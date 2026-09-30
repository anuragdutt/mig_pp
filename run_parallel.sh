#!/usr/bin/env bash
# run_parallel.sh -- one benchmark lane per GPU, all lanes at once.
#
# A lane is one GPU (its MIG slices) running its models one after another.
# Each lane is its own session / process group; each job has its own
# directory (the benchmark's CWD), port and /dev/shm prefix, all planned by
# parallel_plan.py. This script only starts, watches and stops processes.
#
#   ./run_parallel.sh discover             MIG UUIDs per GPU, paste into parallel_config.py
#   ./run_parallel.sh check                validate the config, show sweep sizes
#   ./run_parallel.sh smoke                one tiny config per job on every lane + PASS/FAIL verdict
#   ./run_parallel.sh start [--detach]     the full sweep (--detach: survives logout)
#   ./run_parallel.sh status [RUN_DIR]     progress per lane
#   ./run_parallel.sh stop [RUN_DIR]       abort early only (a finished run exits by itself):
#                                          TERM every lane, KILL after a grace period
#   options: -c CONFIG, --gpus 0,3, --resume RUN_DIR (start: skip jobs already ok)
#
# Runs on bash 3.2 (the Mac tests) and 5 (the box): no associative arrays or
# `wait -n`, and empty arrays expand as ${a[@]+"${a[@]}"} under `set -u`.

set -u

REPO=$(cd -P "$(dirname "$0")" && pwd)
SELF="$REPO/$(basename "$0")"
PYTHON="${PYTHON:-python3}"
PLANNER="$REPO/parallel_plan.py"
BENCH="${MIG_PARALLEL_BENCH:-$REPO/benchmark_pipeline_microbatching.py}"
SHM_DIR="${MIG_SHM_DIR:-/dev/shm}"
RUNS_DIR="${MIG_PARALLEL_RUNS_DIR:-$REPO/runs}"
GRACE="${MIG_PARALLEL_STOP_GRACE:-20}"
POLL="${MIG_PARALLEL_POLL:-5}"
# Box-wide, not per checkout: ports and GPUs are box-wide too.
LOCK="${MIG_PARALLEL_LOCK:-/tmp/mig_pp_run_parallel.lock}"

log() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf 'run_parallel.sh: %s\n' "$*" >&2; exit 1; }
first_line() { local l=""; [ -f "$1" ] && IFS= read -r l < "$1"; printf '%s' "$l"; }

# Runs its arguments as the leader of a new session, so a lane (bash, the
# benchmark parent and its rank processes) is one process group that can be
# signalled as a unit without touching any other lane. util-linux setsid on
# Linux; the Python fallback is for macOS, where the tests run.
if command -v setsid >/dev/null 2>&1; then
    NEW_SESSION=(setsid)
else
    NEW_SESSION=("$PYTHON" -c 'import os, sys
try:
    os.setsid()
except OSError:
    pass
os.execvp(sys.argv[1], sys.argv[1:])')
fi

# Only ever glob a well-formed per-job prefix: an empty or foreign one would
# delete another run's live segments.
shm_cleanup() {
    case "$1" in
        migpp_*[!A-Za-z0-9_]* | '') return 0 ;;
        migpp_*) rm -f "$SHM_DIR/$1"_* ;;
    esac
}

each_job() { # run_dir -> job dirs, lane by lane
    local lane job
    while IFS= read -r lane; do
        while IFS= read -r job; do [ -n "$job" ] && printf '%s\n' "$job"; done < "$1/lanes/$lane/jobs.txt"
    done < "$1/lanes.txt"
}

lane_pgids() { local lane; while IFS= read -r lane; do first_line "$1/lanes/$lane/pgid"; echo; done < "$1/lanes.txt"; }

any_lane_alive() { # run_dir
    local pg
    for pg in $(lane_pgids "$1"); do kill -0 -- "-$pg" 2>/dev/null && return 0; done
    return 1
}

# --- lock: mkdir is atomic; the pid inside tells a live owner from a stale one
lock_take() {
    if ! mkdir "$LOCK" 2>/dev/null; then
        local pid; pid=$(first_line "$LOCK/pid")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            die "another run is active (pid $pid, $(first_line "$LOCK/run_dir")); stop it first"
        fi
        rm -rf "$LOCK"
        mkdir "$LOCK" || die "cannot create $LOCK"
    fi
    echo $$ > "$LOCK/pid"
}
lock_owner() { echo "$1" > "$LOCK/pid"; echo "$2" > "$LOCK/run_dir"; }
lock_drop() { [ "$(first_line "$LOCK/run_dir")" = "$1" ] && rm -rf "$LOCK"; return 0; }

# --- lane: run one GPU's jobs in order ------------------------------------
lane_main() { # run_dir lane
    local run_dir="$1" dir="$1/lanes/$2" cpus job prefix rc state
    ps -o pgid= -p $$ | tr -d ' ' > "$dir/pgid"
    cpus=$(first_line "$dir/cpus")
    local pin=()
    if [ -n "$cpus" ] && command -v taskset >/dev/null 2>&1; then pin=(taskset -c "$cpus"); fi
    while IFS= read -r job; do
        [ -n "$job" ] || continue
        [ -f "$run_dir/STOP" ] && break
        [ "$(first_line "$job/state")" = ok ] && continue
        prefix=$(first_line "$job/shm_prefix")
        echo running > "$job/state"
        date +%s > "$job/started_at"
        shm_cleanup "$prefix"
        printf '===== attempt %s =====\n' "$(date)" >> "$job/stdout.log"
        # CWD = job dir keeps every relative output of the benchmark apart.
        # shellcheck disable=SC1091
        ( cd "$job" && . ./job.env && exec ${pin[@]+"${pin[@]}"} "$PYTHON" "$BENCH" ) >> "$job/stdout.log" 2>&1
        rc=$?
        shm_cleanup "$prefix"
        echo "$rc" > "$job/exit_code"
        date +%s > "$job/finished_at"
        if [ -f "$run_dir/STOP" ]; then state=killed; elif [ "$rc" -eq 0 ]; then state=ok; else state=failed; fi
        echo "$state" > "$job/state"
        # A failed job does not stop the lane: the next model does not depend on it.
    done < "$dir/jobs.txt"
}

# --- stop -----------------------------------------------------------------
stop_lanes() { # run_dir
    local run_dir="$1" pg waited=0 job
    touch "$run_dir/STOP"
    for pg in $(lane_pgids "$run_dir"); do kill -TERM -- "-$pg" 2>/dev/null; done
    while any_lane_alive "$run_dir"; do
        if [ "$waited" -ge "$GRACE" ]; then
            for pg in $(lane_pgids "$run_dir"); do kill -KILL -- "-$pg" 2>/dev/null; done
            sleep 1
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    for job in $(each_job "$run_dir"); do
        shm_cleanup "$(first_line "$job/shm_prefix")"
        [ "$(first_line "$job/state")" = running ] && echo killed > "$job/state"
    done
}

# --- manager: start every lane, wait, report --------------------------------
manager_main() { # run_dir smoke(0|1)
    local run_dir="$1" smoke="$2" lane pids="" pid rc=0
    echo $$ > "$run_dir/manager.pid"
    lock_owner $$ "$run_dir"
    trap 'trap - INT TERM HUP; stop_lanes "$run_dir"; "$PYTHON" "$PLANNER" report "$run_dir"; lock_drop "$run_dir"; exit 130' INT TERM HUP
    rm -f "$run_dir/STOP"
    while IFS= read -r lane; do
        "${NEW_SESSION[@]}" bash "$SELF" __lane "$run_dir" "$lane" >> "$run_dir/lanes/$lane/lane.log" 2>&1 < /dev/null &
        pids="$pids $!"
        log "lane $lane started"
    done < "$run_dir/lanes.txt"
    # Wait on the lane processes (reaps them), then on their groups in case
    # something a lane started outlived it.
    for pid in $pids; do wait "$pid"; done
    while any_lane_alive "$run_dir"; do sleep "$POLL"; done
    log "all lanes finished"
    if [ "$smoke" -eq 1 ]; then
        "$PYTHON" "$PLANNER" report "$run_dir" --verdict | tee "$run_dir/report.txt"
        rc=${PIPESTATUS[0]}
    else
        "$PYTHON" "$PLANNER" report "$run_dir" | tee "$run_dir/report.txt"
        for pid in $(each_job "$run_dir"); do [ "$(first_line "$pid/state")" = ok ] || rc=1; done
    fi
    lock_drop "$run_dir"
    exit "$rc"
}

cmd_start() { # smoke(0|1) args...
    local smoke="$1" config="$REPO/parallel_config.py" gpus="" detach=0 resume="" run_dir out job
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            -c|--config) config="$2"; shift 2 ;;
            --gpus) gpus="$2"; shift 2 ;;
            --detach) detach=1; shift ;;
            --resume) resume="$2"; shift 2 ;;
            *) die "unknown option '$1'" ;;
        esac
    done
    lock_take
    if [ -n "$resume" ]; then
        run_dir=$(cd -P "$resume" && pwd) && [ -f "$run_dir/lanes.txt" ] || { rm -rf "$LOCK"; die "not a run dir: $resume"; }
        for job in $(each_job "$run_dir"); do
            [ "$(first_line "$job/state")" = ok ] || echo pending > "$job/state"
        done
        log "resuming $run_dir (jobs already ok are skipped)"
    else
        local args=(plan -c "$config")
        [ "$smoke" -eq 1 ] && args+=(--smoke)
        [ -n "$gpus" ] && args+=(--gpus "$gpus")
        if ! out=$(MIG_PARALLEL_RUNS_DIR="$RUNS_DIR" "$PYTHON" "$PLANNER" "${args[@]}"); then
            printf '%s\n' "$out"
            rm -rf "$LOCK"
            die "planning failed; nothing launched"
        fi
        printf '%s\n' "$out"
        run_dir=$(printf '%s\n' "$out" | tail -n 1)
    fi
    lock_owner $$ "$run_dir"
    ln -sfn "$(basename "$run_dir")" "$(dirname "$run_dir")/latest"
    if [ "$detach" -eq 1 ]; then
        nohup "${NEW_SESSION[@]}" bash "$SELF" __manager "$run_dir" "$smoke" > "$run_dir/manager.log" 2>&1 < /dev/null &
        local i=0
        while [ ! -s "$run_dir/manager.pid" ] && [ "$i" -lt 40 ]; do sleep 0.25; i=$((i + 1)); done
        echo "running in the background: $run_dir"
        echo "  ./run_parallel.sh status   |   ./run_parallel.sh stop   |   log: $run_dir/manager.log"
        exit 0
    fi
    manager_main "$run_dir" "$smoke"
}

run_dir_arg() { # [dir] -> run dir (default: the active run, else runs/latest)
    local d="${1:-}"
    [ -n "$d" ] || d=$(first_line "$LOCK/run_dir")
    [ -n "$d" ] && [ -d "$d" ] || d="$RUNS_DIR/latest"
    [ -f "$d/lanes.txt" ] || die "no run directory at '$d'"
    (cd -P "$d" && pwd)
}

cmd_stop() {
    local run_dir mpid i=0
    run_dir=$(run_dir_arg "${1:-}")
    mpid=$(first_line "$run_dir/manager.pid")
    [ -n "$mpid" ] && kill -TERM "$mpid" 2>/dev/null
    stop_lanes "$run_dir"
    while [ -n "$mpid" ] && kill -0 "$mpid" 2>/dev/null && [ "$i" -lt 30 ]; do sleep 1; i=$((i + 1)); done
    lock_drop "$run_dir"
    "$PYTHON" "$PLANNER" report "$run_dir"
}

case "${1:-help}" in
    start) shift; cmd_start 0 "$@" ;;
    smoke) shift; cmd_start 1 "$@" ;;
    status) "$PYTHON" "$PLANNER" report "$(run_dir_arg "${2:-}")" ;;
    stop) cmd_stop "${2:-}" ;;
    check | discover) cmd="$1"; shift; "$PYTHON" "$PLANNER" "$cmd" "$@" ;;
    __lane) lane_main "$2" "$3" ;;
    __manager) manager_main "$2" "$3" ;;
    *) sed -n '2,19p' "$SELF" | sed 's/^# \{0,1\}//' ;;
esac
