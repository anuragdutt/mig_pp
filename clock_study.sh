#!/usr/bin/env bash
# clock_study.sh -- max-clock reruns of Vicuna-7B on mig1 / mig2 (2026-11-16), one command per box.
#
#   ./clock_study.sh [--dry-run]       start inside tmux; runs in the foreground (~22 h)
#
# Two conditions, switched between blocks:
#   default  CPU governor schedutil, GPU clocks at the driver default
#   max      CPU governor performance, GPU SM clock locked at its max. If the driver refuses the lock
#            (MIG), max blocks run with the GPU clock logged only, and the log says so.
# Problem pairs: ROUNDS x [default, max, max, default] blocks. Each block runs the 4 problem pairs
# over NEAR_BEST + BY_PAIR splits; the split order is reversed on every other block. Then the 5
# stable pairs at max, NEAR_BEST, once. mig1 and mig2 run the same schedule; their runs add up.
#
# Writes (appends): investigation_runs.txt (one "# run" line, then "<step> <run dir>" per oracle, step =
# clock_blkNN_<condition>_bB_mbM), clock_study.log (what this script prints, including the governors
# and GPU clock read back at every switch), clock_gpu.csv (nvidia-smi every 1 s), clock_cpu.txt
# (scaling_cur_freq in kHz of the GPU's NUMA-local CPUs, which the lane is pinned to, every 2 s).
# On exit (done, error, Ctrl-C): governor back to schedutil, GPU clocks reset. Exit status 1 if any
# oracle failed. Inherited INV_* (but INV_HOST), MIG_*, MASTER_*, PYTORCH_* and thread settings are
# cleared at start, and logged, so the run is what this file says (model, benchmark, logging, threads).
#   --dry-run   print the schedule and hours only, touch nothing (INV_HOST=mig1 to try it elsewhere)

set -u
cd "$(dirname "$0")" || exit 1

CLEARED=""
for v in $(compgen -e); do
    case "$v" in
        INV_HOST) ;;
        INV_* | MIG_* | MASTER_ADDR | MASTER_PORT | PYTORCH_* | OMP_* | MKL_*)
            CLEARED="$CLEARED $v=${!v}"
            unset "$v" ;;
    esac
done

# --- config -----------------------------------------------------------------------------------
# P2 (2026-11-25): run 17's leftover only, (32,2) at max. P1 (mig3) ran ROUNDS=1 with all 5 stable
# pairs and stopped at the lease end after (32,4); for the full study set ROUNDS=1 and
# STABLE="32,16;32,8;16,2;32,4;32,2" again.
ROUNDS=0  # 0: no problem-pair blocks; 1: 2 runs per split per condition (~27 h per box)
PROBLEM="8,4;16,8;16,4;8,2"
STABLE="32,2"
HOST="${INV_HOST:-$(hostname -s)}"
case "$HOST" in
    mig1 | mig2 | mig3 | mig4) ;;
    *) echo "clock_study.sh: this host is not mig1, mig2, mig3 or mig4 (INV_HOST overrides)" >&2; exit 2 ;;
esac

# Wall seconds per split, measured in runs 10-15; only for the time estimate.
sec_per_split() {
    case "$1" in
        8,4) echo 80 ;; 16,8) echo 83 ;; 32,16) echo 100 ;; 8,2) echo 92 ;; 16,4) echo 95 ;;
        32,8) echo 100 ;; 16,2) echo 124 ;; 32,4) echo 127 ;; 32,2) echo 182 ;; *) echo 100 ;;
    esac
}

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
PYTHON="${PYTHON:-python3}"
SYS="${CLOCK_SYS:-/sys}"  # tests point this at a fake tree
MANIFEST="investigation_runs.txt"
LOG="clock_study.log"
[ "$DRY" = 1 ] && LOG=/dev/null
FAILED=""
LOGGERS=""
TOUCHED=0
GPU_LOCK=0
SM_MAX=""
PM_WAS=""
BLOCK=0
NBLOCKS=$((4 * ROUNDS + 1))

say() {
    local line
    line="$(date '+%m-%d %H:%M:%S') [clock] $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG"
}
die() { say "STOP: $*"; exit 1; }
governors() { cat "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c | xargs; }
gpu_clocks() {
    nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw \
        --format=csv,noheader 2>&1 | head -1
}
hours() { awk -v s="$1" 'BEGIN { printf "%.1f", s / 3600 }'; }

# "b,mb" near|union reverse(0|1) -> "a,b,c,d;..." (NEAR_BEST, plus the BY_PAIR splits not in it)
split_list() {
    "$PYTHON" - "$1" "$2" "$3" <<'PY'
import sys
import vicuna7b_40gb_split_order as v
pair = tuple(int(x) for x in sys.argv[1].split(","))
out = [list(s) for s in v.NEAR_BEST[pair]]
if sys.argv[2] == "union":
    out += [list(s) for s in v.BY_PAIR[pair] if list(s) not in out]
if sys.argv[3] == "1":
    out.reverse()
print(";".join(",".join(map(str, s)) for s in out))
PY
}

est_seconds() { # pairs kind
    local s=0 pair n list
    IFS=';' read -r -a list <<< "$1"
    for pair in "${list[@]}"; do
        n=$(split_list "$pair" "$2" 0 | tr ';' '\n' | grep -c .)
        s=$((s + n * $(sec_per_split "$pair")))
    done
    echo "$s"
}

cleanup() {
    local rc=$? bad problems=""
    # shellcheck disable=SC2086  # LOGGERS is a list of pids
    [ -z "$LOGGERS" ] || kill $LOGGERS 2>/dev/null
    if [ "$TOUCHED" = 1 ]; then
        echo schedutil | sudo -n tee "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1
        bad=$(grep -L -x schedutil "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor | wc -l | tr -d ' ')
        [ "$bad" -eq 0 ] || problems="$problems $bad CPUs not on schedutil (echo schedutil | sudo tee \
/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor);"
        # Only if a lock was ever set: where the driver refuses locks, a reset can fail harmlessly.
        if [ "$GPU_LOCK" = 1 ]; then
            sudo -n nvidia-smi -i 0 -rgc > /dev/null 2>&1 || problems="$problems GPU clocks not reset (sudo nvidia-smi -i 0 -rgc);"
        fi
        if [ "$PM_WAS" = Disabled ]; then
            sudo -n nvidia-smi -i 0 -pm 0 > /dev/null 2>&1 || problems="$problems persistence mode still on (sudo nvidia-smi -i 0 -pm 0);"
        fi
        if [ -n "$problems" ]; then
            say "RESTORE FAILED, fix by hand before any other run:$problems governors now $(governors)"
            [ "$rc" -ne 0 ] || rc=1
        else
            say "restored: governors $(governors); GPU (sm, max sm, temp, power) $(gpu_clocks)"
        fi
    fi
    [ -z "$FAILED" ] || say "FAILED:$FAILED"
    exit "$rc"
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM HUP

preflight() {
    local apps bad out
    if pgrep -f run_parallel.sh > /dev/null; then die "run_parallel.sh is already running"; fi
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c -E '^[0-9]+')
    [ "$apps" -eq 0 ] || die "$apps processes are on the GPU"
    sudo -n true 2>/dev/null || die "sudo asks for a password (sudo -n failed)"
    bad=$(grep -L -x schedutil "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor | wc -l | tr -d ' ')
    [ "$bad" -eq 0 ] || die "$bad CPUs are not on schedutil; put them back first"
    say "host $HOST, commit $(git log --oneline -1 2>/dev/null)"
    say "GPU (name, driver, MIG, max sm, power limit, persistence): $(nvidia-smi -i 0 \
--query-gpu=name,driver_version,mig.mode.current,clocks.max.sm,power.limit,persistence_mode \
--format=csv,noheader 2>&1 | head -1)"
    say "CPU: driver $(cat "$SYS"/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null), boost \
$(cat "$SYS"/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a), governors $(governors)"
    SM_MAX=$(nvidia-smi -i 0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits | tr -d ' ')
    PM_WAS=$(nvidia-smi -i 0 --query-gpu=persistence_mode --format=csv,noheader | tr -d ' ')
    TOUCHED=1
    if [ "$PM_WAS" != Enabled ]; then
        # A clock lock only holds between benchmark processes with persistence mode on.
        sudo -n nvidia-smi -i 0 -pm 1 > /dev/null 2>&1 && say "persistence mode on (was $PM_WAS)"
    fi
    if out=$(sudo -n nvidia-smi -i 0 -lgc "$SM_MAX,$SM_MAX" 2>&1); then
        GPU_LOCK=1
        sleep 2
        say "GPU lock test ok at $SM_MAX MHz: (sm, max sm, temp, power) $(gpu_clocks)"
        sudo -n nvidia-smi -i 0 -rgc > /dev/null 2>&1
    else
        say "GPU lock refused; max blocks run with the GPU clock logged only. nvidia-smi said: $out"
    fi
}

cpu_logger() { # "cpu numbers"
    local c f line
    line="epoch_s"
    for c in $1; do line="$line cpu$c"; done
    echo "$line"
    while :; do
        line=$(date +%s)
        for c in $1; do
            f=NA
            read -r f 2>/dev/null < "$SYS/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq"
            line="$line $f"
        done
        echo "$line"
        sleep 2
    done
}

start_loggers() {
    local base q bus dom list cpus d g0 c0
    base="timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu"
    # The first query this driver answers: clock event reasons by the new name, the old name, or none.
    for q in "$base,clocks_event_reasons.active" "$base,clocks_throttle_reasons.active" "$base" ""; do
        [ -n "$q" ] || die "nvidia-smi cannot report the GPU clock: \
$(nvidia-smi -i 0 --query-gpu="$base" --format=csv,noheader 2>&1 | head -1)"
        nvidia-smi -i 0 --query-gpu="$q" --format=csv,noheader 2>&1 | grep -q -E '^[0-9]{4}/' && break
    done
    touch clock_gpu.csv clock_cpu.txt
    g0=$(wc -l < clock_gpu.csv | tr -d ' ')  # both files append across reruns: check only new lines
    c0=$(wc -l < clock_cpu.txt | tr -d ' ')
    nvidia-smi -i 0 --query-gpu="$q" --format=csv,nounits -l 1 >> clock_gpu.csv 2>&1 &
    LOGGERS="$!"
    # The GPU's NUMA-local CPUs, as in parallel_plan._numa_cpus: "00000000:81:00.0" -> "0000:81:00.0"
    bus=$(nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader | tr -d ' ')
    dom=${bus%%:*}
    list=$(cat "$SYS/bus/pci/devices/$(echo "${dom: -4}:${bus#*:}" | tr 'A-F' 'a-f')/local_cpulist" 2>/dev/null)
    cpus=$("$PYTHON" -c 'import sys
print(" ".join(str(c) for p in sys.argv[1].split(",") if p.strip()
               for c in range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1)))' "$list")
    if [ -z "$cpus" ]; then
        for d in "$SYS"/devices/system/cpu/cpu[0-9]*; do cpus="$cpus ${d##*cpu}"; done
        say "no NUMA-local CPU list for the GPU; logging every CPU"
    fi
    cpu_logger "$cpus" >> clock_cpu.txt &
    LOGGERS="$LOGGERS $!"
    # Both logs must be writing before any benchmark runs.
    sleep 5
    tail -n +$((g0 + 1)) clock_gpu.csv | grep -q -E '^[0-9]{4}/' \
        || die "GPU clock log is not writing: $(tail -n +$((g0 + 1)) clock_gpu.csv | head -2 | tr '\n' ' ')"
    tail -n +$((c0 + 1)) clock_cpu.txt \
        | awk '$1 ~ /^[0-9]+$/ { for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]+$/) ok = 1 } END { exit !ok }' \
        || die "CPU frequency log is not writing (scaling_cur_freq unreadable)"
    say "logging: GPU every 1 s -> clock_gpu.csv ($q); CPUs ${list:-all} every 2 s -> clock_cpu.txt"
}

set_condition() { # default|max
    local gov=schedutil bad
    [ "$1" = max ] && gov=performance
    echo "$gov" | sudo -n tee "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null \
        || die "could not set the governor to $gov"
    bad=$(grep -L -x "$gov" "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor | wc -l | tr -d ' ')
    [ "$bad" -eq 0 ] || die "$bad CPUs did not take governor $gov"
    if [ "$GPU_LOCK" = 1 ]; then
        # Stop rather than run a block under a label it does not have.
        if [ "$1" = max ]; then
            sudo -n nvidia-smi -i 0 -lgc "$SM_MAX,$SM_MAX" > /dev/null 2>&1 \
                || die "GPU clock lock failed; not running max blocks with the GPU unlocked"
        else
            sudo -n nvidia-smi -i 0 -rgc > /dev/null 2>&1 \
                || die "GPU clock reset failed; not running default blocks with the GPU locked"
        fi
    fi
    sleep 2
    say "condition $1: governors $(governors); GPU (sm, max sm, temp, power) $(gpu_clocks)"
}

# One oracle: like investigate.sh run_step, with this study's step name.
run_oracle() { # step pair splits
    local before after rc
    # shellcheck disable=SC2012  # run directories are timestamps
    before=$(ls -td runs/*/ 2>/dev/null | head -1)
    INV_RUN=custom INV_PAIRS="$2" INV_SPLITS="$3" ./run_parallel.sh start < /dev/null
    rc=$?
    # shellcheck disable=SC2012
    after=$(ls -td runs/*/ 2>/dev/null | head -1)
    if [ -n "$after" ] && [ "$after" != "$before" ]; then
        printf '%s %s\n' "$1" "${after%/}" >> "$MANIFEST"
        say "$1 -> ${after%/} (exit $rc)"
    else
        say "$1: no new run directory (exit $rc)"
        FAILED="$FAILED $1"
    fi
    [ "$rc" = 0 ] || FAILED="$FAILED $1"
}

run_block() { # condition pairs near|union reverse(0|1)
    local cond="$1" kind="$3" rev="$4" pair list splits n step order=""
    BLOCK=$((BLOCK + 1))
    [ "$DRY" = 1 ] || set_condition "$cond"
    [ "$rev" = 1 ] && order=", reversed"
    IFS=';' read -r -a list <<< "$2"
    for pair in "${list[@]}"; do
        splits=$(split_list "$pair" "$kind" "$rev")
        n=$(echo "$splits" | tr ';' '\n' | grep -c .)
        step=$(printf 'clock_blk%02d_%s_b%s_mb%s' "$BLOCK" "$cond" "${pair%,*}" "${pair#*,}")
        say "== block $BLOCK/$NBLOCKS $cond ($pair): $n splits ($kind$order), ~$((n * $(sec_per_split "$pair") / 60)) min"
        [ "$DRY" = 1 ] || run_oracle "$step" "$pair" "$splits"
    done
}

# --- main ---------------------------------------------------------------------------------------
[ -z "$CLEARED" ] || say "cleared from the environment:$CLEARED"
if [ "$DRY" = 0 ]; then
    preflight
    start_loggers
fi
P=$(est_seconds "$PROBLEM" union)
S=$(est_seconds "$STABLE" near)
lock="locked at $SM_MAX MHz"
[ "$GPU_LOCK" = 1 ] || lock="not locked (logged only)"
[ "$DRY" = 0 ] || lock="tested at start"
say "plan, $HOST: problem pairs $PROBLEM in $((4 * ROUNDS)) blocks ((default, max, max, default) x \
$ROUNDS), then stable pairs $STABLE at max once. GPU at max: $lock. ~$(hours $((4 * ROUNDS * P + S))) h"
[ "$DRY" = 1 ] || printf '# run %s clock_study.sh host=%s rounds=%s gpu_lock=%s\n' \
    "$(date '+%Y-%m-%d_%H:%M:%S')" "$HOST" "$ROUNDS" "$GPU_LOCK" >> "$MANIFEST"

for ((round = 1; round <= ROUNDS; round++)); do  # not seq: BSD `seq 1 0` counts down
    run_block default "$PROBLEM" union 0
    run_block max "$PROBLEM" union 1
    run_block max "$PROBLEM" union 0
    run_block default "$PROBLEM" union 1
done
run_block max "$STABLE" near 0
if [ -n "$FAILED" ]; then
    say "all $NBLOCKS blocks ran, but some oracles failed (listed below)"
    exit 1
fi
say "all $NBLOCKS blocks done"
