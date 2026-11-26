#!/usr/bin/env bash
# clock_study.sh -- max-clock reruns of Vicuna-7B on mig1 / mig2 (2026-11-16), one command per box.
#
#   ./clock_study.sh [--dry-run]       start inside tmux; runs in the foreground (~22 h)
#   ./clock_study.sh --probes          the probes only (CPU, then MIG) at max with THREADS, into PROBE_DIR
#
# MODEL (default "": the setup's model, Vicuna-7B) is passed to every run as INV_MODEL, and step names
# then start with it (mistral_7b -> mistral7b_clock_blk01_...), so a Vicuna step name never matches.
# SPLIT_MODULE holds the split lists; VALIDATE_KIND picks NEAR_BEST ("near"), NEAR_BEST + BY_PAIR
# ("union") or the module's ALL ("all", one list per pair, e.g. mistral7b_40gb_split_order).
#
# Two conditions, switched between blocks:
#   default  CPU governor schedutil, GPU clocks at the driver default
#   max      CPU governor performance, GPU SM clock locked at its max. If the driver refuses the lock
#            (MIG), max blocks run with the GPU clock logged only, and the log says so.
# Problem pairs: ROUNDS x [default, max, max, default] blocks. Each block runs the 4 problem pairs
# over NEAR_BEST + BY_PAIR splits; the split order is reversed on every other block. Then the
# VALIDATE pairs at max over NEAR_BEST + BY_PAIR, VALIDATE_PASSES times (forward, reversed, ...).
# Then the STABLE pairs at max, NEAR_BEST, once. THREADS sets OMP_NUM_THREADS for every benchmark
# run (empty: torch's default, one thread per allowed CPU); a step name then carries it in the
# condition, e.g. clock_blk01_max-t1_b32_mb16. CONTRAST pairs run once at max with torch's default
# threads (step condition "max", as in run 17), between validate passes 1 and 2, so the thread
# setting is compared on one box with drift balanced (1 thread, default, 1 thread).
# The host must be in parallel_config.GPUS_40GB. REQUIRE_GPU_LOCK=1 stops the run unless the GPU
# clock locks at its max and persistence mode is on (both read back). It sends a phone notification
# through ntfy.sh (topic NOTIFY_TOPIC below; CLOCK_NOTIFY_TOPIC overrides it) at start,
# when the run ends for any reason, and when one oracle runs past ~2x its estimate.
#
# Writes (appends): investigation_runs.txt (one "# run" line, then "<step> <run dir>" per oracle, step =
# clock_blkNN_<condition>_bB_mbM), clock_study.log (what this script prints, including the governors
# and GPU clock read back at every switch), clock_gpu.csv (nvidia-smi every 1 s), clock_cpu.txt
# (scaling_cur_freq in kHz of the GPU's NUMA-local CPUs, which the lane is pinned to, every 2 s).
# On exit (done, error, Ctrl-C): governor back to schedutil, GPU clocks reset. Exit status 1 if any
# oracle failed. Inherited INV_* (but INV_HOST), MIG_*, MASTER_*, PYTORCH_* and thread settings are
# cleared at start, and logged, so the run is what this file says (model, benchmark, logging, threads).
#   --dry-run   print the schedule and hours only, touch nothing (INV_HOST=mig1 to try it elsewhere)
#   --probes    probe v7.1 under the main condition (2026-11-26): same preflight, loggers, max condition
#               (performance + GPU locked) and restore as the runs, OMP_NUM_THREADS=$THREADS; the CPU
#               probe (measure_constants_cpu.py) on the first MIG slice, pinned to the GPU's NUMA CPUs,
#               then the MIG probe (measure_constants_mig.py), both run inside PROBE_DIR (must not
#               exist), logs constants_{cpu,mig}_v71.log there. No benchmark runs, no manifest lines (~10 min).

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
# P3 Run B (2026-11-26, current.md): Mistral-7B step 1 under the main condition (performance + GPU
# locked + OMP_NUM_THREADS=1), all 9 pairs, mistral7b_40gb_split_order.ALL (136 splits per pair),
# one run each (~37 h). Sealed until the Mistral picks are frozen (mig_pp/milp/SCORING.md).
# P3 Run A (mig5) ran MODEL="", VALIDATE="32,16;16,8", VALIDATE_KIND="union", VALIDATE_PASSES=2,
# CONTRAST="32,16", NOTIFY_LABEL="P3 RunA" (the 1-thread check, SCORING.md "Pre-registered").
# P2 (mig4) ran ROUNDS=0, STABLE="32,2", THREADS="", VALIDATE="". P1 (mig3) ran ROUNDS=1,
# STABLE="32,16;32,8;16,2;32,4;32,2", THREADS="", VALIDATE="" (the full study).
MODEL="mistral_7b"  # parallel_config.MODELS key, passed as INV_MODEL; "" = the setup's model (Vicuna-7B)
SPLIT_MODULE="mistral7b_40gb_split_order"  # where the split lists come from (vicuna7b_40gb_split_order for Vicuna)
ROUNDS=0  # 0: no problem-pair blocks; 1: 2 runs per split per condition (~27 h per box)
PROBLEM="8,4;16,8;16,4;8,2"
VALIDATE="8,4;16,8;32,16;8,2;16,4;32,8;16,2;32,4;32,2"  # at max, VALIDATE_PASSES passes; "" = none
VALIDATE_KIND="all"  # near | union | all (SPLIT_MODULE.ALL)
EXPECT_SPLITS=136  # kind "all": every pair's list must have exactly this many distinct splits; "" = no count check
EXPECT_LAYERS=32  # every split must have 4 ranks and this many layers; "" = only "all splits alike"
VALIDATE_PASSES=1  # pass 1 forward, pass 2 reversed, ...
STABLE=""  # at max over NEAR_BEST, once, last; "" = none
THREADS=1  # OMP_NUM_THREADS for every run (torchrun's default for multi-process jobs); "" = torch's default
CONTRAST=""  # at max with torch's default threads, NEAR_BEST + BY_PAIR, once, after validate pass 1; "" = none
REQUIRE_GPU_LOCK=1  # 1: stop unless the GPU locks at max and persistence mode is on (P3); 0: max blocks may run unlocked (P1/P2)
NOTIFY_LABEL="P3 RunB"  # title of the phone notifications (no host names or paths in them)
HOST="${INV_HOST:-$(hostname -s)}"
PROBE_DIR="probes_v71_$HOST"  # --probes: output folder, relative to the repo root; never overwritten
PROBE_LABEL="P3 probes"  # --probes: title of the phone notifications

# Wall seconds per split, measured in runs 10-15; only for the time estimate.
sec_per_split() {
    case "$1" in
        8,4) echo 80 ;; 16,8) echo 83 ;; 32,16) echo 100 ;; 8,2) echo 92 ;; 16,4) echo 95 ;;
        32,8) echo 100 ;; 16,2) echo 124 ;; 32,4) echo 127 ;; 32,2) echo 182 ;; *) echo 100 ;;
    esac
}

DRY=0
PROBES=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --probes) PROBES=1 ;;
        *) echo "clock_study.sh: unknown argument '$a' (--dry-run, --probes)" >&2; exit 2 ;;
    esac
done
[ "$PROBES" = 0 ] || NOTIFY_LABEL="$PROBE_LABEL"
PYTHON="${PYTHON:-python3}"
NOTIFY_TOPIC="${CLOCK_NOTIFY_TOPIC:-migpp-871bf4d064194bc2}"  # ntfy.sh topic (user's phone); CLOCK_NOTIFY_TOPIC overrides, "" stays on
if ! INV_HOST="$HOST" "$PYTHON" -c 'import sys, parallel_config as p; sys.exit(p.HOST not in p.GPUS_40GB)' 2> /dev/null; then
    echo "clock_study.sh: host '$HOST' is not in parallel_config.GPUS_40GB (INV_HOST overrides)" >&2
    exit 2
fi
BLOCK_THREADS="$THREADS"  # OMP_NUM_THREADS of the block running now (run_block sets it)
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
STOP_REASON=""
INTERRUPTED=0
OK_RUNS=0
TOTAL_RUNS=0
WATCH=""
BLOCK=0
NV=0  # validate passes that run
[ -z "$VALIDATE" ] || NV=$VALIDATE_PASSES
NS=0  # stable blocks that run
[ -z "$STABLE" ] || NS=1
NC=0  # contrast blocks that run
[ -z "$CONTRAST" ] || NC=1
NBLOCKS=$((4 * ROUNDS + NV + NC + NS))
NUMA_LIST=""  # the GPU's NUMA-local CPU list (start_loggers sets it)
STEP_PREFIX="${MODEL:+${MODEL//_/}_}"  # mistral_7b -> "mistral7b_"; "" for the setup's model

say() {
    local line
    line="$(date '+%m-%d %H:%M:%S') [clock] $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG"
}
die() { STOP_REASON="$*"; say "STOP: $*"; exit 1; }
notify() { # message: one push to the phone; a failure is logged, never fatal
    [ -n "$NOTIFY_TOPIC" ] && [ "$DRY" = 0 ] || return 0
    curl -fsS --max-time 15 -H "Title: $NOTIFY_LABEL" -d "$1" "https://ntfy.sh/$NOTIFY_TOPIC" > /dev/null 2>&1 \
        || say "notification not sent (curl failed): $1"
}
npairs() { [ -n "$1" ] && echo "$1" | tr ';' '\n' | grep -c . || echo 0; }
governors() { cat "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c | xargs; }
gpu_clocks() {
    nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw \
        --format=csv,noheader 2>&1 | head -1
}
hours() { awk -v s="$1" 'BEGIN { printf "%.1f", s / 3600 }'; }

# "b,mb" near|union|all reverse(0|1) -> "a,b,c,d;..." from SPLIT_MODULE: NEAR_BEST, NEAR_BEST plus
# the BY_PAIR splits not in it, or ALL
split_list() {
    "$PYTHON" - "$1" "$2" "$3" "$SPLIT_MODULE" <<'PY'
import importlib
import sys
v = importlib.import_module(sys.argv[4])
pair = tuple(int(x) for x in sys.argv[1].split(","))
if sys.argv[2] == "all":
    out = [list(s) for s in v.ALL[pair]]
else:
    out = [list(s) for s in v.NEAR_BEST[pair]]
if sys.argv[2] == "union":
    out += [list(s) for s in v.BY_PAIR[pair] if list(s) not in out]
if sys.argv[3] == "1":
    out.reverse()
print(";".join(",".join(map(str, s)) for s in out))
PY
}

# Every list the run will use, checked before anything is touched (dry runs too): the module
# imports, and each pair's list is non-empty, has no duplicates, and all its splits have the same
# rank count and layer total; with EXPECT_LAYERS, 4 ranks and that total; with EXPECT_SPLITS, kind
# "all" lists have exactly that many splits (a stale or short list on the box stops the run). Args: "kind:pairs" per block kind that runs, e.g. "all:8,4;16,8".
check_lists() {
    local out
    out=$("$PYTHON" - "$SPLIT_MODULE" "$EXPECT_SPLITS" "$EXPECT_LAYERS" "$@" 2>&1 <<'PY'
import importlib
import sys
try:
    v = importlib.import_module(sys.argv[1])
except Exception as e:
    sys.exit(f"cannot import {sys.argv[1]}: {type(e).__name__}: {e}")
want_n = int(sys.argv[2]) if sys.argv[2] else None
want_layers = int(sys.argv[3]) if sys.argv[3] else None
bad, counts = [], []
for group in sys.argv[4:]:
    kind, pairs = group.split(":", 1)
    for text in [p for p in pairs.split(";") if p.strip()]:
        pair = tuple(int(x) for x in text.split(","))
        try:
            if kind == "all":
                out = [list(s) for s in v.ALL[pair]]
            else:
                out = [list(s) for s in v.NEAR_BEST[pair]]
                if kind == "union":
                    out += [list(s) for s in v.BY_PAIR[pair] if list(s) not in out]
        except Exception as e:
            bad.append(f"{text} {kind}: {type(e).__name__}: {e}")
            continue
        shapes = {(len(s), sum(s)) for s in out}
        if not out:
            bad.append(f"{text} {kind}: empty")
        elif len({tuple(s) for s in out}) != len(out):
            bad.append(f"{text} {kind}: {len(out) - len({tuple(s) for s in out})} duplicate splits")
        elif len(shapes) != 1:
            bad.append(f"{text} {kind}: splits differ in ranks / layer total {sorted(shapes)}")
        elif want_layers is not None and shapes != {(4, want_layers)}:
            bad.append(f"{text} {kind}: splits are {sorted(shapes)} (ranks, layers), want (4, {want_layers})")
        elif kind == "all" and want_n is not None and len(out) != want_n:
            bad.append(f"{text} {kind}: {len(out)} splits, want {want_n}")
        counts.append(f"{text} {len(out)}")
if bad:
    sys.exit("; ".join(bad))
print(", ".join(counts))
PY
) || die "split lists ($SPLIT_MODULE): $(echo "$out" | tail -1)"
    say "split lists ok ($SPLIT_MODULE): $out"
}

est_seconds() { # pairs kind
    local s=0 pair n list
    [ -n "$1" ] || { echo 0; return; }
    IFS=';' read -r -a list <<< "$1"
    for pair in "${list[@]}"; do
        n=$(split_list "$pair" "$2" 0 | tr ';' '\n' | grep -c .)
        s=$((s + n * $(sec_per_split "$pair")))
    done
    echo "$s"
}

cleanup() {
    local rc=$? bad problems="" status restore=""
    # shellcheck disable=SC2086  # LOGGERS is a list of pids
    [ -z "$LOGGERS" ] || kill $LOGGERS 2>/dev/null
    [ -z "$WATCH" ] || kill "$WATCH" 2>/dev/null
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
            restore="; RESTORE FAILED, fix by hand:$problems"
        else
            say "restored: governors $(governors); GPU (sm, max sm, temp, power) $(gpu_clocks)"
            restore="; governor and GPU clock restored"
        fi
    fi
    [ -z "$FAILED" ] || say "FAILED:$FAILED"
    if [ "$INTERRUPTED" = 1 ]; then
        status="interrupted after $OK_RUNS/$TOTAL_RUNS runs"
    elif [ -n "$STOP_REASON" ]; then
        status="STOPPED after $OK_RUNS/$TOTAL_RUNS runs: $STOP_REASON"
    elif [ -n "$FAILED" ]; then
        status="finished, but failed:$FAILED ($OK_RUNS/$TOTAL_RUNS ok)"
    elif [ "$rc" = 0 ]; then
        status="done: $OK_RUNS/$TOTAL_RUNS runs ok"
    else
        status="exited with code $rc after $OK_RUNS/$TOTAL_RUNS runs"
    fi
    notify "$status$restore"
    exit "$rc"
}
trap cleanup EXIT
trap 'INTERRUPTED=1; say "interrupted"; exit 130' INT TERM HUP

preflight() {
    local apps bad out pm sm
    if [ -n "$MODEL" ] && [ "$PROBES" = 0 ]; then  # before anything is touched; probes need no weights
        # The planner's own check (parallel_plan): complete HF weights, not just a config.json.
        out=$("$PYTHON" -c 'import sys, parallel_config as p, model_family as mf
mf.find_weight_files(mf.resolve_model_path(str(p.MODELS[sys.argv[1]]["path"])))' "$MODEL" 2>&1) \
            || die "no complete weights for MODEL=$MODEL ($(echo "$out" | tail -1)); python3 download_models.py --only $MODEL"
        say "model $MODEL (INV_MODEL for every run; step names start with $STEP_PREFIX)"
    fi
    if pgrep -f run_parallel.sh > /dev/null; then die "run_parallel.sh is already running"; fi
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c -E '^[0-9]+')
    [ "$apps" -eq 0 ] || die "$apps processes are on the GPU"
    sudo -n true 2>/dev/null || die "sudo asks for a password (sudo -n failed)"
    bad=$(grep -L -x schedutil "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor | wc -l | tr -d ' ')
    [ "$bad" -eq 0 ] || die "$bad CPUs are not on schedutil; put them back first"
    if [ -n "$THREADS" ]; then
        # The benchmark does not log its thread count: check that torch here takes OMP_NUM_THREADS.
        out=$(OMP_NUM_THREADS="$THREADS" "$PYTHON" -c 'import torch; print(torch.get_num_threads())' 2>&1 | tail -1)
        [ "$out" = "$THREADS" ] || die "OMP_NUM_THREADS=$THREADS, but torch reports $out threads"
        say "threads: OMP_NUM_THREADS=$THREADS for every run (torch reports $out)"
    fi
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
        # A clock lock only holds between benchmark processes with persistence mode on: read it back.
        sudo -n nvidia-smi -i 0 -pm 1 > /dev/null 2>&1
        pm=$(nvidia-smi -i 0 --query-gpu=persistence_mode --format=csv,noheader | tr -d ' ')
        if [ "$pm" = Enabled ]; then
            say "persistence mode on (was $PM_WAS)"
        elif [ "$REQUIRE_GPU_LOCK" = 1 ]; then
            die "persistence mode did not turn on (reads '$pm'); a GPU lock would not hold between runs"
        else
            say "persistence mode did not turn on (reads '$pm'); continuing (REQUIRE_GPU_LOCK=0)"
        fi
    fi
    if out=$(sudo -n nvidia-smi -i 0 -lgc "$SM_MAX,$SM_MAX" 2>&1); then
        GPU_LOCK=1
        sleep 2
        # Read the SM clock back (up to 5 tries: a single read can come back [N/A]).
        for _ in 1 2 3 4 5; do
            sm=$(nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits | tr -d ' ')
            [ "$sm" = "$SM_MAX" ] && break
            sleep 1
        done
        if [ "$sm" != "$SM_MAX" ] && [ "$REQUIRE_GPU_LOCK" = 1 ]; then
            die "GPU lock set, but the SM clock reads '$sm' MHz, not $SM_MAX"
        fi
        say "GPU lock test ok at $SM_MAX MHz: (sm, max sm, temp, power) $(gpu_clocks)"
        sudo -n nvidia-smi -i 0 -rgc > /dev/null 2>&1
    elif [ "$REQUIRE_GPU_LOCK" = 1 ]; then
        die "GPU lock refused (REQUIRE_GPU_LOCK=1). nvidia-smi said: $out"
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
    NUMA_LIST="$list"
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
    local before after rc n limit
    # shellcheck disable=SC2012  # run directories are timestamps
    before=$(ls -td runs/*/ 2>/dev/null | head -1)
    # Watchdog: a push if this oracle runs past ~2x its estimate (+10 min); killed when it ends.
    n=$(echo "$3" | tr ';' '\n' | grep -c .)
    limit="${CLOCK_WATCH_SECONDS:-$((2 * n * $(sec_per_split "$2") + 600))}"
    ( sleep "$limit"; notify "$1 still running after $((limit / 60)) min (about 2x its estimate); check the box" ) &
    WATCH=$!
    env ${BLOCK_THREADS:+"OMP_NUM_THREADS=$BLOCK_THREADS"} ${MODEL:+"INV_MODEL=$MODEL"} INV_RUN=custom \
        INV_PAIRS="$2" INV_SPLITS="$3" ./run_parallel.sh start < /dev/null
    rc=$?
    kill "$WATCH" 2>/dev/null
    WATCH=""
    # shellcheck disable=SC2012
    after=$(ls -td runs/*/ 2>/dev/null | head -1)
    if [ -n "$after" ] && [ "$after" != "$before" ]; then
        printf '%s %s\n' "$1" "${after%/}" >> "$MANIFEST"
        say "$1 -> ${after%/} (exit $rc)"
        [ "$rc" = 0 ] && OK_RUNS=$((OK_RUNS + 1))
    else
        say "$1: no new run directory (exit $rc)"
        FAILED="$FAILED $1"
    fi
    [ "$rc" = 0 ] || FAILED="$FAILED $1"
}

run_block() { # condition pairs near|union reverse(0|1) [threads: default $THREADS, "" = torch's default]
    local cond="$1" kind="$3" rev="$4" pair list splits n step order="" label
    BLOCK_THREADS="${5-$THREADS}"
    label="$cond${BLOCK_THREADS:+-t$BLOCK_THREADS}"  # max -> max-t1 when the block runs 1 thread
    BLOCK=$((BLOCK + 1))
    [ "$DRY" = 1 ] || set_condition "$cond"
    [ "$rev" = 1 ] && order=", reversed"
    IFS=';' read -r -a list <<< "$2"
    for pair in "${list[@]}"; do
        splits=$(split_list "$pair" "$kind" "$rev")
        n=$(echo "$splits" | tr ';' '\n' | grep -c .)
        step=$(printf '%sclock_blk%02d_%s_b%s_mb%s' "$STEP_PREFIX" "$BLOCK" "$label" "${pair%,*}" "${pair#*,}")
        say "== block $BLOCK/$NBLOCKS $label ($pair): $n splits ($kind$order), ~$((n * $(sec_per_split "$pair") / 60)) min"
        [ "$DRY" = 1 ] || run_oracle "$step" "$pair" "$splits"
    done
}

# --probes: one probe script inside PROBE_DIR, with a watchdog push past limit seconds.
run_probe() { # name script log limit gpu ("" = all) [taskset cpu list]
    local rc
    ( sleep "$4"; notify "probe $1 still running after $(($4 / 60)) min; check the box" ) &
    WATCH=$!
    (
        cd "$PROBE_DIR" || exit 1
        env ${THREADS:+"OMP_NUM_THREADS=$THREADS"} ${5:+"CUDA_VISIBLE_DEVICES=$5"} \
            ${6:+taskset -c "$6"} "$PYTHON" "../mig_pp/milp/$2"
    ) < /dev/null 2>&1 | tee -a "$PROBE_DIR/$3"
    rc=${PIPESTATUS[0]}
    kill "$WATCH" 2>/dev/null
    WATCH=""
    if [ "$rc" = 0 ]; then
        OK_RUNS=$((OK_RUNS + 1))
        say "probe $1 ok -> $PROBE_DIR/$3"
    else
        FAILED="$FAILED probe_$1"
        say "probe $1 failed (exit $rc); log $PROBE_DIR/$3"
    fi
}

run_probes() {
    BLOCK=1
    say "== probes, $HOST: CPU probe (first MIG slice, CPUs ${NUMA_LIST:-unpinned}), then MIG probe; \
max condition, OMP_NUM_THREADS ${THREADS:-unset (torch default)} -> $PROBE_DIR (~10 min)"
    [ "$DRY" = 1 ] && return
    set_condition max
    PROBE_GPU=$(nvidia-smi -L | grep -o 'MIG-[^)]*' | head -1)
    [ -n "$PROBE_GPU" ] || die "no MIG slice in nvidia-smi -L"
    mkdir "$PROBE_DIR" || die "could not create $PROBE_DIR"
    run_probe cpu measure_constants_cpu.py constants_cpu_v71.log 1200 "$PROBE_GPU" "$NUMA_LIST"
    # MIG probe as in P2: unpinned, every slice visible; it starts one child per slice type itself
    # (CUDA_VISIBLE_DEVICES per child).
    run_probe mig measure_constants_mig.py constants_mig_v71.log 2700 ""
    say "condition at the end: governors $(governors); GPU (sm, max sm, temp, power) $(gpu_clocks)"
}

# --- main ---------------------------------------------------------------------------------------
[ -z "$CLEARED" ] || say "cleared from the environment:$CLEARED"
if [ "$PROBES" = 1 ]; then
    TOTAL_RUNS=2
    # Checked before anything is touched: an existing folder is earlier probe data.
    [ ! -e "$PROBE_DIR" ] || die "$PROBE_DIR exists (earlier probe data); move it away first, nothing was touched"
    if [ "$DRY" = 0 ]; then
        preflight
        start_loggers
        notify "probes started: CPU + MIG, ~10 min; GPU locked at $SM_MAX MHz"
    fi
    run_probes
    [ -z "$FAILED" ] || exit 1
    say "probes done"
    exit 0
fi
TOTAL_RUNS=$((4 * ROUNDS * $(npairs "$PROBLEM") + NV * $(npairs "$VALIDATE") + NC * $(npairs "$CONTRAST") \
    + NS * $(npairs "$STABLE")))
LISTS=()
[ "$ROUNDS" = 0 ] || LISTS+=("union:$PROBLEM")
[ "$NV" = 0 ] || LISTS+=("$VALIDATE_KIND:$VALIDATE")
[ "$NC" = 0 ] || LISTS+=("union:$CONTRAST")
[ "$NS" = 0 ] || LISTS+=("near:$STABLE")
[ "${#LISTS[@]}" = 0 ] || check_lists "${LISTS[@]}"
if [ "$DRY" = 0 ]; then
    preflight
    start_loggers
fi
P=0
[ "$ROUNDS" = 0 ] || P=$(est_seconds "$PROBLEM" union)
V=$(est_seconds "$VALIDATE" "$VALIDATE_KIND")
C=$(est_seconds "$CONTRAST" union)
S=$(est_seconds "$STABLE" near)
lock="locked at $SM_MAX MHz"
[ "$GPU_LOCK" = 1 ] || lock="not locked (logged only)"
[ "$DRY" = 0 ] || lock="tested at start"
say "plan, $HOST, model ${MODEL:-setup default}, splits from $SPLIT_MODULE: problem pairs $PROBLEM in $((4 * ROUNDS)) blocks ((default, max, max, default) x \
$ROUNDS); validate pairs ${VALIDATE:-none} at max ($VALIDATE_KIND) x $NV; contrast pairs ${CONTRAST:-none} at max, torch's \
default threads, once; stable pairs ${STABLE:-none} at max once. OMP_NUM_THREADS ${THREADS:-unset (torch \
default)} except contrast. GPU at max: $lock. ~$(hours $((4 * ROUNDS * P + NV * V + NC * C + S))) h"
notify "started${MODEL:+ $MODEL}: $TOTAL_RUNS runs, ~$(hours $((4 * ROUNDS * P + NV * V + NC * C + S))) h; GPU $lock"
[ "$DRY" = 1 ] || printf '# run %s clock_study.sh host=%s model=%s splits=%s:%s rounds=%s gpu_lock=%s threads=%s contrast=%s\n' \
    "$(date '+%Y-%m-%d_%H:%M:%S')" "$HOST" "${MODEL:-default}" "$SPLIT_MODULE" "$VALIDATE_KIND" "$ROUNDS" "$GPU_LOCK" \
    "${THREADS:-default}" "${CONTRAST:-none}" >> "$MANIFEST"

for ((round = 1; round <= ROUNDS; round++)); do  # not seq: BSD `seq 1 0` counts down
    run_block default "$PROBLEM" union 0
    run_block max "$PROBLEM" union 1
    run_block max "$PROBLEM" union 0
    run_block default "$PROBLEM" union 1
done
for ((pass = 1; pass <= NV; pass++)); do
    run_block max "$VALIDATE" "$VALIDATE_KIND" $(((pass + 1) % 2))  # pass 1 forward, 2 reversed, ...
    if [ "$pass" = 1 ] && [ "$NC" = 1 ]; then
        run_block max "$CONTRAST" union 0 ""  # torch's default threads, between the 1-thread passes
    fi
done
if [ "$NV" = 0 ] && [ "$NC" = 1 ]; then
    run_block max "$CONTRAST" union 0 ""
fi
[ -z "$STABLE" ] || run_block max "$STABLE" near 0
if [ -n "$FAILED" ]; then
    say "all $NBLOCKS blocks ran, but some oracles failed (listed below)"
    exit 1
fi
say "all $NBLOCKS blocks done"
