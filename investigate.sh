#!/usr/bin/env bash
# investigate.sh -- run the investigation steps one after another, from one command.
#
#   ./investigate.sh [--runs LIST] [--k N] [--raw-logs] [--setup 40gb|80gb] [--smoke] [--dry-run]
#
#   --runs LIST   comma-separated, run in this order (default: control,events,dcgm,timeline,transport,ncu)
#                   control    untraced reference run of the core splits, with power sampling
#                   events     same configs with per-layer timing and token wait (trace_bench.py)
#                   dcgm       same configs with DCGM SM/DRAM sampling
#                   timeline   every nsys timeline the setup defines (nsys_bench.py)
#                   transport  per-microbatch copy/H2D/ACK logging
#                   ncu        ncu_investigate.py: layer on each slice size, then each rank under ncu
#                   neighbors  every neighbors run the setup defines (80gb only so far)
#                   extreme    rank-0 line search (80gb only)
#                   oracle     the Vicuna-7B single-node oracle, all pairs in one run (40gb boxes)
#                   pairs      per (batch, microbatch) pair: that pair's oracle over its own 9-13
#                              candidate splits (vicuna7b_40gb_split_order.BY_PAIR), then for its fastest
#                              and slowest split: layer timing, nsys timeline, ncu on every rank
#                   ncu_pairs  rerun only ncu for every pair whose timelines finished, on the fastest
#                              and slowest split those runs used
#                   deep       per pair, Vicuna-7B: every candidate split plus DEEP_EXTRA, twice:
#                              layer timing (trace_bench.py) then raw transport logs
#                   deep_qwen  the same for Qwen2.5-7B on QWEN7B_BY_PAIR, pairs from QWEN_7B_PAIRS
#                   qwen_ncu   ncu for Qwen2.5-7B: every pair, every rank and the output head; mig1
#                              takes every other profile, mig2 the rest
#   --pairs LIST  pairs for the "pairs" step, e.g. "32,16;8,4" (default: this box's oracle pairs)
#   --pair-steps  what follows each pair's oracle (default: events,timeline,ncu,raw); raw reruns the
#                 fastest and slowest split with per-microbatch transport logs (for reading, not timing)
#   --k N         40gb: how many oracle splits the core runs use (INV_K, default 3)
#   --raw-logs    per-microbatch transport logs on every run (INV_RAW_LOGS=1); slows runs
#   --setup S     force the setup (INV_SETUP); default follows the host
#   --smoke       smoke test first, through trace_bench.py; stops if it fails
#   --dry-run     print the commands only
#
# Each step's run directory is appended to investigation_runs.txt ("step dir"), which
# analyze_investigation.py reads. Each invocation first writes a "# run <time> <args>" line. Start this inside tmux: it runs in the foreground until done.

set -u
ARGS="$*"  # recorded in the run marker line below
cd "$(dirname "$0")" || exit 1

RUNS="control,events,dcgm,timeline,transport,ncu"
PAIRS=""
PAIR_STEPS="events,timeline,ncu,raw"
DRY=0
SMOKE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --runs) RUNS="$2"; shift ;;
        --k) export INV_K="$2"; shift ;;
        --raw-logs) export INV_RAW_LOGS=1 ;;
        --setup) export INV_SETUP="$2"; shift ;;
        --pairs) PAIRS="$2"; shift ;;
        --pair-steps) PAIR_STEPS="$2"; shift ;;
        --smoke) SMOKE=1 ;;
        --dry-run) DRY=1 ;;
        -h | --help) sed -n '2,37p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

PYTHON="${PYTHON:-python3}"
MANIFEST="investigation_runs.txt"
SAMPLER_PID=""
FAILED=""
LAST_DIR=""

say() { printf '%s [investigate] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# Names of the timeline / neighbors runs this setup defines, from parallel_config.INV.
inv_names() { "$PYTHON" -c "import parallel_config as p; print(' '.join(p.INV['$1']))"; }

start_sampler() { # tag [dcgm]
    if [ "$DRY" = 1 ]; then say "would sample: $PYTHON box_sampler.py $*"; return; fi
    "$PYTHON" box_sampler.py "$@" > "investigation_samples_$1.log" 2>&1 &
    SAMPLER_PID=$!
    sleep 5  # let it start nvidia-smi (and DCGM) before the run does
}

stop_sampler() {
    if [ -n "$SAMPLER_PID" ]; then
        kill -TERM "$SAMPLER_PID" 2>/dev/null  # SIGINT is ignored in background jobs
        wait "$SAMPLER_PID" 2>/dev/null
        SAMPLER_PID=""
    fi
}
trap 'stop_sampler; say "interrupted"; exit 130' INT TERM

run_step() { # step inv_run bench(or "")
    local step="$1" inv_run="$2" bench="$3" before after
    say "== $step: INV_RUN=$inv_run${bench:+ MIG_PARALLEL_BENCH=$bench}"
    if [ "$DRY" = 1 ]; then return 0; fi
    # shellcheck disable=SC2012  # run directories are timestamps
    before=$(ls -td runs/*/ 2>/dev/null | head -1)
    if [ -n "$bench" ]; then
        INV_RUN="$inv_run" MIG_PARALLEL_BENCH="$PWD/$bench" ./run_parallel.sh start
    else
        INV_RUN="$inv_run" ./run_parallel.sh start
    fi
    local rc=$?
    # shellcheck disable=SC2012
    after=$(ls -td runs/*/ 2>/dev/null | head -1)
    LAST_DIR=""
    if [ -n "$after" ] && [ "$after" != "$before" ]; then
        LAST_DIR="${after%/}"
        printf '%s %s\n' "$step" "${after%/}" >> "$MANIFEST"
        say "$step -> ${after%/} (exit $rc)"
    else
        say "$step: no new run directory (exit $rc)"
        FAILED="$FAILED $step"
    fi
    [ "$rc" = 0 ] || FAILED="$FAILED $step"
}

# Fastest and slowest ok split of a finished run, as "20,9,2,1 12,10,5,5".
best_worst() {
    "$PYTHON" - "$1" <<'PY'
import csv, glob, sys
rows = [r for f in glob.glob(sys.argv[1] + "/gpu*/*/mig_benchmark_results.csv")
        for r in csv.DictReader(open(f)) if r["status"] == "ok"]
rows.sort(key=lambda r: float(r["total_latency_ms"]))
name = lambda r: r["split"].strip("[]").replace(" ", "")
print(f"{name(rows[0])} {name(rows[-1])}" if rows else "")
PY
}

# One pair: its oracle, then what PAIR_STEPS asks for on its fastest and slowest split.
run_pair() { # "b,mb"
    local pair="$1" tag="b${1%,*}_mb${1#*,}" bw best worst splits
    splits=$("$PYTHON" -c "import vicuna7b_40gb_split_order as v; print(';'.join(','.join(map(str, s)) for s in v.BY_PAIR[($pair)]))") \
        || { say "pair $pair: no candidate splits in vicuna7b_40gb_split_order.BY_PAIR"; FAILED="$FAILED oracle_$tag"; return; }
    INV_PAIRS="$pair" INV_SPLITS="$splits" run_step "oracle_$tag" custom ""
    if [ "$DRY" = 1 ]; then best="BEST"; worst="WORST"
    else
        bw=$(best_worst "$LAST_DIR")
        if [ -z "$bw" ]; then say "pair $pair: no ok results, skipping its investigation"; FAILED="$FAILED oracle_$tag"; return; fi
        best="${bw% *}"; worst="${bw#* }"
    fi
    say "pair $pair: fastest $best, slowest $worst"
    case ",$PAIR_STEPS," in *,events,*)
        INV_PAIRS="$pair" INV_SPLITS="$best;$worst" run_step "events_$tag" custom trace_bench.py ;; esac
    case ",$PAIR_STEPS," in *,timeline,*)
        INV_PAIRS="$pair" INV_SPLITS="$best" run_step "timeline_best_$tag" custom nsys_bench.py
        INV_PAIRS="$pair" INV_SPLITS="$worst" run_step "timeline_worst_$tag" custom nsys_bench.py ;; esac
    case ",$PAIR_STEPS," in *,raw,*)
        INV_RAW_LOGS=1 INV_PAIRS="$pair" INV_SPLITS="$best;$worst" run_step "raw_$tag" custom "" ;; esac
    case ",$PAIR_STEPS," in *,ncu,*)
        say "== ncu_$tag"
        if [ "$DRY" = 0 ]; then
            INV_PAIRS="$pair" INV_SPLITS="$best;$worst" INV_NCU_SCALE=0 "$PYTHON" ncu_investigate.py \
                || FAILED="$FAILED ncu_$tag"
        fi ;; esac
}

say "setup ${INV_SETUP:-host default}, runs: $RUNS, k=${INV_K:-3}, raw logs: ${INV_RAW_LOGS:-0}"
# Marks where this invocation's steps begin in the manifest; readers skip lines starting with #.
[ "$DRY" = 1 ] || printf '# run %s %s\n' "$(date '+%Y-%m-%d_%H:%M:%S')" "$ARGS" >> "$MANIFEST"
if [ "$SMOKE" = 1 ]; then
    say "== smoke"
    if [ "$DRY" = 0 ]; then
        INV_RUN=core MIG_PARALLEL_BENCH="$PWD/trace_bench.py" ./run_parallel.sh smoke || {
            say "smoke failed; stopping"; exit 1; }
    fi
fi

IFS=',' read -r -a STEPS <<< "$RUNS"
for step in "${STEPS[@]}"; do
    case "$step" in
        control) start_sampler core_control; run_step control core ""; stop_sampler ;;
        events) start_sampler core_events; run_step events core trace_bench.py; stop_sampler ;;
        dcgm) start_sampler core_dcgm dcgm; run_step dcgm core ""; stop_sampler ;;
        transport) run_step transport transport "" ;;
        oracle) run_step oracle oracle7b "" ;;
        ncu_pairs)
            tags=$(sed -n 's/^timeline_best_\(b[0-9]*_mb[0-9]*\) .*/\1/p' "$MANIFEST" | sort -u)
            for tag in $tags; do
                best_dir=$(awk -v s="timeline_best_$tag" '$1 == s {d = $2} END {print d}' "$MANIFEST")
                worst_dir=$(awk -v s="timeline_worst_$tag" '$1 == s {d = $2} END {print d}' "$MANIFEST")
                splits=$("$PYTHON" -c "import json, glob, sys
print(';'.join(','.join(map(str, json.load(open(glob.glob(d + '/gpu*/*/job.json')[0]))['splits'][0])) for d in sys.argv[1:]))" "$best_dir" "$worst_dir") \
                    || { say "ncu_pairs $tag: could not read its splits"; FAILED="$FAILED ncu_$tag"; continue; }
                b=${tag%_mb*}; b=${b#b}; mb=${tag#*_mb}
                say "== ncu_$tag: splits $splits"
                if [ "$DRY" = 0 ]; then
                    INV_PAIRS="$b,$mb" INV_SPLITS="$splits" INV_NCU_SCALE=0 "$PYTHON" ncu_investigate.py \
                        || FAILED="$FAILED ncu_$tag"
                fi
            done ;;
        qwen_ncu)
            shard=$("$PYTHON" -c "import parallel_config as p; print({'mig1': '0/2', 'mig2': '1/2'}.get(p.HOST, '0/1'))")
            say "== qwen_ncu: shard $shard"
            if [ "$DRY" = 0 ]; then
                INV_MODEL=qwen_7b INV_NCU_ALL=1 INV_NCU_SHARD="$shard" INV_NCU_SCALE=0 "$PYTHON" ncu_investigate.py \
                    || FAILED="$FAILED qwen_ncu"
            fi ;;
        deep | deep_qwen)
            if [ "$step" = deep ]; then model=vicuna_7b; sets="v.BY_PAIR[p] + v.DEEP_EXTRA.get(p, [])"; table=ORACLE_7B_PAIRS
            else model=qwen_7b; sets="v.QWEN7B_BY_PAIR[p]"; table=QWEN_7B_PAIRS; fi
            step_pairs="$PAIRS"
            [ -n "$step_pairs" ] || step_pairs=$("$PYTHON" -c "import parallel_config as p; print(';'.join(f'{b},{m}' for b, m in p.$table.get(p.HOST, [])))")
            IFS=';' read -r -a PAIR_LIST <<< "$step_pairs"
            for pair in ${PAIR_LIST[@]+"${PAIR_LIST[@]}"}; do
                tag="${model}_b${pair%,*}_mb${pair#*,}"
                splits=$("$PYTHON" -c "import vicuna7b_40gb_split_order as v; p = ($pair); print(';'.join(','.join(map(str, s)) for s in $sets))") \
                    || { say "$step $pair: no splits"; FAILED="$FAILED ${step}_$pair"; continue; }
                INV_MODEL="$model" INV_PAIRS="$pair" INV_SPLITS="$splits" run_step "deep_events_$tag" custom trace_bench.py
                INV_MODEL="$model" INV_PAIRS="$pair" INV_SPLITS="$splits" INV_RAW_LOGS=1 run_step "deep_raw_$tag" custom ""
            done ;;
        pairs)
            [ -n "$PAIRS" ] || PAIRS=$("$PYTHON" -c "import parallel_config as p; print(';'.join(f'{b},{m}' for b, m in p.ORACLE_7B_PAIRS.get(p.HOST, [])))")
            [ -n "$PAIRS" ] || { say "pairs: no pairs for this host; use --pairs"; FAILED="$FAILED pairs"; }
            IFS=';' read -r -a PAIR_LIST <<< "$PAIRS"
            for pair in ${PAIR_LIST[@]+"${PAIR_LIST[@]}"}; do run_pair "$pair"; done ;;
        extreme) run_step extreme extreme "" ;;
        timeline)
            if ! names=$(inv_names timelines); then
                say "timeline: could not read run names from parallel_config"; FAILED="$FAILED timeline"
            fi
            for name in $names; do run_step "$name" "$name" nsys_bench.py; done ;;
        neighbors)
            if ! names=$(inv_names neighbors); then
                say "neighbors: could not read run names from parallel_config"; FAILED="$FAILED neighbors"
            fi
            [ -n "$names" ] || say "neighbors: none defined for this setup"
            for name in $names; do run_step "$name" "$name" ""; done ;;
        ncu)
            say "== ncu"
            if [ "$DRY" = 0 ]; then "$PYTHON" ncu_investigate.py || FAILED="$FAILED ncu"; fi ;;  # exits 1 on any failure
        *) say "unknown step '$step', skipped"; FAILED="$FAILED $step" ;;
    esac
done

if [ -n "$FAILED" ]; then
    say "done; check these steps:$FAILED"
    exit 1
fi
say "done; run directories are listed in $MANIFEST"
