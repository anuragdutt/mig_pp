#!/usr/bin/env bash
# p2.sh -- PCIe session P2 (2026-11-25), one command, one box. Start inside tmux, venv active:
#
#   INV_HOST=<box> ./p2.sh [--dry-run]
#
# Steps, in order (P2 started late, 2026-11-25 ~4 pm: ~15.5 h to the 7:30 am Thu deadline); a failed
# step is logged and the next one still runs:
#   1 drift    Vicuna-7B, default conditions: (8,4) [23,6,2,1] and (16,4) [20,9,2,1], the two splits
#              run 17 measured under default on mig3 (blocks 1 and 4), to compare the boxes (~5 min)
#   2 probes   probe v7 into probes_v7/: CPU probe (pinned to the GPU's NUMA CPUs, first MIG slice),
#              then MIG probe (~1-1.5 h); first, the ILP freeze before P3 needs it
#   3 mistral  Mistral-7B near-best lists, 7 pairs, 2-4 microbatch pairs first (~10.5 h); (32,4) and
#              (32,2) go to P3. Sealed: export it, do not analyse it before the ILP freeze
#   4 check    download_models.py --check for Llama-2-7B and Qwen2.5-7B (gated repo, HF token)
#   5 clock    ./clock_study.sh: run 17's leftover, (32,2) at max (~2.7 h), only if it ends before
#              P2_DEADLINE (default Thu 11/26 7:30 am EST) with 10% to spare; else skipped (-> P3).
#              It sets and restores the governor and GPU clock itself
# After every step: ~/<box>_p2_<N>_<step>.tgz with everything P2 has made so far (manifest, logs,
# clock logs, probes_v7/, every run directory listed after p2's "# run" line). Copy the newest.
# p2.log has every line this script prints.

set -u
cd "$(dirname "$0")" || exit 1

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
HOST="${INV_HOST:-}"
[ -n "$HOST" ] || { echo "p2.sh: set INV_HOST (the box name in parallel_config.GPUS_40GB)" >&2; exit 2; }
export INV_HOST="$HOST"
PYTHON="${PYTHON:-python3}"
# Commands, overridable so a test can stub them; the defaults are what runs on the box.
RUN_PARALLEL="${P2_RUN_PARALLEL:-./run_parallel.sh}"
CLOCK="${P2_CLOCK:-./clock_study.sh}"
INVESTIGATE="${P2_INVESTIGATE:-./investigate.sh}"
SYS="${P2_SYS:-/sys}"
OUT="${P2_OUT:-$HOME}"
MISTRAL_DIR="${P2_MISTRAL_DIR:-$HOME/models/Mistral-7B-Instruct-v0.3}"
MISTRAL_PAIRS="${P2_MISTRAL_PAIRS:-8,4;16,8;32,16;8,2;16,4;32,8;16,2}"  # 2, 2, 2, 4, 4, 4, 8 microbatches
DEADLINE="${P2_DEADLINE:-1795696200}"  # epoch s: 2026-11-26 12:30 UTC = 7:30 am EST, lease end - 1 h
CLOCK_S=$((160 * 60 * 11 / 10))  # (32,2) at max: 53 splits ~160 min, + 10%
DRIFT=("8,4 23,6,2,1" "16,4 20,9,2,1")
MANIFEST="investigation_runs.txt"
LOG="p2.log"
[ "$DRY" = 1 ] && LOG=/dev/null
FAILED=""
STEP=0
MARK=""

say() {
    local line
    line="$(date '+%m-%d %H:%M:%S') [p2] $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG"
}
trap 'say "interrupted"; exit 130' INT TERM HUP
governor() { cat "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | xargs; }
all_schedutil() { [ "$(grep -L -x schedutil "$SYS"/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | wc -l | tr -d ' ')" = 0 ]; }

# Everything P2 has made so far, into ~/<box>_p2_<N>_<step>.tgz.
export_now() { # step name
    local tgz dirs extra=""
    tgz="$OUT/${HOST}_p2_${STEP}_$1.tgz"
    dirs=$(awk -v mark="$MARK" '$0 == mark {on = 1; next} on && !/^#/ && NF == 2 {print $2}' "$MANIFEST")
    for f in "$LOG" clock_study.log clock_gpu.csv clock_cpu.txt probes_v7; do [ -e "$f" ] && extra="$extra $f"; done
    # shellcheck disable=SC2086  # word lists of paths without spaces
    if tar czf "$tgz" "$MANIFEST" $extra $dirs 2>> "$LOG"; then
        say "export: $tgz ($(du -h "$tgz" | cut -f1); $(echo "$dirs" | grep -c .) run directories)"
    else
        say "export FAILED: $tgz"
        FAILED="$FAILED export_$1"
    fi
}

step() { # name description; then runs "${@:3}" and exports
    local name="$1" desc="$2" rc
    shift 2
    STEP=$((STEP + 1))
    say "== step $STEP/5 $name: $desc"
    [ "$DRY" = 1 ] && { say "   (dry run) $*"; return 0; }
    "$@"
    rc=$?
    say "step $STEP $name: exit $rc"
    [ "$rc" = 0 ] || FAILED="$FAILED $name"
    export_now "$name"
}

# One oracle, recorded in the manifest like investigate.sh / clock_study.sh do.
oracle() { # manifest step, pair, splits
    local before after rc
    # shellcheck disable=SC2012  # run directories are timestamps
    before=$(ls -td runs/*/ 2>/dev/null | head -1)
    INV_RUN=custom INV_PAIRS="$2" INV_SPLITS="$3" "$RUN_PARALLEL" start < /dev/null
    rc=$?
    # shellcheck disable=SC2012
    after=$(ls -td runs/*/ 2>/dev/null | head -1)
    if [ -n "$after" ] && [ "$after" != "$before" ]; then
        printf '%s %s\n' "$1" "${after%/}" >> "$MANIFEST"
        say "$1 -> ${after%/} (exit $rc)"
    else
        say "$1: no new run directory (exit $rc)"
        rc=1
    fi
    return "$rc"
}

drift() {
    local d pair split rc=0
    printf '# run %s p2.sh drift host=%s\n' "$(date '+%Y-%m-%d_%H:%M:%S')" "$HOST" >> "$MANIFEST"
    for d in "${DRIFT[@]}"; do
        pair=${d% *}
        split=${d#* }
        oracle "p2_drift_b${pair%,*}_mb${pair#*,}" "$pair" "$split" || rc=1
    done
    return "$rc"
}

probes() {
    local u bus cpus rc=0
    mkdir -p probes_v7
    u=$(nvidia-smi -L | grep -o 'MIG-[^)]*' | head -1)
    bus=$(nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-F' 'a-f')
    cpus=$(cat "$SYS/bus/pci/devices/${bus#0000}/local_cpulist" 2>/dev/null)
    say "probes: CPU probe on $u, CPUs ${cpus:-unpinned (local_cpulist not found)}"
    (
        cd probes_v7 || exit 1
        if [ -n "$cpus" ]; then
            CUDA_VISIBLE_DEVICES="$u" taskset -c "$cpus" "$PYTHON" ../mig_pp/milp/measure_constants_cpu.py
        else
            CUDA_VISIBLE_DEVICES="$u" "$PYTHON" ../mig_pp/milp/measure_constants_cpu.py
        fi
    ) 2>&1 | tee probes_v7/constants_cpu_v7.log
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
    (cd probes_v7 && "$PYTHON" ../mig_pp/milp/measure_constants_mig.py) 2>&1 | tee probes_v7/constants_mig_v7.log
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
    return "$rc"
}

mistral() {
    all_schedutil || { say "CPUs not on schedutil ($(governor)), e.g. clock_study's restore failed: Mistral skipped"; return 1; }
    [ -f "$MISTRAL_DIR/config.json" ] || { say "no Mistral-7B weights at $MISTRAL_DIR: skipped"; return 1; }
    INV_MODEL=mistral_7b "$INVESTIGATE" --runs near --pairs "$MISTRAL_PAIRS" < /dev/null
}

check() { "$PYTHON" download_models.py --check --only llama_7b qwen_7b < /dev/null; }

clock() {
    local now left
    now=$(date +%s)
    left=$((DEADLINE - now))
    if [ "$left" -lt "$CLOCK_S" ]; then
        say "clock: $((left / 60)) min to the deadline, (32,2) needs ~$((CLOCK_S / 60)): skipped (-> P3)"
        return 0
    fi
    say "clock: $((left / 60)) min to the deadline, (32,2) needs ~$((CLOCK_S / 60)): running"
    "$CLOCK" < /dev/null
}

# --- main ---------------------------------------------------------------------------------------
say "host $HOST, commit $(git log --oneline -1 2>/dev/null), governors $(governor)"
if [ "$DRY" = 0 ]; then
    all_schedutil || { say "STOP: CPUs not on schedutil ($(governor)); drift and Mistral need default conditions"; exit 1; }
    MARK="# p2 $(date '+%Y-%m-%d_%H:%M:%S') host=$HOST"
    printf '%s\n' "$MARK" >> "$MANIFEST"
fi
step drift "Vicuna-7B (8,4) [23,6,2,1] + (16,4) [20,9,2,1], default" drift
step probes "probe v7: CPU, then MIG, into probes_v7/" probes
step mistral "Mistral-7B near-best, pairs $MISTRAL_PAIRS (sealed)" mistral
step check "download_models.py --check: Llama-2-7B, Qwen2.5-7B" check
step clock "run 17 leftover: (32,2) at max (clock_study.sh), if it fits" clock
if [ -n "$FAILED" ]; then
    say "done; check these steps:$FAILED"
    exit 1
fi
say "done; copy the last export"
