#!/bin/bash
#SBATCH --job-name=measure_constants
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=15
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=measure_constants_%j.out
#SBATCH --error=measure_constants_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ldcunha@stonybrook.edu

# One-shot: submit from the repo dir on the login node:  sbatch mig_pp/milp/run_measure_constants.sbatch
# Output (diagnostics first, then the DEFAULT_CONSTANTS dict) lands in measure_constants_<jobid>.out.
#
# Before submitting, on the LOGIN node (compute nodes may have no internet):
#   /gpfs/software/Anaconda/bin/python3 -m venv .venv && source .venv/bin/activate
#   python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('lmsys/vicuna-7b-v1.5')"
# That fills the HF cache; the job then runs with HF_HUB_OFFLINE=1.

set -uo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}   # repo root: must contain model_family.py and mig_pp/milp/
VENV=${VENV:-$REPO/.venv}
# The venv is built from /gpfs/software/Anaconda/bin/python3 (absolute path, own OpenSSL),
# so no module is needed. python/3.11.2 lacks libssl; python/3.12.0 is a milan-tree build.
PY_MODULE=${PY_MODULE:-}

if [ -n "$PY_MODULE" ]; then
  module load "$PY_MODULE" || echo "WARNING: module load $PY_MODULE failed on $(hostname)"
fi
module list 2>&1
echo "venv base: $(grep '^home' "$VENV/pyvenv.cfg" 2>/dev/null)"
ls -l "$VENV/bin/python" && readlink -f "$VENV/bin/python" && ls -l "$(readlink -f "$VENV/bin/python")"
source "$VENV/bin/activate" || { echo "no venv at $VENV"; exit 1; }
cd "$REPO" || exit 1

# Own email (notify_mail.py, Gmail SMTP) at start and on any exit; Slurm's --mail-* is the backup.
OUT="$REPO/measure_constants_${SLURM_JOB_ID}.out"
MAIL="python3 $REPO/mig_pp/milp/notify_mail.py"
RC=1
trap '$MAIL "measure_constants job $SLURM_JOB_ID ended, exit $RC" "$OUT" || echo "end mail failed"' EXIT
$MAIL "measure_constants job $SLURM_JOB_ID started on $(hostname)" || echo "start mail failed"

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node ==="
hostname; date
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,mig.mode.current,driver_version --format=csv
nvidia-smi topo -m 2>/dev/null | head -20
echo "=== python ==="
which python
python - <<'EOF'
import torch, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(),
      "n", torch.cuda.device_count())
print("numpy", numpy.__version__)
try:
    import transformers; print("transformers", transformers.__version__)
except Exception as e:
    print("transformers MISSING:", e)
EOF
ls model_family.py mig_pp/milp/measure_constants.py || { echo "wrong REPO: $REPO"; exit 1; }

echo "=== measure_constants ==="
python mig_pp/milp/measure_constants.py
RC=$?
echo "exit code $RC"
