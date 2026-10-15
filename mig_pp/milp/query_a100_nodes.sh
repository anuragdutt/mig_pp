#!/bin/bash
# Which A100 variant (PCIe / SXM4) is in each node of a partition. Run on the login node:
#     bash mig_pp/milp/query_a100_nodes.sh [partition]      (default a100)
# Submits one ~10 s, 1-GPU job per node (pinned with -w), each writing gpu_query_<node>.txt
# and emailing it (notify_mail.py, with the repo's .venv python). Or later:  cat gpu_query_*.txt

PART=${1:-a100}
HERE=$(cd "$(dirname "$0")" && pwd)
MAIL="$HERE/../../.venv/bin/python $HERE/notify_mail.py"

echo "nodes in $PART:"
sinfo -p "$PART" -N -h -o "%N %t %G %f" | sort -u
echo

for node in $(sinfo -p "$PART" -N -h -o "%N" | sort -u); do
  sbatch -p "$PART" -w "$node" --gres=gpu:1 -n 1 -t 00:02:00 -J "gpuq_$node" \
    -o "gpu_query_${node}.txt" \
    --wrap="hostname; nvidia-smi --query-gpu=index,name,memory.total,power.max_limit,clocks.max.sm --format=csv; $MAIL 'gpu query $node' $PWD/gpu_query_${node}.txt"
done
echo
echo "check with: squeue -u \$USER ; then: cat gpu_query_*.txt"
