"""
Runs trace_bench.py under Nsight Systems, for a timeline of all 4 ranks in the Nsight Systems GUI.

    INV_RUN=timeline_best32 MIG_PARALLEL_BENCH=$PWD/nsys_bench.py ./run_parallel.sh start --detach

Writes timeline.nsys-rep in the job directory. Only decode steps 256-259 are captured
(trace_bench.TIMELINE_STEPS): a full run would be millions of kernel records.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

env = dict(os.environ, INV_TIMELINE="1")
cmd = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
       "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
       "-o", "timeline", sys.executable, os.path.join(HERE, "trace_bench.py")]
os.execvpe("nsys", cmd, env)
