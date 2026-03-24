import csv
from pathlib import Path
import pulp

# caps in bytes (GiB)
C = [5*(1024**3), 10*(1024**3), 15*(1024**3)]

# change this if you want to pin embed+lmhead into stage0
E_stage0 = 0

mem_csv = Path.home() / "mig_pp" / "profiles" / "poc_layer_memory.csv"

# load per-layer m_i
layers = []
embed_bytes = 0
with open(mem_csv, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        lid = int(row["layer_id"])
        if lid == -1:
            embed_bytes = int(row["param_bytes"])
            continue
        m_i = int(row["param_bytes"]) + int(row["kv_bytes_peak"])
        layers.append((lid, m_i))

layers.sort(key=lambda t: t[0])
L = len(layers)
m = [mi for _, mi in layers]

prob = pulp.LpProblem("pp_split_poc", pulp.LpMinimize)

# x[i][s] binaries
x = [[pulp.LpVariable(f"x_{i}_{s}", lowBound=0, upBound=1, cat="Binary") for s in range(3)] for i in range(L)]
U = pulp.LpVariable("U", lowBound=0, cat="Continuous")

# each layer assigned once
for i in range(L):
    prob += pulp.lpSum(x[i][s] for s in range(3)) == 1

# contiguity
for i in range(L-1):
    prob += x[i][0] >= x[i+1][0]   # stage 0 prefix
    prob += x[i][2] <= x[i+1][2]   # stage 2 suffix

# non-empty stages
for s in range(3):
    prob += pulp.lpSum(x[i][s] for i in range(L)) >= 1

# cap constraints + utilization constraints
mem0 = pulp.lpSum(m[i] * x[i][0] for i in range(L)) + (embed_bytes if E_stage0 else 0)
mem1 = pulp.lpSum(m[i] * x[i][1] for i in range(L))
mem2 = pulp.lpSum(m[i] * x[i][2] for i in range(L))

prob += mem0 <= C[0]
prob += mem1 <= C[1]
prob += mem2 <= C[2]

prob += mem0 <= U * C[0]
prob += mem1 <= U * C[1]
prob += mem2 <= U * C[2]

# objective
prob += U

prob.solve(pulp.PULP_CBC_CMD(msg=False))

# extract boundaries
stage_of = []
for i in range(L):
    s = max(range(3), key=lambda ss: pulp.value(x[i][ss]))
    stage_of.append(s)

# boundary indices in layer space
# stage0: 0..k-1, stage1: k..m-1, stage2: m..L-1
k = next((i for i,s in enumerate(stage_of) if s != 0), L)
m_idx = next((i for i,s in enumerate(stage_of) if s == 2), L)

print("Status:", pulp.LpStatus[prob.status])
print("U (max utilization):", pulp.value(U))
print("Split boundaries: k =", k, " m =", m_idx)
print("Stage layer counts:", stage_of.count(0), stage_of.count(1), stage_of.count(2))
print("Stage mem (GiB):",
      float(pulp.value(mem0))/1024**3,
      float(pulp.value(mem1))/1024**3,
      float(pulp.value(mem2))/1024**3)
