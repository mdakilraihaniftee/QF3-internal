"""Large-N stage (asymptotics). Usage: python run_asym.py dist1 dist2 ...   (appends to res_asym.csv)"""
import sys, os, time, numpy as np, pandas as pd
from mc import evaluate
rng = np.random.default_rng(abs(hash(tuple(sys.argv[1:]))) % 2**31)
rows = []
for d in sys.argv[1:]:
    t = time.time()
    for N, R in [(1000, 6000), (5000, 4000), (20000, 1500)]:
        rows += evaluate(d, N, R, rng)
    print(d, f"{time.time()-t:.0f}s", flush=True)
new = pd.DataFrame(rows)
if os.path.exists("res_asym.csv"):
    new = pd.concat([pd.read_csv("res_asym.csv"), new])
new.to_csv("res_asym.csv", index=False)
