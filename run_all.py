"""Run the experiment stages and save CSVs.  Usage: python run_all.py [grid|asym|fixed|small|all]"""
import sys, time, numpy as np, pandas as pd
from mc import evaluate
from distributions import DISTS

def stage(name, plan, seed):
    rng = np.random.default_rng(seed); rows = []; t0 = time.time()
    for (d, N, R) in plan:
        rows += evaluate(d, N, R, rng)
        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(f"res_{name}.csv", index=False)
            print(f"[{name}] {d} N={N} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(f"res_{name}.csv", index=False)
    print(f"[{name}] finished in {time.time()-t0:.0f}s", flush=True)

def plans():
    NS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    grid = [(d, N, 4000 if N <= 1000 else 2000) for d in DISTS for N in NS]
    asym = [(d, N, R) for d in ["uniform", "bern_0.5", "beta_2_5", "bern_0.05", "bern_0.005", "tri_skew"]
            for N, R in [(1000, 6000), (5000, 4000), (20000, 2000)]]
    fixed = [(d, N, 20000) for d in ["bern_0.5", "bern_0.4", "uniform"]
             for N in [2, 3, 4, 5, 6, 8, 10, 20, 50, 100, 300, 1000]]
    fixed += [("uniform", 10, 100000), ("bern_0.5", 10, 100000)]
    small = [(d, N, 10000) for d in ["bern_0.5", "uniform", "bern_0.1", "beta_2_5"] for N in [2, 3, 4, 5, 6, 8, 10]]
    return dict(grid=grid, asym=asym, fixed=fixed, small=small)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    P = plans()
    for i, name in enumerate(["fixed", "small", "grid", "asym"]):
        if which in (name, "all"):
            stage(name, P[name], 1000 + i)
