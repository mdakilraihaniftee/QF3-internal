#!/usr/bin/env python3
"""Quick tester for the black-box ci(X,k) functions (also works on a different ci.m / test set).

Examples
  python test_ci.py                                   # all 10 functions, default settings
  python test_ci.py -f 3 9 -N 10 100 1000 -d uniform bern_0.5 -r 5000
  python test_ci.py -f 7 -N 5 --dist bern_0.5 -r 50000

For every (function, distribution, N) it prints the fraction of datasets whose interval
MISSES the true mean, with a 99% Wilson confidence interval, and the average interval width.
A level-(1-alpha) interval must have miss rate <= alpha for every distribution and N.
"""
import argparse, numpy as np, pandas as pd
from mc import evaluate
from distributions import DISTS

p = argparse.ArgumentParser()
p.add_argument("-f", "--funcs", type=int, nargs="+", default=list(range(1, 11)))
p.add_argument("-N", type=int, nargs="+", default=[10, 100, 1000])
p.add_argument("-d", "--dist", nargs="+", default=["bern_0.5", "uniform", "beta_2_5", "bern_0.05"],
               help="choices: " + ", ".join(DISTS))
p.add_argument("-r", "--reps", type=int, default=3000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
rows = []
for d in a.dist:
    for N in a.N:
        rows += [r for r in evaluate(d, N, a.reps, rng) if r["func"] in a.funcs]
df = pd.DataFrame(rows)
df["miss_99%CI"] = [f"[{lo:.3f},{hi:.3f}]" for lo, hi in zip(df.miss_lo, df.miss_hi)]
pd.set_option("display.width", 200)
print(df[["func", "dist", "N", "reps", "miss", "miss_99%CI", "width"]].round(4).to_string(index=False))
