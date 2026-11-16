"""Monte-Carlo core: draw datasets from a known distribution, call the 10 black-box ci functions,
and record how often the true mean is missed plus black-box 'signatures' of the intervals."""
import numpy as np, pandas as pd
from scipy.stats import beta as _b
from ci_bridge import ci_batch
from distributions import DISTS

def wilson(k, n, conf=0.99):
    from scipy.stats import norm
    z = norm.ppf(0.5 + conf / 2); p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def evaluate(dname, N, reps, rng, chunk_elems=4_000_000, funcs=range(1, 11)):
    samp, mu, med = DISTS[dname]
    per = max(1, chunk_elems // N)
    hits = np.zeros(10); hitsm = np.zeros(10); n = 0
    sums = {k: np.zeros(10) for k in ["width", "hw", "z", "cdev", "point", "nan"]}
    sq = {k: np.zeros(10) for k in ["hw", "z"]}
    nz = 0
    while n < reps:
        m = min(per, reps - n)
        D = samp(rng, (m, N))
        res = ci_batch(D, seed=int(rng.integers(1, 2**31 - 1)))
        a, b = res[:, :, 0], res[:, :, 1]
        xbar = D.mean(1)[:, None]
        s = D.std(1, ddof=1)[:, None] if N > 1 else np.zeros((m, 1))
        hits += ((a <= mu) & (mu <= b)).sum(0)
        if med is not None:
            hitsm += ((a <= med) & (med <= b)).sum(0)
        hw = (b - a) / 2
        sums["width"] += np.nansum(b - a, 0)
        sums["hw"] += np.nansum(hw * np.sqrt(N), 0); sq["hw"] += np.nansum((hw * np.sqrt(N)) ** 2, 0)
        ok = (s > 1e-12)
        z = np.where(ok, hw * np.sqrt(N) / np.where(ok, s, 1), 0.0)
        sums["z"] += z.sum(0); sq["z"] += (z ** 2).sum(0); nz += ok.sum()
        sums["cdev"] += np.nansum(np.abs((a + b) / 2 - xbar), 0)
        sums["point"] += (a == b).sum(0)
        sums["nan"] += (np.isnan(a) | np.isnan(b)).sum(0)
        n += m
    rows = []
    for k in range(10):
        lo, hi = wilson(reps - hits[k], reps)
        hwm = sums["hw"][k] / reps
        rows.append(dict(func=k + 1, dist=dname, N=N, reps=reps,
                         miss=1 - hits[k] / reps, miss_lo=lo, miss_hi=hi,
                         miss_median=(np.nan if med is None else 1 - hitsm[k] / reps),
                         width=sums["width"][k] / reps,
                         hw_sqrtN=hwm, hw_sqrtN_sd=np.sqrt(max(sq["hw"][k] / reps - hwm ** 2, 0)),
                         z_implied=(sums["z"][k] / nz if nz else np.nan),
                         z_implied_sd=(np.sqrt(max(sq["z"][k] / nz - (sums["z"][k] / nz) ** 2, 0)) if nz else np.nan),
                         center_dev=sums["cdev"][k] / reps, point_frac=sums["point"][k] / reps,
                         errors=int(sums["nan"][k])))
    return rows
