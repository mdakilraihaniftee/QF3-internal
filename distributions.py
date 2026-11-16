"""Test distributions on [0,1] with known mean (and median where well defined).

name -> (sampler(rng, shape), true_mean, true_median or None)
"""
import numpy as np
from scipy.stats import beta as _beta

def _bern(p):
    return (lambda g, s: (g.random(s) < p).astype(float)), p, None

def _mix_rare(g, s):
    # 98%: U(0,0.02); 2%: U(0.9,1) -> continuous, mean 0.0288, 95%-quantile ~ 0.02 < mean
    big = g.random(s) < 0.02
    return np.where(big, 0.9 + 0.1 * g.random(s), 0.02 * g.random(s))

DISTS = {
    "bern_0.5":   _bern(0.5),
    "bern_0.4":   _bern(0.4),
    "bern_0.9":   _bern(0.9),
    "bern_0.1":   _bern(0.1),
    "bern_0.05":  _bern(0.05),
    "bern_0.02":  _bern(0.02),
    "bern_0.005": _bern(0.005),
    "uniform":    (lambda g, s: g.random(s), 0.5, 0.5),
    "beta_2_5":   (lambda g, s: g.beta(2, 5, s), 2 / 7, float(_beta.ppf(0.5, 2, 5))),
    "beta_.5_.5": (lambda g, s: g.beta(.5, .5, s), 0.5, 0.5),
    "beta_1_20":  (lambda g, s: g.beta(1, 20, s), 1 / 21, 1 - 0.5 ** (1 / 20)),
    "beta_20_1":  (lambda g, s: g.beta(20, 1, s), 20 / 21, 0.5 ** (1 / 20)),
    "const_0.5":  (lambda g, s: np.full(s, 0.5), 0.5, 0.5),
    "tri_skew":   (lambda g, s: g.triangular(0, 0, 1, s), 1 / 3, 1 - np.sqrt(0.5)),
    "mix_rare_big": (_mix_rare, 0.98 * 0.01 + 0.02 * 0.95, 0.01),
}
