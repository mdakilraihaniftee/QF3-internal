"""Combine the experiment CSVs, derive each function's signature and verdict, and make figures.
Everything is computed from the black-box outputs of ci(X,k) only."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from mc import wilson
from ci_bridge import ci_batch

frames = [pd.read_csv(f).assign(stage=f[4:-4]) for f in ["res_fixed.csv", "res_small.csv", "res_grid.csv", "res_asym.csv"]
          if glob.glob(f)]
A = pd.concat(frames, ignore_index=True)
A["k_miss"] = np.round(A.miss * A.reps)

# ---- probe: structure of function 5's output (random, degenerate or [0,1]?) ----
rng = np.random.default_rng(7)
probe = ci_batch(rng.random((20000, 10)), seed=11)[:, 4, :]
p5_full = np.mean((probe[:, 0] == 0) & (probe[:, 1] == 1)); p5_pt = np.mean(probe[:, 0] == probe[:, 1])
print(f"func 5 probe: P(interval == [0,1]) = {p5_full:.3f}, P(degenerate point) = {p5_pt:.3f}")

# ---- signatures ----
rows = []
for k in range(1, 11):
    d = A[A.func == k]
    big = d[(d.N >= 2000) & (d.dist != "const_0.5")]
    worst = d.loc[d.miss.idxmax()]
    wl = big.loc[big.miss.idxmax()] if len(big) else None
    c = d[d.dist == "uniform"].groupby("N").hw_sqrtN.mean()
    const_width = (d.groupby(["dist", "N"]).hw_sqrtN_sd.max().max() < 1e-6) if k not in (5,) else False
    zbig = d[(d.N >= 5000) & (d.dist == "uniform")].z_implied.mean()
    # nominal level from signature
    alpha_nom, kind = np.nan, ""
    if const_width and np.allclose(c.values, c.values[0], rtol=1e-3):
        alpha_nom = 2 * np.exp(-2 * c.values[0] ** 2); kind = f"data-independent width, half-width*sqrt(N)={c.values[0]:.4f} (Hoeffding form)"
    elif d[(d.N >= 5000)].hw_sqrtN_sd.max() > 0 and np.nanstd(d[(d.dist == "uniform") & (d.N >= 1000)].z_implied) < 0.02 and 1.9 < zbig < 3:
        alpha_nom = 2 * (1 - norm.cdf(zbig)); kind = f"half-width = {zbig:.3f}*s/sqrt(N) (normal form)"
    rows.append(dict(func=k, worst_miss_any=worst.miss, worst_dist=worst.dist, worst_N=int(worst.N),
                     worst_miss_largeN=(wl.miss if wl is not None else np.nan),
                     worst_dist_largeN=(wl.dist if wl is not None else ""),
                     alpha_nominal=alpha_nom, signature=kind))
S = pd.DataFrame(rows)

# ---- verdict rules (documented in the report) ----
def verdict(r):
    k = r.func
    if k == 5:
        d = A[(A.func == 5) & (A.dist != "const_0.5")]
        return "VALID, all N", 0.10, "randomised: degenerate point w.p. ~0.10, otherwise [0,1]"
    if r.worst_miss_largeN > 0.5:
        return "INVALID", np.nan, "miss rate -> ~1 for some distributions as N grows" if k != 4 else "fixed interval, misses whenever mean < .035"
    if not np.isnan(r.alpha_nominal):
        d = A[A.func == k]
        d = d[d.dist != "const_0.5"]
        ub = wilson(d.k_miss.values, d.reps.values, 0.99)[1]
        allN_ok = np.all(ub <= r.alpha_nominal * 1.0001) and np.all(d.miss <= r.alpha_nominal)
        if allN_ok:
            return "VALID, all N", r.alpha_nominal, ""
        return "VALID asymptotically only", r.alpha_nominal, ""
    return "VALID asymptotically only", 0.0, "miss -> 0 (coverage -> 100%) for every fixed distribution; fails at small N"
V = S.apply(verdict, axis=1, result_type="expand"); V.columns = ["verdict", "alpha", "note"]
S = pd.concat([S, V], axis=1)
S.to_csv("summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(S[["func", "verdict", "alpha", "alpha_nominal", "worst_miss_any", "worst_dist", "worst_N", "worst_miss_largeN", "worst_dist_largeN"]].round(4).to_string(index=False))

# ---- figures ----
show = ["bern_0.5", "uniform", "beta_2_5", "bern_0.05", "bern_0.005", "mix_rare_big", "beta_1_20", "bern_0.9"]
G = A[A.stage.isin(["grid", "asym"])]
fig, axs = plt.subplots(5, 2, figsize=(9, 12.5), sharex=True)
for k, ax in zip(range(1, 11), axs.ravel()):
    d = G[G.func == k]
    for ds in show:
        e = d[d.dist == ds].groupby("N").miss.mean()
        ax.plot(e.index, e.values, "o-", ms=3, lw=1.1, label=ds)
    a = S.loc[S.func == k, "alpha"].iloc[0]
    if np.isfinite(a) and a > 0: ax.axhline(a, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_title(f"function {k}: {S.verdict[k-1]}", fontsize=10)
    if k > 8: ax.set_xlabel("N")
    ax.set_ylabel("miss rate", fontsize=8)
axs[0, 0].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("fig_miss.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
u = A[(A.dist == "uniform") & A.stage.isin(["grid"])]
for k in (7, 10):
    e = u[u.func == k].groupby("N").hw_sqrtN.mean(); axs[0].plot(e.index, e.values, "o-", label=f"function {k}")
axs[0].set_xscale("log"); axs[0].set_title("half-width x sqrt(N): constant => data-independent Hoeffding-type"); axs[0].set_xlabel("N"); axs[0].legend()
for k in (3, 9):
    e = u[u.func == k].groupby("N").z_implied.mean(); axs[1].plot(e.index, e.values, "o-", label=f"function {k}")
axs[1].axhline(1.96, color="gray", ls=":"); axs[1].axhline(2.576, color="gray", ls=":")
axs[1].set_xscale("log"); axs[1].set_title("half-width x sqrt(N) / s  (-> 1.96 and 2.576)"); axs[1].set_xlabel("N"); axs[1].legend()
f = A[(A.stage == "fixed") & (A.dist == "bern_0.5") & (A.reps == 20000)]
for k, a in ((7, 0.5), (10, 2 * np.exp(-2))):
    e = f[f.func == k].set_index("N").miss; axs[2].plot(e.index, e.values, "o-", label=f"function {k}"); axs[2].axhline(a, ls="--", color="gray")
axs[2].set_xscale("log"); axs[2].set_title("Bernoulli(.5) miss rate vs nominal alpha (dashed)"); axs[2].set_xlabel("N"); axs[2].legend()
plt.tight_layout(); plt.savefig("fig_signature.png", dpi=130); plt.close()
print("figures written")
