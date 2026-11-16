"""Build report.pdf (reportlab).  All numbers come from the experiment CSVs (see analyze.py)."""
import pandas as pd, numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.3, leading=12.2, spaceAfter=5)
S = ParagraphStyle("S", parent=B, fontSize=8.3, leading=10.4, spaceAfter=0)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
P = lambda t, st=B: Paragraph(t, st)

A = pd.concat([pd.read_csv(f).assign(stage=f[4:-4]) for f in ["res_fixed.csv", "res_small.csv", "res_grid.csv", "res_asym.csv"]])
def cell(k, dist, N, stage=None, col="miss"):
    d = A[(A.func == k) & (A.dist == dist) & (A.N == N)]
    if stage: d = d[d.stage == stage]
    return d[col].iloc[0]
m = lambda k, d, N, st=None: f"{cell(k, d, N, st):.3f}"

# ---------------- verdict table ----------------
rows = [["Fn", "Valid?", "Level", "All N or asymptotic?", "Main evidence"]]
rows += [
 ["1", "Yes (asympt. only)", "~100% (miss rate -> 0; no useful finite-N level)", "Asymptotic only",
  f"Returns exactly (min, max) of the sample (checked on 2000 random datasets), so width -> 1. Miss rate {m(1,'bern_0.005',5)} for Bernoulli(.005) at N=5 and {m(1,'bern_0.5',2,'small')} for Bernoulli(.5) at N=2, but miss rate -> 0 for every fixed distribution (0.000 at N=20000)."],
 ["2", "<b>No</b>", "-", "-",
  f"Miss rate -> 1: Bernoulli(.005) {m(2,'bern_0.005',20)} (N=20) -> {m(2,'bern_0.005',5000)} (N=5000); continuous spike mixture {m(2,'mix_rare_big',20)} -> {m(2,'mix_rare_big',5000)}."],
 ["3", "Yes (asympt. only)", "<b>95%</b> (alpha=.05)", "Asymptotic only",
  f"Half-width = 1.960*s/sqrt(N). Miss {m(3,'bern_0.5',6,'small')} for Bernoulli(.5) at N=6, {m(3,'bern_0.005',1000,'asym')} for Bernoulli(.005) at N=1000, but -> 0.05 for large N (pooled N>=5000: 0.051)."],
 ["4", "<b>No</b>", "-", "-",
  f"Always returns the same interval [.035,.975]; misses {m(4,'bern_0.02',100)} of the time whenever the true mean is below .035 (e.g. Bernoulli(.02))."],
 ["5", "Yes", "<b>90%</b> (alpha=.10)", "All N",
  "Randomised: returns the point interval w.p. 0.100 (independent of data and N) and [0,1] otherwise. Miss rate 0.0996 (99% CI .0972-.1021) with 100,000 uniform datasets at N=10; pooled 0.1000 over all continuous tests."],
 ["6", "<b>No</b>", "-", "-",
  f"Miss rate -> 1: Beta(1,20) {m(6,'beta_1_20',20)} (N=20) -> {m(6,'beta_1_20',5000)} (N=5000); Bernoulli(.05) stays near 0.50; spike mixture 1.000."],
 ["7", "Yes", "<b>50%</b> (alpha=.5)", "All N",
  f"Mean +/- 0.8326/sqrt(N) (data-independent width = Hoeffding form with alpha=.5). Worst miss observed {A[A.func==7].miss.max():.3f} (Bernoulli(.5), N=3), always below 0.5; typical worst case ~0.12."],
 ["8", "<b>No</b>", "-", "-",
  f"Miss rate -> 1: Bernoulli(.9) {m(8,'bern_0.9',20)} (N=20) -> {m(8,'bern_0.9',5000)}; Bernoulli(.005) 0.997 -> 1.000; spike mixture 0.936 -> 1.000."],
 ["9", "Yes (asympt. only)", "<b>99%</b> (alpha=.01)", "Asymptotic only",
  f"Half-width = 2.576*s/sqrt(N). Miss {m(9,'bern_0.5',8,'small')} for Bernoulli(.5) at N=8, {m(9,'bern_0.005',1000,'asym')} for Bernoulli(.005) at N=1000, but -> 0.01 for large N (pooled N>=5000: 0.011)."],
 ["10", "Yes", "<b>73%</b> (alpha=2e<super>-2</super>=.271)", "All N",
  f"Mean +/- 1/sqrt(N) (data-independent width = Hoeffding form with alpha=.271). Worst miss observed {A[A.func==10].miss.max():.3f} (Bernoulli(.5), N=8), far below .271."],
]
tdata = [[P(c, S) for c in r] for r in rows]
tbl = Table(tdata, colWidths=[0.38*inch, 0.85*inch, 1.15*inch, 0.8*inch, 3.82*inch], repeatRows=1)
st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6ee")), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
      ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
for i in (2, 4, 6, 8): st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fbeaea")))
tbl.setStyle(TableStyle(st))

story = [P("Project 3 - Confidence intervals: black-box evaluation of ci(X,num)", H1),
 P("Level = 1-alpha, the confidence level the interval can be trusted at. 'Valid' means: for every distribution on [0,1], the interval contains the true mean with probability >= 1-alpha "
   "(all N) or in the limit N -> infinity (asymptotic). Invalid functions are shaded.", B),
 tbl, Spacer(1, 6),
 P("<b>Levels for 7 and 10.</b> These are the levels guaranteed by Hoeffding's inequality for the measured half-width. Real coverage is higher "
   "(worst observed miss 0.255 and 0.074), but 50% and 73% are the levels that hold for every [0,1] distribution and every N.", B),
 PageBreak()]

# ---------------- method ----------------
story += [P("1. Method", H1),
 P("The functions were treated strictly as black boxes: Python generated datasets, called ci(X,num) through GNU Octave, and recorded the returned interval. "
   "The source file was not used for any conclusion. The target parameter is the distribution mean (as in the lecture notes), which is known exactly for each simulated distribution.", B),
 P("<b>Distributions on [0,1] (15):</b> Bernoulli(p) for p = .5, .4, .9, .1, .05, .02, .005; uniform; Beta(2,5); Beta(.5,.5); Beta(1,20); Beta(20,1); triangular; "
   "a constant (all data = .5); and a continuous 'rare spike' mixture (98% U(0,.02), 2% U(.9,1), mean .0288). Skewed and rare-event distributions matter because "
   "validity must hold for all distributions, and CLT-based intervals are known to be fragile there.", B),
 P("<b>Sample sizes and replications:</b> N = 2 to 5000 on a grid (4000 datasets per cell for N<=1000, 2000 above), N = 1000, 5000, 20000 for the asymptotic check "
   "(6000/4000/1500 datasets), a small-N scan N = 2..10 (10,000 each), a worst-case search for the fixed-width functions (20,000 each, plus 100,000 for N=10), "
   "i.e. over 1.3 million simulated datasets, all 10 functions on each. Seeds are fixed for reproducibility.", B),
 P("<b>Measurement:</b> the miss rate = fraction of datasets whose interval [a,b] does not contain the true mean, with a 99% Wilson confidence band. "
   "A level-(1-alpha) interval needs miss rate <= alpha for every distribution and N. <b>Deciding 'all N' vs 'asymptotic':</b> if the miss rate exceeds alpha at some small N "
   "but converges to (at most) alpha as N grows for each fixed distribution, the function is asymptotic only. If the miss rate tends to 1 (or stays far above any sensible alpha) "
   "on some distribution, it is invalid.", B),
 P("<b>Identifying the level from outputs alone.</b> Two signatures were measured from the returned intervals. (i) If the width does not depend on the data, "
   "half-width x sqrt(N) is constant: 0.8326 for function 7 and 1.0000 for function 10 at every N and for every distribution (std = 0). "
   "Hoeffding's inequality (Theorem 10 of the notes) gives half-width eps = sqrt(ln(2/alpha)/(2N)), i.e. alpha = 2 exp(-2 c^2) for c = half-width x sqrt(N): "
   "alpha = 0.500 and 0.271, and these intervals are valid for every N. (ii) For data-dependent widths, half-width x sqrt(N)/s converges to 1.960 (function 3) and 2.576 (function 9), "
   "the normal quantiles of alpha = .05 and .01 (fig. 2, middle). These intervals are centred on the sample mean (centre deviation 0) like the notes' normal interval, "
   "whose guarantee is only asymptotic (CLT).", B),
 PageBreak()]

# ---------------- figures ----------------
story += [P("2. Evidence", H1),
 P("Figure 1 shows, for each function, the miss rate against N for several distributions; dashed lines are the nominal alpha of the valid functions.", B),
 Image("fig_miss.png", width=5.3*inch, height=5.3*inch*12.5/9),
 P("Figure 1. Miss rate of the true mean versus N. Valid-for-all-N functions (5, 7, 10) stay below their dashed alpha everywhere; "
   "the asymptotic ones (1, 3, 9) are bad at small N and rare-event distributions and converge; 2, 4, 6, 8 fail permanently.", S),
 PageBreak(),
 Image("fig_signature.png", width=7.0*inch, height=7.0*inch*4/15),
 P("Figure 2. Left: half-width x sqrt(N) of functions 7 and 10 is exactly constant. Middle: half-width x sqrt(N)/s of functions 3 and 9 tends to 1.96 and 2.576. "
   "Right: Bernoulli(.5) miss rate of functions 7 and 10 versus their nominal alpha (dashed).", S),
 Spacer(1, 10)]

# ---------------- per function ----------------
def ex(k, d, N, st=None): return m(k, d, N, st)
story += [P("3. Function-by-function reasoning", H1),
 P("<b>Function 1 - valid only asymptotically (coverage -> 100%).</b> The interval is wide and its width tends to 1 (about 0.98 at N=100 for uniform data). "
   f"For small N it misses a lot: {ex(1,'bern_0.5',2,'small')} of datasets at N=2 and {ex(1,'bern_0.5',5,'small')} at N=5 for Bernoulli(.5), and 0.976 for Bernoulli(.005) at N=5, "
   "so it is not valid at any non-trivial level for all N. Since the miss rate falls to 0 for every fixed distribution (0.000 for all tested distributions at N=5000 and 20000; "
   "Bernoulli(.005) needs N of a few thousand), it is asymptotically valid, but only because it is trivially conservative (level ~100%).", B),
 P("<b>Function 2 - invalid.</b> It behaves well on benign distributions but its miss rate increases with N for skewed ones: Bernoulli(.005) goes from 0.908 (N=20) to 1.000 (N=1000), "
   "Bernoulli(.02) from 0.665 (N=20) to 0.980 (N=1000) and 1.000 (N=5000), and the continuous spike mixture from 0.666 to 0.977 (N=1000) and 1.000 (N=5000). The mean sits outside the returned interval for essentially every dataset, "
   "so there is no alpha &lt; 1 for which it is valid, even asymptotically.", B),
 P("<b>Function 3 - asymptotic, 95%.</b> Half-width 1.960 s/sqrt(N). For small N the miss rate is far above .05 (Bernoulli(.5): 0.218 at N=6, 0.115 at N=10, uniform 0.093 at N=10, "
   "Beta(2,5) 0.102 at N=10; for Bernoulli(.005) 0.976 at N=5 and 0.131 at N=1000). For large N each distribution's miss rate settles at .05: at N=20000 every distribution's 99% band contains .05 "
   "(e.g. Bernoulli(.005): 0.049; uniform 0.045; Beta(2,5) 0.055); pooled over all N>=5000 it is 0.051. This is exactly the CLT behaviour in the notes (simulation in Sec. 3.3).", B),
 P("<b>Function 4 - invalid.</b> The output is the constant interval [0.035, 0.975] for every dataset. It covers means inside this range (miss 0.000) but misses 100% when the mean is below 0.035 "
   "(Bernoulli(.02), Bernoulli(.005), spike mixture), at every N. A confidence interval must adapt to the data.", B),
 P("<b>Function 5 - valid at 90% for all N.</b> A probe of 20,000 datasets shows the output is [0,1] with probability 0.900 and a single point with probability 0.100 (the proportion is 0.087-0.116 in every cell, "
   "independent of the distribution and N). [0,1] always contains the mean; the point interval almost never does for continuous data, so the miss rate is 0.10 regardless of N or distribution. "
   "100,000 uniform datasets at N=10 give 0.0996 (99% band .0972-.1021); pooled over all continuous tests 0.1000. For discrete data the miss rate is slightly lower (the point can hit the mean), so the guarantee "
   "miss <= 0.10 holds everywhere. It is a valid but silly interval (a randomised one that is uninformative 90% of the time).", B),
 P("<b>Function 6 - invalid.</b> The miss rate is zero for symmetric distributions and for distributions with a large mean, but grows to 1 where the mean is small and the mass is concentrated near 0: Beta(1,20) 0.620 (N=20) -> 1.000 (N=5000), "
   "spike mixture 0.666 -> 1.000, Bernoulli(.02) 0.665 -> 0.966, Bernoulli(.05) stays at about 0.50.", B),
 P("<b>Function 7 - valid for all N, 50%.</b> The interval is X-bar +/- 0.8326/sqrt(N) (std of the half-width = 0), which is Hoeffding's interval for alpha = 2 exp(-2 x 0.8326^2) = 0.500. Hoeffding's inequality "
   "holds for every N, and consistently the miss rate never exceeds 0.5: the worst case is Bernoulli(.5) (maximum variance) with 0.255 at N=3, 0.123 at N=4 and about 0.08-0.12 for N from 10 to 5000 "
   "(0.1115 at N=10 with 100,000 datasets); for uniform data it is under 0.007. Valid at 50% for all N.", B),
 P("<b>Function 8 - invalid.</b> It fails on both skewed tails (small and large means): Bernoulli(.9) 0.674 (N=20) -> 1.000 (N=5000), Bernoulli(.005) 0.997 -> 1.000, "
   "spike mixture 0.936 -> 1.000. Its width is about 0.8 and it does not shrink, so it keeps missing the mean for skewed distributions.", B),
 P("<b>Function 9 - asymptotic, 99%.</b> Half-width 2.576 s/sqrt(N). For small N the miss rate is above .01 (Bernoulli(.5): 0.074 at N=8, uniform 0.041 at N=10, Bernoulli(.005): 0.976 at N=5 and 0.041 at N=1000), "
   "and it converges to .01 (N=20000: 0.008-0.015 over eight distributions, all consistent with .01; pooled over N>=5000: 0.011). Convergence is slowest for rare events, as for function 3.", B),
 P("<b>Function 10 - valid for all N, 73%.</b> The interval is X-bar +/- 1/sqrt(N) exactly, Hoeffding with alpha = 2 exp(-2) = 0.271, hence valid at the 1-alpha = 72.9% level for all N. "
   f"The worst observed miss rate is {A[A.func==10].miss.max():.3f} (Bernoulli(.5), N=8) and about 0.05 for Bernoulli(.5) at large N, so it is in practice very conservative "
   "(it is above 0.05 at N=5 and N=8, so it should not be advertised as 95%).", B),
 P("4. Limitations", H1),
 P("A finite set of distributions cannot prove validity; validity claims for 5, 7 and 10 rest on the observed bounds and on Hoeffding's inequality (7, 10) or on the independence of the randomisation (5). "
   "Asymptotic validity of 3 and 9 means the miss rate converges at every fixed distribution; convergence is non-uniform (rare events need N in the thousands), so these intervals are not valid at any fixed finite N. "
   "The target parameter was assumed to be the mean. All Monte-Carlo percentages have standard errors of about 0.1-0.9 percentage points; 99% Wilson bands are in the CSV files.", B)]

SimpleDocTemplate("report.pdf", pagesize=letter, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch,
                  title="Project 3 - Confidence intervals").build(story)
print("report.pdf written")
