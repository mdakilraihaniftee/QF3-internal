Project 3 - Confidence intervals (black-box testing of ci.m from Python)
=====================================================================

Requirements: Python 3 (numpy, scipy, pandas, matplotlib), GNU Octave on the PATH
(`octave --version` must work; MATLAB-compatible ci.m is run by Octave). ci.m must be
in this folder. The source of ci.m is never read or parsed: only inputs -> outputs.

Quick test on your own settings (miss rate of the TRUE MEAN, with 99% confidence band):
    python test_ci.py                                   # all 10 functions, defaults
    python test_ci.py -f 3 9 -N 10 100 1000 -d uniform bern_0.5 -r 5000
    python test_ci.py -f 7 -N 5 -d bern_0.5 -r 50000
Distributions available: see distributions.py (Bernoulli, uniform, betas, constant, mixture).
A level-(1-alpha) interval must have miss rate <= alpha for EVERY distribution and EVERY N.

Files
  ci_bridge.py   Python <-> Octave bridge: ci_batch(D) / ci(X,k)  (D = reps x N matrix in [0,1])
  run_batch.m    generic Octave driver (loops ci(x,k), k=1..10, over datasets)
  distributions.py  15 test distributions with known mean
  mc.py          Monte-Carlo core: miss rate, Wilson CI, width / implied-z signatures
  run_all.py     experiment stages (run_asym.py: large-N stage in smaller pieces) -> res_fixed.csv res_small.csv res_grid.csv res_asym.csv
                 (python run_all.py all ; about 30 min on one core)
  analyze.py     builds summary.csv, fig_miss.png, fig_signature.png from the CSVs
  make_report.py builds report.pdf
  report.pdf     the report (verdict table on page 1)

To reproduce everything:  python run_all.py all && python analyze.py && python make_report.py
