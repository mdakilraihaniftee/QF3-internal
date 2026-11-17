Project 3 - Confidence intervals (black-box testing of ci.m from Python)
===============================================================================

Python generates samples from distributions on [0,1], calls the 10 confidence-
interval functions in ci.m through GNU Octave, and measures how often each
interval misses the true mean. The source of ci.m is not used by the analysis.

Requirements
------------

- Python 3.9+
- GNU Octave available as `octave`
- Python packages in requirements.txt

Setup (Debian/Ubuntu)
---------------------

    sudo apt-get update
    sudo apt-get install -y octave python3-venv
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Run
---

Quick smoke test:

    python test_ci.py -f 3 7 9 10 -N 10 -d uniform -r 100

Default test of all functions:

    python test_ci.py

Selected settings:

    python test_ci.py -f 3 9 -N 10 100 1000 -d uniform bern_0.5 -r 5000

The output reports the miss rate, its 99% Wilson confidence interval, and average
interval width. A level-(1-alpha) interval must have miss rate <= alpha for every
distribution and every N. Available distributions are listed in distributions.py.

Reproduce all results (about 30 minutes on one core):

    python run_all.py all
    python analyze.py
    python make_report.py

To rebuild only the analysis and PDF from the included CSV files:

    python analyze.py
    python make_report.py

Main files
----------

  ci_bridge.py      Python/Octave bridge
  run_batch.m       Octave batch driver for ci.m
  distributions.py  15 distributions with known means
  mc.py             Monte Carlo evaluation and Wilson intervals
  run_all.py        fixed, small, grid, and asymptotic experiments
  analyze.py        creates summary.csv and the PNG figures
  make_report.py    creates report.pdf
  report.pdf        final report; verdict table is on page 1
