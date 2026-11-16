"""Black-box bridge: call the provided Octave/MATLAB ci(X,k) from Python.

Only the *behaviour* of ci.m (inputs -> intervals) is used; its source is never parsed.
Requires GNU Octave on the PATH (octave --version) and ci.m next to this file.
"""
import os, subprocess, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NFUNC = 10


def ci_batch(D, seed=0):
    """D: (reps, N) array of datasets in [0,1].
    Returns (reps, 10, 2): the interval [a, b] returned by ci(X, k) for k = 1..10 (NaN if ci errored)."""
    D = np.ascontiguousarray(np.atleast_2d(np.asarray(D, dtype=np.float64)))
    R, N = D.shape
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
        D.tofile(fin)
        cmd = ["octave", "--no-gui", "--norc", "--path", HERE, "--eval",
               f"run_batch('{fin}','{fout}',{int(seed)},{R},{N})"]
        subprocess.run(cmd, check=True, capture_output=True, cwd=HERE)
        out = np.fromfile(fout, dtype=np.float64)
    return out.reshape(R, NFUNC, 2)


def ci(X, k):
    """Single call, mirrors Matlab's [a b] = ci(X,k)."""
    r = ci_batch(np.asarray(X, float)[None, :])[0, k - 1]
    return r[0], r[1]
