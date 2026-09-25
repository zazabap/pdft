"""Low-rank matrix completion --- the classical completion incumbent.

The prior-work families the paper names complete by *rank*, not sparsity:
nuclear-norm matrix completion (Candes and Recht 2009) and the LRTC line it
spawned (Liu et al. 2013). On a grayscale 2D image the unfolding-rank
surrogates of LRTC degenerate to the matrix nuclear norm --- the unfoldings
of a 2-way array are the image and its transpose --- so the honest classical
row for the held-out comparison is matrix completion itself, and this module
provides it.

Two solvers, each with exactly one capacity knob:

- ``svp``: rank-r completion by alternating projections (singular value
  projection, Jain et al. 2010). Identical in shape to the paper's IHT map
  with the sparsity projection H_k replaced by rank truncation, so the row
  isolates the prior: rank against sparsity, same solver skeleton. Knob: r.
- ``apg``: FISTA on the nuclear-norm-regularised problem
  min 1/2 ||P_Omega X - Y||^2 + lam ||X||_*, the APGL of Toh and Yun (2010)
  with the singular-value shrink of Cai, Candes and Shen as the prox.
  Knob: lam. The exact-constraint SVT iteration was tried and rejected: its
  held-out quality is non-monotone in the iteration count, which makes the
  stopping point a second, silent capacity knob --- exactly what the budget
  protocol exists to rule out. FISTA converges to a well-defined minimiser per
  lam, so lam is the only knob.

Everything runs in numpy on the CPU: at 512^2 a float64 LAPACK SVD is ~4x
faster than cuSOLVER's, and the SVD is the whole cost. No coherence column
exists for this row --- there is no basis --- which is part of the point.
"""

from __future__ import annotations

import numpy as np


def _shrink(X: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U * np.maximum(s - tau, 0.0)) @ Vt


def svp(Y: np.ndarray, obs: np.ndarray, r: int, iters: int = 300) -> np.ndarray:
    """Rank-r completion by alternating projections, from the zero-filled Y.

    Ends on the data-consistency step, so observed pixels are exact in the
    output, as they are for the paper's solver."""
    Y = np.asarray(Y, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    X = Y.copy()
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X = (U[:, :r] * s[:r]) @ Vt[:r]
        X[obs] = Y[obs]
    return X


def apg(Y: np.ndarray, obs: np.ndarray, lam: float, iters: int = 200) -> np.ndarray:
    """FISTA on 1/2 ||P_Omega X - Y||^2 + lam ||X||_*  (gradient Lipschitz 1)."""
    Y = np.asarray(Y, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    X = np.zeros_like(Y)
    Z, t = X, 1.0
    for _ in range(iters):
        X_new = _shrink(Z - np.where(obs, Z - Y, 0.0), lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = X_new + ((t - 1.0) / t_new) * (X_new - X)
        X, t = X_new, t_new
    return X
