"""Low-rank matrix completion, the classical completion incumbent.

The prior-work families the paper names complete by rank, not sparsity:
nuclear-norm matrix completion (Candes and Recht 2009) and the LRTC line it
spawned. On a grayscale image the unfolding-rank surrogates of LRTC reduce to
the matrix nuclear norm, so the honest classical row is matrix completion
itself. Two solvers, each with exactly one capacity knob:

``svp`` is rank-``r`` completion by alternating projections (Jain et al. 2010),
identical in shape to the paper's IHT map with the sparsity projection replaced
by rank truncation, so the row isolates the prior. ``apg`` is FISTA on
``1/2 ||P_Omega X - Y||^2 + lam ||X||_*`` (Toh and Yun 2010) with the
singular-value shrink as the prox; the exact-constraint SVT iteration was
tried and rejected because its held-out quality is non-monotone in the
iteration count, which makes the stopping point a second, silent knob.

Everything runs in numpy on the CPU, where a ``512^2`` float64 SVD is several
times faster than on the GPU and is the whole cost.
"""

from __future__ import annotations

import numpy as np


def _shrink(X: np.ndarray, tau: float) -> np.ndarray:
    """The singular-value soft threshold, the prox of ``tau ||X||_*``."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U * np.maximum(s - tau, 0.0)) @ Vt


def svp(Y: np.ndarray, obs: np.ndarray, r: int, iters: int = 300) -> np.ndarray:
    """Rank-``r`` completion by alternating projections from the zero-filled ``Y``.

    Ends on the data-consistency step, so observed pixels are exact in the
    output, as they are for the paper's solver.
    """
    Y = np.asarray(Y, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    X = Y.copy()
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X = (U[:, :r] * s[:r]) @ Vt[:r]
        X[obs] = Y[obs]
    return X


def apg(Y: np.ndarray, obs: np.ndarray, lam: float, iters: int = 200) -> np.ndarray:
    """FISTA on ``1/2 ||P_Omega X - Y||^2 + lam ||X||_*`` (gradient Lipschitz constant 1)."""
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
