"""A basis learned from data --- sparsifying transform learning.

The completion paper's introduction claims that neither the wavelets nor "a
basis learned from data" will serve, because the localisation that buys
compression buys coherence with the pixel basis. This module measures the
dictionary-learning half of that claim, with the member of that family that
can actually be scored on the same axis. K-SVD and its descendants learn an
*overcomplete* dictionary over patches: hard thresholding is then no longer
the metric projection, and mu = N max|U_ij|^2 is a property of a basis, not of
a redundant dictionary, so neither the solver nor the mu column survives the
move. The square orthonormal member --- sparsifying transform learning,
Ravishankar and Bresler (IEEE TSP 2013) --- is the one that plugs in unchanged:

    min_{W_r, W_c, C_i}  sum_i || W_r X_i W_c^T - C_i ||_F^2
    subject to  W_r, W_c orthonormal  and  ||C_i||_0 <= k

alternating hard thresholding (the sparse coding step, closed form) with an
orthogonal Procrustes update of each axis (also closed form, one SVD). The
objective is *sparsity on the training set*, so this is the trained competitor
that optimises exactly what the reversal says is the wrong target, at
N^2 = 262,144 free parameters per axis against Model B's 144.

Real and separable, initialised at the DCT-II --- the best fixed row of
Table I, and the conventional initialisation for transform learning --- so the
baseline starts from the strongest fixed transform rather than from noise.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..coherence import coherence
from ..metrics import psnr
from ..solver import iht

# --------------------------------------------------------------------------
# the transform: a separable pair of real orthonormal matrices


def dct_matrix(N: int) -> np.ndarray:
    """Orthonormal DCT-II, the initialisation and the fixed-basis reference."""
    j, kk = np.arange(N)[None, :], np.arange(N)[:, None]
    D = np.cos(np.pi * (2 * j + 1) * kk / (2 * N)) * np.sqrt(2.0 / N)
    D[0] /= np.sqrt(2.0)
    return D


def analysis_t(X, Wr, Wc):
    """C = W_r X W_c^T. Dense O(N^3) --- this family has no fast form, which
    is part of what it costs and is stated as such."""
    return Wr @ X @ Wc.T


def synthesis_t(C, Wr, Wc):
    return Wr.T @ C @ Wc


def coherence_t(W) -> float:
    """mu of one axis. The matrix is already dense, so no dense_operator."""
    return float(coherence(jnp.asarray(W)))


def n_params(N: int) -> int:
    """Free real parameters per axis. Orthonormality removes N(N+1)/2 of the
    N^2 entries; the count reported is the ambient one, as Table I reports the
    ambient count for every other row."""
    return N * N


# --------------------------------------------------------------------------
# learning: alternate hard thresholding with orthogonal Procrustes


def _procrustes(M: np.ndarray) -> np.ndarray:
    """argmax over orthonormal W of tr(W M) --- W = V U^T for M = U S V^T."""
    U, _, Vt = np.linalg.svd(M)
    return Vt.T @ U.T


def hard_k_np(C: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest magnitudes; the sparse-coding step in closed form."""
    flat = np.abs(C).ravel()
    if k >= flat.size:
        return C
    thr = np.partition(flat, -k)[-k]
    return np.where(np.abs(C) >= thr, C, 0.0)


def sparsification_error(images, Wr, Wc, k: int) -> float:
    """||W X W^T - H_k(W X W^T)||_F / ||X||_F, averaged --- the objective, and
    the honest way to say whether the learned basis really sparsifies better."""
    num = den = 0.0
    for X in images:
        C = np.asarray(analysis_t(np.asarray(X), Wr, Wc))
        num += float(np.linalg.norm(C - hard_k_np(C, k)) ** 2)
        den += float(np.linalg.norm(X) ** 2)
    return float(np.sqrt(num / den))


def learn_transform(
    images, k: int, iters: int = 30, init: str = "dct", log_every: int = 5, val_images=None
) -> dict:
    """Fit the separable orthonormal pair on the training images.

    Every step is a closed form, so there is no learning rate to sweep and
    nothing for a tuning budget to decide --- unlike the trained circuits, this
    baseline cannot be under-tuned. What it can be is over-fitted: N^2 free
    parameters per axis against a handful of images. Passing ``val_images``
    returns the iterate that minimises sparsification error on that held-back
    split, which is the standard defence.

    Returns ``{"validated", "final", "history"}``: both candidates, since the
    protocol reads every method at its own optimum.
    """
    X = [np.asarray(im, dtype=np.float64) for im in images]
    N = X[0].shape[0]
    if init == "dct":
        Wr = dct_matrix(N)
    elif init == "identity":
        Wr = np.eye(N)
    else:
        raise ValueError(f"unknown init {init!r}")
    Wc = Wr.copy()

    V = None if val_images is None else [np.asarray(v, dtype=np.float64) for v in val_images]

    def record(it):
        e = {"iter": it, "sparsification_error": sparsification_error(X, Wr, Wc, k)}
        if V is not None:
            e["validation_error"] = sparsification_error(V, Wr, Wc, k)
        return e

    hist = [record(0)]
    best = {
        "Wr": Wr.copy(),
        "Wc": Wc.copy(),
        "iter": 0,
        "validation_error": hist[0].get("validation_error", np.inf),
    }
    for it in range(1, iters + 1):
        C = [hard_k_np(Wr @ x @ Wc.T, k) for x in X]
        Wr = _procrustes(sum((x @ Wc.T) @ c.T for x, c in zip(X, C)))
        C = [hard_k_np(Wr @ x @ Wc.T, k) for x in X]
        Wc = _procrustes(sum((x.T @ Wr.T) @ c for x, c in zip(X, C)))
        if it % log_every == 0 or it == iters:
            e = record(it)
            hist.append(e)
            ve = e.get("validation_error")
            if ve is not None and ve < best["validation_error"]:
                best = {"Wr": Wr.copy(), "Wc": Wc.copy(), "iter": it, "validation_error": ve}
            print(
                f"    iter {it:>3}  sparsification {e['sparsification_error']:.5f}"
                + (f"  validation {ve:.5f}" if ve is not None else "")
                + f"  mu_2d {coherence_t(Wr) * coherence_t(Wc):.3f}",
                flush=True,
            )
    if V is not None:
        print(f"    validation selects iteration {best['iter']}", flush=True)
    return {"validated": best, "final": {"Wr": Wr, "Wc": Wc, "iter": iters}, "history": hist}


# --------------------------------------------------------------------------
# the patch member of the same family --- what "dictionary learning" usually
# means, and the configuration a referee will ask for
#
# A single orthonormal transform on P x P patches, applied blockwise, is still
# an orthonormal basis of the whole image, so the solver, the budget rule and
# mu all carry over unchanged. Its coherence is computed from the global
# block-diagonal operator, whose largest entry is the largest entry of the
# patch transform, so mu = N^2 max|W_ab|^2 on the same scale as the separable
# mu column.


def im2patches(X, P: int):
    """(N, N) -> (N^2/P^2, P^2), non-overlapping patches in row-major order."""
    N = X.shape[0]
    return X.reshape(N // P, P, N // P, P).transpose(0, 2, 1, 3).reshape(-1, P * P)


def patches2im(Ypat, P: int, N: int):
    """The exact inverse of im2patches."""
    return Ypat.reshape(N // P, N // P, P, P).transpose(0, 2, 1, 3).reshape(N, N)


def analysis_p(X, W, P: int):
    """Blockwise analysis; the coefficient array is (n_patches, P^2)."""
    return im2patches(X, P) @ W.T


def synthesis_p(C, W, P: int, N: int):
    return patches2im(C @ W, P, N)


def coherence_patch(W, N: int) -> float:
    """mu of the blockwise operator on an N x N image: N^2 max|W_ab|^2."""
    return float(N * N * np.max(np.abs(np.asarray(W)) ** 2))


def dct2_patch(P: int) -> np.ndarray:
    """The 2-D DCT-II on P x P patches as a P^2 x P^2 matrix --- JPEG's basis,
    and the initialisation transform learning conventionally starts from."""
    D = dct_matrix(P)
    return np.kron(D, D)


def sparsification_error_p(images, W, P: int, k: int) -> float:
    num = den = 0.0
    for X in images:
        X = np.asarray(X)
        C = np.asarray(analysis_p(X, W, P))
        num += float(np.linalg.norm(C - hard_k_np(C, k)) ** 2)
        den += float(np.linalg.norm(X) ** 2)
    return float(np.sqrt(num / den))


def hard_s_rows(C: np.ndarray, s: int) -> np.ndarray:
    """Keep the s largest magnitudes of every row --- the per-patch sparsity
    model transform learning is actually fitted under.

    A single global budget over the whole image leaves most patches with no
    active coefficient at all, which makes the Procrustes update arbitrary in
    the unexcited directions: the objective then barely moves while mu drifts
    to nearly N^2. Per-patch sparsity is both the literature's configuration
    and the well-posed one; s is swept and read on a validation split.
    """
    if s >= C.shape[1]:
        return C
    idx = np.argpartition(-np.abs(C), s - 1, axis=1)[:, s:]
    out = C.copy()
    np.put_along_axis(out, idx, 0.0, axis=1)
    return out


def learn_patch_transform(
    images,
    k: int,
    P: int = 8,
    iters: int = 30,
    sparsities=(1, 2, 4, 8, 16),
    log_every: int = 10,
    val_images=None,
):
    """Alternate per-patch hard thresholding with a Procrustes update.

    The per-patch sparsity s is this baseline's capacity knob, swept and read
    at its optimum exactly as every other method in the table is read at its
    own: selection is by sparsification error at the *recovery* budget k on the
    validation split, so the knob is chosen for the task, not for the
    surrogate.
    """
    Y = [np.asarray(im2patches(np.asarray(x, dtype=np.float64), P)) for x in images]
    N = int(np.asarray(images[0]).shape[0])
    ref = val_images if val_images is not None else images
    sweep, best = [], None
    for sp in sparsities:
        W = dct2_patch(P)
        for it in range(1, iters + 1):
            W = _procrustes(sum(y.T @ hard_s_rows(y @ W.T, sp) for y in Y))
            if it % log_every == 0 or it == iters:
                e = {
                    "s": sp,
                    "iter": it,
                    "sparsification_error": sparsification_error_p(images, W, P, k),
                    "validation_error": sparsification_error_p(ref, W, P, k),
                    "mu": coherence_patch(W, N),
                }
                sweep.append(e)
                print(
                    f"    s = {sp:>2}  iter {it:>3}  "
                    f"sparsification {e['sparsification_error']:.5f}  "
                    f"validation {e['validation_error']:.5f}  "
                    f"mu {e['mu']:.1f}",
                    flush=True,
                )
                if best is None or e["validation_error"] < best["validation_error"]:
                    best = {**e, "W": W.copy()}
    print(
        f"    validation selects s = {best['s']}, iteration {best['iter']} "
        f"(validation {best['validation_error']:.5f})",
        flush=True,
    )
    return {"best": best, "sweep": sweep}


def compress_psnr_p(images, W, P: int, k: int) -> np.ndarray:
    out = []
    for X in images:
        X = np.asarray(X)
        C = hard_k_np(np.asarray(analysis_p(X, W, P)), k)
        out.append(psnr(np.asarray(synthesis_p(C, W, P, X.shape[0])), X))
    return np.array(out)


# --------------------------------------------------------------------------
# recovery, through the one IHT scan every family shares


@functools.partial(jax.jit, static_argnames=("K", "mode"))
def reconstruct_t(Wr, Wc, Y, obs, k: int, K: int, mode: str = "hard"):
    """K unrolled solver steps in the learned basis. The matrices take the
    image's precision (protocol.EVAL_DTYPE hands a float32 Y): a float64 W
    against a float32 carry changes the scan's carry type and jax refuses it."""
    Wr, Wc = Wr.astype(Y.dtype), Wc.astype(Y.dtype)
    return iht(
        lambda X: analysis_t(X, Wr, Wc),
        lambda C: synthesis_t(C, Wr, Wc),
        Y,
        obs,
        k,
        K,
        mode,
        remat=False,
    )


@functools.partial(jax.jit, static_argnames=("P", "N", "K", "mode"))
def reconstruct_p(W, Y, obs, P: int, N: int, k: int, K: int, mode: str = "hard"):
    """The same K unrolled steps in the blockwise learned basis."""
    W = W.astype(Y.dtype)
    return iht(
        lambda X: analysis_p(X, W, P),
        lambda C: synthesis_p(C, W, P, N),
        Y,
        obs,
        k,
        K,
        mode,
        remat=False,
    )


def compress_psnr(images, Wr, Wc, k: int) -> np.ndarray:
    """PSNR of the k-term approximation --- the compression side of the
    reversal, measured for this basis exactly as it is for the fixed ones."""
    out = []
    for X in images:
        X = np.asarray(X)
        C = hard_k_np(np.asarray(analysis_t(X, Wr, Wc)), k)
        out.append(psnr(np.asarray(synthesis_t(C, Wr, Wc)), X))
    return np.array(out)
