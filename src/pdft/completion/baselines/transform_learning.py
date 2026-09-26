"""A basis learned from data --- sparsifying transform learning.

The completion paper's introduction claims that a basis learned from data will
not serve for recovery from pixels, because the localisation that buys
compression buys coherence. This module measures that with the member of the
dictionary-learning family that can be scored on the same axis: K-SVD's
overcomplete patch dictionaries break both the solver (hard thresholding stops
being the metric projection) and the mu column (a property of a basis, not a
redundant dictionary), while the square orthonormal member --- Ravishankar and
Bresler's transform learning (IEEE TSP 2013) --- plugs in unchanged:

    min_{W_r, W_c, C_i}  sum_i || W_r X_i W_c^T - C_i ||_F^2
    subject to  W_r, W_c orthonormal  and  ||C_i||_0 <= k

by alternating hard thresholding (the sparse coding step) with an orthogonal
Procrustes update of each axis (one SVD), both closed form: no learning rate
to sweep, so this baseline cannot be under-tuned, only over-fitted (N^2 free
parameters per axis). Initialised at the DCT-II, the strongest fixed row.
The patch variant (one orthonormal transform on P x P patches, applied
blockwise) is still an orthonormal basis of the whole image, so the solver,
the budget rule and mu carry over unchanged.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..coherence import coherence
from ..families.riemannian import reconstruct_mat
from ..metrics import psnr
from ..solver import iht
from .fixed_bases import hard_k

# --------------------------------------------------------------------------
# the transforms


def dct_matrix(N: int) -> np.ndarray:
    """Orthonormal DCT-II, the initialisation and the fixed-basis reference."""
    j, kk = np.arange(N)[None, :], np.arange(N)[:, None]
    D = np.cos(np.pi * (2 * j + 1) * kk / (2 * N)) * np.sqrt(2.0 / N)
    D[0] /= np.sqrt(2.0)
    return D


def analysis_t(X, Wr, Wc):
    """``C = W_r X W_c^T``: dense, O(N^3). This family has no fast form."""
    return Wr @ X @ Wc.T


def synthesis_t(C, Wr, Wc):
    return Wr.T @ C @ Wc


def im2patches(X, P: int):
    """(N, N) -> (N^2/P^2, P^2), non-overlapping patches in row-major order."""
    N = X.shape[0]
    return X.reshape(N // P, P, N // P, P).transpose(0, 2, 1, 3).reshape(-1, P * P)


def patches2im(Ypat, P: int, N: int):
    return Ypat.reshape(N // P, N // P, P, P).transpose(0, 2, 1, 3).reshape(N, N)


def analysis_p(X, W, P: int):
    """Blockwise analysis; the coefficient array is (n_patches, P^2)."""
    return im2patches(X, P) @ W.T


def synthesis_p(C, W, P: int, N: int):
    return patches2im(C @ W, P, N)


def dct2_patch(P: int) -> np.ndarray:
    """The 2-D DCT-II on P x P patches as a P^2 x P^2 matrix: JPEG's basis."""
    return np.kron(dct_matrix(P), dct_matrix(P))


def coherence_t(W) -> float:
    """mu of one separable axis (the matrix is already dense)."""
    return float(coherence(jnp.asarray(W)))


def coherence_patch(W, N: int) -> float:
    """mu of the blockwise operator on an N x N image: ``N^2 max|W_ab|^2``, the
    largest entry of the block-diagonal operator being the largest of W."""
    return float(N * N * np.max(np.abs(np.asarray(W)) ** 2))


def n_params(N: int) -> int:
    """Ambient real parameters per axis (N^2), as the table counts every row."""
    return N * N


# --------------------------------------------------------------------------
# objective and learning


def _procrustes(M: np.ndarray) -> np.ndarray:
    """argmax over orthonormal W of tr(W M): ``W = V U^T`` for ``M = U S V^T``."""
    U, _, Vt = np.linalg.svd(M)
    return Vt.T @ U.T


def sparsification_error(images, analyse, k: int) -> float:
    """``||C - H_k(C)||_F / ||X||_F`` over the images, ``C = analyse(X)``: the
    objective, and the honest measure of whether a basis sparsifies better."""
    num = den = 0.0
    for X in images:
        C = np.asarray(analyse(np.asarray(X)))
        num += float(np.linalg.norm(C - hard_k(C, k)) ** 2)
        den += float(np.linalg.norm(X) ** 2)
    return float(np.sqrt(num / den))


def compress_psnr(images, analyse, synthesise, k: int) -> np.ndarray:
    """PSNR of the k-term approximation of each image: the compression side of
    the reversal, measured for this basis as it is for the fixed ones."""
    return np.array(
        [
            psnr(np.asarray(synthesise(hard_k(np.asarray(analyse(np.asarray(X))), k))), X)
            for X in images
        ]
    )


def learn_transform(
    images, k: int, iters: int = 30, init: str = "dct", log_every: int = 5, val_images=None
) -> dict:
    """Fit the separable orthonormal pair on the training images. With
    ``val_images``, the iterate minimising sparsification error on that split
    is returned as ``"validated"`` next to the ``"final"`` one, since the
    protocol reads every method at its own optimum."""
    X = [np.asarray(im, dtype=np.float64) for im in images]
    N = X[0].shape[0]
    if init not in ("dct", "identity"):
        raise ValueError(f"unknown init {init!r}")
    Wr = dct_matrix(N) if init == "dct" else np.eye(N)
    Wc = Wr.copy()
    V = None if val_images is None else [np.asarray(v, dtype=np.float64) for v in val_images]

    def record(it):
        e = {
            "iter": it,
            "sparsification_error": sparsification_error(X, lambda x: analysis_t(x, Wr, Wc), k),
        }
        if V is not None:
            e["validation_error"] = sparsification_error(V, lambda x: analysis_t(x, Wr, Wc), k)
        return e

    hist = [record(0)]
    best = {
        "Wr": Wr.copy(),
        "Wc": Wc.copy(),
        "iter": 0,
        "validation_error": hist[0].get("validation_error", np.inf),
    }
    for it in range(1, iters + 1):
        C = [hard_k(Wr @ x @ Wc.T, k) for x in X]
        Wr = _procrustes(sum((x @ Wc.T) @ c.T for x, c in zip(X, C)))
        C = [hard_k(Wr @ x @ Wc.T, k) for x in X]
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


def hard_s_rows(C: np.ndarray, s: int) -> np.ndarray:
    """Keep the s largest magnitudes of every row: the per-patch sparsity model
    transform learning is fitted under. One global budget over the image leaves
    most patches without an active coefficient, which makes the Procrustes
    update arbitrary in the unexcited directions while mu drifts to nearly N^2."""
    if s >= C.shape[1]:
        return C
    out = C.copy()
    np.put_along_axis(out, np.argpartition(-np.abs(C), s - 1, axis=1)[:, s:], 0.0, axis=1)
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
    """Alternate per-patch hard thresholding with a Procrustes update, sweeping
    the per-patch sparsity ``s`` (this baseline's capacity knob) and selecting
    it by sparsification error at the *recovery* budget k on the validation split."""
    Y = [im2patches(np.asarray(x, dtype=np.float64), P) for x in images]
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
                    "sparsification_error": sparsification_error(
                        images, lambda x: analysis_p(x, W, P), k
                    ),
                    "validation_error": sparsification_error(ref, lambda x: analysis_p(x, W, P), k),
                    "mu": coherence_patch(W, N),
                }
                sweep.append(e)
                print(
                    f"    s = {sp:>2}  iter {it:>3}  sparsification {e['sparsification_error']:.5f}  "
                    f"validation {e['validation_error']:.5f}  mu {e['mu']:.1f}",
                    flush=True,
                )
                if best is None or e["validation_error"] < best["validation_error"]:
                    best = {**e, "W": W.copy()}
    print(
        f"    validation selects s = {best['s']}, iteration {best['iter']} (validation {best['validation_error']:.5f})",
        flush=True,
    )
    return {"best": best, "sweep": sweep}


# --------------------------------------------------------------------------
# recovery, through the one solver every family shares


def reconstruct_t(Wr, Wc, Y, obs, k, K: int, mode: str = "hard"):
    """K unrolled steps in the learned separable basis. ``W`` is the analysis
    operator and ``W^T`` the synthesis, so it is the dense family's ``U = W^T``."""
    return reconstruct_mat(Wr.T, Wc.T, Y, obs, k, K, mode, remat=False)


@functools.partial(jax.jit, static_argnames=("P", "K", "mode"))
def reconstruct_p(W, Y, obs, P: int, k, K: int, mode: str = "hard"):
    """The same K steps in the blockwise learned basis."""
    W, N = W.astype(Y.dtype), Y.shape[-1]
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
