"""Coarse-to-fine quantized tensor train --- the PuTT baseline.

Reimplements the method of Loeschcke et al., "Coarse-To-Fine Tensor Trains for
Compact Visual Representations" (ICML 2024, arXiv:2406.04332), which is the
incumbent the paper's rate-sweep comparison is measured against.

A 2^m x 2^m image is written as a rank-capped tensor train over m indices of
dimension 4 (one quadtree split per core), fitted to the observed pixels by
Adam, and grown coarse-to-fine: fit at 64^2, prolongate to 128^2, refit, and so
on to full resolution.

Why coarse-to-fine wins is not subtle: averaging a sparsely observed image into
coarse cells makes the coarse problem *essentially fully observed* (at 1%
nominal sampling the 64^2 level still has 92% cell coverage), so the
ill-conditioned sparse problem is only ever met as a refinement of an already
good solution.

Deviation from the reference implementation, deliberate and documented:
PuTT prolongates by applying a linear-interpolation MPO (Lubasch et al. 2018)
in tensor-train form and then compressing. This module performs the identical
operation densely --- reconstruct, bilinear-upsample, re-compress by TT-SVD ---
which is affordable at these resolutions and much shorter. Numbers from this
module measure this implementation, not the original's floor; cite the method,
never this table.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..adam import adam_init, adam_update, apply_updates

# --------------------------------------------------------------------------
# quantics layout


def to_quantics(img: jnp.ndarray) -> jnp.ndarray:
    """(2^m, 2^m) image -> (4,)*m tensor, coarsest bit first.

    Index j of the result carries (bit j of y, bit j of x) counted from the
    most significant end, packed as 2*by + bx --- one quadtree level per core.
    """
    m = int(np.log2(img.shape[0]))
    t = img.reshape((2,) * (2 * m))  # y0,y1,..,x0,x1,..
    perm = [i // 2 if i % 2 == 0 else m + i // 2 for i in range(2 * m)]
    return jnp.transpose(t, perm).reshape((4,) * m)


def from_quantics(t: jnp.ndarray) -> jnp.ndarray:
    """Inverse of to_quantics."""
    m = t.ndim
    t = t.reshape((2,) * (2 * m))  # y0,x0,y1,x1,..
    perm = [2 * i for i in range(m)] + [2 * i + 1 for i in range(m)]
    return jnp.transpose(t, perm).reshape((2**m, 2**m))


# --------------------------------------------------------------------------
# tensor train


def tt_svd(t: jnp.ndarray, max_rank: int) -> list[jnp.ndarray]:
    """Sequential TT-SVD of a (4,)*m tensor, ranks capped at max_rank."""
    m = t.ndim
    cores = []
    rest = np.asarray(t, dtype=np.float64).reshape(1, -1)
    r = 1
    for _ in range(m - 1):
        rest = rest.reshape(r * 4, -1)
        u, s, vt = np.linalg.svd(rest, full_matrices=False)
        rnew = min(max_rank, len(s), rest.shape[0], rest.shape[1])
        cores.append(u[:, :rnew].reshape(r, 4, rnew))
        rest = s[:rnew, None] * vt[:rnew]
        r = rnew
    cores.append(rest.reshape(r, 4, 1))
    return [jnp.asarray(c, dtype=jnp.float32) for c in cores]


def tt_full(cores) -> jnp.ndarray:
    """Contract the train into the dense (4,)*m tensor."""
    x = cores[0].reshape(4, -1)
    for c in cores[1:]:
        x = jnp.einsum("ar,rbs->abs", x, c).reshape(-1, c.shape[2])
    return x.reshape((4,) * len(cores))


def tt_image(cores) -> jnp.ndarray:
    return from_quantics(tt_full(cores))


def tt_nparams(cores) -> int:
    return int(sum(c.size for c in cores))


def tt_ranks(cores) -> list[int]:
    return [1] + [int(c.shape[2]) for c in cores]


def tt_nparams_at(m: int, max_rank: int) -> int:
    """Free parameters of a rank-capped train over m cores, without fitting one.

    The capacity knob is max_rank, so this is what Table I's Parameters column
    reports for this baseline: the count is fixed by the knob and the image
    size, and _rank_shapes is exactly the shape tt_svd produces at every level.
    """
    return sum(a * b * c for a, b, c in _rank_shapes(m, max_rank))


# --------------------------------------------------------------------------
# masked pooling: the mechanism that makes coarse levels easy


def masked_pool(img: np.ndarray, obs: np.ndarray, factor: int):
    """Average the observed pixels of each factor x factor cell.

    Returns (target, valid). A cell with no observation is not a constraint.
    """
    if factor == 1:
        return img * obs, obs.astype(bool)
    n = img.shape[0] // factor
    s = (img * obs).reshape(n, factor, n, factor).sum(axis=(1, 3))
    c = obs.reshape(n, factor, n, factor).sum(axis=(1, 3))
    return np.where(c > 0, s / np.maximum(c, 1), 0.0), c > 0


def bilinear_up2(x: np.ndarray) -> np.ndarray:
    """2x upsample --- the interpolation the prolongation MPO applies."""
    n = x.shape[0]
    y = jax.image.resize(jnp.asarray(x, dtype=jnp.float32), (2 * n, 2 * n), method="linear")
    return np.asarray(y, dtype=np.float64)


# --------------------------------------------------------------------------
# fitting


def _fit_level(cores, target, valid, iters, lr, seed=0):
    """Adam on the cores against the observed cells of one resolution level."""
    del seed  # the level is deterministic given its initial cores
    tgt = jnp.asarray(target, dtype=jnp.float32)
    msk = jnp.asarray(valid, dtype=jnp.float32)
    denom = jnp.maximum(msk.sum(), 1.0)

    def loss(cs):
        return jnp.sum(((tt_image(cs) - tgt) * msk) ** 2) / denom

    state = adam_init(cores)

    @jax.jit
    def step(cs, st):
        v, g = jax.value_and_grad(loss)(cs)
        upd, st = adam_update(g, st, lr)
        return apply_updates(cs, upd), st, v

    v = jnp.asarray(0.0)
    for _ in range(iters):
        cores, state, v = step(cores, state)
    return cores, float(v)


def level_schedule(
    N: int, init_reso: int, total_iters: int, upsample_at, coarse_to_fine: bool = True
):
    """(resolutions, iteration budgets) of the coarse-to-fine ladder.

    Split out of fit() because the split matters when timing the method: the
    ladder gains a rung per doubling while total_iters stays fixed, so the
    share of iterations that reach FULL resolution shrinks as the image grows.
    Cost per fit is therefore nearly flat in N, and reading that as cheap
    scaling without also reading this would be wrong.
    """
    if not (coarse_to_fine and init_reso < N):
        return [N], [total_iters]
    resos = []
    r = init_reso
    while r < N:
        resos.append(r)
        r *= 2
    resos.append(N)
    # One upsample point per level transition; the last level runs to the end.
    pts = list(upsample_at)[: len(resos) - 1]
    while len(pts) < len(resos) - 1:  # ladder longer than the schedule
        pts.append(pts[-1] + (total_iters - pts[-1]) // 2 if pts else total_iters // 2)
    bounds = pts + [total_iters]
    budgets = [bounds[0]] + [bounds[i] - bounds[i - 1] for i in range(1, len(bounds))]
    if min(budgets) <= 0:
        raise ValueError(
            f"total_iters={total_iters} leaves no budget for some level of the "
            f"upsample schedule {pts}"
        )
    return resos, budgets


def fit(
    img,
    obs,
    init_reso=64,
    max_rank=200,
    total_iters=1024,
    upsample_at=(25, 75, 150, 450),
    lr=0.008,
    lr_decay=0.9,
    seed=42,
    coarse_to_fine=True,
    verbose=False,
):
    """Fit a QTT to the observed pixels of one image.

    coarse_to_fine=False is the 'direct' control: the same model and budget
    fitted at full resolution from the start.
    """
    img = np.asarray(img, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    N = img.shape[0]
    rng = np.random.default_rng(seed)

    resos, budgets = level_schedule(N, init_reso, total_iters, upsample_at, coarse_to_fine)

    # Coarsest level: random init, fitted to data only.
    m0 = int(np.log2(resos[0]))
    cores = [
        jnp.asarray(rng.normal(0, 0.1, size=s), dtype=jnp.float32)
        for s in _rank_shapes(m0, max_rank)
    ]

    cur_lr = lr
    for lvl, (reso, iters) in enumerate(zip(resos, budgets)):
        if lvl > 0:
            dense = np.asarray(tt_image(cores), dtype=np.float64)
            cores = tt_svd(to_quantics(jnp.asarray(bilinear_up2(dense))), max_rank)
            cur_lr *= lr_decay
        target, valid = masked_pool(img, obs, N // reso)
        cores, v = _fit_level(cores, target, valid, iters, cur_lr, seed)
        if verbose:
            print(
                f"    level {reso:>5}  iters {iters:>4}  loss {v:.3e}  "
                f"params {tt_nparams(cores):,}  ranks {tt_ranks(cores)}"
            )
    return cores


def _rank_shapes(m: int, max_rank: int):
    """Core shapes with ranks capped by max_rank and by the 4^i bound."""
    left = [min(max_rank, 4**i) for i in range(m + 1)]
    right = [min(max_rank, 4 ** (m - i)) for i in range(m + 1)]
    r = [min(a, b) for a, b in zip(left, right)]
    r[0] = r[-1] = 1
    return [(r[i], 4, r[i + 1]) for i in range(m)]
