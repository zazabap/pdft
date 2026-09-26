"""The coarse-to-fine quantized tensor train of Loeschcke et al. (ICML 2024).

A ``2^m x 2^m`` image is written as a rank-capped tensor train over ``m``
indices of dimension 4 (one quadtree split per core), fitted to the observed
pixels by Adam, and grown coarse to fine: fit at ``64^2``, prolongate to
``128^2``, refit, and so on. Coarse-to-fine wins because averaging a sparsely
observed image into coarse cells makes the coarse problem essentially fully
observed (at 1% nominal sampling the ``64^2`` level still has 92% cell
coverage), so the ill-conditioned sparse problem is only ever met as a
refinement of an already good solution.

One deliberate deviation from the reference implementation: PuTT prolongates
by a linear-interpolation MPO in tensor-train form and then compresses; this
module does the identical operation densely (reconstruct, bilinear upsample,
re-compress by TT-SVD), affordable at these resolutions and much shorter.
Numbers from this module measure this implementation, not the original's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..adam import adam_init, adam_update, apply_updates

Array = jax.Array


def to_quantics(img: Array) -> Array:
    """``(2^m, 2^m)`` image to a ``(4,)*m`` tensor, coarsest bit first.

    Index ``j`` carries ``(bit j of y, bit j of x)`` from the most significant
    end, packed as ``2 by + bx``: one quadtree level per core.
    """
    m = int(np.log2(img.shape[0]))
    t = img.reshape((2,) * (2 * m))
    perm = [i // 2 if i % 2 == 0 else m + i // 2 for i in range(2 * m)]
    return jnp.transpose(t, perm).reshape((4,) * m)


def from_quantics(t: Array) -> Array:
    """The inverse of ``to_quantics``."""
    m = t.ndim
    t = t.reshape((2,) * (2 * m))
    perm = [2 * i for i in range(m)] + [2 * i + 1 for i in range(m)]
    return jnp.transpose(t, perm).reshape((2**m, 2**m))


def tt_svd(t: Array, max_rank: int) -> list[Array]:
    """Sequential TT-SVD of a ``(4,)*m`` tensor with ranks capped at ``max_rank``."""
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


def tt_full(cores) -> Array:
    """The train contracted into its dense ``(4,)*m`` tensor."""
    x = cores[0].reshape(4, -1)
    for c in cores[1:]:
        x = jnp.einsum("ar,rbs->abs", x, c).reshape(-1, c.shape[2])
    return x.reshape((4,) * len(cores))


def tt_image(cores) -> Array:
    """The image a train denotes."""
    return from_quantics(tt_full(cores))


def tt_nparams(cores) -> int:
    """The number of core entries."""
    return int(sum(c.size for c in cores))


def tt_ranks(cores) -> list[int]:
    """The bond dimensions, ``1`` at both ends."""
    return [1] + [int(c.shape[2]) for c in cores]


def _rank_shapes(m: int, max_rank: int) -> list[tuple[int, int, int]]:
    """Core shapes with ranks capped by ``max_rank`` and by the ``4^i`` bound."""
    left = [min(max_rank, 4**i) for i in range(m + 1)]
    right = [min(max_rank, 4 ** (m - i)) for i in range(m + 1)]
    r = [min(a, b) for a, b in zip(left, right)]
    r[0] = r[-1] = 1
    return [(r[i], 4, r[i + 1]) for i in range(m)]


def tt_nparams_at(m: int, max_rank: int) -> int:
    """The entries of a rank-capped train over ``m`` cores, without fitting one; what the table's parameter column reports."""
    return sum(a * b * c for a, b, c in _rank_shapes(m, max_rank))


def masked_pool(img: np.ndarray, obs: np.ndarray, factor: int):
    """Average the observed pixels of each ``factor x factor`` cell.

    Returns ``(target, valid)``; a cell with no observation is not a constraint.
    """
    if factor == 1:
        return img * obs, obs.astype(bool)
    n = img.shape[0] // factor
    s = (img * obs).reshape(n, factor, n, factor).sum(axis=(1, 3))
    c = obs.reshape(n, factor, n, factor).sum(axis=(1, 3))
    return np.where(c > 0, s / np.maximum(c, 1), 0.0), c > 0


def bilinear_up2(x: np.ndarray) -> np.ndarray:
    """The 2x upsample the prolongation applies."""
    n = x.shape[0]
    y = jax.image.resize(jnp.asarray(x, dtype=jnp.float32), (2 * n, 2 * n), method="linear")
    return np.asarray(y, dtype=np.float64)


def _fit_level(cores, target, valid, iters: int, lr: float):
    """Adam on the cores against the observed cells of one resolution level."""
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
    """The ``(resolutions, iteration budgets)`` of the coarse-to-fine ladder.

    Split out of ``fit`` because it matters when timing the method: the ladder
    gains a rung per doubling while ``total_iters`` stays fixed, so the share
    of iterations at full resolution shrinks as the image grows and the cost
    per fit is nearly flat in ``N``.
    """
    if not (coarse_to_fine and init_reso < N):
        return [N], [total_iters]
    resos = []
    r = init_reso
    while r < N:
        resos.append(r)
        r *= 2
    resos.append(N)
    pts = list(upsample_at)[: len(resos) - 1]
    while len(pts) < len(resos) - 1:  # a ladder longer than the schedule: halve the remainder
        pts.append(pts[-1] + (total_iters - pts[-1]) // 2 if pts else total_iters // 2)
    bounds = pts + [total_iters]
    budgets = [bounds[0]] + [bounds[i] - bounds[i - 1] for i in range(1, len(bounds))]
    if min(budgets) <= 0:
        raise ValueError(
            f"total_iters={total_iters} leaves no budget for some level of the upsample schedule {pts}"
        )
    return resos, budgets


def fit(
    img,
    obs,
    init_reso: int = 64,
    max_rank: int = 200,
    total_iters: int = 1024,
    upsample_at=(25, 75, 150, 450),
    lr: float = 0.008,
    lr_decay: float = 0.9,
    seed: int = 42,
    coarse_to_fine: bool = True,
    verbose: bool = False,
) -> list[Array]:
    """Fit a tensor train to the observed pixels of one image.

    ``coarse_to_fine=False`` is the direct control: the same model and budget
    fitted at full resolution from the start.
    """
    img = np.asarray(img, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    N = img.shape[0]
    rng = np.random.default_rng(seed)
    resos, budgets = level_schedule(N, init_reso, total_iters, upsample_at, coarse_to_fine)
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
        cores, v = _fit_level(cores, target, valid, iters, cur_lr)
        if verbose:
            print(
                f"    level {reso:>5}  iters {iters:>4}  loss {v:.3e}  params {tt_nparams(cores):,}  ranks {tt_ranks(cores)}"
            )
    return cores
