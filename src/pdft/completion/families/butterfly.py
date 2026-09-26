"""A learnable FFT-structured factorisation: the butterfly family of Dao et al. (2019).

It keeps the dataflow of the Cooley-Tukey FFT and frees the 2x2 blocks it is
built from: the natural competitor to the circuit, ``O(N log N)`` and started
at the DFT like it, but with ``2nN`` (unitary) or ``4nN`` (free) parameters per
axis against model B's ``2n(n-1)``, and no pinned coherence, since a butterfly
factor is not diagonal-plus-one-Hadamard and the family can leave the complex
Hadamard set. With ``Pi`` the bit reversal and ``B_l`` the factor mixing
indices at stride ``2^l``,

    U = B_{n-1} ... B_1 B_0 Pi,

and at the DFT every block is ``[[1, w], [1, -w]] / sqrt(2)`` with
``w = exp(+2i pi j / 2^(l+1))`` (a plus sign: the QFT is ``conj(DFT)``).

Two parameterisations, carried in the pytree structure so jit specialises on
it. ``"unitary"`` (``{"gen"}``) writes each block as
``expm(i(a0 I + a . sigma))``, exactly in U(2). ``"free"`` (``{"blk"}``) is an
unconstrained complex 2x2 per block, not an isometry; its ``adjoint=True`` is
the true blockwise inverse rather than the conjugate transpose, the only thing
that keeps the solver's round trip meaningful.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..coherence import coherence, dense_operator
from ..solver import batched, solver_for
from ..training import adam_loop, mu_monitor, task_loss, widths
from ..transform import bitreverse, complex_dtype, register_width, separable

Array = jax.Array

MODES = ("unitary", "free")


def count_params(n: int, mode: str = "unitary") -> int:
    """Real degrees of freedom per axis: ``n`` factors of ``N/2`` blocks of 4 (unitary) or 8 (free) reals."""
    return n * 2 ** (n - 1) * {"unitary": 4, "free": 8}[mode]


def mode_of(params: dict) -> str:
    """Which parameterisation a pytree carries."""
    for key, mode in (("gen", "unitary"), ("blk", "free")):
        if key in params:
            return mode
    raise KeyError(f"not a butterfly parameter dict: keys {sorted(params)}")


def expm_u2(a: Array) -> Array:
    """``exp(i(a0 I + a1 X + a2 Y + a3 Z))`` for ``a`` of shape ``(..., 4)``, in closed form.

    ``exp(i a0) (cos r I + i sinc(r) v.sigma)`` with ``r = |v|``: exact,
    vmappable and analytically differentiable. The ``r = 0`` branch is guarded
    so the gradient through the square root stays finite.
    """
    a0, v = a[..., 0], a[..., 1:]
    r2 = jnp.sum(v * v, axis=-1)
    nz = r2 > 0
    r = jnp.sqrt(jnp.where(nz, r2, 1.0))
    c, s = jnp.where(nz, jnp.cos(r), 1.0), jnp.where(nz, jnp.sin(r) / r, 1.0)
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
    m = jnp.stack(
        [
            jnp.stack([c + 1j * s * v3, s * (1j * v1 + v2)], -1),
            jnp.stack([s * (1j * v1 - v2), c - 1j * s * v3], -1),
        ],
        -2,
    )
    return jnp.exp(1j * a0)[..., None, None] * m


def _logm_u2(M: np.ndarray) -> np.ndarray:
    """The inverse of ``expm_u2`` on U(2), in numpy, used once to sit at the DFT.

    Splits ``M = e^{i alpha} S`` with ``S`` in SU(2) and reads the axis and
    angle off ``S``'s Pauli components; ``arctan2`` rather than ``arccos``
    keeps the digits near ``r = 0`` and ``r = pi``.
    """
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    alpha = np.angle(det) / 2.0
    S = np.exp(-1j * alpha)[..., None, None] * M
    c = np.real(S[..., 0, 0] + S[..., 1, 1]) / 2.0
    n1 = np.imag(S[..., 0, 1] + S[..., 1, 0]) / 2.0
    n2 = np.real(S[..., 0, 1] - S[..., 1, 0]) / 2.0
    n3 = np.imag(S[..., 0, 0] - S[..., 1, 1]) / 2.0
    nrm = np.sqrt(n1**2 + n2**2 + n3**2)
    scale = np.where(nrm > 0, np.arctan2(nrm, c) / np.where(nrm > 0, nrm, 1.0), 0.0)
    return np.stack([alpha, scale * n1, scale * n2, scale * n3], axis=-1)


def dft_blocks(n: int) -> np.ndarray:
    """The ``(n, N/2, 2, 2)`` blocks whose product is ``apply_u(., theta0(n))``.

    The twiddle depends only on the offset within a stage, so every group of a
    stage starts identical; training breaks that, which is exactly the extra
    capacity this family has over the circuit.
    """
    N = 2**n
    out = np.empty((n, N // 2, 2, 2), dtype=np.complex128)
    for lvl in range(n):
        w = np.exp(2j * np.pi * np.arange(2**lvl) / 2 ** (lvl + 1))
        blk = np.stack(
            [np.stack([np.ones_like(w), w], -1), np.stack([np.ones_like(w), -w], -1)], -2
        )
        out[lvl] = np.broadcast_to(blk / np.sqrt(2.0), (N // 2 ** (lvl + 1), 2**lvl, 2, 2)).reshape(
            N // 2, 2, 2
        )
    return out


def init_butterfly(n: int, mode: str = "unitary", dtype=None) -> dict:
    """Parameters for ``n`` factors sitting exactly at the DFT."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    B = dft_blocks(n)
    if mode == "free":
        return {"blk": jnp.asarray(B, dtype=dtype or jnp.complex128)}
    return {"gen": jnp.asarray(_logm_u2(B), dtype=dtype or jnp.float64)}


def blocks(params: dict) -> Array:
    """The ``(n, N/2, 2, 2)`` complex blocks this parameter dict denotes."""
    return expm_u2(params["gen"]) if "gen" in params else params["blk"]


def _inverse_blocks(params: dict) -> Array:
    """Unitary: negate the generator, exactly in U(2). Free: the analytic 2x2 inverse."""
    if "gen" in params:
        return expm_u2(-params["gen"])
    B = params["blk"]
    det = B[..., 0, 0] * B[..., 1, 1] - B[..., 0, 1] * B[..., 1, 0]
    adj = jnp.stack(
        [
            jnp.stack([B[..., 1, 1], -B[..., 0, 1]], -1),
            jnp.stack([-B[..., 1, 0], B[..., 0, 0]], -1),
        ],
        -2,
    )
    return adj / det[..., None, None]


@functools.partial(jax.jit, static_argnames=("adjoint", "axis"))
def apply_butterfly(x: Array, params: dict, adjoint: bool = False, axis: int = -1) -> Array:
    """Apply the butterfly product, or its inverse, along one axis.

    ``O(N log N)``: each stage is ``N/2`` 2x2 blocks applied as two broadcast
    multiplies and an add. A batched matmul lowers twenty times slower and, in
    float32, to TF32.
    """
    n = register_width(x.shape[axis])
    cdtype = complex_dtype(x)
    B = (_inverse_blocks(params) if adjoint else blocks(params)).astype(cdtype)
    if B.shape[0] != n:
        raise ValueError(f"parameters are for another register than the {n} wires of this axis")
    t = jnp.moveaxis(x.astype(cdtype), axis, -1)
    lead, N = t.shape[:-1], 2**n

    def stage(t, lvl):
        # Split the axis as (group, pair bit, offset): stage lvl mixes the
        # indices that differ in bit lvl, i.e. at stride 2**lvl.
        h, j = N >> (lvl + 1), 1 << lvl
        Mb = B[lvl].reshape(h, j, 2, 2)
        tb = t.reshape(lead + (h, 2, j))
        a, b = tb[..., 0, :], tb[..., 1, :]
        out = jnp.stack(
            [Mb[:, :, 0, 0] * a + Mb[:, :, 0, 1] * b, Mb[:, :, 1, 0] * a + Mb[:, :, 1, 1] * b],
            axis=-2,
        )
        return out.reshape(lead + (N,))

    if not adjoint:
        t = bitreverse(t)
        for lvl in range(n):
            t = stage(t, lvl)
    else:  # U^{-1} = Pi B_0^{-1} ... B_{n-1}^{-1}
        for lvl in reversed(range(n)):
            t = stage(t, lvl)
        t = bitreverse(t)
    return jnp.moveaxis(t, -1, axis)


analysis_b, synthesis_b = separable(apply_butterfly)
reconstruct_butterfly = solver_for(apply_butterfly)
reconstruct_butterfly_batch = batched(reconstruct_butterfly)


def unitary_butterfly(params: dict) -> Array:
    """The product formed explicitly. ``O(N^2 log N)``; diagnostics only."""
    return dense_operator(lambda e: apply_butterfly(e, params, axis=0), blocks(params).shape[0])


def coherence_butterfly(params: dict) -> Array:
    """``mu`` of the product. In ``"free"`` mode the operator is not an isometry, so report ``isometry_defect`` with it."""
    return coherence(unitary_butterfly(params))


def isometry_defect(params: dict) -> float:
    """``||U^H U - I||_max``: about 1e-15 in ``"unitary"`` mode, by construction."""
    U = unitary_butterfly(params)
    return float(jnp.abs(jnp.conj(U).T @ U - jnp.eye(U.shape[0], dtype=U.dtype)).max())


def train_butterfly(
    images,
    k,
    K: int = 20,
    p: float = 0.10,
    steps: int = 200,
    lr: float = 2e-3,
    mode: str = "hard",
    param: str = "unitary",
    batch: int = 2,
    seed: int = 0,
    log_every: int = 10,
    remat: bool = True,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Adam on the blocks through the shared loop; the history carries ``mu`` so any drift is on record."""
    images = jnp.asarray(images)
    return adam_loop(
        images,
        {a: init_butterfly(n, param) for a, n in zip(("r", "c"), widths(images))},
        functools.partial(task_loss, reconstruct_butterfly, k=k, K=K, mode=mode, remat=remat),
        lr=lr,
        steps=steps,
        p=p,
        batch=batch,
        seed=seed,
        monitor=mu_monitor(coherence_butterfly),
        log_every=log_every,
        verbose=verbose,
    )
