"""A learnable FFT-structured factorisation --- the butterfly baseline.

Dao et al., "Learning Fast Algorithms for Linear Transforms Using Butterfly
Factorizations" (ICML 2019), and the kaleidoscope (K-matrix) follow-up, learn a
transform by keeping the *dataflow* of the Cooley-Tukey FFT and freeing the 2x2
blocks it is built from. That is the natural competitor to the phase-only
circuit, and a much closer one than a free unitary on U(N): both are
O(N log N), both start at the DFT, both are trained by gradient descent through
the same solver.

    phases (A)   one phase per CP gate; the Hadamards are fixed
                 n(n-1)/2 = 36 params/axis at n=9; mu == 1 everywhere
    diagonals (B)  all four diagonal entries of each CP gate free
                 2n(n-1) = 144 params/axis; mu == 1 everywhere
    butterfly    every 2x2 block of every FFT stage is free
                 2nN = 9,216 (unitary) or 4nN (free) params/axis; mu drifts

B is the opponent, not A. Both satisfy Proposition 1 --- what forces
|U_ij| = N^{-1/2} is one Hadamard per wire with everything else diagonal, and
the phases are free to move without breaking it --- so B differs from this
module in exactly one property while carrying 64x fewer parameters. A butterfly
factor is not diagonal-plus-one-Hadamard: freeing its blocks lets amplitude
concentrate, so the family can and does leave the complex Hadamard set.

Structure. Writing Pi for the bit-reversal permutation and B_l for the factor
that mixes indices at stride 2^l,

    U = B_{n-1} ... B_1 B_0 Pi,

the textbook decimation-in-time FFT: bit-reverse, then n stages of butterflies
of increasing stride. Each B_l is block diagonal with N/2 free 2x2 blocks.
At the DFT the block acting on the pair (a, a + 2^l) is

    [[1,  w], [1, -w]] / sqrt(2),   w = exp(+2i pi j / 2^(l+1)),  j = a mod 2^l,

with a *plus* sign because the circuit at theta0 is the QFT, i.e. conj(DFT).
init_butterfly therefore reproduces transform.apply_u(x, theta0(n), n)
exactly, so the comparison is nested in the same way general.init_general is.

Two parameterisations, selected by init_butterfly(mode=...) and carried in the
pytree structure rather than by a flag, so jax.jit specialises on it and the
optimiser never sees a non-array leaf:

    "unitary"  {"gen"}: each block is expm(i(a0 I + a1 X + a2 Y + a3 Z)),
               4 real params, exactly in U(2) at every parameter value. The
               product is then unitary by construction and adjoint=True is the
               exact inverse. This is the headline baseline: apples to apples
               with the circuit except for the coherence pinning.
    "free"     {"blk"}: an unconstrained complex 2x2 per block, 8 real params.
               Honest "learnable FFT structure", but not an isometry during
               training: ||A(X)||_F != ||X||_F, so keeping the k largest
               coefficients no longer approximates an orthogonal projection.
               adjoint=True inverts each block and reverses the factor order,
               which is the true inverse and the only thing that keeps the
               solver's synthesis(H_k(analysis(X))) round trip meaningful; it
               is *not* the conjugate transpose here.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..coherence import coherence as _mu
from ..coherence import dense_operator
from ..solver import iht
from ..transform import complex_dtype

MODES = ("unitary", "free")


# --------------------------------------------------------------------------
# shape bookkeeping


def n_factors(n: int) -> int:
    """One butterfly factor per bit --- the n stages of the FFT."""
    return n


def n_blocks(n: int) -> int:
    """2x2 blocks per factor. Every index is in exactly one pair, so N/2."""
    return 2 ** (n - 1)


def count_params(n: int, mode: str = "unitary") -> int:
    """Real DOF per axis. 2nN unitary, 4nN free; the phase-only circuit is n(n-1)/2."""
    per_block = {"unitary": 4, "free": 8}[mode]
    return n_factors(n) * n_blocks(n) * per_block


def mode_of(params: dict) -> str:
    """Which parameterisation a pytree carries. Static: it is the structure."""
    if "gen" in params:
        return "unitary"
    if "blk" in params:
        return "free"
    raise KeyError(f"not a butterfly parameter dict: keys {sorted(params)}")


# --------------------------------------------------------------------------
# U(2) as the exponential of an anti-Hermitian generator


def expm_u2(a: jnp.ndarray) -> jnp.ndarray:
    """exp(i(a0 I + a1 X + a2 Y + a3 Z)) for a of shape (..., 4).

    Closed form rather than jax.scipy.linalg.expm: for 2x2 the Pauli expansion
    is exact, it vmaps over N/2 blocks without a Pade iteration per block, and
    it has an analytic derivative, which matters when this sits inside K
    unrolled solver steps. i(v.sigma) = i[[v3, v1 - i v2], [v1 + i v2, -v3]],
    so the whole thing is

        exp(i a0) * (cos r I + i sinc(r) v.sigma),   r = |v|,

    which is in U(2) for every real a, exactly.
    """
    a0, v = a[..., 0], a[..., 1:]
    r2 = jnp.sum(v * v, axis=-1)
    # Double-where: the r2 = 0 branch would otherwise put a NaN in the
    # gradient through sqrt. r2 = 0 means the block is a phase times I, which
    # no DFT butterfly block is, so the branch is never taken at or near init.
    nz = r2 > 0
    r = jnp.sqrt(jnp.where(nz, r2, 1.0))
    c = jnp.where(nz, jnp.cos(r), 1.0)
    s = jnp.where(nz, jnp.sin(r) / r, 1.0)
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
    m00 = c + 1j * s * v3
    m01 = s * (1j * v1 + v2)
    m10 = s * (1j * v1 - v2)
    m11 = c - 1j * s * v3
    m = jnp.stack([jnp.stack([m00, m01], -1), jnp.stack([m10, m11], -1)], -2)
    return jnp.exp(1j * a0)[..., None, None] * m


def _logm_u2(M: np.ndarray) -> np.ndarray:
    """Inverse of expm_u2 on U(2), in numpy: used once, to sit at the DFT.

    Split M = e^{i alpha} S with S in SU(2) via alpha = arg(det M)/2, then read
    (r, n) off the Pauli components of S. arctan2 rather than arccos(tr S / 2)
    because arccos loses half its digits near r = 0 and r = pi.
    """
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    alpha = np.angle(det) / 2.0
    S = np.exp(-1j * alpha)[..., None, None] * M
    c = np.real(S[..., 0, 0] + S[..., 1, 1]) / 2.0  # tr(S)/2
    n1 = np.imag(S[..., 0, 1] + S[..., 1, 0]) / 2.0  # Im tr(S X)/2
    n2 = np.real(S[..., 0, 1] - S[..., 1, 0]) / 2.0  # Im tr(S Y)/2
    n3 = np.imag(S[..., 0, 0] - S[..., 1, 1]) / 2.0  # Im tr(S Z)/2
    nrm = np.sqrt(n1**2 + n2**2 + n3**2)
    r = np.arctan2(nrm, c)
    scale = np.where(nrm > 0, r / np.where(nrm > 0, nrm, 1.0), 0.0)
    return np.stack([alpha, scale * n1, scale * n2, scale * n3], axis=-1)


# --------------------------------------------------------------------------
# initialisation at the DFT


def dft_blocks(n: int) -> np.ndarray:
    """The (n, N/2, 2, 2) blocks whose product is transform.apply_u(., theta0).

    Block (l, h*2^l + j) acts on the pair (h*2^(l+1) + j, h*2^(l+1) + j + 2^l)
    and equals [[1, w], [1, -w]]/sqrt(2). The twiddle depends on j only, so
    every one of the N/2^(l+1) groups of a stage starts identical --- training
    is what breaks that, and breaking it is exactly the extra capacity this
    baseline has over the circuit.
    """
    N = 2**n
    out = np.empty((n, N // 2, 2, 2), dtype=np.complex128)
    for lvl in range(n):
        j = np.arange(2**lvl)
        w = np.exp(2j * np.pi * j / 2 ** (lvl + 1))
        blk = np.empty((2**lvl, 2, 2), dtype=np.complex128)
        blk[:, 0, 0], blk[:, 0, 1] = 1.0, w
        blk[:, 1, 0], blk[:, 1, 1] = 1.0, -w
        blk /= np.sqrt(2.0)
        out[lvl] = np.broadcast_to(blk[None], (N // 2 ** (lvl + 1), 2**lvl, 2, 2)).reshape(
            N // 2, 2, 2
        )
    return out


def init_butterfly(n: int, mode: str = "unitary", dtype=None) -> dict:
    """Parameters for n butterfly factors, sitting exactly at the DFT.

    Same anchoring as init_params and init_general: training deforms the FFT
    factorisation rather than appending a learned correction to it, so any PSNR
    difference against the circuit is a difference of parameterisation and not
    of starting point.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    B = dft_blocks(n)
    if mode == "free":
        return {"blk": jnp.asarray(B, dtype=dtype or jnp.complex128)}
    return {"gen": jnp.asarray(_logm_u2(B), dtype=dtype or jnp.float64)}


def blocks(params: dict) -> jnp.ndarray:
    """The (n, N/2, 2, 2) complex blocks this parameter dict denotes."""
    return expm_u2(params["gen"]) if "gen" in params else params["blk"]


def _inverse_blocks(params: dict) -> jnp.ndarray:
    """Blockwise inverse. Unitary: negate the generator, which stays in U(2)
    exactly rather than to the accuracy of a matrix conjugation. Free: the
    analytic 2x2 inverse --- the conjugate transpose would not invert the map,
    and the solver's round trip depends on it doing so."""
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


# --------------------------------------------------------------------------
# the transform


@functools.partial(jax.jit, static_argnames=("n", "adjoint", "axis"))
def apply_butterfly(x, params: dict, n: int, adjoint: bool = False, axis: int = -1):
    """Apply the butterfly product (or its inverse) along one axis of length 2**n.

    O(N log N): each stage is N/2 blocks of size 2x2 applied elementwise, and
    the matrix is never formed. adjoint=True is the *inverse*, which is the
    Hermitian adjoint in "unitary" mode and is not in "free" mode.
    """
    cdtype = complex_dtype(x)
    B = (_inverse_blocks(params) if adjoint else blocks(params)).astype(cdtype)
    x = jnp.moveaxis(x.astype(cdtype), axis, -1)
    lead = x.shape[:-1]
    nl = len(lead)
    N = 2**n
    t = x

    def bitreverse(t):
        t = t.reshape(lead + (2,) * n)
        t = jnp.transpose(t, tuple(range(nl)) + tuple(nl + n - 1 - i for i in range(n)))
        return t.reshape(lead + (N,))

    def stage(t, lvl, M):
        # Split the axis as (group, pair-bit, offset) = (h, i, j): stage lvl
        # mixes indices differing in bit lvl, i.e. at stride 2**lvl. Written as
        # the two broadcast multiplies and one add per output that a 2x2
        # block is, rather than a batched 2x2 matmul: XLA lowers the einsum to
        # a batched matmul, twenty times slower than the elementwise gates of
        # apply_general and, in float32, run in TF32.
        h, j = N >> (lvl + 1), 1 << lvl
        Mb = M.reshape(h, j, 2, 2)
        tb = t.reshape(lead + (h, 2, j))
        a, b = tb[..., 0, :], tb[..., 1, :]  # (..., h, j)
        out = jnp.stack(
            [
                Mb[:, :, 0, 0] * a + Mb[:, :, 0, 1] * b,
                Mb[:, :, 1, 0] * a + Mb[:, :, 1, 1] * b,
            ],
            axis=-2,
        )
        return out.reshape(lead + (N,))

    if not adjoint:
        t = bitreverse(t)
        for lvl in range(n):
            t = stage(t, lvl, B[lvl])
    else:
        # U^{-1} = Pi^{-1} B_0^{-1} ... B_{n-1}^{-1}; Pi is an involution.
        for lvl in reversed(range(n)):
            t = stage(t, lvl, B[lvl])
        t = bitreverse(t)
    return jnp.moveaxis(t, -1, axis)


def analysis_b(X, pr: dict, pc: dict, n: int):
    """A(X) = U(pr)^{-1} applied along both axes --- mirrors analysis_g."""
    C = apply_butterfly(X, pr, n, adjoint=True, axis=-2)
    return apply_butterfly(C, pc, n, adjoint=True, axis=-1)


def synthesis_b(C, pr: dict, pc: dict, n: int):
    """The exact inverse of analysis_b, in both parameterisations."""
    X = apply_butterfly(C, pr, n, adjoint=False, axis=-2)
    return apply_butterfly(X, pc, n, adjoint=False, axis=-1)


# --------------------------------------------------------------------------
# diagnostics --- mu is imported, never redefined


def unitary_butterfly(params: dict, n: int) -> jnp.ndarray:
    """Form the 2^n x 2^n matrix. O(N^2 log N); diagnostics only."""
    return dense_operator(lambda e: apply_butterfly(e, params, n, adjoint=False, axis=0), n)


def coherence_butterfly(params: dict, n: int) -> jnp.ndarray:
    """mu = N max|U_ij|^2, the definition in pdft.completion.coherence.

    In "unitary" mode this lands in [1, N] as it should. In "free" mode the
    operator is not an isometry, so mu is still N max|U_ij|^2 but no longer
    bounded by N; report isometry_defect alongside it or the number misleads.
    """
    return _mu(unitary_butterfly(params, n))


def isometry_defect(params: dict, n: int) -> float:
    """||U^H U - I||_max. ~1e-15 in "unitary" mode, by construction."""
    U = unitary_butterfly(params, n)
    return float(jnp.abs(jnp.conj(U).T @ U - jnp.eye(2**n, dtype=U.dtype)).max())


# --------------------------------------------------------------------------
# the unrolled solver, identical to solver.reconstruct but butterfly-valued


@functools.partial(jax.jit, static_argnames=("n", "K", "mode", "remat"))
def reconstruct_butterfly(
    pr: dict, pc: dict, Y, obs, n: int, k: int, K: int, mode: str = "hard", remat: bool = True
):
    """K unrolled solver steps with the butterfly basis."""
    return iht(
        lambda X: analysis_b(X, pr, pc, n),
        lambda C: synthesis_b(C, pr, pc, n),
        Y,
        obs,
        k,
        K,
        mode,
        remat,
    )


def reconstruct_butterfly_batch(pr, pc, Y, obs, n, k, K, mode="hard", remat=True):
    """vmap over a leading batch axis; k is per-image, as in solver."""
    f = functools.partial(reconstruct_butterfly, n=n, k=k, K=K, mode=mode, remat=remat)
    return jax.vmap(f, in_axes=(None, None, 0, 0))(pr, pc, Y, obs)


def evaluate_butterfly(params, images, n, p, frac, K, seed, mode="hard"):
    """Held-out PSNR of a butterfly basis under the shared protocol."""
    from ..protocol import evaluate

    return evaluate(
        lambda Y, obs, k: reconstruct_butterfly(
            params["r"], params["c"], Y, obs, n, k, K, mode=mode, remat=False
        ),
        images,
        p,
        frac,
        seed,
    )


# --------------------------------------------------------------------------
# training --- same objective, same optimiser, same mask schedule


def task_loss_b(params, X, obs, n, k, K, mode, remat=True):
    """The task objective with the butterfly basis, so the only change is the family."""
    Xh = reconstruct_butterfly_batch(params["r"], params["c"], X * obs, obs, n, k, K, mode, remat)
    return jnp.mean((Xh - X) ** 2)


def train_butterfly(
    images,
    n,
    k,
    K=20,
    p=0.10,
    steps=200,
    lr=2e-3,
    mode="hard",
    param="unitary",
    batch=2,
    seed=0,
    log_every=10,
    remat=True,
    verbose=True,
):
    """Adam on the butterfly blocks, through training.adam_loop --- so the trained
    families differ only in the circuit they parameterise, never in the
    schedule. Returns (params, history); history carries mu so any drift is on
    record."""
    from ..training import adam_loop

    def loss_fn(params, X, obs):
        return task_loss_b(params, X, obs, n, k, K, mode, remat)

    def monitor(params):
        return {
            "mu_r": float(coherence_butterfly(params["r"], n)),
            "mu_c": float(coherence_butterfly(params["c"], n)),
        }

    params = {"r": init_butterfly(n, param), "c": init_butterfly(n, param)}
    return adam_loop(
        jnp.asarray(images),
        params,
        loss_fn,
        lr=lr,
        steps=steps,
        p=p,
        batch=batch,
        seed=seed,
        monitor=monitor,
        log_every=log_every,
        verbose=verbose,
    )
