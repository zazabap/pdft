"""The phase-only basis: a deformed quantum Fourier circuit, unitary at every angle.

The circuit is

    U(theta) = Pi * prod_{q=n-1}^{0} [ prod_{p=q+1}^{n-1} CP_{pq}(theta_pq) ] H_q,

with Pi the bit-reversal permutation and CP a controlled-phase gate. The
product is ordered: the q = 0 factors stand rightmost and are applied first
(H_0, then CP_{p,0}, then H_1, ...), which is the loop order below. Two
properties do all the work: every factor is unitary for every angle, so
gradient descent on theta never leaves the unitary group; and at the textbook
angles theta0_pq = 2*pi / 2^(p-q+1) the circuit is exactly the DFT (in the QFT
sign convention, so ``U(theta0) == conj(DFT_ortho)``; see ``theta0``).

Gates are applied directly to the image rather than by forming U, so a
transform costs O(N log N) per axis and the n(n-1)/2 angles are the only
parameters. Qubit 0 is the most significant bit of the index along the axis.

This is the same circuit family as :class:`pdft.QFTBasis`, carried in a
different representation: the core package stores one tensor per gate and
trains the tensors on their Riemannian manifolds, while this module stores the
angles and trains them by plain Adam (the phases cover the torus through
``exp(i*phi)``, so no retraction is needed). The two representations are
related exactly, including a bit-reversal of the image; see
:mod:`pdft.completion.bridge`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

_INV_SQRT2 = float(1.0 / np.sqrt(2.0))


def complex_dtype(x):
    """The complex type this input should be carried in.

    Single precision halves both the coefficient array and every carry the
    unrolled solver retains, which is the difference between fitting and not
    fitting at 4096^2. The transform is unitary, so it neither amplifies nor
    accumulates error across gates and single precision is well behaved here;
    :mod:`pdft.completion.unroll` exposes the choice.
    """
    return jnp.complex64 if x.dtype in (jnp.float32, jnp.complex64) else jnp.complex128


def gate_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """(p, q) of every controlled-phase gate, in circuit order.

    The i-th entry names the gate whose angle is ``theta[i]``.
    """
    return tuple((p, q) for q in range(n) for p in range(q + 1, n))


def n_params(n: int) -> int:
    return n * (n - 1) // 2


def n_from_params(n_angles: int) -> int:
    """The register width whose phase-only circuit has ``n_angles`` gates."""
    n = int(round((1 + (1 + 8 * n_angles) ** 0.5) / 2))
    if n_params(n) != n_angles:
        raise ValueError(f"{n_angles} is not n(n-1)/2 for any integer n")
    return n


def theta0(n: int, dtype=jnp.float64) -> jnp.ndarray:
    """The textbook angles. ``U(theta0(n))`` is the DFT exactly (QFT sign)."""
    return jnp.asarray([2.0 * np.pi / 2 ** (p - q + 1) for p, q in gate_pairs(n)], dtype=dtype)


@functools.partial(jax.jit, static_argnames=("n", "adjoint", "axis"))
def apply_u(
    x: jnp.ndarray, theta: jnp.ndarray, n: int, adjoint: bool = False, axis: int = -1
) -> jnp.ndarray:
    """Apply U(theta) (or U(theta)^H) along one axis of length 2**n."""
    cdtype = complex_dtype(x)
    x = jnp.moveaxis(x.astype(cdtype), axis, -1)
    lead = x.shape[:-1]
    nl = len(lead)
    t = x.reshape(lead + (2,) * n)

    def qax(q: int) -> int:
        return nl + q

    def hadamard(t, q):
        a = jnp.take(t, 0, axis=qax(q))
        b = jnp.take(t, 1, axis=qax(q))
        return jnp.stack([(a + b) * _INV_SQRT2, (a - b) * _INV_SQRT2], axis=qax(q))

    def cphase(t, p, q, angle):
        idx = [slice(None)] * t.ndim
        idx[qax(p)] = 1
        idx[qax(q)] = 1
        return t.at[tuple(idx)].multiply(jnp.exp(1j * angle.astype(cdtype)))

    def bitreverse(t):
        return jnp.transpose(t, tuple(range(nl)) + tuple(nl + n - 1 - i for i in range(n)))

    if not adjoint:
        g = 0
        for q in range(n):
            t = hadamard(t, q)
            for p in range(q + 1, n):
                t = cphase(t, p, q, theta[g])
                g += 1
        t = bitreverse(t)
    else:
        # U^H undoes each factor in reverse order; H is self-adjoint, the
        # bit reversal is an involution, and CP(a)^H = CP(-a).
        t = bitreverse(t)
        g = n_params(n)
        for q in reversed(range(n)):
            for p in reversed(range(q + 1, n)):
                g -= 1
                t = cphase(t, p, q, -theta[g])
            t = hadamard(t, q)

    return jnp.moveaxis(t.reshape(lead + (2**n,)), -1, axis)


def analysis(X: jnp.ndarray, theta_r: jnp.ndarray, theta_c: jnp.ndarray, n: int) -> jnp.ndarray:
    """A_theta(X) = U(theta_r)^H X conj(U(theta_c)).

    Both axes contract against U^H; see the note in :func:`synthesis`.
    """
    C = apply_u(X, theta_r, n, adjoint=True, axis=-2)
    return apply_u(C, theta_c, n, adjoint=True, axis=-1)


def synthesis(C: jnp.ndarray, theta_r: jnp.ndarray, theta_c: jnp.ndarray, n: int) -> jnp.ndarray:
    """S_theta(C) = U(theta_r) C U(theta_c)^T --- the exact inverse of analysis.

    (X conj(U))_ib = sum_j X_ij conj(U_jb) = (U^H x)_b applied along axis 1, and
    (C U^T)_ib = sum_j U_bj C_ij = (U c)_b, so each map applies the same
    operator along both axes.
    """
    X = apply_u(C, theta_r, n, adjoint=False, axis=-2)
    return apply_u(X, theta_c, n, adjoint=False, axis=-1)


def unitary_matrix(theta: jnp.ndarray, n: int) -> jnp.ndarray:
    """Form U(theta) explicitly. O(N^2 log N); for diagnostics only."""
    from .coherence import dense_operator

    return dense_operator(lambda e: apply_u(e, theta, n, adjoint=False, axis=0), n)


def coherence(theta: jnp.ndarray, n: int) -> jnp.ndarray:
    """mu(theta) = N max_ij |U_ij|^2 in [1, N]. See :mod:`pdft.completion.coherence`."""
    from .coherence import coherence as _mu

    return _mu(unitary_matrix(theta, n))
