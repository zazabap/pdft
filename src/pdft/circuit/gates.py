"""The QFT circuit applied gate by gate, and the per-axis operators built on it.

:class:`pdft.QFTBasis` stores the circuit as one tensor per gate and contracts
them with an einsum. This module carries the same circuit as its parameters,
the one-qubit gates ``g`` and the four phases ``phi`` of each two-qubit gate,
and applies them to the image directly: no matrix is formed, any register
width works, and the image's dtype sets the precision. It is not part of the
Julia port and has no goldens; the property tests in
``tests/circuit/test_gates.py`` are its correctness criterion.

A per-axis operator is one function ``apply(x, params, adjoint, axis)`` that
applies an operator, or its adjoint, along one axis of length ``2**n``.
``apply_gates`` is the circuit, ``apply_dense`` a matrix, and ``separable``
gives the 2-D analysis and synthesis pair of any such operator. Register
widths are never passed: an axis of length ``2**n`` is the ``n``-wire
register, and ``register_width`` reads it off the shape.

The circuit is a product of one-qubit gates ``g`` (one per wire) and diagonal
two-qubit gates with four phases ``phi`` (one per wire pair), followed by a
bit reversal. At the textbook values, Hadamards and phases
``(0, 0, 0, 2 pi / 2^(p-q+1))``, the product is the DFT in the QFT sign
convention, ``U(theta0) == conj(DFT_ortho)``, and every factor is unitary for
every parameter value. The phase-only circuit, one angle per gate, is the
special case ``theta_to_params(theta)``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .builder import HADAMARD

Array = jax.Array


def complex_dtype(x: Array):
    """The complex type ``x`` is carried in: single precision for float32 inputs, double otherwise.

    The transform is unitary, so it neither amplifies nor accumulates error
    across gates; single precision halves the memory of every intermediate a
    gradient through it retains.
    """
    return jnp.complex64 if x.dtype in (jnp.float32, jnp.complex64) else jnp.complex128


def register_width(size: int) -> int:
    """The ``n`` with ``2**n == size``: the register an axis of that length is."""
    n = int(size).bit_length() - 1
    if 1 << n != size:
        raise ValueError(f"axis of length {size} is not a power of two")
    return n


def gate_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """The ``(p, q)`` of every two-qubit gate in circuit order; gate ``i`` carries ``phi[i]``."""
    return tuple((p, q) for q in range(n) for p in range(q + 1, n))


def n_params(n: int) -> int:
    """The number of two-qubit gates, ``n(n-1)/2``."""
    return n * (n - 1) // 2


def n_from_params(n_angles: int) -> int:
    """The register width whose circuit has ``n_angles`` two-qubit gates."""
    n = int(round((1 + (1 + 8 * n_angles) ** 0.5) / 2))
    if n_params(n) != n_angles:
        raise ValueError(f"{n_angles} is not n(n-1)/2 for any integer n")
    return n


def theta0(n: int, dtype=jnp.float64) -> Array:
    """The textbook controlled phases; the circuit at ``theta0(n)`` is ``conj(DFT_ortho)``."""
    return jnp.asarray([2.0 * np.pi / 2 ** (p - q + 1) for p, q in gate_pairs(n)], dtype=dtype)


def hadamards(n: int) -> Array:
    """One Hadamard per wire: the fixed one-qubit gates of the phase-only circuit."""
    return jnp.broadcast_to(HADAMARD, (n, 2, 2))


def theta_to_params(theta: Array) -> dict:
    """The phase-only angles as a ``{"g", "phi"}`` gate dict: Hadamards and phases ``(0, 0, 0, theta)``."""
    theta = jnp.asarray(theta)
    n = n_from_params(theta.shape[0])
    phi = jnp.zeros((n_params(n), 4), theta.dtype).at[:, 3].set(theta)
    return {"g": hadamards(n), "phi": phi}


@functools.lru_cache
def bitrev_index(n: int) -> tuple[int, ...]:
    """Every index below ``2**n`` with its ``n`` bits reversed."""
    return tuple(int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n))


def bitreverse(x: Array, axis: int = -1) -> Array:
    """The bit reversal ``Pi`` along one axis. An involution."""
    return jnp.take(x, jnp.asarray(bitrev_index(register_width(x.shape[axis]))), axis=axis)


@functools.partial(jax.jit, static_argnames=("adjoint", "axis"))
def apply_gates(x: Array, params: dict, adjoint: bool = False, axis: int = -1) -> Array:
    """Apply the circuit ``{"g", "phi"}``, or its adjoint, along one axis.

    Wire 0 is the most significant bit of the index. Forward, for each wire
    ``q`` in turn: ``g[q]``, then the two-qubit gates ``(p, q)`` with
    ``p > q``; finally the bit reversal. The adjoint undoes each factor in
    reverse order. Gate ``i = (p, q)`` multiplies by ``exp(1j * phi[i][2 * b_p
    + b_q])`` with ``b_p`` and ``b_q`` the bits of its two wires; the
    phase-only circuit uses slot 3 alone.
    """
    n = register_width(x.shape[axis])
    cdtype = complex_dtype(x)
    g, phi = params["g"].astype(cdtype), params["phi"]
    if g.shape[0] != n or phi.shape[0] != n_params(n):
        raise ValueError(f"parameters are for another register than the {n} wires of this axis")
    x = jnp.moveaxis(x.astype(cdtype), axis, -1)
    if adjoint:
        x = bitreverse(x)
    lead = x.shape[:-1]
    nl = len(lead)
    t = x.reshape(lead + (2,) * n)

    def one_qubit(t, q, U):
        a, b = jnp.take(t, 0, axis=nl + q), jnp.take(t, 1, axis=nl + q)
        return jnp.stack([U[0, 0] * a + U[0, 1] * b, U[1, 0] * a + U[1, 1] * b], axis=nl + q)

    def two_qubit(t, p, q, ph):
        # One broadcast multiply by the 2x2 phase table, indexed [b_q, b_p]
        # since q < p. Four scatters instead produce an XLA graph that takes
        # longer to compile than a training run takes.
        table = jnp.exp(1j * ph.astype(cdtype)).reshape(2, 2).T
        shape = [1] * t.ndim
        shape[nl + q] = shape[nl + p] = 2
        return t * table.reshape(shape)

    # The circuit written once: each wire's own gate (no partner), then its
    # two-qubit gates. The adjoint walks the same list backwards and inverts
    # each factor, so the two directions cannot drift apart.
    index = {pair: i for i, pair in enumerate(gate_pairs(n))}
    circuit = [(q, p) for q in range(n) for p in (None, *range(q + 1, n))]
    for q, p in reversed(circuit) if adjoint else circuit:
        if p is None:
            t = one_qubit(t, q, jnp.conj(g[q]).T if adjoint else g[q])
        else:
            ph = phi[index[p, q]]
            t = two_qubit(t, p, q, -ph if adjoint else ph)
    out = t.reshape(lead + (2**n,))
    if not adjoint:
        out = bitreverse(out)
    return jnp.moveaxis(out, -1, axis)


def apply_dense(x: Array, U: Array, adjoint: bool = False, axis: int = -1) -> Array:
    """Apply a matrix ``U``, or ``U^H``, along one axis.

    The dense counterpart of the circuit, for a transform given as a matrix.
    A real matrix keeps a real image real, and the matrix takes the image's
    precision, so the output type is fixed by the image alone. An integer or
    boolean image is carried in double precision, as ``apply_gates`` carries it.
    """
    M = jnp.conj(U).T if adjoint else U
    cdtype = complex_dtype(x)
    M = M.astype(cdtype if jnp.iscomplexobj(M) else jnp.finfo(cdtype).dtype)
    return jnp.moveaxis(jnp.tensordot(M, jnp.moveaxis(x, axis, 0), axes=1), 0, axis)


def separable(apply: Callable) -> tuple[Callable, Callable]:
    """The 2-D analysis and synthesis pair of a per-axis operator.

    ``analysis(X, pr, pc) = U(pr)^H X conj(U(pc))`` applies the adjoint along
    both axes and ``synthesis(C, pr, pc) = U(pr) C U(pc)^T`` is its exact
    inverse: ``(X conj(U))_ib = sum_j X_ij conj(U_jb)`` is ``U^H`` along axis 1,
    and ``(C U^T)_ib = sum_j U_bj C_ij`` is ``U`` along axis 1.
    """

    def analysis(X: Array, pr, pc) -> Array:
        """The coefficients of ``X``: the operator's adjoint along both axes."""
        return apply(apply(X, pr, True, -2), pc, True, -1)

    def synthesis(C: Array, pr, pc) -> Array:
        """The image of the coefficients ``C``: the exact inverse of ``analysis``."""
        return apply(apply(C, pr, False, -2), pc, False, -1)

    return analysis, synthesis


def axis_operator(apply_fn: Callable, n: int, dtype=jnp.complex128) -> Array:
    """The ``2^n x 2^n`` matrix of a per-axis transform given as a closure.

    ``apply_fn`` maps an array to its transform along axis 0. A diagnostic,
    ``N`` transforms of length ``N``: nothing that applies an operator needs
    its matrix.
    """
    return apply_fn(jnp.eye(2**n, dtype=dtype))


def gate_matrix(params: dict) -> Array:
    """The circuit ``{"g", "phi"}`` formed explicitly as a matrix. A diagnostic, like ``axis_operator``."""
    return axis_operator(lambda e: apply_gates(e, params, axis=0), params["g"].shape[0])
