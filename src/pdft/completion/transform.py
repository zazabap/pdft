"""The circuit kernel and the per-axis operators every family is built from.

A transform family is one function ``apply(x, params, adjoint, axis)`` that
applies an operator, or its adjoint, along one axis of an image whose length
is ``2**n``. This module holds the operators (the gate circuit and a dense
matrix; the phase-only circuit is the special case in
:mod:`pdft.completion.families.phases`), the DFT anchor every family starts
from, and ``separable``, the 2-D analysis and synthesis pair of any operator.
Register widths are never passed: an axis of length ``2**n`` is the
``n``-wire register, and ``register_width`` reads it off the shape.

The circuit is a product of one-qubit gates ``g`` (one per wire) and diagonal
two-qubit gates with four phases ``phi`` (one per wire pair), followed by a
bit reversal. At the textbook values, Hadamards and phases
``(0, 0, 0, 2 pi / 2^(p-q+1))``, the product is the DFT in the QFT sign
convention, ``U(theta0) == conj(DFT_ortho)``, and every factor is unitary for
every parameter value. :class:`pdft.QFTBasis` carries the same circuit as one
tensor per gate; :mod:`pdft.completion.bridge` converts between the two.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

_H2 = jnp.asarray(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0), dtype=jnp.complex128)


def complex_dtype(x: Array):
    """The complex type ``x`` is carried in: single precision for float32 inputs, double otherwise.

    The transform is unitary, so it neither amplifies nor accumulates error
    across gates; single precision halves the memory of every carry the
    unrolled solver retains.
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
    """The textbook controlled phases; the circuit at ``theta0(n)`` is the DFT."""
    return jnp.asarray([2.0 * np.pi / 2 ** (p - q + 1) for p, q in gate_pairs(n)], dtype=dtype)


def hadamards(n: int) -> Array:
    """One Hadamard per wire: the fixed one-qubit gates of the phase-only families."""
    return jnp.broadcast_to(_H2, (n, 2, 2))


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
    reverse order.
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

    pairs = gate_pairs(n)
    if not adjoint:
        for i, (p, q) in enumerate(pairs):
            if i == 0 or pairs[i - 1][1] != q:
                t = one_qubit(t, q, g[q])
            t = two_qubit(t, p, q, phi[i])
        t = one_qubit(t, n - 1, g[n - 1])  # the last wire has no two-qubit gate
    else:
        t = one_qubit(t, n - 1, jnp.conj(g[n - 1]).T)
        for i in reversed(range(len(pairs))):
            p, q = pairs[i]
            t = two_qubit(t, p, q, -phi[i])
            if i == 0 or pairs[i - 1][1] != q:
                t = one_qubit(t, q, jnp.conj(g[q]).T)
    out = t.reshape(lead + (2**n,))
    if not adjoint:
        out = bitreverse(out)
    return jnp.moveaxis(out, -1, axis)


def apply_dense(x: Array, U: Array, adjoint: bool = False, axis: int = -1) -> Array:
    """Apply a matrix ``U``, or ``U^H``, along one axis.

    The dense counterpart of the circuits, for the free-unitary and the
    transform-learning families. A real matrix keeps a real image real, and
    the matrix takes the image's precision so the solver's carry type is fixed.
    """
    M = jnp.conj(U).T if adjoint else U
    M = M.astype(complex_dtype(x) if jnp.iscomplexobj(M) else jnp.real(x).dtype)
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
