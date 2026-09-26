"""Exact conversion between the two representations of the QFT circuit family.

:class:`pdft.QFTBasis` stores one tensor per gate, contracts them in Yao's
little-endian convention (qubit 1 is the least significant bit and gets its
Hadamard first, with no final swap layer) and trains the tensors on their
Riemannian manifolds. The compact controlled-phase tensor is
``[[1, 1], [1, e^{i phi}]]`` and ``PhaseManifold`` leaves all four entries
free, so a trained ``QFTBasis`` is exactly the "QFT + diagonals" relaxation
with its Hadamards on U(2). The completion families store the one-qubit gates
``g`` and the four phases ``phi`` of each two-qubit gate, apply them with the
most significant bit as wire 0 and a final bit reversal, and train ``phi`` by
plain Adam.

Applied gate by gate in the same axis frame the two sequences are each other's
reverse, so the operators are transposes of one another up to that missing bit
reversal. With ``Pi`` the bit reversal of an axis and ``U`` the operator of
:func:`pdft.completion.families.phases.apply_u`, a basis built by
``qft_basis_from_angles`` satisfies, for every angle and every complex image,

    basis.forward_transform(X) == U(theta_r)^T (Pi X Pi) U(theta_c)
                               == conj( analysis(Pi conj(X) Pi, theta_r, theta_c) )

(for a real image the inner conjugation drops out), so the coefficient
magnitudes, and with them the top-k support, agree once the image is
bit-reversed. One-qubit gates map to their transposes (a Hadamard is symmetric,
so phase-only bases need none) and the two-qubit phases are re-indexed to the
other bit order. Every function here is exact, tested at random parameters to
round-off, and inverse to its partner.

A completion-trained transform can therefore be serialised by
:func:`pdft.io.save_basis`, certified by :func:`pdft.coherence.certify_flat_modulus`
and drawn by :mod:`pdft.viz`, and a ``QFTBasis`` trained for compression can be
handed to the completion solver with ``bitrev_image`` accounting for the axis
convention. Coherence needs no adjustment: ``mu`` is invariant under
permutation, transposition and conjugation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..bases.base import QFTBasis
from ..bases.circuit.qft import _qft_gates_1d
from ..circuit.builder import sorted_gate_program
from .transform import bitreverse, gate_pairs, hadamards, n_params, theta_to_params

Array = jax.Array

__all__ = [
    "angles_from_qft_basis",
    "bitrev_image",
    "general_from_qft_basis",
    "qft_basis_from_angles",
    "qft_basis_from_general",
]


def bitrev_image(X: Array) -> Array:
    """``Pi X Pi``: the bits of the index reversed along both axes. An involution."""
    return bitreverse(bitreverse(X, -2), -1)


def _program(m: int, n: int) -> list[tuple[str, int, int]]:
    """The stored (Hadamard-first) gate order of ``QFTBasis(m, n)`` in the completion convention.

    Each slot is ``("H", reg, q)`` or ``("CP", reg, i)`` with ``reg`` 0 for
    the row register and 1 for the column register, ``q`` the completion wire
    and ``i`` the index into ``gate_pairs``.
    """
    gates = _qft_gates_1d(m, 0) + _qft_gates_1d(n, m)
    slots = []
    for kind, qubits in sorted_gate_program(gates):
        reg = 0 if max(qubits) <= m else 1
        width, offset = (m, 0) if reg == 0 else (n, m)
        if kind == "H":
            (j,) = qubits
            slots.append(("H", reg, width - (j - offset)))
        else:
            t, q = qubits  # the core's control is the later qubit t, its target q
            pair = (width - (q - offset), width - (t - offset))
            slots.append(("CP", reg, gate_pairs(width).index(pair)))
    return slots


def qft_basis_from_general(pr: dict, pc: dict) -> QFTBasis:
    """A ``QFTBasis`` carrying the ``{"g", "phi"}`` gate dicts of both axes."""
    m, n = pr["g"].shape[0], pc["g"].shape[0]
    params = (pr, pc)
    tensors = []
    for kind, reg, i in _program(m, n):
        if kind == "H":
            tensors.append(jnp.asarray(params[reg]["g"][i], dtype=jnp.complex128).T)
        else:
            phi = jnp.asarray(params[reg]["phi"][i], dtype=jnp.float64)
            tensors.append(jnp.exp(1j * phi).reshape(2, 2).T.astype(jnp.complex128))
    return QFTBasis(m=m, n=n, tensors=tensors)


def general_from_qft_basis(basis: QFTBasis, atol: float = 1e-6) -> tuple[dict, dict]:
    """The ``{"g", "phi"}`` gate dicts of both axes of a ``QFTBasis``.

    The controlled-phase tensors must be unit-modulus (the phase manifold keeps
    them so); anything else has no representation in this family and raises.
    """
    if not isinstance(basis, QFTBasis):
        raise TypeError(f"expected a QFTBasis, got {type(basis).__name__}")
    m, n = basis.m, basis.n
    out = [
        {"g": np.zeros((m, 2, 2), complex), "phi": np.zeros((n_params(m), 4))},
        {"g": np.zeros((n, 2, 2), complex), "phi": np.zeros((n_params(n), 4))},
    ]
    for (kind, reg, i), T in zip(_program(m, n), basis.tensors):
        T = np.asarray(T)
        if kind == "H":
            out[reg]["g"][i] = T.T
        else:
            if np.abs(np.abs(T) - 1.0).max() > atol:
                raise ValueError("a controlled-phase tensor is not unit-modulus; not a phase gate")
            out[reg]["phi"][i] = np.angle(T.T).reshape(-1)
    return tuple({k: jnp.asarray(v) for k, v in p.items()} for p in out)


def qft_basis_from_angles(theta_r: Array, theta_c: Array) -> QFTBasis:
    """A ``QFTBasis`` for the phase-only angles of both axes, Hadamards fixed."""
    return qft_basis_from_general(theta_to_params(theta_r), theta_to_params(theta_c))


def angles_from_qft_basis(basis: QFTBasis, atol: float = 1e-6) -> tuple[Array, Array]:
    """``(theta_r, theta_c)`` of a phase-only ``QFTBasis``.

    Raises if a Hadamard has moved or a controlled-phase gate carries a phase
    off its ``(1, 1)`` entry: such a basis is "QFT + diagonals" or "QFT +
    rotations", and ``general_from_qft_basis`` is its converter.
    """
    pr, pc = general_from_qft_basis(basis, atol=atol)
    out = []
    for p in (pr, pc):
        if not jnp.allclose(p["g"], hadamards(p["g"].shape[0]), atol=atol):
            raise ValueError("a Hadamard has been trained; the basis is not phase-only")
        rest = jnp.angle(jnp.exp(1j * p["phi"][:, :3]))
        if not jnp.allclose(rest, 0.0, atol=atol):
            raise ValueError("a controlled-phase gate has off-(1,1) phases; not phase-only")
        out.append(p["phi"][:, 3])
    return out[0], out[1]
