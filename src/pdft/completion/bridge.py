"""Exact conversion between the two representations of the QFT circuit family.

The core package (:class:`pdft.QFTBasis`) and the completion subpackage carry
the same circuit --- one Hadamard per wire, one controlled-phase gate per pair
--- in two forms:

* ``QFTBasis`` stores one tensor per gate, contracts them by einsum in Yao's
  little-endian convention (qubit 1 is the least significant bit and gets its
  Hadamard first, with no final swap layer), and trains the tensors on their
  Riemannian manifolds. The compact controlled-phase tensor is
  ``[[1, 1], [1, e^{i phi}]]`` and ``PhaseManifold`` leaves all four of its
  entries free, so a trained ``QFTBasis`` is exactly the "QFT + diagonals"
  relaxation with the Hadamards on U(2) ("QFT + rotations").
* :mod:`pdft.completion.families.general` stores the one-qubit gates ``g``
  and the four phases ``phi`` of each two-qubit gate, applies them to the
  image directly with the most significant bit as qubit 0 and a final bit
  reversal, and trains ``phi`` by plain Adam.

Applied gate by gate in the same axis frame, the two sequences are each
other's reverse, so the operators are transposes of one another up to that
missing bit reversal. Concretely, with ``Pi`` the bit-reversal permutation of
an axis and ``U(theta)`` the operator :func:`pdft.completion.transform.apply_u`
applies, a ``QFTBasis`` built by :func:`qft_basis_from_angles` satisfies

    basis.forward_transform(X) == U(theta_r)^T (Pi X Pi) U(theta_c)
                               == conj( analysis(Pi conj(X) Pi, theta_r, theta_c) )

for every angle and every complex image (for a real image the inner
conjugation drops out), ``analysis`` being the completion subpackage's
analysis map ``U^H X conj(U)``; the coefficient magnitudes, hence the top-k
support, agree between the two once the image is bit-reversed. The one-qubit gates map to their
transposes for the same reason (a Hadamard is symmetric, so phase-only bases
need no transposition at all), and the two-qubit phases are re-indexed to the
other convention's bit order. Every function here is exact, tested at random
parameters to round-off, and inverse to its partner.

Why it matters: a completion-trained transform can be serialised by
:func:`pdft.io.save_basis`, certified by :func:`pdft.coherence.certify_flat_modulus`
and drawn by :mod:`pdft.viz`, and a ``QFTBasis`` trained for compression can be
handed to the completion solver, with :func:`bitrev_image` accounting for the
axis convention. Coherence needs no adjustment: ``mu`` is invariant under
permutation, transposition and conjugation, so ``pdft.coherence.coherence`` of
the converted basis equals the product of the per-axis
``coherence_general`` values.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..bases.base import QFTBasis
from ..bases.circuit.qft import _qft_gates_1d
from ..circuit.builder import sorted_gate_program
from .families.general import _H2, init_general
from .transform import gate_pairs, n_from_params, n_params

__all__ = [
    "angles_from_qft_basis",
    "bitrev_image",
    "general_from_qft_basis",
    "qft_basis_from_angles",
    "qft_basis_from_general",
]


def _bitrev_index(n: int) -> np.ndarray:
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)])


def bitrev_image(X):
    """``Pi X Pi``: reverse the bits of the index along both axes.

    The pixel permutation that relates the two conventions; an involution.
    Axis lengths must be powers of two.
    """
    X = jnp.asarray(X)
    out = X
    for axis, size in enumerate(X.shape[-2:]):
        n = int(round(np.log2(size)))
        if 2**n != size:
            raise ValueError(f"axis of length {size} is not a power of two")
        out = jnp.take(out, jnp.asarray(_bitrev_index(n)), axis=axis + X.ndim - 2)
    return out


def _program(m: int, n: int):
    """The gate program of ``QFTBasis(m, n)`` in stored (Hadamard-first) order,
    with each slot mapped to the completion convention: ``("H", reg, q)`` or
    ``("CP", reg, i)`` where reg is 0 (rows) or 1 (columns), q the completion
    qubit, i the index into ``gate_pairs``."""
    gates = _qft_gates_1d(m, 0) + _qft_gates_1d(n, m)
    slots = []
    for kind, qubits in sorted_gate_program(gates):
        reg = 0 if max(qubits) <= m else 1
        width, offset = (m, 0) if reg == 0 else (n, m)
        if kind == "H":
            (j,) = qubits
            slots.append(("H", reg, width - (j - offset)))
        else:
            t, q = qubits  # pdft: control = the later qubit t, target = q
            pair = (width - (q - offset), width - (t - offset))
            slots.append(("CP", reg, gate_pairs(width).index(pair)))
    return slots


def qft_basis_from_general(pr: dict, pc: dict) -> QFTBasis:
    """A ``QFTBasis`` carrying the ``{"g", "phi"}`` parameters of both axes."""
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
    """The ``{"g", "phi"}`` parameters of both axes of a ``QFTBasis``.

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


def qft_basis_from_angles(theta_r, theta_c) -> QFTBasis:
    """A ``QFTBasis`` for the phase-only angles of both axes (fixed Hadamards)."""
    pr = init_general(n_from_params(len(theta_r)))
    pc = init_general(n_from_params(len(theta_c)))
    pr = {"g": pr["g"], "phi": pr["phi"].at[:, 3].set(jnp.asarray(theta_r))}
    pc = {"g": pc["g"], "phi": pc["phi"].at[:, 3].set(jnp.asarray(theta_c))}
    return qft_basis_from_general(pr, pc)


def angles_from_qft_basis(basis: QFTBasis, atol: float = 1e-6) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(theta_r, theta_c)`` of a phase-only ``QFTBasis``.

    Raises if a Hadamard has moved or a controlled-phase gate carries a phase
    off its (1, 1) entry: such a basis is "QFT + diagonals" or "QFT +
    rotations", and :func:`general_from_qft_basis` is the converter for it.
    """
    pr, pc = general_from_qft_basis(basis, atol=atol)
    out = []
    for p in (pr, pc):
        if not jnp.allclose(p["g"], _H2, atol=atol):
            raise ValueError("a Hadamard has been trained; the basis is not phase-only")
        rest = jnp.angle(jnp.exp(1j * p["phi"][:, :3]))
        if not jnp.allclose(rest, 0.0, atol=atol):
            raise ValueError("a controlled-phase gate has off-(1,1) phases; not phase-only")
        out.append(p["phi"][:, 3])
    return out[0], out[1]
