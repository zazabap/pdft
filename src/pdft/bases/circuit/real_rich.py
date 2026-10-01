"""RealRichBasis (Approach A): real-orthogonal restriction of RichBasis.

Motivation: RichBasis (54 free real params per dim, complex U(4) gates) is
a strict 54-dim submanifold of SU(8) — does not contain DCT, can only
approximate. But fully complex U(4) is also wasteful for the natural-image
problem: DCT is real-valued (lives in O(8)), so the IMAGINARY parts of
RichBasis's parameters are doing no useful work for natural images.

RealRichBasis keeps the same QFT topology and gate count but constrains
each tensor to be REAL-valued:
  - 3 H gates per dim → 2×2 real-orthogonal matrices (init: Hadamard).
    Free params per gate: 1 (rotation angle of the connected component
    of Hadamard in O(2)).
  - 3 "U(4)" gates per dim → real 4×4 orthogonal matrices, 6 free real
    params each (init: identity).

Total per dim: 3·1 + 3·6 = 21 free real params (BELOW dim O(8) = 28).
Strict submanifold of O(8); whether DCT is in this family is empirical.

Storage: tensors are stored as complex128 with all-zero imaginary parts.
Cayley retraction on the real-image gradient preserves real-ness
automatically — no manifold change needed; the existing UnitaryManifold
trains the orthogonal subset correctly when initialised with real values
on a real-valued objective.
"""

from __future__ import annotations

import jax

from ...circuit.builder import Gate, identity_tensor, two_registers
from ..core import CircuitBasis
from .qft import qft_gates_1d

Array = jax.Array


def _real_rich_qft_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """QFT topology with H + REAL-orthogonal 2-qubit gates.

    H slots are the canonical Hadamard. 2-qubit slots are the 4×4 identity
    (a real-orthogonal matrix). Both are within the connected component of
    O(d) reachable via Cayley retraction with real updates.
    """
    eye_u4 = identity_tensor("U4")

    def identity(q_ctrl: int, q_tgt: int, phi: float) -> Gate:
        # The QFT phase of the slot is not used: every slot starts at the identity.
        return Gate(kind="U4", qubits=(q_ctrl, q_tgt), tensor=eye_u4, phase=0.0)

    return qft_gates_1d(n_qubits, offset, identity)


def real_rich_gates(m: int, n: int) -> list[Gate]:
    """The gate sequence of the real-rich (H + real-orthogonal 2-qubit) circuit."""
    return two_registers(_real_rich_qft_gates_1d, m, n)


class RealRichBasis(CircuitBasis):
    """QFT topology with H + real-orthogonal 2-qubit gates.

    The U(4) slots are *initialised* to the 4×4 identity (real-orthogonal,
    not the complex controlled-phase) so the basis is NOT bit-identical
    to QFTBasis at training step 0 — the forward circuit at init is
    H ⊗ H ⊗ H per dim followed by identity 2-qubit ops, i.e. just the
    Walsh-Hadamard transform. This is the appropriate starting point for
    a real-valued search; the Walsh-Hadamard is the simplest real-orthogonal
    basis and a natural baseline for natural-image transforms.
    """

    emit = staticmethod(real_rich_gates)
    freezes_to_blocked = True


__all__ = ["RealRichBasis"]
