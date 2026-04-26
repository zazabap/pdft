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

from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ._circuit import HADAMARD, Gate, compile_circuit

Array = jax.Array


def _real_eye_u4() -> Array:
    """4x4 identity reshaped to (2, 2, 2, 2), as real complex128."""
    return jnp.asarray(np.eye(4).reshape(2, 2, 2, 2), dtype=jnp.complex128)


def _real_rich_qft_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """QFT topology with H + REAL-orthogonal 2-qubit gates.

    H slots are the canonical Hadamard. 2-qubit slots are the 4×4 identity
    (a real-orthogonal matrix). Both are within the connected component of
    O(d) reachable via Cayley retraction with real updates.
    """
    eye_u4 = _real_eye_u4()
    gates: list[Gate] = []
    for j in range(1, n_qubits + 1):
        q = offset + j
        gates.append(Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))
        for target in range(j + 1, n_qubits + 1):
            t = offset + target
            gates.append(
                Gate(
                    kind="U4",
                    qubits=(t, q),
                    tensor=eye_u4,
                    phase=0.0,
                )
            )
    return gates


def _real_rich_code(m: int, n: int, *, inverse: bool):
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    gates = _real_rich_qft_gates_1d(m, offset=0) + _real_rich_qft_gates_1d(n, offset=m)
    return compile_circuit(gates, m, n, inverse=inverse)


@dataclass
class RealRichBasis:
    """QFT topology with H + real-orthogonal 2-qubit gates.

    The U(4) slots are *initialised* to the 4×4 identity (real-orthogonal,
    not the complex controlled-phase) so the basis is NOT bit-identical
    to QFTBasis at training step 0 — the forward circuit at init is
    H ⊗ H ⊗ H per dim followed by identity 2-qubit ops, i.e. just the
    Walsh-Hadamard transform. This is the appropriate starting point for
    a real-valued search; the Walsh-Hadamard is the simplest real-orthogonal
    basis and a natural baseline for natural-image transforms.

    Pytree contract:
        leaves   = tensors                                (one list)
        aux data = (m, n, len(tensors), code, inv_code)
    """

    m: int
    n: int
    tensors: list[Array]
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
    ):
        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        _code, init_tensors = _real_rich_code(m, n, inverse=False)
        _inv_code, _ = _real_rich_code(m, n, inverse=True)
        self.tensors = list(tensors) if tensors is not None else init_tensors
        self.code = code if code is not None else _code
        self.inv_code = inv_code if inv_code is not None else _inv_code

    @property
    def inv_tensors(self) -> list[Array]:
        return self.tensors

    @property
    def image_size(self) -> tuple[int, int]:
        return (2**self.m, 2**self.n)

    @property
    def num_parameters(self) -> int:
        return sum(int(t.size) for t in self.tensors)

    def forward_transform(self, pic: Array) -> Array:
        from .loss import _apply_circuit

        return _apply_circuit(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        from .loss import _apply_circuit

        return _apply_circuit(
            [jnp.conj(t) for t in self.tensors],
            self.inv_code,
            self.m,
            self.n,
            pic,
        )


def _realrichbasis_flatten(b: RealRichBasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _realrichbasis_unflatten(aux, leaves) -> RealRichBasis:
    m, n, n_fwd, code, inv_code = aux
    assert len(leaves) == n_fwd
    return RealRichBasis(m=m, n=n, tensors=list(leaves), code=code, inv_code=inv_code)


tree_util.register_pytree_node(RealRichBasis, _realrichbasis_flatten, _realrichbasis_unflatten)


__all__ = ["RealRichBasis"]
