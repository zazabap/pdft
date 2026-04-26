"""RichBasis: single-layer parametric circuit with full 2-qubit unitary gates.

Motivation: at small block sizes (m=n=3 = 8x8) the existing H+CP gate
family hits an expressivity ceiling — all topologies converge ~1.75 dB
below 8x8 DCT. The cause is that diagonal CP gates have only 1 free
parameter each. A general 2-qubit unitary (U(4)) has 15 free parameters,
and the H + U(4) gate family is provably universal for SU(2^n) at any
qubit count >= 2, so it CONTAINS DCT as a special case.

RichBasis emits the same QFT topology gate sequence (H per qubit + 2-qubit
gates between qubit pairs) but each 2-qubit gate is a learnable U(4)
instead of a 1-parameter CP. Initialised so the circuit is BIT-IDENTICAL
to QFT at training step 0 (each U(4) gate equals the 4×4 controlled-phase
diag(1, 1, 1, exp(iφ)) at its standard QFT phase). This gives Adam a
gentle starting point: the optimiser begins exactly where plain QFT does
and can only improve.

Parameter count at m=n=3 (8x8 block):
  - 6 H gates (3 per dim) at 4 real params each = 24
  - 6 U(4) gates (3 per dim) at 15 real params each = 90
  - total: 114 real params per dim
  vs SU(8) dimension = 63 free real params
  → strictly more parameters than needed for any 8x8 unitary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import tree_util

from ._circuit import HADAMARD, Gate, compile_circuit, u4_from_phase

Array = jax.Array


def _rich_qft_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """Same QFT topology as qft._qft_gates_1d, but with U(4) gates instead of CP.

    Each U(4) gate is initialised to the 4x4 unitary equivalent of the
    standard QFT phase (so the basis is bit-identical to QFTBasis at init).
    """
    gates: list[Gate] = []
    for j in range(1, n_qubits + 1):
        q = offset + j
        gates.append(Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))
        for target in range(j + 1, n_qubits + 1):
            k = target - j + 1
            t = offset + target
            phi = float(2 * jnp.pi / (2**k))
            gates.append(
                Gate(
                    kind="U4",
                    qubits=(t, q),  # control, target
                    tensor=u4_from_phase(phi),
                    phase=phi,
                )
            )
    return gates


def _rich_code(m: int, n: int, *, inverse: bool):
    """Build the rich (H + U(4)) circuit einsum + initial tensor list."""
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    gates = _rich_qft_gates_1d(m, offset=0) + _rich_qft_gates_1d(n, offset=m)
    return compile_circuit(gates, m, n, inverse=inverse)


@dataclass
class RichBasis:
    """QFT topology with H + learnable U(4) gates instead of H + CP.

    Drop-in replacement for QFTBasis with strictly more expressivity:
    initialised identically (bit-equal forward output at training step 0)
    but the optimiser is free to deform the U(4) gates away from
    diagonal CP into any 4×4 unitary on each qubit pair.

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
        _code, init_tensors = _rich_code(m, n, inverse=False)
        _inv_code, _ = _rich_code(m, n, inverse=True)
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


def _richbasis_flatten(b: RichBasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _richbasis_unflatten(aux, leaves) -> RichBasis:
    m, n, n_fwd, code, inv_code = aux
    assert len(leaves) == n_fwd
    return RichBasis(m=m, n=n, tensors=list(leaves), code=code, inv_code=inv_code)


tree_util.register_pytree_node(RichBasis, _richbasis_flatten, _richbasis_unflatten)


__all__ = ["RichBasis"]
