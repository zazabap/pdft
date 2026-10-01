"""The concrete circuit bases.

Mirror of upstream src/basis.jl. Each class says how its gates are emitted;
everything else (transforms, pytree registration, parameter count) is
``CircuitBasis`` in ``pdft.bases.core``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import numpy as np

from ..circuit.builder import check_qubits
from .circuit.dct4 import dct4_gates
from .circuit.entangled_qft import entangled_qft_gates
from .circuit.mera import _n_mera_gates, mera_gates
from .circuit.qft import qft_gates
from .circuit.tebd import tebd_gates
from .core import AbstractSparseBasis, CircuitBasis, bases_allclose

Array = jax.Array

__all__ = [
    "AbstractSparseBasis",
    "CircuitBasis",
    "DCT4Basis",
    "EntangledQFTBasis",
    "MERABasis",
    "QFTBasis",
    "TEBDBasis",
    "bases_allclose",
]


def _seeded_phases(
    phases: Sequence[float] | None, seed: int | None, count: int
) -> Sequence[float] | None:
    """The given phases, or ``count`` draws from ``N(0, 0.1^2)`` when only a seed is given.

    Julia's ``randn(n_gates) * 0.1`` convention from
    ``ParametricDFT.jl/src/training.jl::_init_circuit``. With neither, the
    emitter's default of zero phases applies.
    """
    if phases is None and seed is not None:
        return list(np.random.default_rng(seed).normal(0.0, 0.1, count))
    return phases


class QFTBasis(CircuitBasis):
    """QFT tensor-network basis. Mirror of `ParametricDFT.jl/src/basis.jl::QFTBasis`."""

    freezes_to_blocked = True

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
    ):
        self._init(qft_gates(m, n), m, n, tensors, code, inv_code)


@dataclass(init=False)
class EntangledQFTBasis(CircuitBasis):
    """QFT + appended entanglement layer on `min(m, n)` row/col qubit pairs.

    Mirror of upstream src/basis.jl:280-500.
    """

    n_entangle: int

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        entangle_phases: Sequence[float] | None = None,
        entangle_position: str = "back",
        code: object | None = None,
        inv_code: object | None = None,
        seed: int | None = None,
    ):
        """Mirror of `_init_circuit(::Type{EntangledQFTBasis}, ...)` from
        ParametricDFT.jl/src/training.jl. A `seed` without `entangle_phases`
        draws the phases, which breaks the symmetry-collapse where
        `EntangledQFTBasis` initialises identically to `QFTBasis` when all
        entanglement phases are zero.
        """
        check_qubits(m, n)
        gates, self.n_entangle = entangled_qft_gates(
            m,
            n,
            entangle_phases=_seeded_phases(entangle_phases, seed, min(m, n)),
            entangle_position=entangle_position,
        )
        self._init(gates, m, n, tensors, code, inv_code)


@dataclass(init=False)
class TEBDBasis(CircuitBasis):
    """2D TEBD basis with row + column rings of controlled-phase gates.

    Mirror of upstream src/basis.jl:600-745.
    """

    n_row_gates: int
    n_col_gates: int

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        phases: Sequence[float] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
        seed: int | None = None,
        parametrization: str = "cp",
    ):
        """A `seed` without `phases` draws the `m + n` ring phases.

        `parametrization` is "cp" (diagonal ring gates on U(1)^4) or "u4"
        (dense two-qubit ring gates on U(4), the canonical TEBD gate).
        """
        check_qubits(m, n)
        gates, self.n_row_gates, self.n_col_gates = tebd_gates(
            m, n, phases=_seeded_phases(phases, seed, m + n), parametrization=parametrization
        )
        self._init(gates, m, n, tensors, code, inv_code)


@dataclass(init=False)
class MERABasis(CircuitBasis):
    """2D MERA basis. Each dimension with >=2 qubits must be a power of 2.

    Mirror of upstream src/basis.jl:840-1070.
    """

    n_row_gates: int
    n_col_gates: int

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        phases: Sequence[float] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
        seed: int | None = None,
        parametrization: str = "cp",
    ):
        """A `seed` without `phases` draws one phase per disentangler and isometry.

        `parametrization` is "cp" (diagonal disentanglers/isometries on
        U(1)^4) or "u4" (dense two-qubit gates on U(4), the canonical form).
        """
        check_qubits(m, n)
        count = sum(_n_mera_gates(q) for q in (m, n) if q >= 2)
        gates, self.n_row_gates, self.n_col_gates = mera_gates(
            m, n, phases=_seeded_phases(phases, seed, count), parametrization=parametrization
        )
        self._init(gates, m, n, tensors, code, inv_code)


class DCT4Basis(CircuitBasis):
    """DCT-IV tensor-network basis: the real-orthogonal, ancilla-free analogue
    of :class:`QFTBasis`.

    The gates are emitted by :func:`pdft.bases.circuit.dct4.dct4_gates`; at
    initialization the forward operator is the bit-reversed orthonormal DCT-IV
    per dimension, and since DCT-IV is self-inverse the basis reconstructs
    exactly. ``tensors`` holds every gate (the affine ``R_y`` rotation layer,
    the branch Hadamards, the mirror-``Q`` CNOT permutations and the ``Delta``
    sign), each a learnable leaf on its auto-selected Riemannian manifold
    (O(2) / O(4) / phase). The gate tensors are real-valued (stored complex128,
    zero imaginary), so the unitary manifold trains the real-orthogonal subset
    under a real objective: exact DCT-IV at init, then relaxed, just as QFT
    relaxes within U.
    """

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
        parametrization: str = "o4",
    ):
        gates = dct4_gates(m, n, parametrization=parametrization)
        self._init(gates, m, n, tensors, code, inv_code)
