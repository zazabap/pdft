"""Sparse basis abstraction + concrete bases.

Mirror of upstream src/basis.jl. Phase 1 shipped QFTBasis; Phase 3 adds
EntangledQFTBasis, TEBDBasis, and MERABasis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import tree_util

from .circuit.qft import qft_code

Array = jax.Array


@runtime_checkable
class AbstractSparseBasis(Protocol):
    @property
    def image_size(self) -> tuple[int, int]: ...
    @property
    def num_parameters(self) -> int: ...
    def forward_transform(self, pic: Array) -> Array: ...
    def inverse_transform(self, pic: Array) -> Array: ...


@dataclass
class QFTBasis:
    """QFT tensor-network basis. Mirror of `ParametricDFT.jl/src/basis.jl::QFTBasis`.

    Stores ONE tensor list. `inverse_transform` applies `conj(tensors)` through
    `inv_code` — exactly like Julia's `inverse_transform(basis::QFTBasis, ...)`
    which does `basis.inverse_code(conj.(basis.tensors)..., ...)`.

    `code` and `inv_code` are jit-compiled einsum closures and compare by
    identity; they are marked `compare=False` so the dataclass-generated
    `__eq__` doesn't consider them. Use `bases_allclose` for semantic
    comparison.

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
        _code, init_tensors = qft_code(m, n)
        _inv_code, _ = qft_code(m, n, inverse=True)
        self.tensors = list(tensors) if tensors is not None else init_tensors
        self.code = code if code is not None else _code
        self.inv_code = inv_code if inv_code is not None else _inv_code

    @property
    def inv_tensors(self) -> list[Array]:
        """Back-compat alias: Julia stores one tensor list. The "inverse"
        is computed by applying `conj(tensors)` through `inv_code`. We
        expose this property so older callers that read `basis.inv_tensors`
        keep working — they get the same arrays as `basis.tensors`.
        """
        return self.tensors

    # ------ AbstractSparseBasis interface ------

    @property
    def image_size(self) -> tuple[int, int]:
        return (2**self.m, 2**self.n)

    @property
    def num_parameters(self) -> int:
        return sum(int(t.size) for t in self.tensors)

    def forward_transform(self, pic: Array) -> Array:
        from .circuit.qft import ft_mat

        return ft_mat(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        from .circuit.qft import ift_mat

        return ift_mat(
            [jnp.conj(t) for t in self.tensors],
            self.inv_code,
            self.m,
            self.n,
            pic,
        )


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------


def _qftbasis_flatten(b: QFTBasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _qftbasis_unflatten(aux, leaves) -> QFTBasis:
    m, n, n_fwd, code, inv_code = aux
    assert len(leaves) == n_fwd
    tensors = list(leaves)
    return QFTBasis(m=m, n=n, tensors=tensors, code=code, inv_code=inv_code)


tree_util.register_pytree_node(QFTBasis, _qftbasis_flatten, _qftbasis_unflatten)


# ---------------------------------------------------------------------------
# Equality helper
# ---------------------------------------------------------------------------


def bases_allclose(a, b, *, atol: float = 1e-10) -> bool:
    """Semantic equality across any of the registered basis types.

    Checks same concrete type, same (m, n), and all `tensors` within `atol`
    (rtol=0). `code` / `inv_code` are ignored. Per Julia's design we no
    longer store a separate `inv_tensors` to compare.
    """
    if type(a) is not type(b):
        return False
    if (a.m, a.n) != (b.m, b.n):
        return False
    if len(a.tensors) != len(b.tensors):
        return False
    for x, y in zip(a.tensors, b.tensors):
        if not jnp.allclose(x, y, atol=atol, rtol=0.0):
            return False
    return True


# ---------------------------------------------------------------------------
# EntangledQFTBasis
# ---------------------------------------------------------------------------


@dataclass
class EntangledQFTBasis:
    """QFT + appended entanglement layer on `min(m, n)` row/col qubit pairs.

    Mirror of upstream src/basis.jl:280-500 (entangle_position=:back only).
    """

    m: int
    n: int
    n_entangle: int
    tensors: list[Array]
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

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
        ParametricDFT.jl/src/training.jl. When `seed` is provided AND
        `entangle_phases` is None, draws phases from
        `np.random.default_rng(seed).normal(0, 0.1, n_entangle)` — Julia's
        `randn(n_gates) * 0.1` convention. This breaks the symmetry-collapse
        where `EntangledQFTBasis` initialises identically to `QFTBasis` when
        all entanglement phases are zero.
        """
        import numpy as np

        from .circuit.entangled_qft import entangled_qft_code

        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        n_entangle = min(m, n)
        if entangle_phases is None and seed is not None:
            entangle_phases = list(np.random.default_rng(seed).normal(0.0, 0.1, n_entangle))
        _code, init_tensors, self.n_entangle = entangled_qft_code(
            m, n, entangle_phases=entangle_phases, entangle_position=entangle_position
        )
        _inv_code, _, _ = entangled_qft_code(
            m, n, entangle_phases=entangle_phases, entangle_position=entangle_position, inverse=True
        )
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
        from ..circuit.builder import apply_circuit

        return apply_circuit(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        from ..circuit.builder import apply_circuit

        return apply_circuit(
            [jnp.conj(t) for t in self.tensors],
            self.inv_code,
            self.m,
            self.n,
            pic,
        )


def _entangled_flatten(b: EntangledQFTBasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, b.n_entangle, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _entangled_unflatten(aux, leaves) -> EntangledQFTBasis:
    m, n, n_entangle, n_fwd, code, inv_code = aux
    tensors = list(leaves)
    out = EntangledQFTBasis.__new__(EntangledQFTBasis)
    out.m = m
    out.n = n
    out.n_entangle = n_entangle
    out.tensors = tensors
    out.code = code
    out.inv_code = inv_code
    return out


tree_util.register_pytree_node(EntangledQFTBasis, _entangled_flatten, _entangled_unflatten)


# ---------------------------------------------------------------------------
# TEBDBasis
# ---------------------------------------------------------------------------


@dataclass
class TEBDBasis:
    """2D TEBD basis with row + column rings of controlled-phase gates.

    Mirror of upstream src/basis.jl:600-745.
    """

    m: int
    n: int
    n_row_gates: int
    n_col_gates: int
    tensors: list[Array]
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        phases: Sequence[float] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
        seed: int | None = None,
    ):
        """When `seed` is provided AND `phases` is None, draws phases from
        `np.random.default_rng(seed).normal(0, 0.1, n_row_gates + n_col_gates)`
        — Julia's `randn(n_gates) * 0.1` convention from
        `ParametricDFT.jl/src/training.jl::_init_circuit`.
        """
        import numpy as np

        from .circuit.tebd import tebd_code

        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        if phases is None and seed is not None:
            phases = list(np.random.default_rng(seed).normal(0.0, 0.1, m + n))
        _code, init_tensors, self.n_row_gates, self.n_col_gates = tebd_code(m, n, phases=phases)
        _inv_code, _, _, _ = tebd_code(m, n, phases=phases, inverse=True)
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
        from ..circuit.builder import apply_circuit

        return apply_circuit(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        from ..circuit.builder import apply_circuit

        return apply_circuit(
            [jnp.conj(t) for t in self.tensors],
            self.inv_code,
            self.m,
            self.n,
            pic,
        )


def _tebd_flatten(b: TEBDBasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, b.n_row_gates, b.n_col_gates, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _tebd_unflatten(aux, leaves) -> TEBDBasis:
    m, n, nr, nc, n_fwd, code, inv_code = aux
    tensors = list(leaves)
    out = TEBDBasis.__new__(TEBDBasis)
    out.m = m
    out.n = n
    out.n_row_gates = nr
    out.n_col_gates = nc
    out.tensors = tensors
    out.code = code
    out.inv_code = inv_code
    return out


tree_util.register_pytree_node(TEBDBasis, _tebd_flatten, _tebd_unflatten)


# ---------------------------------------------------------------------------
# MERABasis
# ---------------------------------------------------------------------------


@dataclass
class MERABasis:
    """2D MERA basis. Each dimension with >=2 qubits must be a power of 2.

    Mirror of upstream src/basis.jl:840-1070.
    """

    m: int
    n: int
    n_row_gates: int
    n_col_gates: int
    tensors: list[Array]
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        phases: Sequence[float] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
        seed: int | None = None,
    ):
        """When `seed` is provided AND `phases` is None, draws phases from
        `np.random.default_rng(seed).normal(0, 0.1, n_gates)` — Julia's
        `randn(n_gates) * 0.1` convention from
        `ParametricDFT.jl/src/training.jl::_init_circuit`.
        """
        import numpy as np

        from .circuit.mera import _n_mera_gates, mera_code

        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        if phases is None and seed is not None:
            n_row = _n_mera_gates(m) if m >= 2 else 0
            n_col = _n_mera_gates(n) if n >= 2 else 0
            phases = list(np.random.default_rng(seed).normal(0.0, 0.1, n_row + n_col))
        _code, init_tensors, self.n_row_gates, self.n_col_gates = mera_code(m, n, phases=phases)
        _inv_code, _, _, _ = mera_code(m, n, phases=phases, inverse=True)
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
        from ..circuit.builder import apply_circuit

        return apply_circuit(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        from ..circuit.builder import apply_circuit

        return apply_circuit(
            [jnp.conj(t) for t in self.tensors],
            self.inv_code,
            self.m,
            self.n,
            pic,
        )


def _mera_flatten(b: MERABasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, b.n_row_gates, b.n_col_gates, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _mera_unflatten(aux, leaves) -> MERABasis:
    m, n, nr, nc, n_fwd, code, inv_code = aux
    tensors = list(leaves)
    out = MERABasis.__new__(MERABasis)
    out.m = m
    out.n = n
    out.n_row_gates = nr
    out.n_col_gates = nc
    out.tensors = tensors
    out.code = code
    out.inv_code = inv_code
    return out


tree_util.register_pytree_node(MERABasis, _mera_flatten, _mera_unflatten)
