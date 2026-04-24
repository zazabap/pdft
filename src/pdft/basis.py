"""Sparse basis abstraction + QFTBasis implementation.

Mirror of upstream src/basis.jl:1-250 (QFTBasis only; other bases Phase 3).
QFTBasis is a registered JAX pytree so jax.grad / jax.jit traverse it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import tree_util

from .qft import qft_code

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
    """QFT tensor-network basis. See spec Section 4.

    `code` and `inv_code` are jit-compiled einsum closures and compare by
    identity; they are marked `compare=False` so the dataclass-generated
    `__eq__` doesn't consider them. Use `bases_allclose` for semantic
    comparison.

    Pytree contract:
        leaves   = tensors + inv_tensors     (flattened in this order)
        aux data = (m, n, len(tensors), len(inv_tensors), code, inv_code)
    """
    m: int
    n: int
    tensors: list[Array]
    inv_tensors: list[Array]
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        inv_tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
    ):
        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        _code, init_tensors = qft_code(m, n)
        _inv_code, init_inv_tensors = qft_code(m, n, inverse=True)
        self.tensors = list(tensors) if tensors is not None else init_tensors
        self.inv_tensors = list(inv_tensors) if inv_tensors is not None else init_inv_tensors
        self.code = code if code is not None else _code
        self.inv_code = inv_code if inv_code is not None else _inv_code

    # ------ AbstractSparseBasis interface ------

    @property
    def image_size(self) -> tuple[int, int]:
        return (2 ** self.m, 2 ** self.n)

    @property
    def num_parameters(self) -> int:
        return sum(int(t.size) for t in self.tensors)

    def forward_transform(self, pic: Array) -> Array:
        from .qft import ft_mat

        return ft_mat(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        from .qft import ift_mat

        return ift_mat(
            [jnp.conj(t) for t in self.inv_tensors],
            self.inv_code,
            self.m,
            self.n,
            pic,
        )


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------


def _qftbasis_flatten(b: QFTBasis):
    leaves = tuple(b.tensors) + tuple(b.inv_tensors)
    aux = (b.m, b.n, len(b.tensors), len(b.inv_tensors), b.code, b.inv_code)
    return leaves, aux


def _qftbasis_unflatten(aux, leaves) -> QFTBasis:
    m, n, n_fwd, n_inv, code, inv_code = aux
    assert len(leaves) == n_fwd + n_inv
    tensors = list(leaves[:n_fwd])
    inv_tensors = list(leaves[n_fwd : n_fwd + n_inv])
    return QFTBasis(
        m=m,
        n=n,
        tensors=tensors,
        inv_tensors=inv_tensors,
        code=code,
        inv_code=inv_code,
    )


tree_util.register_pytree_node(QFTBasis, _qftbasis_flatten, _qftbasis_unflatten)


# ---------------------------------------------------------------------------
# Equality helper
# ---------------------------------------------------------------------------


def bases_allclose(a: QFTBasis, b: QFTBasis, *, atol: float = 1e-10) -> bool:
    """Semantic equality: same (m, n) and tensor values within tolerance.

    See spec Section 8.1 — `code` / `inv_code` are deliberately ignored.
    """
    if (a.m, a.n) != (b.m, b.n):
        return False
    if len(a.tensors) != len(b.tensors) or len(a.inv_tensors) != len(b.inv_tensors):
        return False
    for x, y in zip(a.tensors, b.tensors):
        if not jnp.allclose(x, y, atol=atol, rtol=0.0):
            return False
    for x, y in zip(a.inv_tensors, b.inv_tensors):
        if not jnp.allclose(x, y, atol=atol, rtol=0.0):
            return False
    return True
