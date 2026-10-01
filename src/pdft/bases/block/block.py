"""BlockedBasis: apply any inner parametric basis independently to each block.

Motivation: the paper's central finding is that **block size dominates the
within-block basis** — BlockDCT 8x8 beats every full-image transform tested.
The full-image learned circuits (QFT/EntangledQFT/TEBD/MERA) are competitive
among full-image bases but lose to BlockDCT 8x8 by ~3 dB.

`BlockedBasis` directly responds to that finding. It wraps an inner basis at
smaller m_inner, n_inner and applies it independently to each
(2^m_inner, 2^n_inner) block of a larger (2^m_outer, 2^n_outer) image.
Parameters are SHARED across blocks (one basis tiled over many blocks).

Concretely: with m_outer = m_inner + block_log_m and n_outer = n_inner +
block_log_n, the image is conceptually decomposed into a grid of
2^block_log_m * 2^block_log_n independent blocks, each transformed by the
inner basis.

Implementation strategy: BlockedBasis exposes ``m, n, tensors, code, inv_code``
just like any other basis, so the existing training pipeline
(``train_basis_batched``, ``loss_function``, ``_build_jit_adam_step``) works
unchanged. The trick is in ``code``/``inv_code``: a ``BlockCode`` permutes the
axes, vmaps the inner code over the block-index dims, and permutes back.

Yao little-endian convention preserved: block-index qubits are the
HIGHER-numbered qubits per dimension (qubits m_inner+1..m_outer for rows),
which correspond to LOW-INDEXED axes [0..block_log_m) in the (2,)^(m+n)
tensor layout, the same convention as `pdft.circuit.builder._axis_of_qubit`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ...circuit.builder import contract_circuit
from ..core import BasisTransforms

Array = jax.Array


@dataclass(frozen=True)
class BlockCode:
    """``code(*tensors, image)`` of a blocked basis: the inner code on every block.

    The image has the outer ``(2,) * (m_outer + n_outer)`` layout (Yao
    little-endian, see ``pdft.circuit.builder._axis_of_qubit``):

        [0..block_log_m)               block-index ROW qubits (msbs)
        [block_log_m..m_outer)         within-block ROW qubits (lsbs)
        [m_outer..m_outer+block_log_n) block-index COL qubits (msbs)
        [m_outer+block_log_n..)        within-block COL qubits (lsbs)

    The block-index axes are moved to the front and the inner code is vmapped
    over them, which works for any inner code, another ``BlockCode`` included.
    A frozen dataclass rather than a closure, so that it compares by value like
    the inner ``CircuitCode`` and two equal blocked bases have equal pytree
    structures.
    """

    inner: Callable[..., Array]
    m_inner: int
    n_inner: int
    block_log_m: int
    block_log_n: int

    def __call__(self, *operands: Any) -> Array:
        *tensors, image = operands
        m_outer = self.m_inner + self.block_log_m
        n_outer = self.n_inner + self.block_log_n
        inner_shape = (2,) * (self.m_inner + self.n_inner)
        outer_shape = (2,) * (m_outer + n_outer)
        if image.shape != outer_shape:
            raise ValueError(f"BlockedBasis expected image shape {outer_shape}, got {image.shape}")
        # Block-index axes (rows then cols) to the front, within-block axes
        # (rows then cols) trailing, which is the layout the inner code expects.
        perm = (
            list(range(self.block_log_m))
            + list(range(m_outer, m_outer + self.block_log_n))
            + list(range(self.block_log_m, m_outer))
            + list(range(m_outer + self.block_log_n, m_outer + n_outer))
        )
        blocks = jnp.transpose(image, perm).reshape((-1,) + inner_shape)
        out = jax.vmap(lambda block: self.inner(*tensors, block))(blocks)
        block_axes = (2,) * (self.block_log_m + self.block_log_n)
        back = [int(axis) for axis in np.argsort(perm)]
        return jnp.transpose(out.reshape(block_axes + inner_shape), back)


# ---------------------------------------------------------------------------
# BlockedBasis
# ---------------------------------------------------------------------------


@dataclass
class BlockedBasis(BasisTransforms):
    """Wraps an inner parametric basis as a within-block transform.

    Parameters
    ----------
    inner : any pdft basis, another BlockedBasis included
        Within-block parametric circuit at smaller m_inner = inner.m,
        n_inner = inner.n.
    block_log_m, block_log_n : int
        Number of block-index qubits per dimension. The image is
        (2^(inner.m + block_log_m), 2^(inner.n + block_log_n)).
        block_log_m=0 (and =0) reduces to the inner basis.

    Notes
    -----
    All learnable parameters live in ``inner.tensors``; ``BlockedBasis`` is
    a pure structural wrapper. Block parameters are SHARED across blocks
    (one inner basis tiled).
    """

    _apply = staticmethod(contract_circuit)

    inner: Any
    block_log_m: int
    block_log_n: int
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    def __init__(
        self,
        inner: Any,
        block_log_m: int,
        block_log_n: int,
        code: object | None = None,
        inv_code: object | None = None,
    ):
        if not hasattr(inner, "m") or not hasattr(inner, "n"):
            raise TypeError(
                f"inner must be a pdft basis with m, n attributes; got {type(inner).__name__}"
            )
        if block_log_m < 0 or block_log_n < 0:
            raise ValueError(
                f"block_log_m and block_log_n must be >= 0; got "
                f"block_log_m={block_log_m}, block_log_n={block_log_n}"
            )
        # A count that cannot size a shape is refused here, not at the first transform.
        (2,) * (block_log_m + block_log_n)
        self.inner = inner
        self.block_log_m = block_log_m
        self.block_log_n = block_log_n
        shape = (inner.m, inner.n, block_log_m, block_log_n)
        self.code = code if code is not None else BlockCode(inner.code, *shape)
        self.inv_code = inv_code if inv_code is not None else BlockCode(inner.inv_code, *shape)

    # m, n and tensors come from the inner basis; BasisTransforms derives the rest.

    @property
    def m(self) -> int:
        return self.inner.m + self.block_log_m

    @property
    def n(self) -> int:
        return self.inner.n + self.block_log_n

    @property
    def tensors(self) -> list[Array]:
        """Forward to inner — pytree leaves order is `inner.tensors`."""
        return self.inner.tensors

    @property
    def num_blocks(self) -> int:
        return 2 ** (self.block_log_m + self.block_log_n)

    @property
    def block_shape(self) -> tuple[int, int]:
        return (2**self.inner.m, 2**self.inner.n)


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------


def _blockedbasis_flatten(b: BlockedBasis):
    """Flatten by recursing into inner via JAX's pytree machinery.

    This delegates inner reconstruction to JAX's tree_unflatten, which
    works for ANY pytree-registered inner — including a nested
    BlockedBasis, whose constructor doesn't take (m, n, tensors, ...).
    """
    inner_leaves, inner_treedef = jax.tree_util.tree_flatten(b.inner)
    aux = (inner_treedef, b.block_log_m, b.block_log_n, b.code, b.inv_code)
    return tuple(inner_leaves), aux


def _blockedbasis_unflatten(aux, leaves) -> BlockedBasis:
    inner_treedef, block_log_m, block_log_n, code, inv_code = aux
    inner = jax.tree_util.tree_unflatten(inner_treedef, list(leaves))
    return BlockedBasis(
        inner=inner,
        block_log_m=block_log_m,
        block_log_n=block_log_n,
        code=code,
        inv_code=inv_code,
    )


tree_util.register_pytree_node(BlockedBasis, _blockedbasis_flatten, _blockedbasis_unflatten)


__all__ = ["BlockedBasis"]
