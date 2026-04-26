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
unchanged. The trick is in ``code``/``inv_code``: they are closures that
permute axes, vmap the inner code over the block-index dims, and permute back.

Yao little-endian convention preserved: block-index qubits are the
HIGHER-numbered qubits per dimension (qubits m_inner+1..m_outer for rows),
which correspond to LOW-INDEXED axes [0..block_log_m) in the (2,)^(m+n)
tensor layout. See CLAUDE.md §2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

Array = jax.Array


# ---------------------------------------------------------------------------
# Block-aware einsum closure
# ---------------------------------------------------------------------------


def _make_block_code(
    inner_code: Callable[..., Array],
    *,
    m_inner: int,
    n_inner: int,
    block_log_m: int,
    block_log_n: int,
) -> Callable[..., Array]:
    """Return a callable with the same signature as ``inner_code`` that operates
    on a (2,)^(m_outer+n_outer) image by vmap-ing ``inner_code`` over the
    block-index axes.

    Outer axis layout (Yao little-endian, see _circuit.build_circuit_einsum):
        [0..block_log_m)              -- block-index ROW qubits (msbs)
        [block_log_m..m_outer)        -- within-block ROW qubits (lsbs, m_inner of them)
        [m_outer..m_outer+block_log_n) -- block-index COL qubits (msbs)
        [m_outer+block_log_n..)        -- within-block COL qubits (lsbs, n_inner of them)
    """
    m_outer = m_inner + block_log_m
    n_outer = n_inner + block_log_n
    n_blocks = 2 ** (block_log_m + block_log_n)
    inner_shape = (2,) * (m_inner + n_inner)
    outer_shape = (2,) * (m_outer + n_outer)

    # Permutation: block-index axes (rows then cols) to the front, within-block
    # axes (rows then cols) trailing. The trailing layout matches the inner
    # einsum's expected (2,)^(m_inner+n_inner) shape.
    perm = (
        list(range(0, block_log_m))  # block row axes
        + list(range(m_outer, m_outer + block_log_n))  # block col axes
        + list(range(block_log_m, m_outer))  # within row axes
        + list(range(m_outer + block_log_n, m_outer + n_outer))  # within col axes
    )
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i

    leading_block_shape = (2,) * (block_log_m + block_log_n)

    def block_code(*args: Any) -> Array:
        *tensors, image = args
        if image.shape != outer_shape:
            raise ValueError(f"BlockedBasis expected image shape {outer_shape}, got {image.shape}")
        x = jnp.transpose(image, perm)
        x_flat = x.reshape((n_blocks,) + inner_shape)

        def apply_one(img_one: Array) -> Array:
            return inner_code(*tensors, img_one)

        out_flat = jax.vmap(apply_one)(x_flat)
        out = out_flat.reshape(leading_block_shape + inner_shape)
        return jnp.transpose(out, inv_perm)

    return block_code


# ---------------------------------------------------------------------------
# BlockedBasis
# ---------------------------------------------------------------------------


@dataclass
class BlockedBasis:
    """Wraps an inner parametric basis as a within-block transform.

    Parameters
    ----------
    inner : any pdft basis (QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis)
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
        self.inner = inner
        self.block_log_m = block_log_m
        self.block_log_n = block_log_n
        if code is None or inv_code is None:
            built_code = _make_block_code(
                inner.code,
                m_inner=inner.m,
                n_inner=inner.n,
                block_log_m=block_log_m,
                block_log_n=block_log_n,
            )
            built_inv = _make_block_code(
                inner.inv_code,
                m_inner=inner.m,
                n_inner=inner.n,
                block_log_m=block_log_m,
                block_log_n=block_log_n,
            )
            self.code = code if code is not None else built_code
            self.inv_code = inv_code if inv_code is not None else built_inv
        else:
            self.code = code
            self.inv_code = inv_code

    # ---- AbstractSparseBasis interface (matches QFTBasis) ----

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
    def inv_tensors(self) -> list[Array]:
        return self.inner.tensors

    @property
    def image_size(self) -> tuple[int, int]:
        return (2**self.m, 2**self.n)

    @property
    def num_parameters(self) -> int:
        return self.inner.num_parameters

    @property
    def num_blocks(self) -> int:
        return 2 ** (self.block_log_m + self.block_log_n)

    @property
    def block_shape(self) -> tuple[int, int]:
        return (2**self.inner.m, 2**self.inner.n)

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


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------


def _blockedbasis_flatten(b: BlockedBasis):
    leaves = tuple(b.inner.tensors)
    aux = (
        type(b.inner),
        b.inner.m,
        b.inner.n,
        b.inner.code,
        b.inner.inv_code,
        len(leaves),
        b.block_log_m,
        b.block_log_n,
        b.code,
        b.inv_code,
    )
    return leaves, aux


def _blockedbasis_unflatten(aux, leaves) -> BlockedBasis:
    (
        inner_class,
        inner_m,
        inner_n,
        inner_code,
        inner_inv_code,
        n_fwd,
        block_log_m,
        block_log_n,
        code,
        inv_code,
    ) = aux
    assert len(leaves) == n_fwd
    inner = inner_class(
        m=inner_m,
        n=inner_n,
        tensors=list(leaves),
        code=inner_code,
        inv_code=inner_inv_code,
    )
    return BlockedBasis(
        inner=inner,
        block_log_m=block_log_m,
        block_log_n=block_log_n,
        code=code,
        inv_code=inv_code,
    )


tree_util.register_pytree_node(BlockedBasis, _blockedbasis_flatten, _blockedbasis_unflatten)


__all__ = ["BlockedBasis"]
