"""Riemannian manifold abstraction with batched `(d, d, n)` operations.

Mirror of upstream src/manifolds.jl. Device-agnostic: works on CPU and GPU
via JAX; no `similar`, `copyto!`, or mutable updates — pure functional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

Array = jax.Array

# ---------------------------------------------------------------------------
# Generalized batched linear algebra
# ---------------------------------------------------------------------------


def batched_matmul(A: Array, B: Array) -> Array:
    """`C[:, :, k] = A[:, :, k] @ B[:, :, k]` for each slice `k`.

    Mirror of upstream src/manifolds.jl:43-53.
    """
    return jnp.einsum("ijk,jlk->ilk", A, B)


def batched_adjoint(A: Array) -> Array:
    """`C[:, :, k] = A[:, :, k].conj().T`. Mirror of upstream src/manifolds.jl:60-62."""
    return jnp.conj(jnp.transpose(A, (1, 0, 2)))


def batched_inv(A: Array) -> Array:
    """Batched matrix inverse via a transpose-and-invert trick.

    Reshape `(d, d, n)` → `(n, d, d)`, call jnp.linalg.inv, reshape back.
    Mirror of upstream src/manifolds.jl:70-78 (but truly vectorized).
    """
    A_nd = jnp.transpose(A, (2, 0, 1))
    inv = jnp.linalg.inv(A_nd)
    return jnp.transpose(inv, (1, 2, 0))


def _make_identity_batch(dtype, d: int, n: int) -> Array:
    """(d, d, n) array of identity matrices. Mirror of upstream src/manifolds.jl:87-92."""
    I_mat = jnp.eye(d, dtype=dtype)
    return jnp.broadcast_to(I_mat[:, :, None], (d, d, n))


# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------


@runtime_checkable
class AbstractRiemannianManifold(Protocol):
    def project(self, points: Array, grads: Array) -> Array: ...
    def retract(self, points: Array, tangent: Array, alpha: float, *, I_batch=None) -> Array: ...
    def transport(self, old: Array, new: Array, vec: Array) -> Array: ...


# ---------------------------------------------------------------------------
# Packing / unpacking
# ---------------------------------------------------------------------------


def stack_tensors(tensors: list[Array], indices: list[int]) -> Array:
    """Pack selected matrices into a `(d1, d2, n)` batch.

    Mirror of upstream src/manifolds.jl:126-135.
    """
    if not indices:
        return jnp.zeros((0, 0, 0), dtype=jnp.complex128)
    batch = jnp.stack([tensors[i] for i in indices], axis=-1)
    return batch


def unstack_tensors(batch: Array, indices: list[int], *, into: list) -> None:
    """Unpack `(d1, d2, n)` back into a Python list, in place.

    Mirror of upstream src/manifolds.jl:146-154. `into` is a mutable list
    long enough to be indexed by each entry of `indices`.
    """
    for k, idx in enumerate(indices):
        into[idx] = batch[:, :, k]


# ---------------------------------------------------------------------------
# Unitarity classification
# ---------------------------------------------------------------------------


def is_unitary_general(t: Array, atol: float = 1e-6) -> bool:
    """True if `t @ t.conj().T ≈ I`. False for non-square.

    Mirror of upstream src/manifolds.jl:103-107.
    """
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        return False
    I_mat = jnp.eye(t.shape[0], dtype=t.dtype)
    return bool(jnp.allclose(t @ jnp.conj(t).T, I_mat, atol=atol))


# Forward declarations — dataclasses defined below — so classify_manifold can
# reference them.  In Python this is a lookup-time concern; classify_manifold
# just names them textually and they're resolved when called.


def classify_manifold(t: Array) -> AbstractRiemannianManifold:
    """Return `UnitaryManifold()` if unitary, else `PhaseManifold()`.

    Mirror of upstream src/manifolds.jl:225-231.
    """
    return UnitaryManifold() if is_unitary_general(t) else PhaseManifold()


def group_by_manifold(tensors: list[Array]) -> dict:
    """`{manifold: [indices]}` bucket map.

    Mirror of upstream src/manifolds.jl:113-119. Dedup on manifold *type*
    (so two PhaseManifold() instances collapse to one bucket).
    """
    groups: dict[AbstractRiemannianManifold, list[int]] = {}
    for i, t in enumerate(tensors):
        m = classify_manifold(t)
        existing = next((k for k in groups if type(k) is type(m)), None)
        if existing is None:
            groups[m] = [i]
        else:
            groups[existing].append(i)
    return groups


# ---------------------------------------------------------------------------
# UnitaryManifold — U(n)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitaryManifold:
    """U(n) unitary group manifold; tensors are n x n unitary matrices.

    Mirror of upstream src/manifolds.jl:161-196.
    """

    def project(self, U: Array, G: Array) -> Array:
        """`U * skew(U^H G)` on `(d, d, n)`."""
        UhG = batched_matmul(batched_adjoint(U), G)
        S = (UhG - batched_adjoint(UhG)) / 2
        return batched_matmul(U, S)

    def retract(self, U: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        """Cayley retraction: `(I - a/2 W)^{-1} (I + a/2 W) U`, W = skew(Xi U^H).

        Mirror of upstream src/manifolds.jl:173-193. `I_batch` pre-allocates the
        batched identity; created on demand if None.
        """
        alpha_half = alpha / 2
        d, _, n = U.shape
        W_raw = batched_matmul(Xi, batched_adjoint(U))
        W = (W_raw - batched_adjoint(W_raw)) / 2
        if I_batch is None:
            I_batch = _make_identity_batch(U.dtype, d, n)
        lhs = I_batch - alpha_half * W
        rhs = I_batch + alpha_half * W
        return batched_matmul(batched_matmul(batched_inv(lhs), rhs), U)

    def transport(self, U_old: Array, U_new: Array, v: Array) -> Array:
        """Parallel transport via re-projection. Upstream src/manifolds.jl:196."""
        return self.project(U_new, v)


# ---------------------------------------------------------------------------
# PhaseManifold — U(1)^d
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseManifold:
    """U(1)^d: each element is a unit complex number.

    Mirror of upstream src/manifolds.jl:203-219.
    """

    def project(self, Z: Array, G: Array) -> Array:
        return 1j * jnp.imag(jnp.conj(Z) * G) * Z

    def retract(self, Z: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        y = Z + alpha * Xi
        return y / jnp.abs(y).astype(y.dtype)

    def transport(self, Z_old: Array, Z_new: Array, v: Array) -> Array:
        return self.project(Z_new, v)
