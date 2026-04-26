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


def _matrix_dim_of(t: Array) -> int | None:
    """Return the matrix dimension of a tensor (rank-2 → t.shape[0]; rank-2k
    → product of first k axes, where 2k matches the storage convention for a
    2-qubit gate stored as (2, 2, 2, 2)). Returns None if not classifiable.

    Used by classify_manifold so that U(2) and U(4) gates bucket separately,
    enabling stack_tensors to operate on homogeneous-shape batches.
    """
    if t.ndim == 2 and t.shape[0] == t.shape[1]:
        return t.shape[0]
    if t.ndim == 4 and t.shape == (2, 2, 2, 2):
        return 4
    return None


def is_unitary_2qubit(t: Array, atol: float = 1e-6) -> bool:
    """True for a (2, 2, 2, 2) tensor whose 4x4 reshape is unitary.

    The (2, 2, 2, 2) layout is the canonical storage of a 2-qubit gate
    (axes: out_ctrl, out_tgt, in_ctrl, in_tgt). We reshape to 4x4 and apply
    the standard unitarity check.
    """
    if t.ndim != 4 or t.shape != (2, 2, 2, 2):
        return False
    M = jnp.reshape(t, (4, 4))
    I_mat = jnp.eye(4, dtype=t.dtype)
    return bool(jnp.allclose(M @ jnp.conj(M).T, I_mat, atol=atol))


def classify_manifold(t: Array) -> AbstractRiemannianManifold:
    """Return the manifold appropriate to ``t`` based on its shape and
    unitarity.

    - rank-2 unitary (d × d) → ``UnitaryManifold(d=d)``
    - (2, 2, 2, 2) tensor whose 4×4 reshape is unitary → ``Unitary2qManifold``
    - otherwise → ``PhaseManifold``

    Note: ``OrthogonalManifold`` and ``Orthogonal2qManifold`` are defined
    in this module for clients that want explicit O(d) constraints, but
    they are NOT auto-selected — selection is the basis class's
    responsibility. Real-valued tensors going through UnitaryManifold
    stay real automatically (Cayley retraction with real W preserves
    real-ness).
    """
    if is_unitary_general(t):
        return UnitaryManifold(d=t.shape[0])
    if is_unitary_2qubit(t):
        return Unitary2qManifold()
    return PhaseManifold()


def group_by_manifold(tensors: list[Array]) -> dict:
    """`{manifold: [indices]}` bucket map.

    Mirror of upstream src/manifolds.jl:113-119, with size-aware bucketing
    (U(2) and U(4) go to separate buckets because UnitaryManifold has a
    `d` field that participates in equality).
    """
    groups: dict[AbstractRiemannianManifold, list[int]] = {}
    for i, t in enumerate(tensors):
        m = classify_manifold(t)
        existing = next((k for k in groups if k == m), None)
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
    """U(d) unitary group manifold; tensors are d × d unitary matrices.

    The ``d`` field defaults to 2 for backward compatibility with all
    pre-RichBasis call sites. Setting ``d`` is what makes U(2) and U(4)
    bucket into separate groups in ``group_by_manifold`` (without it,
    ``stack_tensors`` would try to stack mismatched shapes).

    Mirror of upstream src/manifolds.jl:161-196.
    """

    d: int = 2

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
# Orthogonal manifolds — real subgroups of U(d), used for Approach A
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrthogonalManifold:
    """O(d) for real-valued d×d unitaries (subset of UnitaryManifold).

    Implementation: project gradients to the REAL skew-symmetric tangent
    subspace; Cayley retraction with a real W keeps the tensor real
    throughout training. Hadamards initialised at canonical form
    [[1, 1], [1, -1]] / sqrt(2) live in O(2) (det = -1) — the connected
    component is preserved by retraction, so we stay on the same coset.
    """

    d: int = 2

    def project(self, U: Array, G: Array) -> Array:
        # Project to skew-symmetric (real) tangent direction.
        UhG = batched_matmul(batched_adjoint(U), G)
        S = (UhG - batched_adjoint(UhG)) / 2
        S_real = jnp.real(S).astype(U.dtype)
        return batched_matmul(U, S_real)

    def retract(self, U: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        # Reuse Unitary's Cayley retraction; output stays real if inputs are real.
        return UnitaryManifold(d=self.d).retract(U, Xi, alpha, I_batch=I_batch)

    def transport(self, U_old: Array, U_new: Array, v: Array) -> Array:
        return self.project(U_new, v)


# ---------------------------------------------------------------------------
# Unitary2qManifold — U(4) for 2-qubit gates stored as (2, 2, 2, 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unitary2qManifold:
    """U(4) manifold for 2-qubit gates stored in (2, 2, 2, 2) tensor form.

    Storage convention: axes (out_ctrl, out_tgt, in_ctrl, in_tgt). This is
    the canonical form for a 2-qubit gate as it appears in the einsum
    builder (rank-4 to match 4 qubit-axis subscripts). All Riemannian
    operations reshape to (4, 4, n) internally and reuse the U(d=4) math.
    """

    def _to_mat(self, T: Array) -> Array:
        """(2, 2, 2, 2, n) -> (4, 4, n)."""
        n = T.shape[-1]
        return T.reshape(4, 4, n)

    def _from_mat(self, M: Array) -> Array:
        """(4, 4, n) -> (2, 2, 2, 2, n)."""
        n = M.shape[-1]
        return M.reshape(2, 2, 2, 2, n)

    def project(self, T: Array, G: Array) -> Array:
        return self._from_mat(UnitaryManifold(d=4).project(self._to_mat(T), self._to_mat(G)))

    def retract(self, T: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        # Caller's pre-allocated I_batch (if any) was sized for the storage
        # shape; UnitaryManifold(d=4) builds its own (4,4,n) identity, so
        # we just discard the passed-in I_batch here.
        out_mat = UnitaryManifold(d=4).retract(
            self._to_mat(T), self._to_mat(Xi), alpha, I_batch=None
        )
        return self._from_mat(out_mat)

    def transport(self, T_old: Array, T_new: Array, v: Array) -> Array:
        return self.project(T_new, v)


@dataclass(frozen=True)
class Orthogonal2qManifold:
    """O(4) for real-valued 2-qubit gates stored as (2, 2, 2, 2).

    Same reshape-and-delegate strategy as Unitary2qManifold, but projects
    gradients to the REAL skew-symmetric tangent subspace so the tensor
    stays in O(4) under Cayley retraction.
    """

    def _to_mat(self, T: Array) -> Array:
        n = T.shape[-1]
        return T.reshape(4, 4, n)

    def _from_mat(self, M: Array) -> Array:
        n = M.shape[-1]
        return M.reshape(2, 2, 2, 2, n)

    def project(self, T: Array, G: Array) -> Array:
        return self._from_mat(
            OrthogonalManifold(d=4).project(self._to_mat(T), self._to_mat(G))
        )

    def retract(self, T: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        out_mat = OrthogonalManifold(d=4).retract(
            self._to_mat(T), self._to_mat(Xi), alpha, I_batch=None
        )
        return self._from_mat(out_mat)

    def transport(self, T_old: Array, T_new: Array, v: Array) -> Array:
        return self.project(T_new, v)


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
