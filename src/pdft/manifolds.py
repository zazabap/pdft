"""Riemannian manifold abstraction with batched `(d, d, n)` operations.

Mirror of upstream src/manifolds.jl. Device-agnostic: works on CPU and GPU
via JAX; no `similar`, `copyto!`, or mutable updates — pure functional.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

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


def _skew(A: Array) -> Array:
    """The skew-Hermitian part ``(A - A^H) / 2`` of each matrix in a ``(d, d, n)`` batch."""
    return (A - batched_adjoint(A)) / 2


def _make_identity_batch(dtype, d: int, n: int) -> Array:
    """(d, d, n) array of identity matrices. Mirror of upstream src/manifolds.jl:87-92."""
    I_mat = jnp.eye(d, dtype=dtype)
    return jnp.broadcast_to(I_mat[:, :, None], (d, d, n))


# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------


@runtime_checkable
class AbstractRiemannianManifold(Protocol):
    """Interface the optimizers need from a manifold: ``project`` a Euclidean
    gradient onto the tangent space, ``retract`` from ``points`` along ``tangent``
    by step ``alpha``, and ``transport`` a tangent vector from ``old`` to ``new``.
    """

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
    """Unpack a `(*tensor_shape, n)` batch back into a Python list, in place.

    Mirror of upstream src/manifolds.jl:146-154. `into` is a mutable list
    long enough to be indexed by each entry of `indices`.

    The slice is taken on the LAST axis (`batch[..., k]`), matching
    ``stack_tensors``' ``axis=-1`` stacking, so this works for any tensor
    rank — 2×2 matrices stacked to ``(2, 2, n)`` AND 2-qubit gates
    ``(2, 2, 2, 2)`` stacked to ``(2, 2, 2, 2, n)``. A hardcoded ``[:, :, k]``
    here would slice a qubit axis of the 2-qubit gates instead of the stack
    axis (and JAX silently clamps out-of-range k), corrupting them.
    """
    for k, idx in enumerate(indices):
        into[idx] = batch[..., k]


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


def is_unitary_2qubit(t: Array, atol: float = 1e-6) -> bool:
    """True for a (2, 2, 2, 2) tensor whose 4x4 reshape is unitary.

    The (2, 2, 2, 2) layout is the canonical storage of a 2-qubit gate
    (axes: out_ctrl, out_tgt, in_ctrl, in_tgt). We reshape to 4x4 and apply
    the standard unitarity check.
    """
    return t.shape == (2, 2, 2, 2) and is_unitary_general(jnp.reshape(t, (4, 4)), atol)


# Forward declarations — the manifold dataclasses are defined below, so
# classify_manifold can reference them. In Python this is a lookup-time
# concern; classify_manifold just names them textually and they're resolved
# when called.
def classify_manifold(t: Array) -> AbstractRiemannianManifold:
    """Return the manifold appropriate to ``t`` based on its shape and
    unitarity.

    - rank-2 unitary (d × d) → ``UnitaryManifold(d=d)``
    - (2, 2, 2, 2) tensor whose 4×4 reshape is unitary → ``Unitary2qManifold``
    - otherwise → ``PhaseManifold``

    Note: ``OrthogonalManifold`` and ``Orthogonal2qManifold`` are defined
    in this module for callers that want an explicit O(d) constraint, but
    nothing selects them: not this function, and no basis or trainer in the
    package. A basis whose tensors are all exactly real stays real through
    ``UnitaryManifold`` under a real objective (the Cayley retraction with a
    real ``W`` preserves it), which is how ``RealRichBasis`` trains. One
    complex tensor in the circuit is enough to make the others complex.
    """
    if is_unitary_general(t):
        return UnitaryManifold(d=t.shape[0])
    if is_unitary_2qubit(t):
        return Unitary2qManifold()
    return PhaseManifold()


def group_by_manifold(
    tensors: list[Array], manifolds: Sequence[AbstractRiemannianManifold] | None = None
) -> dict:
    """`{manifold: [indices]}` bucket map.

    Mirror of upstream src/manifolds.jl:113-119, with size-aware bucketing
    (U(2) and U(4) go to separate buckets because UnitaryManifold has a
    `d` field that participates in equality).

    Each tensor's manifold is read off its values by ``classify_manifold``,
    as upstream does, unless ``manifolds`` names it: one manifold per tensor,
    for parameters whose geometry their values cannot tell (a real array of
    angles is flat, and would be classified as a phase tensor).
    """
    if manifolds is not None and len(manifolds) != len(tensors):
        raise ValueError(f"{len(tensors)} tensors but {len(manifolds)} manifolds")
    groups: dict[AbstractRiemannianManifold, list[int]] = {}
    for i, t in enumerate(tensors):
        manifold = classify_manifold(t) if manifolds is None else manifolds[i]
        groups.setdefault(manifold, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# UnitaryManifold — U(n)
# ---------------------------------------------------------------------------


class _ReprojectTransport:
    """Vector transport by projecting onto the tangent space at the new point.

    What every manifold here uses (upstream src/manifolds.jl:196). The two
    points keep the parameter names each manifold has always had (``U_``
    here, ``T_`` for the two-qubit storage, ``Z_`` for phases), so the
    two-qubit and phase manifolds spell the same line out under theirs.
    """

    def transport(self, U_old: Array, U_new: Array, v: Array) -> Array:
        return self.project(U_new, v)


@dataclass(frozen=True)
class UnitaryManifold(_ReprojectTransport):
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
        return batched_matmul(U, _skew(batched_matmul(batched_adjoint(U), G)))

    def retract(self, U: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        """Cayley retraction: `(I - a/2 W)^{-1} (I + a/2 W) U`, W = skew(Xi U^H).

        Mirror of upstream src/manifolds.jl:173-193. `I_batch` pre-allocates the
        batched identity; created on demand if None.
        """
        alpha_half = alpha / 2
        d, _, n = U.shape
        W = _skew(batched_matmul(Xi, batched_adjoint(U)))
        if I_batch is None:
            I_batch = _make_identity_batch(U.dtype, d, n)
        lhs = I_batch - alpha_half * W
        rhs = I_batch + alpha_half * W
        return batched_matmul(batched_matmul(batched_inv(lhs), rhs), U)


# ---------------------------------------------------------------------------
# Orthogonal manifolds — real subgroups of U(d). Not selected by the package.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrthogonalManifold(_ReprojectTransport):
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
        S = _skew(batched_matmul(batched_adjoint(U), G))
        return batched_matmul(U, jnp.real(S).astype(U.dtype))

    def retract(self, U: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        # Reuse Unitary's Cayley retraction; output stays real if inputs are real.
        return UnitaryManifold(d=self.d).retract(U, Xi, alpha, I_batch=I_batch)


# ---------------------------------------------------------------------------
# U(4) and O(4) for 2-qubit gates stored as (2, 2, 2, 2)
# ---------------------------------------------------------------------------


class _TwoQubitStorage:
    """A manifold of 4x4 matrices, for 2-qubit gates stored as ``(2, 2, 2, 2)``.

    Storage convention: axes (out_ctrl, out_tgt, in_ctrl, in_tgt), the
    canonical form for a 2-qubit gate as the circuit applier contracts it
    (rank-4, one axis per qubit leg). Every operation reshapes to
    ``(4, 4, n)``, runs on ``matrix``, the manifold of the 4x4 matrices, and
    reshapes back.
    """

    matrix: ClassVar[AbstractRiemannianManifold]

    @staticmethod
    def _to_mat(T: Array) -> Array:
        """(2, 2, 2, 2, n) -> (4, 4, n)."""
        return T.reshape(4, 4, T.shape[-1])

    @staticmethod
    def _from_mat(M: Array) -> Array:
        """(4, 4, n) -> (2, 2, 2, 2, n)."""
        return M.reshape(2, 2, 2, 2, M.shape[-1])

    def project(self, T: Array, G: Array) -> Array:
        return self._from_mat(self.matrix.project(self._to_mat(T), self._to_mat(G)))

    def retract(self, T: Array, Xi: Array, alpha: float, *, I_batch=None) -> Array:
        # A caller's pre-allocated I_batch was sized for the storage shape;
        # the matrix manifold builds its own (4, 4, n) identity.
        out_mat = self.matrix.retract(self._to_mat(T), self._to_mat(Xi), alpha, I_batch=None)
        return self._from_mat(out_mat)

    def transport(self, T_old: Array, T_new: Array, v: Array) -> Array:
        return self.project(T_new, v)


@dataclass(frozen=True)
class Unitary2qManifold(_TwoQubitStorage):
    """U(4) manifold for 2-qubit gates stored in (2, 2, 2, 2) tensor form."""

    matrix = UnitaryManifold(d=4)


@dataclass(frozen=True)
class Orthogonal2qManifold(_TwoQubitStorage):
    """O(4) for real-valued 2-qubit gates stored as (2, 2, 2, 2).

    Projects gradients to the REAL skew-symmetric tangent subspace so the
    tensor stays in O(4) under Cayley retraction.
    """

    matrix = OrthogonalManifold(d=4)


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


# ---------------------------------------------------------------------------
# EuclideanManifold — flat space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EuclideanManifold:
    """Flat space: arrays of ``shape`` whose entries are free numbers.

    Not in upstream. The tangent space at every point is the space itself, so
    there is nothing to project onto, nothing to retract to and nothing to
    transport: a step is ``points + alpha * tangent``. On real parameters
    ``RiemannianAdam`` is then Adam as Kingma and Ba state it. (A complex array
    is not treated as two real ones: its real and imaginary parts share one
    second moment.) It is the geometry of parameters that are not gate tensors,
    such as the angles a ``ParameterView`` reads off the controlled-phase gates.

    ``classify_manifold`` never returns it, since values cannot tell a free
    array from a phase tensor: name it through ``group_by_manifold``'s
    ``manifolds``. ``shape`` takes part in equality, as ``UnitaryManifold``'s
    ``d`` does, so arrays of different shapes fall into separate groups (a
    group is stacked into one array).
    """

    shape: tuple[int, ...]

    def project(self, points: Array, grads: Array) -> Array:
        return grads

    def retract(self, points: Array, tangent: Array, alpha: float, *, I_batch=None) -> Array:
        return points + alpha * tangent

    def transport(self, old: Array, new: Array, vec: Array) -> Array:
        return vec
