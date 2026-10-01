import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.manifolds import (
    PhaseManifold,
    UnitaryManifold,
    _make_identity_batch,
    batched_adjoint,
    batched_inv,
    batched_matmul,
    classify_manifold,
    group_by_manifold,
    is_unitary_general,
    stack_tensors,
    unstack_tensors,
)

from .helpers import random_unitary


def test_batched_matmul_shape():
    A = jnp.ones((3, 4, 5), dtype=jnp.complex128)
    B = jnp.ones((4, 2, 5), dtype=jnp.complex128)
    C = batched_matmul(A, B)
    assert C.shape == (3, 2, 5)


def test_batched_matmul_matches_loop():
    A = jax.random.normal(jax.random.PRNGKey(1), (3, 3, 4)).astype(jnp.complex128)
    B = jax.random.normal(jax.random.PRNGKey(2), (3, 3, 4)).astype(jnp.complex128)
    C = batched_matmul(A, B)
    for k in range(4):
        assert jnp.allclose(C[:, :, k], A[:, :, k] @ B[:, :, k], atol=1e-12)


def test_batched_adjoint_conjugate_transposes():
    A = jnp.arange(12).reshape(3, 4, 1).astype(jnp.complex128) + 1j * jnp.ones((3, 4, 1))
    H = batched_adjoint(A)
    assert H.shape == (4, 3, 1)
    assert jnp.allclose(H[:, :, 0], jnp.conj(A[:, :, 0]).T)


def test_batched_inv_roundtrip():
    A = jax.random.normal(jax.random.PRNGKey(3), (3, 3, 4)).astype(jnp.complex128)
    A = A + 1j * jax.random.normal(jax.random.PRNGKey(4), (3, 3, 4))
    Ainv = batched_inv(A)
    I3 = jnp.eye(3, dtype=jnp.complex128)
    for k in range(4):
        assert jnp.allclose(A[:, :, k] @ Ainv[:, :, k], I3, atol=1e-8)


def test_identity_batch_is_identity_on_each_slice():
    I_b = _make_identity_batch(jnp.complex128, d=3, n=5)
    assert I_b.shape == (3, 3, 5)
    I3 = jnp.eye(3, dtype=jnp.complex128)
    for k in range(5):
        assert jnp.allclose(I_b[:, :, k], I3)


def test_stack_and_unstack_roundtrip():
    tensors = [jnp.ones((2, 2)) * k for k in range(4)]
    batch = stack_tensors(tensors, [0, 2])
    assert batch.shape == (2, 2, 2)
    assert jnp.allclose(batch[:, :, 0], tensors[0])
    assert jnp.allclose(batch[:, :, 1], tensors[2])

    target = [None, None, None, None]
    unstack_tensors(batch, [1, 3], into=target)
    assert jnp.allclose(target[1], tensors[0])
    assert jnp.allclose(target[3], tensors[2])


def test_stack_tensors_empty_returns_empty_batch():
    """Empty index list → a (0, 0, 0) batch (upstream convention)."""
    batch = stack_tensors([], [])
    assert batch.shape == (0, 0, 0)


def test_stack_and_unstack_roundtrip_2qubit():
    """Rank-4 (2,2,2,2) gates (Unitary2qManifold / U4) must round-trip through
    stack/unstack. Regression: unstack_tensors used a hardcoded 3-axis slice
    `batch[:, :, k]`, mangling 2-qubit tensors to (2,2,2,n) and silently
    clamping out-of-range k — which crashed the GD path for Rich/RealRich."""
    tensors = [jnp.full((2, 2, 2, 2), float(k), dtype=jnp.complex128) for k in range(3)]
    batch = stack_tensors(tensors, [0, 1, 2])
    assert batch.shape == (2, 2, 2, 2, 3)

    target = [None, None, None]
    unstack_tensors(batch, [0, 1, 2], into=target)
    for k in range(3):
        assert target[k].shape == (2, 2, 2, 2)
        assert jnp.allclose(target[k], tensors[k])


def test_is_unitary_general_detects_unitary():
    H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
    assert is_unitary_general(H)
    diag_phase = jnp.diag(jnp.array([1.0 + 0j, jnp.exp(1j * 0.5)]))
    assert is_unitary_general(diag_phase)


def test_is_unitary_general_rejects_non_unitary():
    assert not is_unitary_general(jnp.array([[1.0 + 0j, 2.0], [3.0, 4.0]]))


def test_classify_manifold_dispatches_on_unitarity():
    H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
    assert isinstance(classify_manifold(H), UnitaryManifold)
    nonunit = jnp.array([[1.0 + 0j, 2.0], [3.0, 4.0]])
    assert isinstance(classify_manifold(nonunit), PhaseManifold)


def test_group_by_manifold_buckets_indices():
    H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
    nonunit = jnp.array([[1.0 + 0j, 2.0], [3.0, 4.0]])
    groups = group_by_manifold([H, nonunit, H])
    um = next(k for k in groups if isinstance(k, UnitaryManifold))
    pm = next(k for k in groups if isinstance(k, PhaseManifold))
    assert groups[um] == [0, 2]
    assert groups[pm] == [1]


def test_unitary_manifold_retract_preserves_unitarity():
    d, n = 4, 3
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    A = jax.random.normal(k1, (n, d, d)) + 1j * jax.random.normal(k2, (n, d, d))
    U_nd, _ = jnp.linalg.qr(A)
    U = jnp.transpose(U_nd, (1, 2, 0)).astype(jnp.complex128)
    G = jax.random.normal(k3, (d, d, n)).astype(jnp.complex128) + 1j * jax.random.normal(
        jax.random.PRNGKey(99), (d, d, n)
    )
    M = UnitaryManifold()
    Xi = M.project(U, G)
    for alpha in (1e-4, 1e-2, 1.0):
        U_new = M.retract(U, Xi, alpha)
        I_b = _make_identity_batch(jnp.complex128, d, n)
        UUh = batched_matmul(U_new, batched_adjoint(U_new))
        assert jnp.allclose(UUh, I_b, atol=1e-8)


def test_phase_manifold_retract_preserves_unit_modulus():
    d, n = 5, 2
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key, 2)
    theta = jax.random.uniform(k1, (d, 1, n), minval=0.0, maxval=2 * jnp.pi)
    Z = jnp.exp(1j * theta).astype(jnp.complex128)
    Xi = jax.random.normal(k2, (d, 1, n)).astype(jnp.complex128) * 1j
    M = PhaseManifold()
    Xi_tan = M.project(Z, Xi)
    for alpha in (1e-4, 1e-2, 1.0):
        Z_new = M.retract(Z, Xi_tan, alpha)
        assert jnp.allclose(jnp.abs(Z_new), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Orthogonal manifolds, the 2-qubit storage, transport
# ---------------------------------------------------------------------------


def _unitaries(d, count, seed, real=False):
    """A ``(d, d, count)`` batch of points, the layout the manifolds work on."""
    rng = np.random.default_rng(seed)
    points = [random_unitary(rng, d, real=real) for _ in range(count)]
    return jnp.asarray(np.stack(points, axis=-1), dtype=jnp.complex128)


def _tangent(shape, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))


@pytest.mark.parametrize("d", [2, 4])
def test_orthogonal_manifold_stays_real_and_orthogonal(d):
    from pdft.manifolds import OrthogonalManifold

    manifold = OrthogonalManifold(d=d)
    points = _unitaries(d, 3, seed=d, real=True)
    direction = manifold.project(points, _tangent(points.shape, seed=1))
    assert float(jnp.max(jnp.abs(jnp.imag(direction)))) == 0.0
    moved = manifold.retract(points, direction, 0.3)
    assert float(jnp.max(jnp.abs(jnp.imag(moved)))) == 0.0
    for k in range(3):
        q = moved[:, :, k]
        assert jnp.allclose(q @ q.T, jnp.eye(d), atol=1e-12)
    assert not jnp.allclose(moved, points)


@pytest.mark.parametrize("name", ["Unitary2qManifold", "Orthogonal2qManifold"])
def test_two_qubit_manifolds_are_their_matrix_manifold_through_a_reshape(name):
    import pdft.manifolds as manifolds

    manifold = getattr(manifolds, name)()
    matrix = manifold.matrix
    assert (
        matrix
        == {
            "Unitary2qManifold": manifolds.UnitaryManifold(d=4),
            "Orthogonal2qManifold": manifolds.OrthogonalManifold(d=4),
        }[name]
    )
    mats = _unitaries(4, 3, seed=5, real=name.startswith("Orthogonal"))
    grads = _tangent(mats.shape, seed=6)
    stored, stored_grads = mats.reshape(2, 2, 2, 2, 3), grads.reshape(2, 2, 2, 2, 3)

    projected = manifold.project(stored, stored_grads)
    assert projected.shape == (2, 2, 2, 2, 3)
    assert jnp.array_equal(projected.reshape(4, 4, 3), matrix.project(mats, grads))
    # a caller's identity batch, sized for the storage shape, is not used
    moved = manifold.retract(stored, projected, 0.2, I_batch="not an array")
    assert jnp.array_equal(
        moved.reshape(4, 4, 3), matrix.retract(mats, matrix.project(mats, grads), 0.2)
    )
    # a manifold is a value: the optimiser groups tensors by it
    assert manifold == getattr(manifolds, name)() and hash(manifold) == hash(
        getattr(manifolds, name)()
    )
    assert manifold != manifolds.PhaseManifold()


def test_transport_is_projection_at_the_new_point_on_every_manifold():
    import pdft.manifolds as manifolds

    cases = [
        (manifolds.UnitaryManifold(d=2), _unitaries(2, 2, seed=1)),
        (manifolds.OrthogonalManifold(d=2), _unitaries(2, 2, seed=2, real=True)),
        (manifolds.Unitary2qManifold(), _unitaries(4, 2, seed=3).reshape(2, 2, 2, 2, 2)),
        (
            manifolds.Orthogonal2qManifold(),
            _unitaries(4, 2, seed=4, real=True).reshape(2, 2, 2, 2, 2),
        ),
        (manifolds.PhaseManifold(), jnp.exp(1j * jnp.real(_tangent((2, 2, 2), seed=5)))),
    ]
    for manifold, points in cases:
        vector = _tangent(points.shape, seed=9)
        new = manifold.retract(points, manifold.project(points, vector), 0.1)
        assert jnp.array_equal(
            manifold.transport(points, new, vector), manifold.project(new, vector)
        )
