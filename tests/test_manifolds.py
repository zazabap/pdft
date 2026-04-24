import jax
import jax.numpy as jnp

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
