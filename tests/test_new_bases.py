"""Smoke + property tests for EntangledQFTBasis, TEBDBasis, MERABasis."""
import jax
import jax.numpy as jnp
import pytest

from pdft.basis import EntangledQFTBasis, MERABasis, QFTBasis, TEBDBasis, bases_allclose


def _random_pic(m: int, n: int, key_seed: int = 0):
    return jax.random.normal(
        jax.random.PRNGKey(key_seed), (2 ** m, 2 ** n)
    ).astype(jnp.complex128)


# ---------- EntangledQFTBasis ----------


def test_entangled_qft_basis_init_and_shape():
    b = EntangledQFTBasis(m=2, n=2)
    assert b.m == 2 and b.n == 2 and b.n_entangle == 2
    # 6 QFT gates + 2 entangle gates = 8 tensors
    assert len(b.tensors) == 8
    assert b.image_size == (4, 4)


def test_entangled_qft_basis_zero_phases_matches_standard_qft():
    # When all entangle_phases are zero, the entanglement gates are identity
    # → the circuit reduces to standard 2D QFT.
    eb = EntangledQFTBasis(m=2, n=2, entangle_phases=[0.0, 0.0])
    qb = QFTBasis(m=2, n=2)
    pic = _random_pic(2, 2, key_seed=1)
    eqft = eb.forward_transform(pic)
    qft = qb.forward_transform(pic)
    assert jnp.allclose(eqft, qft, atol=1e-10)


def test_entangled_qft_basis_roundtrip():
    b = EntangledQFTBasis(m=2, n=2, entangle_phases=[0.3, 0.5])
    pic = _random_pic(2, 2, key_seed=2)
    recov = b.inverse_transform(b.forward_transform(pic))
    assert jnp.allclose(recov, pic, atol=1e-10)


def test_entangled_qft_is_pytree():
    b = EntangledQFTBasis(m=2, n=2)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert bases_allclose(restored, b)


# ---------- TEBDBasis ----------


def test_tebd_basis_init_and_shape():
    b = TEBDBasis(m=2, n=2)
    assert b.m == 2 and b.n == 2
    assert b.n_row_gates == 2 and b.n_col_gates == 2
    # 4 Hadamards + 4 phase gates = 8 tensors
    assert len(b.tensors) == 8


def test_tebd_basis_roundtrip():
    phases = [0.1, 0.2, 0.3, 0.4]
    b = TEBDBasis(m=2, n=2, phases=phases)
    pic = _random_pic(2, 2, key_seed=3)
    recov = b.inverse_transform(b.forward_transform(pic))
    assert jnp.allclose(recov, pic, atol=1e-10)


def test_tebd_basis_is_pytree():
    b = TEBDBasis(m=2, n=2)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert bases_allclose(restored, b)


def test_tebd_basis_rejects_wrong_phase_length():
    with pytest.raises(ValueError, match="phases must have length"):
        TEBDBasis(m=2, n=2, phases=[0.1, 0.2])


# ---------- MERABasis ----------


def test_mera_basis_4x4_init_and_shape():
    b = MERABasis(m=2, n=2)
    assert b.m == 2 and b.n == 2
    # _n_mera_gates(2) = 2 per dim → 4 total phase gates + 4 Hadamards = 8
    assert b.n_row_gates == 2 and b.n_col_gates == 2
    assert len(b.tensors) == 8


def test_mera_basis_16x16_counts():
    # MERA needs powers of 2
    b = MERABasis(m=4, n=4)
    # _n_mera_gates(4) = 6 per dim; 8 Hadamards + 12 phase gates = 20
    assert b.n_row_gates == 6 and b.n_col_gates == 6
    assert len(b.tensors) == 20


def test_mera_basis_roundtrip():
    phases = [0.1 * i for i in range(4)]  # 2 + 2 = 4 gates for (2, 2)
    b = MERABasis(m=2, n=2, phases=phases)
    pic = _random_pic(2, 2, key_seed=4)
    recov = b.inverse_transform(b.forward_transform(pic))
    assert jnp.allclose(recov, pic, atol=1e-10)


def test_mera_basis_is_pytree():
    b = MERABasis(m=2, n=2)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert bases_allclose(restored, b)


def test_mera_basis_rejects_non_pow2():
    with pytest.raises(ValueError, match="power of 2"):
        MERABasis(m=3, n=2)


# ---------- Cross-basis training smoke test ----------


def test_training_works_for_all_new_bases():
    from pdft.loss import L1Norm
    from pdft.optimizers import RiemannianGD
    from pdft.training import train_basis

    target = _random_pic(2, 2, key_seed=5)
    for basis in (
        EntangledQFTBasis(m=2, n=2),
        TEBDBasis(m=2, n=2),
        MERABasis(m=2, n=2),
    ):
        result = train_basis(
            basis,
            target=target,
            loss=L1Norm(),
            optimizer=RiemannianGD(lr=0.01),
            steps=5,
            seed=0,
        )
        import numpy as np

        assert np.all(np.isfinite(result.loss_history))
        # Loss decreased at least once over 5 steps.
        assert min(result.loss_history) < result.loss_history[0]
