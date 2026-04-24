import jax
import jax.numpy as jnp

import pdft


def test_full_training_loop_no_nan():
    target = jax.random.normal(jax.random.PRNGKey(11), (4, 4)).astype(jnp.complex128)
    basis = pdft.QFTBasis(m=2, n=2)
    result = pdft.train_basis(
        basis,
        target=target,
        loss=pdft.L1Norm(),
        optimizer=pdft.RiemannianGD(lr=0.01),
        steps=20,
        seed=0,
    )
    losses = jnp.asarray(result.loss_history)
    assert jnp.all(jnp.isfinite(losses))
    # Each tensor preserves its initial manifold classification after training:
    # Hadamards stay unitary, diagonal-CPs stay element-wise unit-modulus.
    for t_init, t_trained in zip(basis.tensors, result.basis.tensors):
        manifold = pdft.classify_manifold(t_init)
        if isinstance(manifold, pdft.UnitaryManifold):
            d = t_trained.shape[0]
            I_mat = jnp.eye(d, dtype=t_trained.dtype)
            assert jnp.allclose(t_trained @ jnp.conj(t_trained).T, I_mat, atol=1e-6)
        elif isinstance(manifold, pdft.PhaseManifold):
            assert jnp.allclose(jnp.abs(t_trained), 1.0, atol=1e-6)


def test_public_api_is_re_exported():
    # All Phase 1 public names available directly on the package
    expected = {
        "AbstractLoss", "L1Norm", "MSELoss", "topk_truncate", "loss_function",
        "qft_code", "ft_mat", "ift_mat",
        "AbstractRiemannianManifold", "UnitaryManifold", "PhaseManifold",
        "classify_manifold", "group_by_manifold",
        "AbstractSparseBasis", "QFTBasis", "bases_allclose",
        "RiemannianGD", "optimize",
        "train_basis", "TrainingResult",
    }
    for name in expected:
        assert hasattr(pdft, name), f"missing export: {name}"


def test_upstream_ref_set():
    assert pdft.__upstream_ref__.startswith("nzy1997/ParametricDFT.jl@")
    assert "unpinned" not in pdft.__upstream_ref__
