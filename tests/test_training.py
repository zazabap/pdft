import jax.numpy as jnp

from pdft.basis import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianGD
from pdft.training import TrainingResult, train_basis


def test_train_basis_returns_training_result_and_does_not_increase_loss():
    # Use a non-trivial target so the initial QFT is far from the L1-optimal
    # parameters — Armijo line search can make real progress.
    import jax

    target = jax.random.normal(jax.random.PRNGKey(7), (4, 4)).astype(jnp.complex128)
    basis = QFTBasis(m=2, n=2)
    result = train_basis(
        basis,
        target=target,
        loss=L1Norm(),
        optimizer=RiemannianGD(lr=0.01),
        steps=10,
        seed=0,
    )
    assert isinstance(result, TrainingResult)
    assert result.steps == 10
    assert len(result.loss_history) == 11  # initial + 10 updates
    # Final loss should be strictly less than initial on a non-optimal start.
    assert result.loss_history[-1] < result.loss_history[0]


def test_train_basis_deterministic_for_fixed_seed():
    target = jnp.ones((4, 4), dtype=jnp.complex128)

    def run():
        basis = QFTBasis(m=2, n=2)
        return train_basis(
            basis,
            target=target,
            loss=L1Norm(),
            optimizer=RiemannianGD(lr=0.01),
            steps=5,
            seed=0,
        )

    a, b = run(), run()
    assert jnp.allclose(jnp.array(a.loss_history), jnp.array(b.loss_history))


def test_train_basis_rejects_zero_steps():
    import pytest

    target = jnp.ones((4, 4), dtype=jnp.complex128)
    with pytest.raises(ValueError):
        train_basis(
            QFTBasis(m=2, n=2),
            target=target,
            loss=L1Norm(),
            optimizer=RiemannianGD(),
            steps=0,
        )
