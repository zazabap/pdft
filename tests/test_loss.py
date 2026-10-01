from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from pdft.bases.circuit.qft import qft_code
from pdft.circuit.builder import GATE_SHAPES
from pdft.loss import L1Norm, MSELoss, loss_function, topk_truncate

from .helpers import random_unitary


def test_l1norm_is_stateless():
    a, b = L1Norm(), L1Norm()
    assert a == b


def test_mseloss_requires_positive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        MSELoss(k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        MSELoss(k=-1)


def test_mseloss_stores_k():
    assert MSELoss(k=5).k == 5


def test_topk_truncate_k_equals_length_is_identity():
    x = jnp.array([[3.0 + 0j, -1.0, 2.0, 0.5]])
    out = topk_truncate(x, k=4)
    assert jnp.allclose(out, x)


def test_topk_truncate_zero_k_zeros_everything():
    x = jnp.array([[3.0 + 0j, -1.0, 2.0, 0.5]])
    out = topk_truncate(x, k=0)
    assert jnp.allclose(out, jnp.zeros_like(x))


def test_topk_truncate_keeps_largest_magnitudes():
    x = jnp.array([[1.0 + 0j, -3.0, 2.0, 0.1]])
    out = topk_truncate(x, k=2)
    expected = jnp.array([[0.0 + 0j, -3.0, 2.0, 0.0]])
    assert jnp.allclose(out, expected)


def test_topk_truncate_k_larger_than_length_clamps():
    x = jnp.array([[1.0 + 0j, 2.0]])
    out = topk_truncate(x, k=10)
    assert jnp.allclose(out, x)


def test_mseloss_no_extra_loss_unchanged():
    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    base_loss = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1), inverse_code=inv_code))

    assert base_loss == base_loss


def test_mseloss_extra_loss_hook_adds_term():
    @dataclass(frozen=True)
    class WithExtra(MSELoss):
        def _extra_loss(self, tensors):
            return jnp.asarray(7.0, dtype=jnp.float64)

    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    base_loss = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1), inverse_code=inv_code))
    extra_loss = float(
        loss_function(tensors, m, n, code, pic, WithExtra(k=1), inverse_code=inv_code)
    )

    assert abs(extra_loss - base_loss - 7.0) < 1e-10


def test_mseloss_extra_loss_uses_tensors():
    @dataclass(frozen=True)
    class TensorSum(MSELoss):
        def _extra_loss(self, tensors):
            return sum(jnp.sum(jnp.abs(t) ** 2) for t in tensors).real

    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.zeros((4, 4), dtype=jnp.complex128)

    loss_val = float(loss_function(tensors, m, n, code, pic, TensorSum(k=1), inverse_code=inv_code))
    expected_reg = float(sum(jnp.sum(jnp.abs(t) ** 2) for t in tensors).real)

    assert abs(loss_val - expected_reg) < 1e-6


def test_mse_reconstruction_uses_the_adjoint_at_non_symmetric_tensors():
    """Keeping every coefficient reconstructs the image exactly, whatever the unitary
    gates are. That needs ``conj(tensors)`` through the inverse code: the inverse
    code alone is the transpose, which only inverts gates that happen to be symmetric,
    as the initial ones are."""
    import jax
    import numpy as np

    import pdft

    rng = np.random.default_rng(0)

    for basis in (pdft.QFTBasis(m=2, n=2), pdft.RichBasis(m=2, n=2)):
        tensors = []
        for kind, _ in basis.program.sorted_steps:
            if kind == "CP":
                tensors.append(jnp.asarray(np.exp(1j * rng.uniform(-3, 3, (2, 2)))))
            else:
                shape = GATE_SHAPES[kind]
                d = round(np.prod(shape) ** 0.5)
                tensors.append(jnp.asarray(random_unitary(rng, d)).reshape(shape))
        pic = jnp.asarray(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        args = (tensors, 2, 2, basis.code, pic)
        full = loss_function(*args, MSELoss(k=16), inverse_code=basis.inv_code)
        assert float(full) < 1e-24
        # and with fewer coefficients it is the energy of the ones dropped
        coefficients = np.sort(
            np.abs(np.asarray(basis.code(*tensors, pic.reshape((2,) * 4)))).ravel()
        )
        kept = loss_function(*args, MSELoss(k=5), inverse_code=basis.inv_code)
        assert float(kept) == pytest.approx(float(np.sum(coefficients[:-5] ** 2)), rel=1e-10)
        gradient = jax.grad(
            lambda ts: loss_function(ts, *args[1:], MSELoss(k=5), inverse_code=basis.inv_code)
        )(tensors)
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in gradient)


def test_basis_loss_and_mean_loss_are_loss_function_over_a_basis():
    import numpy as np

    import pdft
    from pdft.loss import basis_loss, mean_loss

    basis = pdft.RichBasis(m=2, n=2)
    rng = np.random.default_rng(1)
    images = jnp.asarray(rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4)))
    tensors = [t + 0.01 * (i + 1) for i, t in enumerate(basis.tensors)]
    for loss in (L1Norm(), MSELoss(k=5)):
        per_image = basis_loss(basis, loss)
        direct = [
            loss_function(tensors, 2, 2, basis.code, image, loss, inverse_code=basis.inv_code)
            for image in images
        ]
        assert [float(per_image(tensors, image)) for image in images] == [float(v) for v in direct]
        # the tensors are the argument: the basis's own are not read
        assert float(per_image(basis.tensors, images[0])) != float(direct[0])
        assert float(mean_loss(basis, loss)(tensors, images)) == pytest.approx(
            float(np.mean([float(v) for v in direct])), rel=1e-14
        )
