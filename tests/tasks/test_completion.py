"""`complete` and `completion_loss`: recovery of an image from the pixels under a mask."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import cp_diagonals_view, with_tensors
from pdft.loss import topk_truncate
from pdft.tasks import complete, completion, completion_loss

from ..helpers import BASES, case_rng, single_precision


def _problem(basis, rng, rate=0.5):
    """A real image of the basis's size and a mask over it."""
    return (
        jnp.asarray(rng.random(basis.image_size)),
        jnp.asarray(rng.random(basis.image_size) < rate),
    )


@pytest.mark.parametrize("case", list(BASES))
def test_complete_runs_on_every_basis_and_keeps_the_observed_pixels(case):
    basis = BASES[case]()
    image, mask = _problem(basis, case_rng(case))
    out = complete(basis, image, mask, k=image.size // 4, steps=3)
    assert out.shape == basis.image_size and bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_array_equal(out[mask], image[mask])
    assert float(jnp.abs(out - image).max()) > 1e-3  # the rest is a reconstruction, not a copy
    # real, in the precision the basis's transforms return
    assert out.dtype == jnp.real(basis.inverse_transform(basis.forward_transform(image))).dtype


def test_complete_reads_only_the_pixels_under_the_mask():
    basis = pdft.QFTBasis(m=3, n=2)
    image, mask = _problem(basis, case_rng("unread"))
    elsewhere = jnp.where(mask, image, 7.0)
    reference = complete(basis, image, mask, k=9, steps=4)
    np.testing.assert_array_equal(complete(basis, elsewhere, mask, k=9, steps=4), reference)


def test_complete_at_its_two_ends():
    basis = pdft.QFTBasis(m=3, n=2)
    image, mask = _problem(basis, case_rng("ends"))
    # every pixel observed: nothing to fill in
    everything = jnp.ones_like(mask)
    np.testing.assert_array_equal(complete(basis, image, everything, k=5, steps=3), image)
    # every coefficient kept: no sparsity to impose, the zero-filled image is a fixed point
    kept_all = complete(basis, image, mask, k=image.size, steps=3)
    np.testing.assert_allclose(kept_all, jnp.where(mask, image, 0.0), atol=1e-14)


def test_one_step_is_threshold_then_data_consistency():
    basis = pdft.DCT4Basis(m=3, n=2)
    image, mask = _problem(basis, case_rng("one step"))
    zero_filled = jnp.where(mask, image, 0.0)
    sparse = topk_truncate(basis.forward_transform(zero_filled), 7, rtol=1e-8)
    filled = jnp.where(mask, image, jnp.real(basis.inverse_transform(sparse)))
    np.testing.assert_allclose(complete(basis, image, mask, k=7, steps=1), filled, atol=1e-14)
    assert int(jnp.sum(sparse != 0)) == 7
    # the second step starts from the first
    again = complete(basis, filled, jnp.ones_like(mask), k=7, steps=1)
    np.testing.assert_array_equal(again, filled)
    two = complete(basis, image, mask, k=7, steps=2)
    assert float(jnp.abs(two - filled).max()) > 1e-4


@pytest.mark.parametrize(
    "make",
    [
        lambda: pdft.RealRichBasis(m=4, n=4),
        lambda: pdft.DCT4Basis(m=4, n=4),
        lambda: pdft.QFTBasis(m=4, n=4),
    ],
    ids=["real_rich", "dct4", "qft"],
)
def test_complete_recovers_an_image_that_is_sparse_in_the_basis(make):
    basis = make()
    rng = np.random.default_rng(1)
    coefficients = np.zeros(256)
    where = rng.choice(256, size=6, replace=False)
    coefficients[where] = rng.normal(size=6) + 3.0 * np.sign(rng.normal(size=6))
    image = jnp.real(basis.inverse_transform(jnp.asarray(coefficients.reshape(16, 16))))
    k = int(jnp.sum(jnp.abs(basis.forward_transform(image)) > 1e-9))
    mask = jnp.asarray(rng.random((16, 16)) < 0.5)

    errors = [
        float(jnp.abs(complete(basis, image, mask, k=k, steps=s) - image).max())
        for s in (1, 10, 150)
    ]
    assert errors[0] > 0.1 and errors[1] < errors[0] / 10 and errors[2] < 1e-12


def test_the_precision_of_the_result_follows_the_transforms():
    image = np.random.default_rng(0).random((4, 4)).astype(np.float32)
    mask = jnp.asarray(np.random.default_rng(1).random((4, 4)) < 0.5)
    # QFT transforms compute in double whatever they are given; Rich ones do not cast
    qft = single_precision(pdft.QFTBasis(m=2, n=2))
    rich = single_precision(pdft.RichBasis(m=2, n=2))
    assert complete(qft, jnp.asarray(image), mask, k=5, steps=2).dtype == jnp.float64
    assert complete(rich, jnp.asarray(image), mask, k=5, steps=2).dtype == jnp.float32
    assert complete(rich, jnp.asarray(image, dtype=jnp.float64), mask, k=5, steps=2).dtype == (
        jnp.float64
    )


@pytest.mark.parametrize(("k", "steps"), [(0, 3), (3, 0)])
def test_complete_validates_its_budget(k, steps):
    basis = pdft.QFTBasis(m=2, n=2)
    image, mask = _problem(basis, case_rng("budget"))
    with pytest.raises(ValueError, match=f"k and steps must be positive, got k={k}, steps={steps}"):
        complete(basis, image, mask, k=k, steps=steps)


def test_complete_validates_its_image():
    basis = pdft.QFTBasis(m=2, n=3)
    image, mask = _problem(basis, case_rng("image"))
    with pytest.raises(ValueError, match=r"image size \(4, 8\), got \(8, 4\) and \(4, 8\)"):
        complete(basis, image.T, mask, k=3, steps=1)
    with pytest.raises(ValueError, match=r"image size \(4, 8\), got \(4, 8\) and \(32,\)"):
        complete(basis, image, mask.reshape(-1), k=3, steps=1)
    with pytest.raises(ValueError, match="observed is complex"):
        complete(basis, image + 0j, mask, k=3, steps=1)


def test_completion_loss_is_the_mean_squared_error_of_the_solver():
    basis = pdft.QFTBasis(m=3, n=2)
    rng = case_rng("loss")
    images = jnp.asarray(rng.random((3, 8, 4)))
    masks = jnp.asarray(rng.random((3, 8, 4)) < 0.5)
    objective = completion_loss(k=9, steps=3)
    solved = jnp.stack([complete(basis, x, o, k=9, steps=3) for x, o in zip(images, masks)])
    expected = float(jnp.mean((solved - images) ** 2))
    assert float(objective(basis, images, masks)) == pytest.approx(expected, rel=1e-13)
    assert float(jax.jit(objective)(basis, images, masks)) == pytest.approx(expected, rel=1e-13)
    assert expected > 1e-3


def test_the_gradient_through_the_solver_is_the_derivative():
    basis = pdft.QFTBasis(m=3, n=3)
    rng = case_rng("gradient")
    images = jnp.asarray(rng.random((2, 8, 8)))
    masks = jnp.asarray(rng.random((2, 8, 8)) < 0.5)
    objective = completion_loss(k=9, steps=3)
    # away from the Fourier point, where the coefficients are tied in pairs
    (angles,) = cp_diagonals_view.read(basis)
    angles = angles + jnp.asarray(0.3 * rng.normal(size=angles.shape))

    def loss(angles):
        return objective(cp_diagonals_view.write(basis, [angles]), images, masks)

    gradient = jax.grad(loss)(angles)
    assert float(jnp.abs(gradient).max()) > 1e-4
    for index in [(0, 0, 0), (1, 1, 1), (2, 0, 1), (5, 1, 0)]:
        step = jnp.zeros_like(angles).at[index].set(1e-6)
        numeric = (loss(angles + step) - loss(angles - step)) / 2e-6
        assert float(gradient[index]) == pytest.approx(float(numeric), abs=1e-8)


def test_the_real_part_is_taken_behind_an_optimization_barrier():
    """Not a numerical property, a compile-time one that no fast test can time.

    As of JAX 0.11, XLA carries a real part back through the complex gates
    before it, and compiling the solver takes exponentially long in the depth
    of the circuit: half an hour for this suite instead of a minute. The
    barrier in `complete` stops that and changes no number, so nothing else
    would notice it being tidied away.
    """
    basis = pdft.QFTBasis(m=2, n=2)
    image, mask = _problem(basis, case_rng("barrier"))
    traced = jax.make_jaxpr(lambda x: complete(basis, x, mask, k=3, steps=2))(image)
    assert "optimization_barrier" in str(traced)


def _nudged(basis, eps):
    """`basis` with one controlled phase moved by `eps`: far below any tolerance, above rounding."""
    index = basis.program.tensor_indices(kind="CP")[0]
    tensors = list(basis.tensors)
    tensors[index] = tensors[index].at[1, 1].mul(jnp.exp(1j * eps))
    return with_tensors(basis, tensors)


@pytest.mark.parametrize("k", [6, 8, 9])
def test_the_gradient_does_not_depend_on_rounding_when_the_cut_falls_in_a_pair(k, monkeypatch):
    # At the Fourier point the coefficients of a real image are tied in conjugate
    # pairs and an even k cuts through one. Nudging a phase either way decides
    # which of the two is the larger; the gradient must not notice.
    basis = pdft.QFTBasis(m=3, n=3)
    image, mask = _problem(basis, np.random.default_rng(2))

    def gradient(eps):
        nudged = _nudged(basis, eps)

        def loss(tensors):
            solved = complete(with_tensors(nudged, tensors), image, mask, k=k, steps=4)
            return jnp.mean((solved - image) ** 2)

        return np.concatenate([np.asarray(g).ravel() for g in jax.grad(loss)(list(nudged.tensors))])

    assert np.abs(gradient(1e-13) - gradient(-1e-13)).max() < 1e-12

    # the same comparison on upstream's exact rule: it is the band that makes it hold
    monkeypatch.setattr(completion, "topk_truncate", lambda c, k, rtol: topk_truncate(c, k))
    gap = np.abs(gradient(1e-13) - gradient(-1e-13)).max()
    assert (gap > 1e-3) if k % 2 == 0 else (gap < 1e-12)
