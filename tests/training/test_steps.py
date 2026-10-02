"""`train_basis_steps`: a fixed number of Adam steps, each on a fresh batch under a fresh mask."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import CP_DIAGONALS, CP_PHASES, TENSORS, bases_allclose
from pdft.circuit import bit_reverse
from pdft.tasks import completion_loss
from pdft.training import TrainingResult, train_basis_steps

from ..helpers import PlainAdam, case_rng


def _smooth_images(count: int, seed: int = 0) -> list[np.ndarray]:
    """A few 8x8 images with structure a Fourier-like basis can be trained on."""
    yy, xx = np.mgrid[0:8, 0:8] / 8.0
    waves = np.random.default_rng(seed).uniform(0, 2, (count, 4))
    return [
        0.5
        + 0.2 * np.cos(2 * np.pi * (a * xx + b * yy) + c)
        + 0.1 * np.cos(2 * np.pi * (2 * xx - yy) + d)
        for a, b, c, d in waves
    ]


def _draws(images, steps, batch_size, rate, seed, frame=lambda a: a):
    """The batches and masks of a run, restated: the batch, then its masks, from one stream."""
    rng = np.random.default_rng(seed)
    images = np.stack(images)
    for _ in range(steps):
        batch = images[rng.choice(len(images), size=min(batch_size, len(images)), replace=False)]
        masks = rng.random(batch.shape) < rate
        yield frame(jnp.asarray(batch)), frame(jnp.asarray(masks))


def _run(basis, images, objective, **kwargs):
    settings = {"optimizer": pdft.RiemannianAdam(lr=0.02), "steps": 6, "rate": 0.4, "batch_size": 3}
    return train_basis_steps(basis, dataset=images, objective=objective, **(settings | kwargs))


@pytest.mark.parametrize("batch_size", [3, 20])
@pytest.mark.parametrize("frame", [None, bit_reverse], ids=["own frame", "bit_reverse"])
def test_each_step_draws_a_batch_then_its_masks_and_frames_both(batch_size, frame):
    basis = pdft.QFTBasis(m=3, n=3)
    images = _smooth_images(5)
    weights = jnp.asarray(case_rng("weights").normal(size=(8, 8)))

    def probe(basis, images, masks):  # a function of the batch alone, with no symmetry
        return jnp.sum(jnp.where(masks, images * weights, 0.25 * weights))

    result = _run(basis, images, probe, batch_size=batch_size, seed=7, frame=frame)
    restated = _draws(images, 6, batch_size, 0.4, 7, frame or (lambda a: a))
    np.testing.assert_allclose(result.loss_history, [probe(None, x, o) for x, o in restated])
    # an objective that does not depend on the basis leaves it where it was
    assert bases_allclose(result.basis, basis, atol=0.0)
    assert len({round(v, 9) for v in result.loss_history}) == 6  # a fresh draw at every step


@pytest.mark.parametrize("clip", [None, 0.004], ids=["unclipped", "clipped"])
@pytest.mark.parametrize("view", [CP_DIAGONALS, CP_PHASES], ids=lambda v: v.name)
def test_on_a_flat_view_the_run_is_plain_adam_on_what_the_view_reads(view, clip):
    basis = pdft.QFTBasis(m=3, n=3)
    images = _smooth_images(5)
    objective = completion_loss(k=7, steps=3)
    lr, beta1, beta2, eps = 0.02, 0.8, 0.99, 1e-8
    optimizer = pdft.RiemannianAdam(lr=lr, beta1=beta1, beta2=beta2, eps=eps, max_grad_norm=clip)
    result = _run(basis, images, objective, optimizer=optimizer, view=view, seed=3, steps=5)

    # the same run restated: the draws above, jax's gradient, plain Adam
    value_and_grad = jax.value_and_grad(lambda a, x, o: objective(view.write(basis, [a]), x, o))
    (angles,) = (np.asarray(p) for p in view.read(basis))
    adam, losses, norms = PlainAdam(lr, beta1, beta2, eps), [], []
    for x, o in _draws(images, 5, 3, 0.4, 3):
        loss, gradient = value_and_grad(jnp.asarray(angles), x, o)
        losses.append(float(loss))
        norms.append(float(jnp.linalg.norm(gradient)))
        scale = 1.0 if clip is None else min(1.0, clip / norms[-1])
        angles = adam.step(angles, scale * np.asarray(gradient))
    assert clip is None or norms[0] > 1.1 * clip  # the clip bites from the first step

    np.testing.assert_allclose(result.loss_history, losses, rtol=1e-10)
    np.testing.assert_allclose(view.read(result.basis)[0], angles, atol=1e-10)
    assert np.abs(angles - np.asarray(view.read(basis)[0])).max() > 0.01

    # nothing the view does not read has moved, and what it wrote is what it writes
    cp = set(basis.program.tensor_indices(kind="CP"))
    for i, (before, after) in enumerate(zip(basis.tensors, result.basis.tensors)):
        assert (i in cp) or (after is before)
    assert bases_allclose(view.write(basis, view.read(result.basis)), result.basis, atol=1e-15)
    assert isinstance(result, TrainingResult) and type(result.basis) is pdft.QFTBasis
    assert (result.steps, result.seed, len(result.loss_history)) == (5, 3, 5)
    assert result.wall_time_s > 0


@pytest.mark.parametrize("view", [TENSORS, CP_DIAGONALS, CP_PHASES], ids=lambda v: v.name)
def test_training_through_the_solver_lowers_the_loss_on_problems_it_did_not_see(view):
    basis = pdft.QFTBasis(m=3, n=3)
    objective = completion_loss(k=7, steps=6)
    result = _run(
        basis, _smooth_images(8), objective, view=view, steps=60, batch_size=4, frame=bit_reverse
    )
    held_images = bit_reverse(jnp.asarray(np.stack(_smooth_images(8, seed=1))))
    held_masks = bit_reverse(jnp.asarray(np.random.default_rng(9).random((8, 8, 8)) < 0.4))
    before = float(objective(basis, held_images, held_masks))
    after = float(objective(result.basis, held_images, held_masks))
    assert after < 0.9 * before


def test_the_default_view_moves_the_tensors_on_their_manifolds():
    basis = pdft.QFTBasis(m=3, n=3)
    hadamards = basis.program.tensor_indices(kind="H")
    result = _run(basis, _smooth_images(5), completion_loss(k=7, steps=3))
    for i, (before, after) in enumerate(zip(basis.tensors, result.basis.tensors)):
        assert float(jnp.abs(after - before).max()) > 1e-4 and after.dtype == jnp.complex128
        if i in hadamards:
            np.testing.assert_allclose(after @ after.conj().T, np.eye(2), atol=1e-13)
        else:
            np.testing.assert_allclose(jnp.abs(after), 1.0, atol=1e-13)

    frozen = _run(basis, _smooth_images(5), completion_loss(k=7, steps=3), frozen_indices=hadamards)
    for i, (before, after) in enumerate(zip(basis.tensors, frozen.basis.tensors)):
        same = np.asarray(after).tobytes() == np.asarray(before).tobytes()
        assert same == (i in hadamards)


def test_the_callback_sees_every_step():
    basis = pdft.QFTBasis(m=3, n=3)
    seen = []
    result = _run(
        basis,
        _smooth_images(5),
        completion_loss(k=7, steps=3),
        view=CP_DIAGONALS,
        callback=lambda step, trained, loss: seen.append((step, trained, loss)),
    )
    assert [step for step, _, _ in seen] == list(range(6))
    assert [loss for _, _, loss in seen] == result.loss_history
    assert all(type(trained) is pdft.QFTBasis for _, trained, _ in seen)
    assert bases_allclose(seen[-1][1], result.basis, atol=0.0)
    assert not bases_allclose(seen[0][1], seen[1][1], atol=1e-6)


def test_images_keep_their_precision_and_integers_become_float64():
    basis = pdft.QFTBasis(m=3, n=3)
    seen = []

    def objective(basis, images, masks):
        seen.append((images.dtype, masks.dtype, images.shape))
        return jnp.mean(jnp.abs(basis.forward_transform(images[0])))

    images = _smooth_images(4)
    for cast in (np.float64, np.float32, lambda a: np.rint(a * 255).astype(np.uint8)):
        _run(basis, [cast(x) for x in images], objective, steps=1)
    assert seen == [
        (jnp.float64, jnp.bool_, (3, 8, 8)),
        (jnp.float32, jnp.bool_, (3, 8, 8)),
        (jnp.float64, jnp.bool_, (3, 8, 8)),
    ]


def test_a_loss_that_is_not_finite_stops_the_run():
    basis = pdft.QFTBasis(m=3, n=3)

    def objective(basis, images, masks):
        return jnp.log(-jnp.sum(jnp.abs(basis.forward_transform(images[0]))))

    with pytest.raises(FloatingPointError, match="non-finite loss at step 0"):
        _run(basis, _smooth_images(4), objective)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"optimizer": pdft.RiemannianGD()}, TypeError, "takes a RiemannianAdam, got RiemannianGD"),
        ({"optimizer": "adam"}, TypeError, "takes a RiemannianAdam, got str"),
        ({"steps": 0}, ValueError, "steps must be >= 1, got 0"),
        ({"batch_size": 0}, ValueError, "batch_size must be >= 1, got 0"),
        ({"rate": 0.0}, ValueError, r"rate must be in \(0, 1\], got 0.0"),
        ({"rate": 1.5}, ValueError, r"rate must be in \(0, 1\], got 1.5"),
        ({"dataset": []}, ValueError, "dataset must be non-empty"),
        ({"dataset": [np.zeros((4, 8))]}, ValueError, r"dataset\[0\] has shape \(4, 8\)"),
        ({"dataset": [np.zeros((8, 8), dtype=complex)]}, ValueError, r"dataset\[0\] is complex"),
        ({"frozen_indices": [14]}, ValueError, "out-of-range index 14"),
        ({"frozen_indices": [1], "view": CP_PHASES}, ValueError, "out-of-range index 1"),
    ],
)
def test_arguments_are_validated(kwargs, error, message):
    basis = pdft.QFTBasis(m=3, n=3)
    settings = {
        "dataset": _smooth_images(3),
        "objective": completion_loss(k=7, steps=2),
        "optimizer": pdft.RiemannianAdam(),
        "steps": 2,
        "rate": 0.4,
    }
    with pytest.raises(error, match=message):
        train_basis_steps(basis, **(settings | kwargs))


def test_a_rate_of_one_observes_every_pixel():
    basis = pdft.QFTBasis(m=3, n=3)

    def objective(basis, images, masks):
        hidden = jnp.sum(~masks).astype(jnp.float64)
        return hidden + 0.0 * jnp.sum(jnp.abs(basis.forward_transform(images[0])))

    assert _run(basis, _smooth_images(3), objective, rate=1.0, steps=2).loss_history == [0.0, 0.0]
