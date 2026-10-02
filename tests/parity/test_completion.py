"""Parity of image completion with the completion paper's own code.

Completion is not in ParametricDFT.jl. Its golden, `reference/completion/golden.npz`,
is a run of the paper repository (`reference/completion/generate_golden.py`): the
solver at the Fourier point, and Model B trained by plain Adam with a fresh batch
and mask per step. The paper works in the image's frame and this package in the
basis's, hence `frame=bit_reverse` and the reversals around `complete`.

Twenty training steps through a thresholding solver amplify rounding: the paper's
code on a CPU and on a GPU differs from itself by 2e-10 in the trained phases,
and this package from the golden by about as much. The tolerances below leave a
factor of a few hundred over that; a wrong gradient or tie rule is off by 1e-5
after one step.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import CP_DIAGONALS, cp_diagonals
from pdft.circuit import bit_reverse
from pdft.tasks import complete, completion_loss
from pdft.training import train_basis_steps

GOLDEN = Path(__file__).resolve().parent.parent.parent / "reference" / "completion" / "golden.npz"


@pytest.fixture(scope="module")
def golden():
    return dict(np.load(GOLDEN))


def _train(golden):
    """Model B through the solver, with the settings and the seed of the paper's run."""
    return train_basis_steps(
        pdft.QFTBasis(m=4, n=4),
        dataset=list(golden["images"][:6] / 255.0),
        objective=completion_loss(
            k=int(golden["train_k"]), steps=int(golden["train_solver_steps"])
        ),
        optimizer=pdft.RiemannianAdam(lr=float(golden["lr"])),
        view=CP_DIAGONALS,
        steps=int(golden["steps"]),
        rate=float(golden["rate"]),
        batch_size=int(golden["batch_size"]),
        seed=int(golden["seed"]),
        frame=bit_reverse,
    )


@pytest.fixture(scope="module")
def trained(golden):
    return _train(golden)


def _solve_held_out(basis, golden):
    image, mask = jnp.asarray(golden["images"][6] / 255.0), jnp.asarray(golden["mask"])
    return bit_reverse(
        complete(
            basis,
            bit_reverse(image),
            bit_reverse(mask),
            k=int(golden["solve_k"]),
            steps=int(golden["solve_steps"]),
        )
    )


def test_the_solver_matches_the_paper_at_the_fourier_point(golden):
    solved = _solve_held_out(pdft.QFTBasis(m=4, n=4), golden)
    np.testing.assert_allclose(solved, golden["solved_at_the_start"], atol=1e-12)


def test_the_training_losses_match_the_paper(trained, golden):
    np.testing.assert_allclose(trained.loss_history, golden["loss_history"], rtol=1e-7)


def test_the_trained_phases_match_the_paper(trained, golden):
    # the same angles; the paper numbers its gates and their four phases in another order
    ours = np.sort(np.asarray(cp_diagonals(trained.basis)).ravel())
    np.testing.assert_allclose(ours, np.sort(golden["trained_phases"].ravel()), atol=1e-7)
    assert (
        np.abs(ours - np.sort(np.asarray(cp_diagonals(pdft.QFTBasis(m=4, n=4))).ravel())).max()
        > 0.05
    )


def test_the_trained_basis_solves_as_the_papers_does(trained, golden):
    solved = _solve_held_out(trained.basis, golden)
    np.testing.assert_allclose(solved, golden["solved_after_training"], atol=1e-7)
    assert np.abs(solved - golden["solved_at_the_start"]).max() > 1e-3
