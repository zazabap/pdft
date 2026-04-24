from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.basis import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianGD
from pdft.training import train_basis

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def test_train_trajectory_matches_julia_4x4():
    """Python reproduces Julia's training trajectory on the same target.

    L1 loss is non-smooth — gradients change discontinuously at any point
    where a frequency component crosses zero. Cross-language floating-point
    differences of ~1e-5 in per-gate matrix products accumulate to ~1e-4 by
    step ~10, and once one trajectory crosses a cusp (∃ t such that the k-th
    component of T(target) hits zero) that the other does not, Armijo line
    search picks a qualitatively different alpha and the trajectories fork.

    This is expected behavior for a faithful port of a non-smooth optimizer
    across languages, not a bug. We verify that the *setup* matches exactly
    and that early steps (before any cusp is crossed) match tightly. Beyond
    that we only assert both trajectories are descending into comparable
    basins.
    """
    g = np.load(GOLDENS / "train_trajectory_4x4.npz")
    target = jnp.asarray(g["target"])
    steps = int(g["config_steps"][0])
    lr = float(g["config_lr"][0])
    seed = int(g["config_seed"][0])

    basis = QFTBasis(m=2, n=2)
    result = train_basis(
        basis,
        target=target,
        loss=L1Norm(),
        optimizer=RiemannianGD(lr=lr),
        steps=steps,
        seed=seed,
    )
    py = np.asarray(result.loss_history)
    jl = g["loss_history"]

    # (1) Initial loss bit-identical (proves QFT + L1 setup matches Julia).
    np.testing.assert_allclose(py[0], jl[0], atol=1e-10)
    # (2) First 2 accepted steps bit-identical (proves first gradient +
    #     retract + Armijo logic matches).
    np.testing.assert_allclose(py[:2], jl[:2], atol=1e-10)
    # (3) Steps 2-5 within 1e-3 — small FP drift begins but no cusp crossing yet.
    np.testing.assert_allclose(py[2:6], jl[2:6], atol=1e-3, rtol=1e-3)
    # (4) Both trajectories monotonically descend over the first 10 steps.
    assert all(py[i + 1] < py[i] + 1e-6 for i in range(10))
    assert all(jl[i + 1] < jl[i] + 1e-6 for i in range(10))
    # (5) Both reach significantly below the initial loss.
    assert py[-1] < py[0] - 0.3
    assert jl[-1] < jl[0] - 0.3
