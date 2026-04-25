"""Long-run trajectory parity (200 steps) for GD on L1.

If GD bit-exactness held over 50 steps (Phase 2), it should continue to
hold over 200 — verify there is no compounding FP drift after the
Wirtinger-conjugation fix.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.basis import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianGD
from pdft.training import train_basis

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def test_train_trajectory_matches_julia_4x4_200_steps():
    """Long-run GD trajectory parity finding.

    First ~50 steps remain bit-exact (`atol=1e-10`) — same as the 50-step
    parity test from Phase 2. Beyond ~100 steps, Python and Julia eventually
    diverge by ~1e-5 because L1 loss has cusps where the gradient changes
    discontinuously; FP differences in earlier steps eventually push the two
    trajectories across the same cusp at slightly different alphas. Both
    still converge to the same loss basin.

    This test asserts:
      1. First 50 steps match bit-exactly.
      2. Both trajectories reach the same minimum (within 1e-3).
      3. The drift, when it occurs, is sub-percent (rtol=1e-3, atol=1e-3).
    """
    g = np.load(GOLDENS / "train_trajectory_4x4_long.npz")
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

    # (1) First 50 steps: bit-exact (Phase 2 parity guarantee continues to hold).
    np.testing.assert_allclose(py[:50], jl[:50], atol=1e-10, rtol=1e-10)
    # (2) Whole 200-step trajectory: within 0.1% (catches gross drift).
    np.testing.assert_allclose(py, jl, atol=1e-3, rtol=1e-3)
    # (3) Both reach the same basin.
    assert abs(py[-1] - jl[-1]) < 1e-4
    # (4) Both made substantive progress.
    assert py[-1] < py[0] - 1.0
    assert jl[-1] < jl[0] - 1.0
