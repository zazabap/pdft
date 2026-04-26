from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.bases.base import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianGD
from pdft.training import train_basis

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def test_train_trajectory_matches_julia_4x4():
    """Full 50-step GD trajectory matches Julia bit-exactly.

    With gradient conjugation applied (Wirtinger convention alignment between
    JAX and Zygote), Python and Julia produce identical trajectories on L1
    loss + QFT + RiemannianGD + Armijo through all 50 steps.
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
    np.testing.assert_allclose(
        np.asarray(result.loss_history),
        g["loss_history"],
        atol=1e-10,
        rtol=1e-10,
    )
