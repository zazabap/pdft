from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.basis import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianAdam
from pdft.training import train_basis

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def test_adam_trajectory_matches_julia_4x4():
    """Full 50-step Adam trajectory matches Julia to atol=1e-6.

    After the JAX/Zygote gradient conjugation fix (Wirtinger convention
    alignment), Adam trajectories agree within floating-point noise of
    moment-update accumulation (~2e-8 over 50 steps in practice).
    """
    g = np.load(GOLDENS / "adam_trajectory_4x4.npz")
    target = jnp.asarray(g["target"])
    steps = int(g["config_steps"][0])
    lr = float(g["config_lr"][0])
    seed = int(g["config_seed"][0])

    basis = QFTBasis(m=2, n=2)
    result = train_basis(
        basis,
        target=target,
        loss=L1Norm(),
        optimizer=RiemannianAdam(lr=lr),
        steps=steps,
        seed=seed,
    )
    np.testing.assert_allclose(
        np.asarray(result.loss_history),
        g["loss_history"],
        atol=1e-6,
        rtol=1e-6,
    )
