from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.loss import topk_truncate

GOLDENS = Path(__file__).resolve().parent.parent.parent / "reference" / "goldens"


def test_topk_truncate_matches_julia():
    g = np.load(GOLDENS / "loss_values.npz")
    pred = jnp.asarray(g["pred"])
    for k in (1, 3, 5):
        out = topk_truncate(pred, k)
        np.testing.assert_allclose(np.asarray(out), g[f"topk_{k}"], atol=1e-12)
