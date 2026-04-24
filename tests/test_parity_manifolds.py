from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.manifolds import UnitaryManifold

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def test_unitary_project_retract_match_julia():
    g = np.load(GOLDENS / "manifold_project_retract.npz")
    U = jnp.asarray(g["U"])
    G = jnp.asarray(g["G"])
    alpha = float(g["alpha"][0])
    M = UnitaryManifold()
    Xi = M.project(U, G)
    np.testing.assert_allclose(np.asarray(Xi), g["Xi"], atol=1e-10)
    U_new = M.retract(U, Xi, alpha)
    np.testing.assert_allclose(np.asarray(U_new), g["U_new"], atol=1e-10)
