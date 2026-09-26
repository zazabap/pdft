import jax.numpy as jnp
import numpy as np

from pdft.completion.coherence import certify_flat_modulus
from pdft.completion.families import shared as S
from pdft.completion.families.general import init_general
from pdft.completion.transform import apply_gates, hadamards, theta0


def test_dist_index_and_counts():
    assert list(S.dist_index(4)) == [0, 1, 2, 0, 1, 0] and S.n_shared(9) == 32


def test_init_shared_expands_to_the_dft():
    n = 5
    p = S.expand(S.init_shared(n), n)
    assert jnp.allclose(p["phi"], init_general(n)["phi"], atol=1e-12)
    assert jnp.allclose(p["phi"][:, 3], theta0(n), atol=1e-12)
    assert jnp.array_equal(p["g"], hadamards(n))


def test_extend_keeps_fitted_distances_and_falls_back_to_the_dft():
    psi = S.init_shared(4) + 0.1
    up = S.extend(psi, 6)
    assert (
        up.shape == (5, 4)
        and jnp.allclose(up[:3], psi)
        and jnp.allclose(up[3:], S.init_shared(6)[3:])
    )
    down = S.extend(psi, 3)
    assert down.shape == (2, 4) and jnp.allclose(down, psi[:2])
    assert np.asarray(S.init_shared(3)).shape == (2, 4)


def test_shared_phases_stay_flat_modulus():
    """Proposition 1 holds over the whole distance-shared parameter space, since
    nothing in it touches the Hadamards."""
    n = 5
    cert = certify_flat_modulus(
        lambda ps: lambda e: apply_gates(e, S.expand(ps, n), axis=0),
        n,
        lambda rng: jnp.asarray(rng.uniform(-np.pi, np.pi, (n - 1, 4))),
        trials=4,
    )
    assert cert["holds"] and cert["worst_deviation"] < 1e-12
