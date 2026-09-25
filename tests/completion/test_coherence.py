import importlib

import jax.numpy as jnp
import numpy as np

from pdft.completion.families.general import apply_general, init_general
from pdft.completion.families.shared import expand
from pdft.completion.transform import apply_u, n_params, theta0, unitary_matrix

# `pdft.completion.coherence` the attribute is the theta-based function re-exported
# from transform (as `pdft.coherence` is in the core package); this is the module.
C = importlib.import_module("pdft.completion.coherence")
core = importlib.import_module("pdft.coherence")

n = 5


def test_dense_operator_matches_unitary_matrix():
    th = theta0(n) + 0.3
    U = C.dense_operator(lambda e: apply_u(e, th, n, axis=0), n)
    assert jnp.allclose(U, unitary_matrix(th, n), atol=1e-12)


def test_mu_agrees_with_the_core_definition():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))
    U = jnp.asarray(q)
    assert float(C.coherence(U)) == core.coherence(None, operator=U)
    assert C.is_flat_modulus(U) is False
    assert float(C.flat_modulus_deviation(U)) > 0.05
    assert C.is_flat_modulus(unitary_matrix(theta0(n), n))


def _general(params):
    return lambda e: apply_general(e, params, n, adjoint=False, axis=0)


def test_certificate_over_the_parameter_space():
    """Proposition 1: A, B and the shared model stay flat; C, which frees the
    Hadamards, leaves the complex Hadamard set (else the claim is vacuous)."""

    def sample_a(rng):
        return jnp.asarray(rng.uniform(-np.pi, np.pi, n_params(n)))

    def sample_b(rng):
        p = init_general(n)
        return {"g": p["g"], "phi": jnp.asarray(rng.uniform(-np.pi, np.pi, (n_params(n), 4)))}

    def sample_shared(rng):
        return jnp.asarray(rng.uniform(-np.pi, np.pi, (n - 1, 4)))

    def sample_c(rng):
        p = init_general(n)
        a = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
        g = jnp.asarray(np.stack([np.linalg.qr(x)[0] for x in a]))
        return {"g": g, "phi": p["phi"]}

    cert_a = C.certify_flat_modulus(
        lambda th: lambda e: apply_u(e, th, n, axis=0), n, sample_a, trials=4
    )
    cert_b = C.certify_flat_modulus(_general, n, sample_b, trials=4)
    cert_s = C.certify_flat_modulus(lambda ps: _general(expand(ps, n)), n, sample_shared, trials=4)
    cert_c = C.certify_flat_modulus(_general, n, sample_c, trials=4)
    for cert in (cert_a, cert_b, cert_s):
        assert cert["holds"] and cert["worst_deviation"] < 1e-12
        assert abs(cert["worst_mu"] - 1.0) < 1e-12
        assert cert["trials"] == 4 and cert["n"] == n
    assert not cert_c["holds"] and cert_c["worst_mu"] > 1.5
