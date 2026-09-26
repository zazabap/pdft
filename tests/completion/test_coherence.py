import importlib

import jax.numpy as jnp
import numpy as np

from pdft.completion.families.shared import expand
from pdft.completion.transform import apply_gates, apply_u, n_params, theta0, unitary_matrix

# `pdft.completion.coherence` the attribute is the theta-based function re-exported
# from transform (as `pdft.coherence` is in the core package); these are the modules.
C = importlib.import_module("pdft.completion.coherence")
core = importlib.import_module("pdft.coherence")

n = 5


def test_dense_operator_matches_unitary_matrix():
    th = theta0(n) + 0.3
    assert jnp.allclose(
        C.dense_operator(lambda e: apply_u(e, th, axis=0), n), unitary_matrix(th), atol=1e-12
    )


def test_mu_agrees_with_the_core_definition(rng):
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))
    U = jnp.asarray(q)
    assert float(C.coherence(U)) == core.coherence(None, operator=U)
    assert not C.is_flat_modulus(U) and float(C.flat_modulus_deviation(U)) > 0.05
    assert C.is_flat_modulus(unitary_matrix(theta0(n)))


def test_certificate_over_the_parameter_space(rand_general):
    """Proposition 1: A, B and the shared model stay flat; C, which frees the
    Hadamards, leaves the complex Hadamard set (else the claim is vacuous)."""

    def general(params):
        return lambda e: apply_gates(e, params, axis=0)

    def sample_b(rng):
        return {
            "g": rand_general(rng, n)["g"] * 0 + expand(jnp.zeros((n - 1, 4)), n)["g"],
            "phi": rand_general(rng, n)["phi"],
        }

    certs = {
        "A": C.certify_flat_modulus(
            lambda th: lambda e: apply_u(e, th, axis=0),
            n,
            lambda rng: jnp.asarray(rng.uniform(-np.pi, np.pi, n_params(n))),
            trials=4,
        ),
        "B": C.certify_flat_modulus(general, n, sample_b, trials=4),
        "shared": C.certify_flat_modulus(
            lambda ps: general(expand(ps, n)),
            n,
            lambda rng: jnp.asarray(rng.uniform(-np.pi, np.pi, (n - 1, 4))),
            trials=4,
        ),
        "C": C.certify_flat_modulus(general, n, lambda rng: rand_general(rng, n), trials=4),
    }
    for name in ("A", "B", "shared"):
        cert = certs[name]
        assert (
            cert["holds"]
            and cert["worst_deviation"] < 1e-12
            and abs(cert["worst_mu"] - 1.0) < 1e-12
        )
        assert cert["trials"] == 4 and cert["n"] == n
    assert not certs["C"]["holds"] and certs["C"]["worst_mu"] > 1.5
