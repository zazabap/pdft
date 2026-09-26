import importlib

import jax.numpy as jnp
import numpy as np

import pdft.completion.coherence as C
from pdft.completion.transform import apply_gates, hadamards, n_params, theta0, theta_to_params

# The core package re-exports a function named coherence at its root, which shadows its module.
core = importlib.import_module("pdft.coherence")

n = 5


def test_dense_operator_is_the_matrix_of_the_closure():
    th = theta0(n) + 0.3
    U = C.dense_operator(lambda e: apply_gates(e, theta_to_params(th), axis=0), n)
    e = jnp.zeros(2**n, jnp.complex128).at[3].set(1.0)
    assert jnp.allclose(U[:, 3], apply_gates(e, theta_to_params(th)), atol=1e-12)
    assert jnp.allclose(jnp.conj(U).T @ U, jnp.eye(2**n), atol=1e-12)


def test_mu_agrees_with_the_core_definition(rng):
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))
    U = jnp.asarray(q)
    assert float(C.coherence(U)) == core.coherence(None, operator=U)
    assert not C.is_flat_modulus(U) and float(C.flat_modulus_deviation(U)) > 0.05
    assert C.is_flat_modulus(
        C.dense_operator(lambda e: apply_gates(e, theta_to_params(theta0(n)), axis=0), n)
    )


def test_certificate_over_the_parameter_space(rand_general):
    """Proposition 1: the phase-only (A) and four-phase (B) circuits stay flat;
    freeing the Hadamards (C) leaves the complex Hadamard set, else the claim
    would be vacuous."""

    def general(params):
        return lambda e: apply_gates(e, params, axis=0)

    def sample_a(rng):
        return jnp.asarray(rng.uniform(-np.pi, np.pi, n_params(n)))

    def sample_b(rng):
        return {"g": hadamards(n), "phi": rand_general(rng, n)["phi"]}

    certs = {
        "A": C.certify_flat_modulus(lambda th: general(theta_to_params(th)), n, sample_a, trials=4),
        "B": C.certify_flat_modulus(general, n, sample_b, trials=4),
        "C": C.certify_flat_modulus(general, n, lambda rng: rand_general(rng, n), trials=4),
    }
    for name in ("A", "B"):
        cert = certs[name]
        assert (
            cert["holds"]
            and cert["worst_deviation"] < 1e-12
            and abs(cert["worst_mu"] - 1.0) < 1e-12
        )
        assert cert["trials"] == 4 and cert["n"] == n
    assert not certs["C"]["holds"] and certs["C"]["worst_mu"] > 1.5
