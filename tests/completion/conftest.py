"""Shared factories for the completion tests: random parameters of each family
and small image stacks. Fixtures return callables so a test picks its own rng
and register width."""

import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.transform import apply_gates, n_params, separable, theta0, theta_to_params


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def rand_theta():
    def make(rng, n):
        return jnp.asarray(rng.uniform(0, 2 * np.pi, n_params(n)))

    return make


@pytest.fixture
def rand_general():
    """Random U(2) gates and four random phases per pair: model C's parameter space."""

    def make(rng, n):
        a = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
        g = jnp.asarray(np.stack([np.linalg.qr(x)[0] for x in a]))
        return {"g": g, "phi": jnp.asarray(rng.uniform(-np.pi, np.pi, (n_params(n), 4)))}

    return make


@pytest.fixture
def images():
    def make(n=4, count=3, seed=0, dtype=np.float64):
        return np.random.default_rng(seed).random((count, 2**n, 2**n)).astype(dtype)

    return make


@pytest.fixture
def sparse_problem():
    """A real image exactly ``2k``-sparse in the DFT domain and a mask at rate ``p``.

    Well-separated coefficient magnitudes, so top-k has no near-ties and two
    solvers built from different operators of the same matrix agree to round-off.
    """

    def make(n=5, seed=0, k=12, p=0.5):
        N = 2**n
        rng = np.random.default_rng(seed)
        C = np.zeros((N, N), complex)
        C.reshape(-1)[rng.choice(N * N, size=k, replace=False)] = rng.standard_normal(
            k
        ) + 1j * rng.standard_normal(k)
        C = C + np.conj(
            C[(-np.arange(N)) % N][:, (-np.arange(N)) % N]
        )  # Hermitian, so the image is real
        th = theta0(n)
        _, synthesis = separable(apply_gates)
        p0 = theta_to_params(th)
        return jnp.real(synthesis(jnp.asarray(C), p0, p0)), jnp.asarray(rng.random((N, N)) < p), th

    return make
