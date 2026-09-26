"""Shared factories for the completion tests: random parameters of each family
and small image stacks. Fixtures return callables so a test picks its own rng
and register width."""

import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.transform import n_params


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
