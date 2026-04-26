"""Optimizer-test fixtures: random unitary tensors for property checks."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def random_unitary_tensors():
    """Factory: ``random_unitary_tensors(d=4, count=3)`` -> list of d×d unitaries."""

    def _make(d: int, count: int, seed: int = 0):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (count, d, d)) + 1j * jax.random.normal(k2, (count, d, d))
        Q, _ = jnp.linalg.qr(A)
        return [Q[i].astype(jnp.complex128) for i in range(count)]

    return _make
