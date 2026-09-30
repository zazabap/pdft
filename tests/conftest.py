"""Pytest session setup: enable JAX x64 + shared fixtures for all tests.

Without x64, JAX operates in float32/complex64 and parity tolerances are
impossible to hit. Importing pdft does this too (see src/pdft/__init__.py),
but we set it here as well so property tests that use `jax.numpy` directly
(without importing pdft) also run in x64.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.circuit.gates import n_params

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def goldens_dir() -> Path:
    """Path to the Julia-generated reference goldens (read-only)."""
    return Path(__file__).resolve().parent.parent / "reference" / "goldens"


@pytest.fixture(scope="session")
def load_golden(goldens_dir):
    """Factory: ``load_golden("qft_code_4x4.npz")`` returns the loaded npz."""

    def _load(name: str):
        return np.load(goldens_dir / name)

    return _load


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def rand_theta():
    """Factory: random phase-only angles of an ``n``-wire register of the gate kernel."""

    def make(rng, n):
        return jnp.asarray(rng.uniform(0, 2 * np.pi, n_params(n)))

    return make


@pytest.fixture
def rand_general():
    """Factory: random U(2) gates and four random phases per wire pair, the
    gate kernel's whole parameter space."""

    def make(rng, n):
        a = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
        g = jnp.asarray(np.stack([np.linalg.qr(x)[0] for x in a]))
        return {"g": g, "phi": jnp.asarray(rng.uniform(-np.pi, np.pi, (n_params(n), 4)))}

    return make
