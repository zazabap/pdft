"""Training-test fixtures: tiny target image and tiny dataset."""

import jax.numpy as jnp
import pytest


@pytest.fixture
def tiny_target():
    """4×4 complex target with simple structure (uniform amplitude)."""
    return jnp.ones((4, 4), dtype=jnp.complex128) / 4.0


@pytest.fixture
def tiny_dataset(tiny_target):
    """Three-image dataset: tiny_target plus two phase-shifted copies."""
    return [tiny_target, tiny_target * jnp.exp(1j * 0.5), tiny_target * jnp.exp(1j * 1.0)]
