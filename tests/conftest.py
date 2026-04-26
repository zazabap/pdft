"""Pytest session setup: enable JAX x64 + shared fixtures for all tests.

Without x64, JAX operates in float32/complex64 and parity tolerances are
impossible to hit. Importing pdft does this too (see src/pdft/__init__.py),
but we set it here as well so property tests that use `jax.numpy` directly
(without importing pdft) also run in x64.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

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
