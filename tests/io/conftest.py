"""IO-test fixtures: tmp-path helpers."""

import pytest


@pytest.fixture
def tmp_basis_path(tmp_path):
    """Path inside the per-test tmp_path with a `.json` extension for save_basis."""
    return tmp_path / "basis.json"
