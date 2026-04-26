"""Basis-test fixtures: pre-built tiny bases."""

import pytest

from pdft.bases.base import QFTBasis


@pytest.fixture
def qft_2x2():
    """Default-initialised 2x2 QFT basis (deterministic; no seed argument)."""
    return QFTBasis(2, 2)
