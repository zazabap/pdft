"""Test bootstrap: add benchmarks/ to sys.path so sibling modules import."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure x64 mode is on (mirrors tests/conftest.py for pdft itself).
import pdft  # noqa: F401  -- imported for side-effect: jax_enable_x64

_BENCH_DIR = Path(__file__).resolve().parent.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
