"""sys.path bootstrap so sibling modules in benchmarks/ are importable.

This package intentionally has no __init__.py — each run_*.py script is
standalone-runnable. This file is imported first to make `import config`,
`import baselines`, etc. work from any entrypoint inside benchmarks/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

BENCH_ROOT = _BENCH_DIR
REPO_ROOT = _BENCH_DIR.parent
