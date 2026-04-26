"""Auto-mark every test in this directory with @pytest.mark.parity."""

from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            item_path = Path(item.fspath).resolve()
        except (AttributeError, ValueError):
            continue
        # Only mark items under this directory.
        try:
            item_path.relative_to(_THIS_DIR)
        except ValueError:
            continue
        item.add_marker(pytest.mark.parity)
