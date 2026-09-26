"""The hand-written API pages in docs/api/ must list every public name of
their module, so a new export cannot silently stay off the site."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

DOCS_API = Path(__file__).resolve().parents[1] / "docs" / "api"
PAGES = sorted(p for p in DOCS_API.glob("*.rst") if p.name != "index.rst")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_api_page_lists_every_public_name(page: Path) -> None:
    text = page.read_text()
    module = re.search(r"^\.\. currentmodule:: (\S+)$", text, re.M).group(1)
    public = getattr(importlib.import_module(module), "__all__", None)
    if public is None:
        pytest.skip(f"{module} defines no __all__")
    block = text.split(".. autosummary::", 1)[1]
    listed = {
        ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith(":")
    }
    missing = sorted(set(public) - listed)
    assert not missing, f"{page.name} omits {missing}"
