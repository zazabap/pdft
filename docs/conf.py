"""Sphinx configuration for the pdft documentation site."""

from __future__ import annotations

import os

# Keep doc builds from writing into the user's JAX compile cache.
os.environ.setdefault("PDFT_DISABLE_COMPILE_CACHE", "1")

import pdft

project = "pdft"
author = "zazabap"
copyright = "2026, zazabap"
release = pdft.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- API reference -----------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

myst_enable_extensions = ["colon_fence", "dollarmath"]

# -- Example gallery ---------------------------------------------------------
# Each example runs at build time (a few seconds apiece) so the rendered
# pages carry real loss curves. The scripts write their PNGs to examples/out/,
# which is gitignored.
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r".*\.py$",
    "download_all_examples": False,
    "remove_config_comments": True,
    "reference_url": {"pdft": None},
}

# -- HTML --------------------------------------------------------------------
# sphinx-book-theme, the theme JAX's documentation uses.
html_theme = "sphinx_book_theme"
html_title = f"pdft {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/zazabap/pdft",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "show_toc_level": 2,
    "home_page_in_toc": True,
    "navigation_with_keys": False,
}
