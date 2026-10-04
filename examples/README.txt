Example gallery
===============

Short, runnable scripts: three train bases on small random targets for
compression, one fills in an image from a tenth of its pixels and trains the
transform through that solver. Each finishes in a few seconds on a CPU; the
plots below are produced by running them while the docs build. The two
papers behind the package are cited on the pages of the examples that
demonstrate them.

Run any of them locally from the repository root, e.g.
``python examples/basis_demo.py``. They need the ``plot`` extra
(``pip install "pdft[plot]"``) and write their figures to ``out/``.
