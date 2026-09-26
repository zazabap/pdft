.PHONY: docs docs-clean goldens test

test:
	pytest

goldens:
	cd reference/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
	julia --project=reference/julia reference/julia/generate_goldens.jl

# Generated inputs (autosummary stubs, executed gallery) are removed first so a
# local -W build sees what CI's clean checkout sees: no stale stubs for
# renamed entries, no cached figures for changed examples.
docs-clean:
	rm -rf docs/_build docs/api/generated docs/auto_examples docs/sg_execution_times.rst

docs: docs-clean
	sphinx-build -W --keep-going -b html -d docs/_build/doctrees docs docs/_build/html
