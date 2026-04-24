.PHONY: goldens test

test:
	pytest

goldens:
	cd reference/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
	julia --project=reference/julia reference/julia/generate_goldens.jl
