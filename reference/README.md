# Reference Julia harness

This directory produces the `.npz` golden vectors under `reference/goldens/`
that Python parity tests assert against. You only need Julia installed if
you are **regenerating** goldens; normal Python dev + CI consumes the
committed `.npz` files.

## Regeneration

```bash
make goldens
```

equivalent to:

```bash
cd reference/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. generate_goldens.jl
```

The first run resolves `ParametricDFT.jl` at the commit pinned in
`generate_goldens.jl` (variable `UPSTREAM_SHA`). To change the pin, edit
that constant and delete `Manifest.toml` so it is regenerated.

After regeneration, update `pdft.__upstream_ref__` in
`src/pdft/__init__.py` to match `manifest.json["upstream_sha"]`.

## Completion

Image completion is not in ParametricDFT.jl. Its reference is the code of the
completion paper, and `completion/golden.npz` is a run of it:
`completion/generate_golden.py`, executed in that repository (the commit is
recorded in the file as `paper_commit`). `tests/parity/test_completion.py`
asserts against it. The script's docstring says how to run it and why its
settings are what they are.
