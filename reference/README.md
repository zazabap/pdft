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
