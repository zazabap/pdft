# CLAUDE.md

Project-specific guidance for Claude Code working in this repo. Read this before making non-trivial changes — several conventions are *not* discoverable from the code without painful debugging, and at least one is documented here only because the original Python port spent hours rediscovering it.

## What this is

`pdft` is a faithful Python port of [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl) using JAX. Goal: a Python user can reproduce Julia results bit-for-bit (or within documented tolerances) on the same input. **Behavior parity with Julia is the primary correctness criterion** — never sacrifice it for Pythonic-ness.

- Design spec: `docs/superpowers/specs/2026-04-24-pdft-migration-design.md`
- Roadmap: GitHub issue #1
- Reliability hardening backlog: GitHub issue #2
- Upstream is pinned to `nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef`. The pin lives in two places that must stay in sync: `reference/julia/generate_goldens.jl` (`UPSTREAM_SHA`) and `src/pdft/__init__.py` (`__upstream_ref__`).

## Critical conventions (do not break)

These are the non-obvious invariants that took the original port multiple iterations to find. Violating any of them silently breaks parity with Julia.

### 1. Wirtinger gradient conjugation

**JAX returns `∂f/∂z̄` while Julia's Zygote returns `∂f/∂z`** for real-valued functions of complex inputs. They are complex conjugates. `optimize()` in `src/pdft/optimizers.py` conjugates `raw_grads` immediately after `grad_fn(...)`:

```python
raw_grads = grad_fn(state.current_tensors)
raw_grads = [jnp.conj(g) for g in raw_grads]   # MUST stay
```

Without this line, GD trajectories drift ~10% over 50 steps and Adam outright diverges. Any new optimizer added to this module must apply the same conjugation, or define a custom JAX `vjp` that matches Julia's convention.

### 2. Yao little-endian qubit ordering

`pic.reshape((2,)*(m+n))` gives axes in big-endian bit order, but Julia's Yao treats qubit 1 as the LSB. `_circuit.build_circuit_einsum` reverses qubit-to-axis mapping within each block:

```python
row_pic = [input_labels[q - 1] for q in range(m, 0, -1)]   # axis 0 ↔ qubit m
col_pic = [input_labels[q - 1] for q in range(m + n, m, -1)]
```

Same reversal applies to `out_labels`. Don't undo this without first making `ft_mat` parity tests fail in a controlled way.

### 3. Compact 2×2 CP tensor representation

A controlled-phase gate is `diag(1, 1, 1, exp(iφ))` as a 4×4 matrix, but Yao's `yao2einsum` emits the **compact 2×2 form**:

```python
controlled_phase_diag(φ) = [[1, 1], [1, exp(iφ)]]
```

This shares the wire labels of control and target qubits and does NOT introduce new labels (the gate is diagonal so each wire's value is unchanged; only a multiplicative phase is injected). All circuit modules use this representation; never substitute the 4×4 form even though it would be mathematically equivalent — Julia goldens won't match.

### 4. Hadamard-first tensor sort

Julia's `qft_code` does `perm_vec = sortperm(tn.tensors, by=x -> !(x ≈ mat(H)))` — Hadamards come first, CPs after. Python's `_circuit.build_circuit_einsum` applies the same sort. The einsum is rebuilt with the permuted tensor list AND permuted subscripts so it stays valid. **JSON serialization assumes this order**; cross-language interop breaks if you change it.

### 5. JAX x64 mode at import time

`src/pdft/__init__.py` calls `jax.config.update("jax_enable_x64", True)` before any other import. `tests/conftest.py` does the same. Without x64, JAX uses complex64 / float32 and parity tolerances become unreachable. Any new test file that uses `jax.numpy` directly without importing `pdft` first must include the same call (or import pdft to force it).

### 6. Column-major iteration for cross-language hashes

`io_json.basis_hash` and `compression` use **Julia 1-based column-major** flat indices. NumPy is row-major by default. Use `array.flatten(order="F")` or `np.unravel_index(..., order="F")`. Never use plain `array.flatten()` for serialization.

### 7. Julia-compatible float formatting

Python's `repr(5e-7)` gives `"5e-07"`; Julia's `string(5e-7)` gives `"5.0e-7"`. `io_json._format_float_julia_like` post-processes `repr()` to match Julia. Use it (not `repr` or `str`) for any value that participates in a cross-language hash or JSON byte-comparison.

## Repo layout

```
src/pdft/
├── _circuit.py        Shared circuit-to-einsum builder (used by all 4 bases)
├── einsum_cache.py    jnp.einsum_path + jax.jit closure cache
├── loss.py            L1Norm, MSELoss, topk_truncate, loss_function
├── manifolds.py       UnitaryManifold, PhaseManifold, batched (d,d,n) ops
├── qft.py             QFTBasis circuit
├── entangled_qft.py   EntangledQFT (only :back position; :front/:middle are open work in #2)
├── tebd.py            TEBD with row+col rings
├── mera.py            MERA hierarchical (powers of 2 only)
├── basis.py           AbstractSparseBasis + 4 concrete bases (all JAX pytrees)
├── optimizers.py      RiemannianGD (Armijo) + RiemannianAdam
├── training.py        train_basis (single-target; pytree-generic)
├── io_json.py         save_basis / load_basis / basis_hash
├── compression.py     CompressedImage + compress / recover
├── viz.py             matplotlib loss plots (plot extra)
└── circuit_viz.py     matplotlib circuit schematic (plot extra)

reference/julia/       Julia harness — needed only to regenerate goldens
reference/goldens/     Committed .npz + .json files (<200 KB total)
examples/              3 runnable demos, each <10s
tests/                 pytest; one file per src/ module + parity tests
```

`AbstractRiemannianOptimizer` is `RiemannianGD | RiemannianAdam` — a structural union, not a Protocol. Adding a third optimizer means extending that union *and* adding an `isinstance` branch in `optimize()`.

`train_basis` is generic over basis type via JAX pytree flatten/unflatten. The convention: each registered basis pytree must have leaves ordered as `tuple(tensors) + tuple(inv_tensors)`, with everything else (m, n, code, inv_code, counts) in aux data. New basis types must follow this convention.

## Dev workflow

```bash
# Install (Python 3.11+; conda env at /opt/conda/envs/pdft on the dev box)
pip install -e ".[dev]"

# Run tests
pytest                                    # full suite
pytest --cov=pdft --cov-fail-under=90     # CI gate
pytest tests/test_parity_*.py             # parity-only

# Lint (CI fails if this is dirty — check before pushing)
ruff check src tests
ruff format src tests

# Run examples
python examples/basis_demo.py
python examples/optimizer_benchmark.py
python examples/mera_demo.py

# Regenerate Julia goldens (requires Julia 1.10+)
make goldens
```

After regenerating goldens: also update `__upstream_ref__` in `src/pdft/__init__.py` to match `manifest.json`'s `upstream_sha`.

## Tests and parity

Three layers (per spec section 7):

1. **Parity tests** (`tests/test_parity_*.py`) — load committed `.npz` / `.json` goldens from `reference/goldens/` and assert Python matches Julia. These are the load-bearing correctness tests.
2. **Property tests** (`tests/test_<module>.py`) — math-invariant checks (unitarity preserved, round-trip identity, monotone descent, …). Don't depend on Julia.
3. **Smoke / integration** (`tests/test_smoke.py`, `tests/test_training_integration.py`).

Coverage gate is `--cov-fail-under=90`. Don't add tests that reduce per-module coverage below the line; if a new module legitimately needs more code, also add the property tests for it.

`tests/conftest.py` enables x64. Don't import jax in a test before pdft unless the test also calls `jax.config.update("jax_enable_x64", True)`.

## When to break parity

The default answer is "never". The only acceptable cases:

- **Float-precision noise**: test tolerances are documented per case; if you cross a tolerance boundary, investigate before relaxing.
- **Tie-breaking divergences** (e.g. argpartition vs partialsortperm): document and assert downstream behavior matches instead.

If you find another mismatch:
1. Diagnose with `reference/julia/` (write a small Julia script to dump intermediate values).
2. Compare to Python at the same intermediate point.
3. **The bug is in Python**, almost always.
4. If the bug is in upstream Julia, file an issue there and document the divergence in this file.

## CI gotchas

- Ruff is strict (`F401`, `F841`, `E741`, `E731` all errors). Auto-`--fix` handles most.
- Coverage gate runs only on the actual test suite, not examples; don't rely on examples for coverage.
- The `verify-upstream-pin.yml` workflow runs only on PRs touching `reference/`. It uses the GitHub API to confirm the pinned sha exists in upstream — don't push a sha that's only in a fork.
- The matrix is 3.11 / 3.12 / 3.13. JAX dropped 3.10 in 0.10.0; do not lower the floor.

## What NOT to do

- **Don't add `optax`** as a dependency. The optimizer logic is hand-rolled to match Julia's exact moment-update math. `optax`'s defaults and FP order will diverge from Julia.
- **Don't switch from `jnp.einsum_path("greedy")` to `"optimal"`.** "Optimal" is exponential in tensor count and hangs on the 3×3 QFT (12 tensors).
- **Don't add explicit JIT to `train_basis`.** It calls a basis-typed loss closure with Python-list pytrees; JIT decisions are best left to inner functions where the static-vs-leaf split is clearer.
- **Don't introduce backwards-compat shims** for the JSON schema. We're at v0.1.0; if the schema changes, bump the version and regenerate goldens.
- **Don't add ML scaffolding** (no DataLoader, no Trainer-like classes, no Lightning). Upstream is one-target-image-at-a-time and we mirror that. Batched training is open work in #2.
- **Don't run examples in CI.** They write to `out/` (gitignored) and are not coverage-relevant.

## When making changes

1. Make the change.
2. Run `ruff check src tests` and `ruff format src tests`.
3. Run `pytest --cov=pdft --cov-fail-under=90`.
4. If you touched a circuit, basis, optimizer, or io_json module: also run the relevant `tests/test_parity_*.py` to confirm Julia-parity hasn't drifted.
5. If you changed the upstream pin: update both `reference/julia/generate_goldens.jl` (`UPSTREAM_SHA`) and `src/pdft/__init__.py` (`__upstream_ref__`), regenerate goldens, and verify all parity tests still pass.
6. Push and watch the GitHub Actions matrix — three Python versions must all be green before merging.
