# `src/pdft/` modular refactor — design

**Status:** approved, ready for implementation plan
**Date:** 2026-04-26
**Author:** brainstorming session
**Related:** GitHub issue #2 (reliability/cleanup backlog)

## Goal

Reorganize the flat 21-file `src/pdft/` tree into a small set of focused subpackages, eliminate one duplicated bookkeeping block in `training.py`, and lightly modernize the test suite to use standard pytest idioms (fixtures, markers, parametrize). One PR, eight commits, no behavior change beyond the documented DRY fix.

The user is the sole developer. Decisions are biased toward **KISS** (one cohesive thing per file, no over-fragmentation) and **DRY** (collapse genuinely duplicated code), with a **clean break on imports** (no backwards-compat shims).

## Non-goals

- No new features. No new bases, no new optimizers, no new losses.
- No tolerance changes, no logic rewrites in basis/manifold/loss/IO code.
- No new dependencies (no `optax`, no `hypothesis`).
- No JSON schema bump — `io/serialize.py` produces byte-identical output to today's `io_json.py`.
- No upstream-pin update.

## Final structure

```
src/pdft/
├── __init__.py                     # version, __upstream_ref__, slim public re-exports
├── manifolds.py                    # mathematical core (used by bases + optimizers)
├── loss.py                         # mathematical core (public API: L1Norm, MSELoss)
├── profiling.py                    # cross-cutting observability
│
├── bases/
│   ├── __init__.py
│   ├── base.py                     # ← basis.py (AbstractSparseBasis + bases_allclose)
│   ├── circuit/
│   │   ├── __init__.py
│   │   ├── qft.py                  # ← qft.py
│   │   ├── entangled_qft.py        # ← entangled_qft.py
│   │   ├── tebd.py                 # ← tebd.py
│   │   └── mera.py                 # ← mera.py
│   └── block/
│       ├── __init__.py
│       ├── block.py                # ← block_basis.py
│       ├── rich.py                 # ← rich_basis.py
│       └── real_rich.py            # ← real_rich_basis.py
│   # DELETED: dct_basis.py
│
├── circuit/
│   ├── __init__.py
│   ├── builder.py                  # ← _circuit.py
│   └── cache.py                    # ← einsum_cache.py
│
├── optimizers/
│   ├── __init__.py                 # AbstractRiemannianOptimizer union, re-exports
│   ├── core.py                     # _OptimizationState, _common_setup, _batched_project
│   ├── gd.py                       # RiemannianGD + _armijo_step
│   ├── adam.py                     # RiemannianAdam + _init_adam_state + _adam_step
│   └── loop.py                     # optimize() dispatcher
│
├── training/
│   ├── __init__.py                 # TrainingResult dataclass + re-exports
│   ├── schedules.py                # cosine_with_warmup
│   ├── single.py                   # train_basis
│   ├── batched.py                  # train_basis_batched orchestrator (uses eval_loop)
│   ├── adam_step.py                # _build_jit_adam_step (JIT closure factory)
│   └── eval_loop.py                # NEW: evaluate_and_check_early_stop (DRY fix)
│
├── io/
│   ├── __init__.py
│   ├── serialize.py                # ← io_json.py (renamed: avoid `json.py` footgun)
│   └── compression.py              # ← compression.py
│
└── viz/
    ├── __init__.py
    ├── loss.py                     # ← viz.py
    └── circuit.py                  # ← circuit_viz.py
```

Tests mirror this layout:

```
tests/
├── conftest.py                     # x64 setup (existing) + goldens_dir, load_golden fixtures
├── bases/
│   ├── conftest.py                 # common basis fixtures
│   ├── circuit/
│   │   ├── test_qft.py
│   │   ├── test_entangled_qft.py
│   │   ├── test_tebd.py
│   │   └── test_mera.py
│   ├── block/
│   │   ├── test_block.py
│   │   ├── test_rich.py
│   │   └── test_real_rich.py
│   └── test_base.py                # AbstractSparseBasis + bases_allclose
├── circuit/
│   ├── test_builder.py
│   └── test_cache.py
├── optimizers/
│   ├── conftest.py                 # random_unitary_tensors fixture
│   ├── test_gd.py
│   ├── test_adam.py
│   └── test_loop.py
├── training/
│   ├── conftest.py                 # tiny_target, tiny_dataset fixtures
│   ├── test_single.py
│   ├── test_batched.py
│   ├── test_schedules.py
│   ├── test_adam_step.py
│   ├── test_eval_loop.py           # NEW: covers extracted helper
│   └── test_integration.py         # was test_training_integration.py
├── io/
│   ├── conftest.py                 # tmp_basis_path fixture
│   ├── test_serialize.py
│   └── test_compression.py
├── viz/
│   ├── test_loss.py
│   └── test_circuit.py
├── test_manifolds.py               # stays at root (peer of root-level src files)
├── test_loss.py
├── test_profiling.py
├── test_smoke.py
└── parity/
    ├── test_qft.py
    ├── test_loss.py
    ├── test_manifolds.py
    ├── test_training.py
    ├── test_adam.py
    ├── test_long_run.py
    ├── test_io_json.py
    ├── test_compression.py
    ├── test_scale.py
    └── test_new_bases.py
```

Parity tests are grouped under `tests/parity/` (mirror of the existing `test_parity_*.py` flat set) and tagged with `@pytest.mark.parity` so `pytest -m "not parity"` is the fast iteration loop.

## Decisions and rationale

### D1. Clean break on imports (option B from brainstorming)

All imports become subpackage-rooted: `from pdft.bases.circuit import QFTBasis`, `from pdft.training import train_basis`, `from pdft.optimizers import RiemannianGD`. `pdft/__init__.py` is a slim re-export hub for the small public surface — not a 50-line backwards-compat wall.

**Why:** sole developer; in-repo tests/examples/benchmarks are the only consumers; one canonical import path per symbol (DRY); namespace tells you what something is at the import site.

### D2. Delete `DCTBasis` entirely

`dct_basis.py`, `tests/test_dct_basis.py`, the `DCTBasis` export from `__init__.py`, any benchmark/example references — all removed.

**Why:** the user judged it doesn't add value over the other parametric bases.

### D3. `bases/circuit/` and `bases/block/` two-level grouping

Reflects the real conceptual split: `circuit/` bases are FFT/DCT-comparable (full circuit topologies), `block/` bases explore parameter-efficient blocked structure.

**Why:** the user prefers a layout that names the two research lines.

### D4. `_circuit.py` + `einsum_cache.py` → `circuit/` subpackage (NOT `util/`)

`util/` is a known anti-pattern (turns into a dumping ground). The two files are tightly coupled — `circuit/builder.py` builds einsums, `circuit/cache.py` caches their JIT closures. Naming them as a subpackage states their role.

### D5. `viz.py` + `circuit_viz.py` → `viz/` subpackage

Same reason: clear group with one responsibility (presentation).

### D6. `manifolds.py`, `loss.py`, `profiling.py` stay at root

Each is load-bearing and has a different consumer set (`manifolds` → bases + optimizers; `loss` → public + training; `profiling` → cross-cutting). Lumping them under `util/` would hide that they're each a public concept.

### D7. `optimizers/` split: `core.py` / `gd.py` / `adam.py` / `loop.py`

Each strategy's config dataclass lives with its step impl (`RiemannianGD` next to `_armijo_step` in `gd.py`, `RiemannianAdam` next to `_adam_step` in `adam.py`). `core.py` holds shared setup (`_OptimizationState`, `_common_setup`, `_batched_project`). `loop.py` is the `optimize()` dispatcher.

**Why:** "look in `adam.py` for everything Adam." Adding RiemannianMomentum/RMSProp later is a new sibling file + one import in `__init__.py` + one branch in `loop.py`.

### D8. `training/` split: 5 files + DRY fix

```
training/
├── __init__.py        TrainingResult + re-exports
├── schedules.py       cosine_with_warmup
├── single.py          train_basis (~55 lines)
├── batched.py         train_basis_batched orchestrator (~250 lines)
├── adam_step.py       _build_jit_adam_step (~115 lines)
└── eval_loop.py       evaluate_and_check_early_stop (~30 lines, NEW)
```

`TrainingResult` lives in `training/__init__.py` (avoids the silly 10-line `result.py`). `eval_loop.py` extracts the early-stopping bookkeeping currently duplicated at lines 514-533 (Adam path) and lines 584-599 (GD path) of `training.py` — both branches will call one shared helper.

**Why this cut and not finer:** `batched.py` is one cohesive orchestration function; further splitting (e.g. extracting per-epoch loop) would create artificial boundaries. `adam_step.py` is worth its own file because it's a self-contained JIT closure factory with deep Wirtinger/manifold-grouping comments. `schedules.py` is its own file even though small — easy to add MultiStep/ExponentialDecay later.

### D9. DO NOT merge `optimizers/adam.py` with `training/adam_step.py`

They look superficially similar but have different requirements: `optimizers/adam.py` uses Python dicts keyed by manifold (general API); `training/adam_step.py` uses static lists indexed by k (XLA-friendly, no dict lookups inside the JIT'd graph). Each will gain a comment cross-referencing the other.

### D10. `io/serialize.py`, NOT `io/json.py`

A file named `json.py` inside `pdft.io` is a readability footgun (visually shadows stdlib `json` even though Python's absolute imports save you). `io/serialize.py` is unambiguous and accurately names the module's role (serializing bases to/from JSON happens to be the chosen format).

### D11. Test refactor: light, idiomatic pytest

Current state: zero fixtures, zero markers, zero `parametrize`, helpers like `_random_unitary_tensors` and the `GOLDENS = Path(...)` constant duplicated across many files. Light refactor:

- Per-subdir `conftest.py` with shared fixtures (basis factories, `random_unitary_tensors`, `tiny_target`, `tmp_basis_path`)
- Top-level `tests/conftest.py` adds `goldens_dir` + `load_golden(name)` (replaces duplicated `_load`)
- Register `parity` and `slow` markers in `pyproject.toml`; tag `tests/parity/*.py` with `@pytest.mark.parity`; tag `tests/parity/test_long_run.py` and `tests/training/test_integration.py` with `@pytest.mark.slow`
- Use `@pytest.mark.parametrize` to collapse `test_new_bases.py` (currently 17 tests, mostly the same shape across BlockedBasis/RichBasis/RealRichBasis)
- `pytest.approx` for scalar comparisons currently written as `assert abs(a - b) < eps`

**Out of scope:** hypothesis, mocking, new test cases, tolerance changes, unittest-class style.

## Execution plan: one PR, eight commits, dependency-topological order

Each commit is independently green and bisectable. Commits 1–4, 6, 7 are pure file moves + import rewrites (mechanical, same source bytes). Commit 5 contains the only intentional logic restructure (`eval_loop.py` extraction). Commit 8 is the test-idiom pass.

| # | Commit | Scope |
|---|---|---|
| 1 | `circuit/` subpackage | Move `_circuit.py` → `circuit/builder.py`, `einsum_cache.py` → `circuit/cache.py`. Update ~10 import sites. Move `tests/test_einsum_cache.py` → `tests/circuit/test_cache.py`. |
| 2 | `viz/` subpackage | Move `viz.py` → `viz/loss.py`, `circuit_viz.py` → `viz/circuit.py`. Update examples + `tests/test_viz.py`. |
| 3 | `bases/` subpackage + delete DCT | Move 8 basis files into `bases/circuit/` and `bases/block/`; rename `basis.py` → `bases/base.py`. Delete `dct_basis.py`, `tests/test_dct_basis.py`, all DCT references in `__init__.py` / benchmarks. Reorganize basis tests under `tests/bases/`. |
| 4 | `optimizers/` subpackage | Split `optimizers.py` into `core.py` / `gd.py` / `adam.py` / `loop.py`. Delete `optimizers.py`. Reorganize tests under `tests/optimizers/`. |
| 5 | `training/` subpackage + DRY fix | Split `training.py` into 5 files; extract `evaluate_and_check_early_stop` into `eval_loop.py` and call from both Adam and GD paths in `batched.py`. Delete `training.py`. Add `tests/training/test_eval_loop.py`. |
| 6 | `io/` subpackage | Move `io_json.py` → `io/serialize.py`, `compression.py` → `io/compression.py`. Reorganize tests under `tests/io/`. |
| 7 | Final pass | Slim `pdft/__init__.py` to subpackage-rooted public surface; update `examples/`, `benchmarks/`, `CLAUDE.md`. |
| 8 | Test fixtures + markers + parametrize | Move `tests/test_parity_*.py` → `tests/parity/test_*.py` (drop the `parity_` filename prefix; the directory carries that meaning now). Add per-subdir `conftest.py`. Register `parity` and `slow` markers in `pyproject.toml`. Tag every test under `tests/parity/` with `@pytest.mark.parity`; tag `tests/parity/test_long_run.py` and `tests/training/test_integration.py` with `@pytest.mark.slow`. Parametrize cross-basis tests (`test_new_bases.py`). Replace `assert abs(a - b) < eps` patterns with `pytest.approx`. |

### Per-commit verification gate

Every commit must pass:

```bash
ruff check src tests
ruff format --check src tests
pytest -q                              # full suite green
pytest tests/parity/                   # parity tests are load-bearing per CLAUDE.md
                                       # (after commit 8: pytest -m parity)
```

Coverage gate `--cov-fail-under=90` is verified on commit 7 only (commits 5 and 8 may temporarily perturb coverage).

### Pre-flight (before commit 1)

```bash
git checkout main && git pull
pytest -q                              # confirm baseline is green
python -c "import pdft; print(sorted(n for n in dir(pdft) if not n.startswith('_')))" \
    > /tmp/api_before.txt              # public-surface snapshot for diff
git checkout -b refactor/modular-src
```

### Post-refactor verification

```bash
python -c "import pdft; print(sorted(n for n in dir(pdft) if not n.startswith('_')))" \
    > /tmp/api_after.txt
diff /tmp/api_before.txt /tmp/api_after.txt    # expected to shrink (clean break)

pytest --cov=pdft --cov-fail-under=90
pytest -m parity                                # all parity tests still green
pytest -m "not parity" -q                       # fast iteration loop is materially faster
ruff check src tests
ruff format --check src tests

# CI matrix: 3.11 / 3.12 / 3.13 must all be green before merge.
```

## Risk register

| Risk | Mitigation |
|---|---|
| Parity drift (Julia byte-comparison breaks) | Commits 1-4, 6, 7 are pure moves — no source bytes change, only file paths. Commit 5 has one real refactor (`eval_loop.py`); verify byte-exact loss histories on `test_parity_training.py` and `test_parity_adam.py` against the pre-refactor commit by hashing `loss_history` from the seed=0 fixture. |
| Goldens accidentally regenerated | `reference/goldens/` is read-only data; nothing in this refactor regenerates them. |
| Public API drift | Pre/post snapshot of `dir(pdft)` (see pre-flight). Per (B), the diff IS expected to shrink — the snapshot tells us exactly what changed so we can confirm it's intentional. |
| `io/` package shadows stdlib `io` | Mitigated two ways: (a) inside `pdft.io.*` we use absolute imports, so `import io` always resolves to stdlib; (b) the file inside is `serialize.py`, not `json.py`, so the human-readability footgun is also avoided. |
| Coverage gate dips during transition | Gate verified on commit 7 only; intermediate commits use `pytest -q` for green-ness. |
| CI matrix divergence (3.11 / 3.12 / 3.13) | Push after each commit; if any Python version fails, fix before the next commit. |
| Bisectability lost to a squash merge | Use a merge commit, NOT squash, to preserve the 8-commit history for `git bisect`. |

## Open questions / things to confirm during implementation

- `basis.py` may contain more than just `AbstractSparseBasis` — confirm before commit 3 whether anything else needs to split out separately.
- Test-file collapses for `test_new_bases.py` parametrize need a quick read of the existing 17 tests to confirm they share enough structure.
- Benchmarks may import from `pdft.basis` etc.; commit 7 must update those even if not in `src/` or `tests/`.

## What this refactor does NOT do

- Does not split `manifolds.py` (352 lines) — it's cohesive (UnitaryManifold, PhaseManifold, batched ops).
- Does not split `loss.py` (~160 lines) — same reason.
- Does not introduce a `data/` subpackage in `training/` for the train/val split + padding logic — it's used by exactly one function (`train_basis_batched`); extracting it would be premature.
- Does not introduce `steps/` subdir in `training/` — only one step strategy is gnarly enough to warrant its own file (`adam_step.py`); the GD branch's per-batch closure stays inline in `batched.py`.
- Does not change the JSON schema, the upstream Julia pin, or the goldens.
- Does not add `optax`, `hypothesis`, or any new runtime dependency.
