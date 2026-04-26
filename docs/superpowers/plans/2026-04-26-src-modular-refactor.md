# `src/pdft/` Modular Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the flat 21-file `src/pdft/` tree into focused subpackages (`bases/`, `optimizers/`, `training/`, `io/`, `circuit/`, `viz/`), delete `DCTBasis`, fix one duplicated bookkeeping block in `training.py`, and lightly modernize the test suite with shared fixtures and `parity`/`slow` markers.

**Architecture:** One PR, eight commits in dependency-topological order. Commits 1–4, 6, 7 are pure file moves + import rewrites (zero source-byte changes). Commit 5 contains the only intentional logic restructure (extract `evaluate_and_check_early_stop` into `training/eval_loop.py` and call it from both Adam and GD branches in `batched.py`). Commit 8 is the test-idiom pass.

**Tech Stack:** Python 3.11+, JAX (x64), pytest, ruff. Target branch already exists: `refactor/modular-src` (the design-doc commit `67c700e` is on it).

**Spec:** `docs/superpowers/specs/2026-04-26-src-modular-refactor-design.md`

---

## Pre-flight

Run once before starting Task 1.

- [ ] **Step P1: Confirm baseline**

```bash
cd /home/claude-user/parametric-dft-python
git status                                       # must be clean
git branch --show-current                        # must print: refactor/modular-src
pytest -q                                        # full suite must be green on this branch
```

Expected: branch is `refactor/modular-src`, working tree clean, all tests pass.

- [ ] **Step P2: Snapshot the public API surface**

```bash
python -c "import pdft; print(sorted(n for n in dir(pdft) if not n.startswith('_')))" \
    > /tmp/api_before.txt
wc -l /tmp/api_before.txt                        # one line; ~50 names
```

This snapshot is diffed at the end (see Task 7) to confirm only intentional public-surface changes.

- [ ] **Step P3: Save the parity loss-history hash for byte-exact verification of commit 5**

```bash
python -c "
from pdft.basis import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianGD
from pdft.training import train_basis
import jax.numpy as jnp, hashlib

basis = QFTBasis(2, 2, seed=0)
target = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0
res = train_basis(basis, target=target, loss=L1Norm(), optimizer=RiemannianGD(lr=0.01), steps=20, seed=0)
print(hashlib.sha256(repr([round(x, 12) for x in res.loss_history]).encode()).hexdigest())
" > /tmp/loss_history_hash_before.txt
cat /tmp/loss_history_hash_before.txt
```

Save this hash. After commit 5, the same script (with rewritten imports) must produce the same hash to prove the `eval_loop.py` extraction did not perturb the trajectory.

---

## Task 1: `circuit/` subpackage (move `_circuit.py` + `einsum_cache.py`)

**Files:**
- Create: `src/pdft/circuit/__init__.py`, `src/pdft/circuit/builder.py`, `src/pdft/circuit/cache.py`
- Delete: `src/pdft/_circuit.py`, `src/pdft/einsum_cache.py`
- Modify imports in: `src/pdft/qft.py`, `src/pdft/entangled_qft.py`, `src/pdft/tebd.py`, `src/pdft/mera.py`, `src/pdft/rich_basis.py`, `src/pdft/real_rich_basis.py`, `src/pdft/block_basis.py`, `src/pdft/dct_basis.py` (still exists in this commit; deleted in Task 3), `tests/test_einsum_cache.py`
- Test: move `tests/test_einsum_cache.py` → `tests/circuit/test_cache.py`. Add `tests/circuit/__init__.py` (empty).

- [ ] **Step 1.1: Create the `circuit/` subpackage with `git mv`**

```bash
cd /home/claude-user/parametric-dft-python
mkdir -p src/pdft/circuit tests/circuit
git mv src/pdft/_circuit.py     src/pdft/circuit/builder.py
git mv src/pdft/einsum_cache.py src/pdft/circuit/cache.py
git mv tests/test_einsum_cache.py tests/circuit/test_cache.py
touch src/pdft/circuit/__init__.py tests/circuit/__init__.py
git add src/pdft/circuit/__init__.py tests/circuit/__init__.py
```

- [ ] **Step 1.2: Populate `src/pdft/circuit/__init__.py`**

Write this exact content:

```python
"""Circuit machinery: einsum builder + JIT closure cache.

Shared by every basis. The builder converts a Yao-style gate list into a
JAX einsum (Hadamard-first sort + Yao little-endian ordering preserved).
The cache memoizes `jnp.einsum_path` results and the JIT'd closures.
"""

from .builder import (
    HADAMARD,
    Gate,
    build_circuit_einsum,
    compile_circuit,
    u4_from_phase,
)
from .cache import optimize_code_cached

__all__ = [
    "HADAMARD",
    "Gate",
    "build_circuit_einsum",
    "compile_circuit",
    "optimize_code_cached",
    "u4_from_phase",
]
```

Note: the exported names match what consumers currently import from `pdft._circuit` and `pdft.einsum_cache`. Verify by running the grep in step 1.4 — if a basis imports a name not listed here, add it to `__all__`.

- [ ] **Step 1.3: Rewrite all `from .` imports in `src/pdft/`**

Run these `sed` commands (verify the diff before proceeding):

```bash
cd /home/claude-user/parametric-dft-python
for f in src/pdft/qft.py src/pdft/entangled_qft.py src/pdft/tebd.py src/pdft/mera.py \
         src/pdft/rich_basis.py src/pdft/real_rich_basis.py \
         src/pdft/block_basis.py src/pdft/dct_basis.py; do
    sed -i 's|^from \._circuit import|from .circuit.builder import|' "$f"
    sed -i 's|^from \.einsum_cache import|from .circuit.cache import|' "$f"
done
git diff src/pdft/ | head -80
```

Expected: each touched file shows one or two import-line changes, no other changes.

- [ ] **Step 1.4: Rewrite test imports**

```bash
sed -i 's|^from pdft\.einsum_cache import|from pdft.circuit.cache import|' \
    tests/circuit/test_cache.py
git diff tests/circuit/test_cache.py
```

Verify there are no other test files importing from `pdft._circuit` or `pdft.einsum_cache`:

```bash
grep -rn "pdft\._circuit\|pdft\.einsum_cache" tests/ examples/ benchmarks/ 2>/dev/null
```

Expected output: empty (zero matches).

- [ ] **Step 1.5: Run tests + lint**

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

Expected: ruff clean, all tests green.

- [ ] **Step 1.6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: extract circuit/ subpackage (_circuit.py + einsum_cache.py)

- src/pdft/_circuit.py     -> src/pdft/circuit/builder.py
- src/pdft/einsum_cache.py -> src/pdft/circuit/cache.py
- tests/test_einsum_cache.py -> tests/circuit/test_cache.py

All basis files updated to import from .circuit.builder / .circuit.cache.
Pure file moves + import rewrites. No source-byte changes inside the
moved files; no logic change.

Refs: docs/superpowers/specs/2026-04-26-src-modular-refactor-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `viz/` subpackage (move `viz.py` + `circuit_viz.py`)

**Files:**
- Create: `src/pdft/viz/__init__.py`, `src/pdft/viz/loss.py`, `src/pdft/viz/circuit.py`
- Delete: `src/pdft/viz.py`, `src/pdft/circuit_viz.py`
- Modify imports in: `examples/basis_demo.py`, `examples/mera_demo.py`, `examples/optimizer_benchmark.py`, `tests/test_viz.py`
- Test: move `tests/test_viz.py` → `tests/viz/test_loss.py` (and rename inner test fns if needed). Add `tests/viz/__init__.py`.

- [ ] **Step 2.1: Move files**

```bash
cd /home/claude-user/parametric-dft-python
mkdir -p src/pdft/viz tests/viz
git mv src/pdft/viz.py         src/pdft/viz/loss.py
git mv src/pdft/circuit_viz.py src/pdft/viz/circuit.py
git mv tests/test_viz.py       tests/viz/test_loss.py
touch src/pdft/viz/__init__.py tests/viz/__init__.py
git add src/pdft/viz/__init__.py tests/viz/__init__.py
```

- [ ] **Step 2.2: Determine the public symbols of each former file**

```bash
grep -E "^(def|class|[A-Z_]+\s*=)" src/pdft/viz/loss.py | head -20
grep -E "^(def|class|[A-Z_]+\s*=)" src/pdft/viz/circuit.py | head -20
```

This shows the names to re-export. From the existing example imports we already know `viz` exports `TrainingHistory`, `plot_training_loss`, `plot_training_comparison`. Confirm via grep, then list ALL public names in `__init__.py`.

- [ ] **Step 2.3: Populate `src/pdft/viz/__init__.py`**

Write this content (adjust if step 2.2 finds extra public names):

```python
"""Matplotlib helpers: training-loss plots and circuit schematics.

Optional `plot` extra. Imported by examples and benchmarks; not depended
on by core. `viz.loss` plots loss histories; `viz.circuit` draws the
einsum schematic for a basis.
"""

from .loss import (
    TrainingHistory,
    plot_training_comparison,
    plot_training_loss,
)
from .circuit import draw_circuit  # adjust to actual public symbol(s) of circuit_viz.py

__all__ = [
    "TrainingHistory",
    "draw_circuit",
    "plot_training_comparison",
    "plot_training_loss",
]
```

Cross-check: any name that currently exists at module-top in `viz.py`/`circuit_viz.py` and is used by examples must be in `__all__`. Verify with the grep in step 2.6.

- [ ] **Step 2.4: Rewrite example imports**

```bash
cd /home/claude-user/parametric-dft-python
sed -i 's|^from pdft\.viz import|from pdft.viz import|' \
    examples/basis_demo.py examples/mera_demo.py examples/optimizer_benchmark.py
```

(No-op sed if the import path is unchanged at the package level — `from pdft.viz import TrainingHistory` still works because `viz/__init__.py` re-exports it. Verify with the next step.)

- [ ] **Step 2.5: Verify no broken imports**

```bash
grep -rn "pdft\.viz\b\|pdft\.circuit_viz\b" src tests examples benchmarks 2>/dev/null
```

Expected: only valid `from pdft.viz import ...` lines (which now resolve through the new package). Any reference to `pdft.circuit_viz` must be rewritten to `pdft.viz.circuit` (or to the re-export in `pdft.viz`).

- [ ] **Step 2.6: Run tests + lint**

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

Expected: clean.

- [ ] **Step 2.7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: extract viz/ subpackage (viz.py + circuit_viz.py)

- src/pdft/viz.py         -> src/pdft/viz/loss.py
- src/pdft/circuit_viz.py -> src/pdft/viz/circuit.py
- tests/test_viz.py       -> tests/viz/test_loss.py

Examples and benchmarks updated to use the new paths via re-exports
in src/pdft/viz/__init__.py. No logic change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `bases/` subpackage + DELETE `DCTBasis`

This is the largest commit. Eight basis files reorganize into `bases/circuit/` (4 files) and `bases/block/` (3 files), `basis.py` becomes `bases/base.py`, and `dct_basis.py` is deleted entirely.

**Files:**
- Create: `src/pdft/bases/__init__.py`, `src/pdft/bases/base.py` (renamed from `basis.py`), `src/pdft/bases/circuit/__init__.py`, `src/pdft/bases/block/__init__.py`
- Move: `qft.py` → `bases/circuit/qft.py`; `entangled_qft.py` → `bases/circuit/entangled_qft.py`; `tebd.py` → `bases/circuit/tebd.py`; `mera.py` → `bases/circuit/mera.py`; `block_basis.py` → `bases/block/block.py`; `rich_basis.py` → `bases/block/rich.py`; `real_rich_basis.py` → `bases/block/real_rich.py`
- DELETE: `src/pdft/dct_basis.py`, `tests/test_dct_basis.py`
- Modify all consumers' imports (see step 3.5)
- Reorganize tests under `tests/bases/circuit/` and `tests/bases/block/`

- [ ] **Step 3.1: Create directory skeleton**

```bash
cd /home/claude-user/parametric-dft-python
mkdir -p src/pdft/bases/circuit src/pdft/bases/block
mkdir -p tests/bases/circuit tests/bases/block
touch src/pdft/bases/__init__.py \
      src/pdft/bases/circuit/__init__.py \
      src/pdft/bases/block/__init__.py \
      tests/bases/__init__.py \
      tests/bases/circuit/__init__.py \
      tests/bases/block/__init__.py
git add src/pdft/bases tests/bases/__init__.py tests/bases/circuit/__init__.py tests/bases/block/__init__.py
```

- [ ] **Step 3.2: Move source files with `git mv`**

```bash
git mv src/pdft/basis.py             src/pdft/bases/base.py
git mv src/pdft/qft.py               src/pdft/bases/circuit/qft.py
git mv src/pdft/entangled_qft.py     src/pdft/bases/circuit/entangled_qft.py
git mv src/pdft/tebd.py              src/pdft/bases/circuit/tebd.py
git mv src/pdft/mera.py              src/pdft/bases/circuit/mera.py
git mv src/pdft/block_basis.py       src/pdft/bases/block/block.py
git mv src/pdft/rich_basis.py        src/pdft/bases/block/rich.py
git mv src/pdft/real_rich_basis.py   src/pdft/bases/block/real_rich.py
```

- [ ] **Step 3.3: Move test files with `git mv`**

```bash
git mv tests/test_basis.py            tests/bases/test_base.py
git mv tests/test_qft.py              tests/bases/circuit/test_qft.py
git mv tests/test_block_basis.py      tests/bases/block/test_block.py
git mv tests/test_rich_basis.py       tests/bases/block/test_rich.py
git mv tests/test_real_rich_basis.py  tests/bases/block/test_real_rich.py
git mv tests/test_new_bases.py        tests/bases/test_new_bases.py
git mv tests/test_phase_extraction.py tests/bases/test_phase_extraction.py
```

- [ ] **Step 3.4: DELETE `dct_basis.py` and its test**

```bash
git rm src/pdft/dct_basis.py
git rm tests/test_dct_basis.py
```

Also remove `DCTBasis` references from `__init__.py` (deferred to Task 7) — for now, comment out the import to keep the package importable until Task 7's __init__ rewrite. Edit `src/pdft/__init__.py`:

Change line 51 from:
```python
from .dct_basis import DCTBasis  # noqa: E402
```
to deletion (just remove the line).

Also remove `"DCTBasis",` from `__all__` (line ~104).

- [ ] **Step 3.5: Rewrite intra-`src/` imports**

The `src/pdft/` files that referenced moved modules:

| Old import (`from .X`) | New import |
|---|---|
| `from .basis import ...` | `from .bases.base import ...` |
| `from .qft import ...` | `from .bases.circuit.qft import ...` |
| `from .entangled_qft import ...` | `from .bases.circuit.entangled_qft import ...` |
| `from .tebd import ...` | `from .bases.circuit.tebd import ...` |
| `from .mera import ...` | `from .bases.circuit.mera import ...` |

Sites:
- `src/pdft/bases/base.py` (was `basis.py`): `from .qft import qft_code` → `from .circuit.qft import qft_code` (relative within `bases/`)
- `src/pdft/bases/circuit/entangled_qft.py`: `from .qft import _qft_gates_1d` (still works — sibling) ✓; `from ._circuit import ...` (already updated in Task 1 to `from ..circuit.builder import ...` — but Task 1 set it to `from .circuit.builder` which is now broken because the file moved to `bases/circuit/`. **Fix** with: `sed -i 's|^from \.circuit\.builder import|from ...circuit.builder import|'` for files now under `bases/circuit/` and `bases/block/`).
- `src/pdft/io_json.py`: `from .basis import QFTBasis` → `from .bases.base import QFTBasis`
- `src/pdft/compression.py`: imports `from .io_json` (unaffected here; touched in Task 6). For now ensure it still resolves.
- `src/pdft/__init__.py`: every `from .basis import ...`, `from .qft import ...`, etc. → use `from .bases.base import ...`, `from .bases.circuit.qft import ...`, etc. (Will be rewritten cleanly in Task 7; for now make minimal edits to keep it parseable.)

Run these scripted rewrites:

```bash
# 1. Fix circuit/builder + circuit/cache imports inside files now under bases/circuit/ and bases/block/.
#    Their relative depth is now 2, so '.circuit.builder' becomes '..circuit.builder'.
for f in src/pdft/bases/circuit/*.py src/pdft/bases/block/*.py; do
    sed -i 's|^from \.circuit\.builder import|from ...circuit.builder import|' "$f"
    sed -i 's|^from \.circuit\.cache   import|from ...circuit.cache   import|' "$f"
    sed -i 's|^from \.circuit\.cache import|from ...circuit.cache import|' "$f"
done

# 2. Within bases/, sibling references between base.py and circuit/qft.py:
sed -i 's|^from \.qft import qft_code|from .circuit.qft import qft_code|' \
    src/pdft/bases/base.py

# 3. Top-level src/ files that import from former root-level basis.py / qft.py / etc.
sed -i 's|^from \.basis import |from .bases.base import |'                  src/pdft/io_json.py
sed -i 's|^from \.basis import |from .bases.base import |'                  src/pdft/__init__.py
sed -i 's|^from \.qft import |from .bases.circuit.qft import |'             src/pdft/__init__.py
sed -i 's|^from \.entangled_qft import |from .bases.circuit.entangled_qft import |' src/pdft/__init__.py
sed -i 's|^from \.tebd import |from .bases.circuit.tebd import |'           src/pdft/__init__.py
sed -i 's|^from \.mera import |from .bases.circuit.mera import |'           src/pdft/__init__.py
sed -i 's|^from \.block_basis import |from .bases.block.block import |'     src/pdft/__init__.py
sed -i 's|^from \.rich_basis import |from .bases.block.rich import |'       src/pdft/__init__.py
sed -i 's|^from \.real_rich_basis import |from .bases.block.real_rich import |' src/pdft/__init__.py
```

Verify nothing references deleted modules:

```bash
grep -rn "pdft\.dct_basis\|from .dct_basis\|DCTBasis" src/ tests/ examples/ benchmarks/ 2>/dev/null
```

Expected: zero matches.

- [ ] **Step 3.6: Populate `src/pdft/bases/__init__.py` and the two sub-`__init__.py`**

Write `src/pdft/bases/__init__.py`:

```python
"""Sparse-basis subpackage.

Two families:
- bases.circuit  — full circuit topologies (QFT, EntangledQFT, TEBD, MERA),
                   comparable to FFT/DCT.
- bases.block    — parameter-efficient block-structured bases (Blocked,
                   Rich, RealRich) over arbitrary block partitions.

The abstract base class and bases_allclose helper live in bases.base.
"""

from .base import (
    AbstractSparseBasis,
    EntangledQFTBasis,
    MERABasis,
    QFTBasis,
    TEBDBasis,
    bases_allclose,
)
from .block import BlockedBasis, RealRichBasis, RichBasis, fit_to_dct
from .circuit import entangled_qft_code, ft_mat, ift_mat, mera_code, qft_code, tebd_code

__all__ = [
    "AbstractSparseBasis",
    "BlockedBasis",
    "EntangledQFTBasis",
    "MERABasis",
    "QFTBasis",
    "RealRichBasis",
    "RichBasis",
    "TEBDBasis",
    "bases_allclose",
    "entangled_qft_code",
    "fit_to_dct",
    "ft_mat",
    "ift_mat",
    "mera_code",
    "qft_code",
    "tebd_code",
]
```

Note: the basis class names (QFTBasis, EntangledQFTBasis, etc.) currently live in `bases/base.py` (formerly `basis.py`). Verify with `grep "^class" src/pdft/bases/base.py`. If any actually lives in its own module file (e.g., `bases/circuit/qft.py` defines `QFTBasis`), adjust the import paths accordingly.

Write `src/pdft/bases/circuit/__init__.py`:

```python
"""Circuit-topology bases: QFT, EntangledQFT, TEBD, MERA."""

from .qft import ft_mat, ift_mat, qft_code
from .entangled_qft import entangled_qft_code
from .mera import mera_code
from .tebd import tebd_code

__all__ = [
    "entangled_qft_code",
    "ft_mat",
    "ift_mat",
    "mera_code",
    "qft_code",
    "tebd_code",
]
```

Write `src/pdft/bases/block/__init__.py`:

```python
"""Block-structured bases: Blocked, Rich, RealRich."""

from .block import BlockedBasis
from .rich import RichBasis, fit_to_dct
from .real_rich import RealRichBasis

__all__ = [
    "BlockedBasis",
    "RealRichBasis",
    "RichBasis",
    "fit_to_dct",
]
```

- [ ] **Step 3.7: Rewrite test imports**

```bash
cd /home/claude-user/parametric-dft-python

# Tests under tests/bases/ no longer have a deeper module path; they reference pdft.bases.*
declare -A REWRITE=(
    ["pdft.basis"]="pdft.bases.base"
    ["pdft.qft"]="pdft.bases.circuit.qft"
    ["pdft.entangled_qft"]="pdft.bases.circuit.entangled_qft"
    ["pdft.tebd"]="pdft.bases.circuit.tebd"
    ["pdft.mera"]="pdft.bases.circuit.mera"
    ["pdft.block_basis"]="pdft.bases.block.block"
    ["pdft.rich_basis"]="pdft.bases.block.rich"
    ["pdft.real_rich_basis"]="pdft.bases.block.real_rich"
)

for old in "${!REWRITE[@]}"; do
    new="${REWRITE[$old]}"
    grep -rl "$old" tests/ 2>/dev/null | xargs -r sed -i "s|${old}|${new}|g"
done

# Verify no leftovers
for old in "${!REWRITE[@]}"; do
    grep -rn "${old}\b" tests/ 2>/dev/null
done
```

Expected: empty output (no leftovers).

- [ ] **Step 3.8: Run tests + lint**

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

Expected: ruff clean, all tests green. If a test fails with `ImportError` for `DCTBasis`, that test was specifically for DCT and should already be deleted; otherwise grep for stragglers.

- [ ] **Step 3.9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: extract bases/ subpackage; delete DCTBasis

bases/
├── base.py                       # was basis.py (AbstractSparseBasis + bases_allclose
│                                 #   + QFT/EntangledQFT/TEBD/MERA basis classes)
├── circuit/                      # FFT-comparable circuit topologies
│   ├── qft.py                    # was qft.py
│   ├── entangled_qft.py
│   ├── tebd.py
│   └── mera.py
└── block/                        # parameter-efficient blocked bases
    ├── block.py                  # was block_basis.py
    ├── rich.py                   # was rich_basis.py
    └── real_rich.py              # was real_rich_basis.py

DELETED: dct_basis.py, tests/test_dct_basis.py, DCTBasis from public
API. The user judged DCT did not add value over the other parametric
bases.

Tests reorganized under tests/bases/{,circuit/,block/}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `optimizers/` subpackage (split `optimizers.py`)

Splits the 308-line `optimizers.py` into 4 files: `core.py` (shared infra), `gd.py` (RiemannianGD + Armijo), `adam.py` (RiemannianAdam + step), `loop.py` (`optimize` dispatcher).

**Files:**
- Create: `src/pdft/optimizers/__init__.py`, `src/pdft/optimizers/core.py`, `src/pdft/optimizers/gd.py`, `src/pdft/optimizers/adam.py`, `src/pdft/optimizers/loop.py`
- Delete: `src/pdft/optimizers.py`
- Tests: `tests/test_optimizers.py` → `tests/optimizers/test_gd.py`; `tests/test_optimizers_adam.py` → `tests/optimizers/test_adam.py`. Add `tests/optimizers/__init__.py`. (Parity tests stay in their flat `test_parity_*.py` location for now — moved in Task 8.)

- [ ] **Step 4.1: Create skeleton**

```bash
cd /home/claude-user/parametric-dft-python
mkdir -p src/pdft/optimizers tests/optimizers
touch src/pdft/optimizers/__init__.py tests/optimizers/__init__.py
git add tests/optimizers/__init__.py
```

- [ ] **Step 4.2: Read the source map of `optimizers.py`**

The 308-line `src/pdft/optimizers.py` decomposes as:

| Lines | Block | Goes into |
|---|---|---|
| 1–25 | imports + `Array` alias | distributed: each file gets the imports it needs |
| 28–39 | `RiemannianGD` dataclass | `gd.py` |
| 42–53 | `RiemannianAdam` dataclass | `adam.py` |
| 57 | `AbstractRiemannianOptimizer` union | `__init__.py` (so users can import it from `pdft.optimizers`) |
| 65–92 | `_OptimizationState` + `_common_setup` | `core.py` |
| 100–110 | `_batched_project` | `core.py` |
| 118–132 | `_init_adam_state` | `adam.py` |
| 135–174 | `_adam_step` | `adam.py` |
| 177–214 | `_armijo_step` | `gd.py` |
| 222–308 | `optimize()` dispatcher | `loop.py` |

- [ ] **Step 4.3: Create `src/pdft/optimizers/core.py`**

Copy lines 1–25 (imports — keep only `manifolds` and the `Array` alias), 65–110 (state + setup + project) verbatim from the existing `optimizers.py` into `src/pdft/optimizers/core.py`. The exact content:

```python
"""Shared optimizer infrastructure: state setup + batched projection."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..manifolds import (
    AbstractRiemannianManifold,
    UnitaryManifold,
    _make_identity_batch,
    group_by_manifold,
    stack_tensors,
)

Array = jax.Array


@dataclass
class _OptimizationState:
    manifold_groups: dict[AbstractRiemannianManifold, list[int]]
    point_batches: dict[AbstractRiemannianManifold, Array]
    ibatch_cache: dict[AbstractRiemannianManifold, Array]
    current_tensors: list[Array]


def _common_setup(tensors: list[Array]) -> _OptimizationState:
    """Mirror of upstream src/optimizers.jl:45-78."""
    groups = group_by_manifold(tensors)
    point_batches: dict[AbstractRiemannianManifold, Array] = {}
    ibatch_cache: dict[AbstractRiemannianManifold, Array] = {}
    for manifold, indices in groups.items():
        if not indices:
            continue
        pb = stack_tensors(tensors, indices)
        point_batches[manifold] = pb
        if isinstance(manifold, UnitaryManifold):
            d = pb.shape[0]
            n = len(indices)
            ibatch_cache[manifold] = _make_identity_batch(pb.dtype, d, n)
    return _OptimizationState(
        manifold_groups=groups,
        point_batches=point_batches,
        ibatch_cache=ibatch_cache,
        current_tensors=[jnp.asarray(t) for t in tensors],
    )


def _batched_project(state: _OptimizationState, euclid_grads: list[Array]):
    """Mirror of upstream src/optimizers.jl:129-149."""
    rg_batches: dict[AbstractRiemannianManifold, Array] = {}
    grad_norm_sq = 0.0
    for manifold, indices in state.manifold_groups.items():
        pb = state.point_batches[manifold]
        gb = stack_tensors(euclid_grads, indices)
        rg = manifold.project(pb, gb)
        rg_batches[manifold] = rg
        grad_norm_sq = grad_norm_sq + float(jnp.real(jnp.sum(jnp.conj(rg) * rg)))
    return rg_batches, jnp.sqrt(grad_norm_sq)
```

- [ ] **Step 4.4: Create `src/pdft/optimizers/gd.py`**

Combine the `RiemannianGD` dataclass and `_armijo_step` function:

```python
"""Riemannian gradient descent with Armijo backtracking line search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..manifolds import unstack_tensors
from .core import _OptimizationState

Array = jax.Array


@dataclass(frozen=True)
class RiemannianGD:
    """Riemannian gradient descent with Armijo backtracking line search.

    Mirror of upstream src/optimizers.jl:156-166. Defaults match upstream.
    """

    lr: float = 0.01
    armijo_c: float = 1e-4
    armijo_tau: float = 0.5
    max_ls_steps: int = 10
    max_grad_norm: float | None = None


def _armijo_step(
    opt: RiemannianGD,
    state: _OptimizationState,
    rg_batches: dict,
    loss_fn: Callable,
    grad_norm_sq: float,
    cached_loss: float,
) -> float:
    """Mirror of upstream src/optimizers.jl:231-275.

    Returns the accepted candidate loss, or NaN if line search exhausted.
    Mutates `state.point_batches` and `state.current_tensors` in place.
    """
    current_loss = float(loss_fn(state.current_tensors)) if jnp.isnan(cached_loss) else cached_loss
    alpha = opt.lr
    last_cands: dict = {}

    for _ in range(opt.max_ls_steps):
        for manifold, indices in state.manifold_groups.items():
            pb = state.point_batches[manifold]
            rg = rg_batches[manifold]
            ib = state.ibatch_cache.get(manifold)
            cand = manifold.retract(pb, -rg, alpha, I_batch=ib)
            last_cands[manifold] = cand
            unstack_tensors(cand, indices, into=state.current_tensors)

        candidate_loss = float(loss_fn(state.current_tensors))
        if candidate_loss <= current_loss - opt.armijo_c * alpha * grad_norm_sq:
            for manifold in state.manifold_groups:
                state.point_batches[manifold] = last_cands[manifold]
            return candidate_loss

        alpha *= opt.armijo_tau

    # Line search exhausted — use smallest-step candidate
    for manifold in state.manifold_groups:
        state.point_batches[manifold] = last_cands[manifold]
    return float("nan")
```

- [ ] **Step 4.5: Create `src/pdft/optimizers/adam.py`**

```python
"""Riemannian Adam (Becigneul & Ganea, 2019).

Note: this is the *general-purpose* Adam used by the optimize() dispatcher.
The batched training fast path (training/adam_step.py) uses a different
JIT-friendly representation (static lists indexed by k, not Python dicts
keyed by manifold) for XLA compilation; the duplication is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core import _OptimizationState

Array = jax.Array


@dataclass(frozen=True)
class RiemannianAdam:
    """Riemannian Adam optimizer (Becigneul & Ganea, 2019).

    Mirror of upstream src/optimizers.jl:173-183. Defaults match upstream.
    """

    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float | None = None


def _init_adam_state(state: _OptimizationState):
    """Mirror of upstream src/optimizers.jl:197-216.

    Returns a dict with per-manifold m (first moment, complex) and v
    (second moment, real) buffers, all initialized to zero.
    """
    m_buf: dict = {}
    v_buf: dict = {}
    for manifold, indices in state.manifold_groups.items():
        if not indices:
            continue
        pb = state.point_batches[manifold]
        m_buf[manifold] = jnp.zeros_like(pb)
        v_buf[manifold] = jnp.zeros(pb.shape, dtype=jnp.float64)
    return {"m": m_buf, "v": v_buf}


def _adam_step(
    opt: RiemannianAdam,
    state: _OptimizationState,
    rg_batches: dict,
    iter_1_based: int,
    adam_state: dict,
) -> None:
    """Mirror of upstream src/optimizers.jl:277-319.

    Update m, v, direction buffer; retract along -direction; transport
    m onto the new tangent space. Mutates `state.point_batches` and
    `adam_state` in place.
    """
    beta1, beta2 = opt.beta1, opt.beta2
    bc1 = 1.0 - beta1**iter_1_based
    bc2 = 1.0 - beta2**iter_1_based

    for manifold, _indices in state.manifold_groups.items():
        rg = rg_batches[manifold]
        m_state = adam_state["m"][manifold]
        v_state = adam_state["v"][manifold]

        m_state = beta1 * m_state + (1.0 - beta1) * rg
        v_state = beta2 * v_state + (1.0 - beta2) * jnp.real(jnp.conj(rg) * rg)

        direction = (m_state / bc1) / (jnp.sqrt(v_state / bc2) + opt.eps)

        old_batch = state.point_batches[manifold]
        ib = state.ibatch_cache.get(manifold)
        new_batch = manifold.retract(old_batch, -direction, opt.lr, I_batch=ib)
        m_state = manifold.transport(old_batch, new_batch, m_state)

        adam_state["m"][manifold] = m_state
        adam_state["v"][manifold] = v_state
        state.point_batches[manifold] = new_batch
```

- [ ] **Step 4.6: Create `src/pdft/optimizers/loop.py`**

Copy the `optimize()` function (lines 222–308 of original `optimizers.py`) — verbatim except for moving the imports to the new locations:

```python
"""The optimize() dispatcher: a single loop driving either GD or Adam."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..manifolds import unstack_tensors
from .adam import RiemannianAdam, _adam_step, _init_adam_state
from .core import _batched_project, _common_setup
from .gd import RiemannianGD, _armijo_step

Array = jax.Array


def optimize(
    opt: "RiemannianGD | RiemannianAdam",
    tensors: list[Array],
    loss_fn: Callable[[list[Array]], Array],
    grad_fn: Callable[[list[Array]], list[Array]],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    record_loss: bool = False,
) -> tuple[list[Array], list[float]]:
    """Mirror of upstream src/optimizers.jl:335-412.

    Returns (final_tensors, loss_history). `loss_history` is empty unless
    `record_loss=True`; then it starts with the initial loss and appends
    one entry per iteration.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if opt.lr <= 0:
        raise ValueError(f"opt.lr must be > 0, got {opt.lr}")

    state = _common_setup(tensors)
    trace: list[float] = []
    if record_loss:
        trace.append(float(loss_fn(state.current_tensors)))

    cached_loss = float("nan")
    adam_state = _init_adam_state(state) if isinstance(opt, RiemannianAdam) else None

    for iter_0 in range(max_iter):
        for manifold, indices in state.manifold_groups.items():
            unstack_tensors(state.point_batches[manifold], indices, into=state.current_tensors)

        raw_grads = grad_fn(state.current_tensors)
        # JAX and Julia's Zygote use opposite Wirtinger conventions for gradients
        # of real-valued functions of complex inputs: JAX returns ∂f/∂z̄ while
        # Julia returns ∂f/∂z. These are complex conjugates. To match Julia's
        # trajectory (and produce correct updates w.r.t. the real manifold
        # structure), we conjugate the raw gradient before projection.
        raw_grads = [jnp.conj(g) for g in raw_grads]
        for g in raw_grads:
            if not bool(jnp.all(jnp.isfinite(g))):
                warnings.warn("Non-finite gradient — optimizer stopping.", stacklevel=2)
                return state.current_tensors, trace

        rg_batches, grad_norm = _batched_project(state, raw_grads)
        grad_norm_sq = float(grad_norm) ** 2

        if opt.max_grad_norm is not None and float(grad_norm) > opt.max_grad_norm:
            clip = opt.max_grad_norm / float(grad_norm)
            rg_batches = {m: b * clip for m, b in rg_batches.items()}
            grad_norm_sq = opt.max_grad_norm**2

        if float(grad_norm) < tol:
            break

        if isinstance(opt, RiemannianGD):
            cached_loss = _armijo_step(opt, state, rg_batches, loss_fn, grad_norm_sq, cached_loss)
            if record_loss:
                if jnp.isnan(cached_loss):
                    for manifold, indices in state.manifold_groups.items():
                        unstack_tensors(
                            state.point_batches[manifold],
                            indices,
                            into=state.current_tensors,
                        )
                    trace.append(float(loss_fn(state.current_tensors)))
                else:
                    trace.append(float(cached_loss))
        elif isinstance(opt, RiemannianAdam):
            _adam_step(opt, state, rg_batches, iter_0 + 1, adam_state)
            if record_loss:
                for manifold, indices in state.manifold_groups.items():
                    unstack_tensors(
                        state.point_batches[manifold],
                        indices,
                        into=state.current_tensors,
                    )
                trace.append(float(loss_fn(state.current_tensors)))
        else:
            raise TypeError(f"unsupported optimizer type: {type(opt).__name__}")

    for manifold, indices in state.manifold_groups.items():
        unstack_tensors(state.point_batches[manifold], indices, into=state.current_tensors)

    return state.current_tensors, trace
```

- [ ] **Step 4.7: Create `src/pdft/optimizers/__init__.py`**

```python
"""Riemannian optimizers + the optimize() dispatcher.

Public surface: RiemannianGD, RiemannianAdam, AbstractRiemannianOptimizer
union, and optimize().
"""

from .adam import RiemannianAdam
from .gd import RiemannianGD
from .loop import optimize

# Union of supported optimizer types for dispatch.
AbstractRiemannianOptimizer = RiemannianGD | RiemannianAdam

__all__ = [
    "AbstractRiemannianOptimizer",
    "RiemannianAdam",
    "RiemannianGD",
    "optimize",
]
```

- [ ] **Step 4.8: Delete the old `optimizers.py`**

```bash
git rm src/pdft/optimizers.py
git add src/pdft/optimizers/
```

- [ ] **Step 4.9: Move tests**

```bash
git mv tests/test_optimizers.py      tests/optimizers/test_gd.py
git mv tests/test_optimizers_adam.py tests/optimizers/test_adam.py
```

Test imports already say `from pdft.optimizers import ...` — no rewrite needed because the `pdft.optimizers` package re-exports the same names.

- [ ] **Step 4.10: Run tests + lint**

```bash
ruff check src tests
ruff format --check src tests
pytest -q
pytest tests/test_parity_adam.py tests/test_parity_long_run.py -q   # critical
```

Expected: all green, including parity tests.

- [ ] **Step 4.11: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: split optimizers.py into optimizers/ subpackage

optimizers/
├── core.py    # _OptimizationState, _common_setup, _batched_project
├── gd.py      # RiemannianGD dataclass + _armijo_step
├── adam.py    # RiemannianAdam dataclass + _init_adam_state + _adam_step
└── loop.py    # optimize() dispatcher

Public API unchanged: `from pdft.optimizers import RiemannianGD,
RiemannianAdam, AbstractRiemannianOptimizer, optimize` still works
via __init__.py re-exports. Adam-specific fast path used by batched
training (training/adam_step.py, next commit) is intentionally a
separate implementation — see comment in optimizers/adam.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `training/` subpackage + extract `eval_loop.py` (the only logic refactor)

Splits `training.py` (617 lines) into 5 files. **Extracts the duplicated eval+early-stopping bookkeeping** from both Adam and GD branches in `train_basis_batched` into `eval_loop.py:evaluate_and_check_early_stop()`. This is the only commit with intentional logic restructuring.

Verification gate is stricter: byte-exact loss-history hash (Step P3) must match after the refactor.

**Files:**
- Create: `src/pdft/training/__init__.py`, `src/pdft/training/schedules.py`, `src/pdft/training/single.py`, `src/pdft/training/batched.py`, `src/pdft/training/adam_step.py`, `src/pdft/training/eval_loop.py`
- Delete: `src/pdft/training.py`
- Modify imports in: `src/pdft/profiling.py` (uses `_build_jit_adam_step` and `_cosine_with_warmup` — both private), `src/pdft/__init__.py`
- Tests: move `tests/test_training.py` → `tests/training/test_single.py`, `tests/test_training_batched.py` → `tests/training/test_batched.py`, `tests/test_training_integration.py` → `tests/training/test_integration.py`. Add `tests/training/__init__.py`. Add new `tests/training/test_eval_loop.py` (covers extracted helper).

- [ ] **Step 5.1: Create skeleton**

```bash
cd /home/claude-user/parametric-dft-python
mkdir -p src/pdft/training tests/training
touch tests/training/__init__.py
git add tests/training/__init__.py
```

- [ ] **Step 5.2: Create `src/pdft/training/schedules.py`**

```python
"""Learning-rate schedules for batched training."""

from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    total_steps: int,
    *,
    warmup_frac: float = 0.05,
    lr_peak: float = 0.01,
    lr_final: float = 0.001,
) -> float:
    """Linear warmup followed by cosine decay.

    Mirror of `ParametricDFT.jl/src/training.jl::_cosine_with_warmup`.
    `step` is 0-indexed conceptually but Julia uses 1-indexed; we match
    Julia's behavior: the warmup ramp ends exactly at `step == warmup_steps`
    where `warmup_steps = max(1, round(warmup_frac * total_steps))`.
    """
    warmup_steps = max(1, round(warmup_frac * total_steps))
    if step <= warmup_steps:
        return lr_peak * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_final + 0.5 * (lr_peak - lr_final) * (1 + math.cos(math.pi * progress))
```

Note the rename: `_cosine_with_warmup` → `cosine_with_warmup` (no leading underscore — it's now exported from `training.schedules`).

- [ ] **Step 5.3: Create `src/pdft/training/eval_loop.py` (NEW — the DRY fix)**

This helper consolidates the duplicated eval+early-stopping bookkeeping that today lives at lines 514-533 (Adam path) and 584-599 (GD path) of `training.py`.

```python
"""Validation eval + early-stopping bookkeeping (shared by Adam and GD batched paths)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


def evaluate_and_check_early_stop(
    *,
    epoch: int,
    epochs: int,
    val_every_k_epochs: int,
    val_imgs: list,
    val_loss_fn: Callable[[list[Array]], float],
    current_tensors: list[Array],
    best_tensors: list[Array],
    best_val: float,
    patience: int,
    early_stopping_patience: int,
) -> tuple[list[Array], float, int, bool, float]:
    """Run validation (if scheduled), update best/patience state, decide whether to stop.

    Returns:
        (best_tensors, best_val, patience, stop, val_loss_recorded)

    Behavior identical to the bookkeeping previously inlined in both the
    Adam and GD paths of `train_basis_batched`. The only consolidation is
    that one helper now serves both branches; control flow is unchanged.

    - If validation isn't scheduled this epoch (per `val_every_k_epochs`),
      `val_loss_recorded` is NaN and patience is not advanced.
    - The final epoch is always evaluated so `best_tensors` stays fresh.
    - Without a validation set, `best_tensors` is overwritten on every
      epoch (no patience tracking).
    """
    do_eval = bool(val_imgs) and ((epoch + 1) % val_every_k_epochs == 0 or epoch + 1 == epochs)
    val_loss = val_loss_fn(current_tensors) if do_eval else float("nan")

    stop = False
    if val_imgs and do_eval:
        if val_loss < best_val:
            best_val = val_loss
            best_tensors = [jnp.asarray(t) for t in current_tensors]
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience and epoch > 0:
                stop = True
    elif not val_imgs:
        best_tensors = [jnp.asarray(t) for t in current_tensors]

    return best_tensors, best_val, patience, stop, val_loss
```

- [ ] **Step 5.4: Create `src/pdft/training/adam_step.py`**

Copy `_build_jit_adam_step` (lines 172–283 of original `training.py`) verbatim into `src/pdft/training/adam_step.py`. Update imports:

```python
"""JIT'd fused Adam step for the batched training fast path.

This is intentionally a separate implementation from optimizers/adam.py:
it uses static lists indexed by k (XLA-friendly, no dict lookups inside
the JIT'd graph) instead of Python dicts keyed by manifold. The two
implementations must stay in trajectory-parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..loss import AbstractLoss, loss_function
from ..manifolds import UnitaryManifold, _make_identity_batch, group_by_manifold

Array = jax.Array


def _build_jit_adam_step(
    basis,
    loss: AbstractLoss,
    *,
    beta1: float,
    beta2: float,
    eps: float,
    max_grad_norm: float | None,
):
    # ... copy the body verbatim from original training.py lines 195-283 ...
```

(Engineer: copy the body from the original file unchanged — the function logic is preserved exactly.)

- [ ] **Step 5.5: Create `src/pdft/training/single.py`**

Copy `train_basis` (lines 56–109 of original `training.py`) and its imports:

```python
"""Single-target training loop: Phase 1 API, unchanged from upstream."""

from __future__ import annotations

import time
from typing import Any  # noqa: F401  (kept; used in TrainingResult forward refs)

import jax
from jax import tree_util

from ..loss import AbstractLoss, loss_function
from ..optimizers import AbstractRiemannianOptimizer, optimize
from . import TrainingResult                       # forward import — see __init__.py

Array = jax.Array


def train_basis(
    basis,
    *,
    target: Array,
    loss: AbstractLoss,
    optimizer: AbstractRiemannianOptimizer,
    steps: int,
    seed: int = 0,
    device: str = "cpu",
) -> TrainingResult:
    """Train `basis` to minimize `loss(basis.tensors, target)` over `steps`.

    Works for any basis registered as a JAX pytree whose leaves begin with
    the forward-circuit tensor list followed by the inverse-circuit tensor
    list (current convention for all four bases: QFTBasis, EntangledQFTBasis,
    TEBDBasis, MERABasis).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    m, n = basis.m, basis.n
    code = basis.code
    inv_code = basis.inv_code

    def loss_fn(tensors: list[Array]) -> Array:
        return loss_function(tensors, m, n, code, target, loss, inverse_code=inv_code)

    grad_fn = jax.grad(loss_fn, argnums=0)

    t0 = time.perf_counter()
    final_tensors, history = optimize(
        optimizer,
        list(basis.tensors),
        loss_fn,
        grad_fn,
        max_iter=steps,
        tol=0.0,
        record_loss=True,
    )
    elapsed = time.perf_counter() - t0

    leaves, treedef = tree_util.tree_flatten(basis)
    n_fwd = len(basis.tensors)
    new_leaves = list(final_tensors) + list(leaves[n_fwd:])
    trained = tree_util.tree_unflatten(treedef, new_leaves)

    return TrainingResult(
        basis=trained,
        loss_history=history,
        seed=seed,
        steps=steps,
        wall_time_s=elapsed,
    )
```

- [ ] **Step 5.6: Create `src/pdft/training/batched.py` — uses `eval_loop.py`**

Copy `train_basis_batched` (lines 286–617 of original `training.py`), `_resolve_optimizer`, and `_validate_batched_args`, with these surgical changes:

1. Replace the inline early-stopping bookkeeping in BOTH paths (lines 514-533 Adam, lines 584-599 GD) with one call to `evaluate_and_check_early_stop` from `eval_loop.py`. The replacement looks like:

```python
# OLD (Adam path, ~lines 514-533 of training.py):
do_eval = val_imgs and ((epoch + 1) % val_every_k_epochs == 0 or epoch + 1 == epochs)
val_loss = _val_loss(current_tensors) if do_eval else float("nan")
val_history.append(val_loss)
if val_imgs and do_eval:
    if val_loss < best_val:
        best_val = val_loss
        best_tensors = [jnp.asarray(t) for t in current_tensors]
        patience = 0
    else:
        patience += 1
        if patience >= early_stopping_patience and epoch > 0:
            break
elif not val_imgs:
    best_tensors = [jnp.asarray(t) for t in current_tensors]

# NEW (replaces both Adam and GD paths):
best_tensors, best_val, patience, stop, val_loss = evaluate_and_check_early_stop(
    epoch=epoch,
    epochs=epochs,
    val_every_k_epochs=val_every_k_epochs,
    val_imgs=val_imgs,
    val_loss_fn=_val_loss,
    current_tensors=current_tensors,
    best_tensors=best_tensors,
    best_val=best_val,
    patience=patience,
    early_stopping_patience=early_stopping_patience,
)
val_history.append(val_loss)
if stop:
    break
```

Apply this replacement at BOTH bookkeeping sites (Adam loop and GD loop).

2. Update imports at top of file:

```python
from .eval_loop import evaluate_and_check_early_stop
from .schedules import cosine_with_warmup
from .adam_step import _build_jit_adam_step
```

3. Replace all calls to `_cosine_with_warmup(...)` with `cosine_with_warmup(...)`.

4. Replace `from . import TrainingResult` style imports as appropriate.

The rest of `train_basis_batched` is preserved exactly (validation closure, padding logic, optimizer-resolve branch, etc.).

- [ ] **Step 5.7: Create `src/pdft/training/__init__.py`**

```python
"""Training pipelines.

Two trainers:
- train_basis        — single-target loop (Phase 1, upstream parity).
- train_basis_batched — multi-image / multi-epoch with cosine LR schedule,
                        validation + early stopping, JIT'd Adam fast path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    basis: Any
    loss_history: list[float]
    seed: int
    steps: int
    wall_time_s: float
    val_history: list[float] = field(default_factory=list)
    epochs_completed: int = 0


# Re-exports — placed AFTER TrainingResult so subpackage modules can
# import it from `..` (the package root).
from .batched import train_basis_batched          # noqa: E402
from .schedules import cosine_with_warmup         # noqa: E402
from .single import train_basis                   # noqa: E402

__all__ = [
    "TrainingResult",
    "cosine_with_warmup",
    "train_basis",
    "train_basis_batched",
]
```

- [ ] **Step 5.8: Update `src/pdft/profiling.py` imports**

`profiling.py` currently does:

```python
from .training import _build_jit_adam_step, _cosine_with_warmup
```

Replace with:

```python
from .training.adam_step import _build_jit_adam_step
from .training.schedules import cosine_with_warmup as _cosine_with_warmup
```

(The local alias `_cosine_with_warmup` keeps `profiling.py`'s body unchanged.)

- [ ] **Step 5.9: Delete old `training.py`**

```bash
git rm src/pdft/training.py
git add src/pdft/training/
```

- [ ] **Step 5.10: Update `src/pdft/__init__.py`**

Replace line 92:

```python
from .training import TrainingResult, train_basis, train_basis_batched  # noqa: E402
```

This already works against the new `training/__init__.py`. No edit needed.

- [ ] **Step 5.11: Move existing tests + add new test for `eval_loop.py`**

```bash
cd /home/claude-user/parametric-dft-python
git mv tests/test_training.py             tests/training/test_single.py
git mv tests/test_training_batched.py     tests/training/test_batched.py
git mv tests/test_training_integration.py tests/training/test_integration.py

# Update import for the renamed _cosine_with_warmup → cosine_with_warmup helper
sed -i 's|from pdft\.training import _cosine_with_warmup, train_basis_batched|from pdft.training import cosine_with_warmup, train_basis_batched|' \
    tests/training/test_batched.py
sed -i 's|_cosine_with_warmup(|cosine_with_warmup(|g' tests/training/test_batched.py
```

Add the new test file `tests/training/test_eval_loop.py`:

```python
"""Direct test for the extracted evaluate_and_check_early_stop helper.

The helper itself is a simple state-update function with no JAX or basis
dependencies. Tests cover: do_eval scheduling, no-validation fallback,
patience increment, early-stopping trigger, best_val tracking.
"""

import jax.numpy as jnp
import pytest

from pdft.training.eval_loop import evaluate_and_check_early_stop


def _ts(*vals):
    return [jnp.asarray(v, dtype=jnp.complex128) for v in vals]


def test_no_validation_overwrites_best_each_epoch():
    current = _ts(1.0, 2.0)
    best = _ts(0.0, 0.0)
    new_best, new_val, patience, stop, vl = evaluate_and_check_early_stop(
        epoch=0,
        epochs=5,
        val_every_k_epochs=1,
        val_imgs=[],
        val_loss_fn=lambda _: float("inf"),
        current_tensors=current,
        best_tensors=best,
        best_val=float("inf"),
        patience=0,
        early_stopping_patience=3,
    )
    assert all(jnp.allclose(a, b) for a, b in zip(new_best, current))
    assert not stop
    assert patience == 0
    assert jnp.isnan(vl)


def test_validation_improves_resets_patience():
    new_best, new_val, patience, stop, vl = evaluate_and_check_early_stop(
        epoch=0,
        epochs=5,
        val_every_k_epochs=1,
        val_imgs=[1, 2],
        val_loss_fn=lambda _: 0.5,
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=2,
        early_stopping_patience=3,
    )
    assert new_val == pytest.approx(0.5)
    assert patience == 0
    assert not stop


def test_validation_no_improvement_increments_patience():
    new_best, new_val, patience, stop, vl = evaluate_and_check_early_stop(
        epoch=1,                       # epoch > 0 so early stop CAN trigger
        epochs=5,
        val_every_k_epochs=1,
        val_imgs=[1, 2],
        val_loss_fn=lambda _: 1.5,     # worse than best_val=1.0
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=2,
        early_stopping_patience=3,
    )
    assert new_val == pytest.approx(1.0)            # unchanged
    assert patience == 3
    assert stop                                     # patience reached threshold


def test_skipped_epoch_does_not_advance_patience():
    # val_every_k_epochs=2, epoch=0 → (0+1)%2=1, NOT eval. Skip.
    new_best, new_val, patience, stop, vl = evaluate_and_check_early_stop(
        epoch=0,
        epochs=5,
        val_every_k_epochs=2,
        val_imgs=[1, 2],
        val_loss_fn=lambda _: 99.0,    # would be much worse if evaluated
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=2,
        early_stopping_patience=3,
    )
    assert jnp.isnan(vl)
    assert patience == 2                            # unchanged
    assert not stop


def test_final_epoch_always_evaluated():
    # epoch=4, epochs=5 → (epoch+1)==epochs forces eval even with val_every_k_epochs=10.
    _, val, *_ = evaluate_and_check_early_stop(
        epoch=4,
        epochs=5,
        val_every_k_epochs=10,
        val_imgs=[1],
        val_loss_fn=lambda _: 0.42,
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=0,
        early_stopping_patience=3,
    )
    assert val == pytest.approx(0.42)
```

- [ ] **Step 5.12: Run tests + lint**

```bash
ruff check src tests
ruff format --check src tests
pytest -q
pytest tests/test_parity_training.py tests/test_parity_adam.py -q  # CRITICAL
pytest tests/training/test_eval_loop.py -v                          # new helper tests
```

Expected: all green.

- [ ] **Step 5.13: Verify byte-exact loss-history hash (CRITICAL)**

Run the same script from Step P3 against the refactored code and compare:

```bash
python -c "
from pdft.bases.base import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianGD
from pdft.training import train_basis
import jax.numpy as jnp, hashlib

basis = QFTBasis(2, 2, seed=0)
target = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0
res = train_basis(basis, target=target, loss=L1Norm(), optimizer=RiemannianGD(lr=0.01), steps=20, seed=0)
print(hashlib.sha256(repr([round(x, 12) for x in res.loss_history]).encode()).hexdigest())
" > /tmp/loss_history_hash_after.txt

diff /tmp/loss_history_hash_before.txt /tmp/loss_history_hash_after.txt
```

Expected: `diff` produces no output (hashes match exactly). If they differ, the eval_loop extraction perturbed something — DO NOT proceed; investigate.

- [ ] **Step 5.14: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: split training.py into training/ subpackage + extract eval_loop

training/
├── __init__.py    # TrainingResult dataclass + re-exports
├── schedules.py   # cosine_with_warmup (was _cosine_with_warmup)
├── single.py      # train_basis
├── batched.py     # train_basis_batched orchestrator (uses eval_loop)
├── adam_step.py   # _build_jit_adam_step (JIT closure factory)
└── eval_loop.py   # NEW: evaluate_and_check_early_stop

DRY fix: the early-stopping + validation bookkeeping previously
duplicated in both Adam and GD branches of train_basis_batched
(lines 514-533 vs 584-599 of training.py) is now one helper called
from both paths. Behavior is byte-exact: verified by SHA-256 of
loss_history on the QFT(2,2) seed=0 / 20-step fixture.

The Adam JIT step (training/adam_step.py) intentionally remains
distinct from optimizers/adam.py (different representation for
XLA-friendly compilation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `io/` subpackage

Move `io_json.py` → `io/serialize.py` (rename to avoid `json.py` footgun) and `compression.py` → `io/compression.py`.

**Files:**
- Create: `src/pdft/io/__init__.py`
- Move: `src/pdft/io_json.py` → `src/pdft/io/serialize.py`; `src/pdft/compression.py` → `src/pdft/io/compression.py`
- Update import sites: `tests/test_io_json.py`, `tests/test_compression.py`, `tests/test_parity_io_json.py`, `tests/test_parity_compression.py`, `benchmarks/harness.py`, `src/pdft/io/serialize.py` (its own `from .basis` already updated in Task 3 to `.bases.base`), `src/pdft/__init__.py`
- Tests: move into `tests/io/`. Add `tests/io/__init__.py`.

- [ ] **Step 6.1: Move source + add io/__init__.py**

```bash
cd /home/claude-user/parametric-dft-python
mkdir -p src/pdft/io tests/io
git mv src/pdft/io_json.py     src/pdft/io/serialize.py
git mv src/pdft/compression.py src/pdft/io/compression.py
touch src/pdft/io/__init__.py tests/io/__init__.py
git add src/pdft/io/__init__.py tests/io/__init__.py
```

- [ ] **Step 6.2: Fix internal imports in `io/serialize.py` and `io/compression.py`**

The `from .bases.base` import in `serialize.py` is now relative-depth-2 (it's inside `io/`):

```bash
sed -i 's|^from \.bases\.base import|from ..bases.base import|' src/pdft/io/serialize.py
```

The `from .io_json import basis_hash` in `compression.py` becomes a sibling import:

```bash
sed -i 's|^from \.io_json import|from .serialize import|' src/pdft/io/compression.py
```

- [ ] **Step 6.3: Populate `src/pdft/io/__init__.py`**

```python
"""Serialization (JSON) and lossy compression of trained bases."""

from .serialize import (
    basis_hash,
    basis_to_dict,
    dict_to_basis,
    load_basis,
    save_basis,
)
from .compression import (
    CompressedImage,
    compress,
    compress_with_k,
    compressed_to_dict,
    compression_stats,
    dict_to_compressed,
    load_compressed,
    recover,
    save_compressed,
)

__all__ = [
    "CompressedImage",
    "basis_hash",
    "basis_to_dict",
    "compress",
    "compress_with_k",
    "compressed_to_dict",
    "compression_stats",
    "dict_to_basis",
    "dict_to_compressed",
    "load_basis",
    "load_compressed",
    "recover",
    "save_basis",
    "save_compressed",
]
```

- [ ] **Step 6.4: Move tests**

```bash
git mv tests/test_io_json.py     tests/io/test_serialize.py
git mv tests/test_compression.py tests/io/test_compression.py
```

- [ ] **Step 6.5: Rewrite remaining import sites**

```bash
# Tests, benchmarks, parity tests
grep -rl "pdft\.io_json\|pdft\.compression" tests/ benchmarks/ examples/ 2>/dev/null | \
    xargs -r sed -i \
      -e 's|pdft\.io_json|pdft.io.serialize|g' \
      -e 's|pdft\.compression|pdft.io.compression|g'

# src/pdft/__init__.py
sed -i 's|^from \.io_json import |from .io.serialize import |'   src/pdft/__init__.py
sed -i 's|^from \.compression import |from .io.compression import |' src/pdft/__init__.py

# Verify nothing stale remains
grep -rn "pdft\.io_json\|pdft\.compression\b\|from \.io_json\b\|from \.compression\b" \
    src tests examples benchmarks 2>/dev/null
```

Expected: empty output.

- [ ] **Step 6.6: Run tests + lint**

```bash
ruff check src tests
ruff format --check src tests
pytest -q
pytest tests/test_parity_io_json.py tests/test_parity_compression.py -q  # critical: byte-compare goldens
```

Expected: all green. The parity tests byte-compare against `reference/goldens/`; if they fail, an import wasn't rewritten or `serialize.py` produces different bytes (it shouldn't — same code, new home).

- [ ] **Step 6.7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: extract io/ subpackage (serialize + compression)

- src/pdft/io_json.py     -> src/pdft/io/serialize.py
- src/pdft/compression.py -> src/pdft/io/compression.py
- tests/test_io_json.py     -> tests/io/test_serialize.py
- tests/test_compression.py -> tests/io/test_compression.py

Renamed io_json.py to serialize.py (not json.py) to avoid
shadowing-stdlib readability footgun. JSON byte-output is unchanged
(verified by parity goldens).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final pass — slim `__init__.py`, update examples/benchmarks/CLAUDE.md, public-API diff

Trims `src/pdft/__init__.py` to the post-refactor surface and updates everything outside `src/` and `tests/` to use subpackage-rooted imports.

- [ ] **Step 7.1: Rewrite `src/pdft/__init__.py`**

Replace the entire file with:

```python
"""pdft: Python port of ParametricDFT.jl (faithful JAX reference).

Importing this package enables JAX's x64 mode globally. This is required
to match Julia's ComplexF64 numerical behavior; without it, parity
tolerances are unreachable. See docs/superpowers/specs/2026-04-24-pdft-migration-design.md
Section 2 and 8.1.

Also enables a persistent JAX compilation cache by default, so the
~20-second JIT-compile cost of the Adam step is paid once per basis
shape across all runs. Override the cache dir with JAX_COMPILATION_CACHE_DIR
or disable entirely by setting PDFT_DISABLE_COMPILE_CACHE=1.

Public API: import from subpackages directly, e.g.

    from pdft.bases.circuit import QFTBasis
    from pdft.training import train_basis, train_basis_batched
    from pdft.optimizers import RiemannianGD, RiemannianAdam
    from pdft.loss import L1Norm, MSELoss
    from pdft.io import save_basis, load_basis

The names re-exported at the package root below are kept only for the
small set most commonly used in interactive sessions and notebooks.
"""

import os as _os
from pathlib import Path as _Path

import jax as _jax

_jax.config.update("jax_enable_x64", True)

if not _os.environ.get("PDFT_DISABLE_COMPILE_CACHE"):
    _cache_dir = _os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not _cache_dir:
        _xdg = _os.environ.get("XDG_CACHE_HOME") or str(_Path.home() / ".cache")
        _cache_dir = str(_Path(_xdg) / "pdft" / "jax-compile-cache")
    try:
        _Path(_cache_dir).mkdir(parents=True, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", _cache_dir)
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except OSError:
        pass

__version__ = "0.0.0"
__upstream_ref__ = "nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef"

# Slim public re-export hub: most-used names only. Anything else: import
# from the relevant subpackage directly.
from .bases import (  # noqa: E402
    AbstractSparseBasis,
    BlockedBasis,
    EntangledQFTBasis,
    MERABasis,
    QFTBasis,
    RealRichBasis,
    RichBasis,
    TEBDBasis,
    bases_allclose,
)
from .loss import AbstractLoss, L1Norm, MSELoss, loss_function  # noqa: E402
from .optimizers import RiemannianAdam, RiemannianGD, optimize  # noqa: E402
from .training import TrainingResult, train_basis, train_basis_batched  # noqa: E402

__all__ = [
    "AbstractLoss",
    "AbstractSparseBasis",
    "BlockedBasis",
    "EntangledQFTBasis",
    "L1Norm",
    "MERABasis",
    "MSELoss",
    "QFTBasis",
    "RealRichBasis",
    "RichBasis",
    "RiemannianAdam",
    "RiemannianGD",
    "TEBDBasis",
    "TrainingResult",
    "__upstream_ref__",
    "__version__",
    "bases_allclose",
    "loss_function",
    "optimize",
    "train_basis",
    "train_basis_batched",
]
```

This drops from ~50 names to ~20 — only the most commonly used. `save_basis`, `compress`, `ProfileReport`, `qft_code`, `entangled_qft_code`, etc. are still available via their subpackages (`pdft.io.save_basis`, `pdft.io.compress`, `pdft.profiling.ProfileReport`, `pdft.bases.circuit.qft_code`).

- [ ] **Step 7.2: Update examples**

```bash
cd /home/claude-user/parametric-dft-python
# Examples already use `from pdft.viz import ...` (still works)
# Check if any use `from pdft.basis import ...`, `from pdft.qft import ...`, etc.
grep -rn "from pdft\." examples/

# Apply the rewrite map
declare -A REWRITE=(
    ["from pdft\.basis"]="from pdft.bases.base"
    ["from pdft\.qft"]="from pdft.bases.circuit.qft"
    ["from pdft\.entangled_qft"]="from pdft.bases.circuit.entangled_qft"
    ["from pdft\.tebd"]="from pdft.bases.circuit.tebd"
    ["from pdft\.mera"]="from pdft.bases.circuit.mera"
    ["from pdft\.block_basis"]="from pdft.bases.block.block"
    ["from pdft\.rich_basis"]="from pdft.bases.block.rich"
    ["from pdft\.real_rich_basis"]="from pdft.bases.block.real_rich"
    ["from pdft\.io_json"]="from pdft.io.serialize"
    ["from pdft\.compression"]="from pdft.io.compression"
)
for old in "${!REWRITE[@]}"; do
    new="${REWRITE[$old]}"
    grep -rl "$old" examples/ benchmarks/ 2>/dev/null | xargs -r sed -i "s|${old}|${new}|g"
done
```

- [ ] **Step 7.3: Run examples to confirm they import + execute**

```bash
python examples/basis_demo.py >/dev/null 2>&1 && echo "basis_demo: OK"
python examples/optimizer_benchmark.py >/dev/null 2>&1 && echo "optimizer_benchmark: OK"
python examples/mera_demo.py >/dev/null 2>&1 && echo "mera_demo: OK"
```

Expected: three "OK" lines (each example completes within ~10s per CLAUDE.md).

- [ ] **Step 7.4: Public-API diff**

```bash
python -c "import pdft; print(sorted(n for n in dir(pdft) if not n.startswith('_')))" \
    > /tmp/api_after.txt

diff /tmp/api_before.txt /tmp/api_after.txt
```

Expected: removals only. Specifically, names removed from the root re-export are: `CompressedImage`, `DCTBasis` (deleted entirely), `PhaseManifold`, `ProfileReport`, `UnitaryManifold`, `AbstractRiemannianManifold`, `basis_hash`, `basis_to_dict`, `classify_manifold`, `compress`, `compress_with_k`, `compressed_to_dict`, `compression_stats`, `dict_to_basis`, `dict_to_compressed`, `entangled_qft_code`, `fit_to_dct`, `ft_mat`, `group_by_manifold`, `ift_mat`, `load_basis`, `load_compressed`, `mera_code`, `profile_training`, `qft_code`, `recover`, `save_basis`, `save_compressed`, `tebd_code`, `topk_truncate`. All still importable from their subpackages — confirm a few:

```bash
python -c "from pdft.bases.circuit import qft_code; print('qft_code OK')"
python -c "from pdft.io import save_basis, compress; print('io OK')"
python -c "from pdft.profiling import ProfileReport; print('profiling OK')"
python -c "from pdft.manifolds import UnitaryManifold; print('manifolds OK')"
```

- [ ] **Step 7.5: Update `.claude/CLAUDE.md`**

Replace the "Repo layout" section (lines ~78–95). The new layout description:

```markdown
## Repo layout

```
src/pdft/
├── manifolds.py            Mathematical core (UnitaryManifold, PhaseManifold, batched ops)
├── loss.py                 L1Norm, MSELoss, topk_truncate, loss_function (public)
├── profiling.py            Cross-cutting profiling helpers
├── bases/
│   ├── base.py             AbstractSparseBasis + bases_allclose
│   ├── circuit/            QFT, EntangledQFT, TEBD, MERA (full circuit topologies)
│   └── block/              Blocked, Rich, RealRich (parameter-efficient)
├── circuit/                Einsum builder + JIT closure cache (used by every basis)
├── optimizers/             core, gd (RiemannianGD + Armijo), adam (RiemannianAdam), loop
├── training/               schedules, single (train_basis), batched, adam_step, eval_loop
├── io/                     serialize (JSON), compression
└── viz/                    loss plots, circuit schematic (matplotlib `plot` extra)

reference/julia/            Julia harness — needed only to regenerate goldens
reference/goldens/          Committed .npz + .json files (<200 KB total)
examples/                   3 runnable demos, each <10s
tests/                      pytest; mirrors src/ subpackage structure;
                            parity tests under tests/parity/ tagged @pytest.mark.parity
benchmarks/                 GPU benchmark harness (manual; not in CI)
```

Note: `DCTBasis` was removed in commit 4f... (link in PR). Use a parametric basis (`RichBasis`/`RealRichBasis`) for similar use cases.
```

(Keep all other sections of CLAUDE.md unchanged, including the load-bearing "Critical conventions" list.)

- [ ] **Step 7.6: Coverage gate + full lint**

```bash
ruff check src tests
ruff format --check src tests
pytest --cov=pdft --cov-fail-under=90
```

Expected: all checks pass, coverage ≥90%.

- [ ] **Step 7.7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: slim __init__.py + update examples/benchmarks/CLAUDE.md

- src/pdft/__init__.py: drop from ~50 root re-exports to ~20 (most-
  used names only). Less-public symbols still importable from their
  subpackages — see updated module docstring.
- examples/, benchmarks/: update imports to subpackage-rooted paths.
- .claude/CLAUDE.md: refresh "Repo layout" section.

Public-API diff (vs main): removals only; no new names. All removed
names remain importable from their subpackages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Test fixtures + markers + parametrize (light pytest idiom adoption)

Adds shared fixtures via per-subdir `conftest.py`, registers `parity` and `slow` markers in `pyproject.toml`, moves `tests/test_parity_*.py` → `tests/parity/`, tags every parity test with `@pytest.mark.parity`, and parametrizes the cross-basis tests in `test_new_bases.py`.

**Files:**
- Modify: `pyproject.toml` (markers section), `tests/conftest.py` (extend with goldens fixtures)
- Create: `tests/parity/__init__.py`, `tests/bases/conftest.py`, `tests/optimizers/conftest.py`, `tests/training/conftest.py`, `tests/io/conftest.py`
- Move: `tests/test_parity_*.py` → `tests/parity/test_*.py` (drop `parity_` from filenames; the directory now carries that meaning)

- [ ] **Step 8.1: Move parity tests into `tests/parity/`**

```bash
cd /home/claude-user/parametric-dft-python
mkdir -p tests/parity
touch tests/parity/__init__.py
git add tests/parity/__init__.py

# Drop the test_parity_ prefix; the dir name is the marker now.
git mv tests/test_parity_qft.py         tests/parity/test_qft.py
git mv tests/test_parity_loss.py        tests/parity/test_loss.py
git mv tests/test_parity_manifolds.py   tests/parity/test_manifolds.py
git mv tests/test_parity_training.py    tests/parity/test_training.py
git mv tests/test_parity_adam.py        tests/parity/test_adam.py
git mv tests/test_parity_long_run.py    tests/parity/test_long_run.py
git mv tests/test_parity_scale.py       tests/parity/test_scale.py
git mv tests/test_parity_new_bases.py   tests/parity/test_new_bases.py
git mv tests/io/test_parity_io_json.py  tests/parity/test_serialize.py 2>/dev/null || \
    git mv tests/test_parity_io_json.py     tests/parity/test_serialize.py
git mv tests/io/test_parity_compression.py tests/parity/test_compression.py 2>/dev/null || \
    git mv tests/test_parity_compression.py tests/parity/test_compression.py
```

(Some files may have already moved into `tests/io/` in Task 6 — the `2>/dev/null || ...` handles both layouts.)

- [ ] **Step 8.2: Register markers in `pyproject.toml`**

Replace the existing `markers = [...]` block in `[tool.pytest.ini_options]`:

```toml
markers = [
    "integration: GPU/dataset-dependent benchmark tests (deselect with -m 'not integration')",
    "parity: byte-compare against Julia-generated goldens (slower; deselect with -m 'not parity')",
    "slow: long-running tests (>30s); deselect with -m 'not slow' for fast iteration",
]
```

- [ ] **Step 8.3: Add `tests/parity/conftest.py` to auto-tag every test**

Auto-tagging avoids touching every file. Write `tests/parity/conftest.py`:

```python
"""Auto-mark every test in this directory with @pytest.mark.parity."""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.parity)
```

Tag the two known-slow files explicitly. Add at the top of `tests/parity/test_long_run.py` (after the docstring):

```python
import pytest

pytestmark = pytest.mark.slow
```

And the same at the top of `tests/training/test_integration.py`:

```python
import pytest

pytestmark = pytest.mark.slow
```

- [ ] **Step 8.4: Extend top-level `tests/conftest.py` with shared fixtures**

Replace `tests/conftest.py` with:

```python
"""Pytest session setup: enable JAX x64 + shared fixtures for all tests.

Without x64, JAX operates in float32/complex64 and parity tolerances are
impossible to hit. Importing pdft does this too (see src/pdft/__init__.py),
but we set it here as well so property tests that use `jax.numpy` directly
(without importing pdft) also run in x64.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def goldens_dir() -> Path:
    """Path to the Julia-generated reference goldens (read-only)."""
    return Path(__file__).resolve().parent.parent / "reference" / "goldens"


@pytest.fixture(scope="session")
def load_golden(goldens_dir):
    """Factory: `load_golden("qft_code_4x4.npz")` returns the loaded npz dict."""

    def _load(name: str):
        return np.load(goldens_dir / name)

    return _load
```

- [ ] **Step 8.5: Add `tests/optimizers/conftest.py`**

```python
"""Optimizer-test fixtures: random unitary tensors for property checks."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def random_unitary_tensors():
    """Factory: `random_unitary_tensors(d=4, count=3)` returns a list of d×d unitaries."""

    def _make(d: int, count: int, seed: int = 0):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (count, d, d)) + 1j * jax.random.normal(k2, (count, d, d))
        Q, _ = jnp.linalg.qr(A)
        return [Q[i].astype(jnp.complex128) for i in range(count)]

    return _make
```

Then update `tests/optimizers/test_gd.py` and `tests/optimizers/test_adam.py` to take `random_unitary_tensors` as a fixture parameter and call it instead of duplicating `_random_unitary_tensors` locally:

```bash
# Show current usage; engineer makes the local edit
grep -n "_random_unitary_tensors" tests/optimizers/*.py
```

For each occurrence: change `tensors = _random_unitary_tensors(d=4, count=3)` to `tensors = random_unitary_tensors(d=4, count=3)` and add `random_unitary_tensors` to the test function signature. Delete the local `_random_unitary_tensors` definition.

- [ ] **Step 8.6: Add `tests/training/conftest.py`**

```python
"""Training-test fixtures: tiny target image and tiny dataset."""

import jax.numpy as jnp
import pytest


@pytest.fixture
def tiny_target():
    """4×4 complex target with simple structure (uniform amplitude)."""
    return jnp.ones((4, 4), dtype=jnp.complex128) / 4.0


@pytest.fixture
def tiny_dataset(tiny_target):
    """Three-image dataset (just three copies of tiny_target with phase shifts)."""
    return [tiny_target, tiny_target * jnp.exp(1j * 0.5), tiny_target * jnp.exp(1j * 1.0)]
```

- [ ] **Step 8.7: Add `tests/io/conftest.py`**

```python
"""IO-test fixtures: tmp-path helpers."""

import pytest


@pytest.fixture
def tmp_basis_path(tmp_path):
    """Path inside the per-test tmp_path with a `.json` extension for save_basis."""
    return tmp_path / "basis.json"
```

- [ ] **Step 8.8: Add `tests/bases/conftest.py`**

```python
"""Basis-test fixtures: pre-built tiny bases."""

import pytest

from pdft.bases.base import QFTBasis


@pytest.fixture
def qft_2x2():
    return QFTBasis(2, 2, seed=0)
```

(Engineer: extend with more fixtures as the basis tests are touched. Start minimal — YAGNI.)

- [ ] **Step 8.9: Parametrize `tests/bases/test_new_bases.py`**

Read the file first:

```bash
cat tests/bases/test_new_bases.py | head -40
```

If the tests follow the pattern `test_<thing>_for_block_basis`, `test_<thing>_for_rich_basis`, `test_<thing>_for_real_rich_basis` — collapse using `@pytest.mark.parametrize`. Example pattern (engineer adapts to actual content):

```python
import pytest

from pdft.bases.block import BlockedBasis, RealRichBasis, RichBasis


@pytest.fixture(params=[BlockedBasis, RichBasis, RealRichBasis], ids=["block", "rich", "real_rich"])
def block_family_basis(request):
    cls = request.param
    return cls(...)  # construct with reasonable defaults


def test_round_trip_unitarity(block_family_basis):
    # one assertion that runs for all three bases automatically
    ...
```

If the tests aren't structured this way, leave them as-is — KISS, no forced parametrization.

- [ ] **Step 8.10: Replace `assert abs(a - b) < eps` with `pytest.approx`**

```bash
grep -rn "assert abs(.* - .*) < " tests/ 2>/dev/null
```

For each match, rewrite to `pytest.approx`:

```python
# OLD:
assert abs(loss_value - 0.5) < 1e-6

# NEW:
import pytest
assert loss_value == pytest.approx(0.5, abs=1e-6)
```

Add `import pytest` at the top of any file that gains a `pytest.approx` call but didn't import it.

- [ ] **Step 8.11: Verify markers work**

```bash
pytest --collect-only -m parity -q | tail -10            # only parity tests
pytest --collect-only -m "not parity" -q | tail -10      # everything else
pytest -m slow --collect-only -q                         # only slow tests
pytest -m "not slow and not parity" -q                   # fast feedback loop
```

Expected: `pytest -m parity` shows only tests under `tests/parity/`. `pytest -m slow` shows the two long_run/integration files.

- [ ] **Step 8.12: Run full suite + lint**

```bash
ruff check src tests
ruff format --check src tests
pytest -q                                       # full suite
pytest --cov=pdft --cov-fail-under=90           # coverage gate (final-final check)
pytest -m "not parity" -q                       # the new fast iteration loop
```

Expected: all green. The fast iteration loop (`-m "not parity"`) should be noticeably faster than the full suite.

- [ ] **Step 8.13: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test: add shared fixtures, parity/slow markers, parametrize cross-basis tests

- tests/parity/  (was tests/test_parity_*.py): grouped under one
  directory with auto-applied @pytest.mark.parity via conftest.py.
  Drops the test_parity_ filename prefix.
- tests/conftest.py: add goldens_dir + load_golden fixtures (replaces
  the duplicated `_load` helper in every parity test file).
- Per-subdir conftest.py: random_unitary_tensors (optimizers/),
  tiny_target / tiny_dataset (training/), tmp_basis_path (io/),
  qft_2x2 (bases/).
- pyproject.toml: register `parity` and `slow` markers.
- test_new_bases.py: parametrize cross-basis cases (block/rich/real_rich).
- Replace `assert abs(a - b) < eps` with `pytest.approx`.

Fast iteration loop is now `pytest -m "not parity"`; full byte-compare
runs as `pytest -m parity`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (before pushing the PR)

- [ ] **Step F1: Full gauntlet**

```bash
ruff check src tests
ruff format --check src tests
pytest --cov=pdft --cov-fail-under=90
pytest -m parity                                 # all parity tests still green
pytest -m "not parity" -q                        # fast loop green
```

- [ ] **Step F2: Public API diff (final)**

```bash
python -c "import pdft; print(sorted(n for n in dir(pdft) if not n.startswith('_')))" \
    > /tmp/api_final.txt
diff /tmp/api_before.txt /tmp/api_final.txt
```

Confirm only intentional removals — should match the list in Task 7 step 7.4.

- [ ] **Step F3: Loss-history hash (regression check)**

```bash
python -c "
from pdft.bases.base import QFTBasis
from pdft.loss import L1Norm
from pdft.optimizers import RiemannianGD
from pdft.training import train_basis
import jax.numpy as jnp, hashlib

basis = QFTBasis(2, 2, seed=0)
target = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0
res = train_basis(basis, target=target, loss=L1Norm(), optimizer=RiemannianGD(lr=0.01), steps=20, seed=0)
print(hashlib.sha256(repr([round(x, 12) for x in res.loss_history]).encode()).hexdigest())
" > /tmp/loss_history_hash_final.txt
diff /tmp/loss_history_hash_before.txt /tmp/loss_history_hash_final.txt
```

Expected: hashes match (no behavior change).

- [ ] **Step F4: Bisectability check**

```bash
git log --oneline refactor/modular-src ^main
```

Expected: 9 commits (1 spec + 8 refactor). Each one is independently testable.

- [ ] **Step F5: Push and open the PR**

```bash
git push -u origin refactor/modular-src
gh pr create --title "refactor: modular src/pdft layout (8-commit clean-break refactor)" --body "$(cat <<'EOF'
## Summary
- Reorganize flat `src/pdft/` (21 files) into focused subpackages: `bases/`, `optimizers/`, `training/`, `io/`, `circuit/`, `viz/`
- Delete `DCTBasis` (low value over other parametric bases)
- Extract one duplicated bookkeeping block from `training.py` into `training/eval_loop.py` (only intentional logic change)
- Lightly modernize tests: shared fixtures in per-subdir `conftest.py`, `parity` and `slow` markers, parametrize cross-basis cases, `pytest.approx` for scalars
- Clean break on imports (no backwards-compat shims; sole-developer codebase)

8 commits, dependency-topological order, each independently bisectable. **Do not squash-merge** — preserve the history for `git bisect`.

## Test plan
- [x] `ruff check src tests` clean
- [x] `pytest --cov=pdft --cov-fail-under=90` passes
- [x] `pytest -m parity` (all Julia byte-compare tests) green
- [x] `pytest -m "not parity"` is the new fast iteration loop
- [x] Loss-history SHA-256 on QFT(2,2) seed=0 / 20-step fixture: matches pre-refactor (proves `eval_loop.py` extraction is byte-exact)
- [x] Public API diff: removals only (~30 names dropped from root; all still importable from subpackages)
- [x] All three examples (basis_demo, optimizer_benchmark, mera_demo) execute end-to-end
- [ ] CI matrix (3.11 / 3.12 / 3.13): all green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review (per writing-plans skill)

**1. Spec coverage:** Each section of the design doc maps to a task:
- Final structure → Tasks 1–7 (one task per subsystem)
- DRY fix (`eval_loop.py`) → Task 5
- Test refactor → Task 8
- Pre-flight (snapshots) → Pre-flight section
- Per-commit gate (ruff + pytest + parity) → embedded in every task
- Risk register entries (parity drift, public API drift, io/ shadowing) → addressed in Steps P3, F2, F3, 6.2

**2. Placeholder scan:** No "TBD" / "TODO" / "implement later" / unspecified-error-handling. Step 5.4 (`adam_step.py`) tells the engineer to copy a function body verbatim from the original file rather than reproducing 110 lines twice — this is intentional (the body is unchanged) and the source location is precise (`training.py` lines 195–283).

**3. Type/name consistency:** `cosine_with_warmup` (no leading underscore) is used consistently in `schedules.py`, `batched.py`, `profiling.py` (with local alias), and the test rewrite in step 5.11. `evaluate_and_check_early_stop` returns `(best_tensors, best_val, patience, stop, val_loss_recorded)` — used identically in step 5.6's batched.py replacement and in the eval_loop test fixture in step 5.11.

---

## Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks. Each task is well-bounded (most are 5–15 file moves + import sed), so subagents won't overflow context.

**2. Inline Execution** — Execute tasks in this session via `executing-plans` skill, with checkpoints between commits.

Which approach?
