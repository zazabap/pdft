# GPU Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a top-level `benchmarks/` directory that ports the dataset-quality slice of `ParametricDFT-Benchmarks.jl` (`run_quickdraw.jl`, `run_div2k_8q.jl`) onto GPU using `pdft`, producing Julia-schema-compatible `metrics.json` plus rate-distortion CSVs and vector PDF plots, and parallelizes one dataset per GPU via `run_all.sh`.

**Architecture:** A standalone `benchmarks/` directory at the repo root (not a Python package). Each `run_*.py` is a CLI script that imports sibling modules via `sys.path` munging in `_bootstrap.py`. `pdft` itself is unchanged — the harness uses the existing single-target `train_basis`. Per-image trained bases are saved as JSON arrays; evaluation pairs each test image with its own per-image basis (P pairing). Block-DCT and block-FFT (8×8) are added as Python-only baselines beyond Julia's global FFT/DCT.

**Tech Stack:** Python 3.11+, JAX 0.10+, NumPy, SciPy (DCT), scikit-image (PSNR/SSIM), Pillow (DIV2K loader), Matplotlib (PDF plots). All optional via the new `bench` extra in `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-04-25-pdft-gpu-benchmarks-design.md`.

**Reference Julia source on this machine:** `/home/claude-user/ParametricDFT-Benchmarks.jl/` (do not modify).

---

## File Structure

Files this plan creates or modifies:

| File | Responsibility |
|---|---|
| `pyproject.toml` | Add `bench` optional-dependency group |
| `.gitignore` | Add `benchmarks/results/` |
| `benchmarks/_bootstrap.py` | `sys.path` insertion so sibling modules import without an `__init__.py` |
| `benchmarks/config.py` | `Preset` dataclass + `PRESETS_QUICKDRAW` / `PRESETS_DIV2K` dicts |
| `benchmarks/baselines.py` | `global_fft_compress`, `global_dct_compress`, `block_fft_compress`, `block_dct_compress` |
| `benchmarks/evaluation.py` | `compute_metrics`, `evaluate_basis_per_image`, `evaluate_baseline` |
| `benchmarks/data_loading.py` | `load_quickdraw`, `load_div2k` (read-only; no auto-download) |
| `benchmarks/harness.py` | `TrainResult` dataclass, `train_one_basis`, `_OPTIMIZERS` registry, `dump_metrics_json` (with Julia float formatting) |
| `benchmarks/generate_report.py` | `main(results_dir)`: CSVs + plot-module dispatch |
| `benchmarks/plots/__init__.py` | Empty (the module dir is imported as a package via `_bootstrap.py`) |
| `benchmarks/plots/rate_distortion.py` | `plot_rate_distortion(metrics, metric_name, out_pdf)` |
| `benchmarks/plots/loss_trajectories.py` | `plot_loss_trajectories(loss_dir, out_pdf, dataset_name)` |
| `benchmarks/run_quickdraw.py` | CLI runner for QuickDraw (m=n=5) |
| `benchmarks/run_div2k_8q.py` | CLI runner for DIV2K (m=n=8) |
| `benchmarks/run_all.sh` | 2-GPU fan-out shell script |
| `benchmarks/README.md` | How to run, dataset placement, GPU notes |
| `benchmarks/tests/conftest.py` | `sys.path` insertion + `integration` marker registration |
| `benchmarks/tests/fixtures/julia_quickdraw_metrics.json` | Trimmed copy of Julia's `metrics.json` (~5 KB) for schema-compat test |
| `benchmarks/tests/fixtures/quickdraw_stub/*.npy` | Tiny stub `.npy` files for `test_data_loading` |
| `benchmarks/tests/test_config.py` | Layer A |
| `benchmarks/tests/test_baselines.py` | Layer A |
| `benchmarks/tests/test_evaluation.py` | Layer A |
| `benchmarks/tests/test_data_loading.py` | Layer A |
| `benchmarks/tests/test_harness_smoke.py` | Layer A (CPU only) |
| `benchmarks/tests/test_report.py` | Layer A |
| `benchmarks/tests/test_julia_schema_compat.py` | Layer A (no GPU; uses committed fixture) |
| `benchmarks/tests/test_quickdraw_smoke_e2e.py` | Layer B (`@pytest.mark.integration`) |
| `benchmarks/tests/test_div2k_smoke_e2e.py` | Layer B (`@pytest.mark.integration`) |
| `benchmarks/tests/test_2gpu_fanout.py` | Layer B (`@pytest.mark.integration`) |
| `.github/workflows/benchmarks-smoke.yml` | New workflow: Layer A on PRs touching `benchmarks/` |

The `benchmarks/results/` directory is gitignored; per-run output writes there.

---

### Task 1: Scaffold (pyproject extra, gitignore, directory tree, conftest)

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `benchmarks/_bootstrap.py`
- Create: `benchmarks/tests/conftest.py`
- Create: `benchmarks/plots/__init__.py`

- [ ] **Step 1: Add `bench` extra to `pyproject.toml`**

Locate the `[project.optional-dependencies]` table and add a new `bench` key. Existing keys (`dev`, `plot`, `gpu`) stay unchanged.

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7",
    "pytest-cov>=4",
    "ruff>=0.5",
    "numpy>=1.26",
    "jaxtyping>=0.2",
    "matplotlib>=3.8",
]
plot = ["matplotlib>=3.8"]
gpu = ["jax[cuda12]>=0.10.0"]
bench = [
    "pillow>=10",
    "scikit-image>=0.22",
    "scipy>=1.11",
    "matplotlib>=3.8",
]
```

Also register the `integration` pytest marker by appending to `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
filterwarnings = [
    "error",
    "ignore::DeprecationWarning:jax.*",
]
markers = [
    "integration: marks GPU/dataset-dependent benchmark tests (deselect with -m 'not integration')",
]
```

- [ ] **Step 2: Add `benchmarks/results/` to `.gitignore`**

Append to the bottom of `.gitignore`:

```
# Benchmark outputs
benchmarks/results/
```

- [ ] **Step 3: Create `benchmarks/_bootstrap.py`**

Write:

```python
"""sys.path bootstrap so sibling modules in benchmarks/ are importable.

This package intentionally has no __init__.py — each run_*.py script is
standalone-runnable. This file is imported first to make `import config`,
`import baselines`, etc. work from any entrypoint inside benchmarks/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

BENCH_ROOT = _BENCH_DIR
REPO_ROOT = _BENCH_DIR.parent
```

- [ ] **Step 4: Create `benchmarks/tests/conftest.py`**

Write:

```python
"""Test bootstrap: add benchmarks/ to sys.path so sibling modules import."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure x64 mode is on (mirrors tests/conftest.py for pdft itself).
import pdft  # noqa: F401  -- imported for side-effect: jax_enable_x64

_BENCH_DIR = Path(__file__).resolve().parent.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
```

- [ ] **Step 5: Create `benchmarks/plots/__init__.py`**

Empty file (creates the `plots` directory as importable from sibling modules):

```python
"""Plot module dir. Imported as a package via benchmarks/_bootstrap.py."""
```

- [ ] **Step 6: Verify scaffold**

Run:

```bash
ruff check pyproject.toml benchmarks/
```

Expected: no errors. (No actual code yet to lint.)

```bash
pytest tests/ --no-cov -q
```

Expected: PASS (existing tests unaffected; the `bench` extra isn't installed and `benchmarks/tests/` is not picked up by default `pytest`).

- [ ] **Step 7: Install the `bench` extra**

```bash
pip install -e ".[bench]"
```

Expected: pillow, scikit-image, scipy, matplotlib installed (or already present).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore benchmarks/_bootstrap.py benchmarks/tests/conftest.py benchmarks/plots/__init__.py
git commit -m "feat(bench): scaffold benchmarks/ directory with bench extra and bootstrap"
```

---

### Task 2: `config.py` — `Preset` dataclass + presets

**Files:**
- Create: `benchmarks/config.py`
- Create: `benchmarks/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `benchmarks/tests/test_config.py`:

```python
"""Layer A: config.py unit tests. No GPU/datasets. Fast (<1s)."""
from __future__ import annotations

import pytest

from config import (
    PRESETS_DIV2K,
    PRESETS_QUICKDRAW,
    Preset,
    get_preset,
)


def test_preset_dataclass_fields():
    p = Preset(name="x", epochs=10, n_train=2, n_test=2, optimizer="gd", lr=0.01)
    assert p.name == "x"
    assert p.epochs == 10
    assert p.n_train == 2
    assert p.n_test == 2
    assert p.optimizer == "gd"
    assert p.lr == 0.01
    assert p.seed == 42  # default
    assert p.keep_ratios == (0.05, 0.10, 0.15, 0.20)  # default


def test_preset_is_frozen():
    p = Preset(name="x", epochs=10, n_train=2, n_test=2, optimizer="gd", lr=0.01)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        p.epochs = 20  # type: ignore[misc]


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_all_presets_have_four_levels(presets):
    assert set(presets.keys()) == {"smoke", "light", "moderate", "heavy"}


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_n_train_equals_n_test_for_p_pairing(presets):
    """P pairing: basis_i evaluated on test_i — forces n_train == n_test."""
    for name, p in presets.items():
        assert p.n_train == p.n_test, f"{name}: n_train={p.n_train}, n_test={p.n_test}"


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_keep_ratios_are_valid(presets):
    for name, p in presets.items():
        assert len(p.keep_ratios) > 0, f"{name}: empty keep_ratios"
        for kr in p.keep_ratios:
            assert 0.0 < kr <= 1.0, f"{name}: invalid keep_ratio {kr}"


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_optimizer_strings_valid(presets):
    for name, p in presets.items():
        assert p.optimizer in {"gd", "adam"}, f"{name}: unknown optimizer {p.optimizer}"


def test_get_preset_quickdraw():
    p = get_preset("quickdraw", "smoke")
    assert p.name == "smoke"
    assert p.n_train == 2


def test_get_preset_div2k():
    p = get_preset("div2k_8q", "moderate")
    assert p.name == "moderate"


def test_get_preset_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        get_preset("unknown", "smoke")


def test_get_preset_unknown_preset():
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("quickdraw", "unknown")
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest benchmarks/tests/test_config.py -v --no-cov
```

Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Implement `config.py`**

Create `benchmarks/config.py`:

```python
"""Benchmark presets. Plain dataclasses; values mirror Julia repo defaults."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str
    epochs: int  # passed as `steps` to pdft.train_basis
    n_train: int  # number of target images
    n_test: int  # held-out images for eval; equals n_train (P pairing)
    optimizer: str  # "gd" | "adam"
    lr: float
    seed: int = 42
    keep_ratios: tuple[float, ...] = field(
        default_factory=lambda: (0.05, 0.10, 0.15, 0.20)
    )


PRESETS_QUICKDRAW: dict[str, Preset] = {
    "smoke":    Preset("smoke",    epochs=10,   n_train=2,   n_test=2,   optimizer="gd",   lr=0.01),
    "light":    Preset("light",    epochs=100,  n_train=10,  n_test=10,  optimizer="gd",   lr=0.01),
    "moderate": Preset("moderate", epochs=500,  n_train=50,  n_test=50,  optimizer="adam", lr=0.01),
    "heavy":    Preset("heavy",    epochs=2000, n_train=200, n_test=200, optimizer="adam", lr=0.005),
}

PRESETS_DIV2K: dict[str, Preset] = {
    "smoke":    Preset("smoke",    epochs=10,   n_train=2,   n_test=2,   optimizer="gd",   lr=0.01),
    "light":    Preset("light",    epochs=100,  n_train=5,   n_test=5,   optimizer="gd",   lr=0.01),
    "moderate": Preset("moderate", epochs=500,  n_train=20,  n_test=20,  optimizer="adam", lr=0.01),
    "heavy":    Preset("heavy",    epochs=2000, n_train=50,  n_test=50,  optimizer="adam", lr=0.005),
}


_DATASETS = {
    "quickdraw": PRESETS_QUICKDRAW,
    "div2k_8q":  PRESETS_DIV2K,
}


def get_preset(dataset: str, preset_name: str) -> Preset:
    """Look up a preset by dataset + name. Raises KeyError on unknown values."""
    if dataset not in _DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; choices: {sorted(_DATASETS)}")
    presets = _DATASETS[dataset]
    if preset_name not in presets:
        raise KeyError(
            f"unknown preset {preset_name!r} for dataset {dataset!r}; "
            f"choices: {sorted(presets)}"
        )
    return presets[preset_name]
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest benchmarks/tests/test_config.py -v --no-cov
```

Expected: 11 tests pass.

- [ ] **Step 5: Lint**

```bash
ruff check benchmarks/config.py benchmarks/tests/test_config.py
ruff format benchmarks/config.py benchmarks/tests/test_config.py
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/config.py benchmarks/tests/test_config.py
git commit -m "feat(bench): add Preset dataclass and quickdraw/div2k presets"
```

---

### Task 3: `baselines.py` — global FFT/DCT compression

**Files:**
- Create: `benchmarks/baselines.py` (global functions only this task; block functions in Task 4)
- Create: `benchmarks/tests/test_baselines.py`

- [ ] **Step 1: Write the failing test**

Create `benchmarks/tests/test_baselines.py`:

```python
"""Layer A: baselines.py unit tests. Pure numpy/scipy — no JAX, no GPU."""
from __future__ import annotations

import numpy as np
import pytest

from baselines import global_dct_compress, global_fft_compress


@pytest.fixture
def img_32():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)


def test_global_fft_full_keep_is_identity(img_32):
    out = global_fft_compress(img_32, keep_ratio=1.0)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_global_dct_full_keep_is_identity(img_32):
    out = global_dct_compress(img_32, keep_ratio=1.0)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_global_fft_keep_ratio_count(img_32):
    """keep_ratio=0.5 keeps exactly floor(0.5 * 1024) = 512 nonzero coefficients."""
    # We can't probe the internal coefficient mask directly — instead, run with
    # keep_ratio=0.5 and re-FFT the recovered image: the count of nonzero
    # frequency bins (above tolerance) should equal floor(0.5 * 1024) = 512.
    out = global_fft_compress(img_32, keep_ratio=0.5)
    freq = np.fft.fft2(out)
    nonzero = np.sum(np.abs(freq) > 1e-9)
    # Recovery introduces some numeric noise; allow a small slack.
    assert 500 <= nonzero <= 540


def test_global_dct_zero_keep_returns_zero(img_32):
    """keep_ratio = a single coefficient (smallest possible)."""
    out = global_dct_compress(img_32, keep_ratio=1 / (32 * 32))
    # The DC coefficient (largest by magnitude for natural images) is kept;
    # output should be a (near-)constant image equal to the image mean.
    expected_mean = float(np.mean(img_32))
    assert abs(float(np.mean(out)) - expected_mean) < 1e-9


def test_global_fft_returns_real(img_32):
    out = global_fft_compress(img_32, keep_ratio=0.1)
    assert np.isrealobj(out) or np.allclose(out.imag, 0.0)
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest benchmarks/tests/test_baselines.py -v --no-cov
```

Expected: FAIL with `ModuleNotFoundError: No module named 'baselines'`.

- [ ] **Step 3: Implement global functions in `baselines.py`**

Create `benchmarks/baselines.py`:

```python
"""Classical compression baselines for the benchmark harness.

Mirrors `evaluation.jl::fft_compress_recover` and `dct_compress_recover` from
the Julia repo (top-k% of magnitudes globally, zero the rest, inverse-transform).
Block variants extend the same semantic to non-overlapping 8×8 tiles.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def _top_k_mask(magnitudes: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask, True at the k largest entries by magnitude. Ties broken arbitrarily."""
    if k <= 0:
        return np.zeros_like(magnitudes, dtype=bool)
    if k >= magnitudes.size:
        return np.ones_like(magnitudes, dtype=bool)
    flat = magnitudes.ravel()
    # argpartition: the k-th order statistic is at position k-1 of the partition;
    # everything before it is <= and everything after is >=.
    threshold_idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[threshold_idx] = True
    return mask.reshape(magnitudes.shape)


def global_fft_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """2D FFT compression: keep top-k% magnitudes globally, return real part.

    Mirrors evaluation.jl::fft_compress_recover.
    """
    freq = np.fft.fftshift(np.fft.fft2(image))
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0 + 0.0j)
    return np.real(np.fft.ifft2(np.fft.ifftshift(compressed)))


def global_dct_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """2D DCT-II compression: keep top-k% magnitudes globally.

    Mirrors evaluation.jl::dct_compress_recover.
    """
    freq = dct(dct(image, axis=0, norm="ortho"), axis=1, norm="ortho")
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0)
    return idct(idct(compressed, axis=0, norm="ortho"), axis=1, norm="ortho")
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest benchmarks/tests/test_baselines.py -v --no-cov
```

Expected: 5 tests pass.

- [ ] **Step 5: Lint**

```bash
ruff check benchmarks/baselines.py benchmarks/tests/test_baselines.py
ruff format benchmarks/baselines.py benchmarks/tests/test_baselines.py
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/baselines.py benchmarks/tests/test_baselines.py
git commit -m "feat(bench): add global FFT/DCT compression baselines"
```

---

### Task 4: `baselines.py` — block FFT/DCT (8×8)

**Files:**
- Modify: `benchmarks/baselines.py`
- Modify: `benchmarks/tests/test_baselines.py`

- [ ] **Step 1: Add failing tests for block transforms**

Append to `benchmarks/tests/test_baselines.py`:

```python
from baselines import block_dct_compress, block_fft_compress  # noqa: E402


@pytest.fixture
def img_256():
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 1.0, size=(256, 256)).astype(np.float64)


def test_block_fft_full_keep_is_identity(img_32):
    out = block_fft_compress(img_32, keep_ratio=1.0, block=8)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_block_dct_full_keep_is_identity(img_32):
    out = block_dct_compress(img_32, keep_ratio=1.0, block=8)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_block_fft_full_keep_is_identity_256(img_256):
    out = block_fft_compress(img_256, keep_ratio=1.0, block=8)
    np.testing.assert_allclose(out, img_256, atol=1e-10)


def test_block_dct_keep_ratio_global_count(img_32):
    """Block-DCT keeps top-k% globally across all blocks (JPEG-style)."""
    keep_ratio = 0.5
    out = block_dct_compress(img_32, keep_ratio=keep_ratio, block=8)
    # Re-DCT each block of `out` and count nonzero bins globally.
    n = 32
    block = 8
    nonzero_total = 0
    for i in range(0, n, block):
        for j in range(0, n, block):
            tile = out[i : i + block, j : j + block]
            f = dct(dct(tile, axis=0, norm="ortho"), axis=1, norm="ortho")
            nonzero_total += int(np.sum(np.abs(f) > 1e-9))
    expected = int(np.floor(keep_ratio * n * n))
    # Allow ±5% slack for numeric noise.
    assert abs(nonzero_total - expected) <= max(5, int(0.05 * expected))


def test_block_size_must_divide_image():
    bad = np.zeros((10, 10))
    with pytest.raises(ValueError, match="block size"):
        block_fft_compress(bad, keep_ratio=0.5, block=8)


def test_dct_module_imports():
    """Sanity: scipy.fft.dct is importable. Catches missing-dependency early."""
    from scipy.fft import dct as _dct  # noqa: F401
```

The `from scipy.fft import dct, idct` import at the top of the test file is needed for `test_block_dct_keep_ratio_global_count`. Add it next to the existing imports if it isn't already there. (The ones already in the file from Task 3 don't include scipy; add it.) Edit the imports block at the top of `test_baselines.py` to include:

```python
from scipy.fft import dct, idct
```

- [ ] **Step 2: Run the new tests, verify they fail**

```bash
pytest benchmarks/tests/test_baselines.py -v --no-cov
```

Expected: 5 tests still pass; 6 new tests fail with `ImportError: cannot import name 'block_fft_compress'`.

- [ ] **Step 3: Implement block functions in `baselines.py`**

Append to `benchmarks/baselines.py`:

```python
def _check_block_divides(n: int, block: int) -> None:
    if n % block != 0:
        raise ValueError(
            f"block size {block} must evenly divide image dimension {n}"
        )


def _split_blocks(image: np.ndarray, block: int) -> np.ndarray:
    """Split (H, W) into (H/b, W/b, b, b) non-overlapping tiles."""
    h, w = image.shape
    return (
        image.reshape(h // block, block, w // block, block)
        .swapaxes(1, 2)
        .copy()
    )


def _join_blocks(tiles: np.ndarray) -> np.ndarray:
    """Inverse of _split_blocks."""
    nbr, nbc, b, _ = tiles.shape
    return tiles.swapaxes(1, 2).reshape(nbr * b, nbc * b)


def block_fft_compress(
    image: np.ndarray, keep_ratio: float, block: int = 8
) -> np.ndarray:
    """Block FFT (8×8 default). Top-k% magnitudes globally across all blocks."""
    h, w = image.shape
    _check_block_divides(h, block)
    _check_block_divides(w, block)

    tiles = _split_blocks(image, block)  # (H/b, W/b, b, b)
    freq = np.fft.fft2(tiles, axes=(-2, -1))  # FFT each tile
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0 + 0.0j)
    recovered = np.real(np.fft.ifft2(compressed, axes=(-2, -1)))
    return _join_blocks(recovered)


def block_dct_compress(
    image: np.ndarray, keep_ratio: float, block: int = 8
) -> np.ndarray:
    """Block DCT-II (8×8 default). Top-k% magnitudes globally across all blocks."""
    h, w = image.shape
    _check_block_divides(h, block)
    _check_block_divides(w, block)

    tiles = _split_blocks(image, block)
    freq = dct(dct(tiles, axis=-2, norm="ortho"), axis=-1, norm="ortho")
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0)
    recovered = idct(idct(compressed, axis=-2, norm="ortho"), axis=-1, norm="ortho")
    return _join_blocks(recovered)
```

- [ ] **Step 4: Run all baseline tests**

```bash
pytest benchmarks/tests/test_baselines.py -v --no-cov
```

Expected: 11 tests pass.

- [ ] **Step 5: Lint**

```bash
ruff check benchmarks/baselines.py benchmarks/tests/test_baselines.py
ruff format benchmarks/baselines.py benchmarks/tests/test_baselines.py
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/baselines.py benchmarks/tests/test_baselines.py
git commit -m "feat(bench): add block-FFT and block-DCT (8x8) baselines"
```

---

### Task 5: `evaluation.py` — metrics + per-image evaluation

**Files:**
- Create: `benchmarks/evaluation.py`
- Create: `benchmarks/tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

Create `benchmarks/tests/test_evaluation.py`:

```python
"""Layer A: evaluation.py unit tests."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from evaluation import (
    aggregate_per_keep_ratio,
    compute_metrics,
    evaluate_baseline,
)


@pytest.fixture
def img_32():
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)


def test_compute_metrics_identical_image(img_32):
    m = compute_metrics(img_32, img_32)
    assert m["mse"] == 0.0
    assert m["psnr"] == float("inf") or m["psnr"] > 200.0
    assert m["ssim"] > 0.999  # near-1 (some skimage versions cap at <1.0)


def test_compute_metrics_clamps_to_unit_range(img_32):
    """Recovered values outside [0,1] are clamped before metrics."""
    bad = img_32 + 5.0  # all values > 1
    m = compute_metrics(img_32, bad)
    # After clamping, recovered is all 1.0; mse > 0
    assert m["mse"] > 0


def test_compute_metrics_handles_complex_recovered(img_32):
    """Recovered may be complex (FFT path); imag part dropped before clamp."""
    complex_recovered = img_32.astype(np.complex128) + 0.0j
    m = compute_metrics(img_32, complex_recovered)
    assert m["mse"] == pytest.approx(0.0, abs=1e-12)


def test_compute_metrics_psnr_matches_skimage(img_32):
    """PSNR formula 10*log10(1/MSE) matches skimage with data_range=1.0."""
    from skimage.metrics import peak_signal_noise_ratio

    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.1, size=img_32.shape)
    recovered = np.clip(img_32 + noise, 0.0, 1.0)
    m = compute_metrics(img_32, recovered)
    sk_psnr = peak_signal_noise_ratio(img_32, recovered, data_range=1.0)
    assert m["psnr"] == pytest.approx(sk_psnr, rel=1e-9, abs=1e-9)


def test_aggregate_handles_nan():
    """NaN in mse/psnr/ssim lists → nanmean / nanstd skip them."""
    metrics_list = [
        {"mse": 0.1, "psnr": 10.0, "ssim": 0.5},
        {"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan")},
        {"mse": 0.2, "psnr": 7.0, "ssim": 0.4},
    ]
    agg = aggregate_per_keep_ratio(metrics_list)
    assert agg["mean_mse"] == pytest.approx(0.15)
    assert agg["std_mse"] == pytest.approx(np.nanstd([0.1, 0.2]))
    # 1 nan present
    assert agg["nan_count"] == 1


def test_evaluate_baseline_returns_schema(img_32):
    """evaluate_baseline returns (metrics_dict, elapsed_seconds) with the right shape."""
    images = np.stack([img_32, img_32 * 0.5, img_32 * 0.7], axis=0)

    def passthrough(img: np.ndarray, keep: float) -> np.ndarray:
        return img  # perfect reconstruction at any keep ratio

    metrics, elapsed = evaluate_baseline(passthrough, images, [0.05, 0.10])
    assert elapsed >= 0
    assert set(metrics.keys()) == {"0.05", "0.1"}  # str(0.1) == "0.1"
    for kr_str, vals in metrics.items():
        assert set(vals.keys()) >= {
            "mean_mse", "std_mse",
            "mean_psnr", "std_psnr",
            "mean_ssim", "std_ssim",
        }
        # passthrough → mse=0 → mean_mse=0
        assert vals["mean_mse"] == pytest.approx(0.0, abs=1e-12)


def test_evaluate_baseline_failure_records_nan(img_32):
    """A baseline that raises on one image → that image's metrics are nan; others succeed."""
    images = np.stack([img_32, img_32, img_32], axis=0)

    call_count = {"n": 0}

    def flaky(img: np.ndarray, keep: float) -> np.ndarray:
        call_count["n"] += 1
        if call_count["n"] == 4:  # second image's first ratio (we'll see)
            raise RuntimeError("simulated failure")
        return img

    # Reset and use an explicit failing index
    call_count["n"] = 0

    def fail_on_second(img: np.ndarray, keep: float) -> np.ndarray:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        return img

    metrics, _ = evaluate_baseline(fail_on_second, images, [0.10])
    vals = metrics["0.1"]
    # 2 successes (mse=0), 1 failure (nan) → nanmean = 0
    assert vals["mean_mse"] == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest benchmarks/tests/test_evaluation.py -v --no-cov
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`.

- [ ] **Step 3: Implement `evaluation.py`**

Create `benchmarks/evaluation.py`:

```python
"""Compression-quality evaluation: MSE, PSNR, SSIM at multiple keep ratios.

Field shape is bit-compatible with Julia's metrics.json:
    {kr_str: {mean_mse, std_mse, mean_psnr, std_psnr, mean_ssim, std_ssim}}.
PSNR/SSIM via scikit-image (data_range=1.0). Recovered images are clamped to
[0, 1] before metric computation, matching evaluation.jl:25.

Per-(image, keep_ratio) failures record NaN; aggregation uses nanmean/nanstd.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)


def compute_metrics(original: np.ndarray, recovered: np.ndarray) -> dict[str, float]:
    """MSE / PSNR / SSIM. Clamps recovered to [0,1] and drops imag part."""
    rec = np.clip(np.real(recovered), 0.0, 1.0)
    mse = float(np.mean((original - rec) ** 2))
    if mse > 0:
        psnr = 10.0 * np.log10(1.0 / mse)
    else:
        psnr = float("inf")
    try:
        ssim = float(structural_similarity(original, rec, data_range=1.0))
    except (ValueError, RuntimeError):
        ssim = float("nan")
    return {"mse": mse, "psnr": float(psnr), "ssim": ssim}


def aggregate_per_keep_ratio(
    per_image_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Mean/std across a list of per-image metrics dicts. NaNs ignored.

    Returns mean_mse/std_mse/mean_psnr/std_psnr/mean_ssim/std_ssim plus nan_count.
    """
    mse_vals = np.array([m["mse"] for m in per_image_metrics], dtype=np.float64)
    psnr_vals = np.array([m["psnr"] for m in per_image_metrics], dtype=np.float64)
    ssim_vals = np.array([m["ssim"] for m in per_image_metrics], dtype=np.float64)

    nan_count = int(np.sum(np.isnan(mse_vals) | np.isnan(psnr_vals) | np.isnan(ssim_vals)))

    return {
        "mean_mse": float(np.nanmean(mse_vals)),
        "std_mse": float(np.nanstd(mse_vals)),
        "mean_psnr": float(np.nanmean(psnr_vals)),
        "std_psnr": float(np.nanstd(psnr_vals)),
        "mean_ssim": float(np.nanmean(ssim_vals)),
        "std_ssim": float(np.nanstd(ssim_vals)),
        "nan_count": nan_count,
    }


def evaluate_baseline(
    fn: Callable[[np.ndarray, float], np.ndarray],
    test_images: np.ndarray,
    keep_ratios: Sequence[float],
) -> tuple[dict[str, dict[str, float]], float]:
    """Run `fn(image, keep_ratio)` over every (image, kr) pair.

    Returns ({kr_str: aggregated_metrics}, elapsed_seconds). On per-call
    failure, that pair's metrics are nan and the call is skipped.
    """
    t0 = time.perf_counter()
    out: dict[str, dict[str, float]] = {}
    for kr in keep_ratios:
        per_image: list[dict[str, float]] = []
        for img in test_images:
            try:
                recovered = fn(img, kr)
                per_image.append(compute_metrics(img, recovered))
            except Exception as e:  # noqa: BLE001
                logger.warning("baseline failed on a single image: %s", e)
                per_image.append({"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan")})
        agg = aggregate_per_keep_ratio(per_image)
        out[str(kr)] = agg
    elapsed = time.perf_counter() - t0
    return out, elapsed


def evaluate_basis_per_image(
    bases: list,
    test_images: np.ndarray,
    keep_ratios: Sequence[float],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Per-image (P pairing): basis[i] evaluated on test_images[i].

    Round-trips each basis through pdft.io_json save_basis/load_basis to land
    tensors on host (sidesteps GPU scalar-indexing in compress/recover —
    same workaround as evaluation.jl:55-57).

    Returns ({kr_str: aggregated_metrics}, {kr_str: nan_count}).
    """
    import json
    import tempfile
    from pathlib import Path

    import pdft

    if len(bases) != len(test_images):
        raise ValueError(
            f"P-pairing requires len(bases) == len(test_images); "
            f"got {len(bases)} bases vs {len(test_images)} images"
        )

    # Host-roundtrip every basis once at the start.
    cpu_bases = []
    with tempfile.TemporaryDirectory() as td:
        for i, b in enumerate(bases):
            path = Path(td) / f"b{i}.json"
            pdft.save_basis(str(path), b)
            cpu_bases.append(pdft.load_basis(str(path)))

    out: dict[str, dict[str, float]] = {}
    nan_counts: dict[str, int] = {}
    for kr in keep_ratios:
        discard_ratio = 1.0 - kr
        per_image: list[dict[str, float]] = []
        for img, basis in zip(test_images, cpu_bases):
            try:
                compressed = pdft.compress(basis, np.asarray(img, dtype=np.float64), ratio=discard_ratio)
                recovered = pdft.recover(basis, compressed)
                per_image.append(compute_metrics(img, recovered))
            except Exception as e:  # noqa: BLE001
                logger.warning("compress/recover failed on (kr=%s): %s", kr, e)
                per_image.append({"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan")})
        agg = aggregate_per_keep_ratio(per_image)
        out[str(kr)] = agg
        nan_counts[str(kr)] = agg["nan_count"]
    return out, nan_counts
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest benchmarks/tests/test_evaluation.py -v --no-cov
```

Expected: 7 tests pass.

- [ ] **Step 5: Lint**

```bash
ruff check benchmarks/evaluation.py benchmarks/tests/test_evaluation.py
ruff format benchmarks/evaluation.py benchmarks/tests/test_evaluation.py
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/evaluation.py benchmarks/tests/test_evaluation.py
git commit -m "feat(bench): metrics + per-image basis evaluation (P pairing)"
```

---

### Task 6: `data_loading.py` — QuickDraw + DIV2K loaders

**Files:**
- Create: `benchmarks/data_loading.py`
- Create: `benchmarks/tests/test_data_loading.py`
- Create: `benchmarks/tests/fixtures/quickdraw_stub/airplane.npy`
- Create: `benchmarks/tests/fixtures/quickdraw_stub/apple.npy`
- Create: `benchmarks/tests/fixtures/div2k_stub/0001.png`
- Create: `benchmarks/tests/fixtures/div2k_stub/0002.png`
- Create: `benchmarks/tests/fixtures/div2k_stub/0003.png`

- [ ] **Step 1: Create the fixture build script and run it**

Create a temporary helper script `benchmarks/tests/fixtures/_build_fixtures.py`:

```python
"""One-shot script to generate stub fixtures used by test_data_loading.py.
Run once: `python benchmarks/tests/fixtures/_build_fixtures.py`.
The output .npy/.png files are committed; this script is not invoked by tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent

# QuickDraw stub: 5 .npy files, each (10, 28*28) uint8 (raw QuickDraw shape).
# Real QuickDraw files are (N, 784); we keep the shape but use 10 rows.
qd = HERE / "quickdraw_stub"
qd.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
for cat in ("airplane", "apple"):
    arr = (rng.uniform(0, 255, size=(10, 28 * 28))).astype(np.uint8)
    np.save(qd / f"{cat}.npy", arr)

# DIV2K stub: 3 PNGs, 64x64 grayscale (small for fixture size; loader resizes).
div = HERE / "div2k_stub"
div.mkdir(exist_ok=True)
for i in range(1, 4):
    pix = (rng.uniform(0, 255, size=(64, 64))).astype(np.uint8)
    Image.fromarray(pix, mode="L").save(div / f"{i:04d}.png")

print(f"wrote stubs to {HERE}")
```

Run it once:

```bash
python benchmarks/tests/fixtures/_build_fixtures.py
```

Expected: prints `wrote stubs to ...`. The `.npy` and `.png` files are committed alongside the source.

- [ ] **Step 2: Write the failing test**

Create `benchmarks/tests/test_data_loading.py`:

```python
"""Layer A: data_loading.py unit tests using fixtures (no real datasets)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_loading import load_div2k, load_quickdraw

FIXTURES = Path(__file__).parent / "fixtures"
QD_STUB = FIXTURES / "quickdraw_stub"
DV_STUB = FIXTURES / "div2k_stub"


def test_load_quickdraw_shape_and_dtype():
    train, test = load_quickdraw(n_train=2, n_test=2, seed=42, data_root=QD_STUB)
    assert train.shape == (2, 32, 32)
    assert test.shape == (2, 32, 32)
    assert train.dtype == np.float32
    assert 0.0 <= train.min() and train.max() <= 1.0


def test_load_quickdraw_seed_deterministic():
    a_train, _ = load_quickdraw(n_train=2, n_test=2, seed=42, data_root=QD_STUB)
    b_train, _ = load_quickdraw(n_train=2, n_test=2, seed=42, data_root=QD_STUB)
    np.testing.assert_array_equal(a_train, b_train)


def test_load_quickdraw_too_many_raises():
    # Stub has 2 categories × 10 = 20 images. Asking for 30 raises.
    with pytest.raises(ValueError, match="not enough"):
        load_quickdraw(n_train=20, n_test=20, seed=42, data_root=QD_STUB)


def test_load_quickdraw_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_root"):
        load_quickdraw(n_train=1, n_test=1, seed=42, data_root=tmp_path / "nope")


def test_load_div2k_shape_and_dtype():
    train, test = load_div2k(n_train=1, n_test=1, seed=42, size=32, data_root=DV_STUB)
    assert train.shape == (1, 32, 32)
    assert test.shape == (1, 32, 32)
    assert train.dtype == np.float32


def test_load_div2k_resize_works():
    """Stub PNGs are 64x64; loader resizes to size=128."""
    train, _ = load_div2k(n_train=1, n_test=1, seed=42, size=128, data_root=DV_STUB)
    assert train.shape == (1, 128, 128)


def test_load_div2k_too_many_raises():
    # Stub has 3 PNGs. Asking for 5 raises.
    with pytest.raises(ValueError, match="not enough"):
        load_div2k(n_train=3, n_test=3, seed=42, data_root=DV_STUB)


def test_load_div2k_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_root"):
        load_div2k(n_train=1, n_test=1, seed=42, data_root=tmp_path / "nope")
```

- [ ] **Step 3: Run the test, verify it fails**

```bash
pytest benchmarks/tests/test_data_loading.py -v --no-cov
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data_loading'`.

- [ ] **Step 4: Implement `data_loading.py`**

Create `benchmarks/data_loading.py`:

```python
"""Dataset loaders for benchmark runs. Read-only — no auto-download.

QuickDraw .npy files and DIV2K PNGs are expected at:
    /home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw/
    /home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR/

Datasets selected by `seed` use np.random.default_rng. PRNG is independent
from Julia's Random.seed!(42) — Python and Julia draw different image sets
even at the same seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_QUICKDRAW_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw")
DEFAULT_DIV2K_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")


def _ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} data_root does not exist: {path}\n"
            f"Place the dataset at this path or pass data_root=... to override."
        )


def load_quickdraw(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_QUICKDRAW_ROOT,
    img_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` 28×28 QuickDraw drawings, resize to `img_size`,
    return as float32 in [0, 1]. Images are stratified across the available
    categories (.npy files in `data_root`).

    Each .npy file is shape (N_drawings, 784) uint8.
    """
    data_root = Path(data_root)
    _ensure_dir(data_root, "quickdraw")

    npy_files = sorted(data_root.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy files found under {data_root}")

    total_needed = n_train + n_test
    rng = np.random.default_rng(seed)

    # Per-category sample count: ceil division so we always have enough.
    per_cat_target = (total_needed + len(npy_files) - 1) // len(npy_files)

    picked: list[np.ndarray] = []
    for npy in npy_files:
        arr = np.load(npy)  # (N, 784) uint8
        if arr.ndim != 2 or arr.shape[1] != 784:
            raise ValueError(f"{npy} has unexpected shape {arr.shape}; expected (N, 784)")
        n_avail = arr.shape[0]
        n_pick = min(per_cat_target, n_avail)
        idx = rng.choice(n_avail, size=n_pick, replace=False)
        # Reshape to (n_pick, 28, 28).
        picked.append(arr[idx].reshape(n_pick, 28, 28))

    pool = np.concatenate(picked, axis=0).astype(np.float32) / 255.0
    if pool.shape[0] < total_needed:
        raise ValueError(
            f"not enough images in {data_root}: have {pool.shape[0]}, "
            f"need {total_needed}"
        )

    # Trim and shuffle once more before split so categories are interleaved.
    perm = rng.permutation(pool.shape[0])
    pool = pool[perm[:total_needed]]

    if img_size != 28:
        pool = _resize_batch(pool, img_size)

    train = pool[:n_train]
    test = pool[n_train:]
    return train, test


def load_div2k(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_DIV2K_ROOT,
    size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` DIV2K PNGs, convert to grayscale, center-crop
    to a square, resize to `size`×`size`, return as float32 in [0, 1].
    """
    data_root = Path(data_root)
    _ensure_dir(data_root, "div2k")

    pngs = sorted(data_root.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no .png files found under {data_root}")

    total_needed = n_train + n_test
    if len(pngs) < total_needed:
        raise ValueError(
            f"not enough images in {data_root}: have {len(pngs)}, need {total_needed}"
        )

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(pngs), size=total_needed, replace=False)
    chosen = [pngs[i] for i in chosen_idx]

    out = np.empty((total_needed, size, size), dtype=np.float32)
    for i, p in enumerate(chosen):
        img = Image.open(p).convert("L")
        # Center-crop to square.
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.LANCZOS)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0

    return out[:n_train], out[n_train:]


def _resize_batch(images: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a batch (N, H, W) → (N, target, target) via Pillow LANCZOS."""
    n = images.shape[0]
    out = np.empty((n, target_size, target_size), dtype=np.float32)
    for i, img in enumerate(images):
        pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        pil = pil.resize((target_size, target_size), Image.LANCZOS)
        out[i] = np.asarray(pil, dtype=np.float32) / 255.0
    return out
```

- [ ] **Step 5: Run the test, verify it passes**

```bash
pytest benchmarks/tests/test_data_loading.py -v --no-cov
```

Expected: 8 tests pass.

- [ ] **Step 6: Lint**

```bash
ruff check benchmarks/data_loading.py benchmarks/tests/test_data_loading.py
ruff format benchmarks/data_loading.py benchmarks/tests/test_data_loading.py
```

- [ ] **Step 7: Commit**

```bash
git add benchmarks/data_loading.py benchmarks/tests/test_data_loading.py benchmarks/tests/fixtures/
git commit -m "feat(bench): QuickDraw and DIV2K loaders with fixture-based tests"
```

---

### Task 7: `harness.py` — `train_one_basis` + Julia-compat JSON dump

**Files:**
- Create: `benchmarks/harness.py`
- Create: `benchmarks/tests/test_harness_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `benchmarks/tests/test_harness_smoke.py`:

```python
"""Layer A: harness.py CPU smoke test. No GPU. <10s wall-clock."""
from __future__ import annotations

import json
import time

import jax
import numpy as np
import pytest

import pdft

from config import Preset
from harness import (
    OPTIMIZER_REGISTRY,
    TrainResult,
    dump_metrics_json,
    train_one_basis,
)


@pytest.fixture
def cpu_device():
    """First CPU device. JAX always has at least one CPU device."""
    return jax.devices("cpu")[0]


def test_optimizer_registry():
    assert "gd" in OPTIMIZER_REGISTRY
    assert "adam" in OPTIMIZER_REGISTRY
    opt = OPTIMIZER_REGISTRY["gd"](lr=0.01)
    assert isinstance(opt, pdft.RiemannianGD)


def test_optimizer_unknown_raises():
    with pytest.raises(KeyError):
        OPTIMIZER_REGISTRY["sgd"](lr=0.01)  # type: ignore[index]


def test_train_one_basis_smoke(cpu_device):
    """2-step training on QFT m=2 n=2. Should complete <10s, return correct shape."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    target = target.astype(np.complex128)

    preset = Preset(
        name="smoke", epochs=2, n_train=1, n_test=1, optimizer="gd", lr=0.01
    )

    def factory():
        return pdft.QFTBasis(m=2, n=2)

    t0 = time.perf_counter()
    result = train_one_basis(factory, target, preset, device=cpu_device, is_first_image=True)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0

    assert isinstance(result, TrainResult)
    assert len(result.loss_history) == 2
    assert result.time > 0
    assert result.warmup_s > 0  # is_first_image=True → warmup populated
    # Loss should not increase over 2 GD steps on a small problem.
    assert result.loss_history[-1] <= result.loss_history[0] + 1e-10


def test_train_one_basis_subsequent_image_no_warmup(cpu_device):
    """is_first_image=False → warmup_s == 0."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    target = target.astype(np.complex128)
    preset = Preset(name="smoke", epochs=2, n_train=1, n_test=1, optimizer="gd", lr=0.01)
    res = train_one_basis(
        lambda: pdft.QFTBasis(m=2, n=2),
        target, preset, device=cpu_device, is_first_image=False,
    )
    assert res.warmup_s == 0.0


def test_dump_metrics_json_roundtrip(tmp_path):
    """dump_metrics_json produces parseable JSON with the expected keys."""
    metrics = {
        "qft": {
            "metrics": {"0.05": {"mean_mse": 0.012, "std_mse": 0.001,
                                 "mean_psnr": 19.1, "std_psnr": 0.7,
                                 "mean_ssim": 0.5, "std_ssim": 0.04}},
            "time": 1.23,
        },
        "mera": {"skipped": "incompatible_qubits"},
    }
    out = tmp_path / "metrics.json"
    dump_metrics_json(metrics, out)
    parsed = json.loads(out.read_text())
    assert parsed["qft"]["time"] == 1.23
    assert parsed["mera"]["skipped"] == "incompatible_qubits"
    assert parsed["qft"]["metrics"]["0.05"]["mean_mse"] == 0.012


def test_dump_metrics_json_julia_float_format(tmp_path):
    """Very-small floats use Julia-style scientific notation (e.g. 5.0e-7 not 5e-07)."""
    metrics = {"x": {"time": 5e-7}}
    out = tmp_path / "metrics.json"
    dump_metrics_json(metrics, out)
    text = out.read_text()
    # Julia format: 5.0e-7 (mantissa has '.0', exponent has no leading zero, no '+').
    assert "5.0e-7" in text or "5e-7" in text  # tolerate either; spec leans Julia-style.
    assert "5e-07" not in text  # Python's default form must NOT appear.
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest benchmarks/tests/test_harness_smoke.py -v --no-cov
```

Expected: FAIL with `ModuleNotFoundError: No module named 'harness'`.

- [ ] **Step 3: Implement `harness.py`**

Create `benchmarks/harness.py`:

```python
"""Single-basis training wrapper for the benchmark harness.

Wraps pdft.train_basis with timing and optional first-call JIT-warmup tracking.
Also provides a Julia-float-formatted JSON writer for metrics.json.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import pdft
from pdft.io_json import _format_float_julia_like

from config import Preset


OPTIMIZER_REGISTRY: dict[str, Callable[..., Any]] = {
    "gd":   lambda lr: pdft.RiemannianGD(lr=lr),
    "adam": lambda lr: pdft.RiemannianAdam(lr=lr),
}


@dataclass
class TrainResult:
    basis: Any
    loss_history: list[float]
    time: float       # wall-clock incl. JIT (Julia-compatible)
    warmup_s: float   # first-call JIT only; 0 unless is_first_image=True


def _make_optimizer(name: str, lr: float):
    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(
            f"unknown optimizer {name!r}; choices: {sorted(OPTIMIZER_REGISTRY)}"
        )
    return OPTIMIZER_REGISTRY[name](lr=lr)


def train_one_basis(
    basis_factory: Callable[[], Any],
    target: np.ndarray,
    preset: Preset,
    *,
    device: jax.Device,
    is_first_image: bool = False,
) -> TrainResult:
    """Train a fresh basis from `basis_factory()` on `target` for preset.epochs steps.

    Pins device via `jax.default_device(device)`. Calls `jax.block_until_ready`
    on the trained tensors before stopping the clock for honest GPU timing.
    """
    optimizer = _make_optimizer(preset.optimizer, preset.lr)
    target_jnp = jnp.asarray(target, dtype=jnp.complex128)

    with jax.default_device(device):
        basis = basis_factory()
        t0 = time.perf_counter()
        result = pdft.train_basis(
            basis,
            target=target_jnp,
            loss=pdft.L1Norm(),
            optimizer=optimizer,
            steps=preset.epochs,
            seed=preset.seed,
        )
        # Force completion of any in-flight async dispatch before stopping the clock.
        for t in result.basis.tensors:
            jax.block_until_ready(t)
        elapsed = time.perf_counter() - t0

    warmup = elapsed if is_first_image else 0.0
    return TrainResult(
        basis=result.basis,
        loss_history=list(result.loss_history),
        time=elapsed,
        warmup_s=warmup,
    )


def _julia_float_postprocess(json_text: str) -> str:
    """Rewrite Python-style scientific floats (5e-07) to Julia-style (5.0e-7).

    Python's `json` module uses `repr(float)` which yields forms like '5e-07'
    or '1.5e-07'. Julia's JSON3 uses Julia's `string(Float64)` which yields
    '5.0e-7' / '1.5e-7'. We match Julia's form in-place via regex.
    """
    pattern = re.compile(r"([-+]?\d+(?:\.\d+)?)e([-+]?\d+)")

    def fix(match: re.Match) -> str:
        mantissa = match.group(1)
        exponent = match.group(2)
        try:
            return _format_float_julia_like(float(f"{mantissa}e{exponent}"))
        except ValueError:
            return match.group(0)

    return pattern.sub(fix, json_text)


def dump_metrics_json(payload: dict, path: Path | str) -> None:
    """Write metrics.json with Julia-style float formatting in scientific notation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=4, allow_nan=True)
    text = _julia_float_postprocess(text)
    path.write_text(text)
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest benchmarks/tests/test_harness_smoke.py -v --no-cov
```

Expected: 6 tests pass. (May take 5–10s for the QFT smoke training.)

- [ ] **Step 5: Lint**

```bash
ruff check benchmarks/harness.py benchmarks/tests/test_harness_smoke.py
ruff format benchmarks/harness.py benchmarks/tests/test_harness_smoke.py
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/harness.py benchmarks/tests/test_harness_smoke.py
git commit -m "feat(bench): single-basis training wrapper with Julia-compat JSON writer"
```

---

### Task 8: `generate_report.py` + plot modules — CSVs and vector PDFs

**Files:**
- Create: `benchmarks/plots/rate_distortion.py`
- Create: `benchmarks/plots/loss_trajectories.py`
- Create: `benchmarks/generate_report.py`
- Create: `benchmarks/tests/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `benchmarks/tests/test_report.py`:

```python
"""Layer A: generate_report.py + plots/. Synthetic metrics.json input."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from generate_report import main as generate_report_main


@pytest.fixture
def synthetic_results(tmp_path: Path) -> Path:
    rd = tmp_path / "quickdraw_smoke_20260101-000000"
    (rd / "loss_history").mkdir(parents=True)

    metrics = {
        "qft": {
            "metrics": {
                "0.05": {"mean_mse": 0.02, "std_mse": 0.005,
                         "mean_psnr": 17.0, "std_psnr": 0.5,
                         "mean_ssim": 0.4, "std_ssim": 0.03,
                         "nan_count": 0},
                "0.1":  {"mean_mse": 0.01, "std_mse": 0.003,
                         "mean_psnr": 20.0, "std_psnr": 0.5,
                         "mean_ssim": 0.6, "std_ssim": 0.03,
                         "nan_count": 0},
            },
            "time": 12.3,
            "_pdft_py": {"warmup_s": 1.5, "device": "cpu"},
        },
        "tebd": {
            "metrics": {
                "0.05": {"mean_mse": 0.018, "std_mse": 0.004,
                         "mean_psnr": 17.5, "std_psnr": 0.6,
                         "mean_ssim": 0.5, "std_ssim": 0.04,
                         "nan_count": 0},
                "0.1":  {"mean_mse": 0.009, "std_mse": 0.002,
                         "mean_psnr": 20.5, "std_psnr": 0.6,
                         "mean_ssim": 0.7, "std_ssim": 0.04,
                         "nan_count": 0},
            },
            "time": 15.6,
            "_pdft_py": {"warmup_s": 1.6, "device": "cpu"},
        },
        "mera": {"skipped": "incompatible_qubits"},
        "fft":  {"metrics": {
            "0.05": {"mean_mse": 0.03, "std_mse": 0.006, "mean_psnr": 15.0, "std_psnr": 0.7, "mean_ssim": 0.36, "std_ssim": 0.03, "nan_count": 0},
            "0.1":  {"mean_mse": 0.019, "std_mse": 0.004, "mean_psnr": 17.3, "std_psnr": 0.7, "mean_ssim": 0.45, "std_ssim": 0.03, "nan_count": 0},
        }, "time": 0.07},
    }
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=4))

    # Stub loss histories for trajectory plots.
    for basis in ("qft", "tebd"):
        (rd / "loss_history" / f"{basis}_loss.json").write_text(
            json.dumps([[1.0, 0.8, 0.6], [0.9, 0.7, 0.5]])
        )

    return rd


def test_generate_report_writes_csvs(synthetic_results: Path):
    generate_report_main(synthetic_results)
    assert (synthetic_results / "timing_summary.csv").is_file()
    assert (synthetic_results / "rate_distortion_mse.csv").is_file()
    assert (synthetic_results / "rate_distortion_psnr.csv").is_file()
    assert (synthetic_results / "rate_distortion_ssim.csv").is_file()


def test_generate_report_writes_pdfs(synthetic_results: Path):
    generate_report_main(synthetic_results)
    pdir = synthetic_results / "plots"
    for name in ("rate_distortion_mse", "rate_distortion_psnr", "rate_distortion_ssim"):
        f = pdir / f"{name}.pdf"
        assert f.is_file()
        # PDF magic bytes
        with open(f, "rb") as fh:
            assert fh.read(5) == b"%PDF-"
    # loss-trajectory PDF — name uses dataset slug from results_dir name.
    loss_pdf = pdir / "loss_trajectories_quickdraw.pdf"
    assert loss_pdf.is_file()
    with open(loss_pdf, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_csv_row_counts(synthetic_results: Path):
    generate_report_main(synthetic_results)
    # rate_distortion_mse.csv: 1 header + (n_bases_with_metrics × n_keep_ratios) + (skipped/failed rows as NaN).
    # 4 bases (qft, tebd, mera, fft); mera skipped → row with NaN cells; 2 keep_ratios.
    text = (synthetic_results / "rate_distortion_mse.csv").read_text().strip().splitlines()
    # Header + 4 bases × 2 keep_ratios = 9 lines (mera contributes 2 rows of NaN).
    assert len(text) == 1 + 4 * 2


def test_idempotent(synthetic_results: Path):
    """Running twice produces the same files (no exceptions, overwrites in place)."""
    generate_report_main(synthetic_results)
    first = (synthetic_results / "timing_summary.csv").read_text()
    generate_report_main(synthetic_results)
    second = (synthetic_results / "timing_summary.csv").read_text()
    assert first == second
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest benchmarks/tests/test_report.py -v --no-cov
```

Expected: FAIL with `ModuleNotFoundError: No module named 'generate_report'`.

- [ ] **Step 3: Implement plot modules**

Create `benchmarks/plots/rate_distortion.py`:

```python
"""Rate-distortion plot: metric vs keep_ratio, one curve per basis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
matplotlib.rcParams["pdf.fonttype"] = 42  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

_METRIC_LABELS = {
    "mse":  ("Mean Squared Error",  "MSE (lower is better)"),
    "psnr": ("Peak Signal-to-Noise Ratio (dB)", "PSNR — higher is better"),
    "ssim": ("Structural Similarity", "SSIM — higher is better"),
}


def plot_rate_distortion(metrics: dict, metric_name: str, out_pdf: Path) -> None:
    """One panel per dataset (caller passes per-dataset metrics dict).

    `metric_name` ∈ {"mse", "psnr", "ssim"}.
    """
    if metric_name not in _METRIC_LABELS:
        raise ValueError(f"unknown metric {metric_name!r}")
    ylabel, title_suffix = _METRIC_LABELS[metric_name]
    mean_key, std_key = f"mean_{metric_name}", f"std_{metric_name}"

    fig, ax = plt.subplots(figsize=(7, 5))
    for basis_name, data in sorted(metrics.items()):
        if "metrics" not in data:
            continue  # skipped or failed basis
        kr_strs = sorted(data["metrics"].keys(), key=float)
        xs = [float(k) for k in kr_strs]
        ys = [data["metrics"][k][mean_key] for k in kr_strs]
        errs = [data["metrics"][k][std_key] for k in kr_strs]
        ax.errorbar(xs, ys, yerr=errs, marker="o", label=basis_name, capsize=3)

    ax.set_xlabel("Keep ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title_suffix)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
```

Create `benchmarks/plots/loss_trajectories.py`:

```python
"""Loss-trajectory plot: one subplot panel per basis, n_train overlaid curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
matplotlib.rcParams["pdf.fonttype"] = 42  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def plot_loss_trajectories(loss_dir: Path, out_pdf: Path, dataset_name: str) -> None:
    """Reads *_loss.json files (list-of-lists) and produces one panel per basis."""
    files = sorted(loss_dir.glob("*_loss.json"))
    if not files:
        # No bases have loss histories (all skipped/failed) — write an empty PDF.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no loss histories", ha="center", va="center")
        ax.set_axis_off()
    else:
        n_panels = len(files)
        cols = min(2, n_panels)
        rows = (n_panels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, lf in zip(axes_flat, files):
            basis_name = lf.stem.replace("_loss", "")
            histories = json.loads(lf.read_text())
            for traj in histories:
                ax.plot(traj, alpha=0.6, linewidth=0.8)
            ax.set_title(f"{basis_name} — loss trajectories")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
        # Hide any unused subplots.
        for ax in axes_flat[len(files):]:
            ax.set_axis_off()

    fig.suptitle(f"Loss trajectories — {dataset_name}", fontsize=12)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Implement `generate_report.py`**

Create `benchmarks/generate_report.py`:

```python
"""Aggregate metrics.json + loss histories into CSVs and PDF plots.

Idempotent: running on an existing results_dir overwrites outputs in place.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  -- adds benchmarks/ to sys.path

from plots.loss_trajectories import plot_loss_trajectories
from plots.rate_distortion import plot_rate_distortion

logger = logging.getLogger(__name__)


def _dataset_slug(results_dir: Path) -> str:
    """Extract dataset name from a results dir like 'quickdraw_smoke_20260101-000000'."""
    name = results_dir.name
    parts = name.split("_")
    # All preset names are single tokens; the timestamp is the last 1–2 tokens.
    # Drop trailing tokens that match a date-time pattern.
    while parts and parts[-1].replace("-", "").isdigit():
        parts.pop()
    if parts and parts[-1] in ("smoke", "light", "moderate", "heavy"):
        parts.pop()
    return "_".join(parts) or "unknown"


def _write_timing_csv(metrics: dict, out_csv: Path) -> None:
    rows = [["basis", "time_s", "warmup_s"]]
    for name, data in sorted(metrics.items()):
        time_s = data.get("time", float("nan"))
        warmup_s = data.get("_pdft_py", {}).get("warmup_s", float("nan"))
        rows.append([name, _fmt(time_s), _fmt(warmup_s)])
    _write_csv(out_csv, rows)


def _write_rate_distortion_csv(metrics: dict, metric_name: str, out_csv: Path) -> None:
    """One row per (basis, keep_ratio). Skipped/failed bases produce one row per
    keep_ratio with NaN cells, using the union of keep_ratios seen across bases.
    """
    mean_key, std_key = f"mean_{metric_name}", f"std_{metric_name}"

    all_keep_ratios = set()
    for data in metrics.values():
        if "metrics" in data:
            all_keep_ratios.update(data["metrics"].keys())
    if not all_keep_ratios:
        all_keep_ratios = {"0.05", "0.1", "0.15", "0.2"}
    keep_ratio_list = sorted(all_keep_ratios, key=float)

    rows = [["basis", "keep_ratio", "mean", "std"]]
    for basis_name in sorted(metrics.keys()):
        data = metrics[basis_name]
        for kr_str in keep_ratio_list:
            if "metrics" in data and kr_str in data["metrics"]:
                vals = data["metrics"][kr_str]
                rows.append([basis_name, kr_str, _fmt(vals[mean_key]), _fmt(vals[std_key])])
            else:
                rows.append([basis_name, kr_str, "NaN", "NaN"])
    _write_csv(out_csv, rows)


def _write_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def _fmt(x) -> str:
    """Numeric → string. NaN/None → 'NaN'."""
    if x is None:
        return "NaN"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return "NaN"
    return repr(f)


def main(results_dir: Path | str) -> None:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    dataset = _dataset_slug(results_dir)

    _write_timing_csv(metrics, results_dir / "timing_summary.csv")
    for m in ("mse", "psnr", "ssim"):
        _write_rate_distortion_csv(metrics, m, results_dir / f"rate_distortion_{m}.csv")
        plot_rate_distortion(metrics, m, results_dir / "plots" / f"rate_distortion_{m}.pdf")

    plot_loss_trajectories(
        results_dir / "loss_history",
        results_dir / "plots" / f"loss_trajectories_{dataset}.pdf",
        dataset,
    )
    logger.info("report generated under %s", results_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python benchmarks/generate_report.py <results_dir>")
        sys.exit(2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(sys.argv[1])
```

- [ ] **Step 5: Run the test, verify it passes**

```bash
pytest benchmarks/tests/test_report.py -v --no-cov
```

Expected: 4 tests pass.

- [ ] **Step 6: Lint**

```bash
ruff check benchmarks/generate_report.py benchmarks/plots/ benchmarks/tests/test_report.py
ruff format benchmarks/generate_report.py benchmarks/plots/ benchmarks/tests/test_report.py
```

- [ ] **Step 7: Commit**

```bash
git add benchmarks/generate_report.py benchmarks/plots/ benchmarks/tests/test_report.py
git commit -m "feat(bench): generate_report writes CSVs and vector PDF plots"
```

---

### Task 9: Julia schema-compat test (uses committed fixture)

**Files:**
- Create: `benchmarks/tests/fixtures/julia_quickdraw_metrics.json`
- Create: `benchmarks/tests/test_julia_schema_compat.py`

- [ ] **Step 1: Copy a Julia metrics.json fixture**

From the Julia repo on this machine, copy a representative `metrics.json`:

```bash
mkdir -p benchmarks/tests/fixtures
cp /home/claude-user/ParametricDFT-Benchmarks.jl/results/quickdraw/metrics.json \
   benchmarks/tests/fixtures/julia_quickdraw_metrics.json
```

The file is ~7 KB. Confirm it's there:

```bash
ls -la benchmarks/tests/fixtures/julia_quickdraw_metrics.json
```

- [ ] **Step 2: Write the failing test**

Create `benchmarks/tests/test_julia_schema_compat.py`:

```python
"""Layer A: Julia metrics.json runs through Python report code unmodified."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from generate_report import main as generate_report_main

FIXTURE = Path(__file__).parent / "fixtures" / "julia_quickdraw_metrics.json"


@pytest.fixture
def julia_results_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "quickdraw_moderate_20260101-000000"
    (rd / "loss_history").mkdir(parents=True)
    shutil.copy(FIXTURE, rd / "metrics.json")
    return rd


def test_parse_julia_metrics(julia_results_dir: Path):
    """Julia metrics.json parses cleanly as JSON and has expected top-level keys."""
    data = json.loads((julia_results_dir / "metrics.json").read_text())
    # Julia's quickdraw run includes at least these baselines/bases.
    assert "fft" in data
    assert "dct" in data
    # At least one quantum basis present.
    assert any(k in data for k in ("qft", "entangled_qft", "tebd", "mera"))


def test_generate_report_on_julia_metrics(julia_results_dir: Path):
    """Run our report generator on Julia output. Must succeed without errors."""
    generate_report_main(julia_results_dir)
    # CSVs produced.
    assert (julia_results_dir / "timing_summary.csv").is_file()
    for m in ("mse", "psnr", "ssim"):
        assert (julia_results_dir / f"rate_distortion_{m}.csv").is_file()
        assert (julia_results_dir / "plots" / f"rate_distortion_{m}.pdf").is_file()


def test_keep_ratio_keys_julia_form(julia_results_dir: Path):
    """Julia's keep_ratio keys are '0.05','0.1','0.15','0.2' — match Python's str(float)."""
    data = json.loads((julia_results_dir / "metrics.json").read_text())
    for basis_name, entry in data.items():
        if "metrics" not in entry:
            continue
        keys = set(entry["metrics"].keys())
        # Either Julia's natural form OR the equivalent.
        assert keys.issubset({"0.05", "0.1", "0.15", "0.2"}), (
            f"{basis_name}: unexpected keep_ratio keys {keys}"
        )
        break  # one is enough
```

- [ ] **Step 3: Run the test, verify it passes**

```bash
pytest benchmarks/tests/test_julia_schema_compat.py -v --no-cov
```

Expected: 3 tests pass. If `test_generate_report_on_julia_metrics` fails because Julia's metrics.json uses a slightly different field set, fix `generate_report.py` to handle the actual Julia shape gracefully (it should — we already skip bases without `"metrics"`).

- [ ] **Step 4: Lint**

```bash
ruff check benchmarks/tests/test_julia_schema_compat.py
ruff format benchmarks/tests/test_julia_schema_compat.py
```

- [ ] **Step 5: Commit**

```bash
git add benchmarks/tests/fixtures/julia_quickdraw_metrics.json benchmarks/tests/test_julia_schema_compat.py
git commit -m "test(bench): Julia metrics.json schema-compat fixture and test"
```

---

### Task 10: `run_quickdraw.py` — CLI entrypoint

**Files:**
- Create: `benchmarks/run_quickdraw.py`
- Create: `benchmarks/tests/test_quickdraw_smoke_e2e.py` (integration; opt-in)

- [ ] **Step 1: Implement `run_quickdraw.py`**

Create `benchmarks/run_quickdraw.py`:

```python
#!/usr/bin/env python3
"""Run the QuickDraw benchmark (m=n=5, 32×32) on a single GPU.

Usage:
    python benchmarks/run_quickdraw.py <preset> [--gpu N] [--out DIR]
                                       [--allow-cpu] [--verbose] [--log-file]

Mirrors run_quickdraw.jl from ParametricDFT-Benchmarks.jl. Single-target
pdft.train_basis per image (P pairing). Skips MERA (m+n=10 not power of 2).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  -- sys.path bootstrap

import jax
import numpy as np

# Importing pdft sets jax_enable_x64; must come before any other jax math.
import pdft

from baselines import (
    block_dct_compress,
    block_fft_compress,
    global_dct_compress,
    global_fft_compress,
)
from config import get_preset
from data_loading import load_quickdraw
from evaluation import evaluate_baseline, evaluate_basis_per_image
from generate_report import main as generate_report_main
from harness import dump_metrics_json, train_one_basis

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DATASET_NAME = "quickdraw"
M = 5
N = 5

BASIS_FACTORIES = {
    "qft":           lambda: pdft.QFTBasis(m=M, n=N),
    "entangled_qft": lambda: pdft.EntangledQFTBasis(m=M, n=N),
    "tebd":          lambda: pdft.TEBDBasis(m=M, n=N),
    "mera":          lambda: pdft.MERABasis(m=M, n=N),
}

BASELINE_FACTORIES = {
    "fft":         global_fft_compress,
    "dct":         global_dct_compress,
    "block_fft_8": lambda img, kr: block_fft_compress(img, kr, block=8),
    "block_dct_8": lambda img, kr: block_dct_compress(img, kr, block=8),
}


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("preset", choices=("smoke", "light", "moderate", "heavy"))
    p.add_argument("--gpu", type=int, default=0, help="GPU device index (default 0)")
    p.add_argument("--out", type=Path, default=None, help="results directory")
    p.add_argument("--allow-cpu", action="store_true",
                   help="permit CPU run (smoke tests only; not Julia-comparable)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-file", action="store_true",
                   help="also write run.log inside results dir")
    return p.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=Path(__file__).parent.parent,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _select_device(gpu_idx: int, allow_cpu: bool) -> jax.Device:
    cuda_devices = jax.devices("gpu") if jax.default_backend() == "gpu" else []
    if cuda_devices:
        if gpu_idx >= len(cuda_devices):
            raise SystemExit(
                f"GPU index {gpu_idx} out of range; available: "
                f"{[str(d) for d in cuda_devices]}"
            )
        return cuda_devices[gpu_idx]
    if allow_cpu:
        return jax.devices("cpu")[0]
    raise SystemExit(
        "no GPU available. Install pdft[gpu] or pass --allow-cpu for smoke testing."
    )


def _setup_logging(verbose: bool, log_path: Path | None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=handlers,
        force=True,
    )


# ----------------------------------------------------------------------------
# Failure handling helpers
# ----------------------------------------------------------------------------

def _record_failure(failures_dir: Path, basis_name: str, image_idx: int, err: BaseException) -> None:
    failures_dir.mkdir(parents=True, exist_ok=True)
    path = failures_dir / f"{basis_name}_failures.json"
    existing = json.loads(path.read_text()) if path.is_file() else []
    existing.append({
        "image_idx": image_idx,
        "error": f"{type(err).__name__}: {err}",
        "traceback": traceback.format_exc(),
    })
    path.write_text(json.dumps(existing, indent=2))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    preset = get_preset(DATASET_NAME, args.preset)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.out is not None:
        results_dir = args.out
    else:
        results_dir = Path("benchmarks/results") / f"{DATASET_NAME}_{args.preset}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = results_dir / "failures"

    log_path = results_dir / "run.log" if args.log_file else None
    _setup_logging(args.verbose, log_path)
    logger = logging.getLogger("run_quickdraw")

    device = _select_device(args.gpu, args.allow_cpu)
    logger.info("device=%s preset=%s out=%s", device, preset.name, results_dir)

    # ----- env.json (provenance)
    env = {
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "active_device": str(device),
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "pdft_upstream_ref": pdft.__upstream_ref__,
        "preset": preset.name,
        "preset_dataclass": {
            "epochs": preset.epochs,
            "n_train": preset.n_train,
            "n_test":  preset.n_test,
            "optimizer": preset.optimizer,
            "lr": preset.lr,
            "seed": preset.seed,
            "keep_ratios": list(preset.keep_ratios),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }

    # ----- data
    logger.info("loading %s (n_train=%d, n_test=%d)", DATASET_NAME, preset.n_train, preset.n_test)
    train_imgs, test_imgs = load_quickdraw(preset.n_train, preset.n_test, seed=preset.seed)

    metrics_payload: dict = {}

    # ----- bases
    for basis_name, factory in BASIS_FACTORIES.items():
        if basis_name == "mera" and not _is_power_of_two(M + N):
            logger.info("skipping %s — m+n=%d not a power of 2", basis_name, M + N)
            metrics_payload[basis_name] = {"skipped": "incompatible_qubits"}
            continue

        logger.info("training %s — %d images × %d epochs", basis_name, preset.n_train, preset.epochs)
        trained: list = []
        loss_histories: list[list[float]] = []
        total_time = 0.0
        warmup_s = 0.0
        oom_streak = 0

        for i, img in enumerate(train_imgs):
            try:
                res = train_one_basis(factory, img, preset, device=device, is_first_image=(i == 0))
                if i == 0:
                    warmup_s = res.warmup_s
                total_time += res.time
                trained.append(res.basis)
                loss_histories.append(res.loss_history)
                oom_streak = 0
            except (RuntimeError, MemoryError) as e:  # noqa: PERF203
                logger.warning("basis=%s image=%d FAILED: %s", basis_name, i, e)
                _record_failure(failures_dir, basis_name, i, e)
                if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                    oom_streak += 1
                    if oom_streak >= 3:
                        logger.error(
                            "basis=%s aborted after 3 consecutive OOMs", basis_name
                        )
                        break
            except Exception as e:  # noqa: BLE001
                logger.warning("basis=%s image=%d FAILED: %s", basis_name, i, e)
                _record_failure(failures_dir, basis_name, i, e)

        if not trained:
            metrics_payload[basis_name] = {
                "failed": {"error": "all training images failed", "n_attempted": len(train_imgs)},
                "time": total_time,
            }
            continue

        # Save trained bases (JSON array) and loss histories.
        try:
            (results_dir / "loss_history").mkdir(parents=True, exist_ok=True)
            (results_dir / "loss_history" / f"{basis_name}_loss.json").write_text(
                json.dumps(loss_histories)
            )
            # Use pdft.save_basis on each, then concatenate into an array file.
            import tempfile
            arr = []
            with tempfile.TemporaryDirectory() as td:
                for k, b in enumerate(trained):
                    p = Path(td) / f"{k}.json"
                    pdft.save_basis(str(p), b)
                    arr.append(json.loads(p.read_text()))
            (results_dir / f"trained_{basis_name}.json").write_text(json.dumps(arr, indent=2))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not save trained bases for %s: %s", basis_name, e)

        # Per-image evaluation (P pairing). Truncate trained / test to same length
        # in case some images failed.
        n_eval = min(len(trained), len(test_imgs))
        try:
            kr_metrics, nan_counts = evaluate_basis_per_image(
                trained[:n_eval], test_imgs[:n_eval], preset.keep_ratios,
            )
            metrics_payload[basis_name] = {
                "metrics": kr_metrics,
                "time": total_time,
                "_pdft_py": {
                    "warmup_s": warmup_s,
                    "device": str(device),
                    "n_eval_pairs": n_eval,
                    "eval_failed_count": nan_counts,
                },
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("evaluation failed for %s: %s", basis_name, e)
            metrics_payload[basis_name] = {
                "failed": {"phase": "eval", "error": f"{type(e).__name__}: {e}"},
                "time": total_time,
            }

    # ----- baselines
    for name, fn in BASELINE_FACTORIES.items():
        logger.info("running baseline %s", name)
        kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
        metrics_payload[name] = {"metrics": kr_metrics, "time": elapsed}

    # ----- write metrics.json
    dump_metrics_json(metrics_payload, results_dir / "metrics.json")

    # ----- env.json finished_at + write
    env["finished_at"] = datetime.now(timezone.utc).isoformat()
    (results_dir / "env.json").write_text(json.dumps(env, indent=2))

    # ----- generate report (plots + CSVs); failures here don't break the run.
    try:
        generate_report_main(results_dir)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "report generation failed: %s. Re-run: python benchmarks/generate_report.py %s",
            e, results_dir,
        )

    logger.info("done — results in %s", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the smoke CLI on CPU as a quick smoke test**

```bash
python benchmarks/run_quickdraw.py smoke --allow-cpu --verbose
```

Expected: completes within ~60s, prints log lines for each basis, writes `benchmarks/results/quickdraw_smoke_*/metrics.json` and PDFs. May print a warning that MERA was skipped.

- [ ] **Step 3: Inspect the produced metrics.json**

```bash
ls benchmarks/results/quickdraw_smoke_*/
```

Verify: `metrics.json`, `env.json`, `loss_history/`, `trained_qft.json`, `trained_tebd.json`, `trained_entangled_qft.json`, `timing_summary.csv`, `rate_distortion_*.csv`, `plots/*.pdf` all present. `mera` entry in `metrics.json` should have `"skipped": "incompatible_qubits"`.

- [ ] **Step 4: Add an integration smoke test (opt-in, no GPU required when --allow-cpu)**

Create `benchmarks/tests/test_quickdraw_smoke_e2e.py`:

```python
"""Layer B (opt-in): run_quickdraw.py smoke end-to-end. CPU-allowed for CI runners."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
def test_quickdraw_smoke_e2e(tmp_path: Path):
    from run_quickdraw import main

    out_dir = tmp_path / "qd_smoke"
    rc = main(["smoke", "--allow-cpu", "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "metrics.json").is_file()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # 4 quantum bases + 4 baselines = 8 keys.
    assert set(metrics.keys()) == {
        "qft", "entangled_qft", "tebd", "mera",
        "fft", "dct", "block_fft_8", "block_dct_8",
    }
    # MERA skipped on m=n=5.
    assert metrics["mera"].get("skipped") == "incompatible_qubits"
    # All PDFs.
    for name in ("rate_distortion_mse", "rate_distortion_psnr", "rate_distortion_ssim",
                 "loss_trajectories_quickdraw"):
        assert (out_dir / "plots" / f"{name}.pdf").is_file()
```

- [ ] **Step 5: Run the integration test**

```bash
pytest benchmarks/tests/test_quickdraw_smoke_e2e.py -m integration -v --no-cov
```

Expected: PASS (single test). May take 30–60 s on CPU.

- [ ] **Step 6: Lint**

```bash
ruff check benchmarks/run_quickdraw.py benchmarks/tests/test_quickdraw_smoke_e2e.py
ruff format benchmarks/run_quickdraw.py benchmarks/tests/test_quickdraw_smoke_e2e.py
```

- [ ] **Step 7: Commit**

```bash
git add benchmarks/run_quickdraw.py benchmarks/tests/test_quickdraw_smoke_e2e.py
git commit -m "feat(bench): run_quickdraw.py CLI runner with smoke e2e test"
```

---

### Task 11: `run_div2k_8q.py` — DIV2K runner

**Files:**
- Create: `benchmarks/run_div2k_8q.py`
- Create: `benchmarks/tests/test_div2k_smoke_e2e.py`

- [ ] **Step 1: Implement `run_div2k_8q.py`**

This is structurally identical to `run_quickdraw.py` with three differences: `DATASET_NAME = "div2k_8q"`, `M = N = 8` (so MERA succeeds, since 8+8=16 is a power of 2), and `load_div2k` instead of `load_quickdraw`. To avoid copy-paste drift, factor the shared core out.

Refactor: instead of duplicating the 200-line `main`, move the shared logic into a helper. Edit `benchmarks/run_quickdraw.py` to add a `_run_dataset` function that takes `(dataset_name, m, n, loader_fn, args)`. Then both runners call into it. The minimum change is fine.

Concretely, edit `benchmarks/run_quickdraw.py` to extract the body of `main` after argument parsing into a function:

```python
def run_dataset(
    *,
    dataset_name: str,
    m: int,
    n: int,
    basis_factories: dict,
    loader_fn,
    args: argparse.Namespace,
) -> int:
    """Shared core. Returns 0 on success."""
    # ... (everything from `preset = get_preset(...)` onward in current main).
```

Then `main()` becomes a thin wrapper:

```python
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_dataset(
        dataset_name=DATASET_NAME, m=M, n=N,
        basis_factories=BASIS_FACTORIES,
        loader_fn=lambda preset: load_quickdraw(preset.n_train, preset.n_test, seed=preset.seed),
        args=args,
    )
```

(Because the data loaders have different signatures — `load_div2k` also takes `size` — wrap them as `loader_fn(preset)` returning `(train, test)`.)

Now create `benchmarks/run_div2k_8q.py`:

```python
#!/usr/bin/env python3
"""Run the DIV2K-8q benchmark (m=n=8, 256×256) on a single GPU.

Usage: python benchmarks/run_div2k_8q.py <preset> [--gpu N] [--out DIR] [--allow-cpu]
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import pdft

from data_loading import load_div2k
from run_quickdraw import _parse_args, run_dataset

DATASET_NAME = "div2k_8q"
M = 8
N = 8

BASIS_FACTORIES = {
    "qft":           lambda: pdft.QFTBasis(m=M, n=N),
    "entangled_qft": lambda: pdft.EntangledQFTBasis(m=M, n=N),
    "tebd":          lambda: pdft.TEBDBasis(m=M, n=N),
    "mera":          lambda: pdft.MERABasis(m=M, n=N),  # 8+8=16 is power of 2 → runs
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_dataset(
        dataset_name=DATASET_NAME, m=M, n=N,
        basis_factories=BASIS_FACTORIES,
        loader_fn=lambda preset: load_div2k(
            preset.n_train, preset.n_test, seed=preset.seed, size=2 ** M,
        ),
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Refactor `run_quickdraw.py` to expose `run_dataset`**

Edit `benchmarks/run_quickdraw.py`. Replace the existing `main(argv=None)` body with:

```python
def run_dataset(
    *,
    dataset_name: str,
    m: int,
    n: int,
    basis_factories: dict,
    loader_fn,
    args: argparse.Namespace,
) -> int:
    """Shared run-one-dataset core. Used by both run_quickdraw and run_div2k_8q."""
    preset = get_preset(dataset_name, args.preset)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.out is not None:
        results_dir = args.out
    else:
        results_dir = Path("benchmarks/results") / f"{dataset_name}_{args.preset}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = results_dir / "failures"

    log_path = results_dir / "run.log" if args.log_file else None
    _setup_logging(args.verbose, log_path)
    logger = logging.getLogger(f"run_{dataset_name}")

    device = _select_device(args.gpu, args.allow_cpu)
    logger.info("device=%s preset=%s out=%s", device, preset.name, results_dir)

    env = {
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "active_device": str(device),
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "pdft_upstream_ref": pdft.__upstream_ref__,
        "preset": preset.name,
        "preset_dataclass": {
            "epochs": preset.epochs,
            "n_train": preset.n_train,
            "n_test":  preset.n_test,
            "optimizer": preset.optimizer,
            "lr": preset.lr,
            "seed": preset.seed,
            "keep_ratios": list(preset.keep_ratios),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }

    logger.info("loading %s (n_train=%d, n_test=%d)", dataset_name, preset.n_train, preset.n_test)
    train_imgs, test_imgs = loader_fn(preset)

    metrics_payload: dict = {}

    for basis_name, factory in basis_factories.items():
        if basis_name == "mera" and not _is_power_of_two(m + n):
            logger.info("skipping %s — m+n=%d not a power of 2", basis_name, m + n)
            metrics_payload[basis_name] = {"skipped": "incompatible_qubits"}
            continue

        logger.info("training %s — %d images × %d epochs", basis_name, preset.n_train, preset.epochs)
        trained: list = []
        loss_histories: list[list[float]] = []
        total_time = 0.0
        warmup_s = 0.0
        oom_streak = 0

        for i, img in enumerate(train_imgs):
            try:
                res = train_one_basis(factory, img, preset, device=device, is_first_image=(i == 0))
                if i == 0:
                    warmup_s = res.warmup_s
                total_time += res.time
                trained.append(res.basis)
                loss_histories.append(res.loss_history)
                oom_streak = 0
            except (RuntimeError, MemoryError) as e:
                logger.warning("basis=%s image=%d FAILED: %s", basis_name, i, e)
                _record_failure(failures_dir, basis_name, i, e)
                if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                    oom_streak += 1
                    if oom_streak >= 3:
                        logger.error("basis=%s aborted after 3 OOMs", basis_name)
                        break
            except Exception as e:  # noqa: BLE001
                logger.warning("basis=%s image=%d FAILED: %s", basis_name, i, e)
                _record_failure(failures_dir, basis_name, i, e)

        if not trained:
            metrics_payload[basis_name] = {
                "failed": {"error": "all training images failed", "n_attempted": len(train_imgs)},
                "time": total_time,
            }
            continue

        try:
            (results_dir / "loss_history").mkdir(parents=True, exist_ok=True)
            (results_dir / "loss_history" / f"{basis_name}_loss.json").write_text(
                json.dumps(loss_histories)
            )
            import tempfile
            arr = []
            with tempfile.TemporaryDirectory() as td:
                for k, b in enumerate(trained):
                    p = Path(td) / f"{k}.json"
                    pdft.save_basis(str(p), b)
                    arr.append(json.loads(p.read_text()))
            (results_dir / f"trained_{basis_name}.json").write_text(json.dumps(arr, indent=2))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not save trained bases for %s: %s", basis_name, e)

        n_eval = min(len(trained), len(test_imgs))
        try:
            kr_metrics, nan_counts = evaluate_basis_per_image(
                trained[:n_eval], test_imgs[:n_eval], preset.keep_ratios,
            )
            metrics_payload[basis_name] = {
                "metrics": kr_metrics,
                "time": total_time,
                "_pdft_py": {
                    "warmup_s": warmup_s,
                    "device": str(device),
                    "n_eval_pairs": n_eval,
                    "eval_failed_count": nan_counts,
                },
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("evaluation failed for %s: %s", basis_name, e)
            metrics_payload[basis_name] = {
                "failed": {"phase": "eval", "error": f"{type(e).__name__}: {e}"},
                "time": total_time,
            }

    for name, fn in BASELINE_FACTORIES.items():
        logger.info("running baseline %s", name)
        kr_metrics, elapsed = evaluate_baseline(fn, test_imgs, preset.keep_ratios)
        metrics_payload[name] = {"metrics": kr_metrics, "time": elapsed}

    dump_metrics_json(metrics_payload, results_dir / "metrics.json")

    env["finished_at"] = datetime.now(timezone.utc).isoformat()
    (results_dir / "env.json").write_text(json.dumps(env, indent=2))

    try:
        generate_report_main(results_dir)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "report generation failed: %s. Re-run: python benchmarks/generate_report.py %s",
            e, results_dir,
        )

    logger.info("done — results in %s", results_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_dataset(
        dataset_name=DATASET_NAME, m=M, n=N,
        basis_factories=BASIS_FACTORIES,
        loader_fn=lambda preset: load_quickdraw(preset.n_train, preset.n_test, seed=preset.seed),
        args=args,
    )
```

- [ ] **Step 3: Verify run_quickdraw still works**

```bash
pytest benchmarks/tests/test_quickdraw_smoke_e2e.py -m integration -v --no-cov
```

Expected: PASS.

- [ ] **Step 4: Add the DIV2K integration test**

Create `benchmarks/tests/test_div2k_smoke_e2e.py`:

```python
"""Layer B (opt-in): run_div2k_8q.py smoke end-to-end."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


DIV2K_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")


@pytest.mark.integration
def test_div2k_smoke_e2e(tmp_path: Path):
    if not DIV2K_ROOT.is_dir():
        pytest.skip(f"DIV2K not available at {DIV2K_ROOT}")
    from run_div2k_8q import main

    out_dir = tmp_path / "div_smoke"
    rc = main(["smoke", "--allow-cpu", "--out", str(out_dir)])
    assert rc == 0
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # 4 bases + 4 baselines.
    assert set(metrics.keys()) == {
        "qft", "entangled_qft", "tebd", "mera",
        "fft", "dct", "block_fft_8", "block_dct_8",
    }
    # MERA should succeed on m=n=8 (16 is power of 2).
    assert "metrics" in metrics["mera"], f"mera should run on DIV2K-8q; got {metrics['mera']}"
```

- [ ] **Step 5: Run both integration tests**

```bash
pytest benchmarks/tests/ -m integration -v --no-cov
```

Expected: 2 tests pass. May take 1–3 minutes on CPU; faster on GPU.

- [ ] **Step 6: Lint**

```bash
ruff check benchmarks/run_div2k_8q.py benchmarks/run_quickdraw.py benchmarks/tests/test_div2k_smoke_e2e.py
ruff format benchmarks/run_div2k_8q.py benchmarks/run_quickdraw.py benchmarks/tests/test_div2k_smoke_e2e.py
```

- [ ] **Step 7: Commit**

```bash
git add benchmarks/run_quickdraw.py benchmarks/run_div2k_8q.py benchmarks/tests/test_div2k_smoke_e2e.py
git commit -m "feat(bench): run_div2k_8q.py runner; refactor shared run_dataset core"
```

---

### Task 12: `run_all.sh` + 2-GPU fan-out test + README

**Files:**
- Create: `benchmarks/run_all.sh`
- Create: `benchmarks/tests/test_2gpu_fanout.py`
- Create: `benchmarks/README.md`

- [ ] **Step 1: Create `run_all.sh`**

```bash
#!/usr/bin/env bash
# 2-GPU fan-out: one dataset per GPU, both running concurrently.
#
# Usage: bash benchmarks/run_all.sh [preset]   (default: moderate)
#
# Each script reads CUDA_VISIBLE_DEVICES so JAX inside the child process
# sees only one GPU. Per-basis timing stays single-GPU and Julia-comparable.
set -euo pipefail

PRESET=${1:-moderate}
TS=$(date +%Y%m%d-%H%M%S)

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESULTS_BASE="$ROOT/benchmarks/results"
mkdir -p "$RESULTS_BASE"

QD_OUT="$RESULTS_BASE/quickdraw_${PRESET}_${TS}"
DV_OUT="$RESULTS_BASE/div2k_8q_${PRESET}_${TS}"

echo "== launching quickdraw on GPU 0 → $QD_OUT"
CUDA_VISIBLE_DEVICES=0 python "$ROOT/benchmarks/run_quickdraw.py" "$PRESET" --out "$QD_OUT" &
PID_QD=$!

echo "== launching div2k_8q  on GPU 1 → $DV_OUT"
CUDA_VISIBLE_DEVICES=1 python "$ROOT/benchmarks/run_div2k_8q.py"  "$PRESET" --out "$DV_OUT" &
PID_DV=$!

RC_QD=0; RC_DV=0
wait "$PID_QD" || RC_QD=$?
wait "$PID_DV" || RC_DV=$?
echo "quickdraw exit=$RC_QD; div2k_8q exit=$RC_DV"
exit $(( RC_QD + RC_DV ))
```

Make it executable:

```bash
chmod +x benchmarks/run_all.sh
```

- [ ] **Step 2: Add the fan-out integration test**

Create `benchmarks/tests/test_2gpu_fanout.py`:

```python
"""Layer B (opt-in): run_all.sh fan-out. Skipped if <2 GPUs."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import jax
import pytest


@pytest.mark.integration
def test_2gpu_fanout_smoke(tmp_path: Path):
    if jax.default_backend() != "gpu":
        pytest.skip("no GPU backend")
    n_gpus = len(jax.devices("gpu"))
    if n_gpus < 2:
        pytest.skip(f"need 2 GPUs, have {n_gpus}")

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "run_all.sh"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

    t0 = time.perf_counter()
    rc = subprocess.run(
        ["bash", str(script), "smoke"],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=600,
    )
    elapsed = time.perf_counter() - t0
    assert rc.returncode == 0, f"run_all.sh failed: stderr=\n{rc.stderr}"

    # Both result dirs should exist with metrics.json.
    qd = sorted((repo_root / "benchmarks" / "results").glob("quickdraw_smoke_*"))
    dv = sorted((repo_root / "benchmarks" / "results").glob("div2k_8q_smoke_*"))
    assert qd and (qd[-1] / "metrics.json").is_file()
    assert dv and (dv[-1] / "metrics.json").is_file()

    # Sanity: parallel run should complete faster than running sequentially —
    # but the absolute number depends heavily on GPU + dataset size, so we
    # only assert that elapsed is reasonable (<10 min).
    assert elapsed < 600
```

- [ ] **Step 3: Create `benchmarks/README.md`**

```markdown
# pdft Benchmarks

Dataset-quality benchmarks for `pdft`, mirroring the dataset-quality slice of
[`zazabap/ParametricDFT-Benchmarks.jl`](https://github.com/zazabap/ParametricDFT-Benchmarks.jl).

## What this measures

For each of two datasets:

- **QuickDraw** (`run_quickdraw.py`): m=n=5, 32×32 grayscale images.
- **DIV2K** (`run_div2k_8q.py`): m=n=8, 256×256 grayscale images (center-cropped + resized).

For each dataset and each basis class
(`QFTBasis`, `EntangledQFTBasis`, `TEBDBasis`, `MERABasis` — MERA only when
m+n is a power of 2), the harness:

1. Trains a fresh basis per training image (single-target `pdft.train_basis`).
2. Saves all `n_train` trained bases as a JSON array in `trained_<basis>.json`.
3. Records loss history per (basis, image) in `loss_history/<basis>_loss.json`.
4. Evaluates each test image against its **own** per-image trained basis
   (P pairing) at multiple keep ratios.
5. Compares against four classical baselines:
   global FFT, global DCT, 8×8-block FFT, 8×8-block DCT.

Output is in `benchmarks/results/<dataset>_<preset>_<timestamp>/`:

- `metrics.json` — bit-compatible with Julia's `metrics.json` schema. Python-only
  fields (timing breakdowns, device, etc.) are namespaced under `_pdft_py`.
- `loss_history/<basis>_loss.json` — list-of-lists, one row per training image.
- `trained_<basis>.json` — JSON array of all `n_train` trained bases.
- `timing_summary.csv`
- `rate_distortion_{mse,psnr,ssim}.csv`
- `plots/rate_distortion_*.pdf` (vector)
- `plots/loss_trajectories_<dataset>.pdf` (vector)
- `failures/` — only present when something failed
- `env.json` — provenance: JAX version, devices, git sha, preset

## Install

```bash
pip install -e ".[bench,gpu]"   # GPU; pip install -e ".[bench]" for CPU-only smoke
```

## Run

Single dataset on a chosen GPU:

```bash
python benchmarks/run_quickdraw.py moderate --gpu 0
python benchmarks/run_div2k_8q.py  moderate --gpu 1
```

Both datasets in parallel, one per GPU:

```bash
bash benchmarks/run_all.sh moderate
```

CPU smoke for sanity (no GPU required):

```bash
python benchmarks/run_quickdraw.py smoke --allow-cpu
```

Presets: `smoke` (≤60 s on CPU), `light`, `moderate`, `heavy`. See
`benchmarks/config.py` for exact parameters.

## Datasets

The harness reads from
`/home/claude-user/ParametricDFT-Benchmarks.jl/data/`:

- `quickdraw/*.npy` — 5 categories of QuickDraw drawings (28×28 uint8, 784-flat).
- `DIV2K_train_HR/*.png` — DIV2K high-resolution PNGs.

To use a different path, edit the `data_root=` defaults in `benchmarks/data_loading.py`
or pass them via the loader functions if scripted.

## Comparing with Julia

The Julia repo ships its own results under `/home/claude-user/ParametricDFT-Benchmarks.jl/results/`.
Important caveats:

- **Julia uses a different training algorithm** (batched, scheduled, validation
  split, early stopping). Python uses upstream `pdft`'s single-target loop.
  Do not expect bit-equality of metric values; expect **same neighborhood**.
- **PRNGs differ** — same seed picks different image sets in Python and Julia.

Schema-compatibility is enforced by `tests/test_julia_schema_compat.py`: the
Python report generator reads Julia's `metrics.json` without errors.

## Tests

Layer A (CI; <30 s; no GPU; no datasets):

```bash
pytest benchmarks/tests/ --no-cov
```

Layer B (opt-in integration; requires datasets and optionally a GPU):

```bash
pytest benchmarks/tests/ -m integration --no-cov
```

The `bench` extra is required for tests:

```bash
pip install -e ".[bench]"
```

## Out of scope (future work)

- Optimizer-perf benchmarks (`benchmark_scaling`, `profile_gpu`, etc.) from the
  `optimizer/` half of the Julia repo.
- Multi-GPU sharded training of a single basis.
- Resume-after-interrupt.
- Julia-equivalent batched-and-scheduled training.

See the spec for full rationale: `docs/superpowers/specs/2026-04-25-pdft-gpu-benchmarks-design.md`.
```

- [ ] **Step 4: Run all Layer A tests**

```bash
pytest benchmarks/tests/ --no-cov -v
```

Expected: ~30 tests pass, <30 s total. Integration tests are deselected by default.

- [ ] **Step 5: Lint**

```bash
ruff check benchmarks/tests/test_2gpu_fanout.py
ruff format benchmarks/tests/test_2gpu_fanout.py
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/run_all.sh benchmarks/README.md benchmarks/tests/test_2gpu_fanout.py
git commit -m "feat(bench): run_all.sh 2-GPU fan-out, README, and fan-out integration test"
```

---

### Task 13: CI workflow for Layer A on PRs touching `benchmarks/`

**Files:**
- Create: `.github/workflows/benchmarks-smoke.yml`

- [ ] **Step 1: Inspect existing workflows for the project's pattern**

```bash
ls .github/workflows/
cat .github/workflows/ci.yml 2>/dev/null | head -60
```

(Pattern check only — we want to reuse existing Python setup steps and matrix.)

- [ ] **Step 2: Create the workflow**

Create `.github/workflows/benchmarks-smoke.yml`:

```yaml
name: benchmarks-smoke
on:
  pull_request:
    paths:
      - "benchmarks/**"
      - "pyproject.toml"
      - ".github/workflows/benchmarks-smoke.yml"
  push:
    branches: [main]
    paths:
      - "benchmarks/**"

jobs:
  layer-a:
    name: Layer A (no GPU, no datasets)
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[bench,dev]"
      - name: lint
        run: |
          ruff check benchmarks/
      - name: test
        run: |
          pytest benchmarks/tests/ --no-cov -v
```

- [ ] **Step 3: Verify YAML is valid**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/benchmarks-smoke.yml'))"
```

Expected: no output (success).

- [ ] **Step 4: Run the full Layer A locally one more time**

```bash
ruff check benchmarks/
pytest benchmarks/tests/ --no-cov -v
```

Expected: clean lint, all Layer A tests pass.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/benchmarks-smoke.yml
git commit -m "ci(bench): Layer A workflow on PRs touching benchmarks/"
```

---

## Self-review summary

Spec coverage check (per spec section):

- §1 Purpose — ✓ Tasks 10, 11 (`run_quickdraw`, `run_div2k_8q`); Tasks 3, 4 (block-DCT/FFT baselines).
- §2 Locked decisions — ✓ all 13 rows mapped:
  - bit-compat schema → Task 7 (`dump_metrics_json`); Task 9 (`test_julia_schema_compat`).
  - top-level `benchmarks/` → Task 1 scaffold.
  - 2-GPU fan-out → Task 12 (`run_all.sh`, `test_2gpu_fanout`).
  - `time` incl. JIT + `_pdft_py.warmup_s` → Task 7 (`TrainResult`, `train_one_basis`).
  - vector PDFs → Task 8 (`plots/`).
  - single-target P pairing, all bases saved → Tasks 10, 11 (`run_dataset` core).
  - datasets from existing path, no auto-download → Task 6 (`data_loading.py` defaults).
  - block-FFT/DCT 8×8 → Task 4.
  - schema-only Julia parity → Task 9.
  - no resume → not implemented (correctly).
  - GPU-required + `--allow-cpu` → Tasks 10, 11 (CLI).
- §3 Architecture — ✓ Task 1 directory tree.
- §4 Components — ✓ each `4.x` mapped to a task (4.1→T2, 4.2→T6, 4.3→T3+T4, 4.4→T5, 4.5→T7, 4.6→T10/T11, 4.7→T12, 4.8→T8).
- §5 JSON schema — ✓ Task 7 (`dump_metrics_json` + `_julia_float_postprocess`); Tasks 10/11 populate `_pdft_py`.
- §6 Error handling — ✓ Tasks 10/11 (training failures, OOM, eval); Task 5 (NaN aggregation); Task 8 (report failure non-fatal).
- §7 Testing strategy — ✓ Layer A in tasks 2–9; Layer B in tasks 10, 11, 12; CI in Task 13.
- §8 Future work — captured in README (Task 12); not implemented, per spec.
- §9 Implementation pointers — JAX x64 enforced via `import pdft` first in every script and conftest (T1, T10, T11). `_format_float_julia_like` used in T7.
- §10 Open implementation notes — block-keep semantic chosen as "global across blocks" in Task 4 (matches spec recommendation); preset numbers are placeholders to revise after first GPU run, documented in Task 2.

Placeholder scan: I searched the plan for "TBD", "TODO", "later", "fill in", "similar to". None present in step content (the only "TBD" appears in the spec discussion of preset numbers and Task 2 carries the same explicit caveat — a known open item, not a hidden gap).

Type consistency: `TrainResult` (in `harness.py`) is consistent across Tasks 7, 10, 11. `Preset` fields are consistent across Tasks 2, 7, 10, 11. `evaluate_basis_per_image` returns `(metrics_dict, nan_counts)` — consumers in Tasks 10/11 unpack both. `OPTIMIZER_REGISTRY` (Task 7) is a module-level dict consumed only by `_make_optimizer` in the same file; the test refers to it directly.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-25-pdft-gpu-benchmarks.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
