# Design: GPU Benchmark Harness for `pdft`

**Date:** 2026-04-25
**Status:** Draft (pending user review)
**Author:** brainstorming session, this repo

## 1. Purpose

Port the dataset-quality slice of [`zazabap/ParametricDFT-Benchmarks.jl`](https://github.com/zazabap/ParametricDFT-Benchmarks.jl) to the Python `pdft` package, run it on GPU, and produce results suitable for **cross-method analysis** comparing the four `pdft` quantum bases (QFT, EntangledQFT, TEBD, MERA) against classical baselines (global FFT, global DCT, 8×8 block FFT, 8×8 block DCT).

Out of scope for this design (deferred):

- The `optimizer/` half of the Julia repo (`benchmark_scaling`, `profile_gpu`, `benchmark_gradient`, `benchmark_fairness`). Captured as future work.
- Multi-GPU sharded training of a single basis (originally option C in brainstorming). Captured as future work.
- Resume capability after interruption. Captured as future work.
- Julia-equivalent batched / scheduled training (validation split, early stopping, LR warmup+decay, gradient clipping). The Julia benchmark repo uses a forked `train_basis` with these features; the Python port follows upstream `ParametricDFT.jl` main and `pdft`'s single-target `train_basis` instead. This is a deliberate divergence.

## 2. Locked decisions (from brainstorming)

| # | Decision | Notes |
|---|---|---|
| 1 | Two benchmarks: `run_quickdraw` (m=n=5, 32×32) and `run_div2k_8q` (m=n=8, 256×256) | MERA runs only on DIV2K (8 is power of 2). |
| 2 | **Bit-compatible schema** with Julia's `metrics.json` | Same field names, same nesting, Julia-style float formatting. New Python-only fields go inside a `_pdft_py` namespace per basis entry. |
| 3 | **Approach 1**: top-level `benchmarks/` directory mirroring Julia file layout, scripts (not a Python package), separate `benchmarks/tests/`. | `pdft` itself stays untouched. |
| 4 | **Single GPU per training run, two scripts in parallel** (one per dataset on each of the two RTX 3090s) via `run_all.sh` and `CUDA_VISIBLE_DEVICES`. | Per-basis timing stays single-GPU and Julia-comparable. |
| 5 | `time` field is wall-clock incl. JIT compile (Julia-compatible); `_pdft_py.warmup_s` separately reports the first-call JIT cost. | Best of both. |
| 6 | A+C analysis outputs: rate-distortion plots (MSE/PSNR/SSIM vs keep-ratio) + per-basis loss-trajectory plots. | All vector PDF, no PNG. |
| 7 | Single-target `pdft.train_basis` per image, fresh basis per image (P pairing): `basis_i` is evaluated on `test_i`. | Forces `n_train == n_test`. |
| 8 | Save **all** `n_train` trained bases per dataset (JSON array). | Richer downstream analysis. Deviates from Julia's single-basis-per-file shape. |
| 9 | Datasets read from `/home/claude-user/ParametricDFT-Benchmarks.jl/data/`. **No auto-download.** | Avoids re-download. |
| 10 | Block-DCT and block-FFT baselines added (8×8 blocks) in addition to global FFT/DCT. | Keep semantics: top-k% globally across all blocks (verified vs. Julia precedent). |
| 11 | No parity test against Julia *results*. Schema-parity only (Julia `metrics.json` parsed by Python report code). | Different training algorithms ⇒ different numbers. |
| 12 | No resume after interrupt for v1. | Future work. |
| 13 | GPU not available ⇒ hard fail. `--allow-cpu` flag exists for smoke tests only. | A CPU/GPU mix would make the comparison meaningless. |

## 3. Architecture and directory layout

```
benchmarks/                       # top-level, NOT under src/pdft/
├── README.md                     # how to run, dataset placement, GPU notes
├── config.py                     # PRESETS dict (smoke / light / moderate / heavy)
├── data_loading.py               # QuickDraw + DIV2K loaders (read-only, no download)
├── baselines.py                  # global FFT, global DCT, block-FFT(8x8), block-DCT(8x8)
├── evaluation.py                 # MSE / PSNR / SSIM at keep ratios, per-image + aggregated
├── harness.py                    # _train_one_basis(): JIT-warmup step, timing, _pdft_py extras
├── run_quickdraw.py              # mirrors run_quickdraw.jl
├── run_div2k_8q.py               # mirrors run_div2k_8q.jl
├── run_all.sh                    # 2-GPU fan-out via CUDA_VISIBLE_DEVICES
├── generate_report.py            # produces CSVs + matplotlib PDFs
├── plots/
│   ├── rate_distortion.py        # mse/psnr/ssim vs keep_ratio per dataset
│   └── loss_trajectories.py      # per-(dataset,target) overlay across bases
├── results/                      # gitignored; per-run output
└── tests/
    ├── fixtures/                 # tiny Julia metrics.json + stub datasets
    ├── test_baselines.py
    ├── test_evaluation.py
    ├── test_data_loading.py
    ├── test_harness_smoke.py
    ├── test_config.py
    ├── test_report.py
    ├── test_julia_schema_compat.py     # @pytest.mark.integration
    ├── test_quickdraw_smoke_e2e.py     # @pytest.mark.integration
    ├── test_div2k_smoke_e2e.py         # @pytest.mark.integration
    └── test_2gpu_fanout.py             # @pytest.mark.integration
```

**Architectural rules:**

- `benchmarks/` is **not** a Python package. Each `run_*.py` is standalone-runnable. Shared logic is imported from sibling files via a small `_bootstrap.py` that munges `sys.path` (matches Julia's `include("...")` pattern).
- Tests under `benchmarks/tests/` are **not** picked up by the default `pytest` invocation at repo root; they are run on demand with `pytest benchmarks/tests/`.
- `pdft` itself is unchanged. The 90% coverage gate on `--cov=pdft` is unaffected.
- New optional dependency group in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  bench = [
      "pillow>=10",
      "scikit-image>=0.22",
      "scipy>=1.11",
      "matplotlib>=3.8",
  ]
  ```

**Results directory layout** (per run):

```
results/<dataset>_<preset>_<YYYYMMDD-HHMMSS>/
├── metrics.json                 # bit-compat with Julia (see §5)
├── loss_history/
│   └── <basis>_loss.json        # list-of-lists, one row per training image
├── trained_<basis>.json         # JSON array of n_train trained bases
├── timing_summary.csv
├── rate_distortion_mse.csv
├── rate_distortion_psnr.csv
├── rate_distortion_ssim.csv
├── plots/
│   ├── rate_distortion_mse.pdf
│   ├── rate_distortion_psnr.pdf
│   ├── rate_distortion_ssim.pdf
│   └── loss_trajectories_<dataset>.pdf
├── failures/                    # only present if anything failed
│   └── <basis>_failures.json
├── env.json                     # provenance: jax version, devices, git sha, preset
└── run.log                      # only if --log-file passed
```

## 4. Components

### 4.1 `config.py`

```python
@dataclass(frozen=True)
class Preset:
    name: str
    epochs: int            # passed as `steps` to pdft.train_basis
    n_train: int           # number of target images (= n_test)
    n_test: int            # held-out images for eval; equals n_train (P pairing)
    optimizer: str         # "gd" | "adam"
    lr: float
    seed: int = 42
    keep_ratios: tuple = (0.05, 0.10, 0.15, 0.20)
```

Presets per dataset (concrete numbers calibrated to be feasible on a single RTX 3090 within target wall-clock budgets):

```python
PRESETS_QUICKDRAW = {
    "smoke":    Preset("smoke",    epochs=10,   n_train=2,   n_test=2,   optimizer="gd",   lr=0.01),
    "light":    Preset("light",    epochs=100,  n_train=10,  n_test=10,  optimizer="gd",   lr=0.01),
    "moderate": Preset("moderate", epochs=500,  n_train=50,  n_test=50,  optimizer="adam", lr=0.01),
    "heavy":    Preset("heavy",    epochs=2000, n_train=200, n_test=200, optimizer="adam", lr=0.005),
}

PRESETS_DIV2K = {
    "smoke":    Preset("smoke",    epochs=10,   n_train=2,   n_test=2,   optimizer="gd",   lr=0.01),
    "light":    Preset("light",    epochs=100,  n_train=5,   n_test=5,   optimizer="gd",   lr=0.01),
    "moderate": Preset("moderate", epochs=500,  n_train=20,  n_test=20,  optimizer="adam", lr=0.01),
    "heavy":    Preset("heavy",    epochs=2000, n_train=50,  n_test=50,  optimizer="adam", lr=0.005),
}
```

Numbers are placeholders the implementer should sanity-check on first GPU run; they may be adjusted before merge based on observed wall-clock and convergence. The shape of the dataclass is the load-bearing part.

### 4.2 `data_loading.py`

```python
def load_quickdraw(
    n_train: int, n_test: int, *, seed: int,
    data_root: Path = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw"),
) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) of shape (n, 32, 32), float32 in [0,1].
    Reads the .npy files at `data_root` (airplane.npy, apple.npy, bicycle.npy, cat.npy, dog.npy).
    Uses a stratified sample: ceil(n/n_categories) per category, then trimmed.
    """

def load_div2k(
    n_train: int, n_test: int, *, seed: int,
    size: int = 256,
    data_root: Path = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR"),
) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) of shape (n, 256, 256), float32 in [0,1], grayscale.
    Loads PNGs in lexicographic order, converts to grayscale, center-crops to a square,
    resizes to `size` × `size` via Pillow LANCZOS.
    """
```

Both use `np.random.default_rng(seed)` for image-index sampling. The PRNG is independent from Julia's `Random.seed!(42)`, so Python and Julia draw different image *sets* even at the same nominal seed. Documented; no bit-compat attempt on dataset selection.

`n_train + n_test > available` raises `ValueError` with the actual count and the path checked.

### 4.3 `baselines.py`

```python
def global_fft_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray
def global_dct_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray
def block_fft_compress(image: np.ndarray, keep_ratio: float, block: int = 8) -> np.ndarray
def block_dct_compress(image: np.ndarray, keep_ratio: float, block: int = 8) -> np.ndarray
```

Implementation:

- `global_*`: transform whole image, find indices of top-`floor(keep_ratio * N²)` magnitudes, zero everything else, inverse-transform, return real part. Mirrors `evaluation.jl::fft_compress_recover` and `dct_compress_recover` line-for-line including `partialsortperm` semantics — Python uses `np.argpartition`.
- `block_*`: split image into `(N/block)²` non-overlapping `block × block` tiles, transform each tile independently to produce a `(N/block, N/block, block, block)` tensor of coefficients, then keep top-`floor(keep_ratio * N²)` magnitudes **globally across all blocks** (matches the Julia precedent for global keep semantics, extended to blocks). Inverse-transform each tile, reassemble.

CPU-only via `numpy.fft` and `scipy.fft.dct/idct`. Image sizes 32 and 256 are both multiples of 8, so block sizing always divides cleanly.

### 4.4 `evaluation.py`

```python
def evaluate_basis_per_image(
    bases: list,                   # n_test trained bases (P pairing)
    test_images: np.ndarray,       # (n_test, H, W)
    keep_ratios: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Returns {kr_str: {mean_mse, std_mse, mean_psnr, std_psnr, mean_ssim, std_ssim}}.
    Bit-compat with Julia's metrics.json[basis_name]["metrics"] shape.

    For each (image_i, base_i) pair: round-trip the basis through pdft.io_json.save_basis /
    load_basis (forces tensors to host; sidesteps GPU scalar-indexing in compress/recover —
    same workaround as evaluation.jl:55-57). Then compress, recover, compute metrics.
    Failed (basis, image, kr) tuples record nan; aggregation uses np.nanmean / np.nanstd.
    """

def evaluate_baseline(
    fn: Callable[[np.ndarray, float], np.ndarray],
    test_images: np.ndarray,
    keep_ratios: Sequence[float],
) -> tuple[dict, float]:
    """Returns (metrics_dict, elapsed_seconds). Time is wall-clock of the eval loop only."""
```

PSNR via `skimage.metrics.peak_signal_noise_ratio(data_range=1.0)`; SSIM via `skimage.metrics.structural_similarity(data_range=1.0)`. The MSE/PSNR computation matches Julia's:

```julia
mse = mean((original .- recovered_clamped) .^ 2)
psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
```

`recovered_clamped = np.clip(np.real(recovered), 0.0, 1.0)` — matches `evaluation.jl:25`.

### 4.5 `harness.py`

```python
@dataclass
class TrainResult:
    basis: pdft.AbstractSparseBasis
    loss_history: list[float]
    time: float            # wall-clock incl. JIT (Julia-compatible)
    warmup_s: float        # first-call JIT only; populated for image_idx == 0


def train_one_basis(
    basis_factory: Callable[[], pdft.AbstractSparseBasis],
    target: np.ndarray,
    preset: Preset,
    *,
    device: jax.Device,
    is_first_image: bool = False,
) -> TrainResult:
    """Fresh basis from basis_factory(), trained on `target` for preset.epochs steps.

    - Pins device via `with jax.default_device(device):`
    - jax.block_until_ready(...) around the timed region for honest GPU timing.
    - If is_first_image=True, the entire elapsed time is also reported as warmup_s
      (this is the JIT-compile run); otherwise warmup_s = 0.
    """
```

The optimizer is selected by string in `harness.py`:

```python
_OPTIMIZERS = {
    "gd":   lambda lr: pdft.RiemannianGD(lr=lr),
    "adam": lambda lr: pdft.RiemannianAdam(lr=lr),
}
optimizer = _OPTIMIZERS[preset.optimizer](preset.lr)
```

Unknown values raise `KeyError` early in setup. Loss is `pdft.L1Norm()` (matches Julia's default in `evaluation.jl:201`). Seed is `preset.seed`.

### 4.6 `run_quickdraw.py` and `run_div2k_8q.py`

CLI: `python benchmarks/run_quickdraw.py <preset> [--gpu N] [--out DIR] [--allow-cpu] [--verbose]`.

Per-script flow (matches §3 dataflow diagram):

1. Parse CLI.
2. Setup: select device, write `env.json`, create `results_dir`.
3. Load data with `data_loading`.
4. For each basis class in `[QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis]`:
   - Skip MERA when `(m + n)` is not a power of 2 (QuickDraw: m=n=5 ⇒ skip).
   - For each training image: `train_one_basis(...)`, accumulate trained bases and loss histories.
   - Save `trained_<basis>.json` (JSON array) and `loss_history/<basis>_loss.json` (list-of-lists).
5. Per-basis evaluation: `evaluate_basis_per_image(bases, test_images, keep_ratios)`.
6. Baselines: `evaluate_baseline(...)` for each of `[global_fft, global_dct, block_fft_8, block_dct_8]`.
7. Aggregate `metrics.json` with the schema in §5.
8. Call `generate_report.main(results_dir)` for CSVs and PDFs.

Failure handling per §6.

### 4.7 `run_all.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
PRESET=${1:-moderate}
TS=$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=0 python benchmarks/run_quickdraw.py "$PRESET" \
    --out "benchmarks/results/quickdraw_${PRESET}_${TS}" &
PID_QD=$!

CUDA_VISIBLE_DEVICES=1 python benchmarks/run_div2k_8q.py "$PRESET" \
    --out "benchmarks/results/div2k_8q_${PRESET}_${TS}" &
PID_DV=$!

wait $PID_QD || echo "quickdraw exited nonzero"
wait $PID_DV || echo "div2k exited nonzero"
```

One dataset per GPU, both running concurrently. No inter-process coordination. Each `results_dir` is self-contained.

### 4.8 `generate_report.py` and `plots/`

```python
def main(results_dir: Path) -> None:
    """Reads metrics.json + loss_history/*.json, writes:
        - timing_summary.csv (basis, time_s, warmup_s)
        - rate_distortion_{mse,psnr,ssim}.csv (basis, keep_ratio, mean, std)
        - plots/rate_distortion_{mse,psnr,ssim}.pdf
        - plots/loss_trajectories_<dataset>.pdf
    Idempotent: re-running on an existing results_dir overwrites outputs without
    re-running training."""
```

Plots use matplotlib with `format="pdf"` and `mpl.rcParams["pdf.fonttype"] = 42`. Curves are not rasterized. Loss-trajectory plot has one subplot panel per basis with `n_train` overlaid loss curves (one per training image), x-axis = step, y-axis = loss.

## 5. JSON schema (`metrics.json`)

Bit-compatible with Julia's output where field names overlap. Python-only fields are namespaced under `_pdft_py`.

```json
{
  "qft": {
    "metrics": {
      "0.05": {"mean_mse": 0.0123, "std_mse": 0.0021, "mean_psnr": 19.10, "std_psnr": 0.74, "mean_ssim": 0.512, "std_ssim": 0.041},
      "0.10": { /* ... */ },
      "0.15": { /* ... */ },
      "0.20": { /* ... */ }
    },
    "time": 124.3,
    "_pdft_py": {
      "warmup_s": 1.7,
      "device": "CudaDevice(0)",
      "peak_gpu_mb": 412,
      "jax_version": "0.10.0",
      "git_sha": "abc1234",
      "eval_failed_count": {"0.05": 0, "0.10": 0, "0.15": 0, "0.20": 0}
    }
  },
  "entangled_qft": { /* ... */ },
  "tebd": { /* ... */ },
  "mera": { "skipped": "incompatible_qubits" },
  "fft": { "metrics": { /* ... */ }, "time": 0.42 },
  "dct": { "metrics": { /* ... */ }, "time": 0.35 },
  "block_fft_8": { "metrics": { /* ... */ }, "time": 0.51 },
  "block_dct_8": { "metrics": { /* ... */ }, "time": 0.44 }
}
```

Key formatting rules (cross-language hash compatibility):

- Keep-ratio keys are strings via `str(float)` matching Julia's default (`"0.05"` not `"0.050000"`).
- Float values use `pdft.io_json._format_float_julia_like` (already exists in the package).
- Field names are exact: `mean_mse`, `std_mse`, `mean_psnr`, `std_psnr`, `mean_ssim`, `std_ssim`, `metrics`, `time`.

States distinguishable by report code:

- `data["qft"]["metrics"]` is a dict ⇒ success.
- `data["qft"]["skipped"]` is a string ⇒ skipped (e.g. MERA on QuickDraw).
- `data["qft"]["failed"]` is a dict ⇒ training/eval failure; include `error` and `image_idx`.

`env.json` (Python-only, no Julia analog):

```json
{
  "jax_version": "0.10.0",
  "default_backend": "gpu",
  "devices": ["CudaDevice(0)", "CudaDevice(1)"],
  "active_device": "CudaDevice(0)",
  "git_sha": "56913c7",
  "git_branch": "feat/reliability-hardening",
  "pdft_upstream_ref": "a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef",
  "preset": "moderate",
  "preset_dataclass": { "epochs": 500, "n_train": 50, "n_test": 50, "optimizer": "adam", "lr": 0.01, "seed": 42, "keep_ratios": [0.05, 0.10, 0.15, 0.20] },
  "started_at": "2026-04-25T02:30:00Z",
  "finished_at": "2026-04-25T02:32:04Z"
}
```

## 6. Error handling

| Failure | Where | Handling |
|---|---|---|
| GPU not available | Setup | Hard fail. `--allow-cpu` flag exists for smoke tests. |
| Requested GPU id out of range | Setup | Hard fail; list available devices. |
| Dataset path missing | Step 3 | Hard fail with the path checked + how to fix. No auto-download. |
| Dataset has fewer images than `n_train + n_test` | Step 3 | Hard fail. |
| MERA on non-power-of-2 qubits | Per-basis loop | Skip with INFO log. Record `{"skipped": "incompatible_qubits"}`. |
| Per-image training raises (NaN, JAX assertion) | Step 4 | Catch; sidecar in `failures/<basis>_failures.json`; continue with next image. If all images fail for a basis: mark whole basis as `failed`. |
| GPU OOM during training | Step 4 | Same handling. After 3 consecutive OOMs for the same basis: abort the basis. |
| Eval host-roundtrip fails | Step 5 | Catch per-basis; record `{"eval_failed": True, "error": ...}`; preserve trained-basis array file for re-eval. |
| `compress`/`recover` raises on a single (basis, image, keep_ratio) | Step 5 | Record nan; aggregation uses `np.nanmean`/`np.nanstd`. Failure count in `_pdft_py.eval_failed_count`. |
| scikit-image SSIM raises on degenerate image | Step 5 | Same. SSIM nan; mse/psnr still recorded. |
| Baseline (FFT/DCT) raises | Step 6 | Hard fail. These are pure numpy and shouldn't fail. |
| Disk full / write fails | Steps 4, 5, 7 | Best-effort fallback to `/tmp/<run_id>/`; print fallback path. |
| Report generation fails | Step 8 | Catch; raw `metrics.json` already on disk. Print "re-run `python benchmarks/generate_report.py <results_dir>`". |
| `run_all.sh` parallel run: one script crashes | Shell | `wait` returns nonzero; the other keeps running. No mutual cleanup. |
| Interrupt (Ctrl-C) | Any step after 3 | Atexit hook flushes `metrics.json`. Bases trained before the interrupt are recoverable; in-flight basis is dropped. No resume; restart from scratch. |

Logging: stdlib `logging` at INFO; DEBUG via `--verbose`. Per-basis log lines include `(dataset, basis, image_idx)`. `--log-file` writes `results_dir/run.log` (off by default).

## 7. Testing strategy

### Layer A — fast property + smoke tests (CI-runnable, no GPU/datasets)

Run with `pytest benchmarks/tests/`. Excluded from default `pytest`. Excluded from `--cov=pdft`.

| File | Asserts |
|---|---|
| `test_baselines.py` | Full-keep is identity (atol 1e-10). `keep_ratio=0.5` keeps exactly `floor(0.5 * N²)` coefficients. Block transforms keep top-k% globally across blocks. |
| `test_evaluation.py` | Schema. PSNR matches scikit-image. Identical-image case (mse=0, psnr=Inf, ssim=1). nan propagation. |
| `test_data_loading.py` | Stub fixtures — shapes, dtypes, value ranges. `n_train + n_test > available` raises. Same seed → same image set. |
| `test_harness_smoke.py` | `train_one_basis(QFTBasis(2,2), ...)` on CPU, 2-step run. `len(loss_history) == preset.epochs`. Loss monotone non-increasing on 2 steps. <10s wall-clock. |
| `test_config.py` | All 4 presets load. `n_train == n_test` for every preset. `keep_ratios` valid. |
| `test_report.py` | Synthetic `metrics.json` (3 bases, 2 keep_ratios, 1 failed, 1 skipped) → all CSVs and PDFs produced. CSV row counts correct. PDFs start with `%PDF-`. |

Target wall-clock: <30s. Synthetic data only.

### Layer B — integration tests (GPU + datasets, manual)

`@pytest.mark.integration`; run with `pytest benchmarks/tests/ -m integration`.

| File | Asserts |
|---|---|
| `test_julia_schema_compat.py` | Loads a committed Julia `metrics.json` from `benchmarks/tests/fixtures/julia_quickdraw_metrics.json` (~5KB, copied from `/home/claude-user/ParametricDFT-Benchmarks.jl/results/quickdraw/`), parses it with `generate_report` code path, asserts schema OK. **Bit-compat schema test.** |
| `test_quickdraw_smoke_e2e.py` | `run_quickdraw.py smoke` end-to-end. Wall-clock <60s. `metrics.json` has 8 keys. All PDFs generated. MERA marked skipped. |
| `test_div2k_smoke_e2e.py` | `run_div2k_8q.py smoke`. Wall-clock <120s. MERA succeeds. |
| `test_2gpu_fanout.py` | `run_all.sh smoke`. Both children complete. Both `results_dir`s populated. Wall-clock < sum of individual smoke runs. Skipped if only one GPU. |

### Layer C — manual / quality runs

Documented in `benchmarks/README.md`. `moderate` and `heavy` runs are too slow for CI. README points at expected ranges from Julia's results in `/home/claude-user/ParametricDFT-Benchmarks.jl/results/` for eyeball comparison.

### CI

New workflow `.github/workflows/benchmarks-smoke.yml`: runs `pytest benchmarks/tests/` (Layer A only) on every PR touching `benchmarks/`. No GPU runner; Layer B is manual-only.

### Explicitly NOT included

- No bit-equality parity test against Julia *result values* (different training algorithm).
- No coverage gate for `benchmarks/`.
- No PRNG parity with Julia.

## 8. Future work (captured, not in scope)

- Port `optimizer/` benchmarks (`benchmark_scaling`, `profile_gpu`, `benchmark_gradient`) for ms/step CPU-vs-GPU and kernel/alloc profiling.
- Multi-GPU sharded training of a single basis (`jax.sharding`). Likely no speedup at m=n=8; flagged for measurement only.
- Per-basis fan-out *within* a dataset (4 bases on 2 GPUs concurrently) instead of dataset fan-out across GPUs.
- Resume capability: detect `trained_<basis>.json` in `results_dir` and skip re-training.
- Statistical significance tests (paired t-test / Wilcoxon) on per-image MSE/PSNR/SSIM. Better as a notebook than as harness output.
- Batched training in `pdft` itself (per CLAUDE.md "open work in #2"). If this lands, the benchmark harness can be upgraded to use it and become more directly Julia-comparable.

## 9. Implementation pointers

- **Wirtinger gradient conjugation**: not needed at the harness layer — `pdft.train_basis` handles it. (CLAUDE.md §1.)
- **Yao little-endian and compact 2×2 CP**: not touched by this work — handled inside `pdft._circuit`. (CLAUDE.md §§2-4.)
- **JAX x64 mode**: `import pdft` early in every `run_*.py` to enforce. (CLAUDE.md §5.)
- **Column-major hashes**: only relevant if `metrics.json` floats are used in cross-language hashes. They aren't (only `trained_<basis>.json` is, via `pdft.io_json` which already does this correctly).
- **Julia float formatting**: `pdft.io_json._format_float_julia_like` for any value participating in cross-language byte comparison. Use it for `metrics.json`.
- **L1-cusp horizon**: irrelevant here — we're not asserting trajectory bit-equality.
- **Phase-extractor tolerance**: irrelevant here.

## 10. Open implementation notes for the planner

- The block-keep semantics ("top-k% globally across blocks" vs. "top-k% per block") need a quick check against the Julia repo's block transform implementation if any exists. As of this writing, the upstream Julia repo only implements *global* DCT/FFT (`evaluation.jl::fft_compress_recover` / `dct_compress_recover`); block versions are a Python-only addition. Pick *one* semantic, test it, document it. Leaning toward "global across blocks" for consistency with global-keep, but per-block is also defensible (it's what JPEG-1992 does).
- Preset numbers in §4.1 are best-guess; the implementer should run `light` once on a single GPU and adjust epochs/n_train so wall-clock is in a sane range (`light` < 5 min, `moderate` < 30 min, `heavy` < 4 h).
