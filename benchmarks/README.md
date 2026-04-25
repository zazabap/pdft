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
