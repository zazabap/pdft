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

from analyze import analyze_reconstructions
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
    "qft": lambda: pdft.QFTBasis(m=M, n=N),
    "entangled_qft": lambda: pdft.EntangledQFTBasis(m=M, n=N),
    "tebd": lambda: pdft.TEBDBasis(m=M, n=N),
    "mera": lambda: pdft.MERABasis(m=M, n=N),
}

BASELINE_FACTORIES = {
    "fft": global_fft_compress,
    "dct": global_dct_compress,
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
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit CPU run (smoke tests only; not Julia-comparable)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--log-file",
        action="store_true",
        help="also write run.log inside results dir",
    )
    return p.parse_args(argv)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path(__file__).parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _select_device(gpu_idx: int, allow_cpu: bool) -> jax.Device:
    cuda_devices = jax.devices("gpu") if jax.default_backend() == "gpu" else []
    if cuda_devices:
        if gpu_idx >= len(cuda_devices):
            raise SystemExit(
                f"GPU index {gpu_idx} out of range; available: {[str(d) for d in cuda_devices]}"
            )
        return cuda_devices[gpu_idx]
    if allow_cpu:
        return jax.devices("cpu")[0]
    raise SystemExit("no GPU available. Install pdft[gpu] or pass --allow-cpu for smoke testing.")


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


def _record_failure(
    failures_dir: Path, basis_name: str, image_idx: int, err: BaseException
) -> None:
    failures_dir.mkdir(parents=True, exist_ok=True)
    path = failures_dir / f"{basis_name}_failures.json"
    existing = json.loads(path.read_text()) if path.is_file() else []
    existing.append(
        {
            "image_idx": image_idx,
            "error": f"{type(err).__name__}: {err}",
            "traceback": traceback.format_exc(),
        }
    )
    path.write_text(json.dumps(existing, indent=2))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def run_dataset(
    *,
    dataset_name: str,
    m: int,
    n: int,
    basis_factories: dict,
    loader_fn,
    args: argparse.Namespace,
) -> int:
    """Core benchmark runner — dataset-agnostic.

    Parameters
    ----------
    dataset_name:
        Short identifier used for logging and directory naming (e.g. "quickdraw").
    m, n:
        Row / column qubit counts.
    basis_factories:
        Mapping of basis name -> zero-argument factory callable.
    loader_fn:
        Callable(preset) -> (train_imgs, test_imgs).  Receives the resolved
        Preset so it can read n_train / n_test / seed without re-parsing args.
    args:
        Parsed CLI namespace (from _parse_args).
    """
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
            "n_test": preset.n_test,
            "optimizer": preset.optimizer,
            "lr": preset.lr,
            "seed": preset.seed,
            "keep_ratios": list(preset.keep_ratios),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }

    # ----- data
    logger.info("loading %s (n_train=%d, n_test=%d)", dataset_name, preset.n_train, preset.n_test)
    train_imgs, test_imgs = loader_fn(preset)

    metrics_payload: dict = {}
    host_bases_for_analysis: dict[str, list] = {}

    # ----- bases
    for basis_name, factory in basis_factories.items():
        if basis_name == "mera" and not _is_power_of_two(m + n):
            logger.info("skipping %s — m+n=%d not a power of 2", basis_name, m + n)
            metrics_payload[basis_name] = {"skipped": "incompatible_qubits"}
            continue

        logger.info(
            "training %s — %d images × %d epochs",
            basis_name,
            preset.n_train,
            preset.epochs,
        )
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
                        logger.error("basis=%s aborted after 3 consecutive OOMs", basis_name)
                        break
            except Exception as e:  # noqa: BLE001
                logger.warning("basis=%s image=%d FAILED: %s", basis_name, i, e)
                _record_failure(failures_dir, basis_name, i, e)

        if not trained:
            metrics_payload[basis_name] = {
                "failed": {
                    "error": "all training images failed",
                    "n_attempted": len(train_imgs),
                },
                "time": total_time,
            }
            continue

        # Save trained bases (JSON array) and loss histories.
        try:
            (results_dir / "loss_history").mkdir(parents=True, exist_ok=True)
            (results_dir / "loss_history" / f"{basis_name}_loss.json").write_text(
                json.dumps(loss_histories)
            )
            # pdft.save_basis is hardcoded to QFTBasis (Phase 2). For
            # cross-basis serialisation we dump the host-resident tensor list
            # directly with the actual class name so trained_<basis>.json is
            # honest about what was trained. If/when pdft's serialiser supports
            # all basis classes, swap this back to pdft.save_basis.
            arr = []
            for b in trained:
                host_tensors = [jax.device_get(t) for t in b.tensors]
                arr.append(
                    {
                        "type": type(b).__name__,
                        "m": int(b.m),
                        "n": int(b.n),
                        "tensors": [
                            [
                                [float(v.real), float(v.imag)]
                                for v in np.asarray(t).flatten(order="F")
                            ]
                            for t in host_tensors
                        ],
                    }
                )
            (results_dir / f"trained_{basis_name}.json").write_text(json.dumps(arr, indent=2))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not save trained bases for %s: %s", basis_name, e)

        # Per-image evaluation (P pairing). Truncate trained / test to same length
        # in case some images failed. Bases are moved to host once and reused
        # for the post-eval reconstruction analysis.
        n_eval = min(len(trained), len(test_imgs))
        host_bases = [jax.tree_util.tree_map(jax.device_get, b) for b in trained[:n_eval]]
        host_bases_for_analysis[basis_name] = host_bases
        try:
            kr_metrics, nan_counts = evaluate_basis_per_image(
                host_bases,
                test_imgs[:n_eval],
                preset.keep_ratios,
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
            e,
            results_dir,
        )

    # ----- per-image reconstruction analysis (mirrors Julia analyze_frequency_space).
    # Capped at 5 images so the PDF count stays reasonable on heavy presets.
    try:
        if host_bases_for_analysis:
            logger.info("rendering reconstruction PDFs for up to 5 test images")
            analyze_reconstructions(
                test_imgs,
                host_bases_for_analysis,
                BASELINE_FACTORIES,
                preset.keep_ratios,
                results_dir / "analysis",
                max_images=5,
            )
    except Exception as e:  # noqa: BLE001
        logger.error("reconstruction analysis failed: %s", e)

    logger.info("done — results in %s", results_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_dataset(
        dataset_name=DATASET_NAME,
        m=M,
        n=N,
        basis_factories=BASIS_FACTORIES,
        loader_fn=lambda preset: load_quickdraw(preset.n_train, preset.n_test, seed=preset.seed),
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())
