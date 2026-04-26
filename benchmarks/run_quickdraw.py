#!/usr/bin/env python3
"""Run the QuickDraw benchmark (m=n=5, 32×32) on a single GPU.

Usage:
    python benchmarks/run_quickdraw.py <preset> [--gpu N] [--out DIR]
                                       [--allow-cpu] [--verbose] [--log-file]

Mirrors run_quickdraw.jl from ParametricDFT-Benchmarks.jl. Trains one shared
basis per class on the full dataset via `pdft.train_basis_batched` (cosine
LR, validation split, early stopping). Skips MERA on m+n not a power of 2.
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
from evaluation import evaluate_baseline, evaluate_basis_shared
from generate_report import main as generate_report_main
from harness import dump_metrics_json, train_one_basis_batched

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DATASET_NAME = "quickdraw"
M = 5
N = 5


# Basis factories pull `seed` from the active preset at call time so the
# random init for EntangledQFT/TEBD/MERA is deterministic and breaks the
# QFT-clone collapse documented in issue #5. QFT itself is analytic.
def _make_basis_factories(preset_seed: int) -> dict:
    return {
        "qft": lambda: pdft.QFTBasis(m=M, n=N),
        "entangled_qft": lambda: pdft.EntangledQFTBasis(m=M, n=N, seed=preset_seed),
        "tebd": lambda: pdft.TEBDBasis(m=M, n=N, seed=preset_seed),
        "mera": lambda: pdft.MERABasis(m=M, n=N, seed=preset_seed),
    }


BASIS_FACTORIES = {
    "qft": lambda: pdft.QFTBasis(m=M, n=N),
    "entangled_qft": lambda: pdft.EntangledQFTBasis(m=M, n=N, seed=42),
    "tebd": lambda: pdft.TEBDBasis(m=M, n=N, seed=42),
    "mera": lambda: pdft.MERABasis(m=M, n=N, seed=42),
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
    p.add_argument("preset", choices=("smoke", "moderate", "generalized"))
    p.add_argument("--gpu", type=int, default=0, help="GPU device index (default 0)")
    p.add_argument("--out", type=Path, default=None, help="results directory")
    p.add_argument(
        "--bases",
        type=str,
        default=None,
        help="comma-separated subset of bases to train (e.g. 'qft' or 'tebd,mera'). "
        "Default: all four.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override preset batch_size (e.g. 8 on 10q for a single basis run).",
    )
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
    if args.batch_size is not None:
        from dataclasses import replace

        preset = replace(preset, batch_size=args.batch_size)

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
            "batch_size": preset.batch_size,
            "warmup_frac": preset.warmup_frac,
            "lr_peak": preset.lr_peak,
            "lr_final": preset.lr_final,
            "max_grad_norm": preset.max_grad_norm,
            "validation_split": preset.validation_split,
            "early_stopping_patience": preset.early_stopping_patience,
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
    host_bases_for_analysis: dict = {}

    # Re-bind the dataset's basis factories with the active preset's seed so
    # EntangledQFT/TEBD/MERA get deterministic random init. We rebuild factories
    # using the dataset's m, n (NOT the QuickDraw module-level M, N) — earlier
    # code mistakenly pulled QuickDraw-shaped factories for DIV2K runs.
    seeded_factories = {
        "qft": lambda: pdft.QFTBasis(m=m, n=n),
        "entangled_qft": lambda: pdft.EntangledQFTBasis(m=m, n=n, seed=preset.seed),
        "tebd": lambda: pdft.TEBDBasis(m=m, n=n, seed=preset.seed),
        "mera": lambda: pdft.MERABasis(m=m, n=n, seed=preset.seed),
    }
    seeded_factories = {k: seeded_factories[k] for k in basis_factories if k in seeded_factories}

    # Optional --bases filter (e.g. for splitting work across multiple GPUs).
    if args.bases is not None:
        wanted = [b.strip() for b in args.bases.split(",") if b.strip()]
        unknown = [b for b in wanted if b not in seeded_factories]
        if unknown:
            raise SystemExit(
                f"unknown basis name(s) in --bases: {unknown}; available: {list(seeded_factories)}"
            )
        seeded_factories = {k: seeded_factories[k] for k in wanted}
        logger.info("--bases filter active: training only %s", list(seeded_factories))

    # ----- bases (one shared basis per class, batched training)
    for basis_name, factory in seeded_factories.items():
        if basis_name == "mera" and not _is_power_of_two(m + n):
            logger.info("skipping %s — m+n=%d not a power of 2", basis_name, m + n)
            metrics_payload[basis_name] = {"skipped": "incompatible_qubits"}
            continue

        logger.info(
            "training %s — %d images × %d epochs (batch_size=%d, optimizer=%s)",
            basis_name,
            preset.n_train,
            preset.epochs,
            preset.batch_size,
            preset.optimizer,
        )
        try:
            res = train_one_basis_batched(factory, train_imgs, preset, device=device)
        except (RuntimeError, MemoryError) as e:
            logger.warning("basis=%s training FAILED: %s", basis_name, e)
            _record_failure(failures_dir, basis_name, -1, e)
            metrics_payload[basis_name] = {
                "failed": {"phase": "train", "error": f"{type(e).__name__}: {e}"},
                "time": 0.0,
            }
            continue
        except Exception as e:  # noqa: BLE001
            logger.warning("basis=%s training FAILED: %s", basis_name, e)
            _record_failure(failures_dir, basis_name, -1, e)
            metrics_payload[basis_name] = {
                "failed": {"phase": "train", "error": f"{type(e).__name__}: {e}"},
                "time": 0.0,
            }
            continue

        # Save loss history (per-step train losses + per-epoch val losses) and
        # the trained basis. trained_<basis>.json captures actual class name
        # because pdft.save_basis is QFT-only (issue #5 sub-thread).
        try:
            (results_dir / "loss_history").mkdir(parents=True, exist_ok=True)
            (results_dir / "loss_history" / f"{basis_name}_loss.json").write_text(
                json.dumps(
                    {
                        "step_losses": list(res.loss_history),
                        "val_losses": list(res.val_history),
                        "epochs_completed": res.epochs_completed,
                        "steps": res.steps,
                    }
                )
            )
            host_tensors = [jax.device_get(t) for t in res.basis.tensors]
            (results_dir / f"trained_{basis_name}.json").write_text(
                json.dumps(
                    {
                        "type": type(res.basis).__name__,
                        "m": int(res.basis.m),
                        "n": int(res.basis.n),
                        "tensors": [
                            [
                                [float(v.real), float(v.imag)]
                                for v in np.asarray(t).flatten(order="F")
                            ]
                            for t in host_tensors
                        ],
                    },
                    indent=2,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("could not save trained basis for %s: %s", basis_name, e)

        # Move basis to host once; reuse for eval and reconstruction analysis.
        host_basis = jax.tree_util.tree_map(jax.device_get, res.basis)
        host_bases_for_analysis[basis_name] = host_basis
        try:
            kr_metrics, nan_counts = evaluate_basis_shared(
                host_basis,
                test_imgs,
                preset.keep_ratios,
            )
            metrics_payload[basis_name] = {
                "metrics": kr_metrics,
                "time": res.time,
                "_pdft_py": {
                    "warmup_s": res.warmup_s,
                    "device": str(device),
                    "epochs_completed": res.epochs_completed,
                    "steps": res.steps,
                    "n_test": len(test_imgs),
                    "eval_failed_count": nan_counts,
                },
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("evaluation failed for %s: %s", basis_name, e)
            metrics_payload[basis_name] = {
                "failed": {"phase": "eval", "error": f"{type(e).__name__}: {e}"},
                "time": res.time,
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
