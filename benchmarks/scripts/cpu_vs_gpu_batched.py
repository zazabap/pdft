#!/usr/bin/env python3
"""Quick CPU vs GPU comparison for `pdft.train_basis_batched`.

Runs the same batched-training task once on CPU, once on each available
GPU. Reports wall-clock, ms/step (excluding the warmup pass), and final
loss. Intentionally small so it completes in <2 minutes on CPU.

Usage:
    python benchmarks/scripts/cpu_vs_gpu_batched.py

The script is not part of the test suite — it's a standalone tool for
ad-hoc speed checks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Bootstrap so we can import from benchmarks/ siblings if needed.
_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import numpy as np  # noqa: E402

import pdft  # noqa: E402  -- sets jax_enable_x64 before any jax math

import jax  # noqa: E402


def _make_dataset(n_images: int, m: int, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    h, w = 2**m, 2**n
    return [
        (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))).astype(np.complex128)
        for _ in range(n_images)
    ]


def _run_one(
    label: str,
    device,
    *,
    basis_cls,
    m: int,
    n: int,
    dataset,
    epochs: int,
    batch_size: int,
    optimizer: str,
    seed: int,
):
    print(f"\n== {label} on {device}")
    with jax.default_device(device):
        basis = basis_cls(m=m, n=n, seed=seed) if basis_cls is not pdft.QFTBasis else basis_cls(m=m, n=n)

        # JIT-warmup pass (single batch, single image) timed separately.
        t0 = time.perf_counter()
        warmup = pdft.train_basis_batched(
            basis,
            dataset=dataset[:1],
            loss=pdft.L1Norm(),
            epochs=1,
            batch_size=1,
            optimizer=optimizer,
            validation_split=0.0,
            early_stopping_patience=1,
            warmup_frac=0.05,
            lr_peak=0.01,
            lr_final=0.001,
            shuffle=False,
            seed=seed,
        )
        for t in warmup.basis.tensors:
            jax.block_until_ready(t)
        warmup_s = time.perf_counter() - t0

        # Full run from a fresh basis.
        basis = basis_cls(m=m, n=n, seed=seed) if basis_cls is not pdft.QFTBasis else basis_cls(m=m, n=n)
        t0 = time.perf_counter()
        result = pdft.train_basis_batched(
            basis,
            dataset=dataset,
            loss=pdft.L1Norm(),
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            validation_split=0.0,
            early_stopping_patience=1,
            warmup_frac=0.05,
            lr_peak=0.01,
            lr_final=0.001,
            shuffle=True,
            seed=seed,
        )
        for t in result.basis.tensors:
            jax.block_until_ready(t)
        elapsed = time.perf_counter() - t0

    steps = result.steps
    ms_per_step = (elapsed * 1000.0) / max(1, steps)
    final_loss = result.loss_history[-1] if result.loss_history else float("nan")
    return {
        "label": label,
        "device": str(device),
        "warmup_s": warmup_s,
        "total_s": elapsed,
        "steps": steps,
        "ms_per_step": ms_per_step,
        "final_loss": float(final_loss),
    }


BASIS_CHOICES = {
    "qft": pdft.QFTBasis,
    "entangled_qft": pdft.EntangledQFTBasis,
    "tebd": pdft.TEBDBasis,
    "mera": pdft.MERABasis,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basis", default="qft", choices=BASIS_CHOICES.keys())
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--n-train", type=int, default=10)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--optimizer", default="adam", choices=("adam", "gd"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args(argv)

    basis_cls = BASIS_CHOICES[args.basis]
    dataset = _make_dataset(args.n_train, args.m, args.n, args.seed)
    print(
        f"basis={args.basis}  m={args.m} n={args.n}  n_train={args.n_train}  "
        f"epochs={args.epochs}  batch_size={args.batch_size}  optimizer={args.optimizer}"
    )
    print(f"image size = {2**args.m}×{2**args.n}, total optimizer steps per run = "
          f"{args.epochs * ((args.n_train + args.batch_size - 1) // args.batch_size)}")

    rows = []

    cpu_dev = jax.devices("cpu")[0]
    rows.append(_run_one(
        "cpu",
        cpu_dev,
        basis_cls=basis_cls,
        m=args.m,
        n=args.n,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        seed=args.seed,
    ))

    if not args.cpu_only and jax.default_backend() == "gpu":
        for i, gpu_dev in enumerate(jax.devices("gpu")):
            rows.append(_run_one(
                f"gpu{i}",
                gpu_dev,
                basis_cls=basis_cls,
                m=args.m,
                n=args.n,
                dataset=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                optimizer=args.optimizer,
                seed=args.seed,
            ))

    print()
    print(f"{'label':8s}  {'device':24s}  {'warmup_s':>10s}  {'total_s':>10s}  "
          f"{'steps':>6s}  {'ms/step':>10s}  {'final_loss':>12s}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['label']:8s}  {r['device']:24s}  {r['warmup_s']:10.3f}  {r['total_s']:10.3f}  "
            f"{r['steps']:6d}  {r['ms_per_step']:10.2f}  {r['final_loss']:12.4f}"
        )

    if len(rows) >= 2:
        cpu_ms = next((r["ms_per_step"] for r in rows if r["label"] == "cpu"), None)
        for r in rows:
            if r["label"].startswith("gpu") and cpu_ms:
                speedup = cpu_ms / r["ms_per_step"]
                print(f"\n{r['label']} speedup vs cpu (ms/step):  {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
