"""optimizer_benchmark.py — compare RiemannianGD vs RiemannianAdam on QFTBasis.

Run: python examples/optimizer_benchmark.py
"""
from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp

import pdft
from pdft.viz import TrainingHistory, plot_training_comparison


def main(out_dir: str | Path = "out") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    target = jax.random.normal(jax.random.PRNGKey(3), (4, 4)).astype(jnp.complex128)

    histories: list[TrainingHistory] = []
    for label, opt in [
        ("RiemannianGD (lr=0.01)", pdft.RiemannianGD(lr=0.01)),
        ("RiemannianAdam (lr=0.01)", pdft.RiemannianAdam(lr=0.01)),
    ]:
        basis = pdft.QFTBasis(m=2, n=2)
        print(f"Training with {label}...")
        t0 = time.perf_counter()
        result = pdft.train_basis(
            basis,
            target=target,
            loss=pdft.L1Norm(),
            optimizer=opt,
            steps=50,
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.2f}s, final loss {result.loss_history[-1]:.4f}")
        histories.append(TrainingHistory(losses=result.loss_history, label=label))

    fig_path = out / "optimizer_benchmark.png"
    plot_training_comparison(histories, output_path=fig_path,
                             title="RiemannianGD vs RiemannianAdam on QFTBasis (L1 loss)")
    print(f"wrote: {fig_path}")


if __name__ == "__main__":
    main()
