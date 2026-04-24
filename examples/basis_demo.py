"""basis_demo.py — train a QFTBasis on a random 4x4 image and plot the loss.

Run: python examples/basis_demo.py
Requires: pdft + pdft[plot] extra.
"""
from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp

import pdft
from pdft.viz import TrainingHistory, plot_training_loss


def main(out_dir: str | Path = "out") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    target = jax.random.normal(jax.random.PRNGKey(7), (4, 4)).astype(jnp.complex128)
    basis = pdft.QFTBasis(m=2, n=2)

    print(f"Training QFTBasis on a 4x4 target image for 50 steps with RiemannianGD...")
    t0 = time.perf_counter()
    result = pdft.train_basis(
        basis,
        target=target,
        loss=pdft.L1Norm(),
        optimizer=pdft.RiemannianGD(lr=0.01),
        steps=50,
        seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f}s")
    print(f"  initial loss: {result.loss_history[0]:.4f}")
    print(f"  final   loss: {result.loss_history[-1]:.4f}")
    print(f"  reduction   : {result.loss_history[0] - result.loss_history[-1]:.4f}")

    hist = TrainingHistory(losses=result.loss_history, label="RiemannianGD")
    fig_path = out / "basis_demo_loss.png"
    plot_training_loss(hist, output_path=fig_path, smooth_alpha=0.2,
                       title="QFTBasis training loss (L1, RiemannianGD, lr=0.01)")
    print(f"  wrote plot:  {fig_path}")


if __name__ == "__main__":
    main()
