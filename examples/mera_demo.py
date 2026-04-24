"""mera_demo.py — compare QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis on a single target.

Run: python examples/mera_demo.py
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

    target = jax.random.normal(jax.random.PRNGKey(11), (4, 4)).astype(jnp.complex128)

    histories: list[TrainingHistory] = []
    for label, basis in [
        ("QFTBasis", pdft.QFTBasis(m=2, n=2)),
        ("EntangledQFTBasis", pdft.EntangledQFTBasis(m=2, n=2)),
        ("TEBDBasis", pdft.TEBDBasis(m=2, n=2)),
        ("MERABasis", pdft.MERABasis(m=2, n=2)),
    ]:
        print(f"Training {label}...")
        t0 = time.perf_counter()
        result = pdft.train_basis(
            basis,
            target=target,
            loss=pdft.L1Norm(),
            optimizer=pdft.RiemannianAdam(lr=0.01),
            steps=40,
            seed=0,
        )
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.2f}s, initial {result.loss_history[0]:.4f}, final {result.loss_history[-1]:.4f}")
        histories.append(TrainingHistory(losses=result.loss_history, label=label))

    fig_path = out / "mera_demo.png"
    plot_training_comparison(histories, output_path=fig_path,
                             title="Four bases on the same 4x4 target (RiemannianAdam, L1 loss)")
    print(f"wrote: {fig_path}")


if __name__ == "__main__":
    main()
