"""Complete an image from a tenth of its pixels
============================================

Fill in an image from randomly observed pixels with ``pdft.tasks.complete``,
then train the circuit's phases through that solver with
``pdft.training.train_basis_steps`` and complete the image again. The trained
transform is Model B of the inpainting paper: the Hadamards stay fixed and the
four phases of every controlled-phase gate are free, so the coherence of the
basis with the pixels stays at its minimum, 1, whatever the training does.

The images are small synthetic ones with the rough statistics of photographs,
so the script runs in seconds on a CPU; the protocol is the paper's (ten
percent of the pixels, a fresh mask every step, plain Adam on the angles).
Note the bit reversal on the way in and out: a ``QFTBasis`` works on the
bit-reversed image, and a mask lives on the pixels.

Run: ``python examples/completion_demo.py``

Requires: pdft + pdft[plot] extra.

Reference
---------

The method and its evaluation on DIV2K:

.. code-block:: bibtex

   @misc{an2026quantum,
     title         = {Quantum-Inspired Trainable and Parameter-Efficient Tensor
                      Networks for Image Inpainting},
     author        = {An, Shiwen and Slavakis, Konstantinos},
     year          = {2026},
     eprint        = {2609.17298},
     archivePrefix = {arXiv},
     primaryClass  = {eess.IV},
     url           = {https://arxiv.org/abs/2609.17298},
   }
"""

from __future__ import annotations

import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import pdft
from pdft.bases import cp_diagonals_view
from pdft.circuit import bit_reverse
from pdft.tasks import complete, completion_loss
from pdft.viz._figure import require_matplotlib, save


def natural_images(count: int, size: int, seed: int) -> np.ndarray:
    """Random fields with a ``1/f^2`` spectrum and a few sharp edges, in ``[0, 1]``.

    Roughly the statistics of a photograph: a pure sum of cosines would be
    exactly sparse under the DFT and leave nothing to train.
    """
    rng = np.random.default_rng(seed)
    f2 = np.fft.fftfreq(size)[:, None] ** 2 + np.fft.fftfreq(size)[None, :] ** 2
    amplitude = np.where(f2 == 0, 0.0, 1.0 / np.maximum(f2, 1e-12))
    out = []
    for _ in range(count):
        field = np.real(np.fft.ifft2(amplitude * np.fft.fft2(rng.normal(size=(size, size)))))
        for _ in range(3):  # flat patches with sharp borders
            r0, c0 = rng.integers(0, size, 2)
            h, w = rng.integers(size // 6, size // 2, 2)
            field[r0 : r0 + h, c0 : c0 + w] += rng.normal() * field.std()
        out.append((field - field.min()) / (field.max() - field.min()))
    return np.stack(out)


def psnr(a, b) -> float:
    return float(10 * np.log10(1.0 / np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main(out_dir: str | Path = "out") -> None:
    require_matplotlib()
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    size, rate, k, solver_steps = 64, 0.10, 61, 60
    train = natural_images(16, size, seed=0)
    image = natural_images(1, size, seed=1)[0]
    mask = np.random.default_rng(1000).random(image.shape) < rate
    observed = jnp.asarray(image * mask)

    def solve(basis):
        # the mask is on the pixels; the basis works on the bit-reversed image
        filled = complete(
            basis, bit_reverse(observed), bit_reverse(jnp.asarray(mask)), k=k, steps=solver_steps
        )
        return np.asarray(bit_reverse(filled))

    basis = pdft.QFTBasis(m=6, n=6)
    before = solve(basis)
    print(f"DFT (untrained): {psnr(before, image):.2f} dB from {mask.mean():.0%} of the pixels")

    print("Training the four phases of every controlled-phase gate through the solver...")
    t0 = time.perf_counter()
    result = pdft.train_basis_steps(
        basis,
        dataset=train,
        objective=completion_loss(k=k, steps=10),
        view=cp_diagonals_view,
        optimizer=pdft.RiemannianAdam(lr=0.02),
        steps=60,
        rate=rate,
        batch_size=2,
        seed=0,
        frame=bit_reverse,
    )
    after = solve(result.basis)
    print(
        f"done in {time.perf_counter() - t0:.1f}s: {psnr(after, image):.2f} dB with the trained phases"
    )
    print(
        f"  coherence of the trained basis: {pdft.coherence(result.basis):.6f} (the minimum is 1)"
    )

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, (title, panel) in zip(
        axes,
        [
            ("original", image),
            (f"observed ({mask.mean():.0%})", np.where(mask, image, 0.0)),
            (f"DFT, {psnr(before, image):.1f} dB", before),
            (f"trained, {psnr(after, image):.1f} dB", after),
        ],
    ):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig_path = out / "completion_demo.png"
    save(fig, fig_path)
    print(f"  wrote figure: {fig_path}")


if __name__ == "__main__":
    main()
