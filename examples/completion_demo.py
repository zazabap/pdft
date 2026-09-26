"""Inpaint from random pixels with a trained QFT
=============================================

Train the phase-only QFT circuit *through* the completion solver on a few
synthetic images, then recover a held-out image from 30% of its pixels and
compare against the untrained DFT. Finishes with the trained angles converted
to a :class:`pdft.QFTBasis`, certified incoherent, and saved as JSON.

Run: ``python examples/completion_demo.py``
Requires: pdft only (no data download).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import pdft
from pdft.coherence import certify_flat_modulus
from pdft.completion import psnr, reconstruct, theta0, train
from pdft.completion.bridge import qft_basis_from_angles
from pdft.completion.protocol import train_k
from pdft.io import save_basis

n = 5  # 32 x 32 images
N = 2**n
P = 0.30  # observed fraction
K_TRAIN, K_EVAL = 8, 40


def synthetic(rng):
    """Piecewise-constant blocks over a gradient: edges the plain DFT serves poorly."""
    y, x = np.mgrid[0:N, 0:N] / N
    img = 0.2 + 0.3 * x
    for _ in range(4):
        r0, c0 = rng.integers(0, N - 8, 2)
        h, w = rng.integers(6, 16, 2)
        img[r0 : r0 + h, c0 : c0 + w] += rng.uniform(-0.4, 0.4)
    return np.clip(img, 0, 1)


def main():
    rng = np.random.default_rng(0)
    images = np.stack([synthetic(rng) for _ in range(8)])
    train_imgs, test_img = images[:6], images[7]
    k = train_k(N * N, P, 0.125)

    print(f"training the phase-only circuit at n = {n}, p = {P:.0%}, k = {k}, K = {K_TRAIN}")
    params, history = train(train_imgs, k, K=K_TRAIN, p=P, steps=40, lr=2e-2, batch=2, log_every=10)
    print(f"loss {history[0]['loss']:.4e} -> {history[-1]['loss']:.4e}")

    obs = jnp.asarray(rng.random((N, N)) < P)
    Y = jnp.asarray(test_img) * obs
    dft = reconstruct(theta0(n), theta0(n), Y, obs, k, K_EVAL)
    ours = reconstruct(params["r"], params["c"], Y, obs, k, K_EVAL)
    print(
        f"held-out PSNR at K = {K_EVAL}: DFT {psnr(dft, test_img):.2f} dB, "
        f"trained {psnr(ours, test_img):.2f} dB"
    )

    basis = qft_basis_from_angles(params["r"], params["c"])
    cert = certify_flat_modulus(basis, frozen_indices=list(range(2 * n)))
    print(f"mu = {cert.mu:.6f}; {cert.reason}")

    out = Path("out")
    out.mkdir(exist_ok=True)
    path = save_basis(out / "completion_basis.json", basis)
    print(f"saved {path}")
    assert isinstance(pdft.io.load_basis(path), pdft.QFTBasis)


if __name__ == "__main__":
    main()
