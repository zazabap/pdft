#!/usr/bin/env python3
"""Sweep training configurations on QuickDraw QFT to find what reaches Julia's 30 dB.

Each config trains a fresh QFTBasis on `n_train` images for `epochs` × ceil(n_train / batch_size)
optimizer steps, then evaluates PSNR on `n_test` held-out images at the four standard keep ratios.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import pdft  # noqa: E402  -- sets jax_enable_x64

import jax.numpy as jnp  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio  # noqa: E402

from data_loading import load_quickdraw  # noqa: E402

CONFIGS = [
    # Name, epochs, batch_size, optimizer, lr_peak, lr_final, warmup_frac, validation_split, max_grad_norm
    ("baseline",            10,  16,  "adam", 0.01,   0.001,  0.05, 0.2,  1.0),
    ("100ep_same_lr",       100, 16,  "adam", 0.01,   0.001,  0.05, 0.2,  1.0),
    ("10ep_smaller_lr",     10,  16,  "adam", 0.001,  0.0001, 0.05, 0.2,  1.0),
    ("100ep_smaller_lr",    100, 16,  "adam", 0.001,  0.0001, 0.05, 0.2,  1.0),
    ("10ep_gd",             10,  16,  "gd",   0.01,   0.001,  0.05, 0.2,  None),
    ("100ep_gd",            100, 16,  "gd",   0.01,   0.001,  0.05, 0.2,  None),
    ("100ep_no_val",        100, 16,  "adam", 0.01,   0.001,  0.05, 0.0,  1.0),
    ("100ep_b1_per_image",  100, 1,   "adam", 0.01,   0.001,  0.05, 0.2,  1.0),
    ("500ep_smaller_lr",    500, 16,  "adam", 0.001,  0.0001, 0.05, 0.0,  1.0),
]


def run(name, epochs, batch_size, optimizer, lr_peak, lr_final, warmup_frac, val_split, mgn,
        train_imgs, test_imgs, m=5, n=5, k_keep=102):
    target_dataset = [jnp.asarray(t, dtype=jnp.complex128) for t in train_imgs]
    basis = pdft.QFTBasis(m=m, n=n)

    t0 = time.perf_counter()
    res = pdft.train_basis_batched(
        basis,
        dataset=target_dataset,
        loss=pdft.MSELoss(k=k_keep),
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        validation_split=val_split,
        early_stopping_patience=epochs,  # disable early stopping
        warmup_frac=warmup_frac,
        lr_peak=lr_peak,
        lr_final=lr_final,
        max_grad_norm=mgn,
        seed=42,
    )
    elapsed = time.perf_counter() - t0

    psnrs = {}
    for kr in (0.05, 0.10, 0.15, 0.20):
        per_img = []
        for img in test_imgs:
            img64 = np.asarray(img, dtype=np.float64)
            c = pdft.compress(res.basis, img64, ratio=1 - kr)
            r = np.clip(np.real(pdft.recover(res.basis, c)), 0, 1)
            per_img.append(peak_signal_noise_ratio(img64, r, data_range=1.0))
        psnrs[kr] = float(np.mean(per_img))

    return {
        "name": name, "epochs": epochs, "batch_size": batch_size, "optimizer": optimizer,
        "lr_peak": lr_peak, "lr_final": lr_final, "val_split": val_split, "mgn": mgn,
        "time": elapsed, "steps": res.steps,
        "psnr": psnrs,
        "loss_first": res.loss_history[0] if res.loss_history else None,
        "loss_last":  res.loss_history[-1] if res.loss_history else None,
        "loss_min":   min(res.loss_history) if res.loss_history else None,
    }


def main():
    train_imgs, test_imgs = load_quickdraw(20, 50, seed=42)

    print(f"{'config':22s}  {'ep':>4s}  {'bs':>3s}  {'opt':>4s}  {'lr_pk':>7s}  "
          f"{'val':>4s}  {'mgn':>5s}  {'time':>7s}  {'stp':>4s}  "
          f"{'5%':>5s}  {'10%':>5s}  {'15%':>5s}  {'20%':>5s}  {'L0':>7s}  {'Lend':>7s}")
    print("-" * 130)
    rows = []
    for cfg in CONFIGS:
        try:
            r = run(*cfg, train_imgs=train_imgs, test_imgs=test_imgs)
        except Exception as e:
            print(f"{cfg[0]:22s}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        rows.append(r)
        mgn_str = f"{r['mgn']:.1f}" if r['mgn'] is not None else "—"
        print(f"{r['name']:22s}  {r['epochs']:>4d}  {r['batch_size']:>3d}  {r['optimizer']:>4s}  "
              f"{r['lr_peak']:7.4f}  {r['val_split']:4.2f}  {mgn_str:>5s}  "
              f"{r['time']:6.1f}s  {r['steps']:>4d}  "
              f"{r['psnr'][0.05]:5.2f}  {r['psnr'][0.10]:5.2f}  {r['psnr'][0.15]:5.2f}  {r['psnr'][0.20]:5.2f}  "
              f"{r['loss_first']:7.2f}  {r['loss_last']:7.2f}")

    if rows:
        # Best by PSNR @ 0.20
        best = max(rows, key=lambda r: r['psnr'][0.20])
        print(f"\nBest at kr=0.20: {best['name']}  PSNR = {best['psnr'][0.20]:.3f} dB")
        print(f"  config: epochs={best['epochs']} batch_size={best['batch_size']} "
              f"optimizer={best['optimizer']} lr_peak={best['lr_peak']} val_split={best['val_split']}")


if __name__ == "__main__":
    main()
