"""Dataset loaders for benchmark runs. Read-only — no auto-download.

QuickDraw .npy files and DIV2K PNGs are expected at:
    /home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw/
    /home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR/

Datasets selected by `seed` use np.random.default_rng. PRNG is independent
from Julia's Random.seed!(42) — Python and Julia draw different image sets
even at the same seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_QUICKDRAW_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw")
DEFAULT_DIV2K_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")


def _ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} data_root does not exist: {path}\n"
            f"Place the dataset at this path or pass data_root=... to override."
        )


def load_quickdraw(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_QUICKDRAW_ROOT,
    img_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` 28x28 QuickDraw drawings, center-pad with zeros
    to `img_size`, return as float32 in [0, 1].

    Mirrors `pad_to_power_of_two` in `ParametricDFT-Benchmarks.jl/data_loading.jl`.
    Padding (rather than resizing) preserves the sparse-line-drawing structure
    of QuickDraw — the surrounding zeros compress trivially in any frequency
    basis, which is what gives Julia's reported 30 dB at kr=0.20.
    """
    data_root = Path(data_root)
    _ensure_dir(data_root, "quickdraw")

    npy_files = sorted(data_root.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy files found under {data_root}")

    total_needed = n_train + n_test
    rng = np.random.default_rng(seed)

    per_cat_target = (total_needed + len(npy_files) - 1) // len(npy_files)

    picked: list[np.ndarray] = []
    for npy in npy_files:
        arr = np.load(npy)
        if arr.ndim != 2 or arr.shape[1] != 784:
            raise ValueError(f"{npy} has unexpected shape {arr.shape}; expected (N, 784)")
        n_avail = arr.shape[0]
        n_pick = min(per_cat_target, n_avail)
        idx = rng.choice(n_avail, size=n_pick, replace=False)
        picked.append(arr[idx].reshape(n_pick, 28, 28))

    pool = np.concatenate(picked, axis=0).astype(np.float32) / 255.0
    if pool.shape[0] < total_needed:
        raise ValueError(
            f"not enough images in {data_root}: have {pool.shape[0]}, need {total_needed}"
        )

    perm = rng.permutation(pool.shape[0])
    pool = pool[perm[:total_needed]]

    if img_size != 28:
        pool = _pad_batch(pool, img_size)

    train = pool[:n_train]
    test = pool[n_train:]
    return train, test


def _pad_batch(images: np.ndarray, target_size: int) -> np.ndarray:
    """Center-pad each (h, w) image with zeros to (target_size, target_size).

    Mirror of `pad_to_power_of_two` in
    `ParametricDFT-Benchmarks.jl/data_loading.jl`.
    """
    n, h, w = images.shape
    if h > target_size or w > target_size:
        raise ValueError(
            f"can't center-pad ({h},{w}) into ({target_size},{target_size}) — image larger than target"
        )
    out = np.zeros((n, target_size, target_size), dtype=images.dtype)
    y_off = (target_size - h) // 2
    x_off = (target_size - w) // 2
    out[:, y_off:y_off + h, x_off:x_off + w] = images
    return out


def load_div2k(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_DIV2K_ROOT,
    size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` DIV2K PNGs, convert to grayscale, center-crop
    to a square, resize to `size`x`size`, return as float32 in [0, 1].
    """
    data_root = Path(data_root)
    _ensure_dir(data_root, "div2k")

    pngs = sorted(data_root.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no .png files found under {data_root}")

    total_needed = n_train + n_test
    if len(pngs) < total_needed:
        raise ValueError(f"not enough images in {data_root}: have {len(pngs)}, need {total_needed}")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(pngs), size=total_needed, replace=False)
    chosen = [pngs[i] for i in chosen_idx]

    out = np.empty((total_needed, size, size), dtype=np.float32)
    for i, p in enumerate(chosen):
        img = Image.open(p).convert("L")
        # Center-crop to square.
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0

    return out[:n_train], out[n_train:]


def _resize_batch(images: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a batch (N, H, W) -> (N, target, target) via Pillow LANCZOS."""
    n = images.shape[0]
    out = np.empty((n, target_size, target_size), dtype=np.float32)
    for i, img in enumerate(images):
        pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        pil = pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
        out[i] = np.asarray(pil, dtype=np.float32) / 255.0
    return out
