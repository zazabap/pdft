"""Image loading and the dataset splits of the completion paper.

Every function takes the data directory explicitly: this package has no
repository layout to anchor to. Pillow is imported lazily, so the rest of the
subpackage does not need it.

The paper's Table I split is DIV2K: its 800 training images give 750 training
crops and the 50 validation crops the learning-rate sweeps select on, and its
100 validation images are the test set, one 512^2 centre crop per image,
grayscale, no resampling. ``table1_split`` is the entry point every Table I
producer uses; ``kodak_split`` (16 training and 8 test frames of 768x512) is
the earlier protocol.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

from .metrics import psnr  # noqa: F401 -- historical home; new code uses pdft.completion.metrics


def _open_gray(path) -> np.ndarray:
    """One image as float64 grayscale in [0, 1], through Pillow's 8-bit "L"."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without pillow
        raise ImportError(
            "pdft.completion.data needs Pillow to read images: pip install pillow"
        ) from exc
    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def load_gray(path, size: int, offset: str = "center") -> np.ndarray:
    """Load one image as a size x size grayscale crop in [0, 1]."""
    im = _open_gray(path)
    h, w = im.shape
    if h < size or w < size:
        raise ValueError(f"{path}: {h}x{w} is smaller than the {size} crop")
    if offset == "center":
        y0, x0 = (h - size) // 2, (w - size) // 2
    else:
        y0, x0 = 0, 0
    return im[y0 : y0 + size, x0 : x0 + size]


def kodak_split(root, size: int = 512, n_train: int = 16):
    """Disjoint train/test image sets from a directory of ``kodim*.png``.

    Returns (train, test, names)."""
    root = pathlib.Path(root)
    paths = sorted(root.glob("kodim*.png"))
    if not paths:
        raise FileNotFoundError(f"no Kodak images in {root}")
    imgs = np.stack([load_gray(p, size) for p in paths])
    return imgs[:n_train], imgs[n_train:], [p.name for p in paths]


def load_4k(path, size: int) -> np.ndarray:
    """A native-resolution square crop, box-averaged down when a smaller size
    is asked for, so a resolution ladder is the same scene at every size."""
    a = _open_gray(path)
    if size == a.shape[0]:
        return a
    if size > a.shape[0]:
        raise ValueError(f"{size} exceeds the {a.shape[0]} source crop")
    f = a.shape[0] // size
    return a.reshape(size, f, size, f).mean(axis=(1, 3))


def detail_window(img, size: int, stride: int = 32):
    """The size x size window of highest variance: the most textured region,
    read from the image alone so no method's error enters the choice.
    Returns (row, col, size), or None when size is 0."""
    if not size:
        return None
    N = img.shape[0]
    best_w, best_v = None, -1.0
    for r in range(0, N - size + 1, stride):
        for c in range(0, N - size + 1, stride):
            v = float(np.var(img[r : r + size, c : c + size]))
            if v > best_v:
                best_w, best_v = (r, c, size), v
    return best_w


HELDOUT_NAMES = tuple(f"kodim{i:02d}" for i in range(17, 25))  # kodak_split's test half
DIV2K_N_TRAIN, DIV2K_N_VAL = 750, 50  # of DIV2K_train_HR's 800
TEST_NAMES = tuple(f"{i:04d}" for i in range(801, 901))  # table1_split's test ids


def _div2k_paths(root):
    tr = sorted((root / "DIV2K_train_HR").glob("*.png"))
    te = sorted((root / "DIV2K_valid_HR").glob("*.png"))
    if len(tr) != 800 or len(te) != 100:
        raise FileNotFoundError(f"DIV2K incomplete in {root} ({len(tr)} train, {len(te)} valid)")
    return tr, te


def _div2k_cache(root, size: int):
    """``<root>/div2k_<size>.npz``: every crop as uint8, built once from the
    PNGs so a script starts in seconds. uint8 is lossless here: the crops are
    PIL's 8-bit "L" conversion, which load_gray divides by 255."""
    root = pathlib.Path(root)
    path = root / f"div2k_{size}.npz"
    if path.exists():
        z = np.load(path)
        return z["train"], z["valid"], [str(x) for x in z["names"]]
    tr, te = _div2k_paths(root)

    def crops(paths):
        return np.stack([np.rint(load_gray(p, size) * 255).astype(np.uint8) for p in paths])

    train, valid = crops(tr), crops(te)
    names = np.array([p.stem for p in tr + te])
    np.savez_compressed(path, train=train, valid=valid, names=names)
    return train, valid, [str(x) for x in names]


def _smoke() -> int:
    """``PDFT_SMOKE=<n>`` truncates every split to its first n images: the
    smoke test of a producer, never a protocol."""
    return int(os.environ.get("PDFT_SMOKE", "0"))


def div2k_split(root, size: int = 512, n_train: int = DIV2K_N_TRAIN, n_val: int = DIV2K_N_VAL):
    """(train, val, test, names): DIV2K_train_HR 0001-0750 and 0751-0800, then
    DIV2K_valid_HR 0801-0900, one size^2 centre crop each in [0, 1]; names
    lists all 900 ids in that order. The validation crops exist so that the
    learning-rate sweeps select on images that are neither trained on nor
    reported."""
    train, valid, names = _div2k_cache(root, size)
    if n_train + n_val > len(train):
        raise ValueError(f"{n_train} + {n_val} exceeds DIV2K's {len(train)} training images")

    def f(a):
        return a.astype(np.float64) / 255.0

    tr, va, te = train[:n_train], train[n_train : n_train + n_val], valid
    nm = names[:n_train] + names[n_train : n_train + n_val] + names[len(train) :]
    smoke = _smoke()
    if smoke:
        tr, va, te = tr[:smoke], va[:smoke], te[:smoke]
        nm = nm[:smoke] + nm[n_train : n_train + smoke] + nm[len(train) : len(train) + smoke]
    return f(tr), f(va), f(te), nm


def table1_split(root, size: int = 512):
    """The split every Table I producer scores on, in kodak_split's convention:
    (train, test, names) with names[len(train):] the test ids. The sweeps' 50
    validation crops are table1_val's and are in neither half here."""
    tr, va, te, names = div2k_split(root, size)
    return tr, te, names[: len(tr)] + names[len(tr) + len(va) :]


def table1_val(root, size: int = 512):
    """The 50 crops the learning-rate (and Cayley-step) sweeps select on."""
    return div2k_split(root, size)[1]


def table1_test(root, size: int = 512):
    """The test images alone, (test, names), without materialising the
    training crops."""
    train, valid, names = _div2k_cache(root, size)
    te = valid.astype(np.float64) / 255.0
    nm = names[len(train) :]
    smoke = _smoke()
    if smoke:
        te, nm = te[:smoke], nm[:smoke]
    return te, nm
