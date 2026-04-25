"""Image compression via sparse-basis top-k truncation.

Mirror of upstream src/compression.jl. Stores the top-k coefficients of
an image's frequency-domain representation (under a given AbstractSparseBasis)
as a sparse tuple (indices, real parts, imag parts, original size,
basis_hash). Indices use Julia's 1-based column-major convention for
cross-language JSON compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .io_json import basis_hash

_VERSION = "1.0"


@dataclass
class CompressedImage:
    """Sparse frequency-domain representation of an image.

    Mirror of upstream src/compression.jl:27-33. `indices` are 1-based
    column-major linear indices into the frequency-domain matrix (Julia
    convention, chosen for cross-language JSON interop).
    """

    indices: list[int]  # 1-based column-major
    values_real: list[float]
    values_imag: list[float]
    original_size: tuple[int, int]
    basis_hash: str


def _select_top_coefficients(freq: np.ndarray, k: int):
    """Return (indices_1based_colmajor, values) for the top-k by magnitude.

    Mirror of upstream src/compression.jl:138-148. Magnitude-based; ties are
    broken by position (matching `argpartition` stability in practice).
    """
    k = min(k, freq.size)
    # Column-major flatten (Julia vec)
    flat_col = freq.flatten(order="F")
    mags = np.abs(flat_col)
    # Indices of the k largest magnitudes (0-based)
    top = np.argpartition(-mags, k - 1)[:k]
    # Sort to get deterministic order: by magnitude descending, then index
    order = np.argsort(-mags[top], kind="stable")
    top_sorted = top[order]
    values = flat_col[top_sorted]
    # Convert to 1-based for Julia compatibility
    indices_1b = [int(i + 1) for i in top_sorted]
    return indices_1b, values


def compress(basis, image, *, ratio: float = 0.9) -> CompressedImage:
    """Compress `image` under `basis`, keeping top (1 - ratio) fraction.

    Mirror of upstream src/compression.jl:63-90. `ratio=0.9` keeps 10% of
    coefficients.
    """
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    expected = basis.image_size
    image = np.asarray(image)
    if image.shape != expected:
        raise ValueError(f"image shape {image.shape} must match basis size {expected}")

    freq = np.asarray(basis.forward_transform(jnp.asarray(image)))
    total = freq.size
    keep = max(1, round(total * (1.0 - ratio)))
    indices, values = _select_top_coefficients(freq, keep)
    return CompressedImage(
        indices=indices,
        values_real=[float(v.real) for v in values],
        values_imag=[float(v.imag) for v in values],
        original_size=tuple(int(s) for s in image.shape),
        basis_hash=basis_hash(basis),
    )


def compress_with_k(basis, image, *, k: int) -> CompressedImage:
    """Compress keeping exactly `k` coefficients. Mirror of upstream src/compression.jl:105-130."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    expected = basis.image_size
    image = np.asarray(image)
    if image.shape != expected:
        raise ValueError(f"image shape {image.shape} must match basis size {expected}")

    freq = np.asarray(basis.forward_transform(jnp.asarray(image)))
    keep = min(k, freq.size)
    indices, values = _select_top_coefficients(freq, keep)
    return CompressedImage(
        indices=indices,
        values_real=[float(v.real) for v in values],
        values_imag=[float(v.imag) for v in values],
        original_size=tuple(int(s) for s in image.shape),
        basis_hash=basis_hash(basis),
    )


def _reconstruct_frequency_domain(compressed: CompressedImage) -> np.ndarray:
    """Fill a zero matrix with `compressed` non-zero entries (1-based colmajor indices)."""
    h, w = compressed.original_size
    freq_flat_colmajor = np.zeros(h * w, dtype=np.complex128)
    for idx_1b, re, im in zip(compressed.indices, compressed.values_real, compressed.values_imag):
        freq_flat_colmajor[idx_1b - 1] = complex(re, im)
    # Reshape from column-major back to (h, w)
    return freq_flat_colmajor.reshape((h, w), order="F")


def recover(basis, compressed: CompressedImage, *, verify_hash: bool = True) -> np.ndarray:
    """Reconstruct real-valued image from a compressed representation.

    Mirror of upstream src/compression.jl:174-197. Returns `real(T^{-1}(freq))`
    as a float64 numpy array.
    """
    if verify_hash:
        expected = basis_hash(basis)
        if expected != compressed.basis_hash:
            raise ValueError(
                f"Basis hash mismatch. Compressed: {compressed.basis_hash}, basis: {expected}"
            )
    if tuple(compressed.original_size) != tuple(basis.image_size):
        raise ValueError(
            f"Compressed size {compressed.original_size} does not match basis size {basis.image_size}"
        )

    freq = _reconstruct_frequency_domain(compressed)
    recovered = np.asarray(basis.inverse_transform(jnp.asarray(freq)))
    return np.real(recovered)


def compressed_to_dict(c: CompressedImage) -> dict:
    return {
        "version": _VERSION,
        "indices": c.indices,
        "values_real": c.values_real,
        "values_imag": c.values_imag,
        "original_height": int(c.original_size[0]),
        "original_width": int(c.original_size[1]),
        "basis_hash": c.basis_hash,
    }


def dict_to_compressed(d: dict) -> CompressedImage:
    return CompressedImage(
        indices=[int(i) for i in d["indices"]],
        values_real=[float(v) for v in d["values_real"]],
        values_imag=[float(v) for v in d["values_imag"]],
        original_size=(int(d["original_height"]), int(d["original_width"])),
        basis_hash=str(d["basis_hash"]),
    )


def save_compressed(path: str | Path, c: CompressedImage) -> Path:
    p = Path(path)
    with p.open("w") as f:
        json.dump(compressed_to_dict(c), f, indent=2)
    return p


def load_compressed(path: str | Path) -> CompressedImage:
    p = Path(path)
    with p.open("r") as f:
        return dict_to_compressed(json.load(f))


def compression_stats(c: CompressedImage) -> dict:
    """Summary of compression ratio + kept coefficients.

    Mirror of upstream src/compression.jl:318-343.
    """
    h, w = c.original_size
    total = h * w
    kept = len(c.indices)
    return {
        "original_size": (h, w),
        "total_coefficients": total,
        "kept_coefficients": kept,
        "compression_ratio": 1.0 - kept / total,
        "storage_reduction": 1.0 - (3 * kept) / (2 * total),  # 3 floats per kept vs 2 per pixel
    }
