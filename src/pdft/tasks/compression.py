"""Image compression via sparse-basis top-k truncation.

Mirror of upstream src/compression.jl. Stores the top-k coefficients of
an image's frequency-domain representation (under a given AbstractSparseBasis)
as a sparse tuple (indices, real parts, imag parts, original size,
basis_hash). Indices use Julia's 1-based column-major convention for
cross-language JSON compatibility.

A colour image is compressed as a stack of channel planes, ``(C, H, W)``
(``pdft.color`` makes one from an RGB image): the basis transforms every plane
and the ``k`` largest coefficients are kept over all of them together, so a
plane that carries little signal gets few of them. The indices then run over
the ``(C, H, W)`` array in column-major order, channel fastest (the order of
Julia's ``channelview``, though upstream has no colour path: there is no Julia
behaviour to match, and the tests are the specification). The JSON form adds
``original_channels`` and says so with schema version 1.1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..bases.core import _check_planes, _map_planes
from ..io.serialize import basis_hash

_VERSION = "1.0"  # upstream's schema: one plane
_VERSION_PLANES = "1.1"  # with ``original_channels``: a stack of planes


@dataclass
class CompressedImage:
    """Sparse frequency-domain representation of an image.

    Mirror of upstream src/compression.jl:27-33. `indices` are 1-based
    column-major linear indices into the frequency-domain array (Julia
    convention, chosen for cross-language JSON interop). `original_size` is
    ``(H, W)`` for one plane and ``(C, H, W)`` for a stack of channel planes.
    """

    indices: list[int]  # 1-based column-major
    values_real: list[float]
    values_imag: list[float]
    original_size: tuple[int, ...]
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


def _compress_keeping(basis, image, keep: Callable[[int], int]) -> CompressedImage:
    """Transform `image`, plane by plane, and keep the `keep(total)` largest of its `total` coefficients."""
    image = np.asarray(image)
    _check_planes(image, basis.image_size)

    freq = np.asarray(_map_planes(basis.forward_transform, jnp.asarray(image)))
    indices, values = _select_top_coefficients(freq, keep(freq.size))
    return CompressedImage(
        indices=indices,
        values_real=[float(v.real) for v in values],
        values_imag=[float(v.imag) for v in values],
        original_size=tuple(int(s) for s in image.shape),
        basis_hash=basis_hash(basis),
    )


def compress(basis, image, *, ratio: float = 0.9) -> CompressedImage:
    """Compress `image` under `basis`, keeping top (1 - ratio) fraction.

    Mirror of upstream src/compression.jl:63-90. `ratio=0.9` keeps 10% of
    coefficients.
    """
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    return _compress_keeping(basis, image, lambda total: max(1, round(total * (1.0 - ratio))))


def compress_with_k(basis, image, *, k: int) -> CompressedImage:
    """Compress keeping exactly `k` coefficients. Mirror of upstream src/compression.jl:105-130."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return _compress_keeping(basis, image, lambda total: k)


def _reconstruct_frequency_domain(compressed: CompressedImage) -> np.ndarray:
    """Fill a zero array with `compressed` non-zero entries (1-based colmajor indices)."""
    shape = tuple(compressed.original_size)
    freq_flat_colmajor = np.zeros(math.prod(shape), dtype=np.complex128)
    for idx_1b, re, im in zip(compressed.indices, compressed.values_real, compressed.values_imag):
        freq_flat_colmajor[idx_1b - 1] = complex(re, im)
    # Reshape from column-major back to the original shape
    return freq_flat_colmajor.reshape(shape, order="F")


def recover(basis, compressed: CompressedImage, *, verify_hash: bool = True) -> np.ndarray:
    """Reconstruct real-valued image from a compressed representation.

    Mirror of upstream src/compression.jl:174-197. Returns `real(T^{-1}(freq))`
    as a float64 numpy array of the original shape, ``(H, W)`` or ``(C, H, W)``.
    """
    if verify_hash:
        expected = basis_hash(basis)
        if expected != compressed.basis_hash:
            raise ValueError(
                f"Basis hash mismatch. Compressed: {compressed.basis_hash}, basis: {expected}"
            )
    size = tuple(compressed.original_size)
    if len(size) not in (2, 3) or size[-2:] != tuple(basis.image_size):
        raise ValueError(
            f"Compressed size {compressed.original_size} does not match basis size {basis.image_size}"
        )

    freq = _reconstruct_frequency_domain(compressed)
    recovered = np.asarray(_map_planes(basis.inverse_transform, jnp.asarray(freq)))
    return np.real(recovered)


def compressed_to_dict(c: CompressedImage) -> dict:
    """Convert to the JSON-ready dict form.

    One plane is written in upstream's schema (version 1.0). A stack of planes
    adds ``original_channels`` and is marked version 1.1.
    """
    *channels, height, width = c.original_size
    d = {
        "version": _VERSION_PLANES if channels else _VERSION,
        "indices": c.indices,
        "values_real": c.values_real,
        "values_imag": c.values_imag,
        "original_height": int(height),
        "original_width": int(width),
        "basis_hash": c.basis_hash,
    }
    if channels:
        d["original_channels"] = int(channels[0])
    return d


def dict_to_compressed(d: dict) -> CompressedImage:
    """Inverse of `compressed_to_dict`.

    Refuses a document of a version this reader does not know, and one whose
    ``original_channels`` does not go with its version.
    """
    version = str(d["version"])
    if version not in (_VERSION, _VERSION_PLANES):
        raise ValueError(
            f"unknown compressed-image schema version {version!r}; "
            f"this reader knows {_VERSION} (one plane) and {_VERSION_PLANES} (a stack of planes)"
        )
    stack = version == _VERSION_PLANES
    if ("original_channels" in d) != stack:
        raise ValueError(
            f"schema version {version} {'needs' if stack else 'has no'} original_channels"
        )
    size = (int(d["original_height"]), int(d["original_width"]))
    if stack:
        size = (int(d["original_channels"]),) + size
    return CompressedImage(
        indices=[int(i) for i in d["indices"]],
        values_real=[float(v) for v in d["values_real"]],
        values_imag=[float(v) for v in d["values_imag"]],
        original_size=size,
        basis_hash=str(d["basis_hash"]),
    )


def save_compressed(path: str | Path, c: CompressedImage) -> Path:
    """Write a `CompressedImage` to ``path`` as JSON; returns the path."""
    p = Path(path)
    with p.open("w") as f:
        json.dump(compressed_to_dict(c), f, indent=2)
    return p


def load_compressed(path: str | Path) -> CompressedImage:
    """Read a `CompressedImage` written by `save_compressed`."""
    p = Path(path)
    with p.open("r") as f:
        return dict_to_compressed(json.load(f))


def compression_stats(c: CompressedImage) -> dict:
    """Summary of compression ratio + kept coefficients.

    Mirror of upstream src/compression.jl:318-343. For a stack of planes the
    counts are over all of them.
    """
    total = math.prod(c.original_size)
    kept = len(c.indices)
    return {
        "original_size": tuple(c.original_size),
        "total_coefficients": total,
        "kept_coefficients": kept,
        "compression_ratio": 1.0 - kept / total,
        "storage_reduction": 1.0 - (3 * kept) / (2 * total),  # 3 floats per kept vs 2 per pixel
    }
