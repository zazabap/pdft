"""Classical compression baselines for the benchmark harness.

Mirrors `evaluation.jl::fft_compress_recover` and `dct_compress_recover` from
the Julia repo (top-k% of magnitudes globally, zero the rest, inverse-transform).
Block variants extend the same semantic to non-overlapping 8×8 tiles.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def _top_k_mask(magnitudes: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask, True at the k largest entries by magnitude. Ties broken arbitrarily."""
    if k <= 0:
        return np.zeros_like(magnitudes, dtype=bool)
    if k >= magnitudes.size:
        return np.ones_like(magnitudes, dtype=bool)
    flat = magnitudes.ravel()
    # argpartition: the k-th order statistic is at position k-1 of the partition;
    # everything before it is <= and everything after is >=.
    threshold_idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[threshold_idx] = True
    return mask.reshape(magnitudes.shape)


def global_fft_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """2D FFT compression: keep top-k% magnitudes globally, return real part.

    Mirrors evaluation.jl::fft_compress_recover.
    """
    freq = np.fft.fftshift(np.fft.fft2(image))
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0 + 0.0j)
    return np.real(np.fft.ifft2(np.fft.ifftshift(compressed)))


def global_dct_compress(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """2D DCT-II compression: keep top-k% magnitudes globally.

    Mirrors evaluation.jl::dct_compress_recover.
    """
    freq = dct(dct(image, axis=0, norm="ortho"), axis=1, norm="ortho")
    total = freq.size
    keep = max(1, int(np.floor(total * keep_ratio)))
    mask = _top_k_mask(np.abs(freq), keep)
    compressed = np.where(mask, freq, 0.0)
    return idct(idct(compressed, axis=0, norm="ortho"), axis=1, norm="ortho")
