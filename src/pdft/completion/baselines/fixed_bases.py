"""Fixed orthonormal bases, and iterative hard thresholding in them.

The circuit at ``theta0`` is the DFT, so every number at ``theta0`` is a
statement about Fourier-domain IHT, and the contribution is only what training
buys over the best fixed alternative: the DCT-II (even symmetric extension, so
no frame-border discontinuity) and the orthonormal wavelets, the classical
sparsifying bases for natural images and the compression winners whose
completion loss is the paper's motivating reversal. Wavelets run in
periodization mode so they are exact isometries and the ``k`` budget means the
same thing on every side. The DCT comes from scipy when installed and from
``jax.scipy`` otherwise; the wavelets need PyWavelets.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

WAVELETS = ("haar", "db4", "sym8")


def hard_k(C: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest-magnitude entries; numpy, no gradient to protect."""
    a = np.abs(C).ravel()
    if k >= a.size:
        return C
    return np.where(np.abs(C) >= np.partition(a, -k)[-k], C, 0.0)


def iht_fixed(img, obs, k: int, iters: int, fwd: Callable, inv: Callable) -> np.ndarray:
    """The solver's iteration in a fixed basis ``(fwd, inv)``, in numpy."""
    x = np.where(obs, img, 0.0)
    for _ in range(iters):
        x = np.where(obs, img, np.real(inv(hard_k(fwd(x), k))))
    return x


def _dct_backend():
    try:
        from scipy import fft as sf

        return sf.dctn, sf.idctn
    except ImportError:  # pragma: no cover - exercised only without scipy
        from jax.scipy import fft as jf

        def dctn(x, norm):
            return np.asarray(jf.dctn(np.asarray(x), norm=norm))

        def idctn(c, norm):
            return np.asarray(jf.idctn(np.asarray(c), norm=norm))

        return dctn, idctn


def dct_pair() -> tuple[Callable, Callable]:
    """``(fwd, inv)`` of the orthonormal 2-D DCT-II."""
    dctn, idctn = _dct_backend()
    return (lambda x: dctn(x, norm="ortho"), lambda c: idctn(c, norm="ortho"))


def dft_pair() -> tuple[Callable, Callable]:
    """``(fwd, inv)`` of the orthonormal 2-D DFT."""
    return (lambda x: np.fft.fft2(x, norm="ortho"), lambda c: np.fft.ifft2(c, norm="ortho"))


def wavelet_pair(name: str) -> tuple[Callable, Callable]:
    """``(fwd, inv)`` of an orthonormal wavelet.

    ``fwd`` stashes the coefficient slices it produced so ``inv`` can rebuild
    the pyramid; ``inv`` is only ever called on the preceding ``fwd``'s output,
    as ``iht_fixed`` does.
    """
    import pywt

    def fwd(x):
        c = pywt.wavedec2(x, name, mode="periodization")
        arr, slices = pywt.coeffs_to_array(c)
        fwd.slices = slices
        return arr

    def inv(a):
        c = pywt.array_to_coeffs(a, fwd.slices, output_format="wavedec2")
        return pywt.waverec2(c, name, mode="periodization")

    return fwd, inv


def fixed_bases(wavelets=WAVELETS) -> dict:
    """``{name: (fwd, inv)}`` for every control available; the wavelets are absent without PyWavelets."""
    bases = {"DCT-II": dct_pair(), "DFT": dft_pair()}
    try:
        for w in wavelets:
            bases[w] = wavelet_pair(w)
    except ImportError:
        pass
    return bases
