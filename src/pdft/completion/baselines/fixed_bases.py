"""Fixed orthonormal bases, and IHT in them --- the classical controls.

The circuit at theta0 IS the DFT, so every number the subpackage reports at
theta0 is a statement about Fourier-domain iterative hard thresholding. The
contribution is only what training buys over the best *fixed* alternative a
reviewer would reach for: the DCT-II (even symmetric extension, so no
frame-border discontinuity) and the orthonormal wavelets, which are the
classical sparsifying bases for natural images --- and the compression winners
whose completion loss is the paper's motivating reversal.

Wavelets run in periodization mode so they are exact isometries and the k
budget means the same thing on every side. iht_fixed is the same iteration as
pdft.completion.solver.reconstruct, in numpy, for bases that have no angles to
train. The DCT comes from scipy when it is installed and from jax.scipy
otherwise (they agree to round-off); the wavelets need PyWavelets.
"""

from __future__ import annotations

import numpy as np


def hard_k(C: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest-magnitude entries (numpy; no gradient to protect)."""
    a = np.abs(C).ravel()
    if k >= a.size:
        return C
    return np.where(np.abs(C) >= np.partition(a, -k)[-k], C, 0.0)


def iht_fixed(img, obs, k, iters, fwd, inv):
    """The iteration of pdft.completion.solver.reconstruct, in a fixed basis (fwd, inv)."""
    x = np.where(obs, img, 0.0)
    for _ in range(iters):
        x = np.real(inv(hard_k(fwd(x), k)))
        x = np.where(obs, img, x)
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


def dct_pair():
    dctn, idctn = _dct_backend()
    return (lambda x: dctn(x, norm="ortho"), lambda c: idctn(c, norm="ortho"))


def dft_pair():
    return (lambda x: np.fft.fft2(x, norm="ortho"), lambda c: np.fft.ifft2(c, norm="ortho"))


def wavelet_pair(name: str):
    """(fwd, inv) for an orthonormal wavelet. fwd stashes the coefficient
    slices it produced so inv can rebuild the pyramid --- inv is only ever
    called on the output of the preceding fwd, as iht_fixed does."""
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


WAVELETS = ("haar", "db4", "sym8")


def fixed_bases(wavelets=WAVELETS) -> dict:
    """{name: (fwd, inv)} for every control available in this environment.

    The wavelet rows are silently absent without PyWavelets; callers that need
    them check for "haar" and say so, rather than half the controls failing.
    """
    bases = {"DCT-II": dct_pair(), "DFT": dft_pair()}
    try:
        for w in wavelets:
            bases[w] = wavelet_pair(w)
    except ImportError:
        pass
    return bases
