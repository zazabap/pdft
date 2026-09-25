"""Full-reference image quality metrics --- one implementation each.

All three are higher-is-better; SSIM and MS-SSIM equal 1 for identical images.
The Gaussian window is computed here in numpy, matching
``scipy.ndimage.gaussian_filter(sigma=1.5, truncate=3.5, mode="reflect")`` to
round-off (a test pins it when scipy is installed), so the package does not
depend on scipy for its metrics. SSIM agrees with scikit-image to about 6e-4
on real reconstructions.
"""

from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR on [0, 1] data. Inputs are clipped."""
    a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
    b = np.clip(np.asarray(b, dtype=np.float64), 0.0, 1.0)
    return float(10.0 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-15)))


# Wang, Simoncelli & Bovik 2003, table 1.
MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
_C1, _C2 = 0.01**2, 0.03**2


def gaussian_filter(a: np.ndarray, sigma: float = 1.5, truncate: float = 3.5) -> np.ndarray:
    """Separable Gaussian blur with scipy's kernel and "reflect" boundary.

    The kernel spans ``int(truncate * sigma + 0.5)`` taps on each side and is
    normalised to unit sum; "reflect" in scipy's sense repeats the edge sample
    (numpy's ``symmetric``). Applied along every axis in turn.
    """
    a = np.asarray(a, dtype=np.float64)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    for axis in range(a.ndim):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(a, pad, mode="symmetric")
        n = a.shape[axis]
        out = np.zeros_like(a)
        for i, wi in enumerate(w):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + n)
            out += wi * padded[tuple(sl)]
        a = out
    return a


def _stats(a, b, sigma=1.5):
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    return (
        mu_a,
        mu_b,
        gaussian_filter(a * a, sigma) - mu_a**2,
        gaussian_filter(b * b, sigma) - mu_b**2,
        gaussian_filter(a * b, sigma) - mu_a * mu_b,
    )


def ssim(a, b, sigma=1.5):
    """SSIM (Wang et al. 2004), 11x11 Gaussian window, data range 1."""
    a = np.clip(np.asarray(a, float), 0, 1)
    b = np.clip(np.asarray(b, float), 0, 1)
    mu_a, mu_b, saa, sbb, sab = _stats(a, b, sigma)
    num = (2 * mu_a * mu_b + _C1) * (2 * sab + _C2)
    den = (mu_a**2 + mu_b**2 + _C1) * (saa + sbb + _C2)
    return float(np.mean(num / den))


def ms_ssim(a, b, weights=MS_WEIGHTS):
    """MS-SSIM (Wang et al. 2003): contrast/structure at each of five scales,
    luminance at the coarsest only, 2x2 box downsampling between scales."""
    a = np.clip(np.asarray(a, float), 0, 1)
    b = np.clip(np.asarray(b, float), 0, 1)
    out = 1.0
    for i, w in enumerate(weights):
        mu_a, mu_b, saa, sbb, sab = _stats(a, b)
        cs = float(np.mean((2 * sab + _C2) / (saa + sbb + _C2)))
        if i == len(weights) - 1:
            lum = float(np.mean((2 * mu_a * mu_b + _C1) / (mu_a**2 + mu_b**2 + _C1)))
            out *= max(lum, 1e-8) ** w * max(cs, 1e-8) ** w
        else:
            out *= max(cs, 1e-8) ** w
            a = 0.25 * (a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2])
            b = 0.25 * (b[::2, ::2] + b[1::2, ::2] + b[::2, 1::2] + b[1::2, 1::2])
    return float(out)


def score(a, b) -> dict:
    """All three metrics of one reconstruction against its ground truth."""
    return {"psnr": psnr(a, b), "ssim": ssim(a, b), "ms_ssim": ms_ssim(a, b)}
