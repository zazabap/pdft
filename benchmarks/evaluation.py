"""Compression-quality evaluation: MSE, PSNR, SSIM at multiple keep ratios.

Field shape is bit-compatible with Julia's metrics.json:
    {kr_str: {mean_mse, std_mse, mean_psnr, std_psnr, mean_ssim, std_ssim}}.
PSNR/SSIM via scikit-image (data_range=1.0). Recovered images are clamped to
[0, 1] before metric computation, matching evaluation.jl:25.

Per-(image, keep_ratio) failures record NaN; aggregation uses nanmean/nanstd.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

import numpy as np
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)


def compute_metrics(original: np.ndarray, recovered: np.ndarray) -> dict[str, float]:
    """MSE / PSNR / SSIM. Clamps recovered to [0,1] and drops imag part."""
    rec = np.clip(np.real(recovered), 0.0, 1.0)
    mse = float(np.mean((original - rec) ** 2))
    if mse > 0:
        psnr = 10.0 * np.log10(1.0 / mse)
    else:
        psnr = float("inf")
    try:
        ssim = float(structural_similarity(original, rec, data_range=1.0))
    except (ValueError, RuntimeError):
        ssim = float("nan")
    return {"mse": mse, "psnr": float(psnr), "ssim": ssim}


def aggregate_per_keep_ratio(
    per_image_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Mean/std across a list of per-image metrics dicts. NaNs ignored.

    Returns mean_mse/std_mse/mean_psnr/std_psnr/mean_ssim/std_ssim plus nan_count.
    """
    mse_vals = np.array([m["mse"] for m in per_image_metrics], dtype=np.float64)
    psnr_vals = np.array([m["psnr"] for m in per_image_metrics], dtype=np.float64)
    ssim_vals = np.array([m["ssim"] for m in per_image_metrics], dtype=np.float64)

    nan_count = int(np.sum(np.isnan(mse_vals) | np.isnan(psnr_vals) | np.isnan(ssim_vals)))

    # PSNR can be +inf (perfect reconstruction). nanstd raises RuntimeWarning
    # on all-inf arrays, so compute std over finite values only; all-inf → 0.0.
    finite_psnr = psnr_vals[np.isfinite(psnr_vals)]
    std_psnr = float(np.nanstd(finite_psnr)) if len(finite_psnr) > 0 else 0.0

    return {
        "mean_mse": float(np.nanmean(mse_vals)),
        "std_mse": float(np.nanstd(mse_vals)),
        "mean_psnr": float(np.nanmean(psnr_vals)),
        "std_psnr": std_psnr,
        "mean_ssim": float(np.nanmean(ssim_vals)),
        "std_ssim": float(np.nanstd(ssim_vals)),
        "nan_count": nan_count,
    }


def evaluate_baseline(
    fn: Callable[[np.ndarray, float], np.ndarray],
    test_images: np.ndarray,
    keep_ratios: Sequence[float],
) -> tuple[dict[str, dict[str, float]], float]:
    """Run `fn(image, keep_ratio)` over every (image, kr) pair.

    Returns ({kr_str: aggregated_metrics}, elapsed_seconds). On per-call
    failure, that pair's metrics are nan and the call is skipped.
    """
    t0 = time.perf_counter()
    out: dict[str, dict[str, float]] = {}
    for kr in keep_ratios:
        per_image: list[dict[str, float]] = []
        for img in test_images:
            try:
                recovered = fn(img, kr)
                per_image.append(compute_metrics(img, recovered))
            except Exception as e:  # noqa: BLE001
                logger.warning("baseline failed on a single image: %s", e)
                per_image.append({"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan")})
        agg = aggregate_per_keep_ratio(per_image)
        out[str(kr)] = agg
    elapsed = time.perf_counter() - t0
    return out, elapsed


def evaluate_basis_shared(
    basis,
    test_images: np.ndarray,
    keep_ratios: Sequence[float],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """One trained basis evaluated on every image in `test_images`.

    Mirrors `ParametricDFT-Benchmarks.jl/evaluation.jl::evaluate_basis`. Moves
    the basis to host via `jax.tree_util.tree_map(jax.device_get, ...)` (sidesteps
    the GPU scalar-indexing path in compress/recover). Per-(image, keep_ratio)
    failures are recorded as NaN so they don't sink the run.

    Returns ({kr_str: aggregated_metrics}, {kr_str: nan_count}).
    """
    import jax

    import pdft  # noqa: F401  -- ensures jax_enable_x64 is set before any jnp use

    cpu_basis = jax.tree_util.tree_map(jax.device_get, basis)

    out: dict[str, dict[str, float]] = {}
    nan_counts: dict[str, int] = {}
    for kr in keep_ratios:
        discard_ratio = 1.0 - kr
        per_image: list[dict[str, float]] = []
        for img in test_images:
            try:
                compressed = pdft.io.compress(
                    cpu_basis, np.asarray(img, dtype=np.float64), ratio=discard_ratio
                )
                recovered = pdft.io.recover(cpu_basis, compressed)
                per_image.append(compute_metrics(img, recovered))
            except Exception as e:  # noqa: BLE001
                logger.warning("compress/recover failed on (kr=%s): %s", kr, e)
                per_image.append({"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan")})
        agg = aggregate_per_keep_ratio(per_image)
        out[str(kr)] = agg
        nan_counts[str(kr)] = agg["nan_count"]
    return out, nan_counts


def evaluate_basis_per_image(
    bases: list,
    test_images: np.ndarray,
    keep_ratios: Sequence[float],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Per-image (P pairing): basis[i] evaluated on test_images[i].

    Moves each basis to host via jax.tree_util.tree_map(jax.device_get, ...).
    This sidesteps the GPU scalar-indexing path that compress/recover hit when
    tensors are still CuArrays (same intent as evaluation.jl:55-57). We avoid
    pdft.io.serialize save_basis/load_basis here because that path is hardcoded to
    QFTBasis (Phase 2 of the upstream port); the pytree map preserves the
    actual basis class for QFT/EntangledQFT/TEBD/MERA alike.

    Returns ({kr_str: aggregated_metrics}, {kr_str: nan_count}).
    """
    import jax

    import pdft  # noqa: F401  -- ensures jax_enable_x64 is set before any jnp use

    if len(bases) != len(test_images):
        raise ValueError(
            f"P-pairing requires len(bases) == len(test_images); "
            f"got {len(bases)} bases vs {len(test_images)} images"
        )

    cpu_bases = [jax.tree_util.tree_map(jax.device_get, b) for b in bases]

    out: dict[str, dict[str, float]] = {}
    nan_counts: dict[str, int] = {}
    for kr in keep_ratios:
        discard_ratio = 1.0 - kr
        per_image: list[dict[str, float]] = []
        for img, basis in zip(test_images, cpu_bases):
            try:
                compressed = pdft.io.compress(
                    basis, np.asarray(img, dtype=np.float64), ratio=discard_ratio
                )
                recovered = pdft.io.recover(basis, compressed)
                per_image.append(compute_metrics(img, recovered))
            except Exception as e:  # noqa: BLE001
                logger.warning("compress/recover failed on (kr=%s): %s", kr, e)
                per_image.append({"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan")})
        agg = aggregate_per_keep_ratio(per_image)
        out[str(kr)] = agg
        nan_counts[str(kr)] = agg["nan_count"]
    return out, nan_counts
