"""Per-image reconstruction comparison plots.

For each test image, produces a single PDF showing the original alongside
reconstructions from every basis × every keep ratio. PSNR is annotated on
each tile. Mirrors the layout of
`ParametricDFT-Benchmarks.jl/analysis/analyze_frequency_space.jl`'s
`reconstructions.pdf` output.

The analysis runs in-process from `run_dataset` after evaluation has already
host-roundtripped the bases. We assume:

- `host_bases` is a dict mapping {basis_name: list_of_per_image_bases} where
  each basis is on the host (no JAX device tensors). Pass the same list that
  `evaluate_basis_per_image` was called with.
- `baseline_fns` is a dict mapping {baseline_name: callable(image, kr)} that
  returns a real-valued recovered image.
- `test_images` is a numpy array shape (N, H, W) in [0, 1].

Outputs go under `out_dir / "analysis" / "<image_idx>" / reconstructions.pdf`
plus a per-image `summary.txt`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# Lazy-import matplotlib to avoid heavy import unless analysis is actually run.
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def _psnr_clamped(original: np.ndarray, recovered: np.ndarray) -> float:
    rec = np.clip(np.real(recovered), 0.0, 1.0)
    mse = float(np.mean((original - rec) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _recover_basis(basis, image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Run pdft compress/recover and return a real-valued, [0,1]-clipped image."""
    import pdft

    discard_ratio = 1.0 - keep_ratio
    compressed = pdft.io.compress(basis, np.asarray(image, dtype=np.float64), ratio=discard_ratio)
    recovered = pdft.io.recover(basis, compressed)
    return np.clip(np.real(recovered), 0.0, 1.0)


def _forward_magnitude(basis_or_fn, image: np.ndarray) -> np.ndarray:
    """Forward-transform magnitude per method.

    `basis_or_fn`: either a pdft basis (uses `.forward_transform`) OR a callable
    `(image, keep_ratio=1.0) -> recovered_image` for baselines (we then transform
    via the matching numpy/scipy primitive — see _baseline_freq below).
    Returns abs(forward_output) as float64.
    """
    import jax.numpy as jnp

    if hasattr(basis_or_fn, "forward_transform"):
        freq = np.asarray(basis_or_fn.forward_transform(jnp.asarray(image, dtype=jnp.complex128)))
        return np.abs(freq)
    # baseline callables don't have a public forward; we approximate via the
    # caller-provided baseline-freq function (set by _baseline_freq lookup).
    raise ValueError("forward_magnitude needs a basis with forward_transform()")


def _baseline_freq_magnitude(baseline_name: str, image: np.ndarray) -> np.ndarray:
    """Compute frequency magnitude for the four classical baselines."""
    from scipy.fft import dct as scipy_dct

    if baseline_name == "fft":
        return np.abs(np.fft.fftshift(np.fft.fft2(image)))
    if baseline_name == "dct":
        return np.abs(scipy_dct(scipy_dct(image, axis=0, norm="ortho"), axis=1, norm="ortho"))
    if baseline_name in ("block_fft_8", "block_dct_8"):
        # Per-block magnitude — visualise the per-block frequency layout.
        b = 8
        h, w = image.shape
        tiles = (
            image.reshape(h // b, b, w // b, b)
            .swapaxes(1, 2)
            .copy()
        )
        if baseline_name == "block_fft_8":
            freq = np.fft.fft2(tiles, axes=(-2, -1))
        else:
            freq = scipy_dct(scipy_dct(tiles, axis=-2, norm="ortho"), axis=-1, norm="ortho")
        # Reassemble into (h, w) layout for visualisation.
        return (
            np.abs(freq)
            .swapaxes(1, 2)
            .reshape(h, w)
        )
    raise ValueError(f"unknown baseline {baseline_name!r}")


def _cumulative_energy(magnitude: np.ndarray) -> np.ndarray:
    """Sorted-magnitude cumulative energy curve (Julia's analyze_frequency_space.jl:118)."""
    e = np.sort(magnitude.flatten() ** 2)[::-1]
    c = np.cumsum(e)
    return c / c[-1] if c[-1] > 0 else c


def _peak_normalized_log(
    method_magnitudes: dict[str, np.ndarray], floor: float = 1e-6
) -> tuple[dict[str, np.ndarray], float, float]:
    """Per-method peak-normalised log10 magnitude with shared (zmin, zmax).

    Mirror of `analyze_frequency_space.jl::norm_mag` + `logmags` block (line 195-199):
    every method's spectrum has its own peak driven to log10(1)=0, and a fixed
    log10 floor at log10(1e-6) = -6. The shared zmin/zmax range makes panels
    directly comparable as heatmaps and 3D surfaces.
    """
    out: dict[str, np.ndarray] = {}
    for name, mag in method_magnitudes.items():
        peak = max(float(np.max(mag)), 1e-300)
        normed = mag / peak
        out[name] = np.log10(normed + floor)
    zmin = float(min(np.min(v) for v in out.values()))
    zmax = 0.0
    return out, zmin, zmax


def _draw_frequency_spectra(
    image: np.ndarray, image_idx: int,
    method_magnitudes: dict[str, np.ndarray],
    out_pdf: Path,
) -> None:
    """2D frequency spectra: peak-normalised log|F| per method, shared colorbar.

    Mirror of `analyze_frequency_space.jl` (A) panel: one column per method,
    same colormap and colorrange across all methods. Rotation 90° matches
    Julia's `rotr90` orientation convention.
    """
    methods = list(method_magnitudes.keys())
    logmags, zmin, zmax = _peak_normalized_log(method_magnitudes)

    n_panels = 1 + len(methods)
    cols = min(5, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    flat = axes.ravel()
    # Original (linear gray) for reference.
    flat[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    flat[0].set_title("original", fontsize=9)
    flat[0].set_xticks([])
    flat[0].set_yticks([])

    im = None
    for ax, name in zip(flat[1:], methods):
        lm = np.rot90(logmags[name], k=1)  # match Julia's rotr90
        im = ax.imshow(lm, cmap="inferno", vmin=zmin, vmax=zmax)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flat[n_panels:]:
        ax.set_axis_off()
    if im is not None:
        cbar = fig.colorbar(im, ax=flat[1:n_panels].tolist(), shrink=0.85, location="right")
        cbar.set_label(r"log$_{10}\,|F|/\max|F|$", fontsize=9)
    fig.suptitle(
        f"Frequency-domain magnitude (peak-normalised log10) — test image #{image_idx}",
        fontsize=10, fontweight="bold",
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_frequency_spectra_3d(
    image_idx: int,
    method_magnitudes: dict[str, np.ndarray],
    out_pdf: Path,
) -> None:
    """3D surface view of peak-normalised log|F|, downsampled for render.

    Mirror of `analyze_frequency_space.jl` (A2) panel. Downsamples to ~128 per
    axis so the rendered surface stays tractable on 1024×1024 images.
    """
    methods = list(method_magnitudes.keys())
    logmags, zmin, zmax = _peak_normalized_log(method_magnitudes)
    h, w = next(iter(method_magnitudes.values())).shape
    ds = max(1, h // 128)

    n = len(methods)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4.0 * cols, 3.5 * rows))
    fig.suptitle(
        f"3D frequency spectra (peak-normalised log10) — test image #{image_idx}",
        fontsize=10, fontweight="bold",
    )
    for k, name in enumerate(methods):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        z = logmags[name][::ds, ::ds]
        xs = np.arange(z.shape[1])
        ys = np.arange(z.shape[0])
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(X, Y, z, cmap="inferno", vmin=zmin, vmax=zmax,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("col idx", fontsize=7)
        ax.set_ylabel("row idx", fontsize=7)
        ax.set_zlabel(r"log$_{10}\,|F|/\max$", fontsize=7)
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=22, azim=117)  # ≈ Julia's azimuth=0.65π, elevation=0.22π
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _energy_captured(magnitude: np.ndarray, keep_ratio: float) -> float:
    """Fraction of total L2 energy retained when keeping top-k% by magnitude.

    Mirror of `e_fft / e_dct / e_bdct / e_qft` per-ratio fields in
    `analyze_frequency_space.jl::analyze_image`.
    """
    e = magnitude.flatten() ** 2
    total = float(np.sum(e))
    if total <= 0:
        return 0.0
    k = max(1, int(np.floor(e.size * keep_ratio)))
    top_e = np.sort(e)[::-1][:k]
    return float(np.sum(top_e)) / total


def _draw_cumulative_energy(
    image_idx: int,
    method_magnitudes: dict[str, np.ndarray],
    keep_ratios: Sequence[float],
    out_pdf: Path,
) -> None:
    """Cumulative captured-energy curve vs fraction of coefficients kept.

    Log-scale x-axis matches `analyze_frequency_space.jl::ax_cum.xscale=log10`.
    Vertical dashed lines mark the four headline keep_ratios.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))
    n_total = next(iter(method_magnitudes.values())).size
    xs = np.arange(1, n_total + 1) / n_total
    for name, mag in method_magnitudes.items():
        ce = _cumulative_energy(mag)
        ax.plot(xs, ce, label=name, linewidth=1.5, alpha=0.95)
    for kr in keep_ratios:
        ax.axvline(kr, color="grey", linestyle="--", alpha=0.6, linewidth=1.0)
    ax.set_xlabel("Fraction of coefficients kept (sorted by magnitude)")
    ax.set_ylabel("Fraction of total L2 energy")
    ax.set_title(
        f"Energy captured vs. fraction kept — test image #{image_idx}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlim(1.0 / n_total, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_kept_coefficient_masks(
    image_idx: int,
    method_magnitudes: dict[str, np.ndarray],
    keep_ratios: Sequence[float],
    out_pdf: Path,
) -> None:
    """Binary masks of which coefficients are kept, per method × keep_ratio."""
    methods = list(method_magnitudes.keys())
    rows, cols = len(methods), len(keep_ratios)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows), squeeze=False)
    for i, name in enumerate(methods):
        mag = method_magnitudes[name]
        for j, kr in enumerate(keep_ratios):
            k = max(1, int(np.floor(mag.size * kr)))
            flat = mag.flatten()
            idx = np.argpartition(-flat, k)[:k]
            mask_flat = np.zeros_like(flat, dtype=bool)
            mask_flat[idx] = True
            mask = mask_flat.reshape(mag.shape)
            axes[i][j].imshow(mask, cmap="gray", vmin=0, vmax=1)
            if i == 0:
                axes[i][j].set_title(f"keep={kr:g}", fontsize=8)
            if j == 0:
                axes[i][j].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    fig.suptitle(f"Kept-coefficient masks — test image #{image_idx}", fontsize=10)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_grid(
    image: np.ndarray,
    image_idx: int,
    method_recoveries: dict[str, dict[float, np.ndarray]],
    keep_ratios: Sequence[float],
    out_pdf: Path,
) -> None:
    """One PDF per image. Rows = methods (+ original on top), cols = keep ratios.

    method_recoveries: {method_name: {keep_ratio: recovered_image_or_None}}
    """
    methods = list(method_recoveries.keys())
    n_rows = 1 + len(methods)  # +1 for the original-image row
    n_cols = len(keep_ratios)

    # Each cell ~2.0 inches; total figure ~ (2*n_cols) x (2*n_rows). Keep tight.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.0 * n_cols, 2.0 * n_rows),
        squeeze=False,
    )

    # Top row: original image, repeated across all columns for visual reference.
    for j, kr in enumerate(keep_ratios):
        ax = axes[0][j]
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("original", rotation=0, ha="right", va="center", fontsize=8)
        ax.set_title(f"keep={kr:g}", fontsize=8)

    # Method rows.
    for i, method in enumerate(methods):
        for j, kr in enumerate(keep_ratios):
            ax = axes[i + 1][j]
            rec = method_recoveries[method].get(kr)
            if rec is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.imshow(rec, cmap="gray", vmin=0.0, vmax=1.0)
                psnr = _psnr_clamped(image, rec)
                # SSIM via skimage. Caption mirrors Julia's analyze_frequency_space.jl
                # `caption = "X% kept — PSNR Y dB, SSIM Z"`.
                try:
                    from skimage.metrics import structural_similarity

                    rec_clipped = np.clip(np.real(rec), 0.0, 1.0)
                    ssim = float(structural_similarity(image, rec_clipped, data_range=1.0))
                except Exception:
                    ssim = float("nan")
                psnr_str = f"{psnr:5.2f} dB" if np.isfinite(psnr) else "inf"
                ssim_str = f"{ssim:.3f}" if np.isfinite(ssim) else "n/a"
                ax.text(
                    0.02,
                    0.98,
                    f"{psnr_str}\nSSIM {ssim_str}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none"),
                )
                ax.set_xticks([])
                ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(method, rotation=0, ha="right", va="center", fontsize=8)

    fig.suptitle(f"Reconstructions — test image #{image_idx}", fontsize=10)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    image: np.ndarray,
    image_idx: int,
    method_recoveries: dict[str, dict[float, np.ndarray]],
    keep_ratios: Sequence[float],
    out_path: Path,
    method_magnitudes: dict[str, np.ndarray] | None = None,
) -> None:
    """Plain-text summary mirror of `analyze_frequency_space.jl::analyze_image`'s
    return: per-(method, keep_ratio) PSNR, SSIM, and energy fraction.
    """
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        structural_similarity = None  # type: ignore[assignment]

    lines = [
        f"Reconstruction summary — test image #{image_idx}",
        f"Shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]",
        "",
        "PSNR (dB):",
        "method            " + "  ".join(f"keep={kr:5g}" for kr in keep_ratios),
    ]
    for method, by_kr in method_recoveries.items():
        cells = []
        for kr in keep_ratios:
            rec = by_kr.get(kr)
            if rec is None:
                cells.append("    n/a")
            else:
                psnr = _psnr_clamped(image, rec)
                cells.append(f"{psnr:7.2f}" if np.isfinite(psnr) else "    inf")
        lines.append(f"{method:18s}" + "  ".join(cells))

    lines.extend(["", "SSIM:", "method            " + "  ".join(f"keep={kr:5g}" for kr in keep_ratios)])
    for method, by_kr in method_recoveries.items():
        cells = []
        for kr in keep_ratios:
            rec = by_kr.get(kr)
            if rec is None or structural_similarity is None:
                cells.append("    n/a")
            else:
                rec_clipped = np.clip(np.real(rec), 0.0, 1.0)
                try:
                    ssim = float(structural_similarity(image, rec_clipped, data_range=1.0))
                    cells.append(f"{ssim:7.3f}")
                except Exception:
                    cells.append("    n/a")
        lines.append(f"{method:18s}" + "  ".join(cells))

    if method_magnitudes:
        lines.extend([
            "",
            "Fraction of L2 energy captured by top-k% (Parseval-equivalent):",
            "method            " + "  ".join(f"keep={kr:5g}" for kr in keep_ratios),
        ])
        for method, mag in method_magnitudes.items():
            cells = []
            for kr in keep_ratios:
                cells.append(f"{_energy_captured(mag, kr):7.4f}")
            lines.append(f"{method:18s}" + "  ".join(cells))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _resolve_basis(host_bases_entry, image_idx: int):
    """Return the basis to use for image `image_idx`.

    `host_bases_entry` is either a single shared basis (new pipeline, returned
    by `train_one_basis_batched`) or a list of per-image bases (legacy P-pair
    pipeline). The shared form is preferred; lists are kept for back-compat.
    """
    if isinstance(host_bases_entry, list):
        if image_idx >= len(host_bases_entry):
            return None
        return host_bases_entry[image_idx]
    return host_bases_entry


def analyze_reconstructions(
    test_images: np.ndarray,
    host_bases: dict[str, Any],
    baseline_fns: dict[str, Callable[[np.ndarray, float], np.ndarray]],
    keep_ratios: Sequence[float],
    out_dir: Path,
    *,
    max_images: int | None = None,
) -> None:
    """Produce per-image reconstruction PDFs + summaries.

    `host_bases[name]` may be either a single shared basis (new pipeline) or
    a list of per-image bases (legacy P-pair pipeline). Either way the basis
    must already be host-resident.
    """
    n = len(test_images) if max_images is None else min(len(test_images), max_images)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_ratios = list(keep_ratios)

    for i in range(n):
        img = test_images[i]
        recoveries: dict[str, dict[float, np.ndarray]] = {}

        for basis_name, basis_entry in host_bases.items():
            basis = _resolve_basis(basis_entry, i)
            if basis is None:
                recoveries[basis_name] = {kr: None for kr in keep_ratios}
                continue
            per_kr: dict[float, np.ndarray] = {}
            for kr in keep_ratios:
                try:
                    per_kr[kr] = _recover_basis(basis, img, kr)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "analyze: basis=%s img=%d kr=%s recover failed: %s",
                        basis_name,
                        i,
                        kr,
                        e,
                    )
                    per_kr[kr] = None
            recoveries[basis_name] = per_kr

        for baseline_name, fn in baseline_fns.items():
            per_kr = {}
            for kr in keep_ratios:
                try:
                    rec = fn(np.asarray(img, dtype=np.float64), kr)
                    per_kr[kr] = np.clip(np.real(rec), 0.0, 1.0)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "analyze: baseline=%s img=%d kr=%s failed: %s",
                        baseline_name,
                        i,
                        kr,
                        e,
                    )
                    per_kr[kr] = None
            recoveries[baseline_name] = per_kr

        per_image_dir = out_dir / f"{i:04d}"
        _draw_grid(img, i, recoveries, keep_ratios, per_image_dir / "reconstructions.pdf")

        # Frequency-domain analysis (mirrors Julia's analyze_frequency_space.jl).
        method_magnitudes: dict[str, np.ndarray] = {}
        try:
            for basis_name, basis_entry in host_bases.items():
                basis = _resolve_basis(basis_entry, i)
                if basis is None:
                    continue
                method_magnitudes[basis_name] = _forward_magnitude(
                    basis, np.asarray(img, dtype=np.float64)
                )
            for baseline_name in baseline_fns.keys():
                method_magnitudes[baseline_name] = _baseline_freq_magnitude(
                    baseline_name, np.asarray(img, dtype=np.float64)
                )
            if method_magnitudes:
                _draw_frequency_spectra(
                    img, i, method_magnitudes, per_image_dir / "frequency_spectra.pdf"
                )
                _draw_frequency_spectra_3d(
                    i, method_magnitudes, per_image_dir / "frequency_spectra_3d.pdf"
                )
                _draw_cumulative_energy(
                    i, method_magnitudes, keep_ratios, per_image_dir / "cumulative_energy.pdf"
                )
                _draw_kept_coefficient_masks(
                    i, method_magnitudes, keep_ratios,
                    per_image_dir / "kept_coefficient_masks.pdf",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("analyze: frequency-domain plots failed for img=%d: %s", i, e)

        # Summary writer last so it includes Parseval-energy fractions if magnitudes are available.
        _write_summary(
            img, i, recoveries, keep_ratios,
            per_image_dir / "summary.txt",
            method_magnitudes=method_magnitudes if method_magnitudes else None,
        )

    logger.info("analysis: wrote %d per-image analysis bundles under %s", n, out_dir)
