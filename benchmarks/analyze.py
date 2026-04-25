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
from typing import Callable, Sequence

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
    compressed = pdft.compress(basis, np.asarray(image, dtype=np.float64), ratio=discard_ratio)
    recovered = pdft.recover(basis, compressed)
    return np.clip(np.real(recovered), 0.0, 1.0)


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
                psnr_str = f"{psnr:5.2f} dB" if np.isfinite(psnr) else "inf"
                ax.text(
                    0.02,
                    0.98,
                    psnr_str,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
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
) -> None:
    lines = [
        f"Reconstruction summary — test image #{image_idx}",
        f"Shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]",
        "",
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def analyze_reconstructions(
    test_images: np.ndarray,
    host_bases: dict[str, list],
    baseline_fns: dict[str, Callable[[np.ndarray, float], np.ndarray]],
    keep_ratios: Sequence[float],
    out_dir: Path,
    *,
    max_images: int | None = None,
) -> None:
    """Produce per-image reconstruction PDFs + summaries.

    `host_bases[name][i]` is the per-image basis for test image `i` (P pairing).
    Bases that have fewer entries than `len(test_images)` are silently
    truncated to their available length.
    """
    n = len(test_images) if max_images is None else min(len(test_images), max_images)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_ratios = list(keep_ratios)

    for i in range(n):
        img = test_images[i]
        recoveries: dict[str, dict[float, np.ndarray]] = {}

        for basis_name, basis_list in host_bases.items():
            if i >= len(basis_list):
                recoveries[basis_name] = {kr: None for kr in keep_ratios}
                continue
            basis = basis_list[i]
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
        _write_summary(img, i, recoveries, keep_ratios, per_image_dir / "summary.txt")

    logger.info("analysis: wrote %d per-image reconstruction PDFs under %s", n, out_dir)
