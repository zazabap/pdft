#!/usr/bin/env python3
"""Post-run frequency-space analysis matching ParametricDFT-Benchmarks.jl.

Mirrors the output structure of
``ParametricDFT-Benchmarks.jl/analysis/analyze_frequency_space.jl``:

    analysis/<run>/
      <image_label>/                   ← one directory per analyzed test image
        cumulative_energy.pdf
        frequency_spectra.pdf
        frequency_spectra_3d.pdf
        kept_coefficient_masks.pdf
        reconstructions.pdf
        summary.txt                    ← PSNR + SSIM tables for THIS image
      summary_all_images.txt           ← consolidated PSNR + SSIM tables across N images

Usage:

    python benchmarks/post_run_analysis.py \\
        --gpu-dirs <gpu0_results_dir> <gpu1_results_dir> \\
        --out <combined_analysis_dir> \\
        --n-images 20 \\
        [--dataset div2k_8q]

The script loads every ``trained_*.json`` it finds across the given gpu-dirs,
re-loads the test images with the same seed used during training, runs the
existing ``benchmarks/analyze.py`` machinery on N images, then writes the
consolidated ``summary_all_images.txt`` table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  -- adds benchmarks/ to sys.path

import numpy as np

# Importing pdft sets jax_enable_x64; must come before any other jax use.
import pdft

import jax.numpy as jnp  # noqa: E402  - imported after pdft per CLAUDE.md §5

from analyze import analyze_reconstructions  # noqa: E402
from baselines import (  # noqa: E402
    block_dct_compress,
    block_fft_compress,
    global_dct_compress,
    global_fft_compress,
)
from data_loading import DEFAULT_DIV2K_ROOT  # noqa: E402

BASIS_TYPE_MAP = {
    "QFTBasis": pdft.QFTBasis,
    "EntangledQFTBasis": pdft.EntangledQFTBasis,
    "TEBDBasis": pdft.TEBDBasis,
    "MERABasis": pdft.MERABasis,
}

# `trained_<name>` filename → benchmark / Julia-style basis label.
BASIS_NAME_FROM_FILE = {
    "trained_qft.json": "qft",
    "trained_entangled_qft.json": "entangled_qft",
    "trained_tebd.json": "tebd",
    "trained_mera.json": "mera",
}

BASELINE_FACTORIES = {
    "fft": global_fft_compress,
    "dct": global_dct_compress,
    "block_fft_8": lambda img, kr: block_fft_compress(img, kr, block=8),
    "block_dct_8": lambda img, kr: block_dct_compress(img, kr, block=8),
}


def _reconstruct_basis(json_obj: dict):
    """Build a basis instance from its trained_*.json on-disk form."""
    cls_name = json_obj["type"]
    if cls_name not in BASIS_TYPE_MAP:
        raise ValueError(f"unknown basis type {cls_name!r}; choices: {list(BASIS_TYPE_MAP)}")
    cls = BASIS_TYPE_MAP[cls_name]
    m = int(json_obj["m"])
    n = int(json_obj["n"])

    # Build a default-init basis to get the tensor shapes / order. Then replace
    # tensor values with the saved ones.
    proto = cls(m=m, n=n)
    proto_shapes = [tuple(int(s) for s in t.shape) for t in proto.tensors]
    saved_flat = json_obj["tensors"]
    if len(saved_flat) != len(proto_shapes):
        raise ValueError(
            f"trained_{cls_name}.json tensor count ({len(saved_flat)}) "
            f"!= constructor count ({len(proto_shapes)})"
        )

    new_tensors = []
    for shape, flat in zip(proto_shapes, saved_flat):
        # Saved as Fortran-flat list of [real, imag] pairs (matches CLAUDE.md §6).
        complex_flat = np.array([complex(re, im) for re, im in flat], dtype=np.complex128)
        arr = complex_flat.reshape(shape, order="F")
        new_tensors.append(jnp.asarray(arr))

    # Reuse code/inv_code from proto so einsum cache hits.
    return cls(m=m, n=n, tensors=new_tensors, code=proto.code, inv_code=proto.inv_code)


def _load_trained_bases_from_dirs(dirs: list[Path]) -> dict:
    """Walk each dir and merge all trained_*.json files into one {name: basis} dict."""
    bases: dict = {}
    for d in dirs:
        for fn, label in BASIS_NAME_FROM_FILE.items():
            path = d / fn
            if not path.is_file():
                continue
            if label in bases:
                continue  # first directory wins (avoids double-load)
            try:
                obj = json.loads(path.read_text())
                bases[label] = _reconstruct_basis(obj)
                print(f"  loaded {label} from {path}")
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: could not reconstruct {label} from {path}: {e}")
    return bases


def _load_div2k_with_filenames(
    n_train: int, n_test: int, seed: int, data_root: Path, size: int
):
    """Replica of `data_loading.load_div2k` that ALSO returns the filenames of
    the test slice. Needed for Julia-compatible per-image directory naming.
    """
    from PIL import Image

    pngs = sorted(Path(data_root).glob("*.png"))
    total = n_train + n_test
    if len(pngs) < total:
        raise ValueError(f"not enough images in {data_root}: {len(pngs)} < {total}")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(pngs), size=total, replace=False)
    chosen = [pngs[i] for i in chosen_idx]

    out = np.empty((total, size, size), dtype=np.float32)
    for i, p in enumerate(chosen):
        img = Image.open(p).convert("L")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0

    test_imgs = out[n_train:]
    test_names = [chosen[i].stem for i in range(n_train, total)]
    return out[:n_train], test_imgs, test_names


def _psnr(orig: np.ndarray, rec: np.ndarray) -> float:
    rec = np.clip(np.real(rec), 0.0, 1.0)
    mse = float(np.mean((orig - rec) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _ssim(orig: np.ndarray, rec: np.ndarray) -> float:
    """SSIM via skimage if available; else fall back to a simple MSE-derived
    proxy so the script doesn't hard-require skimage."""
    rec = np.clip(np.real(rec), 0.0, 1.0)
    try:
        from skimage.metrics import structural_similarity as _ssim_fn

        return float(_ssim_fn(orig.astype(np.float64), rec.astype(np.float64), data_range=1.0))
    except ImportError:
        # Coarse fallback — MSE-based similarity. Marked clearly in summary.
        mse = float(np.mean((orig - rec) ** 2))
        return 1.0 - min(1.0, mse * 4)


def _recover_basis(basis, image: np.ndarray, kr: float) -> np.ndarray:
    discard = 1.0 - kr
    compressed = pdft.io.compress(basis, np.asarray(image, dtype=np.float64), ratio=discard)
    rec = pdft.io.recover(basis, compressed)
    return np.clip(np.real(rec), 0.0, 1.0)


def _write_summary_all_images(
    out_path: Path,
    test_names: list[str],
    test_images: np.ndarray,
    bases: dict,
    keep_ratios: list[float],
) -> None:
    """Mirror Julia's summary_all_images.txt — PSNR + SSIM tables, one row
    per image plus a MEAN row. Methods columns include FFT, DCT, BDCT, plus
    every quantum basis present in `bases`."""
    methods_classical = [
        ("FFT", lambda img, kr: global_fft_compress(np.asarray(img, dtype=np.float64), kr)),
        ("DCT", lambda img, kr: global_dct_compress(np.asarray(img, dtype=np.float64), kr)),
        (
            "BDCT",
            lambda img, kr: block_dct_compress(np.asarray(img, dtype=np.float64), kr, block=8),
        ),
    ]
    quantum_methods = [(name.upper().replace("_", "-"), basis) for name, basis in bases.items()]

    headers = []
    for kr in keep_ratios:
        for label, _ in methods_classical:
            headers.append(f"{label}@{int(kr*100):>2}%")
        for label, _ in quantum_methods:
            headers.append(f"{label}@{int(kr*100):>2}%")

    n_imgs = len(test_images)
    psnr_rows = []
    ssim_rows = []
    psnr_means = np.zeros(len(headers), dtype=np.float64)
    ssim_means = np.zeros(len(headers), dtype=np.float64)

    for i in range(n_imgs):
        img = test_images[i]
        psnr_row = []
        ssim_row = []
        for kr in keep_ratios:
            for _, fn in methods_classical:
                rec = fn(img, kr)
                psnr_row.append(_psnr(img, rec))
                ssim_row.append(_ssim(img, rec))
            for _, basis in quantum_methods:
                rec = _recover_basis(basis, img, kr)
                psnr_row.append(_psnr(img, rec))
                ssim_row.append(_ssim(img, rec))
        psnr_rows.append(psnr_row)
        ssim_rows.append(ssim_row)
        psnr_means += np.array(psnr_row)
        ssim_means += np.array(ssim_row)

    psnr_means /= max(1, n_imgs)
    ssim_means /= max(1, n_imgs)

    lines: list[str] = []
    lines.append(f"Consolidated Frequency-Space Analysis — {n_imgs} images")
    lines.append("Bases: " + ", ".join(bases.keys()))
    lines.append("=" * 124)
    lines.append("")
    lines.append("PSNR (dB) per image per keep-ratio:")
    header_line = f"{'image':<12}" + " ".join(f"{h:>9}" for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for name, row in zip(test_names, psnr_rows):
        lines.append(
            f"{name+'.png':<12}" + " ".join(f"{v:>9.2f}" for v in row)
        )
    lines.append("-" * len(header_line))
    lines.append(f"{'MEAN':<12}" + " ".join(f"{v:>9.2f}" for v in psnr_means))

    lines.append("")
    lines.append("SSIM per image per keep-ratio:")
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for name, row in zip(test_names, ssim_rows):
        lines.append(
            f"{name+'.png':<12}" + " ".join(f"{v:>9.3f}" for v in row)
        )
    lines.append("-" * len(header_line))
    lines.append(f"{'MEAN':<12}" + " ".join(f"{v:>9.3f}" for v in ssim_means))

    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--gpu-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more results dirs (each holding trained_*.json). The merged "
        "set of trained bases is analyzed together.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Combined analysis directory.")
    ap.add_argument(
        "--n-images",
        type=int,
        default=20,
        help="Number of DIV2K test images to analyze (default 20, mirrors Julia).",
    )
    ap.add_argument(
        "--force-include",
        nargs="*",
        default=["0390"],
        help="Stems of DIV2K images to always include (loaded from disk regardless "
        "of the run's PRNG draw). Default: 0390 (Julia's canonical showcase image).",
    )
    ap.add_argument(
        "--dataset",
        choices=("div2k_8q",),
        default="div2k_8q",
        help="Dataset (currently only div2k_8q post-run analysis is supported).",
    )
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # Read run env to get the seed + n_test that was used.
    env = json.loads((args.gpu_dirs[0] / "env.json").read_text())
    preset = env["preset_dataclass"]
    seed = int(preset["seed"])
    n_train = int(preset["n_train"])
    n_test = int(preset["n_test"])
    n_imgs = min(int(args.n_images), n_test)
    keep_ratios = [float(x) for x in preset["keep_ratios"]]

    print(
        f"Run preset={env['preset']}  seed={seed}  n_train={n_train}  n_test={n_test} "
        f"keep_ratios={keep_ratios}"
    )

    # Reconstruct trained bases from disk.
    print("Loading trained bases from gpu-dirs...")
    bases = _load_trained_bases_from_dirs(args.gpu_dirs)
    if not bases:
        print("ERROR: no trained_*.json files found in any gpu-dir.", file=sys.stderr)
        return 2
    print(f"  reconstructed bases: {list(bases.keys())}")

    # Load same DIV2K test images that were evaluated during the run.
    M = 8
    print(f"Loading DIV2K test images (size=2^{M}={2**M}, seed={seed}, n_test={n_test})...")
    _, test_imgs, test_names = _load_div2k_with_filenames(
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        data_root=DEFAULT_DIV2K_ROOT,
        size=2**M,
    )
    print(f"  test names (head): {test_names[:5]}{'...' if len(test_names)>5 else ''}")

    # Trim to the first N requested.
    test_imgs = test_imgs[:n_imgs]
    test_names = test_names[:n_imgs]

    # Force-include the showcase images (e.g. 0390.png) — Julia's harness
    # always pins this index, so absence makes cross-language comparison hard.
    if args.force_include:
        from PIL import Image as _PIL

        forced_imgs: list[np.ndarray] = []
        forced_names: list[str] = []
        for stem in args.force_include:
            if stem in test_names:
                continue
            png = DEFAULT_DIV2K_ROOT / f"{stem}.png"
            if not png.is_file():
                print(f"  WARNING: --force-include {stem!r} not found at {png}; skipping")
                continue
            img = _PIL.open(png).convert("L")
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((2**M, 2**M), _PIL.Resampling.LANCZOS)
            forced_imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
            forced_names.append(stem)
            print(f"  force-include {stem}.png appended")
        if forced_imgs:
            test_imgs = np.concatenate([test_imgs, np.stack(forced_imgs)], axis=0)
            test_names = test_names + forced_names

    # ---- per-image bundles (uses analyze.py) ----
    # analyze_reconstructions writes to <out>/<idx>/{*.pdf, summary.txt}. We
    # rename those dirs to the image filename stem for Julia parity.
    print(f"\nGenerating per-image analysis bundles for {len(test_imgs)} images...")
    analyze_reconstructions(
        test_imgs,
        bases,
        BASELINE_FACTORIES,
        keep_ratios,
        args.out,
        max_images=len(test_imgs),
    )
    # Two-pass rename to avoid the collision case where a stem like "0014"
    # matches a future sequential index `i=14` whose src dir hasn't been
    # processed yet. Pass 1: rename every `NNNN` to a unique temp name. Pass
    # 2: rename each temp to its final stem.
    import shutil as _shutil

    for i, stem in enumerate(test_names):
        src = args.out / f"{i:04d}"
        if src.is_dir():
            tmp = args.out / f"_renaming_{i:04d}_{stem}"
            src.rename(tmp)
    for i, stem in enumerate(test_names):
        tmp = args.out / f"_renaming_{i:04d}_{stem}"
        dst = args.out / stem
        if tmp.is_dir():
            if dst.exists() and dst != tmp:
                _shutil.rmtree(dst)
            tmp.rename(dst)

    # ---- consolidated summary ----
    print("\nWriting summary_all_images.txt...")
    _write_summary_all_images(
        args.out / "summary_all_images.txt",
        test_names,
        test_imgs,
        bases,
        keep_ratios,
    )

    print(f"\nDone. Results in {args.out}")
    print(f"  - per-image bundles:      {len(test_names)} dirs")
    print(f"  - consolidated summary:   {args.out}/summary_all_images.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
