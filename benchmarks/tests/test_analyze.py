"""Layer A: analyze.py reconstruction comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analyze import analyze_reconstructions


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    test = rng.uniform(0.0, 1.0, size=(3, 8, 8)).astype(np.float64)
    return test


def test_analyze_writes_per_image_pdfs(synthetic_data, tmp_path: Path):
    """With no quantum bases (empty host_bases) and stub baselines, every
    image still gets a PDF + summary.txt."""

    def passthrough(img: np.ndarray, kr: float) -> np.ndarray:
        return img

    def half(img: np.ndarray, kr: float) -> np.ndarray:
        return img * 0.5

    analyze_reconstructions(
        synthetic_data,
        host_bases={},
        baseline_fns={"passthrough": passthrough, "half": half},
        keep_ratios=(0.1, 0.2),
        out_dir=tmp_path,
    )

    for i in range(3):
        sub = tmp_path / f"{i:04d}"
        pdf = sub / "reconstructions.pdf"
        summary = sub / "summary.txt"
        assert pdf.is_file()
        with open(pdf, "rb") as f:
            assert f.read(5) == b"%PDF-"
        assert summary.is_file()
        text = summary.read_text()
        assert "passthrough" in text
        assert "half" in text


def test_analyze_handles_basis_failure(synthetic_data, tmp_path: Path):
    """A basis whose recover raises has 'n/a' tile and entry in summary."""

    class BadBasis:
        m = 2
        n = 2
        tensors = ()

    # Stub host_bases with a 'qft' entry whose recover will fail because
    # BadBasis isn't a real pdft basis. analyze_reconstructions should catch
    # the per-(image, kr) failures and continue.
    analyze_reconstructions(
        synthetic_data[:1],
        host_bases={"qft": [BadBasis()]},
        baseline_fns={},
        keep_ratios=(0.1,),
        out_dir=tmp_path,
    )

    pdf = tmp_path / "0000" / "reconstructions.pdf"
    assert pdf.is_file()
    text = (tmp_path / "0000" / "summary.txt").read_text()
    assert "qft" in text
    assert "n/a" in text


def test_analyze_max_images_caps(synthetic_data, tmp_path: Path):
    analyze_reconstructions(
        synthetic_data,
        host_bases={},
        baseline_fns={"id": lambda img, kr: img},
        keep_ratios=(0.5,),
        out_dir=tmp_path,
        max_images=2,
    )

    assert (tmp_path / "0000").is_dir()
    assert (tmp_path / "0001").is_dir()
    assert not (tmp_path / "0002").exists()
