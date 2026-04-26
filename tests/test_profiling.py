"""Smoke + property tests for `pdft.profiling.profile_training`.

Runs at m=n=2 (16 elements per image) on whatever device JAX picks. Even on CPU
this completes in a few seconds. Covers:
  - profile_training returns a populated ProfileReport
  - phase tagging (compile / warm / val) is correct
  - val_every interleaves val records every K steps
  - dataset shorter than n_steps * batch_size is cycled to fit
  - rejected configs (non-adam, empty dataset)
  - ProfileReport.summary() and to_csv() produce expected shape/content
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import pdft
from pdft.profiling import ProfileReport, StepRecord, _maybe_trace


M, N = 2, 2
SIZE = 2**M


def _synthetic_imgs(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        (rng.standard_normal((SIZE, SIZE)) + 1j * rng.standard_normal((SIZE, SIZE))).astype(
            np.complex128
        )
        for _ in range(n)
    ]


def test_profile_training_returns_populated_report():
    basis = pdft.QFTBasis(m=M, n=N)
    loss = pdft.MSELoss(k=2)
    imgs = _synthetic_imgs(6)

    report = pdft.profile_training(
        basis,
        dataset=imgs,
        loss=loss,
        n_steps=3,
        batch_size=2,
        val_every=0,
        trace_dir=None,
    )

    assert isinstance(report, ProfileReport)
    assert report.basis_class == "QFTBasis"
    assert report.m == M
    assert report.n == N
    assert report.batch_size == 2
    assert report.n_steps == 3
    assert len(report.records) == 3
    # First record is "compile", subsequent are "warm".
    assert report.records[0].phase == "compile"
    assert all(r.phase == "warm" for r in report.records[1:])
    assert report.total_s > 0
    # Loss values are finite floats.
    assert all(r.loss is not None and np.isfinite(r.loss) for r in report.records)


def test_profile_training_with_val_eval_interleaves_records():
    basis = pdft.QFTBasis(m=M, n=N)
    loss = pdft.MSELoss(k=2)
    imgs = _synthetic_imgs(8)
    val_imgs = _synthetic_imgs(2, seed=1)

    report = pdft.profile_training(
        basis,
        dataset=imgs,
        loss=loss,
        n_steps=4,
        batch_size=2,
        val_every=2,
        val_dataset=val_imgs,
        trace_dir=None,
    )

    # 4 train records (1 compile + 3 warm) + 2 val records (every 2 steps).
    train = [r for r in report.records if r.phase in ("compile", "warm")]
    val = [r for r in report.records if r.phase == "val"]
    assert len(train) == 4
    assert len(val) == 2
    assert all(np.isfinite(r.loss) for r in val)


def test_profile_training_cycles_short_dataset():
    """Dataset shorter than n_steps * batch_size is cycled, not rejected."""
    basis = pdft.QFTBasis(m=M, n=N)
    loss = pdft.MSELoss(k=2)
    imgs = _synthetic_imgs(2)  # only 2 images for 6 needed

    report = pdft.profile_training(
        basis, dataset=imgs, loss=loss, n_steps=3, batch_size=2, trace_dir=None
    )
    assert len(report.records) == 3


def test_profile_training_rejects_non_adam():
    basis = pdft.QFTBasis(m=M, n=N)
    loss = pdft.MSELoss(k=2)
    imgs = _synthetic_imgs(2)
    with pytest.raises(NotImplementedError, match="adam"):
        pdft.profile_training(
            basis, dataset=imgs, loss=loss, n_steps=2, batch_size=1, optimizer="gd"
        )


def test_profile_training_rejects_empty_dataset():
    basis = pdft.QFTBasis(m=M, n=N)
    loss = pdft.MSELoss(k=2)
    with pytest.raises(ValueError, match="non-empty"):
        pdft.profile_training(basis, dataset=[], loss=loss, n_steps=2, batch_size=1)


def test_report_summary_contains_key_fields():
    basis = pdft.QFTBasis(m=M, n=N)
    loss = pdft.MSELoss(k=2)
    imgs = _synthetic_imgs(4)
    report = pdft.profile_training(
        basis, dataset=imgs, loss=loss, n_steps=2, batch_size=2, trace_dir=None
    )
    summary = report.summary()
    assert "QFTBasis" in summary
    assert "m=2" in summary
    assert "bs=2" in summary
    assert "per-step" in summary
    assert "warm steps" in summary


def test_report_summary_with_no_warm_records():
    """ProfileReport.summary() handles the edge case where all records are compile/val."""
    report = ProfileReport(
        basis_class="QFTBasis",
        m=2,
        n=2,
        batch_size=2,
        n_steps=1,
        device="cpu:0",
    )
    # n_steps=1 with only a compile record (no warm).
    report.records = [StepRecord(step=0, phase="compile", wall_s=1.0, loss=42.0)]
    summary = report.summary()
    assert "no warm steps recorded" in summary


def test_report_to_csv_roundtrip(tmp_path: Path):
    basis = pdft.QFTBasis(m=M, n=N)
    loss = pdft.MSELoss(k=2)
    imgs = _synthetic_imgs(4)
    report = pdft.profile_training(
        basis, dataset=imgs, loss=loss, n_steps=2, batch_size=2, trace_dir=None
    )
    out = tmp_path / "step_times.csv"
    report.to_csv(out)
    assert out.is_file()

    with out.open() as f:
        rows = list(csv.reader(f))
    # Header + 2 data rows for 2 steps.
    assert rows[0] == ["step", "phase", "wall_s", "loss"]
    assert len(rows) == 1 + len(report.records)
    # Each data row's numeric fields parse cleanly.
    for r in rows[1:]:
        assert r[1] in ("compile", "warm", "val")
        assert float(r[2]) > 0
        assert float(r[3])  # parses


def test_report_to_csv_handles_none_loss(tmp_path: Path):
    """StepRecord.loss=None should round-trip through to_csv as empty string."""
    out = tmp_path / "x.csv"
    rep = ProfileReport(
        basis_class="X", m=1, n=1, batch_size=1, n_steps=1, device="cpu:0",
    )
    rep.records = [StepRecord(step=0, phase="warm", wall_s=0.1, loss=None)]
    rep.to_csv(out)
    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[1][3] == ""  # loss column blank when None


def test_maybe_trace_with_none_is_noop():
    """trace_dir=None must yield a context manager that produces None."""
    with _maybe_trace(None) as t:
        assert t is None
