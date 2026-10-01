"""The package computes today what it computed when the snapshots were taken.

These are not correctness tests: the parity tests against Julia are. They pin
the behaviour of every basis and trainer path so a refactor that is meant to
change nothing can be shown to change nothing, including where no Julia golden
exists (generic tensors, inverses, gradients, the dense and controlled gates,
the batched trainer).

By default a value may differ from its snapshot at rounding level, which keeps
the tests portable across machines and Python versions. ``PDFT_SNAPSHOT_EXACT=1``
demands the same bits and is meant for the machine that generated the file;
``PDFT_SNAPSHOT_FILE`` points the tests at a locally generated one. See
``regenerate.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from .cases import BASES, SNAPSHOT_PATH, TRAININGS, static_record, training_record

EXACT = bool(os.environ.get("PDFT_SNAPSHOT_EXACT"))
STORED = np.load(os.environ.get("PDFT_SNAPSHOT_FILE", SNAPSHOT_PATH))


def _check(prefix: str, record: dict[str, np.ndarray], *, rtol: float, atol: float) -> None:
    stored = {k[len(prefix) :]: STORED[k] for k in STORED.files if k.startswith(prefix)}
    assert sorted(record) == sorted(stored), "the set of recorded arrays changed"
    for name, value in record.items():
        want = stored[name]
        assert value.shape == want.shape and value.dtype == want.dtype, name
        if EXACT or value.dtype.kind in "iuU":
            assert np.array_equal(value, want), name
        else:
            np.testing.assert_allclose(value, want, rtol=rtol, atol=atol, err_msg=name)


@pytest.mark.parametrize("case", list(BASES))
def test_basis_matches_its_snapshot(case):
    """Initial tensors and their order, forward and inverse at the initial and at
    generic tensors, gradients of both losses, output dtypes, frozen indices."""
    _check(f"static/{case}/", static_record(case), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    ("case", "run"), [(case, run) for case, runs in TRAININGS.items() for run in runs]
)
def test_training_matches_its_snapshot(case, run):
    """The loss history of a few steps and what the trained basis computes.

    Looser than the static records: an optimiser amplifies a rounding-level
    difference in the gradient at every step.
    """
    _check(f"train/{case}/{run}/", training_record(case, run), rtol=1e-9, atol=1e-10)


def test_no_snapshot_is_orphaned():
    """Every stored array belongs to a case that is still checked."""
    expected = {f"static/{case}/" for case in BASES} | {
        f"train/{case}/{run}/" for case, runs in TRAININGS.items() for run in runs
    }
    for key in STORED.files:
        if key != "__meta__":
            assert any(key.startswith(prefix) for prefix in expected), key
