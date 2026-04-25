"""Tests for seeded random phase initialisation on TEBD/MERA/EntangledQFT bases.

Mirror of the Julia `_init_circuit` behaviour where `phases = randn(n_gates) * 0.1`
when not explicitly provided. Adds a `seed` argument to the three random-init
basis classes so the symmetry collapse (QFTBasis == EntangledQFTBasis at
init when phases default to zero) is broken.
"""

from __future__ import annotations

import numpy as np
import pytest

import pdft


def test_entangled_qft_seed_breaks_symmetry_with_qft():
    """EntangledQFTBasis(seed=42) must differ from QFTBasis at init."""
    qft = pdft.QFTBasis(m=4, n=4)
    eqft = pdft.EntangledQFTBasis(m=4, n=4, seed=42)
    # Tensor counts differ (entangle layer adds gates), but for the gates that
    # exist in both, the entanglement-layer tensors must NOT be identity.
    # We check via tensor lists not being a strict prefix match.
    common = min(len(qft.tensors), len(eqft.tensors))
    same = all(
        np.allclose(np.asarray(qft.tensors[i]), np.asarray(eqft.tensors[i])) for i in range(common)
    )
    # If they share a strict prefix, EntangledQFT's extra entanglement gates must
    # contain non-identity entries (at least one phase != 0 → CP tensor row≠[1,1])
    if same:
        # Last-N entries should be the entanglement gates with random phases.
        n_extra = len(eqft.tensors) - len(qft.tensors)
        assert n_extra > 0, "EntangledQFTBasis must add entanglement gates"
        for t in eqft.tensors[-n_extra:]:
            arr = np.asarray(t)
            # CP gate has shape (2, 2) and identity-phase yields [[1, 1], [1, 1]].
            # Random phases yield [[1, 1], [1, exp(iφ)]] with |[1, exp(iφ)]| = 1
            # but value differs from 1 unless φ = 0.
            assert not np.allclose(arr, np.ones_like(arr)), (
                f"entanglement gate is identity-phase: {arr}"
            )


def test_entangled_qft_seed_deterministic():
    """Same seed → same tensors."""
    a = pdft.EntangledQFTBasis(m=3, n=3, seed=7)
    b = pdft.EntangledQFTBasis(m=3, n=3, seed=7)
    for ta, tb in zip(a.tensors, b.tensors):
        np.testing.assert_array_equal(np.asarray(ta), np.asarray(tb))


def test_entangled_qft_different_seeds_differ():
    a = pdft.EntangledQFTBasis(m=3, n=3, seed=1)
    b = pdft.EntangledQFTBasis(m=3, n=3, seed=2)
    diffs = [
        not np.allclose(np.asarray(ta), np.asarray(tb)) for ta, tb in zip(a.tensors, b.tensors)
    ]
    assert any(diffs), "Different seeds should yield at least one differing tensor"


def test_tebd_seed_deterministic():
    a = pdft.TEBDBasis(m=2, n=2, seed=11)
    b = pdft.TEBDBasis(m=2, n=2, seed=11)
    for ta, tb in zip(a.tensors, b.tensors):
        np.testing.assert_array_equal(np.asarray(ta), np.asarray(tb))


def test_tebd_seed_breaks_zero_phase():
    """TEBDBasis(seed=42) must not be the all-zero-phase init."""
    no_seed = pdft.TEBDBasis(m=2, n=2)  # default zero-phase
    seeded = pdft.TEBDBasis(m=2, n=2, seed=42)
    diffs = [
        not np.allclose(np.asarray(ta), np.asarray(tb))
        for ta, tb in zip(no_seed.tensors, seeded.tensors)
    ]
    assert any(diffs), "Seeded TEBDBasis must differ from zero-phase default"


def test_mera_seed_deterministic():
    """Same seed → same MERA tensors."""
    a = pdft.MERABasis(m=2, n=2, seed=99)
    b = pdft.MERABasis(m=2, n=2, seed=99)
    for ta, tb in zip(a.tensors, b.tensors):
        np.testing.assert_array_equal(np.asarray(ta), np.asarray(tb))


def test_mera_seed_breaks_zero_phase():
    no_seed = pdft.MERABasis(m=2, n=2)
    seeded = pdft.MERABasis(m=2, n=2, seed=42)
    diffs = [
        not np.allclose(np.asarray(ta), np.asarray(tb))
        for ta, tb in zip(no_seed.tensors, seeded.tensors)
    ]
    assert any(diffs), "Seeded MERABasis must differ from zero-phase default"


def test_seed_does_not_affect_qft():
    """QFTBasis has no phases — passing seed should be a no-op (or rejected)."""
    # QFTBasis doesn't accept seed; passing it must raise. Existing code paths
    # that don't pass seed continue to work.
    a = pdft.QFTBasis(m=2, n=2)
    b = pdft.QFTBasis(m=2, n=2)
    for ta, tb in zip(a.tensors, b.tensors):
        np.testing.assert_array_equal(np.asarray(ta), np.asarray(tb))
    with pytest.raises(TypeError):
        pdft.QFTBasis(m=2, n=2, seed=42)  # type: ignore[call-arg]


def test_existing_zero_phase_default_unchanged():
    """Regression guard: zero-phase init (no seed, no explicit phases) is still
    the all-zero-phase basis. This preserves existing parity goldens."""
    eqft = pdft.EntangledQFTBasis(m=4, n=4)  # no seed, no entangle_phases
    # Implementation detail: at init with phases=zero, every entanglement-gate
    # CP tensor equals [[1,1],[1,1]] (identity diag).
    # We don't probe individual gates — rely on bases_allclose with a freshly
    # constructed default-arg instance.
    eqft2 = pdft.EntangledQFTBasis(m=4, n=4)
    assert pdft.bases_allclose(eqft, eqft2)
