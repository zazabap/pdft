"""What a model built on the circuit bases needs from the core, without a second representation.

Freezing by gate kind or register (``Program.tensor_indices``), the
controlled-phase angles as a parameter view (``cp_phases`` /
``with_cp_phases``), and the pixel frame (``bit_reverse``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import cp_phases, program_of, with_cp_phases
from pdft.bases.circuit.entangled_qft import extract_entangle_phases, get_entangle_tensor_indices
from pdft.circuit import GATE_KINDS, REGISTERS, bit_reverse, is_compact_cp, register_width
from pdft.manifolds import PhaseManifold, Unitary2qManifold, UnitaryManifold, classify_manifold

from .characterisation.cases import BASES


def _image(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


# tensor_indices


def test_tensor_indices_by_kind_on_the_qft():
    program = pdft.QFTBasis(m=2, n=3).program
    assert program.tensor_indices() == list(range(9))
    assert program.tensor_indices(kind="H") == [0, 1, 2, 3, 4]
    assert program.tensor_indices(kind="CP") == [5, 6, 7, 8]
    assert program.tensor_indices(kind="U4") == program.tensor_indices(kind="CRY") == []


def test_tensor_indices_by_register():
    program = pdft.EntangledQFTBasis(m=2, n=3).program
    row, column, both = (program.tensor_indices(register=r) for r in REGISTERS)
    # 2 + 3 Hadamards, 1 + 3 QFT phases, min(m, n) = 2 entanglers across the registers
    assert (len(row), len(column), len(both)) == (3, 6, 2)
    assert sorted(row + column + both) == list(range(11))
    assert program.tensor_indices(kind="H", register="row") == [0, 1]
    assert program.tensor_indices(kind="H", register="both") == []
    stored = program.sorted_steps
    assert all(max(stored[i][1]) <= 2 for i in row)
    assert all(min(stored[i][1]) > 2 for i in column)
    assert all(min(stored[i][1]) <= 2 < max(stored[i][1]) for i in both)


@pytest.mark.parametrize("case", sorted(BASES))
def test_kinds_and_registers_partition_the_tensors(case):
    basis = BASES[case]()
    program = program_of(basis)
    everything = list(range(len(basis.tensors)))
    assert sorted(i for kind in GATE_KINDS for i in program.tensor_indices(kind=kind)) == everything
    assert sorted(i for r in REGISTERS for i in program.tensor_indices(register=r)) == everything
    for kind in GATE_KINDS:
        for i in program.tensor_indices(kind=kind):
            assert basis.tensors[i].shape == ((2, 2, 2, 2) if kind == "U4" else (2, 2))


def test_tensor_indices_refuses_an_unknown_kind_or_register():
    """A typo must not silently freeze nothing."""
    program = pdft.QFTBasis(m=2, n=2).program
    with pytest.raises(ValueError, match="kind must be one of"):
        program.tensor_indices(kind="h")
    with pytest.raises(ValueError, match="register must be one of"):
        program.tensor_indices(register="rows")


@pytest.mark.parametrize("case", sorted(BASES))
def test_the_gate_kind_implies_the_manifold_the_optimiser_picks(case):
    """``classify_manifold`` goes by tensor values, as upstream does. At the
    initial tensors of every basis that agrees with the gate kind."""
    implied = {
        "H": UnitaryManifold(d=2),
        "CRY": UnitaryManifold(d=2),
        "U4": Unitary2qManifold(),
        "CP": PhaseManifold(),
    }
    basis = BASES[case]()
    for (kind, _), tensor in zip(program_of(basis).sorted_steps, basis.tensors):
        assert classify_manifold(tensor) == implied[kind]
        assert is_compact_cp(tensor) == (kind == "CP")


def test_freezing_the_one_qubit_gates_trains_only_the_phases():
    basis = pdft.QFTBasis(m=2, n=2)
    hadamards = basis.program.tensor_indices(kind="H")
    images = [np.asarray(_image((4, 4), seed).real) for seed in range(4)]
    result = pdft.train_basis_batched(
        basis,
        dataset=images,
        loss=pdft.L1Norm(),
        epochs=2,
        batch_size=2,
        frozen_indices=hadamards,
        seed=0,
    )
    trained = result.basis
    for i, (before, after) in enumerate(zip(basis.tensors, trained.tensors)):
        assert jnp.array_equal(before, after) == (i in hadamards)
    # which is the configuration that cannot leave mu == 1
    assert pdft.certify_flat_modulus(trained, frozen_indices=hadamards)


# the controlled-phase view


def test_cp_phases_reads_the_qft_angles():
    phases = cp_phases(pdft.QFTBasis(m=3, n=2))
    # per register: pi/2 between neighbours, pi/4 one apart
    np.testing.assert_allclose(phases, [np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 2], atol=1e-15)


def test_cp_phases_include_the_entanglers_last():
    basis = pdft.EntangledQFTBasis(m=3, n=2, seed=1)
    last = get_entangle_tensor_indices(basis.tensors, basis.n_entangle)
    assert last == basis.program.tensor_indices(kind="CP", register="both")
    np.testing.assert_allclose(
        cp_phases(basis)[-basis.n_entangle :], extract_entangle_phases(basis.tensors, last)
    )


@pytest.mark.parametrize("case", ["qft_3x2", "entangled_3x2", "tebd_cp_3x2", "mera_cp_4x2"])
def test_with_cp_phases_round_trips(case):
    basis = BASES[case]()
    same = with_cp_phases(basis, cp_phases(basis))
    assert type(same) is type(basis) and pdft.bases_allclose(same, basis, atol=1e-15)
    angles = jnp.asarray(np.random.default_rng(3).uniform(-3, 3, len(cp_phases(basis))))
    moved = with_cp_phases(basis, angles)
    np.testing.assert_allclose(cp_phases(moved), angles, atol=1e-14)
    cp = set(basis.program.tensor_indices(kind="CP"))
    for i, (before, after) in enumerate(zip(basis.tensors, moved.tensors)):
        assert (i in cp) or (after is before)
    x = _image(basis.image_size)
    np.testing.assert_allclose(moved.inverse_transform(moved.forward_transform(x)), x, atol=1e-12)


def test_with_cp_phases_is_differentiable_under_jit():
    basis = pdft.QFTBasis(m=2, n=2)
    x = _image((4, 4))

    @jax.jit
    def sparsity(angles):
        return jnp.sum(jnp.abs(with_cp_phases(basis, angles).forward_transform(x)))

    angles = cp_phases(basis)
    gradient = jax.grad(sparsity)(angles)
    assert gradient.shape == angles.shape and bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 1e-3
    # and it is the derivative: one finite-difference check per angle
    for k in range(len(angles)):
        step = jnp.zeros_like(angles).at[k].set(1e-6)
        numeric = (sparsity(angles + step) - sparsity(angles - step)) / 2e-6
        assert float(gradient[k]) == pytest.approx(float(numeric), abs=1e-6)


def test_the_view_reaches_through_a_blocked_basis():
    inner = pdft.QFTBasis(m=2, n=2)
    blocked = pdft.BlockedBasis(inner, 1, 1)
    assert program_of(blocked) is inner.program
    np.testing.assert_array_equal(cp_phases(blocked), cp_phases(inner))
    moved = with_cp_phases(blocked, jnp.asarray([0.3, -0.2]))
    assert type(moved) is pdft.BlockedBasis and moved.image_size == (8, 8)
    np.testing.assert_allclose(cp_phases(moved), [0.3, -0.2], atol=1e-15)


def test_bases_without_controlled_phases_have_an_empty_view():
    basis = pdft.RichBasis(m=2, n=2)
    assert cp_phases(basis).shape == (0,)
    assert pdft.bases_allclose(with_cp_phases(basis, jnp.zeros((0,))), basis, atol=0.0)
    with pytest.raises(ValueError, match="has 0 controlled-phase gates, got 2 phases"):
        with_cp_phases(basis, jnp.zeros(2))


# the pixel frame


def test_bit_reverse_permutes_each_index_by_reversing_its_bits():
    x = jnp.arange(8 * 4).reshape(8, 4)
    rows = [0, 4, 2, 6, 1, 5, 3, 7]
    columns = [0, 2, 1, 3]
    np.testing.assert_array_equal(bit_reverse(x), np.asarray(x)[np.ix_(rows, columns)])
    np.testing.assert_array_equal(bit_reverse(bit_reverse(x)), x)
    assert bit_reverse(x).dtype == x.dtype


def test_bit_reverse_carries_batch_axes():
    stack = _image((3, 2, 4, 8))
    out = bit_reverse(stack)
    assert out.shape == stack.shape
    for i in range(3):
        for j in range(2):
            np.testing.assert_array_equal(out[i, j], bit_reverse(stack[i, j]))


def test_bit_reverse_needs_power_of_two_sides():
    with pytest.raises(ValueError, match="power-of-two number of values, got 6"):
        bit_reverse(jnp.zeros((4, 6)))
    assert [register_width(size) for size in (1, 2, 8, 1024)] == [0, 1, 3, 10]
    for size in (0, 3, 12):
        with pytest.raises(ValueError, match="power-of-two"):
            register_width(size)


@pytest.mark.parametrize(("m", "n"), [(3, 2), (2, 4), (1, 3)])
def test_the_qft_basis_is_the_dft_in_the_bit_reversed_frame(m, n):
    """The sign convention too: ``e^{+2 pi i k x / N}``, numpy's inverse transform."""
    basis = pdft.QFTBasis(m=m, n=n)
    x = _image(basis.image_size, seed=m + n)
    dft = np.fft.ifft2(np.asarray(x), norm="ortho")
    np.testing.assert_allclose(basis.forward_transform(bit_reverse(x)), dft, atol=1e-13)
    np.testing.assert_allclose(
        bit_reverse(basis.inverse_transform(jnp.asarray(dft))), x, atol=1e-13
    )
    # without the adapter it is the transform of the permuted image, not of the image
    assert float(jnp.max(jnp.abs(basis.forward_transform(x) - dft))) > 0.1


def test_a_pixel_mask_commutes_with_the_frame_change():
    """Why reversing the data is the whole adapter: pixelwise operations do not care."""
    x = _image((8, 8))
    mask = jnp.asarray(np.random.default_rng(0).random((8, 8)) < 0.4)
    np.testing.assert_array_equal(bit_reverse(mask * x), bit_reverse(mask) * bit_reverse(x))
