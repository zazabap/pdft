"""What several test files share: the registry of basis configurations and a few random inputs."""

from __future__ import annotations

import zlib
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

import pdft
from pdft.bases import with_tensors
from pdft.circuit.builder import HADAMARD, Gate, controlled_phase_diag, u4_from_phase

Array = jax.Array

# The circuit basis classes, for tests that construct one of each.
CIRCUIT_CLASSES = (
    pdft.QFTBasis,
    pdft.EntangledQFTBasis,
    pdft.TEBDBasis,
    pdft.MERABasis,
    pdft.DCT4Basis,
    pdft.RichBasis,
    pdft.RealRichBasis,
)

# One case per gate kind, parametrization and option the builder supports, at
# small sizes, for tests that run over every basis. Rectangular where the
# basis allows it, so a swapped register cannot go unnoticed.
BASES: dict[str, Callable[[], object]] = {
    "qft_2x2": lambda: pdft.QFTBasis(m=2, n=2),
    "qft_3x2": lambda: pdft.QFTBasis(m=3, n=2),
    "entangled_3x2": lambda: pdft.EntangledQFTBasis(m=3, n=2, seed=1),
    "entangled_front_2x3": lambda: pdft.EntangledQFTBasis(
        m=2, n=3, seed=2, entangle_position="front"
    ),
    "tebd_cp_3x2": lambda: pdft.TEBDBasis(m=3, n=2, seed=1),
    "tebd_u4_2x2": lambda: pdft.TEBDBasis(m=2, n=2, seed=1, parametrization="u4"),
    "mera_cp_4x2": lambda: pdft.MERABasis(m=4, n=2, seed=1),
    "mera_u4_2x2": lambda: pdft.MERABasis(m=2, n=2, seed=1, parametrization="u4"),
    "dct4_o4_3x2": lambda: pdft.DCT4Basis(m=3, n=2),
    "dct4_controlled_3x2": lambda: pdft.DCT4Basis(m=3, n=2, parametrization="controlled"),
    "rich_3x2": lambda: pdft.RichBasis(m=3, n=2),
    "real_rich_2x2": lambda: pdft.RealRichBasis(m=2, n=2),
    "blocked_qft_2x2_in_3x3": lambda: pdft.BlockedBasis(pdft.QFTBasis(m=2, n=2), 1, 1),
}


def small_circuit() -> list[Gate]:
    """Four gates on two qubits, one of each kind the einsum form has.

    Deliberately not Hadamard-first, so the stored order differs from the
    order the gates act in.
    """
    return [
        Gate(kind="U4", qubits=(1, 2), tensor=u4_from_phase(0.3), phase=0.3),
        Gate(kind="H", qubits=(1,), tensor=HADAMARD, phase=0.0),
        Gate(kind="CP", qubits=(2, 1), tensor=controlled_phase_diag(0.7), phase=0.7),
        Gate(kind="H", qubits=(2,), tensor=HADAMARD, phase=0.0),
    ]


def gate_structure(gates: list[Gate]) -> list[tuple[str, tuple[int, ...]]]:
    """``(kind, qubits)`` of each gate: a gate list without its tensors, so it compares with ``==``."""
    return [(g["kind"], g["qubits"]) for g in gates]


def case_rng(case: str) -> np.random.Generator:
    """A generator seeded by the case's name, so each case has its own fixed inputs."""
    return np.random.default_rng(zlib.crc32(case.encode()))


def complex_normal(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def complex_image(shape, seed: int = 0) -> Array:
    """A complex array with no structure: a real image at symmetric tensors hides too much."""
    return jnp.asarray(complex_normal(np.random.default_rng(seed), shape))


def random_unitary(rng: np.random.Generator, d: int, *, real: bool = False) -> np.ndarray:
    """A ``d x d`` unitary with no symmetry, orthogonal when ``real``."""
    a = rng.normal(size=(d, d))
    if not real:
        a = a + 1j * rng.normal(size=(d, d))
    return np.linalg.qr(a)[0]


def generic(basis, rng: np.random.Generator):
    """The basis at tensors with no symmetry left.

    At initialisation most gates are symmetric or diagonal, which hides a
    swapped leg or a transposed gate. Every entry is scaled by its own complex
    factor; the applier is linear in each tensor, so unitarity is not needed to
    pin what it computes.
    """
    return with_tensors(
        basis, [t * jnp.asarray(1 + 0.2 * complex_normal(rng, t.shape)) for t in basis.tensors]
    )
