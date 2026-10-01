"""Quantum Fourier Transform circuit as a hand-rolled tensor network.

Mirror of upstream src/qft.jl. Replaces Yao.EasyBuild.qft_circuit +
yao2einsum with an explicit gate chain. The gate sequence is the standard
QFT decomposition (upstream src/entangled_qft.jl:51-77):

    For j = 1..n_qubits:
        H on qubit j
        For target in j+1..n_qubits:
            CP(control=target, target=j, phase=2*pi/2^(target-j+1))

2D QFT = (m-qubit QFT on row qubits) tensor (n-qubit QFT on col qubits);
no entanglement between blocks.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ...circuit.builder import (
    HADAMARD,
    Gate,
    apply_circuit,
    compile_circuit,
    controlled_phase_diag,
    cp_gate,
    hadamard_gate,
    two_registers,
)

Array = jax.Array


# Re-export canonical primitives (imported by tests / external callers)
__all__ = [
    "HADAMARD",
    "_qft_gates_1d",
    "controlled_phase_diag",
    "ft_mat",
    "ift_mat",
    "qft_code",
    "qft_gates",
    "qft_gates_1d",
]


def qft_gates_1d(
    n_qubits: int, offset: int, two_qubit: Callable[[int, int, float], Gate] = cp_gate
) -> list[Gate]:
    """Emit the 1D QFT gate sequence on qubits (offset+1, ..., offset+n_qubits).

    Matches upstream src/entangled_qft.jl:64-78 exactly. ``two_qubit(q_ctrl,
    q_tgt, phi)`` builds the gate between a qubit and each later one; the
    default is the controlled phase of the QFT itself, and the bases that keep
    this topology but train another gate there pass their own.
    """
    gates: list[Gate] = []
    for j in range(1, n_qubits + 1):
        gates.append(hadamard_gate(offset + j))
        for target in range(j + 1, n_qubits + 1):
            phi = float(2 * jnp.pi / 2 ** (target - j + 1))
            gates.append(two_qubit(offset + target, offset + j, phi))
    return gates


_qft_gates_1d = qft_gates_1d


def qft_gates(m: int, n: int) -> list[Gate]:
    """The gate sequence of the 2D QFT on (2^m, 2^n) images."""
    return two_registers(qft_gates_1d, m, n)


def qft_code(m: int, n: int, *, inverse: bool = False) -> tuple[Callable[..., Array], list[Array]]:
    """Return `(code, initial_tensors)` for 2D QFT on (2^m, 2^n) images."""
    return compile_circuit(qft_gates(m, n), m, n, inverse=inverse)


# Julia's names for applying a circuit to an image. The inverse is the same
# call with the inverse code and conjugated tensors, so one function is both.
ft_mat = ift_mat = apply_circuit
