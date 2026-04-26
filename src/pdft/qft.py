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

from .circuit.builder import (
    HADAMARD,
    Gate,
    apply_circuit,
    compile_circuit,
    controlled_phase_diag,
)

Array = jax.Array


# Re-export canonical primitives (imported by tests / external callers)
__all__ = [
    "HADAMARD",
    "controlled_phase_diag",
    "qft_code",
    "ft_mat",
    "ift_mat",
    "_qft_gates_1d",
]


def _qft_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """Emit the 1D QFT gate sequence on qubits (offset+1, ..., offset+n_qubits).

    Matches upstream src/entangled_qft.jl:64-78 exactly.
    """
    gates: list[Gate] = []
    for j in range(1, n_qubits + 1):
        q = offset + j
        gates.append(Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))
        for target in range(j + 1, n_qubits + 1):
            k = target - j + 1
            t = offset + target
            phi = 2 * jnp.pi / (2**k)
            gates.append(
                Gate(
                    kind="CP",
                    qubits=(t, q),
                    tensor=controlled_phase_diag(float(phi)),
                    phase=float(phi),
                )
            )
    return gates


def qft_code(m: int, n: int, *, inverse: bool = False) -> tuple[Callable[..., Array], list[Array]]:
    """Return `(einsum_fn, initial_tensors)` for 2D QFT on (2^m, 2^n) images."""
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    gates = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    return compile_circuit(gates, m, n, inverse=inverse)


def ft_mat(tensors: list[Array], code: Callable, m: int, n: int, pic: Array) -> Array:
    """Apply 2D QFT circuit to a (2^m, 2^n) image."""
    return apply_circuit(tensors, code, m, n, pic)


def ift_mat(tensors: list[Array], code: Callable, m: int, n: int, pic: Array) -> Array:
    """Apply 2D inverse QFT circuit. Caller must have conjugated the tensors."""
    return apply_circuit(tensors, code, m, n, pic)
