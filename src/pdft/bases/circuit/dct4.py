"""DCT-IV circuit as a hand-rolled tensor network.

Real-orthogonal analogue of the QFT (cf. :mod:`pdft.bases.circuit.qft`). The
DCT-IV is the self-dual "bulk" cosine transform: real, orthogonal, and its
own inverse. Unlike the DCT-II it is **ancilla-free** and uses only 1- and
2-qubit gates, so it drops straight into the H + 2-qubit-gate tensor-network
builder with no multi-controlled scaffolding.

The 1D recursion (uniform: same child on both halves), for N = 2M:

    C^IV_{2M} = (H layer)(Z sign)(I_2 (x) C^IV_M)(R_y layer)(mirror Q)

Per level k on the within-block qubits (b = top qubit of the level):

    * mirror Q      : CX(control=b, target=q) for q below b
    * rotation T    : R_y(pi / 2N) on b, then CR_y(pi 2^p / N) on b
                      controlled by each lower bit (affine-angle decomposition
                      -> one base rotation + one controlled rotation per bit,
                      never a 2^m multiplexer)
    * recurse       : C^IV_M on the lower register (uncontrolled)
    * Z sign Delta  : CZ(control=b, target=next qubit)
    * H merge       : H on b

All angles are fixed by the transform at initialization. The trainable leaves
are the *bulk* gates — the affine ``R_y`` rotation layer and the branch
Hadamards (the real-orthogonal degrees of freedom; cf. the relaxation in
``all_real_dct_zero_ancilla.tex``). The structural gates — the mirror-``Q``
CNOT permutations and the ``Delta`` sign — are emitted ``trainable=False``, so
they stay fixed wiring and never enter the optimizer. The forward operator
equals the bit-reversed orthonormal DCT-IV per dimension, matching the
bit-reversed output convention of :func:`qft_code`.

2D DCT-IV = (m-qubit DCT-IV on row qubits) tensor (n-qubit DCT-IV on col
qubits); no entanglement between blocks.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ...circuit.builder import (
    HADAMARD,
    Gate,
    apply_circuit,
    compile_circuit,
    controlled_phase_diag,
)

Array = jax.Array


__all__ = [
    "dct4_code",
    "dct4_ft_mat",
    "dct4_ift_mat",
    "_dct4_gates_1d",
]


def _ry(theta: float) -> Array:
    """Single-qubit real R_y(theta) as a (2, 2) complex tensor `T[out, in]`."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return jnp.asarray(np.array([[c, -s], [s, c]]), dtype=jnp.complex128)


def _cnot_u4() -> Array:
    """CNOT as a (2, 2, 2, 2) tensor `T[out_ctrl, out_tgt, in_ctrl, in_tgt]`."""
    M = np.zeros((2, 2, 2, 2))
    for ic in (0, 1):
        for it in (0, 1):
            M[ic, it ^ ic, ic, it] = 1.0
    return jnp.asarray(M, dtype=jnp.complex128)


def _cry_u4(theta: float) -> Array:
    """Controlled-R_y(theta) as a (2, 2, 2, 2) tensor (control fires on 1)."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    R = np.array([[c, -s], [s, c]])
    M = np.zeros((2, 2, 2, 2))
    for it in (0, 1):
        M[0, it, 0, it] = 1.0  # control = 0 -> identity on target
    for ot in (0, 1):
        for it in (0, 1):
            M[1, ot, 1, it] = R[ot, it]  # control = 1 -> R_y on target
    return jnp.asarray(M, dtype=jnp.complex128)


def _dct4_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """Emit the 1D DCT-IV gate sequence on qubits (offset+1, ..., offset+n_qubits).

    Within-block qubit ``q`` (0 = top/most significant) maps to builder qubit
    ``offset + n_qubits - q`` (Yao little-endian), which reproduces the
    bit-reversed orthonormal DCT-IV exactly.
    """

    def Q(q: int) -> int:
        return offset + n_qubits - q

    gates: list[Gate] = []

    def level(k: int) -> None:
        if k == n_qubits:
            return
        size = 2 ** (n_qubits - k)
        b = k
        lower = list(range(k + 1, n_qubits))
        # mirror Q: controlled bit-complement of the lower register (structural
        # permutation — fixed, not a learnable degree of freedom)
        for q in lower:
            gates.append(Gate(kind="U4", qubits=(Q(b), Q(q)), tensor=_cnot_u4(), phase=0.0, trainable=False))
        # rotation layer T (affine angles) — the learnable real-orthogonal bulk
        gates.append(Gate(kind="H", qubits=(Q(b),), tensor=_ry(np.pi / (2 * size)), phase=0.0))
        for p in range(n_qubits - 1 - k):
            theta = np.pi * (2**p) / size
            gates.append(Gate(kind="U4", qubits=(Q(n_qubits - 1 - p), Q(b)), tensor=_cry_u4(theta), phase=0.0))
        # mirror Q again (structural — fixed)
        for q in lower:
            gates.append(Gate(kind="U4", qubits=(Q(b), Q(q)), tensor=_cnot_u4(), phase=0.0, trainable=False))
        # recurse on the lower register (uncontrolled)
        level(k + 1)
        # Z sign (Delta, fixed) + learnable branch Hadamard merge
        if lower:
            gates.append(
                Gate(kind="CP", qubits=(Q(b), Q(k + 1)), tensor=controlled_phase_diag(float(np.pi)), phase=float(np.pi), trainable=False)
            )
        gates.append(Gate(kind="H", qubits=(Q(b),), tensor=HADAMARD, phase=0.0))

    level(0)
    return gates


def dct4_code(m: int, n: int, *, inverse: bool = False) -> tuple[Callable[..., Array], list[Array]]:
    """Return `(einsum_fn, initial_tensors)` for 2D DCT-IV on (2^m, 2^n) images."""
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    gates = _dct4_gates_1d(m, offset=0) + _dct4_gates_1d(n, offset=m)
    return compile_circuit(gates, m, n, inverse=inverse)


def dct4_ft_mat(tensors: list[Array], code: Callable, m: int, n: int, pic: Array) -> Array:
    """Apply 2D DCT-IV circuit to a (2^m, 2^n) image."""
    return apply_circuit(tensors, code, m, n, pic)


def dct4_ift_mat(tensors: list[Array], code: Callable, m: int, n: int, pic: Array) -> Array:
    """Apply 2D inverse DCT-IV circuit. Caller must have conjugated the tensors."""
    return apply_circuit(tensors, code, m, n, pic)
