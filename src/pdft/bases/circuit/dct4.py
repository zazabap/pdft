"""DCT-IV circuit as a hand-rolled tensor network.

Real-orthogonal analogue of the QFT (cf. :mod:`pdft.bases.circuit.qft`). The
DCT-IV is the self-dual "bulk" cosine transform: real, orthogonal, and its
own inverse. Unlike the DCT-II it is **ancilla-free** and uses only 1- and
2-qubit gates, so it drops straight into the H + 2-qubit-gate tensor-network
builder with no multi-controlled scaffolding.

The 1D recursion is *uniform* — both halves run the same child, so DCT-IV
unrolls into n qubit-local stages with no multi-controls (the real-orthogonal
analogue of the QFT butterfly). For N = 2M, with top qubit b splitting the
register (cf. ``all_real_dct_one_ancilla.tex`` §3.2):

    C^IV_{2M} = P_{2M} (H (x) I_M) (I_M (+) D_M) (I_2 (x) C^IV_M)
                (I_M (+) R_M) T_{2M} Q_{2M}

Per level k on the within-block qubits (b = top qubit), in emission order
(= the operator product above read right-to-left):

    * Q  pair layout : CX(control=b, target=q) for each q below b
    * T  rotation    : R_y(pi / 2N) on b, then CR_y(pi 2^p / N) on b controlled
                       by each lower bit — the affine phase gradient (one base
                       rotation + one controlled rotation per bit, never a 2^m
                       multiplexer)
    * R  reverse     : the mirror-Q permutation repeated (odd-branch reversal)
    * recurse        : C^IV_M on the lower register, uncontrolled (I_2 (x) C^IV_M)
    * D  sign Delta  : CZ(control=b, target=next qubit)
    * H  merge       : H on b

The interleave P_{2M} is deferred: per-level interleaves compose into a single
bit reversal, realised as the bit-reversed output convention (matching
:func:`qft_code`), so no explicit P gate is emitted.

At init the forward operator equals the bit-reversed orthonormal DCT-IV per
dimension. Every gate is a learnable leaf on its auto-selected Riemannian
manifold — R_y and branch-H on O(2), CR_y and mirror-Q on O(4), the Delta
sign on the phase manifold. Like QFT/RealRich, real init plus a real objective
keep the operator real-orthogonal as it relaxes from the exact transform (cf.
``all_real_dct_zero_ancilla.tex``).

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
        # Q: pair-layout permutation — controlled bit-complement of the lower
        # register (exact CNOT at init; a learnable leaf like every gate)
        for q in lower:
            gates.append(Gate(kind="U4", qubits=(Q(b), Q(q)), tensor=_cnot_u4(), phase=0.0))
        # T: affine R_y phase-gradient (one base R_y on b, one CR_y per lower bit)
        gates.append(Gate(kind="H", qubits=(Q(b),), tensor=_ry(np.pi / (2 * size)), phase=0.0))
        for p in range(n_qubits - 1 - k):
            theta = np.pi * (2**p) / size
            gates.append(Gate(kind="U4", qubits=(Q(n_qubits - 1 - p), Q(b)), tensor=_cry_u4(theta), phase=0.0))
        # R: odd-branch reversal — the mirror-Q permutation repeated
        for q in lower:
            gates.append(Gate(kind="U4", qubits=(Q(b), Q(q)), tensor=_cnot_u4(), phase=0.0))
        # recurse on the lower register (uncontrolled)
        level(k + 1)
        # D: Delta sign  +  H: branch Hadamard merge
        if lower:
            gates.append(
                Gate(kind="CP", qubits=(Q(b), Q(k + 1)), tensor=controlled_phase_diag(float(np.pi)), phase=float(np.pi))
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
