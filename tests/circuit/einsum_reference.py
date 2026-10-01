"""A circuit as one einsum string: an independent reference for the gate walk.

This is the builder the package applied circuits with before it walked them
gate by gate, moved here unchanged. Nothing in the package calls it any more,
so it lives with the tests, where it is useful as a second implementation of
what a gate list means: wire labels instead of axes, one contraction instead
of a walk. It is limited to 52 labels and has no ``CRY`` gate.
"""

from __future__ import annotations

import string

import jax
import jax.numpy as jnp

from pdft.circuit.builder import Gate, _hadamard_first_perm

Array = jax.Array


def build_circuit_einsum(
    gates: list[Gate],
    m: int,
    n: int,
    *,
    inverse: bool,
) -> tuple[str, list[Array], list[tuple[int, ...]]]:
    """Convert a gate sequence to (subscripts, tensors, shapes).

    `gates` is applied in order to a circuit on `m + n` qubits (1-indexed).
    For CP gates, the 2x2 tensor shares the current wire labels of its
    control and target qubits and does NOT introduce new labels (the gate
    is diagonal).

    The returned tensor list and tensor-shape list are sorted so Hadamards
    come first (matching Julia's `perm_vec`).
    """
    N = m + n
    pool = list(string.ascii_lowercase + string.ascii_uppercase)
    next_idx = 0

    def fresh() -> str:
        nonlocal next_idx
        if next_idx >= len(pool):
            raise ValueError(f"too many qubits: need > {len(pool)} einsum labels")
        ch = pool[next_idx]
        next_idx += 1
        return ch

    input_labels = [fresh() for _ in range(N)]
    wire_state: dict[int, str] = {q + 1: input_labels[q] for q in range(N)}

    tensor_subscripts: list[str] = []
    tensor_list: list[Array] = []
    tensor_shapes: list[tuple[int, ...]] = []

    for g in gates:
        if g["kind"] == "H":
            (q,) = g["qubits"]
            in_lbl = wire_state[q]
            out_lbl = fresh()
            tensor_subscripts.append(out_lbl + in_lbl)
            tensor_list.append(g["tensor"])
            tensor_shapes.append((2, 2))
            wire_state[q] = out_lbl
        elif g["kind"] == "CP":
            q_ctrl, q_tgt = g["qubits"]
            ctrl_lbl = wire_state[q_ctrl]
            tgt_lbl = wire_state[q_tgt]
            tensor_subscripts.append(ctrl_lbl + tgt_lbl)
            tensor_list.append(g["tensor"])
            tensor_shapes.append((2, 2))
        elif g["kind"] == "U4":
            # General 2-qubit unitary stored as (out_c, out_t, in_c, in_t).
            # Unlike CP this is NOT diagonal; it INTRODUCES new wire labels
            # for both qubits' outputs.
            q_ctrl, q_tgt = g["qubits"]
            in_c = wire_state[q_ctrl]
            in_t = wire_state[q_tgt]
            out_c = fresh()
            out_t = fresh()
            tensor_subscripts.append(out_c + out_t + in_c + in_t)
            tensor_list.append(g["tensor"])
            tensor_shapes.append((2, 2, 2, 2))
            wire_state[q_ctrl] = out_c
            wire_state[q_tgt] = out_t
        else:
            raise AssertionError(f"unknown gate kind: {g['kind']}")

    # Hadamard-first sort (matches Julia's perm_vec).
    perm = _hadamard_first_perm(tensor_list)
    tensor_list = [tensor_list[i] for i in perm]
    tensor_subscripts = [tensor_subscripts[i] for i in perm]
    tensor_shapes = [tensor_shapes[i] for i in perm]

    # Little-endian qubit mapping within each block
    row_pic = [input_labels[q - 1] for q in range(m, 0, -1)]
    col_pic = [input_labels[q - 1] for q in range(m + n, m, -1)]
    pic_labels = "".join(row_pic + col_pic)

    row_out = [wire_state[q] for q in range(m, 0, -1)]
    col_out = [wire_state[q] for q in range(m + n, m, -1)]
    out_labels = "".join(row_out + col_out)

    if inverse:
        lhs = ",".join(tensor_subscripts + [out_labels])
        rhs = pic_labels
    else:
        lhs = ",".join(tensor_subscripts + [pic_labels])
        rhs = out_labels

    subscripts = f"{lhs}->{rhs}"
    tensor_shapes.append((2,) * N)  # pic operand
    return subscripts, tensor_list, tensor_shapes


def contract(gates: list[Gate], m: int, n: int, pic: Array, *, inverse: bool = False) -> Array:
    """Apply ``gates`` to ``pic``, shape ``(2,) * (m + n)``, as a single einsum.

    "greedy" and not "optimal": the optimal path search is exponential in the
    number of tensors and hangs on a 3x3 QFT.
    """
    subscripts, tensors, _ = build_circuit_einsum(gates, m, n, inverse=inverse)
    return jnp.einsum(subscripts, *tensors, pic, optimize="greedy")
