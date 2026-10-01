"""Circuit machinery shared by every basis: gates, programs and the applier.

A family emits a Yao-style gate list; `compile_program` turns it into a
`Program` and the tensors in stored order (Hadamard-first, Yao little-endian
qubits); `CircuitCode` applies the program one gate at a time.
"""

from .builder import (
    HADAMARD,
    CircuitCode,
    Gate,
    Program,
    apply_circuit,
    apply_program,
    compile_circuit,
    compile_program,
    controlled_phase_diag,
    extract_phase_from_cp,
    extract_phases,
    hadamard_gate,
    hadamards_then_layers,
    is_compact_cp,
    phase_gate,
    select_last_n_cp_indices,
    two_registers,
    u4_from_phase,
)

__all__ = [
    "HADAMARD",
    "CircuitCode",
    "Gate",
    "Program",
    "apply_circuit",
    "apply_program",
    "compile_circuit",
    "compile_program",
    "controlled_phase_diag",
    "extract_phase_from_cp",
    "extract_phases",
    "hadamard_gate",
    "hadamards_then_layers",
    "is_compact_cp",
    "phase_gate",
    "select_last_n_cp_indices",
    "two_registers",
    "u4_from_phase",
]
