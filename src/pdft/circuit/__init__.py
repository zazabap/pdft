"""Circuit machinery: einsum builder + JIT closure cache.

Shared by every basis. The builder converts a Yao-style gate list into a
JAX einsum (Hadamard-first sort + Yao little-endian ordering preserved).
The cache memoizes `jnp.einsum_path` results and the JIT'd closures.
"""

from .builder import (
    HADAMARD,
    CircuitCode,
    Gate,
    Program,
    apply_circuit,
    apply_program,
    build_circuit_einsum,
    compile_circuit,
    compile_program,
    controlled_phase_diag,
    extract_phase_from_cp,
    extract_phases,
    hadamard_gate,
    is_compact_cp,
    phase_gate,
    select_last_n_cp_indices,
    sorted_gate_program,
    two_registers,
    u4_from_phase,
)
from .cache import optimize_code_cached

__all__ = [
    "HADAMARD",
    "CircuitCode",
    "Gate",
    "Program",
    "apply_circuit",
    "apply_program",
    "build_circuit_einsum",
    "compile_circuit",
    "compile_program",
    "controlled_phase_diag",
    "extract_phase_from_cp",
    "extract_phases",
    "hadamard_gate",
    "is_compact_cp",
    "optimize_code_cached",
    "phase_gate",
    "select_last_n_cp_indices",
    "sorted_gate_program",
    "two_registers",
    "u4_from_phase",
]
