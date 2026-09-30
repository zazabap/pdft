"""Circuit machinery: einsum builder + JIT closure cache + gate-by-gate kernel.

Shared by every basis. The builder converts a Yao-style gate list into a
JAX einsum (Hadamard-first sort + Yao little-endian ordering preserved).
The cache memoizes `jnp.einsum_path` results and the JIT'd closures.
The gate kernel (`gates`) applies the QFT circuit to an image directly from
its gate parameters, with no tensors and no matrix; it is not a Julia port.
"""

from .builder import (
    HADAMARD,
    Gate,
    apply_circuit,
    build_circuit_einsum,
    compile_circuit,
    controlled_phase_diag,
    extract_phase_from_cp,
    is_compact_cp,
    select_last_n_cp_indices,
    sorted_gate_program,
    u4_from_phase,
)
from .cache import optimize_code_cached
from .gates import (
    apply_dense,
    apply_gates,
    axis_operator,
    bitreverse,
    gate_matrix,
    gate_pairs,
    hadamards,
    register_width,
    separable,
    theta0,
    theta_to_params,
)

__all__ = [
    "HADAMARD",
    "Gate",
    "apply_circuit",
    "apply_dense",
    "apply_gates",
    "axis_operator",
    "bitreverse",
    "build_circuit_einsum",
    "compile_circuit",
    "controlled_phase_diag",
    "extract_phase_from_cp",
    "gate_matrix",
    "gate_pairs",
    "hadamards",
    "is_compact_cp",
    "optimize_code_cached",
    "register_width",
    "select_last_n_cp_indices",
    "separable",
    "sorted_gate_program",
    "theta0",
    "theta_to_params",
    "u4_from_phase",
]
