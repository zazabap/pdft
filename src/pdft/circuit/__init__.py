"""Circuit machinery: einsum builder + JIT closure cache.

Shared by every basis. The builder converts a Yao-style gate list into a
JAX einsum (Hadamard-first sort + Yao little-endian ordering preserved).
The cache memoizes `jnp.einsum_path` results and the JIT'd closures.
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
    u4_from_phase,
)
from .cache import optimize_code_cached

__all__ = [
    "HADAMARD",
    "Gate",
    "apply_circuit",
    "build_circuit_einsum",
    "compile_circuit",
    "controlled_phase_diag",
    "extract_phase_from_cp",
    "is_compact_cp",
    "optimize_code_cached",
    "select_last_n_cp_indices",
    "u4_from_phase",
]
