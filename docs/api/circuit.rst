Circuit machinery
=================

.. currentmodule:: pdft.circuit

Turns a Yao-style gate list into a JAX einsum, and caches the contraction path and the JIT-compiled closure. Every basis uses it.

.. automodule:: pdft.circuit
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   Program
   compile_program
   CircuitCode
   compile_circuit
   build_circuit_einsum
   apply_circuit
   optimize_code_cached
   sorted_gate_program
   Gate
   HADAMARD
   controlled_phase_diag
   u4_from_phase
   extract_phase_from_cp
   is_compact_cp
   select_last_n_cp_indices
