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
   apply_program
   apply_circuit
   build_circuit_einsum
   optimize_code_cached
   sorted_gate_program
   Gate
   HADAMARD
   hadamard_gate
   phase_gate
   two_registers
   controlled_phase_diag
   u4_from_phase
   extract_phase_from_cp
   extract_phases
   is_compact_cp
   select_last_n_cp_indices
