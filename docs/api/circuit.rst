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

   build_circuit_einsum
   compile_circuit
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

.. rubric:: The gate-by-gate kernel

.. currentmodule:: pdft.circuit.gates

.. automodule:: pdft.circuit.gates
   :no-members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   apply_gates
   apply_dense
   separable
   gate_matrix
   axis_operator
   theta0
   theta_to_params
   hadamards
   gate_pairs
   n_params
   n_from_params
   register_width
   bitreverse
