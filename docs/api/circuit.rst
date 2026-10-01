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
   contract_circuit
   check_image_shape
   bit_reverse
   register_width
   Gate
   GATE_KINDS
   REGISTERS
   HADAMARD
   hadamard_gate
   phase_gate
   cp_gate
   u4_gate
   two_registers
   hadamards_then_layers
   check_qubits
   phase_list
   controlled_phase_diag
   u4_from_phase
   controlled
   identity_tensor
   extract_phase_from_cp
   extract_phases
   is_compact_cp
   select_last_n_cp_indices
