Circuit machinery
=================

.. currentmodule:: pdft.circuit

Turns a Yao-style gate list into a program and applies it to an image one gate at a time; every basis with the same program shares one compiled applier. Every basis uses it.

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
   controlled_phase_diag
   u4_from_phase
   controlled
   identity_tensor
   extract_phase_from_cp
   extract_phases
   is_compact_cp
   select_last_n_cp_indices
