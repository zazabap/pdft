Coherence
=========

.. currentmodule:: pdft.coherence

Mutual coherence with the pixel basis, ``mu(U) = N max_ij |U_ij|^2`` in ``[1, N]``, together with a certificate that tells you before training whether training can raise it above 1.

.. automodule:: pdft.coherence
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   coherence
   certify_flat_modulus
   FlatModulusCertificate
   is_flat_modulus
   diagonal_tensor_indices
   dense_operator

.. rubric:: Operators given as closures

The same quantity for a transform that is applied rather than stored as a basis, such as the gate kernel in :mod:`pdft.circuit.gates`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   axis_operator
   operator_coherence
   flat_modulus_deviation
   sampled_flat_modulus
