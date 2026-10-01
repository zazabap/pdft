Coherence
=========

.. currentmodule:: pdft.coherence

Mutual coherence with the pixel basis, ``mu(U) = N max_ij |U_ij|^2`` in ``[1, N]``, together with a certificate that tells you before training whether training can raise it above 1 (for the package's bases, which have one Hadamard per wire).

.. automodule:: pdft.coherence
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   coherence
   certify_flat_modulus
   FlatModulusCertificate
   sampled_flat_modulus
   is_flat_modulus
   diagonal_tensor_indices
   dense_operator
   operator_coherence
   flat_modulus_deviation
