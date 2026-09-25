Manifolds
=========

The manifold geometry used by the optimizers: projections, retractions and batched linear algebra over stacked gate tensors.

.. currentmodule:: pdft.manifolds

.. autosummary::
   :toctree: generated
   :nosignatures:

   AbstractRiemannianManifold
   UnitaryManifold
   PhaseManifold
   OrthogonalManifold
   Unitary2qManifold
   Orthogonal2qManifold
   classify_manifold
   group_by_manifold
   stack_tensors
   unstack_tensors
   batched_matmul
   batched_adjoint
   batched_inv
   is_unitary_2qubit
   is_unitary_general
