Optimizers
==========

.. currentmodule:: pdft.optimizers

Riemannian optimizers that keep each gate tensor on its manifold (unitary or phase, picked from the tensor's values; a real unitary tensor under a real objective stays orthogonal). Their update rules follow ParametricDFT.jl exactly; the package does not use optax.

.. automodule:: pdft.optimizers
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   RiemannianGD
   RiemannianAdam
   AbstractRiemannianOptimizer
   optimize
