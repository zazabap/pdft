Optimizers
==========

.. currentmodule:: pdft.optimizers

Riemannian optimizers that keep each gate tensor on its manifold (unitary or phase, picked from the tensor's values; a basis whose tensors are all exactly real stays real under a real objective). Their update rules follow ParametricDFT.jl exactly; the package does not use optax.

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
