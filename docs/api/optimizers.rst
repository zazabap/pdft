Optimizers
==========

Riemannian optimizers that keep each gate tensor on its manifold (unitary, orthogonal or phase). Their update rules follow ParametricDFT.jl exactly; the package does not use optax.

.. currentmodule:: pdft.optimizers

.. autosummary::
   :toctree: generated
   :nosignatures:

   RiemannianGD
   RiemannianAdam
   optimize
