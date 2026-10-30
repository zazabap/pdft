Training
========

.. currentmodule:: pdft.training

:func:`train_basis` trains one basis on one target image, the loop the Julia goldens are recorded with. :func:`train_basis_batched` is the port of upstream's dataset trainer: several images, epochs, a cosine learning-rate schedule, validation and early stopping. :func:`train_basis_steps` is not in upstream: a fixed number of Adam steps, each on a fresh batch under a fresh observation mask, for training a basis through a solver. :func:`train_basis_exact` is not in upstream either: no gradient steps, but exact block-coordinate descent on the sparsity objective, each gate of a separable circuit set in closed form to its best given the others (the polar factor of its environment for a unitary gate, the environment's phases for a phase tensor).

.. automodule:: pdft.training
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   train_basis
   train_basis_batched
   train_basis_exact
   train_basis_steps
   TrainingResult
   cosine_with_warmup
