Training
========

.. currentmodule:: pdft.training

:func:`train_basis` trains one basis on one target image, the loop the Julia goldens are recorded with. :func:`train_basis_batched` is the port of upstream's dataset trainer: several images, epochs, a cosine learning-rate schedule, validation and early stopping.

.. automodule:: pdft.training
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   train_basis
   train_basis_batched
   TrainingResult
   cosine_with_warmup
