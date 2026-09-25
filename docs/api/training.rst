Training
========

:func:`train_basis` trains one basis on one target and mirrors the upstream loop. :func:`train_basis_batched` adds several images, epochs, a cosine learning-rate schedule, validation and early stopping.

.. currentmodule:: pdft.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   train_basis
   train_basis_batched
   TrainingResult
   cosine_with_warmup
