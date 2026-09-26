Completion
==========

.. currentmodule:: pdft.completion

Image inpainting from random pixels with a trainable QFT: the circuit applied to the image gate by gate, the unrolled recovery solver it is trained through, and the exact bridge to :class:`pdft.QFTBasis`. A transform family is one per-axis operator ``apply(x, params, adjoint, axis)``; its 2-D pair, its solver and its evaluation are derived from it. The names below are grouped by the module that defines them; the ones most used interactively are also re-exported at ``pdft.completion``.

.. automodule:: pdft.completion
   :no-members:

Operators
---------

.. currentmodule:: pdft.completion.transform

.. autosummary::
   :toctree: generated
   :nosignatures:

   apply_gates
   apply_dense
   separable
   theta0
   theta_to_params
   hadamards
   gate_pairs
   n_params
   n_from_params
   register_width
   bitreverse

The solver
----------

.. currentmodule:: pdft.completion.solver

.. autosummary::
   :toctree: generated
   :nosignatures:

   hard_k
   soft_k
   iht
   solver_for
   batched

.. currentmodule:: pdft.completion.unroll

.. autosummary::
   :toctree: generated
   :nosignatures:

   solver_for
   reconstruct
   plan
   estimate_peak
   split_k
   report

Training and evaluation
-----------------------

.. currentmodule:: pdft.completion.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   minibatches
   task_loss
   mu_monitor
   adam_loop
   widths

.. currentmodule:: pdft.completion.protocol

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate
   evaluate_params
   table1_scores
   heldout_mask
   budget_k
   train_k
   per_metric_best

.. currentmodule:: pdft.completion.metrics

.. autosummary::
   :toctree: generated
   :nosignatures:

   psnr
   ssim
   ms_ssim
   score

Coherence and the bridge
------------------------

.. currentmodule:: pdft.completion.coherence

.. autosummary::
   :toctree: generated
   :nosignatures:

   coherence
   dense_operator
   flat_modulus_deviation
   is_flat_modulus
   certify_flat_modulus

.. currentmodule:: pdft.completion.bridge

.. autosummary::
   :toctree: generated
   :nosignatures:

   qft_basis_from_angles
   qft_basis_from_general
   angles_from_qft_basis
   general_from_qft_basis
   bitrev_image

Families
--------

Every family starts at the DFT, so the comparisons are nested.

.. currentmodule:: pdft.completion.families.phases

.. autosummary::
   :toctree: generated
   :nosignatures:

   apply_u
   analysis
   synthesis
   reconstruct
   reconstruct_batch
   train
   comp_loss
   init_params
   unitary_phases
   coherence_phases

.. currentmodule:: pdft.completion.families.general

.. autosummary::
   :toctree: generated
   :nosignatures:

   init_general
   reconstruct_g
   train_general
   train_c
   unitary_general
   coherence_general
   count_params
   count_params_b

.. currentmodule:: pdft.completion.families.shared

.. autosummary::
   :toctree: generated
   :nosignatures:

   init_shared
   expand
   extend
   dist_index
   n_shared

.. currentmodule:: pdft.completion.families.butterfly

.. autosummary::
   :toctree: generated
   :nosignatures:

   apply_butterfly
   init_butterfly
   reconstruct_butterfly
   train_butterfly
   unitary_butterfly
   coherence_butterfly
   isometry_defect
   count_params
   mode_of

.. currentmodule:: pdft.completion.families.riemannian

.. autosummary::
   :toctree: generated
   :nosignatures:

   reconstruct_mat
   train_unitary
   dft_matrix
   skew
   cayley

.. currentmodule:: pdft.completion.families.transform_learning

.. autosummary::
   :toctree: generated
   :nosignatures:

   learn_transform
   learn_patch_transform
   reconstruct_t
   reconstruct_p
   sparsification_error
   compress_psnr
   dct_matrix
   dct2_patch

Baselines
---------

Per-image methods with no trained basis.

.. currentmodule:: pdft.completion.baselines.fixed_bases

.. autosummary::
   :toctree: generated
   :nosignatures:

   iht_fixed
   fixed_bases
   dct_pair
   dft_pair
   wavelet_pair
   hard_k

.. currentmodule:: pdft.completion.baselines.nuclear

.. autosummary::
   :toctree: generated
   :nosignatures:

   svp
   apg

.. currentmodule:: pdft.completion.baselines.qtt

.. autosummary::
   :toctree: generated
   :nosignatures:

   fit
   tt_image
   tt_svd
   tt_nparams_at
   level_schedule

Data
----

.. currentmodule:: pdft.completion.data

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_gray
   kodak_split
   div2k_split
   table1_split
   table1_val
   table1_test
   detail_window
   load_4k
