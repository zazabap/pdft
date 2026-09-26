Completion
==========

.. currentmodule:: pdft.completion

Image inpainting from random pixels with a trainable QFT: the circuit applied to the image gate by gate, the unrolled recovery solver it is trained through, and the exact bridge to :class:`pdft.QFTBasis`. A transform family is one per-axis operator ``apply(x, params, adjoint, axis)``; the 2-D pair, the solver and the evaluation are derived from it.

.. automodule:: pdft.completion
   :no-members:

.. rubric:: The circuit and its operators

.. autosummary::
   :toctree: generated
   :nosignatures:

   apply_gates
   apply_u
   apply_dense
   theta0
   theta_to_params
   gate_pairs
   n_params
   separable
   analysis
   synthesis
   unitary_matrix
   coherence

.. rubric:: The solver

.. autosummary::
   :toctree: generated
   :nosignatures:

   hard_k
   soft_k
   iht
   solver_for
   batched
   reconstruct
   reconstruct_batch
   reconstruct_bounded
   plan
   estimate_peak
   split_k
   report

.. rubric:: Training and evaluation

.. autosummary::
   :toctree: generated
   :nosignatures:

   train
   adam_loop
   minibatches
   task_loss
   init_params
   evaluate
   evaluate_params
   psnr
   ssim
   ms_ssim
   score

.. rubric:: Coherence and the bridge

.. autosummary::
   :toctree: generated
   :nosignatures:

   certify_flat_modulus
   dense_operator
   is_flat_modulus
   qft_basis_from_angles
   qft_basis_from_general
   angles_from_qft_basis
   general_from_qft_basis
   bitrev_image

.. rubric:: Data

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_gray
   kodak_split
   table1_split
   table1_val
   table1_test

The trained families (``pdft.completion.families``: ``general`` for QFT + diagonals and QFT + rotations, ``shared`` for phases tied by gate distance, ``butterfly`` and ``riemannian``) and the per-image baselines (``pdft.completion.baselines``: fixed bases, nuclear norm, tensor train, transform learning) are imported from their modules; each family module declares its operator and exposes ``reconstruct_*`` and ``train_*`` built from the helpers above.
