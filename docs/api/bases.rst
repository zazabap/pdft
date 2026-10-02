Bases
=====

.. currentmodule:: pdft.bases

Trainable sparsifying bases. Circuit bases (QFT, entangled QFT, TEBD, MERA, Rich, RealRich, DCT-IV) act on a whole ``2^m x 2^n`` image; :class:`BlockedBasis` tiles an inner basis over image blocks. Every circuit basis is a :class:`CircuitBasis`: a gate program and the tensors of its gates. The ``*_code`` helpers return a family's applier and initial tensors; ``ft_mat`` and ``ift_mat`` apply one to an image.

.. automodule:: pdft.bases
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   QFTBasis
   EntangledQFTBasis
   TEBDBasis
   MERABasis
   DCT4Basis
   RichBasis
   RealRichBasis
   BlockedBasis
   CircuitBasis
   AbstractSparseBasis
   freeze_as_blocked
   fit_to_dct
   bases_allclose
   program_of
   with_tensors
   cp_phases
   with_cp_phases
   cp_diagonals
   with_cp_diagonals
   ParameterView
   TENSORS
   CP_PHASES
   CP_DIAGONALS
   qft_code
   entangled_qft_code
   tebd_code
   mera_code
   dct4_code
   ft_mat
   ift_mat
