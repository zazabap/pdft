Bases
=====

Trainable sparsifying bases. Circuit bases (QFT, entangled QFT, TEBD, MERA, Rich, RealRich, DCT-IV) act on a whole ``2^m x 2^n`` image; :class:`BlockedBasis` tiles an inner basis over image blocks. The ``*_code`` and ``*_mat`` helpers build the underlying einsum circuits and reference DFT matrices.

.. currentmodule:: pdft.bases

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
   AbstractSparseBasis
   freeze_as_blocked
   fit_to_dct
   bases_allclose
   qft_code
   entangled_qft_code
   tebd_code
   mera_code
   ft_mat
   ift_mat
