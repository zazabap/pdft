"""Per-image competitors: methods with no trainable basis.

``fixed_bases`` (DCT-II / DFT / wavelet IHT), ``nuclear`` (low-rank matrix
completion, the classical row), ``qtt`` (the coarse-to-fine quantized tensor
train after PuTT) and ``transform_learning`` (the square orthonormal member of
dictionary learning). None of these import the circuit machinery; they are what
the trained families in :mod:`pdft.completion.families` are measured against.
"""
