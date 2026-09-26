"""Per-image methods with no trained basis.

``fixed_bases`` (DCT-II, DFT and wavelet IHT in numpy), ``nuclear`` (low-rank
matrix completion) and ``qtt`` (the coarse-to-fine quantized tensor train after
PuTT) refit nothing, or refit to each test image alone. They are what the
trained families in :mod:`pdft.completion.families` are measured against.
"""
