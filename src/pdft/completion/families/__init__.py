"""The trainable transform families the completion paper compares.

Every module here parameterises a basis and starts it at the DFT so the
comparisons are nested: ``general`` (the relaxed circuit of arXiv:2608.00053
--- Models A/B ride it, C frees its one-qubit gates), ``shared`` (phases tied
by gate distance, the resolution-transferable form), ``butterfly`` (the
learnable FFT factorisation of Dao et al.) and ``riemannian`` (a free unitary
on U(N) by Cayley SGD). The core method --- circuit, solver, coherence,
training loop, protocol --- lives one level up.
"""
