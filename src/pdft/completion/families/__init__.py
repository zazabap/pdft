"""The transform families, one per-axis operator each.

``phases`` is the completion paper's own family (one controlled phase per wire
pair, Hadamards fixed); ``general`` relaxes the same circuit (all four phases
of each gate, then the one-qubit gates too); ``shared`` ties the phases by gate
distance so a fit transfers across resolutions; ``butterfly`` frees the 2x2
blocks of the FFT dataflow; ``riemannian`` is a free unitary on U(N);
``transform_learning`` is a separable orthonormal pair fitted for sparsity.
Every family starts at the DFT, so the comparisons are nested, and every one
gets its solver, batched solver and evaluation from the generic helpers.
"""
