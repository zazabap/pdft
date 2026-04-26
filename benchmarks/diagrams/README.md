# Circuit diagrams for the BlockedBasis paper extension

`circuits.typ` is the master diagram document — three quantum-circuit topologies
side-by-side (QFT, Rich, Dense Rich) with parameter accounting and the empirical
results that motivate each promotion.

The structure mirrors `ParametricDFT.jl/note/main.typ`: typst source with
`@preview/quill` for circuit drawing.

## Compile

```bash
# Local install (one-time):
typst compile circuits.typ circuits.pdf

# Or use a sandboxed local compile:
# https://typst.app/  (paste the file in, no install needed)
```

The first compile fetches `@preview/quill:0.6.0` automatically; subsequent
compiles are offline.

## What's drawn

1. **Outer block decomposition** — how the 256×256 image splits into a
   32×32 grid of 8×8 blocks, each transformed by the SAME parametric
   within-block circuit.
2. **QFTBasis** — H + 1-parameter controlled-phase. 12-dim parametric
   family. Cannot express DCT.
3. **RichBasis** — H + 15-parameter $U^{(4)}$. 54-dim family. Closer to
   DCT but still not containing it.
4. **RichBasis(dense=True)** — H + $U^{(4)}$ × 2 passes. 99-dim family,
   universal for $U(8)$. **Beats BlockDCT 8×8 by +0.06 dB**.

## Source files this depicts

| topology | source |
|---|---|
| QFTBasis | `src/pdft/qft.py::_qft_gates_1d` |
| RichBasis | `src/pdft/rich_basis.py::_rich_qft_gates_1d` |
| RichBasis(dense=True) | `src/pdft/rich_basis.py::_dense_qft_gates_1d` |
| BlockedBasis wrapper | `src/pdft/block_basis.py` |
