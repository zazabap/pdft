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
| RealRichBasis (Approach A) | `src/pdft/real_rich_basis.py::_real_rich_qft_gates_1d` |
| DCTBasis (Approach B) | `src/pdft/dct_basis.py` |
| BlockedBasis wrapper | `src/pdft/block_basis.py` |

## `parameter_efficiency.{typ,pdf}` + `parameter_efficiency_table.tex`

Pareto frontier figure (free-real-parameter count vs PSNR @ kr=0.20) and a
paper-ready LaTeX `tabular` block summarising the same data. The two
defensible parameter-efficiency claims are:

1. **Strict Pareto win over BlockFFT $8{\times}8$**: `QFTBasis` (24 free
   params) beats `BlockFFT` (128 free params) by $+1.47$ dB.
2. **DCT-comparable PSNR with 25% fewer params**: `RealRichBasis` (42 free
   params, real-orthogonal) reaches within 0.31 dB of `BlockDCT 8x8` (56
   free params).

Both use the manifold-aware free-parameter count: $\dim \mathrm{SU}(2) = 3$,
$\dim \mathrm{SU}(4) = 15$, $\dim \mathrm{O}(8) = 28$.

The `.tex` file is meant to be `\input`'d directly into the paper's main
table list; it uses the standard `booktabs` + `xcolor` packages.
