// Quantum circuit topologies used in pdft for the BlockedBasis 8q benchmark.
// Style mirrors ParametricDFT.jl/note/main.typ.
//
// Compile:  typst compile circuits.typ circuits.pdf

#import "@preview/quill:0.6.0": *

#set page(paper: "a4", margin: (x: 1.6cm, y: 1.8cm))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#show heading: set block(above: 1.4em, below: 0.8em)
#show heading.where(level: 1): set text(size: 14pt, weight: "bold")
#show heading.where(level: 2): set text(size: 12pt, weight: "bold")

#align(center)[
  #text(size: 18pt, weight: "bold")[Parametric circuit topologies for 8 × 8 within-block transforms]

  #v(0.2em)
  #text(size: 9pt, fill: gray)[
    Three sub-`SU(8)` parametric families, all with strictly fewer free real
    parameters than `dim SU(8) = 63`. Each acts on $m=n=3$ qubits per dim.
  ]
]

#v(0.6em)


= Outer block decomposition

The 256 × 256 image is reshaped into a $32 × 32$ grid of $8 × 8$ blocks (`BlockedBasis` with `block_log_m = block_log_n = 5`). Every block is transformed by the same shared parametric circuit.

$ "image"_(256 × 256) #h(0.3em) underbrace(arrow.r, "reshape") #h(0.3em) "blocks"_(32 × 32 × 8 × 8) #h(0.3em) underbrace(arrow.r, U^("inner") "applied per block") #h(0.3em) "coeffs"_(32 × 32 × 8 × 8) $

The within-block transform is separable: $U^("inner") = U^("row")_(8 × 8) ⊗ U^("col")_(8 × 8)$, with each 1D component built from the same $m = 3$ qubit circuit shown below.


= `RichBasis` — H + complex U(4)

QFT-style topology with each controlled-phase gate replaced by a fully learnable 4 × 4 *complex* unitary. Storage: `(2, 2, 2, 2)` per 2-qubit gate, classified by `Unitary2qManifold`.

#align(center)[
  #quantum-circuit(
    scale: 110%,
    row-spacing: 1.0em,
    column-spacing: 0.6em,
    lstick($q_1$), gate($H$), mqgate($U^((4))$, n: 2), 1, mqgate($U^((4))$, n: 3), 1, 1, 1, [\ ],
    lstick($q_2$), 1, 1,         1,                          1, gate($H$), mqgate($U^((4))$, n: 2), 1, [\ ],
    lstick($q_3$), 1, 1,         1,                          1, 1,         1,                          gate($H$),
  )
]

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  inset: (x: 6pt, y: 3pt),
  stroke: (0.5pt + gray),
  table.header(
    [*gate role*], [*count*], [*free real params*], [*manifold*],
  ),
  [Hadamard ($H$)], [3], [3 × 3 = 9], [`UnitaryManifold(d=2)`],
  [Two-qubit unitary ($U^((4))$)], [3], [3 × 15 = 45], [`Unitary2qManifold`],
  [*total per dim*], [*6*], [*54*], [],
)

54-dim submanifold of `SU(8)`. Empirically does not contain DCT (`fit_to_dct` plateaus at Frobenius² ≈ 63.7); achieves 33.71 dB on DIV2K 8q at kr = 0.20 (BlockDCT 8 reference: 34.01).


#pagebreak()


= Approach A — `RealRichBasis` (H + real-orthogonal U(4))

Same QFT topology, but every tensor is constrained to be REAL-valued. The H slots are canonical Hadamards (in O(2)); the 2-qubit slots are 4 × 4 *real-orthogonal* matrices initialised at the identity, evolving in the connected component of $italic("SO")(4)$. Real images and real gradients keep the Cayley retraction inside the orthogonal manifold automatically.

#align(center)[
  #quantum-circuit(
    scale: 110%,
    row-spacing: 1.0em,
    column-spacing: 0.6em,
    lstick($q_1$), gate($H$), mqgate($O^((4))$, n: 2), 1, mqgate($O^((4))$, n: 3), 1, 1, 1, [\ ],
    lstick($q_2$), 1, 1,         1,                          1, gate($H$), mqgate($O^((4))$, n: 2), 1, [\ ],
    lstick($q_3$), 1, 1,         1,                          1, 1,         1,                          gate($H$),
  )
]

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  inset: (x: 6pt, y: 3pt),
  stroke: (0.5pt + gray),
  table.header(
    [*gate role*], [*count*], [*free real params*], [*manifold*],
  ),
  [Hadamard ($H$, real)], [3], [3 × 1 = 3], [`UnitaryManifold(d=2)` ∩ real],
  [Two-qubit orthogonal ($O^((4))$)], [3], [3 × 6 = 18], [`Unitary2qManifold` ∩ real],
  [*total per dim*], [*6*], [*21*], [],
)

21 free real parameters per dim — far below `dim O(8) = 28`. A strict submanifold of $italic("O")(8)$. Whether it contains DCT exactly is empirical; if it does not, training cannot match BlockDCT.


= Approach B — `DCTBasis` (1D macro-gate, init at canonical DCT)

Bypasses the gate-decomposition path entirely. Each 1D direction is parametrised by a single 8 × 8 real-orthogonal matrix, initialised at the canonical orthonormal DCT-II. The Cooley–Tukey style butterfly factorisation (Loeffler 1989 etc.) is a known *fast* implementation of this same matrix and would give an isomorphic circuit; we don't expand it here, instead exposing the matrix as a single trainable rotation in $italic("O")(8)$.

#align(center)[
  #quantum-circuit(
    scale: 110%,
    row-spacing: 1.0em,
    column-spacing: 1.0em,
    lstick($q_1$), mqgate($D_("row")^((8 × 8))$, n: 3), [\ ],
    lstick($q_2$), 1, [\ ],
    lstick($q_3$), 1,
  )
]

#v(0.2em)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  inset: (x: 6pt, y: 3pt),
  stroke: (0.5pt + gray),
  table.header(
    [*gate role*], [*count*], [*free real params*], [*manifold*],
  ),
  [Macro-gate ($D_("row")$, $D_("col")$)], [2], [2 × 28 = 56], [`UnitaryManifold(d=8)`],
  [*total*], [*2*], [*56*], [],
)

56 free real parameters total. Provably *contains* DCT (it IS DCT at initialisation); also strictly below `dim SU(8) = 63`. The experiment Approach B answers cleanly: starting AT DCT and training on natural-image MSE, does Adam find a better basis (move away from DCT) or stay near DCT?


#pagebreak()


= Why each approach is meaningful

A meaningful parametric family must have strictly fewer free parameters than $dim italic("SU")(8) = 63$, otherwise the structural constraint becomes vacuous and training collapses to "learn an arbitrary 8 × 8 unitary on the Stiefel manifold." Both Approach A and Approach B respect this.

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  inset: (x: 6pt, y: 4pt),
  stroke: (0.5pt + gray),
  table.header(
    [*basis*], [*free params per dim*], [*total*], [*relationship to DCT*],
  ),
  [`RichBasis` (H + complex U(4))], [54], [108], [strict 54-dim ⊂ SU(8); does NOT contain DCT],
  [`RealRichBasis` (Approach A)], [21], [42], [strict 21-dim ⊂ O(8); empirical whether DCT is in family],
  [`DCTBasis` (Approach B)], [28 (per dim)], [56], [contains DCT exactly at init],
  [`BlockDCT 8 × 8`], [— (fixed)], [—], [the canonical reference (= 34.01 dB)],
)

The progression is meant to *separate* two sources of inability to beat DCT: structural (the family does not contain DCT) versus optimisation (DCT is in the family but Adam cannot leave it). Approach A tests the former; Approach B tests the latter.


= Inverse circuit and unitarity

For each topology the `inverse_transform` reverses the gate order and conjugates each tensor (Yao convention). Both A and B preserve unitarity exactly during training because every tensor sits on a Riemannian manifold with Cayley retraction:

$ U_("new") = (I - alpha/2 W)^(-1) (I + alpha/2 W) U_("old"), space W = "skew"(Xi U_("old")^dagger) $

For `RealRichBasis`, the input image is real and the gradient is therefore real, so the projection $W$ stays real and Cayley retraction stays in $italic("O")(d)$. For `DCTBasis`, the same machinery applies at $d = 8$.
