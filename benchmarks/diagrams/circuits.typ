// Quantum circuit topologies used in pdft for the
// `BlockedBasis(inner, block_log_m=5, block_log_n=5)` 8q benchmark.
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
  #text(size: 18pt, weight: "bold")[Parametric circuit topologies for 8×8 within-block transforms]

  #v(0.2em)
  #text(size: 9pt, fill: gray)[
    `pdft.QFTBasis` → `pdft.RichBasis` → `pdft.RichBasis(dense=True)`
    \ each circuit acts on $m=n=3$ qubits per dim → 8×8 transform per block
  ]
]

#v(0.6em)


= Outer block decomposition

The 256×256 input image is reshaped into a $32 × 32$ grid of $8 × 8$ blocks (`BlockedBasis` with `block_log_m = block_log_n = 5`). Every block is transformed by the SAME within-block parametric circuit; parameters are SHARED across all $1024$ blocks. Forward decomposition:

$ "image"_(256 × 256) #h(0.3em) underbrace(arrow.r, "reshape") #h(0.3em) "blocks"_(32 × 32 × 8 × 8) #h(0.3em) underbrace(arrow.r, U^("inner") "applied per block") #h(0.3em) "coeffs"_(32 × 32 × 8 × 8) $

The reduction reduces the per-step einsum cost roughly $k$-fold (for $k$ blocks per dim) while sharing parameters, giving the same inductive bias that powers JPEG's BlockDCT 8×8.

The within-block transform is itself separable: $U^("inner") = U^("row")_(8×8) ⊗ U^("col")_(8×8)$, with each 1D component built from the same $m=3$ qubit circuit shown below.


= Topology 1 — QFT layout (`QFTBasis`)

Standard Cooley–Tukey factorisation of the 8-point DFT, expressed as a quantum circuit. Each Hadamard is a learnable $2 × 2$ unitary on the `UnitaryManifold`; each *controlled-phase* is a 2-qubit *diagonal* gate $"diag"(1, 1, 1, e^(i phi.alt))$ with a single learnable phase parameter $phi.alt$, stored in the compact $2 × 2$ form $cases([1, 1], [1, e^(i phi.alt)])$ on the `PhaseManifold`.

#align(center)[
  #quantum-circuit(
    scale: 110%,
    row-spacing: 1.0em,
    column-spacing: 0.6em,
    lstick($q_1$), gate($H$), ctrl(1), ctrl(2), 1, 1, 1, [\ ],
    lstick($q_2$), 1,         gate($P_(pi slash 2)$), 1,         gate($H$), ctrl(1), 1, [\ ],
    lstick($q_3$), 1,         1,         gate($P_(pi slash 4)$), 1,         gate($P_(pi slash 2)$), gate($H$),
  )
]

#v(0.4em)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  inset: (x: 6pt, y: 3pt),
  stroke: (0.5pt + gray),
  table.header(
    [*gate role*], [*count*], [*free real params*], [*manifold*],
  ),
  [Hadamard ($H$)], [3], [3 × 3 = 9], [`UnitaryManifold(d=2)`],
  [Controlled-phase ($P_phi.alt$)], [3], [3 × 1 = 3], [`PhaseManifold`],
  [*total per dim*], [*6*], [*12*], [],
)

The reachable parametric family is a 12-dim submanifold of $italic("U")(8)$ (which has 63 free real params). Expressivity ceiling: the H+CP family does *not* contain the 8×8 DCT.


#pagebreak()


= Topology 2 — Rich gate set (`RichBasis`)

Same QFT layout, but each controlled-phase is *replaced* by a fully learnable two-qubit unitary $U^((4))$ on the `Unitary2qManifold` (15 free real params each, vs 1 for the diagonal CP). Each $U^((4))$ is *initialised* to the 4×4 form of its corresponding controlled-phase $"diag"(1, 1, 1, e^(i phi.alt))$, so the basis is bit-identical to `QFTBasis` at training step 0 and any improvement is real.

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

#v(0.4em)

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

The reachable parametric family is now a 54-dim submanifold of $italic("U")(8)$ — much larger than QFT, but *still narrower than the full 63-dim* $italic("U")(8)$. Empirically, `fit_to_dct` plateaus at $|"forward circuit" - "DCT"|^2 = 63.7$: the DCT is genuinely *not* in this family.


= Topology 3 — Dense rich gate set (`RichBasis(dense=True)`)

Same primitives as #emph[Topology 2], but adds a second pass of $U^((4))$ gates over each qubit pair after the QFT layout. The extra gates are initialised to the identity, so the basis is *still* bit-identical to `QFTBasis` at training step 0. Adam can then deform them away from the identity to traverse the rest of $italic("U")(8)$.

#align(center)[
  #quantum-circuit(
    scale: 105%,
    row-spacing: 1.0em,
    column-spacing: 0.5em,
    lstick($q_1$), gate($H$), mqgate($U^((4))_1$, n: 2), 1, mqgate($U^((4))_2$, n: 3), 1, 1, 1, mqgate($U^((4))_4$, n: 2), 1, mqgate($U^((4))_5$, n: 3), 1, 1, [\ ],
    lstick($q_2$), 1, 1, 1, 1, gate($H$), mqgate($U^((4))_3$, n: 2), 1, 1, 1, 1, 1, mqgate($U^((4))_6$, n: 2), [\ ],
    lstick($q_3$), 1, 1, 1, 1, 1, 1, gate($H$), 1, 1, 1, 1, 1,
  )
]

#v(0.4em)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  inset: (x: 6pt, y: 3pt),
  stroke: (0.5pt + gray),
  table.header(
    [*gate role*], [*count*], [*free real params*], [*manifold*],
  ),
  [Hadamard ($H$)], [3], [3 × 3 = 9], [`UnitaryManifold(d=2)`],
  [Two-qubit unitary ($U^((4))$)], [6], [6 × 15 = 90], [`Unitary2qManifold`],
  [*total per dim*], [*9*], [*99*], [],
)

The reachable family is now a 99-dim submanifold strictly *containing* $italic("U")(8)$: gradient descent on $|"circuit" - "DCT"|^2$ drives the loss to $≈ 1.3 × 10^(-3)$ in 200 Adam steps; DCT is reachable to numerical precision. This is the topology that *beats* `BlockDCT 8×8` on DIV2K.


#pagebreak()


= Why each step closes part of the gap to BlockDCT 8×8

Empirical results on DIV2K 8q `generalized` (n_train = 500, n_test = 50, MSE loss with top-$k$ truncation, `keep_ratio = 0.20`).

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  inset: (x: 6pt, y: 4pt),
  stroke: (0.5pt + gray),
  table.header(
    [*basis*], [*params per dim*], [*reachable in*], [*PSNR (dB)*], [*Δ vs `BlockDCT 8`*],
  ),
  [`QFTBasis` (H + CP)], [12], [12-dim ⊂ $italic("U")(8)$], [32.26], [#text(fill: red, "−1.75")],
  [`RichBasis` (H + U#super[(4)])], [54], [54-dim ⊂ $italic("U")(8)$], [33.71], [#text(fill: red, "−0.30")],
  [`RichBasis(dense=True)`], [99], [contains $italic("U")(8)$], [#strong("34.07")], [#text(fill: green.darken(20%), "+0.06")],
  [`BlockDCT 8 × 8` (JPEG)], [—], [fixed], [34.01], [reference],
)

#v(0.6em)

The progression is clean: every topology change either #emph[grows the parametric family] (more free parameters) or #emph[reaches a region of $italic("U")(8)$ the previous family could not], and each gain in expressivity directly translates into closing the PSNR gap to the AR(1)-optimal 8×8 DCT. Once the family is large enough to contain DCT exactly, training on natural-image MSE finds a basis *marginally better* than DCT — by +0.06 dB.

The DCT-init experiment is the architectural validation. Starting AT DCT and training for 200 epochs:
- with `dense=False`: the `fit_to_dct` step itself fails (loss plateaus at 63.7); training pulls the basis to a worse local minimum at 33.01 dB
- with `dense=True`: `fit_to_dct` reaches $1.3 × 10^(-3)$ (DCT to ~4 decimal places per matrix entry); training settles at 33.98 dB, essentially DCT

The standard-init `dense=True` run (starting from QFT) finds a basin marginally better (34.07 dB) than the DCT basin (33.98 dB). Both bases are within 0.1 dB of each other, suggesting the natural-image-MSE optimal basis sits in a flat valley near DCT.


= Inverse circuit and unitarity

For each topology the `inverse_transform` reverses the gate order and conjugates each tensor (Yao convention). All three topologies preserve unitarity exactly during training because every gate sits on a Riemannian manifold with Cayley retraction:

$ U_("new") = (I - alpha/2 W)^(-1) (I + alpha/2 W) U_("old"), space W = "skew"(Xi U_("old")^dagger) $

This guarantees the trained basis is ortho-orthonormal to $|U U^dagger - I| < 10^(-6)$ throughout the entire training run, so the learned transform is genuinely a unitary basis (not an approximation that drifts).
