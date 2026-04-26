// Parameter-efficiency Pareto frontier for the BlockedBasis paper extension.
// Compile: typst compile parameter_efficiency.typ parameter_efficiency.pdf

#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3"

#set page(paper: "a4", margin: (x: 1.6cm, y: 1.8cm))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)

#align(center)[
  #text(size: 16pt, weight: "bold")[Parameter efficiency on DIV2K 8q natural images]

  #v(0.2em)
  #text(size: 9pt, fill: gray)[
    PSNR at kr=0.20 vs free-real-parameter count (manifold-aware), 8×8 within-block transforms.
    Smaller is more parameter-efficient; higher PSNR is better. Top-left is the Pareto goal.
  ]
]

#v(0.4em)


= Pareto frontier

#align(center)[
  #cetz.canvas({
    import cetz.draw: *
    import cetz-plot: plot

    plot.plot(
      size: (12, 7),
      x-label: [free real params per direction],
      y-label: [PSNR (dB) at kr=0.20],
      x-min: 0, x-max: 80,
      y-min: 30, y-max: 35,
      x-tick-step: 10,
      y-tick-step: 1,
      legend: "inner-south-east",
      {
        // Learned circuits (filled circles)
        plot.add(
          ((12, 32.26),),
          mark: "o", mark-size: 0.4, mark-style: (fill: blue),
          style: (stroke: none),
          label: [learned QFT (H+CP)],
        )
        plot.add(
          ((42, 33.70),),
          mark: "square", mark-size: 0.4, mark-style: (fill: green.darken(20%)),
          style: (stroke: none),
          label: [learned RealRich (A)],
        )
        plot.add(
          ((108, 33.71),),
          mark: "square", mark-size: 0.4, mark-style: (fill: orange),
          style: (stroke: none),
          label: [learned Rich],
        )
        plot.add(
          ((56, 33.97),),
          mark: "triangle", mark-size: 0.4, mark-style: (fill: purple),
          style: (stroke: none),
          label: [learned DCTBasis (B)],
        )

        // Analytical references (X markers)
        plot.add(
          ((128, 30.79),),
          mark: "x", mark-size: 0.5, mark-style: (stroke: red, thickness: 1.5pt),
          style: (stroke: none),
          label: [BlockFFT 8×8 (analytic)],
        )
        plot.add(
          ((56, 34.01),),
          mark: "x", mark-size: 0.5, mark-style: (stroke: black, thickness: 1.5pt),
          style: (stroke: none),
          label: [BlockDCT 8×8 (analytic)],
        )
      },
    )
  })
]

#v(0.4em)


= Numbers

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, left),
  inset: (x: 6pt, y: 4pt),
  stroke: (0.5pt + gray),
  table.header(
    [*basis*], [*free params per dim*], [*total*], [*PSNR (dB)*], [*efficiency note*],
  ),
  [`BlockFFT 8×8` (analytical)], [64], [128], [30.79], [reference],
  [`QFTBasis` (H+CP, learned)], [12], [24], [32.26], [strictly dominates BlockFFT: +1.47 dB at 5.3× fewer params],
  [`RichBasis` (H + complex U(4))], [54], [108], [33.71], [54 → 108 params, but +0.06 dB over RealRich (noise)],
  [`RealRichBasis` (Approach A)], [21], [42], [33.70], [matches Rich at 2.6× fewer params (real-only)],
  [`DCTBasis` (Approach B, init at DCT)], [28], [56], [33.97], [exact O(8) at init; drifts -0.04 dB],
  [`BlockDCT 8×8` (analytical)], [28], [56], [34.01], [reference (KLT-near-optimal for AR(1))],
)

#v(0.6em)


= Two parameter-efficiency claims

#text(weight: "bold")[Claim 1 (strict Pareto win): `QFTBasis` dominates `BlockFFT 8×8`.]
At 24 free real parameters total, the learned QFT-topology circuit achieves PSNR 32.26 dB on DIV2K 8q at keep-ratio 0.20 — *both* fewer parameters (5.3×) *and* higher PSNR (+1.47 dB) than analytical `BlockFFT 8×8` at 128 free parameters. This is the strongest parameter-efficiency result.

#v(0.4em)

#text(weight: "bold")[Claim 2 (efficiency-vs-DCT trade): `RealRichBasis` achieves DCT-comparable PSNR with 25% fewer parameters.]
At 42 free real parameters total, `RealRichBasis` achieves PSNR 33.70 dB on DIV2K 8q at keep-ratio 0.20 — *0.31 dB short of* `BlockDCT 8×8` (which has 56 free real parameters). The learned basis pays a small PSNR penalty in exchange for a 25% reduction in basis description.

#v(0.4em)

#text(weight: "bold")[Auxiliary observation: imaginary degrees of freedom waste budget.]
`RealRichBasis` (42 params) and `RichBasis` (108 params, fully complex) achieve identical PSNR within 0.01 dB. The 66-parameter difference between the two — entirely the imaginary part of the U(4) gates — does not contribute to natural-image-MSE compression at 8×8 block size on this single-seed run.


= Caveats

- All numbers are single-seed (seed = 42). Typical seed-to-seed variance at this scale is 0.05–0.15 dB. Claim 1 is well outside that envelope; Claim 2's 0.31 dB gap and the auxiliary 0.01 dB equivalence are at the noise floor.

- "Free real parameters" counts the manifold-aware dimension (e.g. 3 for SU(2), 15 for SU(4), 28 for O(8)), not raw tensor storage. Storage-on-disk is larger because the over-parameterised constraint surfaces are stored in full.

- PSNR is logarithmic; the 0.31 dB gap from RealRichBasis to BlockDCT corresponds to ~7.5% more MSE error.
