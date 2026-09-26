"""Generate the pdft logo assets in docs/_static/.

The mark is a two-wire circuit fragment (Hadamard, controlled phase) whose
wires leave the phase gate as waves, on a plum-to-coral tile with a gold phase
dot. The wordmark is "pdft" set in Inter Bold and converted to outlines, so
the SVGs render identically without the font installed.

Run:  python docs/logo/build_logo.py path/to/Inter-Bold.ttf
Needs fonttools. Inter is SIL OFL: https://github.com/rsms/inter
Writes: logo-light.svg, logo-dark.svg (lockups), mark.svg (bare mark).
favicon.png is mark.svg rasterised at 64 px (any SVG renderer).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parents[1] / "_static"
VB = 512
WHITE, INK, PAPER = "#ffffff", "#1f2933", "#f3f5f7"
TILE = ("#2a0b3d", "#7a2a6d", "#e2557a")  # dark floor, mid, bright corner
DOT = ("#ffe7a3", "#ffb43c")  # phase dot: lit centre, gold rim (also the halo)
WORD = "#b03a72"  # the "dft" of the wordmark
MARK = "D"

DEFS = f"""<defs>
  <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{TILE[0]}"/><stop offset="0.55" stop-color="{TILE[1]}"/><stop offset="1" stop-color="{TILE[2]}"/>
  </linearGradient>
  <radialGradient id="dotg" cx="0.35" cy="0.35" r="0.85">
    <stop offset="0" stop-color="{DOT[0]}"/><stop offset="1" stop-color="{DOT[1]}"/>
  </radialGradient>
</defs>"""


def wire(x1, y1, x2, y2, w=16, color=WHITE):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>'


def path(d, w=16, color=WHITE):
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>'


def dot(x, y, r, color=WHITE):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>'


def glow_dot(x, y, r):
    # The halo is two translucent discs rather than a blur filter, so it
    # survives every renderer, including favicon rasterisers.
    return (
        f'<circle cx="{x}" cy="{y}" r="{r + 26}" fill="{DOT[1]}" fill-opacity="0.16"/>'
        f'<circle cx="{x}" cy="{y}" r="{r + 12}" fill="{DOT[1]}" fill-opacity="0.32"/>'
        + dot(x, y, r, "url(#dotg)")
    )


def h_gate(cx, cy, s=100):
    b, ih, g = 17, s * 0.5, TILE[0]
    return (
        f'<rect x="{cx - s / 2}" y="{cy - s / 2}" width="{s}" height="{s}" rx="20" fill="{WHITE}"/>'
        f'<rect x="{cx - s * 0.25 - b / 2}" y="{cy - ih / 2}" width="{b}" height="{ih}" fill="{g}"/>'
        f'<rect x="{cx + s * 0.25 - b / 2}" y="{cy - ih / 2}" width="{b}" height="{ih}" fill="{g}"/>'
        f'<rect x="{cx - s * 0.25}" y="{cy - b / 2}" width="{s * 0.5}" height="{b}" fill="{g}"/>'
    )


def sine(x0, x1, y, amp, lam, phase=0.0, n=60):
    xs = [x0 + (x1 - x0) * i / n for i in range(n + 1)]
    return "M" + " L".join(
        f"{x:.1f},{y + amp * math.sin(2 * math.pi * (x - x0) / lam + phase):.1f}" for x in xs
    )


def mark():
    y1, y2 = 186, 326
    tile = f'<rect width="{VB}" height="{VB}" rx="104" fill="url(#tile)"/>'
    cp = wire(300, y1, 300, y2) + dot(300, y1, 26) + glow_dot(300, y2, 30)
    return (
        tile
        + wire(56, y1, 300, y1)
        + wire(56, y2, 300, y2)
        + h_gate(150, y1)
        + cp
        + path(sine(300, 456, y1, 14, 104))
        + path(sine(300, 456, y2, 26, 104, math.pi), color=DOT[1])
    )


def wordmark(font_path, size=268, tracking=-8):
    """Outline 'pdft' at the given pixel size; returns (paths, advance) with the
    'p' and 'dft' as separate path strings so they can be coloured apart."""
    font = TTFont(font_path)
    scale = size / font["head"].unitsPerEm
    cmap, glyphs, hmtx = font.getBestCmap(), font.getGlyphSet(), font["hmtx"]
    x, out = 0.0, {}
    for ch in "pdft":
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, 0)))
        out[ch] = pen.getCommands()
        x += hmtx[name][0] * scale + tracking
    return out["p"], out["d"] + out["f"] + out["t"], x - tracking


def svg(body, w=VB, h=VB):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n{DEFS}\n{body}\n</svg>\n'


def main(font_path):
    OUT.mkdir(exist_ok=True)
    (OUT / "mark.svg").write_text(svg(mark()))
    p_path, dft_path, advance = wordmark(font_path)
    x0, base, pad = 440, 292, 28
    width = round(x0 + advance + pad)
    for name, p_fill in (("logo-light.svg", INK), ("logo-dark.svg", PAPER)):
        body = (
            f'<g transform="scale(0.75)">{mark()}</g>'
            f'<g transform="translate({x0},{base})"><path d="{p_path}" fill="{p_fill}"/><path d="{dft_path}" fill="{WORD}"/></g>'
        )
        (OUT / name).write_text(svg(body, width, 384))
    print(f"wrote mark.svg, logo-light.svg, logo-dark.svg ({width}x384) to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
