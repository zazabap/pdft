"""Generate the pdft logo assets in docs/_static/.

The mark is the two-qubit QFT circuit: a Hadamard, a controlled phase, a
second Hadamard, and the final swap drawn as crossing wires, on a
navy-to-teal tile lit from the top left, with an amber phase dot. The
wordmark is "pdft" set in Inter Bold, two-tone, converted to outlines so the
SVGs render identically without the font installed.

Run:  python docs/logo/build_logo.py path/to/Inter-Bold.ttf
Needs fonttools. Inter is SIL OFL: https://github.com/rsms/inter
Writes: logo-light.svg, logo-dark.svg (lockups), mark.svg (bare mark) and
favicon.svg, a reduced mark (no Hadamards, heavier strokes) that stays
legible at 16 px, and favicon.png (favicon.svg at 64 px) when resvg-py is
installed; conf.py uses the PNG because Safari ignores SVG favicons.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parents[1] / "_static"
VB = 512
WHITE, INK, PAPER = "#ffffff", "#1f2933", "#f3f5f7"
TILE = ("#06263a", "#0a5f7c", "#17a0ad")  # dark floor, mid, bright corner
DOT = ("#ffc77a", "#ef7f3a")  # phase dot: lit centre, amber rim (also the halo)
WORD = "#0f8a9c"  # the "dft" of the wordmark
Y1, Y2 = 186, 326  # the two wires

DEFS = f"""<defs>
  <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{TILE[0]}"/><stop offset="0.55" stop-color="{TILE[1]}"/><stop offset="1" stop-color="{TILE[2]}"/>
  </linearGradient>
  <radialGradient id="light" cx="0.22" cy="0.18" r="0.75">
    <stop offset="0" stop-color="{WHITE}" stop-opacity="0.22"/><stop offset="1" stop-color="{WHITE}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="vign" cx="0.85" cy="0.9" r="0.7">
    <stop offset="0" stop-color="#000000" stop-opacity="0.22"/><stop offset="1" stop-color="#000000" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="dotg" cx="0.35" cy="0.35" r="0.85">
    <stop offset="0" stop-color="{DOT[0]}"/><stop offset="1" stop-color="{DOT[1]}"/>
  </radialGradient>
</defs>"""


def stroke(d, w, color=WHITE, op=1.0):
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" '
        f'stroke-linecap="round" stroke-linejoin="round" stroke-opacity="{op}"/>'
    )


def dot(x, y, r, color=WHITE):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>'


def glow_dot(x, y, r):
    # The halo is translucent discs rather than a blur filter, so it survives
    # every renderer, including favicon rasterisers.
    return (
        f'<circle cx="{x}" cy="{y}" r="{r + 34}" fill="{DOT[1]}" fill-opacity="0.10"/>'
        f'<circle cx="{x}" cy="{y}" r="{r + 22}" fill="{DOT[1]}" fill-opacity="0.18"/>'
        f'<circle cx="{x}" cy="{y}" r="{r + 10}" fill="{DOT[1]}" fill-opacity="0.34"/>'
        + dot(x, y, r, "url(#dotg)")
        + f'<circle cx="{x - r * 0.32}" cy="{y - r * 0.34}" r="{r * 0.28}" fill="{WHITE}" fill-opacity="0.55"/>'
    )


def h_gate(cx, cy, s=88):
    b, ih, g = 16, s * 0.5, TILE[0]
    box = f'x="{cx - s / 2}" width="{s}" height="{s}" rx="18"'
    return (
        f'<rect {box} y="{cy - s / 2 + 6}" fill="#000000" fill-opacity="0.22"/>'
        f'<rect {box} y="{cy - s / 2}" fill="{WHITE}"/>'
        f'<rect x="{cx - s * 0.25 - b / 2}" y="{cy - ih / 2}" width="{b}" height="{ih}" rx="3" fill="{g}"/>'
        f'<rect x="{cx + s * 0.25 - b / 2}" y="{cy - ih / 2}" width="{b}" height="{ih}" rx="3" fill="{g}"/>'
        f'<rect x="{cx - s * 0.25}" y="{cy - b / 2}" width="{s * 0.5}" height="{b}" rx="3" fill="{g}"/>'
    )


def tile():
    return (
        f'<rect width="{VB}" height="{VB}" rx="104" fill="url(#tile)"/>'
        f'<rect width="{VB}" height="{VB}" rx="104" fill="url(#light)"/>'
        f'<rect width="{VB}" height="{VB}" rx="104" fill="url(#vign)"/>'
        f'<rect x="4" y="4" width="{VB - 8}" height="{VB - 8}" rx="100" fill="none" '
        f'stroke="{WHITE}" stroke-opacity="0.16" stroke-width="3"/>'
    )


def swap(x0, x1, w):
    """The QFT's final swap: each wire crosses to the other's position."""
    c0, c1 = x0 + 0.42 * (x1 - x0), x0 + 0.58 * (x1 - x0)
    return stroke(f"M{x0},{Y1} C{c0},{Y1} {c1},{Y2} {x1},{Y2}", w) + stroke(
        f"M{x0},{Y2} C{c0},{Y2} {c1},{Y1} {x1},{Y1}", w
    )


def mark():
    """H, controlled phase, H, swap: the two-qubit QFT."""
    x_h1, x_cp, x_h2, x_sw0, x_sw1 = 110, 218, 316, 372, 468
    wires = stroke(f"M44,{Y1 + 6} H{x_sw0}", 16, "#000000", 0.2) + stroke(
        f"M44,{Y2 + 6} H{x_sw0}", 16, "#000000", 0.2
    )
    wires += stroke(f"M44,{Y1} H{x_sw0}", 16) + stroke(f"M44,{Y2} H{x_sw0}", 16)
    cp = stroke(f"M{x_cp},{Y1} V{Y2}", 16) + dot(x_cp, Y1, 24)
    return (
        tile()
        + wires
        + swap(x_sw0, x_sw1, 16)
        + h_gate(x_h1, Y1)
        + h_gate(x_h2, Y2)
        + cp
        + glow_dot(x_cp, Y2, 28)
    )


def favicon_mark():
    """Reduced mark for 16-32 px: the phase gate and the swap, heavier strokes, no Hadamards."""
    x_cp, x_sw0, x_sw1 = 190, 300, 460
    wires = stroke(f"M52,{Y1} H{x_sw0}", 26) + stroke(f"M52,{Y2} H{x_sw0}", 26)
    cp = stroke(f"M{x_cp},{Y1} V{Y2}", 26) + dot(x_cp, Y1, 36)
    return tile() + wires + swap(x_sw0, x_sw1, 26) + cp + glow_dot(x_cp, Y2, 44)


def wordmark(font_path, size=268, tracking=-8):
    """Outline 'pdft'; returns the 'p' and 'dft' paths separately plus the advance."""
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
    out = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        f"{DEFS}\n{body}\n</svg>\n"
    )
    # Two decimals is far below anything visible at 512 px; keeps diffs readable.
    return re.sub(r"-?\d+\.\d+", lambda m: f"{round(float(m.group()), 2):g}", out)


def main(font_path):
    OUT.mkdir(exist_ok=True)
    (OUT / "mark.svg").write_text(svg(mark()))
    (OUT / "favicon.svg").write_text(svg(favicon_mark()))
    p_path, dft_path, advance = wordmark(font_path)
    x0, base, pad = 440, 292, 28
    width = round(x0 + advance + pad)
    for name, p_fill in (("logo-light.svg", INK), ("logo-dark.svg", PAPER)):
        body = (
            f'<g transform="scale(0.75)">{mark()}</g>'
            f'<g transform="translate({x0},{base})">'
            f'<path d="{p_path}" fill="{p_fill}"/><path d="{dft_path}" fill="{WORD}"/></g>'
        )
        (OUT / name).write_text(svg(body, width, 384))
    print(f"wrote mark.svg, favicon.svg, logo-light.svg, logo-dark.svg ({width}x384) to {OUT}")
    try:
        import resvg_py
    except ImportError:
        print("resvg-py not installed: favicon.png NOT regenerated (pip install resvg-py)")
    else:
        png = resvg_py.svg_to_bytes(svg_path=str(OUT / "favicon.svg"), width=64, height=64)
        (OUT / "favicon.png").write_bytes(bytes(png))
        print("wrote favicon.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} path/to/Inter-Bold.ttf")
    main(sys.argv[1])
