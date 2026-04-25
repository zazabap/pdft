"""One-shot script to generate stub fixtures used by test_data_loading.py.
Run once: `python benchmarks/tests/fixtures/_build_fixtures.py`.
The output .npy/.png files are committed; this script is not invoked by tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent

# QuickDraw stub: 5 .npy files, each (10, 28*28) uint8 (raw QuickDraw shape).
# Real QuickDraw files are (N, 784); we keep the shape but use 10 rows.
qd = HERE / "quickdraw_stub"
qd.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
for cat in ("airplane", "apple"):
    arr = (rng.uniform(0, 255, size=(10, 28 * 28))).astype(np.uint8)
    np.save(qd / f"{cat}.npy", arr)

# DIV2K stub: 3 PNGs, 64x64 grayscale (small for fixture size; loader resizes).
div = HERE / "div2k_stub"
div.mkdir(exist_ok=True)
for i in range(1, 4):
    pix = (rng.uniform(0, 255, size=(64, 64))).astype(np.uint8)
    Image.fromarray(pix, mode="L").save(div / f"{i:04d}.png")

print(f"wrote stubs to {HERE}")
