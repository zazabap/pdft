"""Rewrite the characterisation snapshots from the code as it is now.

    python -m tests.characterisation.regenerate            # overwrite snapshots.npz
    python -m tests.characterisation.regenerate --out x.npz # write elsewhere

Run this deliberately, never to make a failing test pass: a snapshot that moves
is a behaviour change, and the commit that regenerates it has to say which one
and why. Writing to ``--out`` and pointing ``PDFT_SNAPSHOT_FILE`` at the result
gives a local reference for the bit-exact mode (``PDFT_SNAPSHOT_EXACT=1``),
which only means something on the machine that produced the file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

# Snapshots are CPU results; a GPU agrees with them to rounding, not to the bit.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402

import pdft  # noqa: E402

from .cases import SNAPSHOT_PATH, all_records  # noqa: E402


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(SNAPSHOT_PATH))
    args = parser.parse_args()
    records = all_records()
    meta = {
        # the commit of the checkout this runs in, and the package that was
        # actually imported: they differ when PYTHONPATH points elsewhere
        "commit": _commit(),
        "source": str(Path(pdft.__file__).resolve().parent),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "device": jax.devices()[0].platform,
    }
    np.savez_compressed(args.out, __meta__=np.asarray(json.dumps(meta)), **records)
    print(f"wrote {len(records)} arrays to {args.out}: {meta}")


if __name__ == "__main__":
    main()
