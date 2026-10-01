"""Run `workload.py` on the source of a git ref and on the working tree; compare bit for bit.

    python -m tests.characterisation.compare_refs origin/main

The ref's `src/` is extracted with `git archive` into a temporary directory.
Each side runs in its own process, on the CPU, with the compilation cache off,
so neither can see the other's modules or compiled code. Every recorded array
must have the same shape, dtype and bytes on both sides; the ones that do not
are listed with the size of the difference, and the exit status is 1.

Use it for a change that is meant to alter no behaviour. It only means
something on one machine, since both sides are computed there.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = Path(__file__).with_name("workload.py")


def _run(src: Path, target: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(src), JAX_PLATFORMS="cpu", PDFT_DISABLE_COMPILE_CACHE="1")
    done = subprocess.run(
        [sys.executable, str(WORKLOAD), str(target)], env=env, capture_output=True, text=True
    )
    if done.returncode != 0:
        raise SystemExit(f"the workload failed on {src}:\n{done.stderr[-3000:]}")


def differences(old: np.lib.npyio.NpzFile, new: np.lib.npyio.NpzFile) -> list[tuple[str, str]]:
    """``(key, what differs)`` for every array that is not the same on both sides."""
    found = [(key, "only on one side") for key in sorted(set(old.files) ^ set(new.files))]
    for key in sorted(set(old.files) & set(new.files)):
        a, b = old[key], new[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            found.append((key, f"{a.dtype} {a.shape} against {b.dtype} {b.shape}"))
        elif a.dtype.kind in "US":
            if not np.array_equal(a, b):
                found.append((key, f"{str(a)[:60]!r} against {str(b)[:60]!r}"))
        elif a.tobytes() != b.tobytes():
            found.append((key, f"max abs difference {np.nanmax(np.abs(a - b)):.3e}"))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref", help="the git ref whose src/ is the reference, e.g. origin/main")
    parser.add_argument("--show", type=int, default=40, help="how many differences to list")
    args = parser.parse_args()

    archive = subprocess.run(
        ["git", "archive", args.ref, "src"], cwd=REPO, capture_output=True, check=True
    ).stdout
    with tempfile.TemporaryDirectory() as scratch:
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(scratch, filter="data")
        records = {}
        for label, src in ((args.ref, Path(scratch) / "src"), ("working tree", REPO / "src")):
            target = Path(scratch) / f"{len(records)}.npz"
            _run(src, target)
            records[label] = np.load(target)
            print(f"{label}: {len(records[label].files)} arrays")
        found = differences(*records.values())

    if not found:
        print("identical: every array has the same shape, dtype and bytes")
        return
    print(f"{len(found)} arrays differ")
    for key, what in found[: args.show]:
        print(f"    {key}: {what}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
