"""Golden for tests/parity/test_completion.py, computed by the completion paper's own code.

Image completion is not in ParametricDFT.jl, so its reference is the code of
the completion paper (the `pdft-completion` repository): its solver, its
Model B (Hadamards fixed, the four phases of every controlled-phase gate free)
and its training loop, plain Adam with a fresh batch and mask per step.

That repository's package is also named ``pdft``, so run this with its source
on the path and this package off it:

    cd <pdft-completion checkout>
    PYTHONPATH=src python <pdft checkout>/reference/completion/generate_golden.py \
        <pdft checkout>/reference/completion/golden.npz

The images are synthetic and stored as uint8, so both sides read exactly the
same numbers.

The settings below were picked so that no threshold in the run comes near a
tie: the smallest relative gap between the ``k``-th and the next magnitude,
over every solver step of every training step, is 3.7e-4. That is what makes
the run a reference. The paper's threshold keeps every magnitude that reaches
the ``k``-th, so ``k + 1`` of them when two are equal, and which it does when
they are equal up to rounding is decided by rounding; this package keeps
exactly ``k`` and settles a tie by position. The two agree away from ties and
only there. Ties are common: at the Fourier point the coefficients of a real
image come in conjugate pairs, so an even ``k`` cuts one, and at this size an
odd ``k`` often does too. Check a change of settings the same way before
trusting it (the paper's code on a CPU and on a GPU must agree to rounding).
"""

import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from pdft.families import general  # the paper repository's package, not this one

N_QUBITS = 4
SIZE = 2**N_QUBITS
TRAIN = {"k": 15, "K": 8, "p": 0.30, "steps": 20, "lr": 1e-2, "batch": 2, "seed": 2}
SOLVE = {"k": 21, "K": 30}


def synthetic_images(count: int, seed: int) -> np.ndarray:
    """Smooth images with an edge, as uint8."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    out = []
    for _ in range(count):
        a, b, c, d, e, f = rng.uniform(0, 2, 6)
        img = 0.5 + 0.2 * np.cos(2 * np.pi * (a * xx + b * yy) + c)
        img += 0.1 * np.cos(2 * np.pi * (2 * xx - yy) * d + e)
        img += 0.15 * (xx + f * yy > 0.9)
        out.append(np.clip(np.rint(img * 255), 0, 255).astype(np.uint8))
    return np.stack(out)


def main(out_path: str) -> None:
    repo = pathlib.Path(general.__file__).resolve().parents[3]
    git = ["git", "-C", str(repo)]
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    if dirty.stdout.strip():
        raise SystemExit(f"{repo} has uncommitted changes; a golden must come from a commit")

    images_u8 = synthetic_images(7, seed=2026)
    images = images_u8.astype(np.float64) / 255.0
    train, test = images[:6], images[6]

    params, history = general.train_general(
        train, N_QUBITS, TRAIN["k"], K=TRAIN["K"], p=TRAIN["p"], steps=TRAIN["steps"],
        lr=TRAIN["lr"], model="B", batch=TRAIN["batch"], seed=TRAIN["seed"], verbose=False,
    )  # fmt: skip

    mask = np.random.default_rng(1000).random(test.shape) < TRAIN["p"]
    observed, obs = jnp.asarray(test * mask), jnp.asarray(mask)
    start = general.init_general(N_QUBITS)

    def solve(par_r, par_c):
        solved = general.reconstruct_g(
            par_r, par_c, observed, obs, N_QUBITS, SOLVE["k"], SOLVE["K"]
        )
        return np.asarray(solved)

    np.savez_compressed(
        out_path,
        images=images_u8,
        loss_history=np.array([h["loss"] for h in history]),
        trained_phases=np.stack([np.asarray(params[a]["phi"]) for a in ("r", "c")]),
        mask=mask,
        solved_at_the_start=solve(start, start),
        solved_after_training=solve(params["r"], params["c"]),
        train_k=TRAIN["k"], train_solver_steps=TRAIN["K"], rate=TRAIN["p"], steps=TRAIN["steps"],
        lr=TRAIN["lr"], batch_size=TRAIN["batch"], seed=TRAIN["seed"],
        solve_k=SOLVE["k"], solve_steps=SOLVE["K"],
        paper_commit=commit.stdout.strip(),
    )  # fmt: skip
    print(f"wrote {out_path} from {commit.stdout.strip()[:7]} on {jax.devices()[0]}")


if __name__ == "__main__":
    main(sys.argv[1])
