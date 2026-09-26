"""The evaluation protocol every completion experiment scores with, defined once.

Numbers from different transform families are comparable only if they were
produced the same way: masks drawn per image from a stated seed, the
retained-coefficient rule ``k = max(m * frac, 64)``, PSNR on the clipped
reconstruction. A family plugs in as a closure ``solve(Y, obs, k)`` with ``Y``
the zero-filled observation, so this module knows nothing about how a
transform is parameterised. The constants are Table I of the completion paper
(DIV2K, ``512^2`` centre crops, ``p = 10%``).
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .metrics import ms_ssim, psnr, ssim

Array = jax.Array

TABLE1_P = 0.10
# Single precision for every jax transform in training and evaluation: the
# circuit is unitary, so float32 neither amplifies nor accumulates error across
# the gates, and at 512^2, K = 300, the PSNRs agree with float64 to 0.0003 dB at
# a tenth of the time. The angles stay float64; complex_dtype picks the working
# type from the image, so this constant is the one switch. Families that
# multiply matrices (transform learning, the butterfly blocks) lose up to
# 0.55 dB under XLA's default TF32 float32 matmuls and reproduce float64 under
# "highest", which is why every scoring loop runs inside MATMUL_EXACT.
EVAL_DTYPE = jnp.float32
MATMUL_EXACT = "highest"
# Wide enough that every method's optimum is interior at p = 10%.
TABLE1_FRACS = (0.015, 0.03, 0.0625, 0.125, 0.25, 0.5)
TABLE1_K = 300
QTT_RANKS = (48, 96, 200, 400)
QTT_ITERS = 512
NUC_RANKS = (2, 4, 6, 8, 12, 16, 32)
NUC_LAMS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)
# The rate sweep of the paper's Fig. 3; its p = 10% column is Table I verbatim.
RATE_SWEEP = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def heldout_mask(i: int, shape, p: float = TABLE1_P) -> np.ndarray:
    """Held-out image ``i``'s observation mask, seeded ``1000 + i``.

    Every method scored on image ``i`` sees the same pixels, the invariant the
    whole table rests on. A numpy bool array; wrap in ``jnp.asarray`` for the
    jax solvers.
    """
    return np.random.default_rng(1000 + i).random(shape) < p


def fracs_for(p: float) -> tuple[float, ...]:
    """The ``k/m`` budgets swept at sampling rate ``p``, wide enough that every optimum is interior."""
    if p < 0.10:
        return (0.0075,) + TABLE1_FRACS
    if p == 0.10:
        return TABLE1_FRACS
    return (0.00375, 0.0075, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 0.75)


def budget_k(obs, frac: float) -> int:
    """Retained coefficients for a mask: ``max(m * frac, 64)`` with ``m`` the observed count."""
    return max(int(int(np.asarray(obs).sum()) * frac), 64)


def train_k(n_pixels: int, p: float, frac: float) -> int:
    """``budget_k`` from the expected observed count, for training, where the mask is redrawn every step."""
    return max(int(p * n_pixels * frac), 64)


def evaluate(solve: Callable, images, p: float, frac: float, seed: int) -> np.ndarray:
    """PSNR per image at fixed masks: the same masks for every method at this seed, never one the optimiser saw."""
    rng = np.random.default_rng(seed)
    out = []
    for img in images:
        obs = jnp.asarray(rng.random(img.shape) < p)
        with jax.default_matmul_precision(MATMUL_EXACT):
            xh = solve(jnp.asarray(img, dtype=EVAL_DTYPE) * obs, obs, budget_k(obs, frac))
        out.append(psnr(np.asarray(xh), np.asarray(img)))
    return np.array(out)


def evaluate_params(
    reconstruct: Callable,
    params: dict,
    images,
    p: float,
    frac: float,
    K: int,
    seed: int,
    mode: str = "hard",
) -> np.ndarray:
    """``evaluate`` for any family's solver and its ``{"r", "c"}`` parameters."""
    return evaluate(
        lambda Y, obs, k: reconstruct(params["r"], params["c"], Y, obs, k, K, mode, remat=False),
        images,
        p,
        frac,
        seed,
    )


def per_metric_best(cands, img) -> dict:
    """Each metric read at the candidate that maximises that metric.

    Selecting by PSNR and reporting MS-SSIM would penalise whichever method is
    not tuned for PSNR; this is the read-off rule of Table I.
    """
    zs = [np.asarray(z) for z in cands]
    return {
        "psnr": max(psnr(z, img) for z in zs),
        "ssim": max(ssim(z, img) for z in zs),
        "msssim": max(ms_ssim(z, img) for z in zs),
    }


def table1_scores(solve: Callable, images, fracs=TABLE1_FRACS, p: float = TABLE1_P) -> dict:
    """Score one method under Table I: the ``rng(1000 + i)`` masks, the budget sweep, each metric at its own optimum.

    Returns ``{"psnr", "ssim", "msssim"}`` lists with one entry per image.
    """
    out = {"psnr": [], "ssim": [], "msssim": []}
    for i, img in enumerate(images):
        obs_np = heldout_mask(i, img.shape, p)
        obs = jnp.asarray(obs_np)
        Y = jnp.asarray(img, dtype=EVAL_DTYPE) * obs
        with jax.default_matmul_precision(MATMUL_EXACT):
            cands = [np.asarray(solve(Y, obs, budget_k(obs_np, f))) for f in fracs]
        for met, v in per_metric_best(cands, img).items():
            out[met].append(v)
    return out


def write_json(path, obj) -> pathlib.Path:
    """Write ``obj`` as JSON, creating the directory, and say where."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"wrote {path}")
    return path


def _floats(a) -> list:
    """An array as a JSON list, complex entries interleaved as (re, im)."""
    a = np.asarray(a)
    return (a.view(float) if np.iscomplexobj(a) else a).tolist()


def _complex(values, shape) -> Array:
    """The inverse of ``_floats`` for a complex array of the given shape."""
    return jnp.asarray(np.ascontiguousarray(np.array(values)).view(complex).reshape(shape))


def encode_gphi(par: dict) -> dict:
    """An ``{"r", "c"}`` pair of ``{"g", "phi"}`` gate dicts as JSON lists; ``decode_gphi`` inverts it."""
    return {a: {"g": _floats(par[a]["g"]), "phi": _floats(par[a]["phi"])} for a in ("r", "c")}


def decode_gphi(d: dict) -> dict:
    """The gate dicts back from ``encode_gphi``'s lists."""
    return {
        a: {"g": _complex(d[a]["g"], (-1, 2, 2)), "phi": jnp.asarray(np.array(d[a]["phi"]))}
        for a in ("r", "c")
    }


def encode_butterfly(par: dict) -> dict:
    """An ``{"r", "c"}`` pair of butterfly parameters as JSON lists: real generators as they are, complex blocks with their shape."""

    def one(d):
        if "gen" in d:
            return {"gen": _floats(d["gen"])}
        return {"blk": _floats(d["blk"]), "shape": list(np.shape(d["blk"]))}

    return {a: one(par[a]) for a in ("r", "c")}


def decode_butterfly(d: dict) -> dict:
    """The butterfly parameters back from ``encode_butterfly``'s lists."""
    return {
        a: {"gen": jnp.asarray(np.array(d[a]["gen"]))}
        if "gen" in d[a]
        else {"blk": _complex(d[a]["blk"], tuple(d[a]["shape"]))}
        for a in ("r", "c")
    }
