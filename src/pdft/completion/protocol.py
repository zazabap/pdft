"""The evaluation protocol every completion experiment scores with --- one definition.

Numbers from different transform families are only comparable if they were
produced the same way: masks drawn per image from a stated seed, the retained-
coefficient rule k = max(m * k/m, 64), PSNR on the clipped reconstruction.

A family plugs in as a closure ``solve(Y, obs, k) -> reconstruction``, where Y
is the zero-filled observation, so this module needs to know nothing about how
any transform is parameterised. The constants are the published Table I
protocol of the completion paper (DIV2K, 512^2 centre crops, p = 10%).
"""

from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from .metrics import ms_ssim, psnr, ssim

# The published Table I protocol: sampling rate, the budget sweep each
# transform is read over, the solver depth, and the capacity grids of the
# per-image baselines.
TABLE1_P = 0.10
# Single precision for every jax transform, training and evaluation: the
# circuit is unitary, so float32 neither amplifies nor accumulates error
# across the gates, and at 512^2, K = 300, the PSNRs agree with float64 to
# 0.0003 dB while a recovery costs a tenth of the time. The angles themselves
# stay float64; complex_dtype picks the working type from the image's dtype,
# so this constant is the one switch. The circuit has no matmul; the solvers
# that do (transform learning's dense pair, the butterfly's block products)
# lose up to 0.55 dB under XLA's default TF32 float32 matmuls and reproduce
# float64 exactly under "highest", which is why every scoring loop runs inside
# MATMUL_EXACT.
EVAL_DTYPE = jnp.float32
MATMUL_EXACT = "highest"
# Wide enough that every method's optimum is interior at p = 10%.
TABLE1_FRACS = (0.015, 0.03, 0.0625, 0.125, 0.25, 0.5)
TABLE1_K = 300
QTT_RANKS = (48, 96, 200, 400)
# The rate sweep of the paper's Fig. 3: the rates, and the budget grid each
# rate sweeps k/m over. The p = 10% grid is Table I's verbatim.
RATE_SWEEP = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
QTT_ITERS = 512
NUC_RANKS = (2, 4, 6, 8, 12, 16, 32)
NUC_LAMS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)


def heldout_mask(i: int, shape, p: float = TABLE1_P) -> np.ndarray:
    """Held-out image i's observation mask under the Table I protocol.

    Seeded 1000+i, so every method scored on image i sees the same pixels ---
    the invariant the whole table rests on. Returns a numpy bool array; wrap in
    jnp.asarray for the jax solvers.
    """
    return np.random.default_rng(1000 + i).random(shape) < p


def fracs_for(p: float):
    """The k/m budgets swept at sampling rate p in the rate figure, wide
    enough that every row's optimum is interior."""
    if p < 0.10:
        return (0.0075,) + TABLE1_FRACS
    if p == 0.10:
        return TABLE1_FRACS  # Table I's sweep, verbatim
    return (0.00375, 0.0075, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 0.75)


def budget_k(obs, frac: float) -> int:
    """Retained coefficients for a mask: k = max(m * frac, 64), m observed."""
    return max(int(int(np.asarray(obs).sum()) * frac), 64)


def train_k(n_pixels: int, p: float, frac: float) -> int:
    """budget_k's training-time counterpart, from the expected observed count.

    Training redraws the mask every step, so k is pinned to p * n_pixels rather
    than to one realisation of Omega; the floor matches budget_k's.
    """
    return max(int(p * n_pixels * frac), 64)


def evaluate(solve, images, p: float, frac: float, seed: int) -> np.ndarray:
    """PSNR per image at fixed masks --- the same masks for every method
    compared at this seed, and never a mask the optimiser saw."""
    rng = np.random.default_rng(seed)
    out = []
    for img in images:
        obs = jnp.asarray(rng.random(img.shape) < p)
        with jax.default_matmul_precision(MATMUL_EXACT):
            xh = solve(jnp.asarray(img, dtype=EVAL_DTYPE) * obs, obs, budget_k(obs, frac))
        out.append(psnr(np.asarray(xh), np.asarray(img)))
    return np.array(out)


def per_metric_best(cands, img) -> dict:
    """Each metric read at the candidate that maximises *that* metric.

    Selecting by PSNR and then reporting MS-SSIM systematically penalises
    whichever method is not tuned for PSNR; this is the published read-off rule
    of Table I.
    """
    zs = [np.asarray(z) for z in cands]
    return {
        "psnr": max(psnr(z, img) for z in zs),
        "ssim": max(ssim(z, img) for z in zs),
        "msssim": max(ms_ssim(z, img) for z in zs),
    }


def write_json(path, obj) -> pathlib.Path:
    """``<path>.json``, creating the directory; announces what it wrote."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"wrote {path}")
    return path


def encode_gphi(par) -> dict:
    """A {"r", "c"} pair of {"g", "phi"} params as JSON-serialisable lists.

    The complex g blocks are stored as interleaved floats; decode_gphi is the
    exact inverse.
    """
    return {
        a: {
            "g": np.asarray(par[a]["g"]).view(float).tolist(),
            "phi": np.asarray(par[a]["phi"]).tolist(),
        }
        for a in ("r", "c")
    }


def decode_gphi(d) -> dict:
    """The inverse of encode_gphi, in the form pdft.completion.families.general consumes."""
    par = {}
    for a in ("r", "c"):
        g = np.ascontiguousarray(np.array(d[a]["g"])).view(complex).reshape(-1, 2, 2)
        par[a] = {"g": jnp.asarray(g), "phi": jnp.asarray(np.array(d[a]["phi"]))}
    return par


def encode_butterfly(par) -> dict:
    """A {"r", "c"} pair of butterfly parameters as JSON-serialisable lists.

    Same contract as encode_gphi, for the other trained FFT factorisation: the
    real generators of the unitary mode go out as they are, the complex blocks
    of the free mode as interleaved floats.
    """

    def one(d):
        if "gen" in d:
            return {"gen": np.asarray(d["gen"]).tolist()}
        b = np.asarray(d["blk"])
        return {"blk": b.view(float).tolist(), "shape": list(b.shape)}

    return {a: one(par[a]) for a in ("r", "c")}


def decode_butterfly(d) -> dict:
    """The inverse of encode_butterfly, in the form pdft.completion.families.butterfly consumes."""
    par = {}
    for a in ("r", "c"):
        if "gen" in d[a]:
            par[a] = {"gen": jnp.asarray(np.array(d[a]["gen"]))}
        else:
            b = np.ascontiguousarray(np.array(d[a]["blk"])).view(complex)
            par[a] = {"blk": jnp.asarray(b.reshape(tuple(d[a]["shape"])))}
    return par


def table1_scores(solve, images, fracs=TABLE1_FRACS, p: float = TABLE1_P) -> dict:
    """Score one method under Table I's protocol: the rng(1000+i) masks, the
    budget sweep over ``fracs``, each metric at its own optimum per image.

    ``solve(Y, obs, k)`` is the family's closure (Y zero-filled, obs a jnp bool
    mask, k the retained count). Returns {"psnr", "ssim", "msssim"} lists, one
    entry per image.
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
