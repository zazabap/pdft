"""Memory-bounded differentiation of the unrolled solver.

Evaluating the solver is cheap: ``jax.lax.scan`` keeps one carry alive, so
memory is O(N^2) and flat in the depth K. Differentiating it is not. The
reverse pass needs each step's input to recompute that step, so a scan of K
steps retains K carries --- O(K N^2) --- and at 4096^2, K = 100, float64 that
is 12.5 GiB of carries before a single gate intermediate.

That is a rematerialisation schedule, not a property of the method, and this
module fixes it. Splitting the scan into n_outer blocks of n_inner steps and
rematerialising the *blocks* retains n_outer carries between blocks and n_inner
inside whichever block the backward pass is currently recomputing:

    peak carries = n_outer + n_inner,   minimised at n_outer = n_inner = sqrt(K)

so O(K N^2) becomes O(sqrt(K) N^2). The cost is one extra forward evaluation
per block, about 1.2x the flat schedule. This is the classical binomial-
checkpointing trade at one level, which is all that is needed.

Three schedules, all computing the same function:

    "none"      no rematerialisation. Correct only for evaluation, where
                nothing is retained anyway; the fastest choice there.
    "step"      rematerialise each step. Kills the ~300 gate intermediates per
                step but still retains K carries.
    "nested"    rematerialise blocks of steps as well. Retains sqrt(K) carries.

``reconstruct`` accepts square or rectangular registers and either precision,
and ``plan`` picks a schedule for a memory budget.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .families.general import analysis_rect, synthesis_rect
from .solver import _THRESH

STRATEGIES = ("none", "step", "nested")

# Live intermediates during the backward pass, in units of one complex image.
# Calibrated against twelve measurements (n = 10, 11, 12 x {step, nested} x
# {float64, float32}, K = 100, RTX 5070): within 12% on every case that ran,
# and it reproduces the fit/OOM verdict in all twelve. It is a constant fitted
# on one XLA version and one card, so plan() keeps a margin on top; the
# retained-carry term is exact and is the one that scales with K.
_WORKING_SET = 60
_PLAN_MARGIN = 1.15


# --------------------------------------------------------------------------
# schedule arithmetic


def split_k(K: int, n_outer: int | None = None) -> tuple[int, int, int]:
    """(n_outer, n_inner, remainder) for the nested schedule.

    Defaults to the square split, which minimises n_outer + n_inner. K need not
    be a perfect square or even composite: whatever does not divide evenly is
    run as a trailing flat scan.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if n_outer is None:
        n_outer = max(1, int(round(math.sqrt(K))))
    n_outer = int(min(max(1, n_outer), K))
    n_inner = K // n_outer
    return n_outer, n_inner, K - n_outer * n_inner


def carry_bytes(shape, dtype=jnp.float64) -> int:
    return int(np.prod(shape)) * jnp.dtype(dtype).itemsize


def retained_carries(K: int, strategy: str, n_outer: int | None = None) -> int:
    """How many carries the backward pass holds. Exact, and the term in K."""
    if strategy == "none":
        return 1
    if strategy == "step":
        return K
    if strategy == "nested":
        no, ni, rem = split_k(K, n_outer)
        return no + ni + (1 if rem else 0)
    raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")


def estimate_peak(
    shape, K: int, strategy: str = "nested", dtype=jnp.float64, n_outer: int | None = None
) -> dict:
    """Approximate peak device bytes for one gradient evaluation.

    The retained term is exact; the working set is a calibrated constant, so
    treat the total as an estimate and the ordering as reliable.
    """
    real = jnp.dtype(dtype)
    cplx = jnp.complex64 if real == jnp.dtype(jnp.float32) else jnp.complex128
    held = retained_carries(K, strategy, n_outer) * carry_bytes(shape, real)
    work = _WORKING_SET * carry_bytes(shape, cplx)
    return {
        "retained_bytes": held,
        "working_bytes": work,
        "total_bytes": held + work,
        "carries": retained_carries(K, strategy, n_outer),
    }


def plan(shape, K: int, budget_bytes: int, dtype=jnp.float64) -> dict:
    """Cheapest schedule that is estimated to fit, with the split to use.

    Prefers the flat schedule when it fits, since nesting costs an extra forward
    pass. Falls back to the deepest split available rather than raising: a plan
    that overshoots the budget is more useful than no plan, and the caller can
    see the estimate.
    """
    limit = budget_bytes / _PLAN_MARGIN
    flat = estimate_peak(shape, K, "step", dtype)
    if flat["total_bytes"] <= limit:
        return {"strategy": "step", "n_outer": None, **flat, "fits": True}
    best = None
    for no in range(1, K + 1):
        est = estimate_peak(shape, K, "nested", dtype, no)
        if best is None or est["total_bytes"] < best[1]["total_bytes"]:
            best = (no, est)
    no, est = best
    return {"strategy": "nested", "n_outer": no, **est, "fits": est["total_bytes"] <= limit}


# --------------------------------------------------------------------------
# the solver


def _scan(step, X0, K: int, strategy: str, n_outer: int | None):
    """K applications of ``step``, under the requested rematerialisation."""

    def body(c, _):
        return step(c), None

    if strategy == "none":
        return jax.lax.scan(body, X0, None, length=K)[0]

    inner = jax.checkpoint(body)  # drop the per-step gate intermediates
    if strategy == "step":
        return jax.lax.scan(inner, X0, None, length=K)[0]
    if strategy != "nested":
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

    no, ni, rem = split_k(K, n_outer)

    def outer_body(c, _):
        return jax.lax.scan(inner, c, None, length=ni)[0], None

    X = jax.lax.scan(jax.checkpoint(outer_body), X0, None, length=no)[0]
    if rem:
        X = jax.lax.scan(inner, X, None, length=rem)[0]
    return X


def reconstruct(
    pr,
    pc,
    Y,
    obs,
    k: int,
    K: int,
    *,
    nr: int | None = None,
    nc: int | None = None,
    mode: str = "hard",
    strategy: str = "auto",
    n_outer: int | None = None,
    budget_bytes: int | None = None,
):
    """The unrolled solver on a 2^nr x 2^nc image, differentiable at bounded memory.

    ``pr``, ``pc`` are the ``{"g", "phi"}`` parameter dicts of
    :mod:`pdft.completion.families.general`; distance-shared phases expand to
    that form through :func:`pdft.completion.families.shared.expand`, and
    phase-only angles ``theta`` are ``init_general(n)`` with
    ``phi[:, 3]`` replaced by ``theta``. nr, nc default to
    log2 of Y's trailing two axes. strategy="auto" reads the device's free
    memory and plans against 80% of it; pass a schedule by name to pin it. The
    dtype of Y decides the precision throughout.
    """
    if nr is None:
        nr = int(round(math.log2(Y.shape[-2])))
    if nc is None:
        nc = int(round(math.log2(Y.shape[-1])))
    for name, val, size in (("nr", nr, Y.shape[-2]), ("nc", nc, Y.shape[-1])):
        if 2**val != size:
            raise ValueError(f"axis of length {size} is not a power of two ({name})")

    if strategy == "auto":
        budget = budget_bytes if budget_bytes is not None else _free_bytes()
        chosen = plan(Y.shape[-2:], K, int(0.8 * budget), Y.dtype)
        strategy, n_outer = chosen["strategy"], chosen["n_outer"]

    thresh = _THRESH[mode]
    X0 = jnp.where(obs, Y, 0.0)

    def step(X):
        C = thresh(analysis_rect(X, pr, pc, nr, nc), k)
        return jnp.where(obs, Y, jnp.real(synthesis_rect(C, pr, pc, nr, nc)))

    return _scan(step, X0, K, strategy, n_outer)


def _free_bytes(default: int = 8 * 2**30) -> int:
    try:
        st = jax.local_devices()[0].memory_stats()
        return int(st["bytes_limit"] - st["bytes_in_use"])
    except Exception:
        return default


def device_peak_mb() -> float:
    """Peak device bytes in use, in MB; NaN off-GPU."""
    try:
        return jax.local_devices()[0].memory_stats()["peak_bytes_in_use"] / 2**20
    except Exception:
        return float("nan")


def report(shape, K: int, dtype=jnp.float64) -> str:
    """One-line-per-schedule table, for logs and for deciding a split."""

    def gb(b):
        return b / 2**30

    out = [
        f"{shape[0]}x{shape[1]}, K={K}, {jnp.dtype(dtype).name}",
        f"{'schedule':>10} {'carries':>8} {'retained':>11} {'total (est)':>13}",
    ]
    for s in STRATEGIES:
        e = estimate_peak(shape, K, s, dtype)
        out.append(
            f"{s:>10} {e['carries']:>8} {gb(e['retained_bytes']):>10.2f}G "
            f"{gb(e['total_bytes']):>12.2f}G"
        )
    return "\n".join(out)
