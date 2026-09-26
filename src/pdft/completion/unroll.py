"""Memory-bounded differentiation of the unrolled solver.

Evaluating the solver is cheap: ``jax.lax.scan`` keeps one carry alive, so
memory is ``O(N^2)`` and flat in the depth ``K``. Differentiating it is not.
The reverse pass needs each step's input to recompute that step, so a scan of
``K`` steps retains ``K`` carries, ``O(K N^2)``: at ``4096^2``, ``K = 100``,
float64, that is 12.5 GiB of carries before a single gate intermediate.

That is a rematerialisation schedule, not a property of the method. Splitting
the scan into ``n_outer`` blocks of ``n_inner`` steps and rematerialising the
blocks retains ``n_outer`` carries between blocks and ``n_inner`` inside
whichever block the backward pass is recomputing, so the peak is
``n_outer + n_inner``, minimised at ``sqrt(K)`` each: ``O(K N^2)`` becomes
``O(sqrt(K) N^2)`` for one extra forward evaluation per block (about 1.2x).
This is binomial checkpointing at one level, which is all that is needed.

Three schedules compute the same function: ``"none"`` (no rematerialisation,
correct only for evaluation), ``"step"`` (each step rematerialised, ``K``
carries retained) and ``"nested"`` (blocks rematerialised as well, ``sqrt(K)``
carries). ``solver_for`` builds the bounded solver of any operator and
``plan`` picks a schedule for a memory budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .solver import THRESHOLDS
from .transform import apply_gates, separable

Array = jax.Array

STRATEGIES = ("none", "step", "nested")

# Live intermediates during the backward pass, in units of one complex image.
# Calibrated against twelve measurements (n = 10, 11, 12; step and nested;
# float64 and float32; K = 100; one RTX 5070): within 12% on every case that
# ran, and it reproduces the fit-or-OOM verdict in all twelve. A constant fitted
# on one XLA version and one card, so plan() keeps a margin on top; the
# retained-carry term is exact and is the one that scales with K.
_WORKING_SET = 60
_PLAN_MARGIN = 1.15


def split_k(K: int, n_outer: int | None = None) -> tuple[int, int, int]:
    """``(n_outer, n_inner, remainder)`` of the nested schedule.

    Defaults to the square split, which minimises ``n_outer + n_inner``.
    Whatever does not divide evenly runs as a trailing flat scan.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if n_outer is None:
        n_outer = max(1, int(round(math.sqrt(K))))
    n_outer = int(min(max(1, n_outer), K))
    n_inner = K // n_outer
    return n_outer, n_inner, K - n_outer * n_inner


def carry_bytes(shape, dtype=jnp.float64) -> int:
    """The bytes of one solver carry of this shape and dtype."""
    return int(np.prod(shape)) * jnp.dtype(dtype).itemsize


def retained_carries(K: int, strategy: str, n_outer: int | None = None) -> int:
    """How many carries the backward pass holds: exact, and the term in ``K``."""
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
    """Approximate peak device bytes of one gradient evaluation.

    The retained term is exact and the working set a calibrated constant, so
    the total is an estimate and the ordering between schedules is reliable.
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
    """The cheapest schedule estimated to fit the budget, with the split to use.

    Prefers the flat schedule when it fits, since nesting costs an extra
    forward pass. Falls back to the deepest split rather than raising: a plan
    that overshoots is more useful than none, and the caller sees the estimate.
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


def _scan(step: Callable, X0: Array, K: int, strategy: str, n_outer: int | None) -> Array:
    """``K`` applications of ``step`` under the requested rematerialisation."""

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


def solver_for(apply: Callable) -> Callable:
    """The bounded-memory ``K``-step solver of a per-axis operator.

    Returns ``reconstruct(pr, pc, Y, obs, k, K, *, mode, strategy, n_outer,
    budget_bytes)``. ``strategy="auto"`` reads the device's free memory and
    plans against 80% of it; a schedule by name pins it. The dtype of ``Y``
    decides the precision. Not jitted, so a Python-int ``k`` reaches
    ``top_k``; jit the loss that calls it.
    """
    analysis, synthesis = separable(apply)

    def reconstruct(
        pr,
        pc,
        Y: Array,
        obs: Array,
        k,
        K: int,
        *,
        mode: str = "hard",
        strategy: str = "auto",
        n_outer: int | None = None,
        budget_bytes: int | None = None,
    ) -> Array:
        """``K`` solver steps at bounded memory; see ``solver_for``."""
        if strategy == "auto":
            budget = budget_bytes if budget_bytes is not None else _free_bytes()
            chosen = plan(Y.shape[-2:], K, int(0.8 * budget), Y.dtype)
            strategy, n_outer = chosen["strategy"], chosen["n_outer"]
        thresh = THRESHOLDS[mode]
        X0 = jnp.where(obs, Y, 0.0)

        def step(X):
            return jnp.where(obs, Y, jnp.real(synthesis(thresh(analysis(X, pr, pc), k), pr, pc)))

        return _scan(step, X0, K, strategy, n_outer)

    return reconstruct


reconstruct = solver_for(apply_gates)


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
    """A one-line-per-schedule table of estimated peaks, for logs and for choosing a split."""

    def gb(b):
        return b / 2**30

    out = [
        f"{shape[0]}x{shape[1]}, K={K}, {jnp.dtype(dtype).name}",
        f"{'schedule':>10} {'carries':>8} {'retained':>11} {'total (est)':>13}",
    ]
    for s in STRATEGIES:
        e = estimate_peak(shape, K, s, dtype)
        out.append(
            f"{s:>10} {e['carries']:>8} {gb(e['retained_bytes']):>10.2f}G {gb(e['total_bytes']):>12.2f}G"
        )
    return "\n".join(out)
