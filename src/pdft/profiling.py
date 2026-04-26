"""Per-step profiling for `train_basis_batched`.

Two outputs:
  1) Python-side per-step wall-clock CSV (no JIT distortion — uses
     `jax.block_until_ready` after each step). Lets you see step-to-step
     variance and identify outliers (recompiles, val passes).
  2) Optional JAX/XLA HLO trace dumped to `trace_dir` for TensorBoard
     viewing. Annotated with `StepTraceAnnotation` so each step shows
     up as a discrete unit.

Usage
-----

    from pdft.profiling import profile_training

    report = profile_training(
        basis=basis,
        dataset=imgs,
        loss=pdft.MSELoss(k=...),
        n_steps=20,
        batch_size=4,
        trace_dir="profile_out",  # optional — TensorBoard trace
    )
    print(report.summary())
    report.to_csv("step_times.csv")

This is a profiling utility — not used by the main training path. It
re-implements the inner Adam loop (calling the same JIT'd `step_fn`) so
that per-step block_until_ready can be inserted without disturbing
`train_basis_batched`'s production behavior.
"""

from __future__ import annotations

import csv
import statistics
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .loss import AbstractLoss
from .manifolds import group_by_manifold
from .training import _build_jit_adam_step, _cosine_with_warmup


@dataclass
class StepRecord:
    step: int
    phase: str  # "compile" (first step JIT) | "warm" (post-JIT) | "val"
    wall_s: float
    loss: float | None = None


@dataclass
class ProfileReport:
    basis_class: str
    m: int
    n: int
    batch_size: int
    n_steps: int
    device: str
    records: list[StepRecord] = field(default_factory=list)
    setup_s: float = 0.0  # construction + first-step compile (approx)
    total_s: float = 0.0
    peak_gb: float = 0.0
    trace_dir: str | None = None

    def summary(self) -> str:
        warm = [r.wall_s for r in self.records if r.phase == "warm"]
        compile_s = sum(r.wall_s for r in self.records if r.phase == "compile")
        val = [r.wall_s for r in self.records if r.phase == "val"]
        if not warm:
            return f"<no warm steps recorded for {self.basis_class}>"
        med = statistics.median(warm)
        p10 = sorted(warm)[max(0, int(0.1 * len(warm)))]
        p90 = sorted(warm)[min(len(warm) - 1, int(0.9 * len(warm)))]
        lines = [
            f"=== {self.basis_class} m={self.m} n={self.n} bs={self.batch_size} on {self.device} ===",
            f"  total wall:      {self.total_s:.2f} s ({self.n_steps} steps)",
            f"  setup+compile:   {self.setup_s + compile_s:.2f} s",
            f"  per-step (warm): median={med * 1000:.1f} ms  p10={p10 * 1000:.1f} ms  p90={p90 * 1000:.1f} ms",
            f"  warm steps:      {len(warm)} / {self.n_steps}",
        ]
        if val:
            lines.append(
                f"  val passes:      {len(val)} × ~{statistics.mean(val) * 1000:.0f} ms each"
            )
        if self.peak_gb:
            lines.append(f"  peak GPU:        {self.peak_gb:.2f} GB")
        if self.trace_dir:
            lines.append(
                f"  trace:           {self.trace_dir} (open with `tensorboard --logdir {self.trace_dir}`)"
            )
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "phase", "wall_s", "loss"])
            for r in self.records:
                w.writerow(
                    [r.step, r.phase, f"{r.wall_s:.6f}", "" if r.loss is None else f"{r.loss:.6e}"]
                )


@contextmanager
def _maybe_trace(trace_dir: str | Path | None):
    """Optionally wrap a region in `jax.profiler.trace()`.

    NOTE: On some JAX/CUDA combinations (observed on JAX 0.10 + CUDA 12.9
    with RTX 3090) `jax.profiler.trace()` triggers a CUDA launch failure
    on the first step. If you hit `CUDA_ERROR_LAUNCH_FAILED`, leave
    `trace_dir=None` — Python-side per-step timing still works and is
    usually sufficient to find the bottleneck.
    """
    if trace_dir is None:
        with nullcontext():
            yield None
    else:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(trace_dir)):
            yield trace_dir


def profile_training(
    basis: Any,
    *,
    dataset: list,
    loss: AbstractLoss,
    n_steps: int,
    batch_size: int = 4,
    optimizer: str = "adam",
    lr_peak: float = 0.003,
    lr_final: float = 0.0003,
    warmup_frac: float = 0.05,
    max_grad_norm: float | None = 1.0,
    val_every: int = 0,  # 0 = no val passes; >0 = run val_eval every K steps
    val_dataset: list | None = None,
    trace_dir: str | Path | None = None,
    seed: int = 42,
    device: jax.Device | None = None,
) -> ProfileReport:
    """Profile `n_steps` of `_build_jit_adam_step` end-to-end on `dataset`.

    Pads / cycles the dataset to fill exactly `n_steps` batches of
    `batch_size` images each. Inserts `jax.block_until_ready` after every
    step so per-step wall-clock is honest (no async dispatch overlap).

    Returns a `ProfileReport` with per-step records and optional XLA
    HLO trace at `trace_dir` (open with `tensorboard --logdir <dir>`).

    The first step's wall-clock is dominated by JIT compile and tagged
    "compile"; subsequent steps are tagged "warm". Val passes (when
    `val_every > 0`) are tagged "val".
    """
    if optimizer.lower() != "adam":
        raise NotImplementedError("profile_training currently supports optimizer='adam' only")
    if not dataset:
        raise ValueError("dataset must be non-empty")

    device = device or jax.devices()[0]

    # Materialize dataset on device.
    with jax.default_device(device):
        imgs = [jax.device_put(np.asarray(img).astype(np.complex128), device) for img in dataset]
        val_imgs = None
        if val_every > 0 and val_dataset:
            val_imgs = jnp.stack(
                [
                    jax.device_put(np.asarray(img).astype(np.complex128), device)
                    for img in val_dataset
                ],
                axis=0,
            )

        # Pad/cycle dataset to fill n_steps * batch_size.
        needed = n_steps * batch_size
        if needed > len(imgs):
            reps = (needed + len(imgs) - 1) // len(imgs)
            imgs = (imgs * reps)[:needed]
        else:
            imgs = imgs[:needed]

        # Build the JIT'd Adam step (same one train_basis_batched uses).
        step_fn = _build_jit_adam_step(
            basis,
            loss,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            max_grad_norm=max_grad_norm,
        )
        # Init Adam moments.
        groups = group_by_manifold(list(basis.tensors))
        m_state, v_state = [], []
        from .manifolds import stack_tensors

        for _, idxs in groups.items():
            pb = stack_tensors(list(basis.tensors), list(idxs))
            m_state.append(jnp.zeros_like(pb))
            v_state.append(jnp.zeros(pb.shape, dtype=jnp.float64))

        # Optional val closure mirrors training.py:402.
        from .loss import loss_function as _lf

        m_qb, n_qb = basis.m, basis.n
        code, inv_code = basis.code, basis.inv_code

        def _per_image_loss(tensors, img):
            return _lf(tensors, m_qb, n_qb, code, img, loss, inverse_code=inv_code)

        _batched_val = jax.vmap(_per_image_loss, in_axes=(None, 0))
        _val_eval = (
            jax.jit(lambda ts, b: jnp.mean(_batched_val(ts, b))) if val_imgs is not None else None
        )

        report = ProfileReport(
            basis_class=type(basis).__name__,
            m=m_qb,
            n=n_qb,
            batch_size=batch_size,
            n_steps=n_steps,
            device=str(device),
            trace_dir=str(trace_dir) if trace_dir else None,
        )
        current = [jnp.asarray(t) for t in basis.tensors]
        t_setup0 = time.perf_counter()
        # Touch the moments / tensors once so allocation is realised.
        jax.block_until_ready(current[0])
        report.setup_s = time.perf_counter() - t_setup0

        t_total0 = time.perf_counter()
        with _maybe_trace(trace_dir):
            for s in range(n_steps):
                lr_t = _cosine_with_warmup(
                    s + 1,
                    n_steps,
                    warmup_frac=warmup_frac,
                    lr_peak=lr_peak,
                    lr_final=lr_final,
                )
                batch = jnp.stack(imgs[s * batch_size : (s + 1) * batch_size], axis=0)

                with jax.profiler.StepTraceAnnotation("train_step", step_num=s):
                    t0 = time.perf_counter()
                    current, m_state, v_state, loss_val = step_fn(
                        current,
                        m_state,
                        v_state,
                        batch,
                        jnp.asarray(lr_t),
                        jnp.asarray(s + 1, dtype=jnp.int32),
                    )
                    jax.block_until_ready(loss_val)
                    dt = time.perf_counter() - t0

                phase = "compile" if s == 0 else "warm"
                report.records.append(
                    StepRecord(step=s, phase=phase, wall_s=dt, loss=float(loss_val))
                )

                if val_every > 0 and val_imgs is not None and (s + 1) % val_every == 0:
                    with jax.profiler.StepTraceAnnotation("val_eval", step_num=s):
                        t0 = time.perf_counter()
                        v = _val_eval(current, val_imgs)
                        jax.block_until_ready(v)
                        dt = time.perf_counter() - t0
                    report.records.append(StepRecord(step=s, phase="val", wall_s=dt, loss=float(v)))

        report.total_s = time.perf_counter() - t_total0

        # Peak memory snapshot (best-effort; backend-dependent).
        try:
            stats = device.memory_stats() or {}
            report.peak_gb = (stats.get("peak_bytes_in_use") or 0) / (1024**3)
        except Exception:
            pass

    return report


__all__ = ["profile_training", "ProfileReport", "StepRecord"]
