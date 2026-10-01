"""The ``optimize`` loop around the two optimisers: what it records and when it stops."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.optimizers import RiemannianAdam, RiemannianGD, optimize
from pdft.optimizers.core import _common_setup, _write_back

from ..helpers import complex_image


def _problem(loss=None):
    basis = pdft.QFTBasis(m=2, n=2)
    target = jnp.asarray(np.random.default_rng(0).normal(size=(4, 4)))
    loss = loss or pdft.L1Norm()

    def loss_fn(tensors):
        return pdft.loss_function(
            tensors, 2, 2, basis.code, target, loss, inverse_code=basis.inv_code
        )

    return list(basis.tensors), loss_fn, jax.grad(loss_fn)


@pytest.mark.parametrize("opt", [RiemannianGD(lr=0.01), RiemannianAdam(lr=0.01)])
def test_the_trace_is_the_loss_at_every_point_visited(opt):
    tensors, loss_fn, grad_fn = _problem()
    final, trace = optimize(opt, tensors, loss_fn, grad_fn, max_iter=4, tol=0.0, record_loss=True)
    assert len(trace) == 5 and trace[0] == float(loss_fn(tensors))
    assert trace[-1] == float(loss_fn(final))
    # one more step from the same start reproduces the prefix
    _, shorter = optimize(opt, tensors, loss_fn, grad_fn, max_iter=3, tol=0.0, record_loss=True)
    assert shorter == trace[:4]
    assert optimize(opt, tensors, loss_fn, grad_fn, max_iter=2)[1] == []


def test_an_exhausted_line_search_takes_its_last_candidate_and_rereads_the_loss():
    """As upstream: when no step satisfies the Armijo condition, the smallest one tried
    is taken anyway. The line search has no loss to hand back for it, so the trace
    re-reads the loss at the new point instead of recording a NaN."""
    tensors, loss_fn, grad_fn = _problem()
    stuck = RiemannianGD(lr=1e6, max_ls_steps=1)
    final, trace = optimize(stuck, tensors, loss_fn, grad_fn, max_iter=3, tol=0.0, record_loss=True)
    assert len(trace) == 4 and all(math.isfinite(value) for value in trace)
    assert trace[1] > trace[0]
    assert trace[-1] == float(loss_fn(final))
    assert not all(jnp.array_equal(a, b) for a, b in zip(final, tensors))


def test_a_non_finite_gradient_stops_the_loop_with_a_warning():
    tensors, loss_fn, _ = _problem()

    def bad_grad(ts):
        return [jnp.full_like(t, jnp.nan) for t in ts]

    with pytest.warns(UserWarning, match="Non-finite gradient"):
        final, trace = optimize(
            RiemannianAdam(lr=0.01), tensors, loss_fn, bad_grad, max_iter=3, record_loss=True
        )
    assert len(trace) == 1 and all(jnp.array_equal(a, b) for a, b in zip(final, tensors))


def test_the_loop_stops_below_the_gradient_tolerance():
    tensors, loss_fn, grad_fn = _problem()
    final, trace = optimize(
        RiemannianGD(lr=0.01), tensors, loss_fn, grad_fn, max_iter=5, tol=1e9, record_loss=True
    )
    assert len(trace) == 1 and all(jnp.array_equal(a, b) for a, b in zip(final, tensors))


def test_an_unknown_optimiser_is_refused():
    @dataclasses.dataclass(frozen=True)
    class Momentum:
        lr: float = 0.1
        max_grad_norm: float | None = None

    tensors, loss_fn, grad_fn = _problem()
    with pytest.raises(TypeError, match="unsupported optimizer type: Momentum"):
        optimize(Momentum(), tensors, loss_fn, grad_fn, max_iter=1)


def test_write_back_unstacks_every_group_into_the_tensor_list():
    tensors = list(pdft.RichBasis(m=2, n=2).tensors)
    state = _common_setup(tensors)
    for manifold in state.point_batches:
        state.point_batches[manifold] = 2 * state.point_batches[manifold]
    _write_back(state)
    assert all(
        jnp.array_equal(now, 2 * before) for now, before in zip(state.current_tensors, tensors)
    )


def test_a_clipped_gd_step_has_the_length_of_the_clip():
    """The line search judges a clipped step against the clipped norm. Against the
    unclipped one a small clip never shows sufficient decrease: the search runs out
    and the step shrinks to its last candidate."""
    basis = pdft.RichBasis(m=2, n=2)
    image = complex_image((4, 4))

    def loss_fn(tensors):
        return pdft.loss_function(tensors, 2, 2, basis.code, image, pdft.L1Norm())

    moved, _ = optimize(
        RiemannianGD(lr=0.1, max_grad_norm=1e-6),
        list(basis.tensors),
        loss_fn,
        jax.grad(loss_fn),
        max_iter=1,
        tol=0.0,
    )
    distance = math.sqrt(
        sum(float(jnp.sum(jnp.abs(a - b) ** 2)) for a, b in zip(moved, basis.tensors))
    )
    assert distance == pytest.approx(0.1 * 1e-6, rel=1e-3)
