import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.unroll as U
from pdft.completion.families.general import init_general
from pdft.completion.families.riemannian import dft_matrix
from pdft.completion.families.shared import expand, init_shared
from pdft.completion.protocol import train_k
from pdft.completion.transform import apply_dense


def test_split_k_cases():
    assert U.split_k(9) == (3, 3, 0)
    assert U.split_k(10) == (3, 3, 1)
    assert U.split_k(7, 2) == (2, 3, 1)
    assert U.split_k(5, 10) == (5, 1, 0)
    with pytest.raises(ValueError):
        U.split_k(0)


def test_retained_carries():
    assert U.retained_carries(20, "none") == 1
    assert U.retained_carries(20, "step") == 20
    assert U.retained_carries(20, "nested") == 4 + 5
    assert U.retained_carries(10, "nested") == 3 + 3 + 1
    with pytest.raises(ValueError):
        U.retained_carries(4, "bogus")


def test_estimate_peak_and_plan():
    e64 = U.estimate_peak((64, 64), 16, "step")
    e32 = U.estimate_peak((64, 64), 16, "step", jnp.float32)
    assert e64["retained_bytes"] == 16 * 64 * 64 * 8 == 2 * e32["retained_bytes"]
    assert e64["total_bytes"] == e64["retained_bytes"] + e64["working_bytes"]
    big = U.plan((64, 64), 16, 10**12)
    assert big["strategy"] == "step" and big["fits"]
    small = U.plan((64, 64), 16, e64["total_bytes"] // 2)
    assert small["strategy"] == "nested" and small["total_bytes"] < e64["total_bytes"]
    assert not U.plan((64, 64), 16, 1)["fits"]


def test_report_lists_every_schedule():
    text = U.report((4096, 4096), 100)
    assert all(s in text for s in U.STRATEGIES) and "float64" in text
    assert "float32" in U.report((64, 64), 4, jnp.float32)


def _problem(n, seed=0):
    N = 2**n
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.random((N, N)))
    obs = jnp.asarray(rng.random((N, N)) < 0.3)
    return X, obs, train_k(N * N, 0.3, 0.125), jnp.asarray(init_shared(n)) + 0.05


def _loss(psi, X, obs, k, K, n, strategy, n_outer=None):
    p = expand(psi, n)
    return jnp.mean(
        (U.reconstruct(p, p, X * obs, obs, k, K, strategy=strategy, n_outer=n_outer) - X) ** 2
    )


def test_every_schedule_computes_the_same_function():
    n, K = 4, 6
    X, obs, k, psi = _problem(n)
    ref_v, ref_g = jax.value_and_grad(lambda q: _loss(q, X, obs, k, K, n, "none"))(psi)
    for s, no in (("step", None), ("nested", None), ("nested", 4), ("nested", 6), ("nested", 1)):
        v, g = jax.value_and_grad(lambda q, s=s, no=no: _loss(q, X, obs, k, K, n, s, no))(psi)
        assert abs(float(v) - float(ref_v)) < 1e-12 and float(jnp.abs(g - ref_g).max()) < 1e-9


def test_auto_strategy_plans_against_the_budget():
    n, K = 4, 6
    X, obs, k, psi = _problem(n)
    p = expand(psi, n)
    a = U.reconstruct(p, p, X * obs, obs, k, K, strategy="auto", budget_bytes=1)
    assert jnp.allclose(a, U.reconstruct(p, p, X * obs, obs, k, K, strategy="step"), atol=1e-12)
    assert jnp.allclose(a, U.reconstruct(p, p, X * obs, obs, k, K), atol=1e-12)  # reads the device


def test_solver_for_any_operator(sparse_problem):
    """The bounded solver of the dense operator at the DFT matrix equals the
    default circuit solver at theta0, schedule for schedule."""
    X, obs, th = sparse_problem(n=4, seed=2, k=6)
    p, F = init_general(4), dft_matrix(16)
    dense = U.solver_for(apply_dense)
    for s in U.STRATEGIES:
        assert jnp.allclose(
            dense(F, F, X * obs, obs, 8, 5, strategy=s),
            U.reconstruct(p, p, X * obs, obs, 8, 5, strategy=s),
            atol=1e-10,
        )


def test_rectangular_registers():
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.random((16, 8)))
    obs = jnp.asarray(rng.random(X.shape) < 0.5)
    out = U.reconstruct(
        expand(init_shared(4), 4), expand(init_shared(3), 3), X * obs, obs, 8, 3, strategy="nested"
    )
    assert out.shape == X.shape and bool(jnp.all(jnp.isfinite(out)))


def test_errors():
    p = expand(init_shared(4), 4)
    with pytest.raises(ValueError, match="power of two"):
        U.reconstruct(p, p, jnp.zeros((12, 16)), jnp.zeros((12, 16), bool), 4, 2, strategy="step")
    with pytest.raises(ValueError, match="unknown strategy"):
        U.reconstruct(p, p, jnp.zeros((16, 16)), jnp.zeros((16, 16), bool), 4, 2, strategy="bogus")


def test_device_helpers_return_numbers():
    assert isinstance(U.device_peak_mb(), float) and U._free_bytes() > 0
