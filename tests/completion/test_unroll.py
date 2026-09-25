import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.unroll as U
from pdft.completion.families.shared import expand, init_shared
from pdft.completion.protocol import train_k


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
    assert e64["retained_bytes"] == 16 * 64 * 64 * 8
    assert e32["retained_bytes"] == e64["retained_bytes"] // 2
    assert e64["total_bytes"] == e64["retained_bytes"] + e64["working_bytes"]
    big = U.plan((64, 64), 16, 10**12)
    assert big["strategy"] == "step" and big["fits"]
    small = U.plan((64, 64), 16, e64["total_bytes"] // 2)
    assert small["strategy"] == "nested" and small["n_outer"] is not None
    assert small["total_bytes"] < e64["total_bytes"]
    tiny = U.plan((64, 64), 16, 1)
    assert tiny["strategy"] == "nested" and not tiny["fits"]


def test_report_lists_every_schedule():
    text = U.report((4096, 4096), 100)
    for s in U.STRATEGIES:
        assert s in text
    assert "float64" in text and "float32" in U.report((64, 64), 4, jnp.float32)


def _problem(n, K, seed=0):
    N = 2**n
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.random((N, N)))
    obs = jnp.asarray(rng.random((N, N)) < 0.3)
    return X, obs, train_k(N * N, 0.3, 0.125), jnp.asarray(init_shared(n)) + 0.05


def _loss(psi, X, obs, k, K, n, strategy, n_outer=None):
    p = expand(psi, n)
    Xh = U.reconstruct(p, p, X * obs, obs, k, K, nr=n, nc=n, strategy=strategy, n_outer=n_outer)
    return jnp.mean((Xh - X) ** 2)


def test_every_schedule_computes_the_same_function():
    n, K = 4, 6
    X, obs, k, psi = _problem(n, K)
    ref_v, ref_g = jax.value_and_grad(lambda q: _loss(q, X, obs, k, K, n, "none"))(psi)
    for s, no in (("step", None), ("nested", None), ("nested", 4), ("nested", 6), ("nested", 1)):
        v, g = jax.value_and_grad(lambda q, s=s, no=no: _loss(q, X, obs, k, K, n, s, no))(psi)
        assert abs(float(v) - float(ref_v)) < 1e-12
        assert float(jnp.abs(g - ref_g).max()) < 1e-9


def test_auto_strategy_plans_against_the_budget():
    n, K = 4, 6
    X, obs, k, psi = _problem(n, K)
    p = expand(psi, n)
    a = U.reconstruct(p, p, X * obs, obs, k, K, strategy="auto", budget_bytes=1)
    b = U.reconstruct(p, p, X * obs, obs, k, K, strategy="step")
    assert jnp.allclose(a, b, atol=1e-12)
    c = U.reconstruct(p, p, X * obs, obs, k, K, strategy="auto")  # reads the device
    assert jnp.allclose(a, c, atol=1e-12)


def test_rectangular_registers():
    rng = np.random.default_rng(1)
    nr, nc = 4, 3
    X = jnp.asarray(rng.random((2**nr, 2**nc)))
    obs = jnp.asarray(rng.random(X.shape) < 0.5)
    pr, pc = expand(init_shared(nr), nr), expand(init_shared(nc), nc)
    out = U.reconstruct(pr, pc, X * obs, obs, 8, 3, strategy="nested")
    assert out.shape == X.shape and bool(jnp.all(jnp.isfinite(out)))


def test_errors():
    X = jnp.zeros((12, 16))
    p = expand(init_shared(4), 4)
    with pytest.raises(ValueError, match="power of two"):
        U.reconstruct(p, p, X, X > 0, 4, 2, strategy="step")
    X = jnp.zeros((16, 16))
    with pytest.raises(ValueError, match="unknown strategy"):
        U.reconstruct(p, p, X, X > 0, 4, 2, strategy="bogus")


def test_device_helpers_return_numbers():
    assert isinstance(U.device_peak_mb(), float)
    assert U._free_bytes() > 0
