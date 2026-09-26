import jax
import jax.numpy as jnp
import numpy as np

import pdft.completion.solver as S
from pdft.completion.families.phases import reconstruct
from pdft.completion.transform import apply_dense, theta0

n = 5
N = 2**n


def _dft_matrix(N):
    """The DFT in the QFT sign convention, exp(+2 pi i jk / N) / sqrt(N), as a dense matrix."""
    j = np.arange(N)
    return jnp.asarray(np.exp(2j * np.pi * np.outer(j, j) / N) / np.sqrt(N))


def test_kth_largest_static_and_traced_agree(rng):
    mag = jnp.asarray(rng.random((8, 8)))
    assert float(S.kth_largest(mag, 5)) == float(jax.jit(S.kth_largest)(mag, 5))
    assert float(S.kth_largest(mag, 5)) == float(np.sort(np.asarray(mag).ravel())[-5])


def test_hard_k_keeps_exactly_k_entries(rng):
    C = jnp.asarray(rng.standard_normal((6, 6)))
    out = S.hard_k(C, 7)
    assert int(jnp.count_nonzero(out)) == 7 and jnp.allclose(jnp.where(out != 0, C, 0.0), out)


def test_soft_k_shrinks_towards_zero(rng):
    C = jnp.asarray(rng.standard_normal((6, 6)))
    out = S.soft_k(C, 7)
    assert int(jnp.count_nonzero(out)) <= 7
    assert bool(jnp.all(jnp.abs(out) <= jnp.abs(C) + 1e-12)) and bool(
        jnp.all(jnp.sign(out) * jnp.sign(C) >= 0)
    )


def test_solver_for_builds_the_same_solver_from_any_operator(sparse_problem):
    """The dense operator at the DFT matrix and the circuit at theta0 are the same
    matrix, so their solvers agree (on a well-separated spectrum, where top-k
    has no rounding-level ties to break differently)."""
    X, obs, th = sparse_problem(seed=5)
    dense = S.solver_for(apply_dense)
    U = _dft_matrix(N)
    assert jnp.allclose(
        dense(U, U, X * obs, obs, 20, 6), reconstruct(th, th, X * obs, obs, 20, 6), atol=1e-10
    )


def test_batched_vmaps_over_images(rng):
    X = jnp.asarray(rng.random((2, N, N)))
    obs = jnp.asarray(rng.random((2, N, N)) < 0.5)
    th = theta0(n)
    out = S.batched(reconstruct)(th, th, X * obs, obs, 20, 3)
    for i in range(2):
        assert jnp.allclose(out[i], reconstruct(th, th, X[i] * obs[i], obs[i], 20, 3), atol=1e-12)


def test_iht_accepts_any_exact_pair(rng):
    X = jnp.asarray(rng.random((8, 8)))
    obs = jnp.asarray(rng.random((8, 8)) < 0.5)
    out = S.iht(
        lambda x: x, lambda c: c, X * obs, obs, 10, 3, remat=False
    )  # the identity basis: nothing to recover
    assert jnp.allclose(out, jnp.where(obs, X, 0.0))
