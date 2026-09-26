import numpy as np

from pdft.completion.baselines import nuclear as Nc


def _low_rank(seed=0, N=32, r=3, p=0.6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, r)) @ rng.standard_normal((r, N)) / r
    obs = rng.random((N, N)) < p
    return X, obs


def test_svp_recovers_a_low_rank_matrix():
    X, obs = _low_rank()
    out = Nc.svp(X * obs, obs, 3, iters=300)
    assert np.linalg.norm(out - X) < 1e-3 * np.linalg.norm(X)
    assert np.array_equal(out[obs], X[obs])


def test_apg_improves_over_zero_fill():
    X, obs = _low_rank(seed=1)
    out = Nc.apg(X * obs, obs, 0.05, iters=200)
    assert np.linalg.norm(out - X) < 0.5 * np.linalg.norm(np.where(obs, X, 0) - X)
    assert np.linalg.matrix_rank(out, tol=1e-6) <= 32
