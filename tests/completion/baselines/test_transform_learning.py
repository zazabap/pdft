import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.baselines import transform_learning as TL

N, P = 8, 4


def test_dct_matrix_is_orthonormal_and_the_pair_inverts():
    D = TL.dct_matrix(N)
    assert np.allclose(D @ D.T, np.eye(N), atol=1e-12)
    X = np.random.default_rng(0).random((N, N))
    assert np.allclose(TL.synthesis_t(TL.analysis_t(X, D, D), D, D), X, atol=1e-12)
    assert 1.0 < TL.coherence_t(D) <= 2.0  # below 2 at small N: no cosine hits 1 exactly
    assert TL.n_params(N) == 64


def test_procrustes_and_thresholds():
    rng = np.random.default_rng(1)
    W = TL._procrustes(rng.standard_normal((N, N)))
    assert np.allclose(W @ W.T, np.eye(N), atol=1e-12)
    C = rng.standard_normal((6, 5))
    assert np.count_nonzero(TL.hard_k_np(C, 4)) == 4
    assert np.array_equal(TL.hard_k_np(C, 100), C)
    rows = TL.hard_s_rows(C, 2)
    assert np.all(np.count_nonzero(rows, axis=1) == 2)
    assert np.array_equal(TL.hard_s_rows(C, 5), C)


def _images(count=3, seed=2):
    return list(np.random.default_rng(seed).random((count, N, N)))


def test_learn_transform_with_validation(capsys):
    out = TL.learn_transform(_images(), 6, iters=4, log_every=2, val_images=_images(2, 9))
    assert set(out) == {"validated", "final", "history"}
    Wr = out["final"]["Wr"]
    assert np.allclose(Wr @ Wr.T, np.eye(N), atol=1e-10)
    assert 0 <= out["history"][0]["sparsification_error"] <= 1
    assert "validation selects" in capsys.readouterr().out
    ident = TL.learn_transform(_images(), 6, iters=1, init="identity", log_every=1)
    assert ident["final"]["iter"] == 1
    with pytest.raises(ValueError):
        TL.learn_transform(_images(), 6, iters=1, init="random")


def test_patch_machinery():
    X = np.random.default_rng(3).random((N, N))
    Y = TL.im2patches(X, P)
    assert Y.shape == (4, 16)
    assert np.array_equal(TL.patches2im(Y, P, N), X)
    W = TL.dct2_patch(P)
    assert np.allclose(W @ W.T, np.eye(P * P), atol=1e-12)
    assert np.allclose(TL.synthesis_p(TL.analysis_p(X, W, P), W, P, N), X, atol=1e-12)
    assert TL.coherence_patch(W, N) == pytest.approx(N * N * np.max(W**2))
    assert 0 <= TL.sparsification_error_p([X], W, P, 10) <= 1
    out = TL.learn_patch_transform(_images(), 10, P=P, iters=2, sparsities=(1, 2), log_every=1)
    assert set(out) == {"best", "sweep"} and out["best"]["s"] in (1, 2)
    assert TL.compress_psnr_p(_images(2), W, P, 10).shape == (2,)


def test_recovery_through_the_shared_solver():
    rng = np.random.default_rng(4)
    D = jnp.asarray(TL.dct_matrix(N))
    X = jnp.asarray(rng.random((N, N)), dtype=jnp.float32)
    obs = jnp.asarray(rng.random((N, N)) < 0.5)
    out = TL.reconstruct_t(D, D, X * obs, obs, 10, 3)
    assert out.dtype == jnp.float32 and out.shape == (N, N)
    W = jnp.asarray(TL.dct2_patch(P))
    outp = TL.reconstruct_p(W, X * obs, obs, P, N, 10, 3)
    assert outp.shape == (N, N) and bool(jnp.all(jnp.isfinite(outp)))
    assert TL.compress_psnr(_images(2), np.asarray(D), np.asarray(D), 10).shape == (2,)
