import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.baselines import fixed_bases as F
from pdft.completion.baselines import transform_learning as TL

N, P = 8, 4


def _analysis(D):
    return lambda x: TL.analysis_t(x, D, D)


def test_dct_matrix_is_orthonormal_and_the_pair_inverts(rng):
    D = TL.dct_matrix(N)
    assert np.allclose(D @ D.T, np.eye(N), atol=1e-12)
    X = rng.random((N, N))
    assert np.allclose(TL.synthesis_t(TL.analysis_t(X, D, D), D, D), X, atol=1e-12)
    assert 1.0 < TL.coherence_t(D) <= 2.0  # below 2 at small N: no cosine hits 1 exactly
    assert TL.n_params(N) == 64


def test_procrustes_and_row_threshold(rng):
    W = TL._procrustes(rng.standard_normal((N, N)))
    assert np.allclose(W @ W.T, np.eye(N), atol=1e-12)
    C = rng.standard_normal((6, 5))
    assert np.all(np.count_nonzero(TL.hard_s_rows(C, 2), axis=1) == 2)
    assert np.array_equal(TL.hard_s_rows(C, 5), C)


def test_objective_and_compression(images):
    imgs = list(images(3, 2))
    D = TL.dct_matrix(N)
    e = TL.sparsification_error(imgs, _analysis(D), 6)
    assert 0 <= e <= 1 and TL.sparsification_error(imgs, _analysis(D), N * N) == 0
    assert TL.compress_psnr(imgs, _analysis(D), lambda c: TL.synthesis_t(c, D, D), 10).shape == (2,)
    W = TL.dct2_patch(P)
    assert TL.compress_psnr(
        imgs, lambda x: TL.analysis_p(x, W, P), lambda c: TL.synthesis_p(c, W, P, N), 10
    ).shape == (2,)


def test_learn_transform_with_validation(images, capsys):
    out = TL.learn_transform(
        list(images(3, 3)), 6, iters=4, log_every=2, val_images=list(images(3, 2, seed=9))
    )
    assert set(out) == {"validated", "final", "history"}
    Wr = out["final"]["Wr"]
    assert np.allclose(Wr @ Wr.T, np.eye(N), atol=1e-10)
    assert "validation selects" in capsys.readouterr().out
    assert (
        TL.learn_transform(list(images(3, 3)), 6, iters=1, init="identity", log_every=1)["final"][
            "iter"
        ]
        == 1
    )
    with pytest.raises(ValueError):
        TL.learn_transform(list(images(3, 3)), 6, iters=1, init="random")


def test_patch_machinery(images, rng):
    X = rng.random((N, N))
    Y = TL.im2patches(X, P)
    assert Y.shape == (4, 16) and np.array_equal(TL.patches2im(Y, P, N), X)
    W = TL.dct2_patch(P)
    assert np.allclose(W @ W.T, np.eye(P * P), atol=1e-12)
    assert np.allclose(TL.synthesis_p(TL.analysis_p(X, W, P), W, P, N), X, atol=1e-12)
    assert TL.coherence_patch(W, N) == pytest.approx(N * N * np.max(W**2))
    out = TL.learn_patch_transform(
        list(images(3, 3)), 10, P=P, iters=2, sparsities=(1, 2), log_every=1
    )
    assert set(out) == {"best", "sweep"} and out["best"]["s"] in (1, 2)


def test_recovery_through_the_shared_solver(rng):
    D = jnp.asarray(TL.dct_matrix(N))
    X = jnp.asarray(rng.random((N, N)), dtype=jnp.float32)
    obs = jnp.asarray(rng.random((N, N)) < 0.5)
    out = TL.reconstruct_t(D, D, X * obs, obs, 10, 3)
    assert out.dtype == jnp.float32 and out.shape == (N, N)
    outp = TL.reconstruct_p(jnp.asarray(TL.dct2_patch(P)), X * obs, obs, P, 10, 3)
    assert outp.shape == (N, N) and bool(jnp.all(jnp.isfinite(outp)))
    # the dense separable solver at the DCT is the numpy fixed-basis IHT at the DCT
    X64 = jnp.asarray(rng.random((16, 16)))
    o64 = jnp.asarray(rng.random((16, 16)) < 0.5)
    D16 = jnp.asarray(TL.dct_matrix(16))
    fwd, inv = F.dct_pair()
    ref = F.iht_fixed(np.asarray(X64), np.asarray(o64), 20, 5, fwd, inv)
    assert np.allclose(
        np.asarray(TL.reconstruct_t(D16, D16, X64 * o64, o64, 20, 5)), ref, atol=1e-10
    )
