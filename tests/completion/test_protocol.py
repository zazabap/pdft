import json

import jax.numpy as jnp
import numpy as np

import pdft.completion.protocol as P
from pdft.completion.families.butterfly import init_butterfly
from pdft.completion.families.general import init_general
from pdft.completion.solver import reconstruct
from pdft.completion.transform import theta0


def test_heldout_mask_is_deterministic_and_at_rate():
    a = P.heldout_mask(3, (64, 64))
    b = P.heldout_mask(3, (64, 64))
    assert a.dtype == bool and np.array_equal(a, b)
    assert abs(a.mean() - P.TABLE1_P) < 0.03
    assert not np.array_equal(a, P.heldout_mask(4, (64, 64)))


def test_budget_rules():
    obs = np.zeros((32, 32), bool)
    obs[:16] = True  # 512 observed
    assert P.budget_k(obs, 0.5) == 256
    assert P.budget_k(obs, 0.01) == 64  # the floor
    assert P.train_k(32 * 32, 0.5, 0.5) == 256
    assert P.train_k(100, 0.1, 0.1) == 64


def test_fracs_for_every_regime():
    assert P.fracs_for(0.10) == P.TABLE1_FRACS
    assert P.fracs_for(0.05)[0] == 0.0075 and P.fracs_for(0.05)[1:] == P.TABLE1_FRACS
    assert P.fracs_for(0.30)[-1] == 0.75


def _images(n=4, count=2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random((2**n, 2**n)) for _ in range(count)]


def test_evaluate_and_table1_scores_with_the_dft():
    n = 4
    th = theta0(n)

    def solve(Y, obs, k):
        return reconstruct(th, th, Y, obs, n, k, 3)

    psnrs = P.evaluate(solve, _images(n), 0.5, 0.25, seed=0)
    assert psnrs.shape == (2,) and np.all(psnrs > 5)
    scores = P.table1_scores(solve, _images(n), fracs=(0.25, 0.5), p=0.5)
    assert set(scores) == {"psnr", "ssim", "msssim"}
    assert all(len(v) == 2 for v in scores.values())


def test_per_metric_best_reads_each_metric_at_its_own_optimum():
    rng = np.random.default_rng(2)
    img = rng.random((32, 32))
    cands = [np.clip(img + s * rng.standard_normal(img.shape), 0, 1) for s in (0.3, 0.05, 0.1)]
    best = P.per_metric_best(cands, img)
    from pdft.completion.metrics import ms_ssim, psnr, ssim

    assert best["psnr"] == max(psnr(c, img) for c in cands)
    assert best["ssim"] == max(ssim(c, img) for c in cands)
    assert best["msssim"] == max(ms_ssim(c, img) for c in cands)


def test_gphi_and_butterfly_encodings_round_trip():
    p = init_general(4)
    p = {"g": p["g"] * jnp.exp(0.3j), "phi": p["phi"] + 0.1}
    par = {"r": p, "c": init_general(4)}
    back = P.decode_gphi(json.loads(json.dumps(P.encode_gphi(par))))
    for a in ("r", "c"):
        assert jnp.allclose(back[a]["g"], par[a]["g"])
        assert jnp.allclose(back[a]["phi"], par[a]["phi"])
    for mode in ("unitary", "free"):
        bpar = {"r": init_butterfly(3, mode), "c": init_butterfly(3, mode)}
        back = P.decode_butterfly(json.loads(json.dumps(P.encode_butterfly(bpar))))
        key = "gen" if mode == "unitary" else "blk"
        for a in ("r", "c"):
            assert back[a][key].shape == bpar[a][key].shape
            assert jnp.allclose(back[a][key], bpar[a][key])


def test_write_json(tmp_path, capsys):
    path = P.write_json(tmp_path / "sub" / "out.json", {"a": 1, "b": np.float64(2.5)})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2.5}
    assert "wrote" in capsys.readouterr().out
