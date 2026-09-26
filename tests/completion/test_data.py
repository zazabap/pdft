import numpy as np
import pytest

import pdft.completion.data as D

Image = pytest.importorskip("PIL.Image")


def _png(path, h, w, seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return arr


def test_load_gray_crops(tmp_path):
    arr = _png(tmp_path / "a.png", 20, 24)
    gray = np.asarray(Image.fromarray(arr).convert("L"), dtype=np.float64) / 255.0
    centre = D.load_gray(tmp_path / "a.png", 16)
    assert centre.shape == (16, 16) and np.array_equal(centre, gray[2:18, 4:20])
    corner = D.load_gray(tmp_path / "a.png", 16, offset="corner")
    assert np.array_equal(corner, gray[:16, :16])
    with pytest.raises(ValueError):
        D.load_gray(tmp_path / "a.png", 32)


def test_kodak_split(tmp_path):
    for i in range(3):
        _png(tmp_path / f"kodim{i + 1:02d}.png", 24, 32, seed=i)
    tr, te, names = D.kodak_split(tmp_path, size=16, n_train=2)
    assert tr.shape == (2, 16, 16) and te.shape == (1, 16, 16)
    assert names == ["kodim01.png", "kodim02.png", "kodim03.png"]
    with pytest.raises(FileNotFoundError):
        D.kodak_split(tmp_path / "empty")


def test_load_4k_and_detail_window(tmp_path):
    _png(tmp_path / "big.png", 16, 16)
    full = D.load_4k(tmp_path / "big.png", 16)
    half = D.load_4k(tmp_path / "big.png", 8)
    assert half.shape == (8, 8)
    assert np.allclose(half, full.reshape(8, 2, 8, 2).mean(axis=(1, 3)))
    with pytest.raises(ValueError):
        D.load_4k(tmp_path / "big.png", 32)
    img = np.zeros((64, 64))
    img[40:48, 8:16] = np.random.default_rng(0).random((8, 8))
    assert D.detail_window(img, 8, stride=8) == (40, 8, 8)
    assert D.detail_window(img, 0) is None


def test_div2k_splits_and_cache(tmp_path, monkeypatch):
    root = tmp_path / "div2k"
    (root / "DIV2K_train_HR").mkdir(parents=True)
    (root / "DIV2K_valid_HR").mkdir()
    with pytest.raises(FileNotFoundError):
        D.div2k_split(root, size=4)
    for i in range(1, 801):
        _png(root / "DIV2K_train_HR" / f"{i:04d}.png", 4, 4, seed=i)
    for i in range(801, 901):
        _png(root / "DIV2K_valid_HR" / f"{i:04d}.png", 4, 4, seed=i)
    tr, va, te, names = D.div2k_split(root, size=4)
    assert tr.shape == (750, 4, 4) and va.shape == (50, 4, 4) and te.shape == (100, 4, 4)
    assert names[0] == "0001" and names[-1] == "0900" and len(names) == 900
    assert (root / "div2k_4.npz").exists()
    tr2, te2, names2 = D.table1_split(root, size=4)
    assert np.array_equal(tr2, tr) and np.array_equal(te2, te)
    assert names2[750:] == list(D.TEST_NAMES)
    assert np.array_equal(D.table1_val(root, size=4), va)
    te3, nm3 = D.table1_test(root, size=4)
    assert np.array_equal(te3, te) and nm3 == list(D.TEST_NAMES)
    with pytest.raises(ValueError):
        D.div2k_split(root, size=4, n_train=790, n_val=20)
    monkeypatch.setenv("PDFT_SMOKE", "3")
    tr, va, te, names = D.div2k_split(root, size=4)
    assert tr.shape[0] == va.shape[0] == te.shape[0] == 3 and len(names) == 9
    assert names == ["0001", "0002", "0003", "0751", "0752", "0753", "0801", "0802", "0803"]
    assert D.table1_test(root, size=4)[1] == ["0801", "0802", "0803"]
    assert len(D.HELDOUT_NAMES) == 8
