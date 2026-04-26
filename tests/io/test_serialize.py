import hashlib
from pathlib import Path

import jax.numpy as jnp
import pytest

from pdft.bases.base import QFTBasis, bases_allclose
from pdft.io.serialize import basis_hash, basis_to_dict, dict_to_basis, load_basis, save_basis


def test_basis_hash_deterministic():
    b = QFTBasis(m=2, n=2)
    assert basis_hash(b) == basis_hash(b)


def test_basis_hash_changes_on_tensor_perturbation():
    b1 = QFTBasis(m=2, n=2)
    b2 = QFTBasis(m=2, n=2, tensors=[t + 1e-10 for t in b1.tensors])
    assert basis_hash(b1) != basis_hash(b2)


def test_basis_hash_is_64_hex_chars():
    h = basis_hash(QFTBasis(m=2, n=2))
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_basis_hash_format_matches_spec():
    """Canonical string is 'QFTBasis:m=M:n=N:<re>,<im>;...' and sha256 of it matches."""
    from pdft.io.serialize import _format_float_julia_like as f

    b = QFTBasis(m=1, n=1)
    h = 1.0 / float(jnp.sqrt(2.0))
    import numpy as np

    H = np.array([[h, h], [h, -h]], dtype=np.complex128)
    vals = H.flatten(order="F")
    parts = ["QFTBasis:m=1:n=1:"]
    for _ in range(2):  # two Hadamards
        for v in vals:
            parts.append(f"{f(v.real)},{f(v.imag)};")
    expected = hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()
    assert basis_hash(b) == expected


def test_basis_to_dict_has_required_fields():
    b = QFTBasis(m=2, n=2)
    d = basis_to_dict(b)
    assert d["type"] == "QFTBasis"
    assert d["version"] == "1.0"
    assert d["m"] == 2 and d["n"] == 2
    assert d["hash"] == basis_hash(b)
    assert len(d["tensors"]) == len(b.tensors)
    # Each tensor serialized as list of [re, im] pairs
    for t, serialized in zip(b.tensors, d["tensors"]):
        assert len(serialized) == t.size
        assert all(len(pair) == 2 for pair in serialized)


def test_dict_roundtrip_preserves_basis():
    b = QFTBasis(m=2, n=2)
    d = basis_to_dict(b)
    b2 = dict_to_basis(d)
    assert bases_allclose(b, b2, atol=1e-14)


def test_dict_roundtrip_with_custom_tensors():
    # Use a slightly perturbed basis so we're not just checking the default init.
    b = QFTBasis(m=2, n=2)
    perturbed = QFTBasis(m=2, n=2, tensors=[t + 1e-6 * (0.3 + 0.5j) for t in b.tensors])
    d = basis_to_dict(perturbed)
    loaded = dict_to_basis(d)
    assert bases_allclose(perturbed, loaded, atol=1e-14)
    assert d["hash"] == basis_hash(loaded)


def test_file_roundtrip(tmp_path: Path):
    b = QFTBasis(m=2, n=2)
    path = tmp_path / "basis.json"
    save_basis(path, b)
    assert path.exists()
    loaded = load_basis(path)
    assert bases_allclose(b, loaded, atol=1e-14)


def test_dict_to_basis_rejects_wrong_type():
    with pytest.raises(ValueError, match="Unknown basis type"):
        dict_to_basis(
            {"type": "MERABasis", "version": "1.0", "m": 2, "n": 2, "tensors": [], "hash": ""}
        )


def test_dict_to_basis_warns_on_hash_mismatch():
    b = QFTBasis(m=2, n=2)
    d = basis_to_dict(b)
    d["hash"] = "0" * 64
    with pytest.warns(UserWarning, match="hash mismatch"):
        dict_to_basis(d)


def test_dict_to_basis_warns_on_version_mismatch():
    b = QFTBasis(m=2, n=2)
    d = basis_to_dict(b)
    d["version"] = "9.9"
    with pytest.warns(UserWarning, match="version"):
        dict_to_basis(d)
