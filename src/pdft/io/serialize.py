"""Cross-language JSON serialization for AbstractSparseBasis instances.

Mirror of upstream src/serialization.jl (Phase 2 scope: QFTBasis only;
EntangledQFTBasis / TEBDBasis / MERABasis are deferred to Phase 3).

Schema (QFTBasis):
    {
      "type": "QFTBasis",
      "version": "1.0",
      "m": <int>,
      "n": <int>,
      "tensors": [[[re, im], ...], ...],
      "hash": "<sha256 hex>"
    }

Each tensor is serialized as an array of [real, imag] pairs, traversed in
**column-major (Fortran-order)** order to match Julia's default array
iteration. `basis_hash` likewise iterates column-major. Without this,
the hash and JSON byte-layout would differ from Julia's output even
for identical tensor values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..bases.base import QFTBasis

_VERSION = "1.0"


def _iter_column_major(arr: np.ndarray):
    """Yield elements of `arr` in column-major (Fortran) order.

    Julia iterates multi-dimensional arrays in column-major order by default;
    for cross-language hash/JSON parity, Python must do the same regardless
    of its native row-major default.
    """
    return np.asarray(arr).flatten(order="F")


def _format_float_julia_like(x) -> str:
    """Format a Python float the way Julia's `string(x::Float64)` does.

    Python and Julia agree on decimal forms ("1.0", "3.14", "-0.0") and on
    the base algorithm (shortest round-trip representation). They differ in
    scientific notation:

        Python repr    Julia string
        ------------   ------------
        5e-07          5.0e-7         ← Julia always has '.0' in mantissa
        1e+20          1.0e20         ← Julia drops '+' from positive exponent
        1.5e-07        1.5e-7         ← Julia drops leading zero in exponent

    We produce Julia's form by post-processing Python's `repr`.
    """
    # np.float64 reprs as 'np.float64(1.5)'; cast to Python float so repr
    # yields the bare decimal form that Julia uses too.
    s = repr(float(x))
    if "e" not in s:
        return s

    mantissa, exp = s.split("e", 1)

    # Python always includes sign in exponent (e+20, e-07); Julia drops '+'.
    if exp.startswith("+"):
        exp = exp[1:]

    # Strip leading zeros in the exponent magnitude, but keep at least one digit.
    if exp.startswith("-"):
        exp = "-" + exp[1:].lstrip("0") or "-0"
        if exp == "-":
            exp = "-0"
    else:
        exp = exp.lstrip("0") or "0"

    # Mantissa must contain '.0' if it's integer-valued (Julia convention).
    if "." not in mantissa:
        mantissa = mantissa + ".0"

    return f"{mantissa}e{exp}"


def basis_hash(basis: QFTBasis) -> str:
    """SHA256 of a canonical string form, matching Julia's `basis_hash(QFTBasis)`.

    Canonical form:
        "QFTBasis:m=<m>:n=<n>:<re0>,<im0>;<re1>,<im1>;..."

    Tensor elements are iterated tensor-by-tensor, column-major within each
    tensor. Floats are formatted with Python `repr` — this matches Julia's
    `string(Float64)` for values produced by deterministic numerical
    operations (Julia and Python agree that `string(0.7071067811865475)`
    equals Python's `repr(0.7071067811865475)`).
    """
    parts: list[str] = [f"QFTBasis:m={basis.m}:n={basis.n}:"]
    for t in basis.tensors:
        arr = np.asarray(t)
        for val in _iter_column_major(arr):
            parts.append(
                f"{_format_float_julia_like(val.real)},{_format_float_julia_like(val.imag)};"
            )
    canonical = "".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def basis_to_dict(basis: QFTBasis) -> dict:
    """Serialize a QFTBasis to a plain Python dict (JSON-ready).

    Mirror of upstream src/serialization.jl:477-495. Tensor flattening is
    column-major to match Julia's `for val in tensor` iteration.
    """
    serialized_tensors: list[list[list[float]]] = []
    for t in basis.tensors:
        arr = np.asarray(t)
        flat = _iter_column_major(arr)
        serialized_tensors.append([[float(v.real), float(v.imag)] for v in flat])
    return {
        "type": "QFTBasis",
        "version": _VERSION,
        "m": int(basis.m),
        "n": int(basis.n),
        "tensors": serialized_tensors,
        "hash": basis_hash(basis),
    }


def dict_to_basis(d: dict) -> QFTBasis:
    """Inverse of `basis_to_dict`. Verifies hash match; warns (not raises) on mismatch.

    Mirror of upstream src/serialization.jl:569 / 285-323.
    """
    if d.get("type") != "QFTBasis":
        raise ValueError(
            f"Unknown basis type: {d.get('type')!r}. Only QFTBasis is supported in Phase 2."
        )
    version = d.get("version", _VERSION)
    if version != _VERSION:
        import warnings

        warnings.warn(
            f"Basis version {version!r} may not be fully compatible with current version {_VERSION!r}",
            stacklevel=2,
        )

    m, n = int(d["m"]), int(d["n"])

    # Rebuild the circuit to get template tensor shapes.
    from ..bases.circuit.qft import qft_code

    _code, template_tensors = qft_code(m, n)

    serialized = d["tensors"]
    if len(serialized) != len(template_tensors):
        raise ValueError(
            f"Tensor count mismatch: expected {len(template_tensors)} (from qft_code(m={m}, n={n})), "
            f"got {len(serialized)}"
        )

    tensors: list = []
    for i, tensor_data in enumerate(serialized):
        shape = template_tensors[i].shape
        complex_vals = np.array(
            [complex(pair[0], pair[1]) for pair in tensor_data],
            dtype=np.complex128,
        )
        if complex_vals.size != int(np.prod(shape)):
            raise ValueError(
                f"Tensor {i} element count mismatch: expected {int(np.prod(shape))} "
                f"(shape {shape}), got {complex_vals.size}"
            )
        # Julia flattened column-major; reshape back with order='F' to undo.
        tensor = jnp.asarray(complex_vals.reshape(shape, order="F"))
        tensors.append(tensor)

    basis = QFTBasis(m=m, n=n, tensors=tensors)

    expected_hash = d.get("hash")
    if expected_hash is not None:
        computed = basis_hash(basis)
        if computed != expected_hash:
            import warnings

            warnings.warn(
                f"Basis hash mismatch. File hash: {expected_hash}, computed: {computed}. "
                "Basis may have been corrupted or written by a non-canonical serializer.",
                stacklevel=2,
            )

    return basis


def save_basis(path: str | Path, basis: QFTBasis) -> Path:
    """Write basis to a JSON file. Mirror of upstream src/serialization.jl:100-106."""
    p = Path(path)
    d = basis_to_dict(basis)
    with p.open("w") as f:
        json.dump(d, f, indent=2)
    return p


def load_basis(path: str | Path) -> QFTBasis:
    """Read basis from a JSON file. Mirror of upstream src/serialization.jl:247-278."""
    p = Path(path)
    with p.open("r") as f:
        d: dict[str, Any] = json.load(f)
    return dict_to_basis(d)
