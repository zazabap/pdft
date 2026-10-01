"""The cases the characterisation snapshots cover, and what is recorded for each.

The tests and ``regenerate.py`` both call ``static_record`` and
``training_record``, so what is stored and what is compared cannot drift apart.

A record is a flat ``{name: numpy array}`` dict. Everything in it is observable
behaviour of the package as it stands: initial tensors and their order, the
output of the applier at the initial and at generic tensors, gradients through
the package's own losses, what ``freeze_as_blocked`` returns, the dtype each
input dtype comes back as, and short training trajectories.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import pdft

SNAPSHOT_PATH = Path(__file__).with_name("snapshots.npz")

# One case per gate kind and parametrization the builder supports, at sizes
# small enough to keep the snapshot file small. Rectangular where the basis
# allows it, so a swapped register cannot go unnoticed.
BASES: dict[str, Callable[[], object]] = {
    "qft_2x2": lambda: pdft.QFTBasis(m=2, n=2),
    "qft_3x2": lambda: pdft.QFTBasis(m=3, n=2),
    "entangled_3x2": lambda: pdft.EntangledQFTBasis(m=3, n=2, seed=1),
    "tebd_cp_3x2": lambda: pdft.TEBDBasis(m=3, n=2, seed=1),
    "tebd_u4_2x2": lambda: pdft.TEBDBasis(m=2, n=2, seed=1, parametrization="u4"),
    "mera_cp_4x2": lambda: pdft.MERABasis(m=4, n=2, seed=1),
    "mera_u4_2x2": lambda: pdft.MERABasis(m=2, n=2, seed=1, parametrization="u4"),
    "dct4_o4_3x2": lambda: pdft.DCT4Basis(m=3, n=2),
    "dct4_controlled_3x2": lambda: pdft.DCT4Basis(m=3, n=2, parametrization="controlled"),
    "rich_3x2": lambda: pdft.RichBasis(m=3, n=2),
    "real_rich_2x2": lambda: pdft.RealRichBasis(m=2, n=2),
    "blocked_qft_2x2_in_3x3": lambda: pdft.BlockedBasis(pdft.QFTBasis(m=2, n=2), 1, 1),
}

# The trainer paths, as ``(name, runner)``; a runner takes ``(basis, images, k)``.
_RUNS: dict[str, Callable] = {
    "single_gd_l1": lambda basis, images, k: pdft.train_basis(
        basis, target=images[0], loss=pdft.L1Norm(), optimizer=pdft.RiemannianGD(lr=0.01), steps=4
    ),
    "single_adam_mse": lambda basis, images, k: pdft.train_basis(
        basis,
        target=images[0],
        loss=pdft.MSELoss(k=k),
        optimizer=pdft.RiemannianAdam(lr=0.01),
        steps=4,
    ),
    "batched_adam_mse": lambda basis, images, k: pdft.train_basis_batched(
        basis,
        dataset=list(images),
        loss=pdft.MSELoss(k=k),
        epochs=2,
        batch_size=2,
        optimizer="adam",
        seed=0,
    ),
    "batched_gd_l1": lambda basis, images, k: pdft.train_basis_batched(
        basis,
        dataset=list(images),
        loss=pdft.L1Norm(),
        epochs=2,
        batch_size=2,
        optimizer="gd",
        seed=0,
    ),
}

# Every trainer path on the two circuits that differ most (diagonal two-qubit
# gates against dense ones), and the fused batched step on each remaining gate
# kind and on the block wrapper.
TRAININGS: dict[str, tuple[str, ...]] = {
    "qft_3x2": tuple(_RUNS),
    "rich_3x2": tuple(_RUNS),
    "tebd_u4_2x2": ("batched_adam_mse",),
    "dct4_controlled_3x2": ("batched_adam_mse",),
    "blocked_qft_2x2_in_3x3": ("batched_adam_mse",),
}

_FREEZABLE = (pdft.QFTBasis, pdft.RichBasis, pdft.RealRichBasis)
_INPUT_DTYPES = (jnp.float32, jnp.float64, jnp.complex64)


def case_rng(case: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(case.encode()))


def complex_normal(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _packed(arrays) -> np.ndarray:
    """A list of arrays as one flat array, in list order; ``_shapes`` records how to read it."""
    return np.concatenate([np.asarray(a).ravel() for a in arrays])


def _shapes(arrays) -> np.ndarray:
    """One row per array, its shape padded with zeros to rank 4."""
    return np.asarray([tuple(a.shape) + (0,) * (4 - a.ndim) for a in arrays])


def with_tensors(basis, tensors):
    """The same basis carrying other tensors, through its pytree like the trainers do."""
    _, treedef = jax.tree_util.tree_flatten(basis)
    return jax.tree_util.tree_unflatten(treedef, list(tensors))


def generic(basis, rng: np.random.Generator):
    """The basis at tensors with no symmetry left.

    At initialisation most gates are symmetric or diagonal, which hides a
    swapped leg or a transposed gate. Every entry is scaled by its own complex
    factor; the applier is linear in each tensor, so unitarity is not needed to
    pin what it computes.
    """
    return with_tensors(
        basis, [t * jnp.asarray(1 + 0.2 * complex_normal(rng, t.shape)) for t in basis.tensors]
    )


def static_record(case: str) -> dict[str, np.ndarray]:
    """Everything recorded for ``case`` that involves no training."""
    basis = BASES[case]()
    rng = case_rng(case)
    rows, cols = basis.image_size
    probe = jnp.asarray(complex_normal(rng, (rows, cols)))
    image = jnp.asarray(rng.random((rows, cols)))
    moved = generic(basis, rng)
    m, n, k = basis.m, basis.n, rows * cols // 4

    out: dict[str, np.ndarray] = {
        "sizes": np.asarray([rows, cols, basis.num_parameters, len(basis.tensors)]),
        "tensor_shapes": _shapes(basis.tensors),
        "tensors": _packed(basis.tensors),
        "forward": np.asarray(basis.forward_transform(probe)),
        "inverse": np.asarray(basis.inverse_transform(probe)),
        "forward_generic": np.asarray(moved.forward_transform(probe)),
        "inverse_generic": np.asarray(moved.inverse_transform(probe)),
    }

    for name, loss in (("l1", pdft.L1Norm()), ("mse", pdft.MSELoss(k=k))):
        grads = jax.grad(
            lambda ts, loss=loss: pdft.loss_function(
                ts, m, n, moved.code, image, loss, inverse_code=moved.inv_code
            )
        )(list(moved.tensors))
        assert np.array_equal(_shapes(grads), out["tensor_shapes"])
        out[f"grad_{name}"] = _packed(grads)

    dtypes = []
    for dtype in _INPUT_DTYPES:
        complex_input = jnp.issubdtype(dtype, jnp.complexfloating)
        x = (probe if complex_input else image).astype(dtype)
        for transform in (basis.forward_transform, basis.inverse_transform):
            dtypes.append(f"{jnp.dtype(dtype).name}->{transform(x).dtype}")
    out["dtypes"] = np.asarray(dtypes)

    if isinstance(basis, _FREEZABLE):
        frozen_basis, frozen = pdft.freeze_as_blocked(basis, 1, 1)
        out["frozen_indices"] = np.asarray(frozen, dtype=np.int64)
        out["frozen_forward"] = np.asarray(frozen_basis.forward_transform(probe))
    return out


def training_record(case: str, run: str) -> dict[str, np.ndarray]:
    """The loss history of one trainer path on ``case``, and what the trained basis computes.

    The images are complex on purpose. At its initial tensors a Fourier-like
    basis maps a real image to coefficients that come in conjugate pairs of
    exactly equal magnitude, so where the top-k cut of ``MSELoss`` falls inside
    a pair the kept set is decided by the last bit, and the trajectory then
    differs between machines by percents. A complex image has no such pairs,
    and runs the same code.
    """
    basis = BASES[case]()
    rng = case_rng(f"{case}/{run}")
    rows, cols = basis.image_size
    images = jnp.asarray(complex_normal(rng, (4, rows, cols)))
    probe = jnp.asarray(complex_normal(rng, (rows, cols)))
    result = _RUNS[run](basis, images, rows * cols // 4)
    return {
        "loss_history": np.asarray(result.loss_history, dtype=np.float64),
        "trained_forward": np.asarray(result.basis.forward_transform(probe)),
    }


def all_records() -> dict[str, np.ndarray]:
    """Every record under its snapshot key, ``static/<case>/<name>`` or ``train/<case>/<run>/<name>``."""
    out: dict[str, np.ndarray] = {}
    for case in BASES:
        for name, value in static_record(case).items():
            out[f"static/{case}/{name}"] = value
    for case, runs in TRAININGS.items():
        for run in runs:
            for name, value in training_record(case, run).items():
                out[f"train/{case}/{run}/{name}"] = value
    return out
