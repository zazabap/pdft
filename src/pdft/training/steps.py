"""Step-based trainer for objectives that need a mask: `train_basis_steps`.

Not in upstream. `train_basis_batched` mirrors Julia's dataset trainer, where
a training example is an image and the loss a function of it alone. Training
a basis through a solver that fills in unobserved pixels has a second random
thing per example, the mask, and it is redrawn every step so that the basis
adapts to the sampling rate and not to one mask. This trainer owns that draw;
the epoch trainer never sees a mask.

Below the loop nothing is separate: the step is `training.adam_step`'s fused
one, the update is the package's one Adam update, and what moves is chosen by
a `ParameterView`, so a flat view gives plain Adam and the tensors themselves
give Riemannian Adam.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..bases.core import ParameterView, tensors_view
from ..optimizers import RiemannianAdam
from .adam_step import adam_stepper
from .batched import _at_least_one, _check_dataset, _check_image_shape, _validate_frozen_indices
from .result import TrainingResult

Array = jax.Array


def _stack_real_images(dataset: Sequence, expected_size: tuple[int, int]) -> np.ndarray:
    """The dataset as one real array, in the precision it came in (integers become float64)."""
    _check_dataset(dataset)
    images = []
    for i, img in enumerate(dataset):
        arr = np.asarray(img)
        _check_image_shape(i, arr, expected_size)
        if np.iscomplexobj(arr):
            raise ValueError(f"dataset[{i}] is complex; a mask is drawn over real images")
        images.append(arr)
    stacked = np.stack(images)
    return stacked if np.issubdtype(stacked.dtype, np.floating) else stacked.astype(np.float64)


def train_basis_steps(
    basis,
    *,
    dataset: Sequence,
    objective: Callable[[object, Array, Array], Array],
    optimizer: RiemannianAdam,
    steps: int,
    rate: float,
    batch_size: int = 2,
    view: ParameterView = tensors_view,
    frozen_indices: list[int] | None = None,
    frame: Callable[[Array], Array] | None = None,
    seed: int = 0,
    callback: Callable[[int, object, float], None] | None = None,
) -> TrainingResult:
    """Train `basis` for `steps` steps, each on a fresh batch under a fresh mask.

    Parameters
    ----------
    dataset
        Real images of the basis's ``image_size``, in the image's own frame.
    objective
        ``(basis, images, masks) -> scalar``, traceable; ``images`` and
        ``masks`` are stacked along a leading axis. For completion,
        ``pdft.tasks.completion_loss``. One that ignores the masks trains on
        the batches alone.
    optimizer
        A ``RiemannianAdam``; its learning rate is used as it is, at every step.
    rate
        The probability that a pixel is observed.
    view
        What is trained. The default is the tensors, each on its manifold.
        ``cp_phases_view`` and ``cp_diagonals_view`` train the controlled-phase angles
        as free numbers, on which the update is plain Adam; every other tensor
        is then left as it is.
    frozen_indices
        Indices into the view's parameters (for the default view, into
        ``basis.tensors``) that are not updated.
    frame
        Applied to each batch and to its masks once they are drawn, to take
        them from the image's frame to the basis's: ``pdft.circuit.bit_reverse``
        for a basis with the QFT topology.
    callback
        ``callback(step, basis, loss)`` after every step: the basis as that
        step left it, and the loss the step started from.

    Returns
    -------
    TrainingResult
        Its ``loss_history`` holds, per step, the loss of that step's batch
        before its update.

    Raises
    ------
    FloatingPointError
        On a loss that is not finite.

    Notes
    -----
    The draw, per step and from one ``np.random.default_rng(seed)``, is the
    batch (``choice`` over the dataset, without replacement within a batch)
    and then its masks (``random(batch.shape) < rate``). That order is the
    completion paper's, so a seed names the same batches and masks as it does
    there.

    Under the default view the tensors come back as complex128 whatever they
    went in as, as ``RiemannianAdam`` returns them everywhere (a frozen tensor
    comes back as it went in). A flat view trains its angles in double
    precision and writes them into tensors of the precision the basis had.

    Single-precision tensors on a GPU cannot be trained under the default
    view: the manifold of a tensor is read off its values, and there that test
    runs in reduced precision and takes unitary gates for phase tensors, so
    the run fails or leaves the manifold. It is the same defect as in the
    other trainers. Train in double precision, or through a flat view, which
    names its manifold and is not affected.
    """
    if not isinstance(optimizer, RiemannianAdam):
        raise TypeError(
            f"train_basis_steps takes a RiemannianAdam, got {type(optimizer).__name__}: "
            "a line search needs a loss that is the same function on every evaluation"
        )
    if not optimizer.lr > 0:
        raise ValueError(f"optimizer.lr must be > 0, got {optimizer.lr}")
    _at_least_one("steps", steps)
    _at_least_one("batch_size", batch_size)
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    images = _stack_real_images(dataset, basis.image_size)

    params = [jnp.asarray(p) for p in view.read(basis)]
    if not any(p.size for p in params):
        raise ValueError(f"the view {view.name!r} reads no parameters from this basis")
    frozen_set = _validate_frozen_indices(
        frozen_indices, len(params), holder=f"view {view.name!r}", held="parameters"
    )

    def in_parameters(params: list[Array], batch: tuple[Array, Array]) -> Array:
        return objective(view.write(basis, params), *batch)

    adam_step = adam_stepper(
        in_parameters,
        params,
        manifolds=view.manifolds(params),
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        eps=optimizer.eps,
        max_grad_norm=optimizer.max_grad_norm,
        frozen_set=frozen_set,
    )

    rng = np.random.default_rng(seed)
    loss_history: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        picked = images[rng.choice(len(images), size=min(batch_size, len(images)), replace=False)]
        batch = (jnp.asarray(picked), jnp.asarray(rng.random(picked.shape) < rate))
        if frame is not None:
            batch = tuple(frame(a) for a in batch)
        params, loss = adam_step(params, batch, optimizer.lr, step + 1)
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss_history.append(loss)
        if callback is not None:
            callback(step, view.write(basis, params), loss)

    return TrainingResult(
        basis=view.write(basis, params),
        loss_history=loss_history,
        seed=seed,
        steps=steps,
        wall_time_s=time.perf_counter() - t0,
    )
