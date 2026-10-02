"""What every basis shares: the transforms, and the circuit basis they are derived for.

A circuit basis is a gate program and the tensors of its gates.
``CircuitBasis`` keeps both, so nothing downstream has to rebuild or guess the
gate sequence, and it derives the rest once for every circuit family: the
transforms, the parameter count, and the JAX pytree whose leaves are the
tensors. A concrete basis adds only how its gates are emitted.

This module imports nothing from ``pdft.bases.circuit``, which is what lets
the family modules there and ``pdft.bases.base`` both build on it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import tree_util

from ..circuit.builder import (
    CircuitCode,
    Gate,
    Program,
    apply_circuit,
    compile_program,
    controlled_phase_diag,
)
from ..manifolds import EuclideanManifold

Array = jax.Array


@runtime_checkable
class AbstractSparseBasis(Protocol):
    """Structural interface shared by every basis: an ``image_size``, a
    ``num_parameters`` count, and ``forward_transform`` / ``inverse_transform``
    on a ``2^m x 2^n`` image.
    """

    @property
    def image_size(self) -> tuple[int, int]: ...
    @property
    def num_parameters(self) -> int: ...
    def forward_transform(self, pic: Array) -> Array: ...
    def inverse_transform(self, pic: Array) -> Array: ...


class BasisTransforms:
    """What follows from ``m``, ``n``, ``tensors``, ``code`` and ``inv_code``, however a basis holds them.

    ``_apply`` is how a transform reaches the circuit. The default,
    ``apply_circuit``, checks the image's shape and works in double precision.
    The Rich, RealRich and Blocked bases set it to ``contract_circuit``, which
    does neither: they were written that way, single-precision tensors stay
    single precision through them, and changing either side would change
    results that exist.
    """

    _apply: ClassVar[Callable[..., Array]] = staticmethod(apply_circuit)

    @property
    def inv_tensors(self) -> list[Array]:
        """The tensor list itself.

        Julia stores one list and computes the inverse by applying
        ``conj(tensors)`` through the inverse code; the name is kept for
        callers that read ``basis.inv_tensors``.
        """
        return self.tensors

    @property
    def image_size(self) -> tuple[int, int]:
        return (2**self.m, 2**self.n)

    @property
    def num_parameters(self) -> int:
        return sum(int(t.size) for t in self.tensors)

    def forward_transform(self, pic: Array) -> Array:
        return self._apply(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        """``conj(tensors)`` through ``inv_code``, exactly like Julia's
        ``basis.inverse_code(conj.(basis.tensors)..., ...)``."""
        return self._apply([jnp.conj(t) for t in self.tensors], self.inv_code, self.m, self.n, pic)


@dataclass(init=False)
class CircuitBasis(BasisTransforms):
    """A basis that is a gate circuit on ``m`` row qubits and ``n`` column qubits.

    ``tensors`` holds one tensor per gate, Hadamards first (Julia's
    ``perm_vec``); ``program`` says which gate each one is and when it acts.
    ``code`` and ``inv_code`` are the program's forward and inverse appliers;
    they compare by program, so they and ``program`` are left out of the
    ``repr``. To compare two bases use ``bases_allclose``: ``==`` compares the
    tensor lists and raises as soon as it reaches two distinct arrays (it is
    ``True`` only when every tensor is the same object).
    ``program`` is derived, never passed in, which keeps
    ``dataclasses.replace(basis, tensors=...)`` working on the bases whose
    fields are all constructor arguments.

    Every subclass is registered as a JAX pytree: the leaves are ``tensors``,
    in order, which is the contract ``train_basis`` relies on. Every other
    attribute of the instance is aux data, so it must be hashable.
    Unflattening restores the attributes without running the constructor, so
    no gate list is rebuilt.

    A family whose circuit depends on ``m`` and ``n`` alone sets ``emit`` to
    its gate emitter and is done. One with options of its own defines a
    constructor that emits its gates and hands them to ``_init``.
    """

    m: int
    n: int
    tensors: list[Array]
    program: Program = field(init=False, compare=False, repr=False)
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    # True for the bases with the QFT topology, where the gates on the first k
    # qubits of a register are the whole circuit of a k-qubit register. That is
    # what makes resetting the gates on the other qubits to the identity
    # equivalent to a blocked basis; see ``freeze_as_blocked``.
    freezes_to_blocked: ClassVar[bool] = False

    # The family's gate emitter, a plain function ``(m, n) -> gates``. A
    # subclass sets it as ``emit = staticmethod(<family>_gates)``: a function
    # stored bare on a class becomes a method, and calling ``self.emit(m, n)``
    # would then pass the instance as its first argument.
    emit: ClassVar[Callable[[int, int], list[Gate]]]

    def __init__(
        self,
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
    ):
        self._init(self.emit(m, n), m, n, tensors, code, inv_code)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        tree_util.register_pytree_node(cls, _flatten, partial(_unflatten, cls))

    def _init(
        self,
        gates: list[Gate],
        m: int,
        n: int,
        tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
    ) -> None:
        """Compile ``gates`` and set the shared fields; an argument left ``None`` gets the circuit's own."""
        self.m = m
        self.n = n
        compiled, initial = compile_program(gates, m, n)
        self.tensors = list(tensors) if tensors is not None else initial
        # A ``CircuitCode`` that is passed in defines the circuit: the program
        # is read from it, and when it comes alone its counterpart is derived
        # from it (the other direction, the same arithmetic), so the two cannot
        # disagree when a basis is rebuilt with another instance's code.
        default = CircuitCode(compiled)
        if code is None:
            code = _other_direction(inv_code) if isinstance(inv_code, CircuitCode) else default
        if inv_code is None:
            inv_code = _other_direction(code if isinstance(code, CircuitCode) else default)
        self.code = code
        self.inv_code = inv_code
        self.program = getattr(code, "program", compiled)


def _other_direction(code: CircuitCode) -> CircuitCode:
    return replace(code, inverse=not code.inverse)


def _flatten(basis: CircuitBasis):
    # Every attribute of the instance, not only its dataclass fields: a
    # subclass that sets an attribute in its constructor without declaring a
    # field must not lose it on the way through a pytree.
    static = tuple(
        sorted((name, value) for name, value in vars(basis).items() if name != "tensors")
    )
    return tuple(basis.tensors), static


def _unflatten(cls: type[CircuitBasis], static, leaves) -> CircuitBasis:
    basis = object.__new__(cls)
    for name, value in static:
        setattr(basis, name, value)
    basis.tensors = list(leaves)
    return basis


def with_tensors(basis, tensors: Sequence[Array]):
    """A copy of ``basis`` holding ``tensors`` in place of its own; everything else is kept.

    Works on any basis registered as a pytree whose leaves begin with its
    tensors, which is the convention the trainers rely on.
    """
    leaves, treedef = tree_util.tree_flatten(basis)
    return tree_util.tree_unflatten(treedef, list(tensors) + leaves[len(basis.tensors) :])


def program_of(basis) -> Program:
    """The gate program of a basis; for a blocked basis, the one of the circuit it tiles.

    The stored tensor list of either is in that program's order, so its
    ``tensor_indices`` index ``basis.tensors`` directly.
    """
    program = getattr(basis, "program", None)
    return program if program is not None else program_of(basis.inner)


def _cp_indices(basis) -> list[int]:
    return program_of(basis).tensor_indices(kind="CP")


def _with_cp_tensors(basis, phases: Array, tensor_of: Callable[[Array], Array]):
    """A copy of ``basis`` whose ``i``-th controlled-phase tensor is ``tensor_of(phases[i])``."""
    indices = _cp_indices(basis)
    if len(phases) != len(indices):
        raise ValueError(
            f"basis has {len(indices)} controlled-phase gates, got {len(phases)} phases"
        )
    tensors = list(basis.tensors)
    for i, phi in zip(indices, phases):
        tensors[i] = tensor_of(phi).astype(tensors[i].dtype)
    return with_tensors(basis, tensors)


def _read_cp(basis, read: Callable[[Array], Array], shape: tuple[int, ...]) -> Array:
    """``read`` of every controlled-phase tensor, stacked; ``shape`` is what it reads per gate."""
    indices = _cp_indices(basis)
    if not indices:
        return jnp.zeros((0, *shape))
    return jnp.stack([read(basis.tensors[i]) for i in indices])


def cp_phases(basis) -> Array:
    """The angle of every controlled-phase gate, in stored order: the phase-only parameters.

    A compact controlled-phase tensor is ``[[1, 1], [1, exp(i*phi)]]``; this
    reads ``phi`` off each one. ``with_cp_phases`` is the way back. Together
    they are the view a phase-only model trains through: the angles are
    ordinary real parameters, the basis and its circuit stay what they are.

    The view is exact for tensors of that form, which is what every basis
    starts with and what ``with_cp_phases`` writes. The package's Riemannian
    trainers move all four entries of such a tensor around the unit circle
    (the phase manifold is ``U(1)^4``); on a basis trained that way this reads
    one of the four phases, and writing it back resets the other three.
    ``cp_diagonals`` is the view of all four.
    """
    return _read_cp(basis, lambda tensor: jnp.angle(tensor[1, 1]), ())


def with_cp_phases(basis, phases: Array):
    """A copy of ``basis`` whose controlled-phase gates have the angles ``phases``.

    Every other tensor is kept. Traceable: ``phases`` may be a traced array,
    so a loss can be differentiated with respect to the angles through the
    transforms of the returned basis.
    """
    return _with_cp_tensors(basis, phases, controlled_phase_diag)


def cp_diagonals(basis) -> Array:
    """The four phases of every controlled-phase tensor, in stored order: shape ``(gates, 2, 2)``.

    The tensor of a controlled-phase gate holds the diagonal of a two-qubit
    gate. Freeing all four of its entries on the unit circle keeps the gate
    diagonal, and with it the flat modulus of the basis; this reads their
    angles, and ``with_cp_diagonals`` is the way back. Unlike ``cp_phases``
    the view is exact for every tensor with unit-modulus entries, so it also
    round-trips a basis the Riemannian trainers have moved.
    """
    return _read_cp(basis, jnp.angle, (2, 2))


def with_cp_diagonals(basis, phases: Array):
    """A copy of ``basis`` whose controlled-phase tensors are ``exp(i * phases)``. Traceable.

    ``phases`` has the shape ``cp_diagonals`` returns, ``(gates, 2, 2)``.
    """
    phases = jnp.asarray(phases)
    if phases.shape[1:] != (2, 2):
        raise ValueError(f"a gate has phases of shape (2, 2), got {phases.shape[1:]}")
    return _with_cp_tensors(basis, phases, lambda phi: jnp.exp(1j * phi))


@dataclass(frozen=True)
class ParameterView:
    """What a trainer moves in place of a basis's tensors.

    ``read(basis)`` gives the parameters as a list of arrays, ``write(basis,
    params)`` a copy of the basis that holds them, and ``manifolds(params)``
    the manifold each one lives on, or ``None`` to have it read off their
    values as ``classify_manifold`` does. ``write`` is traceable, so a loss
    differentiates through it and the parameters are trained like any others:
    the basis and its circuit stay what they are, and whatever the view does
    not read is not trained.

    The package's views: ``TENSORS``, the tensors themselves, each on the
    manifold its values put it on (what the trainers move when no view is
    named); ``CP_PHASES``, one angle per controlled-phase gate
    (``cp_phases``); ``CP_DIAGONALS``, all four phases of every
    controlled-phase tensor (``cp_diagonals``). The last two are free real
    numbers on ``EuclideanManifold``, read in double precision whatever the
    precision of the tensors, and written back in the tensors' own.
    """

    name: str
    read: Callable = field(repr=False)
    write: Callable = field(repr=False)
    manifolds: Callable = field(repr=False)


def _flat(params: Sequence[Array]) -> list[EuclideanManifold]:
    return [EuclideanManifold(tuple(p.shape)) for p in params]


TENSORS = ParameterView(
    "tensors",
    read=lambda basis: list(basis.tensors),
    write=with_tensors,
    manifolds=lambda params: None,
)
CP_PHASES = ParameterView(
    "cp_phases",
    read=lambda basis: [cp_phases(basis).astype(jnp.float64)],
    write=lambda basis, params: with_cp_phases(basis, *params),
    manifolds=_flat,
)
CP_DIAGONALS = ParameterView(
    "cp_diagonals",
    read=lambda basis: [cp_diagonals(basis).astype(jnp.float64)],
    write=lambda basis, params: with_cp_diagonals(basis, *params),
    manifolds=_flat,
)


def bases_allclose(a, b, *, atol: float = 1e-10) -> bool:
    """Semantic equality across any of the registered basis types.

    Checks same concrete type, same (m, n), and all `tensors` within `atol`
    (rtol=0). `code` / `inv_code` are ignored. Per Julia's design we no
    longer store a separate `inv_tensors` to compare.
    """
    if type(a) is not type(b):
        return False
    if (a.m, a.n) != (b.m, b.n):
        return False
    if len(a.tensors) != len(b.tensors):
        return False
    for x, y in zip(a.tensors, b.tensors):
        if not jnp.allclose(x, y, atol=atol, rtol=0.0):
            return False
    return True
