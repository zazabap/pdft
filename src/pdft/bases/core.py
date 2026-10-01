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
from dataclasses import dataclass, field, fields
from functools import partial
from typing import ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import tree_util

from ..circuit.builder import CircuitCode, Gate, Program, apply_circuit, compile_program

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
    """What follows from ``m``, ``n``, ``tensors``, ``code`` and ``inv_code``, however a basis holds them."""

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
        return apply_circuit(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        """``conj(tensors)`` through ``inv_code``, exactly like Julia's
        ``basis.inverse_code(conj.(basis.tensors)..., ...)``."""
        return apply_circuit(
            [jnp.conj(t) for t in self.tensors], self.inv_code, self.m, self.n, pic
        )


@dataclass(init=False)
class CircuitBasis(BasisTransforms):
    """A basis that is a gate circuit on ``m`` row qubits and ``n`` column qubits.

    ``tensors`` holds one tensor per gate, Hadamards first (Julia's
    ``perm_vec``); ``program`` says which gate each one is and when it acts.
    ``code`` and ``inv_code`` are the program's forward and inverse appliers;
    they compare by program, so they and ``program`` are left out of the
    generated ``__eq__`` and ``repr``. Use ``bases_allclose`` for semantic
    comparison.

    Every subclass is registered as a JAX pytree: the leaves are ``tensors``,
    in order, and everything else is aux data. That is the contract
    ``train_basis`` relies on. Unflattening restores the fields without
    running the constructor, so no gate list is rebuilt.

    A family whose circuit depends on ``m`` and ``n`` alone sets ``emit`` to
    its gate emitter and is done. One with options of its own defines a
    constructor that emits its gates and hands them to ``_init``.
    """

    m: int
    n: int
    tensors: list[Array]
    program: Program = field(compare=False, repr=False)
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    # True for the bases with the QFT topology, where the gates on the first k
    # qubits of a register are the whole circuit of a k-qubit register. That is
    # what makes resetting the gates on the other qubits to the identity
    # equivalent to a blocked basis; see ``freeze_as_blocked``.
    freezes_to_blocked: ClassVar[bool] = False

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
        self.program, initial = compile_program(gates, m, n)
        self.tensors = list(tensors) if tensors is not None else initial
        self.code = code if code is not None else CircuitCode(self.program)
        self.inv_code = (
            inv_code if inv_code is not None else CircuitCode(self.program, inverse=True)
        )


def _flatten(basis: CircuitBasis):
    static = tuple((f.name, getattr(basis, f.name)) for f in fields(basis) if f.name != "tensors")
    return tuple(basis.tensors), static


def _unflatten(cls: type[CircuitBasis], static, leaves) -> CircuitBasis:
    basis = object.__new__(cls)
    for name, value in static:
        setattr(basis, name, value)
    basis.tensors = list(leaves)
    return basis


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
