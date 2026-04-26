"""Riemannian optimizers + the optimize() dispatcher.

Public surface: RiemannianGD, RiemannianAdam, AbstractRiemannianOptimizer
union, and optimize().
"""

from .adam import RiemannianAdam
from .gd import RiemannianGD
from .loop import optimize

# Union of supported optimizer types for dispatch.
AbstractRiemannianOptimizer = RiemannianGD | RiemannianAdam

__all__ = [
    "AbstractRiemannianOptimizer",
    "RiemannianAdam",
    "RiemannianGD",
    "optimize",
]
