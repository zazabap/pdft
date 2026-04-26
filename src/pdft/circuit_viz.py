"""Simple circuit topology visualization.

Mirror of the *essentials* of upstream src/circuit_visualization.jl (867
lines). Renders a left-to-right wire diagram with H gates as boxes and
CP gates as vertical links. Not intended to be publication-quality; use
upstream for that. Requires the `plot` extra.
"""

from __future__ import annotations

from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for pdft.circuit_viz. Install with: pip install pdft[plot]"
        ) from e


def _topology_from_basis(basis):
    """Return (n_qubits, gates) where gates are (kind, qubits, phase)."""
    from .circuit.builder import HADAMARD
    import numpy as np

    m, n = basis.m, basis.n
    # Re-derive the gate sequence by reading tensor values: Hadamards sort
    # first in basis.tensors (Phase 1/2 convention), so the remainder are CPs.
    # We don't have the original gate sequence metadata, so this is
    # approximate — we render all H tensors as individual 1-qubit gates
    # across all qubits and CPs as generic "two-qubit link" markers spaced
    # along the X axis. For full topology, use upstream Julia viz.
    H_np = np.asarray(HADAMARD)
    tensors = [np.asarray(t) for t in basis.tensors]
    hadamard_count = sum(1 for t in tensors if np.allclose(t, H_np, atol=1e-12))
    cp_count = len(tensors) - hadamard_count
    return m + n, {"n_hadamards": hadamard_count, "n_cps": cp_count, "n_qubits": m + n}


def plot_circuit(
    basis,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
):
    """Render a schematic of `basis`'s circuit structure.

    Deliberately simple: horizontal wires for each qubit, marker positions
    for Hadamards and CPs. Full-fidelity circuit rendering with gate labels
    and coloring is upstream's domain.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    n_qubits, info = _topology_from_basis(basis)
    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * n_qubits))

    # Draw qubit wires
    for q in range(n_qubits):
        ax.axhline(y=q, color="lightgray", linewidth=1, zorder=0)

    # Place Hadamard markers (simple: one per qubit in the first column)
    for q in range(n_qubits):
        ax.scatter([0], [q], marker="s", s=200, color="steelblue", zorder=2)
    # Place CP markers (a single aggregate column; real positions would need
    # the explicit gate sequence which we don't preserve in Basis)
    if info["n_cps"] > 0:
        for i in range(info["n_cps"]):
            ax.scatter([i + 1], [0.5], marker="o", s=80, color="salmon", zorder=2)

    ax.set_xlim(-1, max(2, info["n_cps"] + 1))
    ax.set_ylim(-1, n_qubits)
    ax.set_xlabel("gate order")
    ax.set_ylabel("qubit")
    ax.set_yticks(range(n_qubits))
    ax.set_yticklabels([f"q{q + 1}" for q in range(n_qubits)])
    if title is None:
        title = f"{type(basis).__name__}(m={basis.m}, n={basis.n}): {info['n_hadamards']} H + {info['n_cps']} CP"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    if output_path is not None:
        fig.savefig(str(output_path), bbox_inches="tight", dpi=120)
    return fig
