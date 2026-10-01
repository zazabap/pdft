"""Simple circuit topology visualization.

Mirror of the *essentials* of upstream src/circuit_visualization.jl (867
lines). Renders a left-to-right wire diagram with one-qubit gates as boxes
and two-qubit gates as vertical links, read from the program the basis
keeps. Not intended to be publication-quality; use upstream for that.
Requires the `plot` extra.
"""

from __future__ import annotations

from pathlib import Path

from ._figure import require_matplotlib, save


def plot_circuit(
    basis,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
):
    """Render `basis`'s circuit: one column per gate, in the order the gates act.

    Horizontal wires for each qubit, a box for a one-qubit gate, a vertical
    link for a two-qubit gate (round ends for a diagonal controlled phase,
    square ends for a dense or controlled-rotation gate). Deliberately
    simple; full-fidelity rendering with gate labels is upstream's domain.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt

    from ..bases.core import program_of

    program = program_of(basis)
    n_qubits = program.m + program.n
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(program.steps)), 2 + 0.4 * n_qubits))

    for q in range(n_qubits):
        ax.axhline(y=q, color="lightgray", linewidth=1, zorder=0)

    counts: dict[str, int] = {}
    for x, (kind, qubits) in enumerate(program.steps):
        counts[kind] = counts.get(kind, 0) + 1
        ys = [q - 1 for q in qubits]
        if len(ys) == 1:
            ax.scatter([x], ys, marker="s", s=200, color="steelblue", zorder=2)
        else:
            ax.plot([x, x], ys, color="salmon", linewidth=2, zorder=1)
            ax.scatter(
                [x, x], ys, marker="o" if kind == "CP" else "s", s=80, color="salmon", zorder=2
            )

    ax.set_xlim(-1, max(2, len(program.steps)))
    ax.set_ylim(-1, n_qubits)
    ax.set_xlabel("gate order")
    ax.set_ylabel("qubit")
    ax.set_yticks(range(n_qubits))
    ax.set_yticklabels([f"q{q + 1}" for q in range(n_qubits)])
    if title is None:
        gates = " + ".join(f"{count} {kind}" for kind, count in counts.items())
        title = f"{type(basis).__name__}(m={basis.m}, n={basis.n}): {gates}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    save(fig, output_path)
    return fig
