"""Seeded random semi-Markovian ADMG generator.

Used by scripts/random_admg_sweep.py for additional graph-variation checks. Generates ADMGs
over n labelled nodes with independent directed-edge and bidirected-edge
probabilities; guarantees at least one bidirected edge. No hidden semantics —
just a sampler.
"""

from __future__ import annotations

import numpy as np

from canonred.graph import ADMG, admg


def random_admg(
    n: int,
    *,
    rng: np.random.Generator,
    p_dir: float = 0.4,
    p_bidir: float = 0.3,
    min_bidir: int = 1,
    cardinality: int = 2,
    name: str | None = None,
) -> ADMG:
    """Sample a semi-Markovian ADMG on `n` nodes named 'V0'..'V{n-1}'.

    - Node order in `ADMG.nodes` is also a valid topological order (edges
      only go from earlier nodes to later ones).
    - Each ordered pair (i, j) with i<j has a directed edge with probability
      `p_dir`.
    - Each unordered pair {i, j} has a bidirected edge with probability
      `p_bidir`; if fewer than `min_bidir` survive, random pairs are added.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    nodes = tuple(f"V{i}" for i in range(n))

    directed: list[tuple[str, str]] = []
    for j in range(1, n):
        for i in range(j):
            if rng.random() < p_dir:
                directed.append((nodes[i], nodes[j]))

    pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
    bidir: list[set[str]] = []
    for p in pairs:
        if rng.random() < p_bidir:
            bidir.append({*p})
    if len(bidir) < min_bidir:
        remaining = [p for p in pairs if {*p} not in bidir]
        rng.shuffle(remaining)
        for p in remaining[: min_bidir - len(bidir)]:
            bidir.append({*p})

    return admg(
        nodes=nodes,
        cardinality=cardinality,
        directed=directed,
        bidirected=[tuple(e) for e in bidir],
    )


__all__ = ["random_admg"]
