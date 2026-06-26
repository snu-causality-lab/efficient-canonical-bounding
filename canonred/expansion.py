"""Structural expansion of a CFQuery into its edge-split graph.

See docs/01_concepts.md §3.5 (G*) and §10.1 (S_V(Q)).

The edge-split graph G* is constructed by unrolling each CFVar's structural
recursion. A non-fixed node in G* corresponds to one invocation of the base
variable's response function; a fixed node corresponds to a do-intervened
copy (not an invocation of f_V). S_V(Q) is the set of distinct labeled
parent-value tuples at which f_V is invoked across all of Q's CFVars.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

from canonred.counterfactual import CFQuery, CFVar, CFVarError
from canonred.graph import ADMG
from canonred.types import Label, Value, VarName
from canonred.worlds import World


class ConflictError(ValueError):
    """Raised when a query's sigma-map is not L_{2.5}-realizable.

    Currently detects: two distinct World objects sharing the same label
    (same name, different intervention map).
    """


@dataclass(frozen=True)
class SplitNode:
    """A vertex of the edge-split graph G*.

    Identity
    --------
    Two SplitNodes are equal iff (base, label) match. Internal nodes reuse the
    label from their incoming edge's World; CFVar roots use a per-CFVar marker
    label so that roots from distinct CFVars do not accidentally merge.
    """

    cfvar_index: int        # -1 for internal nodes; i >= 0 for the root of CFVar i
    base: VarName
    label: Label


@dataclass(frozen=True)
class SplitGraph:
    """Edge-split graph G* plus fixed-value annotations.

    - `nodes`: all SplitNodes reachable by expansion.
    - `directed`: edges (p_split, c_split) induced by the original graph's
      directed edges under sigma.
    - `fixed_values`: map from fixed SplitNodes (intervened copies) to their
      do-value. Non-fixed SplitNodes correspond to f_V invocations.
    - `roots`: CFVar roots, one per event in the query (same order).
    """

    nodes: frozenset[SplitNode]
    directed: frozenset[tuple[SplitNode, SplitNode]]
    fixed_values: Mapping[SplitNode, Value]
    roots: tuple[SplitNode, ...]
    original_graph: ADMG
    original_query: CFQuery

    @property
    def non_fixed(self) -> frozenset[SplitNode]:
        return frozenset(n for n in self.nodes if n not in self.fixed_values)

    def parents_in_star(self, sn: SplitNode) -> tuple[SplitNode, ...]:
        """Parents of `sn` in G* in a deterministic order (by original graph nodes)."""
        G = self.original_graph
        parents = [p for (p, c) in self.directed if c == sn]
        return tuple(sorted(parents, key=lambda s: (G.nodes.index(s.base), s.label)))


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------


def _collect_worlds(Q: CFQuery) -> dict[Label, World]:
    """Return {label: World} across all CFVars; raise ConflictError on label collisions."""
    by_label: dict[Label, World] = {}
    for cfvar, _ in Q.events:
        for w in cfvar.worlds:
            prev = by_label.get(w.label)
            if prev is None:
                by_label[w.label] = w
            elif prev != w:
                raise ConflictError(
                    f"label {w.label!r} is bound to two incompatible worlds: "
                    f"{sorted(prev.do)} vs {sorted(w.do)}"
                )
    return by_label


def _edge_world_map(cfvar: CFVar) -> Mapping[tuple[VarName, VarName], World]:
    """Flatten CFVar.edge_worlds (frozenset of (edge, world)) into a dict for lookup."""
    return {edge: world for edge, world in cfvar.edge_worlds}


# ----------------------------------------------------------------------
# Build G*
# ----------------------------------------------------------------------


def edge_split_graph(G: ADMG, Q: CFQuery) -> SplitGraph:
    """Construct the edge-split graph G* of Q on G.

    Each CFVar contributes a subtree rooted at its base. A root SplitNode gets
    a unique per-CFVar label (`"__cfvar<i>__"`). Internal SplitNodes use the
    World's label directly; they merge across CFVars when (base, label) match.

    Raises `ConflictError` if the query is not label-consistent.
    """
    if G != Q.graph:
        raise CFVarError("edge_split_graph received a graph different from Q.graph")
    _collect_worlds(Q)  # raises on label collision

    nodes: set[SplitNode] = set()
    edges: set[tuple[SplitNode, SplitNode]] = set()
    fixed: dict[SplitNode, Value] = {}
    roots: list[SplitNode] = []

    for i, (cfvar, _target) in enumerate(Q.events):
        sigma = _edge_world_map(cfvar)
        root = SplitNode(cfvar_index=i, base=cfvar.base, label=f"__cfvar{i}__")
        roots.append(root)
        nodes.add(root)
        if cfvar.base in cfvar.root_world.intervened:
            val = cfvar.root_world.value_of(cfvar.base)
            assert val is not None
            fixed[root] = val
        else:
            _expand(cfvar.base, root, sigma, G, nodes, edges, fixed)

    return SplitGraph(
        nodes=frozenset(nodes),
        directed=frozenset(edges),
        fixed_values=dict(fixed),
        roots=tuple(roots),
        original_graph=G,
        original_query=Q,
    )


def _expand(
    cur_base: VarName,
    cur_split: SplitNode,
    sigma: Mapping[tuple[VarName, VarName], World],
    G: ADMG,
    nodes: set[SplitNode],
    edges: set[tuple[SplitNode, SplitNode]],
    fixed: dict[SplitNode, Value],
) -> None:
    """Expand the subtree under `cur_split` (base=cur_base) following sigma."""
    for p in G.parents(cur_base):
        edge = (p, cur_base)
        if edge not in sigma:
            raise CFVarError(
                f"CFVar expansion for {cur_base!r} is missing sigma edge {edge!r}"
            )
        w = sigma[edge]
        p_split = SplitNode(cfvar_index=-1, base=p, label=w.label)

        newly_added = p_split not in nodes
        nodes.add(p_split)
        edges.add((p_split, cur_split))

        if p in w.intervened:
            # Fixed copy: record value, do not recurse.
            val = w.value_of(p)
            assert val is not None  # invariant of `intervened`
            if p_split in fixed and fixed[p_split] != val:
                # Impossible because label collisions are ruled out upstream.
                raise ConflictError(
                    f"fixed-value conflict at {p_split}: {fixed[p_split]} vs {val}"
                )
            fixed[p_split] = val
            # No recursion — fixed nodes have no ancestors in G*.
        else:
            if newly_added:
                _expand(p, p_split, sigma, G, nodes, edges, fixed)


# ----------------------------------------------------------------------
# Enumerate S_V(Q)
# ----------------------------------------------------------------------


def enumerate_contexts(
    G: ADMG, Q: CFQuery
) -> dict[VarName, frozenset[tuple[tuple[VarName, Value, Label], ...]]]:
    """Return S_V(Q) for every node V in G as a frozenset of labeled parent tuples.

    Each tuple is sorted by parent-name for deterministic comparison. An empty
    tuple () means V was invoked with no parent arguments (root node).
    """
    gstar = edge_split_graph(G, Q)
    out: dict[VarName, set[tuple[tuple[VarName, Value, Label], ...]]] = {
        v: set() for v in G.nodes
    }
    for sn in gstar.non_fixed:
        parents = gstar.parents_in_star(sn)
        # For each non-fixed parent, iterate over its domain; fixed parents
        # contribute their unique fixed value.
        parent_bundles: list[list[tuple[VarName, Value, Label]]] = []
        for p_sn in parents:
            p_label = p_sn.label
            p_base = p_sn.base
            if p_sn in gstar.fixed_values:
                parent_bundles.append([(p_base, gstar.fixed_values[p_sn], p_label)])
            else:
                d = G.cardinality[p_base]
                parent_bundles.append([(p_base, v, p_label) for v in range(d)])
        if not parent_bundles:
            out[sn.base].add(())
        else:
            for combo in itertools.product(*parent_bundles):
                ordered = tuple(sorted(combo, key=lambda e: e[0]))
                out[sn.base].add(ordered)
    return {v: frozenset(s) for v, s in out.items()}


__all__ = [
    "ConflictError",
    "SplitGraph",
    "SplitNode",
    "edge_split_graph",
    "enumerate_contexts",
]
