"""ADMG (acyclic directed mixed graph) primitives.

See docs/01_concepts.md §1 for the formalism. Every method here is pure
(returns new objects), and the ADMG class is
immutable and hashable.
"""

import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from canonred.types import EdgeKey, VarName


class GraphError(ValueError):
    """Raised for malformed ADMG specs."""


def _freeze_bidir(bidir: Iterable[Iterable[VarName]]) -> frozenset[frozenset[VarName]]:
    out: set[frozenset[VarName]] = set()
    for e in bidir:
        s = frozenset(e)
        if len(s) != 2:
            raise GraphError(f"bidirected edge must be a 2-element set, got {tuple(e)!r}")
        out.add(s)
    return frozenset(out)


def _freeze_directed(directed: Iterable[EdgeKey]) -> frozenset[EdgeKey]:
    out: set[EdgeKey] = set()
    for e in directed:
        if len(e) != 2:
            raise GraphError(f"directed edge must be a 2-tuple, got {e!r}")
        p, c = e
        if p == c:
            raise GraphError(f"self-loop not allowed: {e!r}")
        out.add((p, c))
    return frozenset(out)


@dataclass(frozen=True)
class ADMG:
    """A semi-Markovian ADMG over a finite observed-variable set.

    Parameters
    ----------
    nodes : tuple of VarName
        Unique names of observed nodes. Order establishes a default topological
        tie-breaker but is NOT assumed to be a valid topological order.
    card_pairs : tuple of (VarName, int)
        Hashable encoding of the cardinality map. One (node, domain size) pair
        per node in `nodes`, in the same order. Access via the `cardinality`
        property.
    directed : iterable of (parent, child)
        Directed edges. Must define a DAG on `nodes`.
    bidirected : iterable of 2-element sets of VarName
        Undirected unobserved-confounder edges.

    Invariants (checked in __post_init__):
      - every node in every edge is in `nodes`;
      - no self-loops;
      - directed subgraph is acyclic;
      - cardinalities >= 2 for all nodes;
      - `card_pairs` covers exactly `nodes` (one pair per node).

    Use the `admg(...)` builder below for user-friendly construction from a
    dict.
    """

    nodes: tuple[VarName, ...]
    card_pairs: tuple[tuple[VarName, int], ...]
    directed: frozenset[EdgeKey] = field(default_factory=frozenset)
    bidirected: frozenset[frozenset[VarName]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if len(set(self.nodes)) != len(self.nodes):
            raise GraphError(f"duplicate node names in {self.nodes!r}")
        node_set = set(self.nodes)

        object.__setattr__(self, "directed", _freeze_directed(self.directed))
        object.__setattr__(self, "bidirected", _freeze_bidir(self.bidirected))

        for p, c in self.directed:
            if p not in node_set:
                raise GraphError(f"directed-edge source {p!r} is not a declared node")
            if c not in node_set:
                raise GraphError(f"directed-edge target {c!r} is not a declared node")
        for e in self.bidirected:
            for v in e:
                if v not in node_set:
                    raise GraphError(f"bidirected-edge node {v!r} is not declared")

        # card_pairs validation
        seen: set[VarName] = set()
        for v, d in self.card_pairs:
            if v in seen:
                raise GraphError(f"duplicate cardinality entry for {v!r}")
            seen.add(v)
            if v not in node_set:
                raise GraphError(f"cardinality has key not in nodes: {v!r}")
            if not isinstance(d, int) or isinstance(d, bool) or d < 2:
                raise GraphError(f"cardinality[{v!r}] = {d!r}; must be int >= 2")
        missing = node_set - seen
        if missing:
            raise GraphError(f"missing cardinality for node(s): {sorted(missing)}")

        # DAG check on directed subgraph
        if not self._is_dag():
            raise GraphError("directed subgraph has a cycle")

    @property
    def cardinality(self) -> Mapping[VarName, int]:
        """Read-only {node: domain_size} map."""
        return types.MappingProxyType(dict(self.card_pairs))

    # ------------------------------------------------------------------
    # Basic introspection
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.nodes)

    def has_node(self, v: VarName) -> bool:
        return v in self.cardinality

    def parents(self, v: VarName) -> tuple[VarName, ...]:
        """Strict directed parents of v, returned in the order of self.nodes."""
        pa = {p for (p, c) in self.directed if c == v}
        return tuple(n for n in self.nodes if n in pa)

    def children(self, v: VarName) -> tuple[VarName, ...]:
        ch = {c for (p, c) in self.directed if p == v}
        return tuple(n for n in self.nodes if n in ch)

    def bidir_neighbors(self, v: VarName) -> tuple[VarName, ...]:
        nbrs: set[VarName] = set()
        for e in self.bidirected:
            if v in e:
                nbrs.update(e)
        nbrs.discard(v)
        return tuple(n for n in self.nodes if n in nbrs)

    # ------------------------------------------------------------------
    # Ancestors / descendants / topological order
    # ------------------------------------------------------------------

    def ancestors(self, S: Iterable[VarName], inclusive: bool = True) -> frozenset[VarName]:
        """Return the set of ancestors of the given set (directed-only).

        `inclusive=True` includes S itself (matching the paper's `An(Y)` convention).
        """
        seen: set[VarName] = set(S) if inclusive else set()
        frontier: list[VarName] = list(S)
        while frontier:
            v = frontier.pop()
            for p in self.parents(v):
                if p not in seen:
                    seen.add(p)
                    frontier.append(p)
        return frozenset(seen)

    def descendants(self, S: Iterable[VarName], inclusive: bool = True) -> frozenset[VarName]:
        seen: set[VarName] = set(S) if inclusive else set()
        frontier: list[VarName] = list(S)
        while frontier:
            v = frontier.pop()
            for c in self.children(v):
                if c not in seen:
                    seen.add(c)
                    frontier.append(c)
        return frozenset(seen)

    def topo_order(self) -> tuple[VarName, ...]:
        """A topological order on the directed subgraph (Kahn's algorithm).

        Tie-breaker: node order as declared in `self.nodes`.
        """
        indeg = {v: 0 for v in self.nodes}
        for _p, c in self.directed:
            indeg[c] += 1

        ready = [v for v in self.nodes if indeg[v] == 0]
        order: list[VarName] = []
        while ready:
            v = ready.pop(0)  # preserve declaration order among ready nodes
            order.append(v)
            for c in self.children(v):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            # maintain ready in declaration order
            ready.sort(key=self.nodes.index)

        if len(order) != self.n:
            raise GraphError("graph contains a cycle")
        return tuple(order)

    def _is_dag(self) -> bool:
        indeg = {v: 0 for v in self.nodes}
        for _p, c in self.directed:
            indeg[c] += 1
        ready = [v for v in self.nodes if indeg[v] == 0]
        seen = 0
        while ready:
            v = ready.pop()
            seen += 1
            for c in self.children(v):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        return seen == self.n

    def has_directed_path(self, src: VarName, dst: VarName) -> bool:
        """Return True if there exists a directed path src -> ... -> dst in G."""
        if src == dst:
            return True
        return dst in self.descendants({src}, inclusive=False)

    # ------------------------------------------------------------------
    # C-components and c-cliques
    # ------------------------------------------------------------------

    def c_components(self) -> tuple[frozenset[VarName], ...]:
        """Partition of nodes by bidirected-connectivity (transitive closure).

        Returns components sorted by first-declared-node order.
        """
        # Union-find over self.bidirected.
        parent = {v: v for v in self.nodes}

        def find(v: VarName) -> VarName:
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        def union(a: VarName, b: VarName) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for e in self.bidirected:
            it = iter(e)
            a = next(it)
            b = next(it)
            union(a, b)

        buckets: dict[VarName, set[VarName]] = {}
        for v in self.nodes:
            r = find(v)
            buckets.setdefault(r, set()).add(v)

        comps = sorted(
            (frozenset(bucket) for bucket in buckets.values()),
            key=lambda s: min(self.nodes.index(v) for v in s),
        )
        return tuple(comps)

    def c_component_of(self, v: VarName) -> frozenset[VarName]:
        for comp in self.c_components():
            if v in comp:
                return comp
        raise GraphError(f"{v!r} is not in this ADMG")

    def c_cliques(self) -> tuple[frozenset[VarName], ...]:
        """Return all maximal c-cliques (appendix Def. A.2).

        A c-clique is a subset K in which every pair is directly joined by a
        bidirected edge, and K is maximal w.r.t. that property. Singletons
        are excluded unless the node has no bidirected neighbors (in which case
        the singleton is a maximal clique by vacuous truth).

        We enumerate maximal cliques on the bidirected graph via Bron-Kerbosch
        with pivoting.
        """
        # Adjacency on bidirected graph
        adj: dict[VarName, set[VarName]] = {v: set() for v in self.nodes}
        for e in self.bidirected:
            a, b = tuple(e)
            adj[a].add(b)
            adj[b].add(a)

        cliques: list[frozenset[VarName]] = []

        def bron_kerbosch(R: set[VarName], P: set[VarName], X: set[VarName]) -> None:
            if not P and not X:
                if R:
                    cliques.append(frozenset(R))
                return
            # pivot u maximizing |P ∩ N(u)|
            pivot = max(P | X, key=lambda u: len(P & adj[u]))
            for v in P - adj[pivot]:
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}

        bron_kerbosch(set(), set(self.nodes), set())

        # Include every node with no bidirected neighbor as a trivial singleton clique.
        covered: set[VarName] = set().union(*cliques) if cliques else set()
        for v in self.nodes:
            if v not in covered:
                cliques.append(frozenset({v}))

        cliques.sort(
            key=lambda s: (-len(s), min(self.nodes.index(v) for v in s)),
        )
        return tuple(cliques)

    # ------------------------------------------------------------------
    # Subgraph and mutilation
    # ------------------------------------------------------------------

    def induce(self, S: Iterable[VarName]) -> "ADMG":
        """Return the vertex-induced subgraph on S, preserving declaration order."""
        S_set = frozenset(S)
        for v in S_set:
            if v not in self.cardinality:
                raise GraphError(f"induce: {v!r} is not in this ADMG")
        new_nodes = tuple(n for n in self.nodes if n in S_set)
        new_card = tuple((n, self.cardinality[n]) for n in new_nodes)
        new_dir = frozenset((p, c) for (p, c) in self.directed if p in S_set and c in S_set)
        new_bi = frozenset(e for e in self.bidirected if e.issubset(S_set))
        return ADMG(new_nodes, new_card, new_dir, new_bi)

    def mutilate_incoming(self, X: Iterable[VarName]) -> "ADMG":
        """G_{\\overline{X}}: delete edges (directed and bidirected) incident on X.

        Removes:
          - every directed edge p -> v with v in X;
          - every bidirected edge incident on any X.
        """
        X_set = frozenset(X)
        for v in X_set:
            if v not in self.cardinality:
                raise GraphError(f"mutilate_incoming: {v!r} is not in this ADMG")
        new_dir = frozenset((p, c) for (p, c) in self.directed if c not in X_set)
        new_bi = frozenset(e for e in self.bidirected if e.isdisjoint(X_set))
        return ADMG(self.nodes, self.card_pairs, new_dir, new_bi)

    def mutilate_outgoing(self, X: Iterable[VarName]) -> "ADMG":
        """G_{\\underline{X}}: delete directed edges out of X (bidirected unchanged)."""
        X_set = frozenset(X)
        for v in X_set:
            if v not in self.cardinality:
                raise GraphError(f"mutilate_outgoing: {v!r} is not in this ADMG")
        new_dir = frozenset((p, c) for (p, c) in self.directed if p not in X_set)
        return ADMG(self.nodes, self.card_pairs, new_dir, self.bidirected)

    # ------------------------------------------------------------------
    # Cardinality helpers
    # ------------------------------------------------------------------

    def parent_domain_size(self, v: VarName) -> int:
        """|Omega_{pa(v)}| = product of parents' cardinalities (1 if empty)."""
        size = 1
        for p in self.parents(v):
            size *= self.cardinality[p]
        return size

    def response_function_count(self, v: VarName) -> int:
        """|F_V| = d_V ^ |Omega_{pa(V)}|."""
        return self.cardinality[v] ** self.parent_domain_size(v)

    def lifted_canonical_size(self) -> int:
        """|Omega_full| = product over V of response_function_count(V)."""
        size = 1
        for v in self.nodes:
            size *= self.response_function_count(v)
        return size

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dir_str = ", ".join(f"{p}->{c}" for p, c in sorted(self.directed))
        bi_str = ", ".join(
            "<->".join(sorted(e, key=self.nodes.index)) for e in sorted(self.bidirected, key=lambda s: sorted(s))
        )
        card_str = ", ".join(f"{v}:{d}" for v, d in self.card_pairs)
        return (
            f"ADMG(nodes={self.nodes!r}, cardinality={{{card_str}}}, "
            f"directed=[{dir_str}], bidirected=[{bi_str}])"
        )


# ----------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------


def admg(
    *,
    nodes: Iterable[VarName],
    cardinality: Mapping[VarName, int] | int = 2,
    directed: Iterable[EdgeKey] = (),
    bidirected: Iterable[Iterable[VarName]] = (),
) -> ADMG:
    """User-friendly builder.

    If `cardinality` is an int, apply it to every node.
    """
    nodes_t = tuple(nodes)
    if isinstance(cardinality, int):
        pairs = tuple((v, cardinality) for v in nodes_t)
    else:
        # Preserve the node ordering when building the pair tuple.
        extra = set(cardinality).difference(nodes_t)
        if extra:
            raise GraphError(f"cardinality has keys not in nodes: {sorted(extra)}")
        pairs = tuple((v, cardinality[v]) for v in nodes_t if v in cardinality)
    return ADMG(
        nodes=nodes_t,
        card_pairs=pairs,
        directed=frozenset(tuple(e) for e in directed),
        bidirected=frozenset(frozenset(e) for e in bidirected),
    )


__all__ = ["ADMG", "GraphError", "admg"]
