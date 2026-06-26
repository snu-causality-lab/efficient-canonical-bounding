"""Counterfactual variables (CFVars) and queries.

See docs/01_concepts.md §3 and §4. A CFVar is a base variable V with an
edge-label map σ that assigns a World to every directed edge in the
ancestor-subgraph of V. A CFQuery is a tuple of (CFVar, target_value) events.

Constructors on CFVar:
  - observational(V, G): every edge gets World.observational().
  - standard(V, w, G): every edge gets `w` (the standard potential outcome V_w).
  - nested(V, edge_worlds, G): explicit per-edge mapping. `edge_worlds` must
    cover every directed edge in G.induce(An(V)).directed exactly.

Design note: we do not (yet) perform label inheritance. The user supplies the
full σ explicitly in `nested`. This is intentional for TDD: semantics are
unambiguous. A `with_inheritance` helper can be added later once we have the
structural-expansion machinery in `expansion.py`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from canonred.graph import ADMG
from canonred.types import EdgeKey, Value, VarName
from canonred.worlds import World


class CFVarError(ValueError):
    """Raised for malformed CFVar or CFQuery specs."""


def _ancestor_edges(G: ADMG, base: VarName) -> frozenset[EdgeKey]:
    """Return the set of directed edges of G whose both endpoints are in An(base)."""
    if not G.has_node(base):
        raise CFVarError(f"base {base!r} is not a node of the graph")
    an = G.ancestors({base}, inclusive=True)
    return frozenset((p, c) for (p, c) in G.directed if p in an and c in an)


def _graph_signature(G: ADMG) -> tuple:
    """Small structural fingerprint used to reject CFVars built on another graph."""
    return (
        tuple(G.nodes),
        tuple(sorted(G.directed)),
        tuple(sorted(tuple(sorted(e)) for e in G.bidirected)),
        tuple((v, G.cardinality[v]) for v in G.nodes),
    )


def _validate_world_against_graph(G: ADMG, w: World, *, context: str) -> None:
    for v, val in w.do:
        if not G.has_node(v):
            raise CFVarError(f"{context}: world {w.label!r} intervenes on unknown node {v!r}")
        d = G.cardinality[v]
        if val >= d:
            raise CFVarError(
                f"{context}: world {w.label!r} assigns {v}={val}, "
                f"outside domain [0, {d})"
            )


@dataclass(frozen=True)
class CFVar:
    """Counterfactual variable (V, σ).

    Attributes
    ----------
    base : VarName
    edge_worlds : frozenset of ((parent, child), World)
        Must cover every directed edge in G.induce(An(base)).directed.
    root_world : World
        World attached to the queried root itself. This matters for
        self-interventions such as X_x; if root_world intervenes on `base`, the
        CFVar evaluates to that fixed value without invoking f_base.
    """

    base: VarName
    edge_worlds: frozenset[tuple[EdgeKey, World]]
    root_world: World = field(default_factory=World.observational)
    graph_signature: tuple | None = None

    # --- constructors ---

    @staticmethod
    def observational(base: VarName, G: ADMG) -> "CFVar":
        return CFVar.standard(base, World.observational(), G)

    @staticmethod
    def standard(base: VarName, world: World, G: ADMG) -> "CFVar":
        edges = _ancestor_edges(G, base)
        return CFVar(
            base=base,
            edge_worlds=frozenset((e, world) for e in edges),
            root_world=world,
            graph_signature=_graph_signature(G),
        )

    @staticmethod
    def nested(base: VarName, edge_worlds: Mapping[EdgeKey, World], G: ADMG) -> "CFVar":
        required = _ancestor_edges(G, base)
        provided = frozenset(edge_worlds.keys())
        missing = required - provided
        if missing:
            raise CFVarError(
                f"nested CFVar for base {base!r}: missing edge_worlds for {sorted(missing)}"
            )
        extra = provided - required
        if extra:
            raise CFVarError(
                f"nested CFVar for base {base!r}: edge(s) not in An({base}): {sorted(extra)}"
            )
        return CFVar(
            base=base,
            edge_worlds=frozenset((e, w) for e, w in edge_worlds.items()),
            root_world=World.observational(),
            graph_signature=_graph_signature(G),
        )

    # --- accessors ---

    @property
    def worlds(self) -> frozenset[World]:
        """Distinct worlds appearing in σ."""
        worlds = {w for _, w in self.edge_worlds}
        if self.root_world.intervened or not worlds:
            worlds.add(self.root_world)
        return frozenset(worlds)


@dataclass(frozen=True)
class CFQuery:
    """Joint counterfactual event Q = P(∧_i CFVar_i = target_i) on a given ADMG.

    Carries the graph so that target values can be domain-validated at
    construction time (target_i ∈ Ω_{V_i}). Keep G on the query; do not try
    to share a single query across graphs with different cardinalities.
    """

    events: tuple[tuple[CFVar, Value], ...]
    graph: ADMG
    name: str = "Q"

    def __post_init__(self) -> None:
        for cfvar, target in self.events:
            if not isinstance(cfvar, CFVar):
                raise CFVarError(f"event must be a (CFVar, value) pair; got {cfvar!r}")
            if not isinstance(target, int) or isinstance(target, bool) or target < 0:
                raise CFVarError(
                    f"target value for {cfvar.base!r} must be a non-negative int (got {target!r})"
                )
            if not self.graph.has_node(cfvar.base):
                raise CFVarError(f"base {cfvar.base!r} not in the query's graph")
            expected_sig = _graph_signature(self.graph)
            if cfvar.graph_signature is not None and cfvar.graph_signature != expected_sig:
                raise CFVarError(
                    f"CFVar for {cfvar.base!r} was built on a different graph"
                )
            expected_edges = _ancestor_edges(self.graph, cfvar.base)
            provided_edges = frozenset(edge for edge, _world in cfvar.edge_worlds)
            if provided_edges != expected_edges:
                missing = expected_edges - provided_edges
                extra = provided_edges - expected_edges
                raise CFVarError(
                    f"CFVar for {cfvar.base!r} does not match the query graph "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
                )
            for w in cfvar.worlds:
                _validate_world_against_graph(
                    self.graph, w, context=f"CFVar for {cfvar.base!r}"
                )
            d = self.graph.cardinality[cfvar.base]
            if target >= d:
                raise CFVarError(
                    f"target {target} outside domain of {cfvar.base!r} (cardinality={d})"
                )

    # --- derived properties ---

    @property
    def cfvars(self) -> tuple[CFVar, ...]:
        return tuple(cv for cv, _ in self.events)

    @property
    def worlds(self) -> frozenset[World]:
        """Union of σ-worlds across all CFVars in the query."""
        out: set[World] = set()
        for cv, _ in self.events:
            out.update(cv.worlds)
        return frozenset(out)

    @property
    def intervened_any(self) -> frozenset[VarName]:
        """Union of do-intervened variables across all worlds in the query."""
        out: set[VarName] = set()
        for w in self.worlds:
            out.update(w.intervened)
        return frozenset(out)

    @property
    def labels(self) -> frozenset[str]:
        """Λ(Q): the set of world labels used anywhere in σ."""
        return frozenset(w.label for w in self.worlds)


__all__ = ["CFQuery", "CFVar", "CFVarError"]
