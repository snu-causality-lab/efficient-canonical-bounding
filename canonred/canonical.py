"""Canonical state space: response functions, enumeration, query evaluation.

See docs/01_concepts.md §5 and §6. A CanonicalState u is a map
VarName -> ResponseFn (one per node). Omega_full is the product of all nodes'
response-function spaces. Evaluating a CFVar under u = propagate values
through the edge-split graph of the CFVar, looking up f_V from u at each
non-fixed SplitNode.
"""

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import prod

from canonred.counterfactual import CFQuery, CFVar
from canonred.expansion import SplitGraph, SplitNode, edge_split_graph
from canonred.graph import ADMG
from canonred.types import Value, VarName


class CanonicalError(ValueError):
    """Raised for malformed response functions or state evaluations."""


# ----------------------------------------------------------------------
# ResponseFn
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseFn:
    """A deterministic structural function f_V : Omega_pa(V) -> Omega_V.

    The `table` is the flat row-major encoding: for parents with cardinalities
    (d_1, ..., d_k) and values (v_1, ..., v_k), the flat index is

        idx = v_1 * d_2 * ... * d_k + v_2 * d_3 * ... * d_k + ... + v_k.

    For a root (no parents), the table has exactly one entry.
    """

    var: VarName
    parent_cards: tuple[int, ...]
    codomain: int
    table: tuple[Value, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.codomain, int) or isinstance(self.codomain, bool) or self.codomain < 2:
            raise ValueError(f"codomain must be int >= 2 (got {self.codomain!r})")
        for d in self.parent_cards:
            if not isinstance(d, int) or d < 2:
                raise ValueError(f"parent_cards entries must be ints >= 2 (got {self.parent_cards!r})")
        expected = prod(self.parent_cards) if self.parent_cards else 1
        if len(self.table) != expected:
            raise ValueError(
                f"table length {len(self.table)} != prod(parent_cards)={expected} for {self.var!r}"
            )
        for v in self.table:
            if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= self.codomain:
                raise ValueError(
                    f"table entry {v!r} for {self.var!r} is outside [0, {self.codomain})"
                )

    def __call__(self, *pa_values: Value) -> Value:
        if len(pa_values) != len(self.parent_cards):
            raise ValueError(
                f"ResponseFn({self.var!r}) called with {len(pa_values)} args; "
                f"expected {len(self.parent_cards)}"
            )
        idx = 0
        for v, d in zip(pa_values, self.parent_cards, strict=True):
            if v < 0 or v >= d:
                raise ValueError(f"parent value {v} outside domain [0,{d}) for {self.var!r}")
            idx = idx * d + v
        return self.table[idx]


# ----------------------------------------------------------------------
# CanonicalState
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalState:
    """A single element u of Omega_full: mapping base name -> ResponseFn.

    The state is graph-free — it doesn't validate coverage of any particular
    ADMG at construction time. Callers that need full coverage (e.g.,
    enumerate_states) construct complete maps directly; ad-hoc constructions
    may have missing keys, which raise on use.
    """

    funcs: Mapping[VarName, ResponseFn]

    def __post_init__(self) -> None:
        # Freeze the mapping by converting to a plain dict view; unhashable
        # by design (Mapping may be any dict-like). We manually implement
        # equality and hashing so CanonicalState is usable in tests.
        object.__setattr__(self, "funcs", dict(self.funcs))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CanonicalState):
            return NotImplemented
        return dict(self.funcs) == dict(other.funcs)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.funcs.items())))


# ----------------------------------------------------------------------
# Enumeration of Omega_full
# ----------------------------------------------------------------------


def state_cardinality(G: ADMG) -> int:
    """|Omega_full| = prod_V d_V^{prod_p d_p}."""
    return G.lifted_canonical_size()


def _enumerate_response_fns(G: ADMG, v: VarName) -> Iterator[ResponseFn]:
    """Yield every possible response function for V over G.parents(V)."""
    parent_cards = tuple(G.cardinality[p] for p in G.parents(v))
    codomain = G.cardinality[v]
    n_entries = prod(parent_cards) if parent_cards else 1
    for tbl in itertools.product(range(codomain), repeat=n_entries):
        yield ResponseFn(var=v, parent_cards=parent_cards, codomain=codomain, table=tbl)


def enumerate_states(G: ADMG) -> Iterator[CanonicalState]:
    """Yield each CanonicalState in Omega_full exactly once.

    Order: nodes in G's topological order; for each node, tables in
    lexicographic order.
    """
    topo = G.topo_order()
    per_node: list[list[ResponseFn]] = [list(_enumerate_response_fns(G, v)) for v in topo]
    for combo in itertools.product(*per_node):
        yield CanonicalState(funcs=dict(zip(topo, combo, strict=True)))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def _topo_order_split_graph(gstar: SplitGraph) -> list[SplitNode]:
    """Topological order of G* by Kahn's algorithm."""
    indeg: dict[SplitNode, int] = {n: 0 for n in gstar.nodes}
    for (_p, c) in gstar.directed:
        indeg[c] += 1
    # Stable tie-breaker: use original graph's node order + label.
    G = gstar.original_graph

    def key(n: SplitNode) -> tuple[int, str, int]:
        return (G.nodes.index(n.base), n.label, n.cfvar_index)

    ready = sorted([n for n in gstar.nodes if indeg[n] == 0], key=key)
    order: list[SplitNode] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        successors = [c for (p, c) in gstar.directed if p == n]
        for c in successors:
            indeg[c] -= 1
            if indeg[c] == 0:
                # Insert keeping `ready` sorted.
                ready.append(c)
                ready.sort(key=key)
    if len(order) != len(gstar.nodes):
        raise CanonicalError("edge-split graph is not a DAG")
    return order


def _evaluate_split_graph(u: CanonicalState, gstar: SplitGraph) -> dict[SplitNode, Value]:
    """Return the value of every SplitNode in gstar under state `u`."""
    vals: dict[SplitNode, Value] = {}
    fixed = gstar.fixed_values
    for sn in _topo_order_split_graph(gstar):
        if sn in fixed:
            vals[sn] = fixed[sn]
            continue
        if sn.base not in u.funcs:
            raise CanonicalError(
                f"state is missing a ResponseFn for base {sn.base!r}"
            )
        f = u.funcs[sn.base]
        parents = gstar.parents_in_star(sn)
        # Order parents to match G.parents(sn.base): the ResponseFn expects
        # arguments in that order.
        G = gstar.original_graph
        expected_parent_bases = G.parents(sn.base)
        pa_vals_by_base: dict[VarName, Value] = {}
        for p in parents:
            pa_vals_by_base[p.base] = vals[p]
        ordered_vals = tuple(pa_vals_by_base[b] for b in expected_parent_bases)
        vals[sn] = f(*ordered_vals)
    return vals


def eval_cfvar(u: CanonicalState, cfvar: CFVar, G: ADMG) -> Value:
    """Compute the value of `cfvar` under canonical state `u`."""
    # Wrap in a dummy query (target value irrelevant for eval).
    dummy = CFQuery(events=((cfvar, 0),), graph=G, name="__eval__")
    gstar = edge_split_graph(G, dummy)
    vals = _evaluate_split_graph(u, gstar)
    root = gstar.roots[0]
    return vals[root]


def eval_query_indicator(u: CanonicalState, Q: CFQuery, G: ADMG) -> int:
    """Return 1 iff every event in Q holds under state u; else 0."""
    gstar = edge_split_graph(G, Q)
    vals = _evaluate_split_graph(u, gstar)
    for root, (_cfvar, target) in zip(gstar.roots, Q.events, strict=True):
        if vals[root] != target:
            return 0
    return 1


__all__ = [
    "CanonicalError",
    "CanonicalState",
    "ResponseFn",
    "enumerate_states",
    "eval_cfvar",
    "eval_query_indicator",
    "state_cardinality",
]
