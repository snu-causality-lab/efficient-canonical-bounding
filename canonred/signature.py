"""Coarsest safe quotient via (c_u, M_{:,u}) signature (Prop 3.1).

See docs/01_concepts.md §9 and paper Prop. 3.1 / Alg. 1 Phase 3.

Given an ADMG G, a CFQuery Q (defining the linear objective c), and a list
of MomentConstraints (defining M), we compute the coarsest equivalence
relation u ~ u' ⟺ (c_u = c_{u'}) ∧ (M_{:, u} = M_{:, u'}), and the reduced
objective / constraint-matrix on the quotient.

Two builders:
  - `build_signature_quotient(G, Q, cs)` — Phase 2+3 explicit: enumerates
    Omega_full, computes signatures, buckets. O(|Omega_full|) time/memory.
    Works for any constraint family; the conservative default.

  - `build_signature_quotient_lazy_pobs(G, Q)` — pobs-specialised lazy
    builder. Iterates over the n_obs_cells = prod_V d_V observed cells of
    P(V_1,...,V_n) and, per cell, runs a DFS over G* to determine which
    c-indicator values ∈ {0,1} are achievable for canonical states u that
    produce that cell. Stops as soon as both 0 and 1 are seen (early exit).
    Σ-classes are exactly {(cell_idx, c_value): c_value ∈ achievable}, so
    the reduced LP is M̃ = block-1-per-cell, c̃ = c_value. Avoids
    materialising Omega_full; the only enumeration is per-cell over
    response-function entries reachable from Q's expansion graph.
"""

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from canonred.canonical import enumerate_states, eval_query_indicator, state_cardinality
from canonred.counterfactual import CFQuery
from canonred.expansion import SplitNode, edge_split_graph
from canonred.graph import ADMG
from canonred.observation import MomentConstraint


@dataclass(frozen=True)
class SignatureQuotient:
    """Reduced LP data (on Omega_red = Omega_full / ~).

    Attributes
    ----------
    lifted_dim : int
        |Omega_full|.
    reduced_dim : int
        Number of equivalence classes.
    objective : np.ndarray, shape (reduced_dim,)
        `c_tilde` — one value per class (the shared indicator).
    constraint_matrix : np.ndarray, shape (n_constraints, reduced_dim)
        `M_tilde`. Column t is the per-class signature across all constraints.
    class_sizes : tuple[int, ...], length reduced_dim
        # of lifted states in each class. Sums to `lifted_dim`.
    state_class_assignment : tuple[int, ...], length lifted_dim
        `state_class_assignment[i] = t` iff the i-th lifted state (in the
        order yielded by `enumerate_states(G)`) is in class `t`.
    class_lifted_c : tuple[tuple[int, ...], ...]
        Per-class list of c_u values across its lifted members (all equal, by
        construction — used for testing Prop 3.1 invariants).
    class_lifted_M : tuple[tuple[tuple[int, ...], ...], ...]
        Per-class list of M_{:, u} rows across its lifted members (all equal).
    """

    lifted_dim: int
    reduced_dim: int
    objective: np.ndarray
    constraint_matrix: np.ndarray
    class_sizes: tuple[int, ...]
    state_class_assignment: tuple[int, ...]
    class_lifted_c: tuple[tuple[int, ...], ...]
    class_lifted_M: tuple[tuple[tuple[int, ...], ...], ...]


def build_signature_quotient(
    G: ADMG,
    Q: CFQuery,
    constraints: Iterable[MomentConstraint] = (),
) -> SignatureQuotient:
    """Build the coarsest safe quotient by bucketing states on sigma(u).

    Enumerates Omega_full explicitly. For each state u, computes
    ``c_u = eval_query_indicator(u, Q, G)`` and
    ``M[:, u] = (constraint_i.evaluate(u, G) for each i)``.
    States sharing the same (c_u, tuple(M[:, u])) go in one class.
    """
    cs = tuple(constraints)
    n_cons = len(cs)

    buckets: dict[tuple[int, tuple[int, ...]], int] = {}
    class_sizes: list[int] = []
    class_lifted_c: list[list[int]] = []
    class_lifted_M: list[list[tuple[int, ...]]] = []
    reduced_c: list[int] = []
    reduced_M: list[tuple[int, ...]] = []
    assignment: list[int] = []

    total = 0
    for u in enumerate_states(G):
        total += 1
        c_u = eval_query_indicator(u, Q, G)
        M_col = tuple(c.evaluate(u, G) for c in cs)
        sig = (c_u, M_col)
        if sig not in buckets:
            buckets[sig] = len(buckets)
            class_sizes.append(0)
            class_lifted_c.append([])
            class_lifted_M.append([])
            reduced_c.append(c_u)
            reduced_M.append(M_col)
        idx = buckets[sig]
        class_sizes[idx] += 1
        class_lifted_c[idx].append(c_u)
        class_lifted_M[idx].append(M_col)
        assignment.append(idx)

    reduced_dim = len(buckets)
    c_arr = np.asarray(reduced_c, dtype=np.int64)
    if n_cons > 0:
        M_arr = np.asarray(reduced_M, dtype=np.int64).T  # shape (n_cons, reduced_dim)
    else:
        M_arr = np.zeros((0, reduced_dim), dtype=np.int64)

    return SignatureQuotient(
        lifted_dim=total,
        reduced_dim=reduced_dim,
        objective=c_arr,
        constraint_matrix=M_arr,
        class_sizes=tuple(class_sizes),
        state_class_assignment=tuple(assignment),
        class_lifted_c=tuple(tuple(xs) for xs in class_lifted_c),
        class_lifted_M=tuple(tuple(xs) for xs in class_lifted_M),
    )


# ----------------------------------------------------------------------
# Lazy pobs builder (no Omega_full enumeration)
# ----------------------------------------------------------------------


def _topo_order_split_graph_for_lazy(gstar) -> list[SplitNode]:
    """Local topo-sort of G* (avoid importing the private helper from canonical.py).

    Same algorithm as `canonical._topo_order_split_graph` but inlined so the
    lazy builder is self-contained.
    """
    indeg: dict[SplitNode, int] = {n: 0 for n in gstar.nodes}
    for (_p, c) in gstar.directed:
        indeg[c] += 1
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
                ready.append(c)
                ready.sort(key=key)
    if len(order) != len(gstar.nodes):
        raise RuntimeError("edge-split graph is not a DAG")
    return order


def _observation_pins(G: ADMG, cell_values: dict) -> dict[tuple, int]:
    """Compute the response-fn entries pinned by observation of `cell_values`.

    For each V in topo order, the observed value of V is determined by f_V at
    the parent context (parent_obs_values), which is recursively determined.

    Returns: dict mapping (V, parent_value_tuple) -> value.
    """
    pins: dict[tuple, int] = {}
    obs: dict = {}  # observed value of each V
    for v in G.topo_order():
        parents = G.parents(v)
        ctx = tuple(obs[p] for p in parents)
        target = cell_values[v]
        pins[(v, ctx)] = target
        obs[v] = target
    return pins


def _c_achievable_for_cell(
    G: ADMG,
    Q: CFQuery,
    cell_values: dict,
    gstar=None,
) -> frozenset[int]:
    """DFS over G* with cell-pinned f_V entries; return set of c_u ∈ {0, 1}.

    Per-cell complexity bound: O(prod_V d_V^{|S_V(Q) \\ pinned_by_cell|}) — typically
    much smaller than |Omega_full| when Q is sparse. Early exit when both 0 and 1
    are observed.
    """
    if gstar is None:
        gstar = edge_split_graph(G, Q)
    topo = _topo_order_split_graph_for_lazy(gstar)
    pins = _observation_pins(G, cell_values)

    achievable: set[int] = set()
    sn_value: dict[SplitNode, int] = {}    # split-node values along current branch
    fn_choice: dict[tuple, int] = {}        # (V, ctx_key) -> chosen value (current branch)

    def dfs(idx: int) -> None:
        if {0, 1}.issubset(achievable):
            return
        if idx == len(topo):
            c = 1
            for root, (_cf, target) in zip(gstar.roots, Q.events, strict=True):
                if sn_value.get(root) != target:
                    c = 0
                    break
            achievable.add(c)
            return

        sn = topo[idx]
        if sn in gstar.fixed_values:
            sn_value[sn] = gstar.fixed_values[sn]
            dfs(idx + 1)
            del sn_value[sn]
            return

        # Compute parent-value tuple in G.parents(sn.base) order
        parent_sns = gstar.parents_in_star(sn)
        pa_by_base = {p.base: sn_value[p] for p in parent_sns}
        ordered_vals = tuple(pa_by_base[b] for b in G.parents(sn.base))
        fn_key = (sn.base, ordered_vals)

        # 1. Already chosen earlier in this branch (same f_V at same parent context)
        if fn_key in fn_choice:
            sn_value[sn] = fn_choice[fn_key]
            dfs(idx + 1)
            del sn_value[sn]
            return

        # 2. Pinned by observation
        if fn_key in pins:
            fn_choice[fn_key] = pins[fn_key]
            sn_value[sn] = pins[fn_key]
            dfs(idx + 1)
            del fn_choice[fn_key]
            del sn_value[sn]
            return

        # 3. Free — enumerate
        for val in range(G.cardinality[sn.base]):
            fn_choice[fn_key] = val
            sn_value[sn] = val
            dfs(idx + 1)
            del fn_choice[fn_key]
            del sn_value[sn]
            if {0, 1}.issubset(achievable):
                return

    dfs(0)
    return frozenset(achievable)


def build_signature_quotient_lazy_pobs(
    G: ADMG, Q: CFQuery
) -> SignatureQuotient:
    """Lazy σ-quotient build for pobs constraints (no Omega_full enumeration).

    Constraints are *implicitly* taken to be `pobs_constraints(G)` — one per
    cell of P(V_1, ..., V_n). The returned SignatureQuotient has:

      reduced_dim = sum over cells of |c_achievable(cell)|  ∈ [n_cells, 2*n_cells]
      objective[t] = c_value of the t-th class
      constraint_matrix[cell_idx_of_class, t] = 1, 0 elsewhere
      class_sizes / state_class_assignment / class_lifted_* are unset (0 / empty)
        because the lazy builder does not enumerate Omega_full and therefore
        cannot count members per class.

    For graphs that are tractable both ways, this gives the same reduced LP
    bounds as `build_signature_quotient(G, Q, pobs_constraints(G))` — verified
    empirically against bow / IV / non-IV / front-door / NDE / PNS / ternary-NDE
    in tests.
    """
    ordered = G.nodes
    domains = tuple(range(G.cardinality[v]) for v in ordered)
    n_cells = 1
    for d in domains:
        n_cells *= len(d)

    # Pre-build G* for Q once (re-used across cells).
    gstar = edge_split_graph(G, Q)

    classes: list[tuple[int, int]] = []     # list of (cell_idx, c_value)
    reduced_c: list[int] = []
    reduced_M_rows: list[tuple[int, int]] = []  # (row=cell_idx, col=class_idx)

    for cell_idx, cell_tup in enumerate(itertools.product(*domains)):
        cell = dict(zip(ordered, cell_tup, strict=True))
        c_set = _c_achievable_for_cell(G, Q, cell, gstar=gstar)
        for c in sorted(c_set):
            class_idx = len(classes)
            classes.append((cell_idx, c))
            reduced_c.append(c)
            reduced_M_rows.append((cell_idx, class_idx))

    reduced_dim = len(classes)
    M_arr = np.zeros((n_cells, reduced_dim), dtype=np.int64)
    for (row, col) in reduced_M_rows:
        M_arr[row, col] = 1
    c_arr = np.asarray(reduced_c, dtype=np.int64)

    return SignatureQuotient(
        lifted_dim=state_cardinality(G),
        reduced_dim=reduced_dim,
        objective=c_arr,
        constraint_matrix=M_arr,
        class_sizes=tuple(0 for _ in range(reduced_dim)),
        state_class_assignment=tuple(),
        class_lifted_c=tuple(),
        class_lifted_M=tuple(),
    )


__all__ = [
    "SignatureQuotient",
    "build_signature_quotient",
    "build_signature_quotient_lazy_pobs",
]
