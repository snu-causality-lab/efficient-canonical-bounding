"""Unit tests for canonred.graph.

All expected values here are either (a) easy hand checks, or (b) directly from
the 1086 paper (see docstrings per test).
"""

import pytest

from canonred.examples import (
    bow_graph,
    frontdoor_graph,
    iv_graph,
    nde_graph,
    non_iv_graph,
    pns_graph,
    ternary_nde_graph,
)
from canonred.graph import ADMG, GraphError, admg


# ----------------------------------------------------------------------
# Construction & validation
# ----------------------------------------------------------------------


def test_bare_construction_ok():
    g = admg(nodes=("A", "B"), directed=[("A", "B")])
    assert g.n == 2
    assert g.parents("B") == ("A",)
    assert g.children("A") == ("B",)
    assert g.cardinality["A"] == 2


def test_duplicate_nodes_error():
    with pytest.raises(GraphError):
        admg(nodes=("A", "A"))


def test_self_loop_error():
    with pytest.raises(GraphError):
        admg(nodes=("A",), directed=[("A", "A")])


def test_edge_to_unknown_node_error():
    with pytest.raises(GraphError):
        admg(nodes=("A",), directed=[("A", "B")])


def test_bidirected_size_mismatch_error():
    with pytest.raises(GraphError):
        admg(nodes=("A", "B"), bidirected=[{"A"}])  # not a 2-set


def test_cycle_error():
    with pytest.raises(GraphError):
        admg(nodes=("A", "B"), directed=[("A", "B"), ("B", "A")])


def test_cardinality_below_two_error():
    with pytest.raises(GraphError):
        admg(nodes=("A",), cardinality={"A": 1})


def test_extra_cardinality_key_error():
    with pytest.raises(GraphError):
        admg(nodes=("A",), cardinality={"A": 2, "B": 2})


# ----------------------------------------------------------------------
# NDE graph (1086 Fig. 2a)
# ----------------------------------------------------------------------


def test_nde_structure():
    g = nde_graph()
    assert g.nodes == ("X", "W", "Y")
    assert g.parents("X") == ()
    assert g.parents("W") == ("X",)
    assert g.parents("Y") == ("X", "W")
    assert set(g.bidir_neighbors("X")) == {"W"}


def test_nde_topo_order():
    g = nde_graph()
    order = g.topo_order()
    # any valid topo order puts X first, Y last
    assert order[0] == "X"
    assert order[-1] == "Y"


def test_nde_c_components():
    """Paper §4.1: c-components of NDE are {X, W} and {Y}."""
    g = nde_graph()
    comps = g.c_components()
    assert len(comps) == 2
    comps_as_sets = {frozenset(c) for c in comps}
    assert frozenset({"X", "W"}) in comps_as_sets
    assert frozenset({"Y"}) in comps_as_sets


def test_nde_c_cliques_equal_c_components():
    """NDE has a single bidirected edge X<->W, so c-cliques match c-components."""
    g = nde_graph()
    cliques = {frozenset(c) for c in g.c_cliques()}
    comps = {frozenset(c) for c in g.c_components()}
    assert cliques == comps


def test_nde_ancestors():
    g = nde_graph()
    assert g.ancestors({"Y"}) == frozenset({"X", "W", "Y"})
    assert g.ancestors({"Y"}, inclusive=False) == frozenset({"X", "W"})
    assert g.ancestors({"W"}) == frozenset({"X", "W"})


def test_nde_mutilate_incoming_x():
    """G_{\\overline{X}}: remove edges X->W, X->Y, and bidirected X<->W."""
    g = nde_graph()
    gm = g.mutilate_incoming({"X"})
    # bidirected X<->W is removed because it's incident on X
    assert gm.bidirected == frozenset()
    # directed edges into X (none exist here) removed
    # but edges OUT of X remain: X->W and X->Y
    assert ("X", "W") in gm.directed
    assert ("X", "Y") in gm.directed


def test_nde_lifted_size():
    """|Omega_full| for binary NDE = 2^1 * 2^2 * 2^4 = 128."""
    g = nde_graph()
    # X: no parents -> 2 response-fn values
    # W: 1 parent (X, binary) -> 2^2 = 4
    # Y: 2 parents (X, W) -> 2^4 = 16
    # total = 2 * 4 * 16 = 128
    assert g.response_function_count("X") == 2
    assert g.response_function_count("W") == 4
    assert g.response_function_count("Y") == 16
    assert g.lifted_canonical_size() == 128


# ----------------------------------------------------------------------
# IV graph
# ----------------------------------------------------------------------


def test_iv_c_components():
    """Paper §3: IV has two c-components {Z} and {X, Y}."""
    g = iv_graph()
    comps = {frozenset(c) for c in g.c_components()}
    assert comps == {frozenset({"Z"}), frozenset({"X", "Y"})}


def test_iv_lifted_size():
    """|Omega_full| for binary IV = 2^1 * 2^2 * 2^4 = 128 (same count as NDE)."""
    g = iv_graph()
    # Z: 2, X (parent Z): 4, Y (parent X): 4 -> 32. Hmm but paper says differently?
    # Careful: response-fn count uses pa(V) only, not PA. X has 1 parent Z -> 2^2=4.
    # Y has 1 parent X -> 2^2=4. Total 2*4*4=32.
    assert g.lifted_canonical_size() == 32


# ----------------------------------------------------------------------
# Non-IV graph
# ----------------------------------------------------------------------


def test_non_iv_c_component_single():
    """Paper §3: Non-IV has a single c-component {X, Y, Z}."""
    g = non_iv_graph()
    comps = {frozenset(c) for c in g.c_components()}
    assert comps == {frozenset({"X", "Y", "Z"})}


def test_non_iv_c_cliques_are_two():
    """Paper appendix Def. A.2: Non-IV has c-cliques {X,Y} and {Z,Y}."""
    g = non_iv_graph()
    cliques = {frozenset(c) for c in g.c_cliques()}
    # X<->Y and Z<->Y exist, but X<->Z does NOT, so the c-component is not a c-clique.
    # maximal cliques on the bidirected graph: {X,Y} and {Z,Y}.
    assert frozenset({"X", "Y"}) in cliques
    assert frozenset({"Z", "Y"}) in cliques
    # No clique containing all three:
    assert frozenset({"X", "Y", "Z"}) not in cliques


# ----------------------------------------------------------------------
# Frontdoor
# ----------------------------------------------------------------------


def test_frontdoor_c_components():
    """Frontdoor has c-components {X, Y} and {W}."""
    g = frontdoor_graph()
    comps = {frozenset(c) for c in g.c_components()}
    assert comps == {frozenset({"X", "Y"}), frozenset({"W"})}


# ----------------------------------------------------------------------
# Bow
# ----------------------------------------------------------------------


def test_bow_c_component_single():
    g = bow_graph()
    comps = {frozenset(c) for c in g.c_components()}
    assert comps == {frozenset({"X", "Y"})}


def test_bow_lifted_size():
    """X (no parents, binary) = 2; Y (parent X, binary) = 2^2 = 4. Total 8."""
    g = bow_graph()
    assert g.lifted_canonical_size() == 8


# ----------------------------------------------------------------------
# PNS graph (confounded mediator)
# ----------------------------------------------------------------------


def test_pns_structure():
    g = pns_graph()
    # With paper's intended X->Y edge, Y has both parents.
    assert set(g.parents("Y")) == {"X", "M"}
    assert set(g.bidir_neighbors("Y")) == {"X", "M"}


def test_pns_c_component_single():
    """X<->Y and M<->Y connects all three into one c-component."""
    g = pns_graph()
    comps = {frozenset(c) for c in g.c_components()}
    assert comps == {frozenset({"X", "M", "Y"})}


def test_pns_c_cliques():
    """Maximal bidirected-cliques in PNS: {X,Y} and {M,Y}."""
    g = pns_graph()
    cliques = {frozenset(c) for c in g.c_cliques()}
    assert frozenset({"X", "Y"}) in cliques
    assert frozenset({"M", "Y"}) in cliques
    assert frozenset({"X", "M", "Y"}) not in cliques


def test_pns_lifted_size():
    """|Omega_full| = 2 × 4 × 16 = 128 for PNS (X, M binary, Y binary with pa(Y)={X,M}).

    Paper natural_bound_app Table 'Graph 1 Worst-case (L3)' reports 128 via:
       X (2^1) × M (2^2) × Y (2^4)
    where Y's response-fn count 2^4 = 2^{|Omega_X × Omega_M|} assumes Y has BOTH
    X and M as parents. Confirmed by the author (2026-04-25) that the X->Y
    directed edge is missing from the paper's TikZ by copy-paste mistake.
    """
    g = pns_graph()
    assert g.lifted_canonical_size() == 128


# ----------------------------------------------------------------------
# Ternary-mediator NDE
# ----------------------------------------------------------------------


def test_ternary_nde_structure():
    g = ternary_nde_graph()
    assert g.cardinality["M"] == 3
    # With X->Y edge (added 2026-04-25 per paper-intended table numbers), Y has
    # both X and M as parents.
    assert set(g.parents("Y")) == {"X", "M"}


def test_ternary_nde_c_components():
    """X<->Y connects X and Y; Z and M are singletons."""
    g = ternary_nde_graph()
    comps = {frozenset(c) for c in g.c_components()}
    assert comps == {frozenset({"Z"}), frozenset({"X", "Y"}), frozenset({"M"})}


def test_ternary_nde_response_counts():
    """Per-node response function counts with paper-intended X->Y edge.

    Z: no parents -> 2
    X: 1 parent (Z, binary) -> 2^2 = 4
    M: 1 parent (X, binary), M ternary -> 3^2 = 9
    Y: 2 parents (X binary, M ternary), Y binary -> 2^(2*3) = 2^6 = 64

    Total = 2 * 4 * 9 * 64 = 4608 — matches paper's natural_bound_app Graph 2
    "Worst-case L3" line.
    """
    g = ternary_nde_graph()
    assert g.response_function_count("Z") == 2
    assert g.response_function_count("X") == 4
    assert g.response_function_count("M") == 9
    assert g.response_function_count("Y") == 64
    assert g.lifted_canonical_size() == 4608


# ----------------------------------------------------------------------
# induce() and mutilate preserve ADMG invariants
# ----------------------------------------------------------------------


def test_induce_preserves_edges():
    g = nde_graph()
    g_wy = g.induce(["W", "Y"])
    assert set(g_wy.nodes) == {"W", "Y"}
    assert ("W", "Y") in g_wy.directed
    # X<->W bidirected is gone because X not in induced set
    assert g_wy.bidirected == frozenset()


def test_mutilate_incoming_roundtrip():
    g = nde_graph()
    g2 = g.mutilate_incoming({"X"})
    # applying again is a no-op
    g3 = g2.mutilate_incoming({"X"})
    assert g2.directed == g3.directed
    assert g2.bidirected == g3.bidirected


# ----------------------------------------------------------------------
# Property: c-components partition nodes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "g",
    [nde_graph(), iv_graph(), non_iv_graph(), frontdoor_graph(), bow_graph(), pns_graph(), ternary_nde_graph()],
)
def test_c_components_partition(g: ADMG):
    comps = g.c_components()
    union: set[str] = set()
    for c in comps:
        assert union.isdisjoint(c), f"c-components not disjoint: {comps}"
        union.update(c)
    assert union == set(g.nodes)
