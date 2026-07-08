"""Unit tests for canonred.expansion (spec: docs/01_concepts.md §3.5, §10.1).

This is the module that turns a CFQuery into its structural-expansion / edge-
split representation and enumerates S_V(Q). Every expected number below is
hand-derived from the paper (paper §4.2) or from YB25 L_{2.5} semantics; cite in
the test docstring.

Key concepts under test:
  - SplitNode identity and compatibility across CFVars.
  - `edge_split_graph(G, Q)`: build G* as an ADMG-like structure with fixed-
    value annotations for intervened split copies.
  - `enumerate_contexts(G, Q)`: return S_V(Q) as sets of labeled parent tuples
    per base V.
"""

import pytest

from canonred.counterfactual import CFQuery, CFVar, CFVarError
from canonred.examples import bow_graph, iv_graph, nde_graph, pns_graph
from canonred.expansion import (
    ConflictError,
    SplitNode,
    edge_split_graph,
    enumerate_contexts,
)
from canonred.worlds import World


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def nde():
    return nde_graph()


@pytest.fixture
def pns():
    return pns_graph()


@pytest.fixture
def iv():
    return iv_graph()


@pytest.fixture
def bow():
    return bow_graph()


@pytest.fixture
def w_obs():
    return World.observational()


@pytest.fixture
def w_x0():
    return World.from_dict("x0", {"X": 0})


@pytest.fixture
def w_x1():
    return World.from_dict("x1", {"X": 1})


@pytest.fixture
def nde_l25_query(nde, w_x0, w_x1):
    """Q = P(Y_{0, W_1} = 1), paper paper running example."""
    cf = CFVar.nested(
        base="Y",
        edge_worlds={
            ("X", "W"): w_x1,
            ("X", "Y"): w_x0,
            ("W", "Y"): w_x1,
        },
        G=nde,
    )
    return CFQuery(events=((cf, 1),), graph=nde, name="Y_{0,W_1}=1")


@pytest.fixture
def nde_obs_query(nde, w_obs):
    """Observational Y = 1 on NDE."""
    cf = CFVar.observational("Y", nde)
    return CFQuery(events=((cf, 1),), graph=nde, name="Y=1")


@pytest.fixture
def pns_query(pns):
    """PNS: Q = P(Y_{x=0} = 1, Y_{x=1} = 0) — a cross-world L_3 event."""
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    y_x0 = CFVar.standard("Y", w0, pns)
    y_x1 = CFVar.standard("Y", w1, pns)
    return CFQuery(events=((y_x0, 1), (y_x1, 0)), graph=pns, name="PNS")


# ---------------------------------------------------------------------
# SplitNode basics
# ---------------------------------------------------------------------


def test_splitnode_is_frozen_and_hashable():
    a = SplitNode(cfvar_index=0, base="X", label="x0")
    b = SplitNode(cfvar_index=0, base="X", label="x0")
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_splitnode_distinct_by_label():
    a = SplitNode(cfvar_index=0, base="X", label="x0")
    b = SplitNode(cfvar_index=0, base="X", label="x1")
    assert a != b


def test_splitnode_distinct_by_cfvar_index():
    """Two CFVars' roots at the same (base, label) are still distinct split nodes."""
    a = SplitNode(cfvar_index=0, base="Y", label="x")
    b = SplitNode(cfvar_index=1, base="Y", label="x")
    assert a != b


# ---------------------------------------------------------------------
# edge_split_graph: NDE L_{2.5}
# ---------------------------------------------------------------------


def test_nde_l25_split_graph_has_four_nodes(nde, nde_l25_query):
    """paper §6 Case 2: Y_{0, W_1} has 4 split nodes: {Y, X[x0], W[x1], X[x1]}."""
    gstar = edge_split_graph(nde, nde_l25_query)
    bases = {(n.base, n.label) for n in gstar.nodes}
    assert ("Y", "__root__0") in bases or any(n.base == "Y" for n in gstar.nodes)
    # Two copies of X with distinct labels:
    x_labels = {n.label for n in gstar.nodes if n.base == "X"}
    assert x_labels == {"x0", "x1"}, f"got X labels {x_labels}"
    # One copy of W (world x1):
    w_labels = {n.label for n in gstar.nodes if n.base == "W"}
    assert w_labels == {"x1"}, f"got W labels {w_labels}"


def test_nde_l25_fixed_values(nde, nde_l25_query):
    """X[x0] fixed to 0; X[x1] fixed to 1; W[x1] not fixed; Y not fixed."""
    gstar = edge_split_graph(nde, nde_l25_query)
    fixed = gstar.fixed_values

    # Identify the X split nodes by their label.
    x0 = next(n for n in gstar.nodes if n.base == "X" and n.label == "x0")
    x1 = next(n for n in gstar.nodes if n.base == "X" and n.label == "x1")
    w1 = next(n for n in gstar.nodes if n.base == "W" and n.label == "x1")
    y = next(n for n in gstar.nodes if n.base == "Y")

    assert fixed[x0] == 0
    assert fixed[x1] == 1
    assert w1 not in fixed
    assert y not in fixed


def test_nde_l25_split_edges(nde, nde_l25_query):
    """Edges in G*: X[x0]->Y, W[x1]->Y, X[x1]->W[x1]."""
    gstar = edge_split_graph(nde, nde_l25_query)
    x0 = next(n for n in gstar.nodes if n.base == "X" and n.label == "x0")
    x1 = next(n for n in gstar.nodes if n.base == "X" and n.label == "x1")
    w1 = next(n for n in gstar.nodes if n.base == "W" and n.label == "x1")
    y = next(n for n in gstar.nodes if n.base == "Y")

    assert (x0, y) in gstar.directed
    assert (w1, y) in gstar.directed
    assert (x1, w1) in gstar.directed
    assert len(gstar.directed) == 3


# ---------------------------------------------------------------------
# edge_split_graph: NDE observational
# ---------------------------------------------------------------------


def test_nde_obs_split_graph_same_structure_as_original(nde, nde_obs_query):
    """Observational σ assigns 'obs' everywhere: G* = NDE with 'obs' labels."""
    gstar = edge_split_graph(nde, nde_obs_query)
    bases_labels = {(n.base, n.label) for n in gstar.nodes}
    # One copy of each node at label "obs".
    # (The CFVar root Y may have a special root-marker label; allow either.)
    for b in ("X", "W"):
        assert any(n.base == b and n.label == "obs" for n in gstar.nodes), f"missing {b}@obs"
    assert any(n.base == "Y" for n in gstar.nodes)


def test_nde_obs_no_fixed_values(nde_obs_query, nde):
    """Observational world intervenes on nothing → no fixed split nodes."""
    gstar = edge_split_graph(nde, nde_obs_query)
    assert gstar.fixed_values == {}


def test_root_self_intervention_is_fixed_without_expanding_parents(bow):
    """Y_{y=1} should be a single fixed root, with no X parent expansion."""
    w = World.from_dict("y1", {"Y": 1})
    cf = CFVar.standard("Y", w, bow)
    Q = CFQuery(events=((cf, 1),), graph=bow)

    gstar = edge_split_graph(bow, Q)
    root = gstar.roots[0]
    assert gstar.nodes == frozenset({root})
    assert gstar.directed == frozenset()
    assert gstar.fixed_values == {root: 1}


def test_edge_split_graph_rejects_query_graph_mismatch(bow, nde):
    cf = CFVar.observational("X", bow)
    Q = CFQuery(events=((cf, 1),), graph=bow)
    with pytest.raises(CFVarError):
        edge_split_graph(nde, Q)


# ---------------------------------------------------------------------
# edge_split_graph: PNS cross-world
# ---------------------------------------------------------------------


def test_pns_split_graph_has_two_chains(pns, pns_query):
    """PNS Q: Y_x and Y_{x'} live in separate subtrees; no shared non-fixed node."""
    gstar = edge_split_graph(pns, pns_query)
    # Two copies of each of Y, M.
    assert sum(1 for n in gstar.nodes if n.base == "Y") == 2
    assert sum(1 for n in gstar.nodes if n.base == "M") == 2
    # Two fixed copies of X (one for each world).
    x_nodes = [n for n in gstar.nodes if n.base == "X"]
    assert len(x_nodes) == 2
    for n in x_nodes:
        assert n in gstar.fixed_values


def test_pns_x_values(pns, pns_query):
    """X[x0]=0 and X[x1]=1 in PNS split graph."""
    gstar = edge_split_graph(pns, pns_query)
    vals = {gstar.fixed_values[n] for n in gstar.nodes if n.base == "X"}
    assert vals == {0, 1}


# ---------------------------------------------------------------------
# enumerate_contexts: S_V(Q) for named queries
# ---------------------------------------------------------------------


def test_enumerate_contexts_nde_l25(nde, nde_l25_query):
    """paper §4.2: for Y_{0, W_1}, m_X=0, m_W=1, m_Y=2."""
    S = enumerate_contexts(nde, nde_l25_query)

    # m_X = 0: X has no NON-FIXED split copies.
    assert S["X"] == frozenset(), f"expected m_X=0, got {S['X']}"

    # m_W = 1: W called once at X=1 under label x1.
    assert len(S["W"]) == 1
    # The single tuple is ((X=1)_{x1},)
    only = next(iter(S["W"]))
    assert only == (("X", 1, "x1"),)

    # m_Y = 2: Y called at (X=0 under x0, W=0 under x1) and (X=0 under x0, W=1 under x1).
    assert len(S["Y"]) == 2
    tuples = {tuple(sorted(t, key=lambda e: e[0])) for t in S["Y"]}
    expected = {
        tuple(sorted((("X", 0, "x0"), ("W", 0, "x1")), key=lambda e: e[0])),
        tuple(sorted((("X", 0, "x0"), ("W", 1, "x1")), key=lambda e: e[0])),
    }
    assert tuples == expected


def test_enumerate_contexts_nde_obs(nde, nde_obs_query):
    """Observational Y on NDE: m_X=1, m_W=2, m_Y=4 (full L_3 count)."""
    S = enumerate_contexts(nde, nde_obs_query)
    assert len(S["X"]) == 1       # X called once with empty parent tuple
    assert len(S["W"]) == 2       # W called at X=0 and X=1
    assert len(S["Y"]) == 4       # Y called at (X,W) ∈ {0,1}^2


def test_enumerate_contexts_pns(pns, pns_query):
    """PNS Q = P(Y_x=1, Y_{x'}=0): m_X=0, m_M=2, m_Y=4 (paper natural_bound_app)."""
    S = enumerate_contexts(pns, pns_query)
    assert len(S["X"]) == 0
    assert len(S["M"]) == 2
    assert len(S["Y"]) == 4


def test_enumerate_contexts_iv_yx(iv):
    """IV graph Y_{x=1}: X fixed, so m_Z=0, m_X=0, m_Y=1."""
    w = World.from_dict("x1", {"X": 1})
    y_x1 = CFVar.standard("Y", w, iv)
    Q = CFQuery(events=((y_x1, 1),), graph=iv, name="Y_x1")

    S = enumerate_contexts(iv, Q)
    assert len(S["Z"]) == 0   # Z not ancestor of Y in mutilated graph... wait Z IS ancestor of X, but X is fixed.
    assert len(S["X"]) == 0   # X fully intervened in world x1
    assert len(S["Y"]) == 1   # Y called at the single parent tuple (X=1)_{x1}


# ---------------------------------------------------------------------
# Conflict detection (L_{2.5} realizability)
# ---------------------------------------------------------------------


def test_conflict_detected_when_same_label_gets_incompatible_worlds(nde):
    """If the user builds σ with two worlds sharing a label but different do-maps,
    the expansion should detect this (or, equivalently, CFVar construction should).

    Our current CFVar construction accepts inconsistent (edge → world) maps as
    long as the `edge_worlds` keys cover An(base). This test checks that the
    expansion surfaces the inconsistency: if (X, W) carries world_a and (W, Y)
    carries world_b where both have label "x" but different values, conflict.
    """
    w_a = World.from_dict("x", {"X": 0})
    w_b = World.from_dict("x", {"X": 1})  # same label, different value → conflict
    cf = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w_a, ("X", "Y"): w_a, ("W", "Y"): w_b},
        G=nde,
    )
    Q = CFQuery(events=((cf, 1),), graph=nde)
    with pytest.raises(ConflictError):
        edge_split_graph(nde, Q)
