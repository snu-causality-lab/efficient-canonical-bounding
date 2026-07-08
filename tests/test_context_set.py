"""Unit tests for canonred.context_set (spec: docs/01_concepts.md §10).

m_V(Q) = |S_V(Q)|; Prop 4.2 cardinality d_U(Q) = prod_V d_V^{m_V(Q)}.

All numbers are from paper §4.2, §4.1, and the natural_bound_app.
"""

import pytest

from canonred.context_set import (
    context_multiplicities,
    m_V,
    pch_cardinality,
    prop42_cardinality,
)
from canonred.counterfactual import CFQuery, CFVar
from canonred.examples import iv_graph, nde_graph, pns_graph
from canonred.worlds import World


# ---------------------------------------------------------------------
# m_V: NDE running example
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
def nde_l25(nde):
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    cf = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w1, ("X", "Y"): w0, ("W", "Y"): w1},
        G=nde,
    )
    return CFQuery(events=((cf, 1),), graph=nde, name="Y_{0,W_1}")


@pytest.fixture
def nde_obs(nde):
    return CFQuery(events=((CFVar.observational("Y", nde), 1),), graph=nde, name="Y=1")


@pytest.fixture
def pns_q(pns):
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    return CFQuery(
        events=(
            (CFVar.standard("Y", w0, pns), 1),
            (CFVar.standard("Y", w1, pns), 0),
        ),
        graph=pns,
    )


def test_m_V_nde_l25(nde, nde_l25):
    """paper §4.2: for Y_{0, W_1} on NDE, m_X=0, m_W=1, m_Y=2."""
    assert m_V(nde, nde_l25, "X") == 0
    assert m_V(nde, nde_l25, "W") == 1
    assert m_V(nde, nde_l25, "Y") == 2


def test_m_V_nde_obs_recovers_l3_worst_case(nde, nde_obs):
    """Observational Y = 1 on NDE hits the L_3 worst-case:
    m_X=1, m_W=|Omega_X|=2, m_Y=|Omega_{X,W}|=4.
    """
    assert m_V(nde, nde_obs, "X") == 1
    assert m_V(nde, nde_obs, "W") == 2
    assert m_V(nde, nde_obs, "Y") == 4


def test_context_multiplicities_returns_dict(nde, nde_l25):
    m = context_multiplicities(nde, nde_l25)
    assert set(m.keys()) == set(nde.nodes)
    assert m["X"] == 0
    assert m["W"] == 1
    assert m["Y"] == 2


# ---------------------------------------------------------------------
# Prop 4.2 cardinality
# ---------------------------------------------------------------------


def test_prop42_nde_l25_equals_8(nde, nde_l25):
    """paper §6 Case 2 + §4.2: d_U(Y_{0,W_1}) = 2^0 × 2^1 × 2^2 = 8."""
    assert prop42_cardinality(nde, nde_l25) == 8


def test_prop42_nde_obs_equals_128(nde, nde_obs):
    """Observational Y: d_U = full L_3 count = 128."""
    assert prop42_cardinality(nde, nde_obs) == 128


def test_prop42_pns_query_only_equals_64(pns, pns_q):
    """Paper natural_bound_app: PNS query-only → d_U = 64.

    With X->Y edge (added 2026-04-25 per author confirmation):
      c-components: {X, Y} (via X<->Y), {M} (via M<->Y - wait, M<->Y connects them).
      Actually: X<->Y connects {X,Y}; M<->Y connects M with Y; so all three are
      one c-component {X, M, Y}.
      m_X = 0 (intervened in both worlds).
      m_M = 2 (M invoked at X=0 and X=1 contexts).
      m_Y = 4 (Y invoked at 4 (X,M) combinations across both worlds).
      d_U = 2^0 * 2^2 * 2^4 = 64.
    """
    assert prop42_cardinality(pns, pns_q) == 64


def test_pns_lifted_size_is_128():
    """Paper natural_bound_app: PNS lifted L3 dim = 128 (with X->Y edge)."""
    from canonred.examples import pns_graph

    g = pns_graph()
    assert g.lifted_canonical_size() == 128


def test_ternary_nde_query_only_equals_24():
    """Paper natural_bound_app Graph 2: ternary NDE query-only → d_U = 24.

    Graph: Z -> X -> M -> Y, X -> Y, X <-> Y, Z/X/Y binary, M ternary.
    Query: Q_NDE = P(Y_{1, M_0} = 1).

    Edge labels (σ on An(Y) = {Z, X, M, Y}):
      (X, Y): world x1  (Y's X-input is fixed to 1)
      (M, Y): world x0  (Y's M-input is M evaluated in world x0 → M_0)
      (X, M): world x0  (inherited along M's subtree)
      (Z, X): world x0  (inherited; but X is intervened in x0, so Z never feeds X)

    Structural counting:
      m_Z = 0 (X is intervened in every world of the query → Z is never invoked)
      m_X = 0 (X intervened in both worlds)
      m_M = 1 (M invoked once at X=0 under label x0)
      m_Y = 3 (Y invoked at (X=1, M=v) for v ∈ {0,1,2})

    c-components of induced ancestor subgraph = {Z}, {X, Y}, {M}.
    d_U = 2^0 × (2^0 × 2^3) × 3^1 = 1 × 8 × 3 = 24.
    """
    from canonred.counterfactual import CFQuery, CFVar
    from canonred.examples import ternary_nde_graph
    from canonred.worlds import World

    g = ternary_nde_graph()
    w1 = World.from_dict("x1", {"X": 1})
    w0 = World.from_dict("x0", {"X": 0})
    y_cf = CFVar.nested(
        base="Y",
        edge_worlds={
            ("Z", "X"): w0,
            ("X", "M"): w0,
            ("X", "Y"): w1,
            ("M", "Y"): w0,
        },
        G=g,
    )
    Q = CFQuery(events=((y_cf, 1),), graph=g, name="Y_{1,M_0}=1")
    assert prop42_cardinality(g, Q) == 24


def test_prop42_iv_y_x1_is_small(iv):
    """IV Y_{x=1}: m_Z=0 (not invoked), m_X=0 (fully fixed), m_Y=1 → d_U = 2.

    c-components of IV: {Z} and {X, Y}.
    d_{U_1} = |Omega_Z|^0 = 1.
    d_{U_2} = |Omega_X|^0 × |Omega_Y|^1 = 1 × 2 = 2.
    Total = 2.
    """
    w = World.from_dict("x1", {"X": 1})
    Q = CFQuery(events=((CFVar.standard("Y", w, iv), 1),), graph=iv, name="Y_x1")
    assert prop42_cardinality(iv, Q) == 2


# ---------------------------------------------------------------------
# PCH layer cardinalities (analytic, query-independent)
# ---------------------------------------------------------------------


def test_pch_L3_nde(nde):
    """L_3 worst-case for NDE = 128."""
    assert pch_cardinality(nde, layer="L3") == 128


def test_pch_L3_iv(iv):
    """L_3 worst-case for IV = 32 (Z:2, X:4, Y:4)."""
    assert pch_cardinality(iv, layer="L3") == 32


def test_pch_L1_nde(nde):
    """L_1 count on NDE uses closed parent domains C union pa(C).

    c-components of NDE: {X, W} and {Y}.
      C_1 = {X, W}: C_1 union pa(C_1) = {X, W} -> d_{U_1}=4.
      C_2 = {Y}: C_2 union pa(C_2) = {X, W, Y} -> d_{U_2}=8.
    Total L_1 = 4 × 8 = 32.
    """
    assert pch_cardinality(nde, layer="L1") == 32


def test_pch_L2_nde(nde):
    """L_2 count on NDE = prod over c-comp of prod V in C of d_{pa(V) x V}.

    C_1 = {X, W}:
      V=X: pa(X)=∅, |Omega_{pa(X) × X}| = |Omega_X| = 2.
      V=W: pa(W)={X}, |Omega_{pa(W) × W}| = |Omega_X| × |Omega_W| = 4.
      d_{U_1} = 2 × 4 = 8.
    C_2 = {Y}:
      V=Y: pa(Y)={X,W}, |Omega_{pa(Y) × Y}| = 2 × 2 × 2 = 8.
      d_{U_2} = 8.
    Total = 8 × 8 = 64.
    """
    assert pch_cardinality(nde, layer="L2") == 64


def test_pch_unknown_layer_errors(nde):
    with pytest.raises(ValueError):
        pch_cardinality(nde, layer="L4")  # type: ignore[arg-type]
