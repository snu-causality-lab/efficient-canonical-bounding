"""Unit tests for canonred.counterfactual (spec: docs/01_concepts.md §3, §4).

A CFVar is a base variable V plus an edge-label map σ that assigns a World
to every directed edge in the ancestor subgraph of V. Constructors:
  - observational(V, G): every edge → World.observational()
  - standard(V, w, G): every edge → w
  - nested(V, edge_worlds, G): explicit per-edge assignment (must cover all
    edges in An(V)).

A CFQuery is a tuple of (CFVar, target_value) events.
"""

import pytest

from canonred.counterfactual import CFQuery, CFVar, CFVarError
from canonred.examples import iv_graph, nde_graph
from canonred.worlds import World


# ---------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def nde():
    return nde_graph()


@pytest.fixture
def iv():
    return iv_graph()


@pytest.fixture
def world_obs():
    return World.observational()


@pytest.fixture
def world_x0():
    return World.from_dict("x0", {"X": 0})


@pytest.fixture
def world_x1():
    return World.from_dict("x1", {"X": 1})


# ---------------------------------------------------------------------
# CFVar.observational
# ---------------------------------------------------------------------


def test_cfvar_observational_base_is_correct(nde):
    v = CFVar.observational("Y", nde)
    assert v.base == "Y"


def test_cfvar_observational_covers_every_ancestor_edge(nde):
    """An(Y) in NDE has edges X->W, X->Y, W->Y — exactly 3 edges."""
    v = CFVar.observational("Y", nde)
    edges = {e for (e, _w) in v.edge_worlds}
    assert edges == {("X", "W"), ("X", "Y"), ("W", "Y")}


def test_cfvar_observational_every_edge_has_obs_world(nde, world_obs):
    v = CFVar.observational("Y", nde)
    for _edge, w in v.edge_worlds:
        assert w == world_obs


def test_cfvar_observational_root_node_has_no_edges(nde):
    """X is a root in NDE: An(X)={X}, no directed edges in the induced subgraph."""
    v = CFVar.observational("X", nde)
    assert v.edge_worlds == frozenset()


# ---------------------------------------------------------------------
# CFVar.standard
# ---------------------------------------------------------------------


def test_cfvar_standard_assigns_world_to_every_edge(nde, world_x1):
    """Y_{x=1} on NDE: σ assigns world_x1 to every edge in An(Y)."""
    v = CFVar.standard("Y", world_x1, nde)
    for _edge, w in v.edge_worlds:
        assert w == world_x1


def test_cfvar_standard_distinct_from_observational(nde, world_x0):
    a = CFVar.observational("Y", nde)
    b = CFVar.standard("Y", world_x0, nde)
    assert a != b


def test_cfvar_standard_root_preserves_world(nde, world_x1):
    """Root potential outcomes need the root world to evaluate X_x correctly."""
    v = CFVar.standard("X", world_x1, nde)
    assert v.edge_worlds == frozenset()
    assert v.worlds == frozenset({world_x1})


def test_cfvar_standard_on_iv_graph(iv, world_x1):
    """IV graph: An(Y) = {Z, X, Y}; edges are Z->X, X->Y."""
    v = CFVar.standard("Y", world_x1, iv)
    edges = {e for (e, _w) in v.edge_worlds}
    assert edges == {("Z", "X"), ("X", "Y")}


# ---------------------------------------------------------------------
# CFVar.nested — the NDE Y_{0, W_1} case
# ---------------------------------------------------------------------


def test_cfvar_nested_nde_y_0_w1(nde, world_x0, world_x1):
    """paper running example: Y_{0, W_1}.

    edges in An(Y):
      (X, W): world x1  (because W's parent X is evaluated in world x1, W=W_1)
      (X, Y): world x0  (Y's X-input is fixed to 0)
      (W, Y): world x1  (Y's W-input is W_1, so come from world x1)
    """
    v = CFVar.nested(
        base="Y",
        edge_worlds={
            ("X", "W"): world_x1,
            ("X", "Y"): world_x0,
            ("W", "Y"): world_x1,
        },
        G=nde,
    )
    assert v.base == "Y"
    # Check that the stored edge_worlds round-trips as a dict
    by_edge = dict(v.edge_worlds)
    assert by_edge[("X", "W")] == world_x1
    assert by_edge[("X", "Y")] == world_x0
    assert by_edge[("W", "Y")] == world_x1


def test_cfvar_nested_incomplete_coverage_errors(nde, world_x0):
    """Missing any edge in An(base) is an error (explicit API)."""
    with pytest.raises(CFVarError):
        CFVar.nested(
            base="Y",
            edge_worlds={("X", "Y"): world_x0},  # missing (X,W) and (W,Y)
            G=nde,
        )


def test_cfvar_nested_extra_edge_errors(nde, world_x0):
    """Providing an edge that is not in An(base) is an error."""
    # ("Y", "X") is not a directed edge of NDE at all.
    with pytest.raises(CFVarError):
        CFVar.nested(
            base="Y",
            edge_worlds={
                ("X", "W"): world_x0,
                ("X", "Y"): world_x0,
                ("W", "Y"): world_x0,
                ("Y", "X"): world_x0,  # extra, invalid
            },
            G=nde,
        )


def test_cfvar_nested_unknown_base_errors(nde, world_x0):
    with pytest.raises(CFVarError):
        CFVar.nested(base="Q", edge_worlds={}, G=nde)


# ---------------------------------------------------------------------
# Equality / hashability (frozen dataclass)
# ---------------------------------------------------------------------


def test_cfvar_equality(nde, world_x0):
    a = CFVar.standard("Y", world_x0, nde)
    b = CFVar.standard("Y", world_x0, nde)
    assert a == b
    assert hash(a) == hash(b)


def test_cfvar_inequality_by_base(nde, world_x0):
    """Different base → distinct CFVars even when edge_worlds overlap."""
    a = CFVar.standard("Y", world_x0, nde)
    b = CFVar.standard("W", world_x0, nde)
    assert a != b


def test_cfvar_inequality_by_world(nde, world_x0, world_x1):
    a = CFVar.standard("Y", world_x0, nde)
    b = CFVar.standard("Y", world_x1, nde)
    assert a != b


def test_cfvar_is_frozen(nde, world_x0):
    v = CFVar.standard("Y", world_x0, nde)
    with pytest.raises((AttributeError, TypeError)):
        v.base = "W"  # type: ignore[misc]


# ---------------------------------------------------------------------
# CFVar.worlds (helper accessor)
# ---------------------------------------------------------------------


def test_cfvar_worlds_standard_is_single(nde, world_x1):
    v = CFVar.standard("Y", world_x1, nde)
    assert v.worlds == frozenset({world_x1})


def test_cfvar_worlds_nested_multi(nde, world_x0, world_x1):
    v = CFVar.nested(
        base="Y",
        edge_worlds={
            ("X", "W"): world_x1,
            ("X", "Y"): world_x0,
            ("W", "Y"): world_x1,
        },
        G=nde,
    )
    assert v.worlds == frozenset({world_x0, world_x1})


def test_cfvar_worlds_observational_is_obs_only(nde, world_obs):
    v = CFVar.observational("Y", nde)
    assert v.worlds == frozenset({world_obs})


# ---------------------------------------------------------------------
# CFQuery
# ---------------------------------------------------------------------


def test_cfquery_construction(nde, world_x0, world_x1):
    """PNS-style joint event: (Y_0 = 1) ∧ (Y_1 = 0)."""
    v0 = CFVar.standard("Y", world_x0, nde)
    v1 = CFVar.standard("Y", world_x1, nde)
    q = CFQuery(events=((v0, 1), (v1, 0)), graph=nde)
    assert len(q.events) == 2
    assert q.events[0] == (v0, 1)
    assert q.events[1] == (v1, 0)


def test_cfquery_worlds_union(nde, world_x0, world_x1):
    v0 = CFVar.standard("Y", world_x0, nde)
    v1 = CFVar.standard("Y", world_x1, nde)
    q = CFQuery(events=((v0, 1), (v1, 0)), graph=nde)
    assert q.worlds == frozenset({world_x0, world_x1})


def test_cfquery_intervened_any(nde, world_x0, world_x1):
    v0 = CFVar.standard("Y", world_x0, nde)
    v1 = CFVar.standard("Y", world_x1, nde)
    q = CFQuery(events=((v0, 1), (v1, 0)), graph=nde)
    # Both worlds intervene on X only.
    assert q.intervened_any == frozenset({"X"})


def test_cfquery_labels_property(nde, world_x0, world_x1):
    v0 = CFVar.standard("Y", world_x0, nde)
    v1 = CFVar.standard("Y", world_x1, nde)
    q = CFQuery(events=((v0, 1), (v1, 0)), graph=nde)
    # Λ(Q) — the symbolic label set.
    assert q.labels == frozenset({"x0", "x1"})


def test_cfquery_single_event_observational(nde):
    """Q = P(Y = 1) — an observational joint event."""
    v = CFVar.observational("Y", nde)
    q = CFQuery(events=((v, 1),), graph=nde)
    assert q.worlds == frozenset({World.observational()})
    assert q.intervened_any == frozenset()


def test_cfquery_target_outside_domain_errors(nde, world_x0):
    """Target value must be in Omega_{base}."""
    v = CFVar.standard("Y", world_x0, nde)
    with pytest.raises(CFVarError):
        CFQuery(events=((v, 99),), graph=nde)


def test_cfquery_rejects_cfvar_from_different_graph(nde, iv, world_x1):
    """A CFVar built on IV should not silently expand on the NDE graph."""
    v = CFVar.standard("Y", world_x1, iv)
    with pytest.raises(CFVarError):
        CFQuery(events=((v, 1),), graph=nde)


def test_cfquery_is_frozen(nde, world_x0):
    v = CFVar.standard("Y", world_x0, nde)
    q = CFQuery(events=((v, 1),), graph=nde)
    with pytest.raises((AttributeError, TypeError)):
        q.events = ()  # type: ignore[misc]


def test_cfquery_equality(nde, world_x0):
    v = CFVar.standard("Y", world_x0, nde)
    a = CFQuery(events=((v, 1),), graph=nde)
    b = CFQuery(events=((v, 1),), graph=nde)
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------
# Integration: NDE L_{2.5} running example
# ---------------------------------------------------------------------


def test_nde_l25_running_example_query(nde, world_x0, world_x1):
    """Reconstruct Q = P(Y_{0, W_1} = 1) fully."""
    y_nested = CFVar.nested(
        base="Y",
        edge_worlds={
            ("X", "W"): world_x1,
            ("X", "Y"): world_x0,
            ("W", "Y"): world_x1,
        },
        G=nde,
    )
    q = CFQuery(events=((y_nested, 1),), graph=nde)
    assert q.labels == frozenset({"x0", "x1"})
    assert q.intervened_any == frozenset({"X"})
    # Only one CFVar in the joint event, but it uses two worlds.
    assert q.worlds == frozenset({world_x0, world_x1})
