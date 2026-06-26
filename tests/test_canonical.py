"""Unit tests for canonred.canonical (spec: docs/01_concepts.md §5, §6).

Responsibilities:
  - ResponseFn: a deterministic f_V encoded as a lookup table over parent-value
    combinations (row-major flat index).
  - CanonicalState: a (VarName -> ResponseFn) map.
  - enumerate_states(G): iterate all states in Omega_full.
  - state_cardinality(G): |Omega_full|.
  - eval_cfvar(u, cfvar, G): propagate CFVar through G*.
  - eval_query_indicator(u, Q, G): 0/1 indicator of the joint event.
"""

import pytest

from canonred.canonical import (
    CanonicalState,
    ResponseFn,
    enumerate_states,
    eval_cfvar,
    eval_query_indicator,
    state_cardinality,
)
from canonred.counterfactual import CFQuery, CFVar
from canonred.examples import bow_graph, iv_graph, nde_graph
from canonred.worlds import World


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def bow():
    return bow_graph()


@pytest.fixture
def iv():
    return iv_graph()


@pytest.fixture
def nde():
    return nde_graph()


# ---------------------------------------------------------------------
# ResponseFn: tabular callable
# ---------------------------------------------------------------------


def test_responsefn_root_no_parents():
    """A root (no parents) has a 1-entry table; f() returns the fixed value."""
    f = ResponseFn(var="X", parent_cards=(), codomain=2, table=(1,))
    assert f() == 1


def test_responsefn_one_parent_binary():
    """1 binary parent → 2-entry table."""
    f = ResponseFn(var="Y", parent_cards=(2,), codomain=2, table=(0, 1))
    assert f(0) == 0
    assert f(1) == 1


def test_responsefn_two_parents_binary_rowmajor():
    """2 binary parents → 4-entry table, flat index = p0*2 + p1."""
    # table[p0, p1] laid out row-major: (0,0), (0,1), (1,0), (1,1)
    f = ResponseFn(var="Y", parent_cards=(2, 2), codomain=2, table=(0, 1, 1, 0))
    assert f(0, 0) == 0
    assert f(0, 1) == 1
    assert f(1, 0) == 1
    assert f(1, 1) == 0


def test_responsefn_ternary_output():
    """Codomain 3 tabular function."""
    f = ResponseFn(var="M", parent_cards=(2,), codomain=3, table=(0, 2))
    assert f(0) == 0
    assert f(1) == 2


def test_responsefn_table_length_mismatch_errors():
    """Table length must equal prod(parent_cards)."""
    with pytest.raises(ValueError):
        ResponseFn(var="Y", parent_cards=(2, 2), codomain=2, table=(0, 1))


def test_responsefn_table_entry_out_of_codomain_errors():
    with pytest.raises(ValueError):
        ResponseFn(var="Y", parent_cards=(2,), codomain=2, table=(0, 2))


def test_responsefn_codomain_below_two_errors():
    with pytest.raises(ValueError):
        ResponseFn(var="Y", parent_cards=(), codomain=1, table=(0,))


def test_responsefn_is_frozen():
    f = ResponseFn(var="Y", parent_cards=(2,), codomain=2, table=(0, 1))
    with pytest.raises((AttributeError, TypeError)):
        f.var = "Z"  # type: ignore[misc]


def test_responsefn_argc_mismatch_errors():
    """Calling with wrong arity is an error."""
    f = ResponseFn(var="Y", parent_cards=(2,), codomain=2, table=(0, 1))
    with pytest.raises(ValueError):
        f(0, 0)


# ---------------------------------------------------------------------
# CanonicalState
# ---------------------------------------------------------------------


def test_canonical_state_roundtrip(bow):
    """Construct a specific state on bow graph; look up response fns."""
    fx = ResponseFn(var="X", parent_cards=(), codomain=2, table=(1,))
    fy = ResponseFn(var="Y", parent_cards=(2,), codomain=2, table=(1, 0))
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    assert u.funcs["X"] is fx
    assert u.funcs["Y"] is fy


def test_canonical_state_accepts_incomplete_funcs(bow):
    """CanonicalState is graph-free. Construction with missing vars does NOT
    raise; domain validation happens at use time (eval_cfvar).
    """
    fx = ResponseFn(var="X", parent_cards=(), codomain=2, table=(0,))
    u = CanonicalState(funcs={"X": fx})  # Y is missing — allowed
    assert u.funcs["X"] is fx


# ---------------------------------------------------------------------
# enumerate_states / state_cardinality
# ---------------------------------------------------------------------


def test_state_cardinality_bow(bow):
    """|Omega_full| for bow = 2 * 4 = 8."""
    assert state_cardinality(bow) == 8


def test_state_cardinality_iv(iv):
    """|Omega_full| for IV = 2 * 4 * 4 = 32."""
    assert state_cardinality(iv) == 32


def test_state_cardinality_nde(nde):
    """|Omega_full| for NDE = 2 * 4 * 16 = 128."""
    assert state_cardinality(nde) == 128


def test_enumerate_states_count_matches_cardinality(bow):
    """enumerate_states yields exactly `state_cardinality(G)` distinct states."""
    states = list(enumerate_states(bow))
    assert len(states) == state_cardinality(bow)


def test_enumerate_states_are_distinct(bow):
    """Every enumerated state is unique."""
    states = list(enumerate_states(bow))
    fingerprints = {(s.funcs["X"].table, s.funcs["Y"].table) for s in states}
    assert len(fingerprints) == len(states)


def test_enumerate_states_iv_count(iv):
    """Exhaustive enumeration on IV reaches exactly 32 states."""
    count = sum(1 for _ in enumerate_states(iv))
    assert count == 32


# ---------------------------------------------------------------------
# eval_cfvar: bow graph
# ---------------------------------------------------------------------


def test_eval_obs_on_bow_root_x(bow):
    """Observational X = f_X()."""
    fx = ResponseFn("X", (), 2, (1,))
    fy = ResponseFn("Y", (2,), 2, (0, 0))
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    x = CFVar.observational("X", bow)
    assert eval_cfvar(u, x, bow) == 1


def test_eval_self_intervention_on_bow_root_x(bow):
    """X_{x=0} is fixed by the intervention, not by the natural root f_X()."""
    fx = ResponseFn("X", (), 2, (1,))
    fy = ResponseFn("Y", (2,), 2, (0, 0))
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    w0 = World.from_dict("x0", {"X": 0})
    x_x0 = CFVar.standard("X", w0, bow)
    assert eval_cfvar(u, x_x0, bow) == 0


def test_eval_self_intervention_on_bow_y(bow):
    """Y_{y=1} is fixed even when Y has parents and f_Y would return 0."""
    fx = ResponseFn("X", (), 2, (0,))
    fy = ResponseFn("Y", (2,), 2, (0, 0))
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    w1 = World.from_dict("y1", {"Y": 1})
    y_y1 = CFVar.standard("Y", w1, bow)
    assert eval_cfvar(u, y_y1, bow) == 1


def test_eval_obs_on_bow_y_uses_fx(bow):
    """Observational Y = f_Y(f_X())."""
    fx = ResponseFn("X", (), 2, (1,))        # X=1
    fy = ResponseFn("Y", (2,), 2, (0, 1))    # Y=0 if X=0, Y=1 if X=1
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    y = CFVar.observational("Y", bow)
    assert eval_cfvar(u, y, bow) == 1


def test_eval_yx_on_bow_ignores_fx(bow):
    """Y_{x=0} = f_Y(0) regardless of f_X."""
    w0 = World.from_dict("x0", {"X": 0})
    fx = ResponseFn("X", (), 2, (1,))        # natural X = 1, but should be overridden
    fy = ResponseFn("Y", (2,), 2, (3 % 2, 1))  # equivalent: (1, 1). Use non-trivial to be safe.
    # Actually simplify:
    fy = ResponseFn("Y", (2,), 2, (0, 1))    # Y_0=0, Y_1=1
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    y_x0 = CFVar.standard("Y", w0, bow)
    assert eval_cfvar(u, y_x0, bow) == 0    # f_Y(0) = 0 despite f_X()=1


# ---------------------------------------------------------------------
# eval_cfvar: NDE with Y_{0, W_1}
# ---------------------------------------------------------------------


def test_eval_nde_y_0_w1_hand_worked_state(nde):
    """Pick a specific state and verify Y_{0, W_1} evaluates as expected.

    State:
      f_X() = 1           (irrelevant: X is intervened)
      f_W(X=0) = 0
      f_W(X=1) = 1        -> W_1 = 1
      f_Y(X=0, W=0) = 0
      f_Y(X=0, W=1) = 1   -> Y_{0, W_1} = f_Y(0, 1) = 1
      f_Y(X=1, W=0) = ?   (irrelevant)
      f_Y(X=1, W=1) = ?
    """
    fx = ResponseFn("X", (), 2, (1,))
    fw = ResponseFn("W", (2,), 2, (0, 1))               # W_0=0, W_1=1
    # Y's parents in NDE: order = (X, W) per G.parents("Y") which returns nodes-order.
    # Verify:
    assert nde.parents("Y") == ("X", "W")
    # table row-major: (X=0,W=0), (0,1), (1,0), (1,1)
    fy = ResponseFn("Y", (2, 2), 2, (0, 1, 0, 1))       # f_Y(0,0)=0, f_Y(0,1)=1, ...
    u = CanonicalState(funcs={"X": fx, "W": fw, "Y": fy})

    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    y_nested = CFVar.nested(
        base="Y",
        edge_worlds={
            ("X", "W"): w1,
            ("X", "Y"): w0,
            ("W", "Y"): w1,
        },
        G=nde,
    )
    assert eval_cfvar(u, y_nested, nde) == 1


# ---------------------------------------------------------------------
# eval_query_indicator
# ---------------------------------------------------------------------


def test_eval_query_indicator_true(bow):
    """Q = P(Y_{x=0}=0). With state that makes Y_0=0, indicator=1."""
    fx = ResponseFn("X", (), 2, (0,))
    fy = ResponseFn("Y", (2,), 2, (0, 1))
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    w = World.from_dict("x0", {"X": 0})
    v = CFVar.standard("Y", w, bow)
    Q = CFQuery(events=((v, 0),), graph=bow)
    assert eval_query_indicator(u, Q, bow) == 1


def test_eval_query_indicator_false(bow):
    fx = ResponseFn("X", (), 2, (0,))
    fy = ResponseFn("Y", (2,), 2, (1, 1))  # Y is always 1
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    w = World.from_dict("x0", {"X": 0})
    v = CFVar.standard("Y", w, bow)
    Q = CFQuery(events=((v, 0),), graph=bow)
    assert eval_query_indicator(u, Q, bow) == 0


def test_eval_query_indicator_conjunction_true(bow):
    """Joint event (Y_{x=0}=0) AND (Y_{x=1}=1) — PNS-style on bow."""
    fx = ResponseFn("X", (), 2, (0,))
    fy = ResponseFn("Y", (2,), 2, (0, 1))  # Y_0=0, Y_1=1
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    v0 = CFVar.standard("Y", w0, bow)
    v1 = CFVar.standard("Y", w1, bow)
    Q = CFQuery(events=((v0, 0), (v1, 1)), graph=bow)
    assert eval_query_indicator(u, Q, bow) == 1


def test_eval_query_indicator_conjunction_false(bow):
    """One conjunct fails → indicator 0."""
    fx = ResponseFn("X", (), 2, (0,))
    fy = ResponseFn("Y", (2,), 2, (0, 0))  # Y_0=0, Y_1=0
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    v0 = CFVar.standard("Y", w0, bow)
    v1 = CFVar.standard("Y", w1, bow)
    Q = CFQuery(events=((v0, 0), (v1, 1)), graph=bow)  # v1=1 target, but Y_1=0
    assert eval_query_indicator(u, Q, bow) == 0


# ---------------------------------------------------------------------
# Integration: sum of indicators over Omega_full equals closed-form counts
# ---------------------------------------------------------------------


def test_total_indicator_sum_on_bow():
    """Over all 8 states of bow, sum of indicator for (X=0) obs should equal 4.

    Because f_X is independent of parents and has 2 states (X=0 or X=1), half
    of the 8-state product (X: 2 values * Y: 4 tables = 8 total) has X=0 → 4.
    """
    G = bow_graph()
    w_obs = World.observational()
    x_obs = CFVar.observational("X", G)
    Q = CFQuery(events=((x_obs, 0),), graph=G)
    total = sum(eval_query_indicator(u, Q, G) for u in enumerate_states(G))
    assert total == 4
