"""Unit tests for canonred.observation (spec: docs/01_concepts.md §7).

A MomentConstraint represents one linear functional on psi: the indicator that
a specified conjunction of counterfactual events holds at a canonical state u.

Factories:
  - pobs_constraints(G): the full observational distribution P_obs — one
    MomentConstraint per observable cell (tuple over Omega_V).
  - l25_marginal_constraints(cfvars, targets=1): one constraint per CFVar
    marginal.
"""

import pytest

from canonred.canonical import CanonicalState, ResponseFn
from canonred.counterfactual import CFVar
from canonred.examples import bow_graph, nde_graph
from canonred.observation import (
    MomentConstraint,
    l25_marginal_constraints,
    pobs_constraints,
)
from canonred.worlds import World


@pytest.fixture
def bow():
    return bow_graph()


@pytest.fixture
def nde():
    return nde_graph()


# ---------------------------------------------------------------------
# MomentConstraint basics
# ---------------------------------------------------------------------


def test_moment_constraint_indicator_true(bow):
    """M[row, u] = 1 when the conjunctive event holds at u."""
    v_x = CFVar.observational("X", bow)
    c = MomentConstraint(name="X=1", events=((v_x, 1),))
    fx = ResponseFn("X", (), 2, (1,))
    fy = ResponseFn("Y", (2,), 2, (0, 0))
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    assert c.evaluate(u, bow) == 1


def test_moment_constraint_indicator_false(bow):
    v_x = CFVar.observational("X", bow)
    c = MomentConstraint(name="X=1", events=((v_x, 1),))
    fx = ResponseFn("X", (), 2, (0,))
    fy = ResponseFn("Y", (2,), 2, (0, 0))
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    assert c.evaluate(u, bow) == 0


def test_moment_constraint_conjunction(bow):
    """Conjunctive events: (X=0) ∧ (Y=1) observed."""
    v_x = CFVar.observational("X", bow)
    v_y = CFVar.observational("Y", bow)
    c = MomentConstraint(name="X=0,Y=1", events=((v_x, 0), (v_y, 1)))
    fx = ResponseFn("X", (), 2, (0,))
    fy = ResponseFn("Y", (2,), 2, (1, 1))  # Y always 1
    u = CanonicalState(funcs={"X": fx, "Y": fy})
    assert c.evaluate(u, bow) == 1


def test_moment_constraint_empty_events_errors():
    """A constraint with no events is degenerate."""
    with pytest.raises(ValueError):
        MomentConstraint(name="", events=())


def test_moment_constraint_is_frozen(bow):
    v = CFVar.observational("X", bow)
    c = MomentConstraint(name="c", events=((v, 1),))
    with pytest.raises((AttributeError, TypeError)):
        c.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------
# pobs_constraints
# ---------------------------------------------------------------------


def test_pobs_bow_has_4_cells(bow):
    """P_obs(X, Y) on binary bow has 4 cells: (0,0),(0,1),(1,0),(1,1)."""
    cs = pobs_constraints(bow)
    assert len(cs) == 4


def test_pobs_nde_has_8_cells(nde):
    """P_obs(X, W, Y) on binary NDE = 8 cells."""
    cs = pobs_constraints(nde)
    assert len(cs) == 8


def test_pobs_constraints_named_by_cell(bow):
    cs = pobs_constraints(bow)
    names = {c.name for c in cs}
    # Names should uniquely identify the cell (e.g., "P_obs(X=0,Y=1)").
    assert len(names) == 4


def test_pobs_each_cell_is_one_event_per_node(bow):
    cs = pobs_constraints(bow)
    for c in cs:
        # bow has 2 nodes X, Y → each cell event should have 2 cfvar-value pairs.
        assert len(c.events) == 2


def test_pobs_total_indicator_sum_is_one_per_state(bow):
    """Every canonical state satisfies exactly one pobs cell."""
    from canonred.canonical import enumerate_states
    cs = pobs_constraints(bow)
    for u in enumerate_states(bow):
        total = sum(c.evaluate(u, bow) for c in cs)
        assert total == 1


# ---------------------------------------------------------------------
# l25_marginal_constraints
# ---------------------------------------------------------------------


def test_l25_marginal_single_cfvar(bow):
    """One CFVar → one constraint for the target."""
    v = CFVar.observational("X", bow)
    cs = l25_marginal_constraints([(v, 1)])
    assert len(cs) == 1
    assert cs[0].events == ((v, 1),)


def test_l25_marginal_nde_running(nde):
    """1086 running example: marginals of W_1, Y_{0,0}, Y_{0,1}, each =1."""
    w1 = World.from_dict("x1", {"X": 1})
    w00 = World.from_dict("x0", {"X": 0})
    # Build CFVars for W_1, Y_{0,0}, Y_{0,1}.
    # W_1: standard with w=x1, base W.
    v_w1 = CFVar.standard("W", w1, nde)
    # Y_{0,0}: need to fix both X=0 and W=0. But NDE W isn't an intervenable
    # parent of Y via a CFVar.standard alone; we'd need a nested form or
    # treat Y_{0,0} as the quantity computed under a world that fixes both.
    w0_and_w0 = World.from_dict("x0w0", {"X": 0, "W": 0})
    v_y00 = CFVar.standard("Y", w0_and_w0, nde)
    w0_and_w1 = World.from_dict("x0w1", {"X": 0, "W": 1})
    v_y01 = CFVar.standard("Y", w0_and_w1, nde)

    cs = l25_marginal_constraints([(v_w1, 1), (v_y00, 1), (v_y01, 1)])
    assert len(cs) == 3
