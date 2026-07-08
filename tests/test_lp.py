"""Unit tests for canonred.lp (spec: docs/01_concepts.md §8).

CVXPY-backed LP solver. Tests target two invariants:
  - Closed-form correctness on the NDE L_{2.5} running example.
  - Agreement between the lifted LP and the reduced (signature-quotient) LP.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow  # cvxpy builds can be slow; gate the whole file.

cp = pytest.importorskip("cvxpy")

from canonred.counterfactual import CFQuery, CFVar
from canonred.examples import nde_graph
from canonred.lp import (  # noqa: E402
    LPCoefficients,
    build_lp_lifted,
    build_lp_reduced,
    solve_sharp_bounds,
)
from canonred.observation import l25_marginal_constraints
from canonred.signature import build_signature_quotient
from canonred.worlds import World


# ---------------------------------------------------------------------
# Fixture: NDE L_{2.5} running example
# ---------------------------------------------------------------------


@pytest.fixture
def nde():
    return nde_graph()


@pytest.fixture
def nde_l25_query(nde):
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    cf = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w1, ("X", "Y"): w0, ("W", "Y"): w1},
        G=nde,
    )
    return CFQuery(events=((cf, 1),), graph=nde, name="Y_{0,W_1}=1")


@pytest.fixture
def nde_l25_marginals(nde):
    w1 = World.from_dict("x1", {"X": 1})
    w00 = World.from_dict("x0w0", {"X": 0, "W": 0})
    w01 = World.from_dict("x0w1", {"X": 0, "W": 1})
    v_w1 = CFVar.standard("W", w1, nde)
    v_y00 = CFVar.standard("Y", w00, nde)
    v_y01 = CFVar.standard("Y", w01, nde)
    return l25_marginal_constraints([(v_w1, 1), (v_y00, 1), (v_y01, 1)])


# ---------------------------------------------------------------------
# LPCoefficients: build_lp_lifted / build_lp_reduced
# ---------------------------------------------------------------------


def test_build_lp_lifted_nde_shape(nde, nde_l25_query, nde_l25_marginals):
    coef = build_lp_lifted(nde, nde_l25_query, nde_l25_marginals)
    assert isinstance(coef, LPCoefficients)
    assert coef.c.shape == (128,)
    assert coef.M.shape == (3, 128)
    assert len(coef.row_labels) == 3


def test_build_lp_reduced_nde_shape(nde, nde_l25_query, nde_l25_marginals):
    quotient = build_signature_quotient(nde, nde_l25_query, nde_l25_marginals)
    coef = build_lp_reduced(quotient)
    assert coef.c.shape == (quotient.reduced_dim,)
    assert coef.M.shape == (3, quotient.reduced_dim)


# ---------------------------------------------------------------------
# Closed-form sanity on NDE L_{2.5}
# ---------------------------------------------------------------------


def _closed_form_bounds(p_w1: float, p_y00: float, p_y01: float) -> tuple[float, float]:
    """Paper §6 eq (closed-form-bounds)."""
    L = max(0.0, p_y00 - p_w1) + max(0.0, p_y01 - (1.0 - p_w1))
    U = min(p_y00, 1.0 - p_w1) + min(p_y01, p_w1)
    return L, U


def test_solve_sharp_bounds_reduced_matches_closed_form(nde, nde_l25_query, nde_l25_marginals):
    """For (p_w1, p_y00, p_y01) = (0.6, 0.3, 0.7), closed form gives [0.3, 0.9]."""
    quotient = build_signature_quotient(nde, nde_l25_query, nde_l25_marginals)
    coef = build_lp_reduced(quotient)
    p = np.array([0.6, 0.3, 0.7])

    lb, ub, _ = solve_sharp_bounds(coef, p)
    L_exp, U_exp = _closed_form_bounds(0.6, 0.3, 0.7)
    # LP solver tolerances can accumulate on larger lifted formulations; match
    # the paper's own "solver tolerance" convention.
    assert abs(lb - L_exp) < 1e-4
    assert abs(ub - U_exp) < 1e-4


# ---------------------------------------------------------------------
# Lifted / reduced LP bounds agree — this is the Prop 3.1 empirical check.
# ---------------------------------------------------------------------


def test_lifted_and_reduced_bounds_agree_on_grid(nde, nde_l25_query, nde_l25_marginals):
    """For multiple p values, lifted LP and reduced LP give identical bounds."""
    coef_l = build_lp_lifted(nde, nde_l25_query, nde_l25_marginals)
    quotient = build_signature_quotient(nde, nde_l25_query, nde_l25_marginals)
    coef_r = build_lp_reduced(quotient)

    grid = [
        (0.6, 0.3, 0.7),
        (0.5, 0.5, 0.5),
        (0.2, 0.4, 0.8),
        (0.9, 0.1, 0.3),
    ]
    for p_w1, p_y00, p_y01 in grid:
        p = np.array([p_w1, p_y00, p_y01])
        lb_l, ub_l, _ = solve_sharp_bounds(coef_l, p)
        lb_r, ub_r, _ = solve_sharp_bounds(coef_r, p)
        assert abs(lb_l - lb_r) < 1e-4, f"LB disagree at {p}: {lb_l} vs {lb_r}"
        assert abs(ub_l - ub_r) < 1e-4, f"UB disagree at {p}: {ub_l} vs {ub_r}"


def test_solve_returns_diagnostics(nde, nde_l25_query, nde_l25_marginals):
    """Diagnostics include wall-clock and solver time."""
    quotient = build_signature_quotient(nde, nde_l25_query, nde_l25_marginals)
    coef = build_lp_reduced(quotient)
    p = np.array([0.6, 0.3, 0.7])
    _, _, diag = solve_sharp_bounds(coef, p)
    assert "wall_time" in diag
    assert "solver_time" in diag
    assert diag["wall_time"] >= 0.0
