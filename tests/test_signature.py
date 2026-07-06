"""Unit tests for canonred.signature (spec: docs/01_concepts.md §9).

The coarsest safe quotient declares u ~ u' iff sigma(u) = sigma(u'), where
sigma(u) = (c_u, M_{:,u}). Prop 3.1 guarantees the reduced LP on this
quotient has the same sharp bounds as the full Omega_full LP.

Critical paper checkpoints (paper §6 Case 1 / Case 2, NDE running example):
  - (NDE, Y_{0,W_1}, L2.5 marginals) → reduced_dim = 8
  - (NDE, Y_{0,W_1}, P_obs)          → reduced_dim = 64
"""

import pytest

from canonred.counterfactual import CFQuery, CFVar
from canonred.examples import (
    bow_graph,
    frontdoor_graph,
    iv_graph,
    nde_graph,
    non_iv_graph,
    pns_graph,
    ternary_nde_graph,
)
from canonred.observation import l25_marginal_constraints, pobs_constraints
from canonred.signature import (
    SignatureQuotient,
    build_signature_quotient,
    build_signature_quotient_lazy_pobs,
)
from canonred.worlds import World


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def bow():
    return bow_graph()


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
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    w00 = World.from_dict("x0w0", {"X": 0, "W": 0})
    w01 = World.from_dict("x0w1", {"X": 0, "W": 1})
    v_w1 = CFVar.standard("W", w1, nde)
    v_y00 = CFVar.standard("Y", w00, nde)
    v_y01 = CFVar.standard("Y", w01, nde)
    return l25_marginal_constraints([(v_w1, 1), (v_y00, 1), (v_y01, 1)])


# ---------------------------------------------------------------------
# SignatureQuotient structure
# ---------------------------------------------------------------------


def test_quotient_is_frozen_dataclass(bow):
    """Construction returns a frozen SignatureQuotient with required fields."""
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    q = build_signature_quotient(bow, Q, constraints=())
    assert isinstance(q, SignatureQuotient)
    assert q.lifted_dim == 8  # bow |Omega_full|
    assert q.reduced_dim <= q.lifted_dim


def test_quotient_classes_partition_omega_full(bow):
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    q = build_signature_quotient(bow, Q, constraints=())
    total = sum(q.class_sizes)
    assert total == q.lifted_dim
    assert len(q.class_sizes) == q.reduced_dim


def test_state_class_assignment_length(bow):
    """state_class_assignment has one entry per lifted state."""
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    q = build_signature_quotient(bow, Q, constraints=())
    assert len(q.state_class_assignment) == q.lifted_dim


def test_state_class_assignment_values_in_range(bow):
    """Each entry is a class index in [0, reduced_dim)."""
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    q = build_signature_quotient(bow, Q, constraints=())
    for t in q.state_class_assignment:
        assert 0 <= t < q.reduced_dim


def test_state_class_assignment_consistent_with_class_sizes(bow):
    """Counts induced by state_class_assignment match class_sizes."""
    from collections import Counter

    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    q = build_signature_quotient(bow, Q, constraints=())
    counts = Counter(q.state_class_assignment)
    for t, sz in enumerate(q.class_sizes):
        assert counts[t] == sz


def test_quotient_no_constraints_binary_objective(bow):
    """Without constraints and with a binary indicator objective, the reduced
    space has at most 2 classes (one for c_u=0, one for c_u=1)."""
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    q = build_signature_quotient(bow, Q, constraints=())
    assert q.reduced_dim <= 2


def test_quotient_objective_vector_bounded_binary(bow):
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    q = build_signature_quotient(bow, Q, constraints=())
    assert len(q.objective) == q.reduced_dim
    assert set(q.objective).issubset({0, 1})


def test_quotient_constraint_matrix_shape(bow):
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    cs = pobs_constraints(bow)
    q = build_signature_quotient(bow, Q, constraints=cs)
    assert q.constraint_matrix.shape == (len(cs), q.reduced_dim)


# ---------------------------------------------------------------------
# Paper-verified reduced dimensions
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_nde_l25_reduced_dim_is_8(nde, nde_l25_query, nde_l25_marginals):
    """paper §6 Case 2: L_{2.5} marginals reduce NDE 128 -> 8 states."""
    q = build_signature_quotient(nde, nde_l25_query, constraints=nde_l25_marginals)
    assert q.lifted_dim == 128
    assert q.reduced_dim == 8


@pytest.mark.slow
def test_nde_pobs_reduced_dim_is_16(nde, nde_l25_query):
    """Coarsest safe quotient under P_obs constraints on NDE = 16.

    Paper §6 Case 1 shows a specific non-coarsest structural reduction to 64
    (dropping one bit of the response table), but the coarsest safe signature
    quotient is strictly tighter: each state maps to exactly one of 8
    observation cells, and the query indicator contributes 1 extra bit,
    giving at most 2*8 = 16 distinct signatures — all realized.
    """
    cs = pobs_constraints(nde)
    q = build_signature_quotient(nde, nde_l25_query, constraints=cs)
    assert q.lifted_dim == 128
    assert q.reduced_dim == 16


# ---------------------------------------------------------------------
# Correctness: pushforward relations c = Pi^T c_tilde and M = M_tilde Pi
# ---------------------------------------------------------------------


def test_pushforward_objective_reconstruction_bow(bow):
    """For each lifted state u in class t, c_u = c_tilde[t]. (Prop 3.1 invariant.)"""
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    cs = pobs_constraints(bow)
    q = build_signature_quotient(bow, Q, constraints=cs)
    for class_idx, cls_c in enumerate(q.objective):
        for u_c in q.class_lifted_c[class_idx]:
            assert u_c == cls_c


def test_pushforward_constraints_reconstruction_bow(bow):
    """For each lifted state u in class t and each constraint i, M[i,u] = M_tilde[i,t]."""
    w = World.from_dict("x0", {"X": 0})
    Q = CFQuery(events=((CFVar.standard("Y", w, bow), 1),), graph=bow)
    cs = pobs_constraints(bow)
    q = build_signature_quotient(bow, Q, constraints=cs)
    n_cons = q.constraint_matrix.shape[0]
    for class_idx in range(q.reduced_dim):
        cls_col = q.constraint_matrix[:, class_idx]
        for row_u in q.class_lifted_M[class_idx]:
            for i in range(n_cons):
                assert cls_col[i] == row_u[i]


# ---------------------------------------------------------------------
# Lazy pobs builder — equivalence with full enumeration
# ---------------------------------------------------------------------


def _bow_query(g):
    w = World.from_dict("x1", {"X": 1})
    return CFQuery(events=((CFVar.standard("Y", w, g), 1),), graph=g, name="bow_Y_x1")


def _iv_like_query(g):
    w = World.from_dict("x1", {"X": 1})
    return CFQuery(events=((CFVar.standard("Y", w, g), 1),), graph=g, name="iv_like_Y_x1")


def _nde_running_query(g):
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    cf = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w1, ("X", "Y"): w0, ("W", "Y"): w1},
        G=g,
    )
    return CFQuery(events=((cf, 1),), graph=g, name="nde_running")


def _pns_query(g):
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    return CFQuery(
        events=((CFVar.standard("Y", w0, g), 1), (CFVar.standard("Y", w1, g), 0)),
        graph=g,
        name="pns",
    )


def _ternary_query(g):
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    cf = CFVar.nested(
        base="Y",
        edge_worlds={("Z", "X"): w0, ("X", "M"): w0, ("X", "Y"): w1, ("M", "Y"): w0},
        G=g,
    )
    return CFQuery(events=((cf, 1),), graph=g, name="ternary_running")


@pytest.mark.parametrize(
    "graph_factory,query_factory,expected_lifted,expected_reduced",
    [
        (bow_graph, _bow_query, 8, 6),
        (iv_graph, _iv_like_query, 32, 12),
        (non_iv_graph, _iv_like_query, 32, 12),
        (frontdoor_graph, _iv_like_query, 32, 12),
        (nde_graph, _nde_running_query, 128, 16),
        (pns_graph, _pns_query, 128, 12),
        pytest.param(
            ternary_nde_graph,
            _ternary_query,
            4608,
            48,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_lazy_pobs_matches_full_quotient_dimensions(
    graph_factory, query_factory, expected_lifted, expected_reduced,
):
    """Lazy pobs builder produces the same lifted_dim and reduced_dim as full."""
    g = graph_factory()
    Q = query_factory(g)
    cs = pobs_constraints(g)
    full = build_signature_quotient(g, Q, cs)
    lazy = build_signature_quotient_lazy_pobs(g, Q)
    assert full.lifted_dim == expected_lifted == lazy.lifted_dim
    assert full.reduced_dim == expected_reduced == lazy.reduced_dim


@pytest.mark.parametrize(
    "graph_factory,query_factory",
    [
        (bow_graph, _bow_query),
        (iv_graph, _iv_like_query),
        (nde_graph, _nde_running_query),
        (pns_graph, _pns_query),
    ],
)
def test_lazy_pobs_lp_bounds_match_full(graph_factory, query_factory):
    """Lazy LP gives same sharp bounds as full LP on K=3 random feasible p."""
    import numpy as np  # local import keeps top of file clean

    from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced

    g = graph_factory()
    Q = query_factory(g)
    cs = pobs_constraints(g)
    full = build_signature_quotient(g, Q, cs)
    lazy = build_signature_quotient_lazy_pobs(g, Q)
    coef_lifted = build_lp_lifted(g, Q, cs)
    coef_full = build_lp_reduced(full)
    coef_lazy = build_lp_reduced(lazy)

    s_lift = CachedSharpLP(coef_lifted)
    s_full = CachedSharpLP(coef_full)
    s_lazy = CachedSharpLP(coef_lazy)

    warm = np.full(coef_lifted.n_constraints, 1.0 / coef_lifted.n_constraints)
    s_lift.solve(warm)
    s_full.solve(warm)
    s_lazy.solve(warm)

    rng = np.random.default_rng(42)
    for _ in range(3):
        alpha = rng.dirichlet(np.ones(coef_lifted.dim))
        p = coef_lifted.M @ alpha
        L_lift, U_lift, _ = s_lift.solve(p)
        L_lazy, U_lazy, _ = s_lazy.solve(p)
        L_full, U_full, _ = s_full.solve(p)
        assert abs(L_lift - L_lazy) < 1e-7
        assert abs(U_lift - U_lazy) < 1e-7
        assert abs(L_full - L_lazy) < 1e-7
        assert abs(U_full - U_lazy) < 1e-7


@pytest.mark.slow
def test_lazy_pobs_handles_v6_random_admg():
    """Spot-check on a random ADMG that lazy build is fast and yields correct dims.

    Lifted ≤ 4096 here so we can also build the full-enumeration quotient and
    cross-check.
    """
    import numpy as np

    from canonred.random_admg import random_admg

    rng = np.random.default_rng(4042)
    g = random_admg(6, rng=rng, p_dir=0.4, p_bidir=0.3)
    topo = g.topo_order()
    y = topo[-1]
    anc = g.ancestors({y}, inclusive=False)
    x = min(anc, key=lambda v: topo.index(v)) if anc else topo[0]
    w0 = World.from_dict("x0", {x: 0})
    w1 = World.from_dict("x1", {x: 1})
    Q = CFQuery(
        events=((CFVar.standard(y, w0, g), 1), (CFVar.standard(y, w1, g), 1)),
        graph=g,
        name="rand_v6_cross",
    )
    lifted = g.lifted_canonical_size()
    if lifted > 4096:
        pytest.skip(
            f"random fixture lifted_dim={lifted} exceeds explicit cross-check cap"
        )
    cs = pobs_constraints(g)
    full = build_signature_quotient(g, Q, cs)
    lazy = build_signature_quotient_lazy_pobs(g, Q)
    assert full.reduced_dim == lazy.reduced_dim


def test_lazy_pobs_handles_v8_intractable_lifted():
    """Lazy builder succeeds where lifted enumeration is hopeless (|V|=8)."""
    import numpy as np

    from canonred.random_admg import random_admg

    rng = np.random.default_rng(4042)
    g = random_admg(8, rng=rng, p_dir=0.4, p_bidir=0.3)
    topo = g.topo_order()
    y = topo[-1]
    anc = g.ancestors({y}, inclusive=False)
    x = min(anc, key=lambda v: topo.index(v)) if anc else topo[0]
    w0 = World.from_dict("x0", {x: 0})
    w1 = World.from_dict("x1", {x: 1})
    Q = CFQuery(
        events=((CFVar.standard(y, w0, g), 1), (CFVar.standard(y, w1, g), 1)),
        graph=g,
        name="rand_v8_cross",
    )
    # Lifted size for V=8 binary at this density typically exceeds 1e8 — too large
    # for explicit enumeration. The lazy builder must still succeed and produce a
    # bounded reduced_dim ≤ 2 * prod_V d_V = 2 * 256 = 512.
    lazy = build_signature_quotient_lazy_pobs(g, Q)
    assert lazy.lifted_dim == g.lifted_canonical_size()
    assert 1 <= lazy.reduced_dim <= 2 * (2 ** 8)
