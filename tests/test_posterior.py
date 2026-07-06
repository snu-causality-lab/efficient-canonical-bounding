"""Unit tests for canonred.posterior (spec: docs/01_concepts.md — posterior
sampling for phi on the reduced simplex; lift helpers for phi ↔ psi).

Focus:
  - dirichlet_posterior(counts, alpha, S): closed-form sampler for Regime II
    of the paper (direct multinomial query-state observations).
  - lift_phi_uniform(phi, quotient): deterministic uniform-within-fiber lift.
  - lift_phi_dirichlet(phi, quotient, alpha, rng): random Dirichlet lift
    with per-fiber Dir(alpha/|fiber|) concentrations.

Key invariants:
  - For any lift map, Pi @ psi == phi  (aggregation preserved).
  - For the sharp-bound LP on NDE L_{2.5}, Q_8 = g^T phi == Q_128 = c^T psi
    for ANY valid lift (since g is constant on fibers).
"""

import numpy as np
import pytest

from canonred.counterfactual import CFQuery, CFVar
from canonred.examples import bow_graph, nde_graph
from canonred.observation import l25_marginal_constraints
from canonred.posterior import (
    binomial_marginal_gibbs,
    dirichlet_posterior,
    lift_phi_dirichlet,
    lift_phi_uniform,
)
from canonred.signature import build_signature_quotient
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
def bow_quotient(bow):
    """Trivial query Q = P(Y_{x=0}=1) + no constraints → few classes, nice for tests."""
    w = World.from_dict("x0", {"X": 0})
    v = CFVar.standard("Y", w, bow)
    Q = CFQuery(events=((v, 1),), graph=bow)
    return build_signature_quotient(bow, Q, constraints=())


@pytest.fixture
def nde_l25_quotient(nde):
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    cf = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w1, ("X", "Y"): w0, ("W", "Y"): w1},
        G=nde,
    )
    Q = CFQuery(events=((cf, 1),), graph=nde, name="Y_{0,W_1}")
    w00 = World.from_dict("x0w0", {"X": 0, "W": 0})
    w01 = World.from_dict("x0w1", {"X": 0, "W": 1})
    v_w1 = CFVar.standard("W", w1, nde)
    v_y00 = CFVar.standard("Y", w00, nde)
    v_y01 = CFVar.standard("Y", w01, nde)
    cs = l25_marginal_constraints([(v_w1, 1), (v_y00, 1), (v_y01, 1)])
    return build_signature_quotient(nde, Q, cs)


# ---------------------------------------------------------------------
# dirichlet_posterior
# ---------------------------------------------------------------------


def test_dirichlet_posterior_shape_and_simplex(bow_quotient):
    """Output shape (S, reduced_dim) and each row is on the simplex."""
    counts = np.zeros(bow_quotient.reduced_dim, dtype=int)
    counts[0] = 3
    samples = dirichlet_posterior(counts, alpha=1.0, S=50, seed=0)
    assert samples.shape == (50, bow_quotient.reduced_dim)
    # Simplex: non-negative, rows sum to 1.
    assert (samples >= 0).all()
    np.testing.assert_allclose(samples.sum(axis=1), 1.0, atol=1e-10)


def test_dirichlet_posterior_seed_determinism(bow_quotient):
    counts = np.ones(bow_quotient.reduced_dim, dtype=int)
    a = dirichlet_posterior(counts, alpha=1.0, S=20, seed=42)
    b = dirichlet_posterior(counts, alpha=1.0, S=20, seed=42)
    np.testing.assert_array_equal(a, b)


def test_dirichlet_posterior_shifts_mass_to_observed(bow_quotient):
    """Large counts on one class → posterior concentrates there."""
    d = bow_quotient.reduced_dim
    counts = np.zeros(d, dtype=int)
    counts[0] = 10_000
    samples = dirichlet_posterior(counts, alpha=1.0, S=200, seed=1)
    # Mean should be close to the MLE p_0 ≈ 1.
    assert samples[:, 0].mean() > 0.99


# ---------------------------------------------------------------------
# lift_phi_uniform
# ---------------------------------------------------------------------


def test_lift_phi_uniform_aggregation(bow_quotient):
    """Pi @ psi == phi : aggregating the uniform lift recovers phi."""
    rng = np.random.default_rng(0)
    phi = rng.dirichlet(np.ones(bow_quotient.reduced_dim))
    psi = lift_phi_uniform(phi, bow_quotient)
    assert psi.shape == (bow_quotient.lifted_dim,)
    # Build aggregation: sum psi over states assigned to each class.
    aggregated = np.zeros_like(phi)
    for i, t in enumerate(bow_quotient.state_class_assignment):
        aggregated[t] += psi[i]
    np.testing.assert_allclose(aggregated, phi, atol=1e-12)


def test_lift_phi_uniform_is_on_simplex(bow_quotient):
    rng = np.random.default_rng(1)
    phi = rng.dirichlet(np.ones(bow_quotient.reduced_dim))
    psi = lift_phi_uniform(phi, bow_quotient)
    assert (psi >= -1e-12).all()
    assert abs(psi.sum() - 1.0) < 1e-10


def test_lift_phi_uniform_within_fiber_is_constant(bow_quotient):
    """All states in the same class get the same psi mass."""
    rng = np.random.default_rng(2)
    phi = rng.dirichlet(np.ones(bow_quotient.reduced_dim))
    psi = lift_phi_uniform(phi, bow_quotient)
    # Group psi by class and check within-class stdev is 0.
    per_class: dict[int, list[float]] = {}
    for i, t in enumerate(bow_quotient.state_class_assignment):
        per_class.setdefault(t, []).append(psi[i])
    for t, values in per_class.items():
        assert max(values) - min(values) < 1e-14


# ---------------------------------------------------------------------
# lift_phi_dirichlet (random within-fiber lift)
# ---------------------------------------------------------------------


def test_lift_phi_dirichlet_aggregation(bow_quotient):
    """Random lift still satisfies Pi @ psi == phi (deterministically)."""
    rng = np.random.default_rng(3)
    phi = rng.dirichlet(np.ones(bow_quotient.reduced_dim))
    psi = lift_phi_dirichlet(phi, bow_quotient, alpha=1.0, rng=np.random.default_rng(10))
    aggregated = np.zeros_like(phi)
    for i, t in enumerate(bow_quotient.state_class_assignment):
        aggregated[t] += psi[i]
    np.testing.assert_allclose(aggregated, phi, atol=1e-12)


def test_lift_phi_dirichlet_on_simplex(bow_quotient):
    rng = np.random.default_rng(4)
    phi = rng.dirichlet(np.ones(bow_quotient.reduced_dim))
    psi = lift_phi_dirichlet(phi, bow_quotient, alpha=1.0, rng=np.random.default_rng(11))
    assert (psi >= -1e-12).all()
    assert abs(psi.sum() - 1.0) < 1e-10


def test_lift_phi_dirichlet_seed_determinism(bow_quotient):
    rng = np.random.default_rng(5)
    phi = rng.dirichlet(np.ones(bow_quotient.reduced_dim))
    psi_a = lift_phi_dirichlet(phi, bow_quotient, alpha=1.0, rng=np.random.default_rng(99))
    psi_b = lift_phi_dirichlet(phi, bow_quotient, alpha=1.0, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(psi_a, psi_b)


# ---------------------------------------------------------------------
# Prop 3.1 empirical: Q_8 = Q_128 (via any valid lift) on NDE L_{2.5}
# ---------------------------------------------------------------------


def test_nde_l25_q8_equals_q128_via_uniform_lift(nde_l25_quotient):
    """For any phi, Q_8 = g_tilde^T phi == c^T psi = Q_128 (uniform lift)."""
    q = nde_l25_quotient
    rng = np.random.default_rng(7)
    phi = rng.dirichlet(np.ones(q.reduced_dim))

    # Q_8 on the reduced simplex:
    Q_8 = float(q.objective @ phi)

    # Q_128 on the lifted simplex — we need the lifted c vector.
    # Build it from state_class_assignment: c_lifted[i] = c_tilde[class_of(i)].
    c_lifted = q.objective[list(q.state_class_assignment)]
    psi = lift_phi_uniform(phi, q)
    Q_128 = float(c_lifted @ psi)
    assert abs(Q_8 - Q_128) < 1e-10


def test_nde_l25_q8_equals_q128_via_dirichlet_lift(nde_l25_quotient):
    """Same with random Dirichlet lift — Prop 3.1 says any valid lift agrees."""
    q = nde_l25_quotient
    rng = np.random.default_rng(8)
    phi = rng.dirichlet(np.ones(q.reduced_dim))
    Q_8 = float(q.objective @ phi)
    c_lifted = q.objective[list(q.state_class_assignment)]
    psi = lift_phi_dirichlet(phi, q, alpha=1.0, rng=np.random.default_rng(20))
    Q_128 = float(c_lifted @ psi)
    assert abs(Q_8 - Q_128) < 1e-10


# ---------------------------------------------------------------------
# binomial_marginal_gibbs (Regime I): allocation-augmented Gibbs
# ---------------------------------------------------------------------


def test_gibbs_shape_and_simplex(nde_l25_quotient):
    """Output shape (S, reduced_dim); each row on the simplex."""
    # Three fake binomial observations (K, n) for the three L_{2.5} marginals.
    obs = [(60, 100), (30, 100), (70, 100)]
    samples = binomial_marginal_gibbs(
        nde_l25_quotient, observations=obs, alpha=1.0, S=100, burn=50, seed=0
    )
    assert samples.shape == (100, nde_l25_quotient.reduced_dim)
    assert (samples >= 0).all()
    np.testing.assert_allclose(samples.sum(axis=1), 1.0, atol=1e-10)


def test_gibbs_deterministic_under_seed(nde_l25_quotient):
    obs = [(60, 100), (30, 100), (70, 100)]
    a = binomial_marginal_gibbs(nde_l25_quotient, observations=obs, S=50, burn=20, seed=7)
    b = binomial_marginal_gibbs(nde_l25_quotient, observations=obs, S=50, burn=20, seed=7)
    np.testing.assert_array_equal(a, b)


def test_gibbs_recovers_marginals_for_large_n(nde_l25_quotient):
    """For n=10000 counts consistent with p=(0.6, 0.3, 0.7), posterior mean of
    each marginal should land near the observed rate.

    We read each marginal as sum_{t : M_tilde[i,t] = 1} phi[t].
    """
    q = nde_l25_quotient
    # K = round(p * n) for each of 3 marginals
    p_true = np.array([0.6, 0.3, 0.7])
    n = 10_000
    obs = [(int(round(p * n)), n) for p in p_true]
    samples = binomial_marginal_gibbs(q, observations=obs, S=1000, burn=500, seed=0)
    # Posterior means of the three marginals
    M = q.constraint_matrix  # (3, reduced_dim)
    p_posterior = samples @ M.T   # (S, 3)
    p_mean = p_posterior.mean(axis=0)
    for i, expected in enumerate(p_true):
        assert abs(p_mean[i] - expected) < 0.02, f"marginal {i}: {p_mean[i]} vs {expected}"


def test_gibbs_rejects_mismatched_obs_length(nde_l25_quotient):
    """observations length must match number of constraints."""
    bad = [(60, 100), (30, 100)]  # only 2 but quotient has 3 constraints
    with pytest.raises(ValueError):
        binomial_marginal_gibbs(nde_l25_quotient, observations=bad, S=10, seed=0)
