"""Unit tests for canonred.random_admg."""

import numpy as np
import pytest

from canonred.graph import GraphError
from canonred.random_admg import random_admg


def test_random_admg_is_dag():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        g = random_admg(5, rng=rng)
        # Construction enforces DAG invariant; if cycles existed, ADMG raised.
        assert len(g.topo_order()) == 5


def test_random_admg_nodes_in_order():
    rng = np.random.default_rng(0)
    g = random_admg(4, rng=rng)
    assert g.nodes == ("V0", "V1", "V2", "V3")


def test_random_admg_deterministic_under_seed():
    a = random_admg(5, rng=np.random.default_rng(42))
    b = random_admg(5, rng=np.random.default_rng(42))
    assert a == b


def test_random_admg_differs_under_different_seed():
    a = random_admg(6, rng=np.random.default_rng(1))
    b = random_admg(6, rng=np.random.default_rng(2))
    # With 15 edge slots each, disagreement is near-certain.
    assert a != b


def test_random_admg_min_bidirected():
    """Even with p_bidir=0 we still get at least min_bidir edges."""
    rng = np.random.default_rng(0)
    g = random_admg(5, rng=rng, p_bidir=0.0, min_bidir=2)
    assert len(g.bidirected) >= 2


def test_random_admg_n_too_small_errors():
    with pytest.raises(ValueError):
        random_admg(1, rng=np.random.default_rng(0))


def test_random_admg_cardinality_respected():
    rng = np.random.default_rng(0)
    g = random_admg(3, rng=rng, cardinality=3)
    for v in g.nodes:
        assert g.cardinality[v] == 3
