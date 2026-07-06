"""Named (graph, query, constraints) fixtures with paper-reported numbers.

See docs/03_modules.md §12.

For the initial scaffold only the graphs are populated. Queries, constraints, and
expected cardinalities will be filled in as the downstream modules land.
"""

from canonred.graph import ADMG, admg

# ----------------------------------------------------------------------
# NDE graph (paper Fig. 2a): X -> W -> Y, X -> Y, X <-> W.
# ----------------------------------------------------------------------


def nde_graph() -> ADMG:
    """Natural Direct Effect graph (binary nodes)."""
    return admg(
        nodes=("X", "W", "Y"),
        cardinality=2,
        directed=[("X", "W"), ("W", "Y"), ("X", "Y")],
        bidirected=[{"X", "W"}],
    )


# ----------------------------------------------------------------------
# IV (instrumental variable) graph (appendix Fig. 3a): Z -> X -> Y, X <-> Y.
# ----------------------------------------------------------------------


def iv_graph() -> ADMG:
    return admg(
        nodes=("Z", "X", "Y"),
        cardinality=2,
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[{"X", "Y"}],
    )


# ----------------------------------------------------------------------
# Non-IV graph (appendix Fig. 3b): Z -> X -> Y, X <-> Y, Z <-> Y.
# ----------------------------------------------------------------------


def non_iv_graph() -> ADMG:
    return admg(
        nodes=("Z", "X", "Y"),
        cardinality=2,
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[{"X", "Y"}, {"Z", "Y"}],
    )


# ----------------------------------------------------------------------
# Front-door graph (ZTB22 Fig. 1c): X -> W -> Y, X <-> Y.
# ----------------------------------------------------------------------


def frontdoor_graph() -> ADMG:
    return admg(
        nodes=("X", "W", "Y"),
        cardinality=2,
        directed=[("X", "W"), ("W", "Y")],
        bidirected=[{"X", "Y"}],
    )


# ----------------------------------------------------------------------
# Bow graph (ZTB22 Fig. 1d): X -> Y, X <-> Y.
# ----------------------------------------------------------------------


def bow_graph() -> ADMG:
    return admg(
        nodes=("X", "Y"),
        cardinality=2,
        directed=[("X", "Y")],
        bidirected=[{"X", "Y"}],
    )


# ----------------------------------------------------------------------
# PNS graph (paper appendix Fig. confounded-mediator-pns):
#   X -> M -> Y, X -> Y, X <-> Y, M <-> Y.
#
# The direct X -> Y edge is missing from the paper's TikZ as of the submission
# snapshot (author confirmed 2026-04-25: TikZ copy-paste miss from the NDE
# template). Including it here to match the paper's intended Y parent set
# {X, M} and thereby the paper's reported lifted dim = 128 for this graph.
# ----------------------------------------------------------------------


def pns_graph() -> ADMG:
    return admg(
        nodes=("X", "M", "Y"),
        cardinality=2,
        directed=[("X", "M"), ("M", "Y"), ("X", "Y")],
        bidirected=[{"X", "Y"}, {"M", "Y"}],
    )


# ----------------------------------------------------------------------
# Ternary-mediator NDE graph (paper appendix Fig. iv-ternary-mediator-nde):
#   Z -> X -> M -> Y, X -> Y, X <-> Y. M is ternary, rest are binary.
#
# As with pns_graph(), the paper's TikZ omits the `X -> Y` edge — same
# copy-paste miss from the NDE template. The associated table reports lifted
# L3 dim = 4608, which requires Y's parents to be {X, M}. With X->Y included
# here, lifted dim = 2^1 × 2^2 × 3^2 × 2^6 = 4608 matches the paper.
# ----------------------------------------------------------------------


def ternary_nde_graph() -> ADMG:
    return admg(
        nodes=("Z", "X", "M", "Y"),
        cardinality={"Z": 2, "X": 2, "M": 3, "Y": 2},
        directed=[("Z", "X"), ("X", "M"), ("M", "Y"), ("X", "Y")],
        bidirected=[{"X", "Y"}],
    )


__all__ = [
    "bow_graph",
    "frontdoor_graph",
    "iv_graph",
    "nde_graph",
    "non_iv_graph",
    "pns_graph",
    "ternary_nde_graph",
]
