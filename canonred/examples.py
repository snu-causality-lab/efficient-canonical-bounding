"""Named graph fixtures used by examples, tests, and paper reproductions."""

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
# IV (instrumental variable) graph: Z -> X -> Y, X <-> Y.
# ----------------------------------------------------------------------


def iv_graph() -> ADMG:
    return admg(
        nodes=("Z", "X", "Y"),
        cardinality=2,
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[{"X", "Y"}],
    )


# ----------------------------------------------------------------------
# Non-IV graph: Z -> X -> Y, X <-> Y, Z <-> Y.
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
