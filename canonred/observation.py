"""Observation / information-set constraints.

See docs/01_concepts.md §7.

A MomentConstraint is one linear functional on psi: the indicator that a
specified conjunction of counterfactual events holds at a canonical state u.
For each row i of the LP's `M` matrix, `M[i, u] = MomentConstraint_i.evaluate(u, G)`.
"""

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from canonred.canonical import CanonicalState, eval_cfvar
from canonred.counterfactual import CFVar
from canonred.graph import ADMG
from canonred.types import Value


@dataclass(frozen=True)
class MomentConstraint:
    """One linear equality row `M[i, :] psi = p_i` on Omega_full.

    `M[i, u] = indicator( all (events_j := CFVar_j == target_j) hold at u )`.
    """

    name: str
    events: tuple[tuple[CFVar, Value], ...]

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("MomentConstraint must have at least one event")

    def evaluate(self, u: CanonicalState, G: ADMG) -> int:
        """Return 1 iff every event (cfvar == target) holds at u; else 0."""
        for cfvar, target in self.events:
            if eval_cfvar(u, cfvar, G) != target:
                return 0
        return 1


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------


def pobs_constraints(G: ADMG) -> tuple[MomentConstraint, ...]:
    """One MomentConstraint per observable cell of prod_V Omega_V (all nodes)."""
    ordered = G.nodes
    obs_cfvars = tuple(CFVar.observational(v, G) for v in ordered)
    domains = tuple(range(G.cardinality[v]) for v in ordered)
    out: list[MomentConstraint] = []
    for cell in itertools.product(*domains):
        name = "P_obs(" + ",".join(f"{v}={val}" for v, val in zip(ordered, cell, strict=True)) + ")"
        events = tuple((cf, val) for cf, val in zip(obs_cfvars, cell, strict=True))
        out.append(MomentConstraint(name=name, events=events))
    return tuple(out)


def l25_marginal_constraints(
    pairs: Iterable[tuple[CFVar, Value]],
) -> tuple[MomentConstraint, ...]:
    """One MomentConstraint per (CFVar, target) marginal.

    Typical use:

        v_w1   = CFVar.standard("W", world_x1, G)
        v_y00  = CFVar.standard("Y", world_x0_w0, G)
        v_y01  = CFVar.standard("Y", world_x0_w1, G)
        cs = l25_marginal_constraints([(v_w1, 1), (v_y00, 1), (v_y01, 1)])
    """
    out: list[MomentConstraint] = []
    for cf, target in pairs:
        name = f"P({cf.base}={target})"
        out.append(MomentConstraint(name=name, events=((cf, target),)))
    return tuple(out)


__all__ = [
    "MomentConstraint",
    "l25_marginal_constraints",
    "pobs_constraints",
]
