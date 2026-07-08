"""LP construction and solve for partial counterfactual bounds.

See docs/01_concepts.md §8. Two construction paths:

  - ``build_lp_lifted(G, Q, constraints)`` — enumerate Omega_full directly.
  - ``build_lp_reduced(signature_quotient)`` — consume a precomputed quotient.

Both return ``LPCoefficients(c, M, row_labels)`` with:

  min/max  c^T psi   s.t.   M psi = p,   sum(psi) = 1,   psi >= 0.

Solved with CVXPY (CLARABEL primary, SCS backup). Paper timing scripts pass
ECOS explicitly for reproducibility. The ``CachedSharpLP`` class wraps the
problem in ``cp.Parameter`` so repeated p-sweeps reuse compilation.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from canonred.canonical import (
    enumerate_states,
    eval_query_indicator,
    state_cardinality,
)
from canonred.counterfactual import CFQuery
from canonred.graph import ADMG
from canonred.observation import MomentConstraint
from canonred.signature import SignatureQuotient


class LPError(RuntimeError):
    """Raised for infeasibility or solver failure."""


@dataclass(frozen=True)
class LPCoefficients:
    """Linear coefficients for the sharp-bound LP.

    Attributes
    ----------
    c : np.ndarray, shape (dim,)
        Objective (query indicator).
    M : np.ndarray, shape (n_constraints, dim)
        Moment constraint matrix.
    row_labels : tuple[str, ...]
        Human-readable names of constraint rows.
    """

    c: np.ndarray
    M: np.ndarray
    row_labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dim(self) -> int:
        return int(self.c.shape[0])

    @property
    def n_constraints(self) -> int:
        return int(self.M.shape[0])


# ----------------------------------------------------------------------
# Build: lifted (explicit Omega_full enumeration)
# ----------------------------------------------------------------------


def build_lp_lifted(
    G: ADMG, Q: CFQuery, constraints: Iterable[MomentConstraint]
) -> LPCoefficients:
    """Enumerate Omega_full and build (c, M). Warning: exponential in graph size."""
    cs = tuple(constraints)
    dim = state_cardinality(G)
    c = np.zeros(dim, dtype=np.float64)
    M = np.zeros((len(cs), dim), dtype=np.float64)
    for i, u in enumerate(enumerate_states(G)):
        c[i] = eval_query_indicator(u, Q, G)
        for j, con in enumerate(cs):
            M[j, i] = con.evaluate(u, G)
    return LPCoefficients(c=c, M=M, row_labels=tuple(con.name for con in cs))


# ----------------------------------------------------------------------
# Build: reduced (signature quotient)
# ----------------------------------------------------------------------


def build_lp_reduced(quotient: SignatureQuotient) -> LPCoefficients:
    """Build reduced LP from a precomputed SignatureQuotient."""
    return LPCoefficients(
        c=quotient.objective.astype(np.float64),
        M=quotient.constraint_matrix.astype(np.float64),
        row_labels=tuple(),  # names aren't threaded through the quotient yet
    )


# ----------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------


def _solve_direction(coef: LPCoefficients, p: np.ndarray, maximize: bool, solver: str, backup: str):
    import cvxpy as cp

    x = cp.Variable(coef.dim, nonneg=True)
    cons = [cp.sum(x) == 1]
    if coef.n_constraints > 0:
        cons.append(coef.M @ x == p)
    obj = cp.Maximize(coef.c @ x) if maximize else cp.Minimize(coef.c @ x)
    prob = cp.Problem(obj, cons)

    start = time.perf_counter()
    try:
        val = prob.solve(solver=getattr(cp, solver), verbose=False)
    except Exception:
        val = prob.solve(solver=getattr(cp, backup), verbose=False)
    wall = time.perf_counter() - start

    status = prob.status or ""
    if "infeasible" in status.lower() or val is None:
        raise LPError(f"LP failed (status={prob.status})")
    stats = prob.solver_stats
    solver_time = float(getattr(stats, "solve_time", 0.0) or 0.0)
    return float(val), wall, solver_time


def solve_sharp_bounds(
    coef: LPCoefficients,
    p: np.ndarray,
    *,
    solver: str = "CLARABEL",
    backup: str = "SCS",
) -> tuple[float, float, dict]:
    """Return (lower, upper, diagnostics)."""
    if coef.n_constraints != len(p):
        raise LPError(
            f"LP has {coef.n_constraints} constraints but p has length {len(p)}"
        )
    lb, w_lb, st_lb = _solve_direction(coef, p, maximize=False, solver=solver, backup=backup)
    ub, w_ub, st_ub = _solve_direction(coef, p, maximize=True, solver=solver, backup=backup)
    diag = {
        "wall_time": w_lb + w_ub,
        "solver_time": st_lb + st_ub,
        "wall_time_min": w_lb,
        "wall_time_max": w_ub,
    }
    return lb, ub, diag


# ----------------------------------------------------------------------
# CachedSharpLP: DPP-friendly repeated solve over the same (c, M).
# ----------------------------------------------------------------------


class CachedSharpLP:
    """Repeatedly solve sharp bounds with varying p. Caches CVXPY compilation."""

    def __init__(self, coef: LPCoefficients, *, solver: str = "CLARABEL", backup: str = "SCS",
                 solver_opts: dict | None = None, strict: bool = False) -> None:
        import cvxpy as cp

        self._cp = cp
        self.coef = coef
        self.solver = solver
        self.backup = backup
        # Pinned solver tolerances for reproducible timing (e.g. ECOS
        # feastol/abstol/reltol/max_iters). Empty dict = solver defaults.
        self.solver_opts = dict(solver_opts or {})
        # strict=True: never silently fall back to ``backup`` (would corrupt
        # timing tables by mixing solvers). Raise instead and let the caller
        # record the failure.
        self.strict = strict
        self.last_solver_used = solver

        self.x = cp.Variable(coef.dim, nonneg=True)
        self.p_param = cp.Parameter(coef.n_constraints) if coef.n_constraints > 0 else None
        cons: list = [cp.sum(self.x) == 1]
        if self.p_param is not None:
            cons.append(coef.M @ self.x == self.p_param)
        self.prob_min = cp.Problem(cp.Minimize(coef.c @ self.x), cons)
        self.prob_max = cp.Problem(cp.Maximize(coef.c @ self.x), cons)

    def _solve(self, prob) -> tuple[float, float, float]:
        cp = self._cp
        start = time.perf_counter()
        try:
            val = prob.solve(solver=getattr(cp, self.solver), warm_start=True,
                             verbose=False, **self.solver_opts)
            self.last_solver_used = self.solver
        except Exception:
            if self.strict:
                raise LPError(f"{self.solver} failed and strict=True (no fallback)")
            val = prob.solve(solver=getattr(cp, self.backup), warm_start=True, verbose=False)
            self.last_solver_used = self.backup
        wall = time.perf_counter() - start
        status = prob.status or ""
        if "infeasible" in status.lower() or val is None:
            raise LPError(f"LP failed (status={prob.status})")
        stats = prob.solver_stats
        solver_time = float(getattr(stats, "solve_time", 0.0) or 0.0)
        return float(val), wall, solver_time

    def solve(self, p: np.ndarray) -> tuple[float, float, dict]:
        if self.p_param is not None:
            if len(p) != self.coef.n_constraints:
                raise LPError(
                    f"p has length {len(p)} but LP has {self.coef.n_constraints} constraints"
                )
            self.p_param.value = np.asarray(p, dtype=np.float64)
        lb, w_lb, st_lb = self._solve(self.prob_min)
        ub, w_ub, st_ub = self._solve(self.prob_max)
        diag = {
            "wall_time": w_lb + w_ub,
            "solver_time": st_lb + st_ub,
            "solver_used": self.last_solver_used,
        }
        return lb, ub, diag


__all__ = [
    "CachedSharpLP",
    "LPCoefficients",
    "LPError",
    "build_lp_lifted",
    "build_lp_reduced",
    "solve_sharp_bounds",
]
