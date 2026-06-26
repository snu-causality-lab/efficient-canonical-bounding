"""Paper Appendix F Table 5: stress scaling via synthetic fiber inflation.

Paper text: "We synthetically increase the fiber size of the lifted full-space
LP (thus increasing the full dimension) while keeping the reduced 8-state LP
fixed."

Interpretation: tile the reduced (c_tilde, M_tilde) into a larger LP of
dimension `fiber * reduced_dim`, with c and M repeated along the fiber axis
such that the aggregated optimum equals the reduced one. Then time both
solves.

For fiber ∈ {16, 64, 256} the inflated dim ∈ {128, 512, 2048}. Paper's
observed speedup: 12×, 44×, 218×.

Output CSV:
  table5_stress_scaling.csv
    columns: fiber, dim_lifted, med_8_ms, p90_8_ms, max_8_ms,
             med_lifted_ms, p90_lifted_ms, max_lifted_ms, speedup
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import dump_csv, dump_meta, timed  # noqa: E402

from canonred.counterfactual import CFQuery, CFVar  # noqa: E402
from canonred.examples import nde_graph  # noqa: E402
from canonred.lp import CachedSharpLP, LPCoefficients, build_lp_reduced  # noqa: E402
from canonred.observation import l25_marginal_constraints  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402
from canonred.worlds import World  # noqa: E402


def inflate(coef: LPCoefficients, fiber: int) -> LPCoefficients:
    """Tile (c, M) by repeating each column `fiber` times (no rescaling).

    Given the reduced LP ``min c^T phi  s.t. M phi = p, sum(phi)=1, phi>=0``,
    the inflated LP introduces `fiber` per-class duplicate variables:

        y in R^{dim * fiber}   with  y[t, f]   (t = class, f = fiber idx)

    and defines
        c_new[t, f] = c[t]                  (column repeat)
        M_new[i, t, f] = M[i, t]            (column repeat)

    Then ``M_new @ y = sum_{t} M[i,t] * (sum_f y[t,f])`` and ``sum(y) =
    sum_t (sum_f y[t,f])``, so setting `phi_t := sum_f y[t,f]` recovers the
    original LP. The optimum is identical, but the solver now has `fiber - 1`
    redundant degrees of freedom per class that it must traverse.
    """
    c_new = np.repeat(coef.c, fiber)
    M_new = np.repeat(coef.M, fiber, axis=1)
    return LPCoefficients(c=c_new, M=M_new, row_labels=coef.row_labels)


def build_nde_l25():
    G = nde_graph()
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    w00 = World.from_dict("x0w0", {"X": 0, "W": 0})
    w01 = World.from_dict("x0w1", {"X": 0, "W": 1})
    y = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w1, ("X", "Y"): w0, ("W", "Y"): w1},
        G=G,
    )
    Q = CFQuery(events=((y, 1),), graph=G)
    cs = l25_marginal_constraints([
        (CFVar.standard("W", w1, G), 1),
        (CFVar.standard("Y", w00, G), 1),
        (CFVar.standard("Y", w01, G), 1),
    ])
    return G, Q, cs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fibers", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--K", type=int, default=50, help="# solves per fiber for timing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_meta({"fibers": args.fibers, "K": args.K, "seed": args.seed}, out / "meta.json")

    G, Q, cs = build_nde_l25()
    q = build_signature_quotient(G, Q, cs)
    coef_r = build_lp_reduced(q)
    cached_r = CachedSharpLP(coef_r)
    cached_r.solve(np.array([0.5, 0.5, 0.5]))  # warmup

    rng = np.random.default_rng(args.seed)
    # Sample K feasible p values (matching the running example's feasible region)
    p_samples = []
    for _ in range(args.K):
        # Uniformly sample on simplex for phi, project to p
        alpha = rng.dirichlet(np.ones(q.reduced_dim))
        p_samples.append(np.asarray(q.constraint_matrix, dtype=np.float64) @ alpha)

    rows: list[dict] = []
    for fiber in args.fibers:
        with timed(f"inflate_fiber{fiber}"):
            coef_lifted = inflate(coef_r, fiber)
        cached_lifted = CachedSharpLP(coef_lifted)
        cached_lifted.solve(np.array([0.5, 0.5, 0.5]))  # warmup

        solver_r = np.zeros(args.K)
        solver_l = np.zeros(args.K)
        for k, p in enumerate(p_samples):
            _, _, dR = cached_r.solve(p)
            _, _, dL = cached_lifted.solve(p)
            solver_r[k] = dR["solver_time"]
            solver_l[k] = dL["solver_time"]
        row = {
            "fiber": fiber,
            "dim_lifted": coef_lifted.dim,
            "med_8_ms": round(float(np.median(solver_r)) * 1e3, 4),
            "p90_8_ms": round(float(np.quantile(solver_r, 0.9)) * 1e3, 4),
            "max_8_ms": round(float(np.max(solver_r)) * 1e3, 4),
            "med_lifted_ms": round(float(np.median(solver_l)) * 1e3, 4),
            "p90_lifted_ms": round(float(np.quantile(solver_l, 0.9)) * 1e3, 4),
            "max_lifted_ms": round(float(np.max(solver_l)) * 1e3, 4),
            "speedup": round(float(np.median(solver_l) / max(np.median(solver_r), 1e-12)), 3),
        }
        rows.append(row)
        print(
            f"fiber={fiber:3d} dim={coef_lifted.dim:5d}  "
            f"med_8={row['med_8_ms']:.3f}ms  med_lifted={row['med_lifted_ms']:.3f}ms  "
            f"speedup={row['speedup']:.1f}x", flush=True,
        )

    dump_csv(rows, out / "table5_stress_scaling.csv")
    print(f"\n[done] outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
