"""Rebuttal B4: named fixtures (IV / Non-IV / front-door / bow / PNS / ternary NDE).

Reviewer U3kb asked for examples beyond the NDE graph. This script runs the
constraint-aware reduction + LP on every named graph in `canonred.examples`
with an appropriate query and full `P_obs` constraints.

Fixtures and queries:
  - IV       : Y_{X=1} single-world
  - Non-IV   : Y_{X=1} single-world
  - Frontdoor: Y_{X=1} single-world
  - Bow      : Y_{X=1} single-world (tiny; baseline sanity)
  - NDE      : Y_{0, W_1} the paper's running example (L_{2.5})
  - PNS      : Y_x ∧ Y_{x'} cross-world
  - Ternary NDE: Y_{1, M_0} cross-world with ternary M (lifted=4608)

For each: report lifted_dim, reduced_dim, Prop 4.2 d_U, build time, LP solve
time (reduced vs lifted), bounds agreement.

Output: `named_fixtures.csv`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import dump_csv, dump_meta  # noqa: E402

from canonred.context_set import prop42_cardinality  # noqa: E402
from canonred.counterfactual import CFQuery, CFVar  # noqa: E402
from canonred.examples import (  # noqa: E402
    bow_graph,
    frontdoor_graph,
    iv_graph,
    nde_graph,
    non_iv_graph,
    pns_graph,
    ternary_nde_graph,
)
from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced  # noqa: E402
from canonred.observation import pobs_constraints  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402
from canonred.worlds import World  # noqa: E402


def _bow():
    g = bow_graph()
    w = World.from_dict("x1", {"X": 1})
    Q = CFQuery(events=((CFVar.standard("Y", w, g), 1),), graph=g, name="bow_Y_x1")
    return "bow", g, Q


def _iv():
    g = iv_graph()
    w = World.from_dict("x1", {"X": 1})
    Q = CFQuery(events=((CFVar.standard("Y", w, g), 1),), graph=g, name="iv_Y_x1")
    return "iv", g, Q


def _non_iv():
    g = non_iv_graph()
    w = World.from_dict("x1", {"X": 1})
    Q = CFQuery(events=((CFVar.standard("Y", w, g), 1),), graph=g, name="non_iv_Y_x1")
    return "non_iv", g, Q


def _fd():
    g = frontdoor_graph()
    w = World.from_dict("x1", {"X": 1})
    Q = CFQuery(events=((CFVar.standard("Y", w, g), 1),), graph=g, name="fd_Y_x1")
    return "frontdoor", g, Q


def _nde():
    g = nde_graph()
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    y = CFVar.nested(
        base="Y",
        edge_worlds={("X", "W"): w1, ("X", "Y"): w0, ("W", "Y"): w1},
        G=g,
    )
    Q = CFQuery(events=((y, 1),), graph=g, name="nde_Y_0_W1")
    return "nde", g, Q


def _pns():
    g = pns_graph()
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    Q = CFQuery(
        events=(
            (CFVar.standard("Y", w0, g), 1),
            (CFVar.standard("Y", w1, g), 0),
        ),
        graph=g, name="pns_Yx_Yxp",
    )
    return "pns", g, Q


def _ternary_nde():
    g = ternary_nde_graph()
    w1 = World.from_dict("x1", {"X": 1})
    w0 = World.from_dict("x0", {"X": 0})
    y = CFVar.nested(
        base="Y",
        edge_worlds={
            ("Z", "X"): w0, ("X", "M"): w0, ("X", "Y"): w1, ("M", "Y"): w0,
        },
        G=g,
    )
    Q = CFQuery(events=((y, 1),), graph=g, name="ternary_Y_1_M0")
    return "ternary_nde", g, Q


FIXTURES = [_bow, _iv, _non_iv, _fd, _nde, _pns, _ternary_nde]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--out-csv", type=str, default="named_fixtures.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_meta({"p_samples": args.p_samples, "seed": args.seed}, out / "meta.json")

    rows: list[dict] = []
    rng = np.random.default_rng(args.seed)

    for factory in FIXTURES:
        name, G, Q = factory()
        cs = pobs_constraints(G)
        lifted_dim = G.lifted_canonical_size()

        row: dict = {"fixture": name, "n_nodes": len(G.nodes), "lifted_dim": lifted_dim}

        t0 = time.perf_counter()
        coef_l = build_lp_lifted(G, Q, cs)
        t_build_l = time.perf_counter() - t0

        t0 = time.perf_counter()
        q = build_signature_quotient(G, Q, cs)
        coef_r = build_lp_reduced(q)
        t_build_r = time.perf_counter() - t0

        try:
            d_u = prop42_cardinality(G, Q)
        except Exception:  # noqa: BLE001
            d_u = None

        row.update({
            "reduced_dim": q.reduced_dim,
            "prop42_d_U": d_u,
            "n_constraints": coef_l.n_constraints,
            "build_lifted_s": round(t_build_l, 4),
            "build_reduced_s": round(t_build_r, 4),
        })

        cached_l = CachedSharpLP(coef_l)
        cached_r = CachedSharpLP(coef_r)
        warmup_p = np.full(coef_l.n_constraints, 1.0 / coef_l.n_constraints)
        try:
            cached_l.solve(warmup_p)
            cached_r.solve(warmup_p)
        except Exception:
            pass

        rt_l: list[float] = []
        rt_r: list[float] = []
        diffs_lb: list[float] = []
        diffs_ub: list[float] = []
        for _ in range(args.p_samples):
            alpha = rng.dirichlet(np.ones(coef_l.dim))
            p = coef_l.M @ alpha
            p = np.clip(p, 0.0, 1.0)
            try:
                lb_l, ub_l, dL = cached_l.solve(p)
                lb_r, ub_r, dR = cached_r.solve(p)
            except Exception as e:  # noqa: BLE001
                row["solve_err"] = str(e)
                break
            rt_l.append(dL["solver_time"])
            rt_r.append(dR["solver_time"])
            diffs_lb.append(abs(lb_l - lb_r))
            diffs_ub.append(abs(ub_l - ub_r))

        if rt_l:
            row.update({
                "max_abs_lb_diff": round(max(diffs_lb), 10),
                "max_abs_ub_diff": round(max(diffs_ub), 10),
                "med_lifted_ms": round(float(np.median(rt_l)) * 1e3, 4),
                "med_reduced_ms": round(float(np.median(rt_r)) * 1e3, 4),
                "speedup": round(float(np.median(rt_l) / max(np.median(rt_r), 1e-12)), 3),
            })

        rows.append(row)
        print(
            f"{name:12s}  lifted={lifted_dim:5d}  reduced={q.reduced_dim:5d}  "
            f"d_U={d_u}  speedup={row.get('speedup')}  "
            f"LB_diff={row.get('max_abs_lb_diff')}  UB_diff={row.get('max_abs_ub_diff')}",
            flush=True,
        )

    dump_csv(rows, out / args.out_csv)
    print(f"[wrote] {out / args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
