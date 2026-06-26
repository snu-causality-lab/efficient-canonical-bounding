"""Aggregate per-seed stable-timing CSV shards into paper-ready numbers.

The run_stable.py drivers write one CSV per seed (sharded for slurm arrays):
  paper_posterior_sweep/results_stable/table3_solve_time_seed{S}.csv
  paper_posterior_sweep/results_stable/table4_wall_time_seed{S}.csv
  paper_stress_scaling/results_stable/table5_stress_scaling_seed{S}.csv

This module is the SINGLE source of the reported numbers: it reads those shards
and computes median / IQR (Q1,Q3) / min / max / n_seeds for every reported
quantity, writes `*_summary.csv`, and prints a paper-ready block. Run it after
retrieving the slurm results so Table 3/4/5 and the conclusion are reproducibly
derived from the CSVs rather than hand-computed.

Usage:
  python experiments/aggregate_stable.py            # uses default results dirs
  python experiments/aggregate_stable.py --results-name results_stable
"""

import argparse
import csv
import glob
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def _col(rows: list[dict], key: str) -> list:
    """Column values, skipping rows that lack the key (tolerates older shards;
    avoids KeyError if a shard predates a newly-added column)."""
    return [r[key] for r in rows if r.get(key) not in (None, "")]


def _stats(xs) -> dict:
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    if n == 0:
        return {"median": None, "q1": None, "q3": None, "min": None, "max": None, "n": 0}
    med = st.median(xs)
    if n >= 2:
        q = st.quantiles(xs, n=4, method="inclusive")  # [Q1, Q2, Q3]
        q1, q3 = q[0], q[2]
    else:
        q1 = q3 = med
    return {"median": med, "q1": q1, "q3": q3, "min": xs[0], "max": xs[-1], "n": n}


def _fmt(s: dict, prec: int = 3) -> str:
    if s["n"] == 0:
        return "n/a"
    return (f"{s['median']:.{prec}f}  IQR[{s['q1']:.{prec}f},{s['q3']:.{prec}f}]  "
            f"range[{s['min']:.{prec}f},{s['max']:.{prec}f}]  n={s['n']}")


def _summary_rows(label: str, cols: dict, prec: int = 4) -> list[dict]:
    out = []
    for name, s in cols.items():
        out.append({
            "table": label, "quantity": name,
            "median": round(s["median"], prec) if s["n"] else None,
            "q1": round(s["q1"], prec) if s["n"] else None,
            "q3": round(s["q3"], prec) if s["n"] else None,
            "min": round(s["min"], prec) if s["n"] else None,
            "max": round(s["max"], prec) if s["n"] else None,
            "n_seeds": s["n"],
        })
    return out


def _validate(label: str, rows: list[dict], *, expect_seeds: int | None) -> list[str]:
    """Guardrails so a partial/contaminated result set is never silently reported."""
    warns: list[str] = []
    if not rows:
        return warns
    seeds = sorted({r.get("seed") for r in rows if r.get("seed") is not None})
    if expect_seeds is not None and len(seeds) != expect_seeds:
        warns.append(f"{label}: {len(seeds)} seeds present, expected {expect_seeds}")
    solvers = {r.get("solver") for r in rows if r.get("solver")}
    if solvers - {"ECOS"}:
        warns.append(f"{label}: non-ECOS in 'solver' column: {solvers}")
    used: set = set()
    for r in rows:
        used.update(t for t in str(r.get("solvers_used", "")).split("+") if t)
    if used - {"ECOS"}:
        warns.append(f"{label}: solvers_used = {used} (SCS-fallback contamination?)")
    hosts = {r.get("host") for r in rows if r.get("host")}
    if len(hosts) > 1:
        warns.append(f"{label}: mixed hosts {hosts}; timings not comparable")
    return warns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-name", default="results_stable",
                    help="subdir name under each experiment dir holding the seed shards")
    ap.add_argument("--out", default=None, help="where to write *_summary.csv (default: each results dir)")
    ap.add_argument("--expect-seeds", type=int, default=None,
                    help="warn if a table does not have exactly this many seed shards")
    args = ap.parse_args()

    post = HERE / "paper_posterior_sweep" / args.results_name
    stress = HERE / "paper_stress_scaling" / args.results_name

    summary: list[dict] = []

    # ---- Table 3 (solver, 8 vs 128) ----
    t3 = _load(str(post / "table3_solve_time_seed*.csv"))
    if t3:
        cols = {
            "speedup": _stats(_col(t3, "speedup")),
            "solver_med_8_ms": _stats(_col(t3, "solver_med_8_ms")),
            "solver_med_128_ms": _stats(_col(t3, "solver_med_128_ms")),
        }
        summary += _summary_rows("table3_solver", cols)
        print("== Table 3 (solver, 8 vs 128) ==")
        print(f"  speedup        : {_fmt(cols['speedup'], 2)}")
        print(f"  8-state  (ms)  : {_fmt(cols['solver_med_8_ms'], 4)}")
        print(f"  128-state(ms)  : {_fmt(cols['solver_med_128_ms'], 4)}")

    # ---- Table 4 (wall) ----
    t4 = _load(str(post / "table4_wall_time_seed*.csv"))
    if t4:
        cols = {
            "speedup_wall": _stats(_col(t4, "speedup_wall")),
            "wall_med_8_ms": _stats(_col(t4, "wall_med_8_ms")),
            "wall_med_128_ms": _stats(_col(t4, "wall_med_128_ms")),
        }
        summary += _summary_rows("table4_wall", cols)
        print("== Table 4 (wall) ==")
        print(f"  speedup_wall   : {_fmt(cols['speedup_wall'], 2)}")
        print(f"  8-state  (ms)  : {_fmt(cols['wall_med_8_ms'], 3)}")
        print(f"  128-state(ms)  : {_fmt(cols['wall_med_128_ms'], 3)}")

    # ---- Table 5 (stress, per dim_lifted) ----
    t5 = _load(str(stress / "table5_stress_scaling_seed*.csv"))
    if t5:
        bydim: dict[int, list] = {}
        for r in t5:
            bydim.setdefault(int(r["dim_lifted"]), []).append(r)
        print("== Table 5 (stress scaling) ==")
        for dim in sorted(bydim):
            rs = bydim[dim]
            cols = {
                f"dim{dim}_speedup": _stats(_col(rs, "speedup")),
                f"dim{dim}_med_8_ms": _stats(_col(rs, "solver_med_8_ms")),
                f"dim{dim}_med_lifted_ms": _stats(_col(rs, "solver_med_lifted_ms")),
            }
            summary += _summary_rows("table5_stress", cols)
            print(f"  dim={dim:>5}: speedup {_fmt(cols[f'dim{dim}_speedup'], 2)}  "
                  f"8-state {_fmt(cols[f'dim{dim}_med_8_ms'], 4)}")

    # ---- named fixtures (per-graph, P_obs) ----
    named = _load(str(HERE / "reb_named_fixtures" / args.results_name /
                      "named_fixtures_stable_seed*.csv"))
    if named:
        byfix: dict[str, list] = {}
        for r in named:
            byfix.setdefault(r["fixture"], []).append(r)
        print("== Named fixtures (P_obs, reduced vs lifted) ==")
        # keep a stable display order
        order = ["bow", "iv", "non_iv", "frontdoor", "nde", "pns", "ternary_nde"]
        for fix in [f for f in order if f in byfix] + [f for f in byfix if f not in order]:
            rs = byfix[fix]
            sp = _stats(_col(rs, "speedup"))
            dims = (rs[0]["lifted_dim"], rs[0]["reduced_dim"])
            agree = max(float(r["max_abs_ub_diff"]) for r in rs)
            cols = {
                f"{fix}_speedup": sp,
                f"{fix}_reduced_med_ms": _stats(_col(rs, "reduced_med_ms")),
                f"{fix}_lifted_med_ms": _stats(_col(rs, "lifted_med_ms")),
            }
            summary += _summary_rows("named_fixtures", cols)
            print(f"  {fix:12s} lifted={dims[0]:>5} reduced={dims[1]:>3}: "
                  f"speedup {_fmt(sp, 2)}  max|bound diff|={agree:.1e}")

    # ---- validation guardrails ----
    warns = (_validate("table3", t3, expect_seeds=args.expect_seeds)
             + _validate("table4", t4, expect_seeds=args.expect_seeds)
             + _validate("table5", t5, expect_seeds=args.expect_seeds)
             + _validate("named_fixtures", named, expect_seeds=args.expect_seeds))
    if warns:
        print("\n[VALIDATION WARNINGS] do NOT report these numbers until resolved:")
        for w in warns:
            print("  - " + w)
    else:
        print("\n[validation] OK: solver=ECOS, no SCS contamination, single host per table.")

    # ---- write consolidated summary ----
    if summary:
        out_path = Path(args.out) if args.out else (post / "STABLE_SUMMARY.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"\n[written] {out_path}")
    else:
        print("[warn] no seed shards found — run the slurm array first.")


if __name__ == "__main__":
    main()
