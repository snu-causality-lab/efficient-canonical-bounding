"""Strict release validation for committed experiment artifacts.

This script checks that the release contains the 20-seed stable timing shards
used for the paper tables, that the shards have the expected dimensions and
solver provenance, and that re-aggregating them reproduces the committed
``STABLE_SUMMARY.csv``. It exits non-zero on failure so it can be used as a
local pre-release gate or a CI check.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_SEEDS = set(range(42, 62))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_shards(pattern: str) -> tuple[list[Path], list[dict[str, str]]]:
    paths = sorted(HERE.glob(pattern))
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_load_csv(path))
    return paths, rows


def _ints(rows: list[dict[str, str]], key: str) -> set[int]:
    return {int(r[key]) for r in rows}


def _strings(rows: list[dict[str, str]], key: str) -> set[str]:
    return {r[key] for r in rows}


class Validator:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        suffix = f" ({detail})" if detail else ""
        print(f"[{status}] {label}{suffix}")
        if not condition:
            self.failures.append(f"{label}{suffix}")

    def require_columns(
        self, label: str, rows: list[dict[str, str]], columns: set[str]
    ) -> None:
        if not rows:
            self.check(f"{label}: columns present", False, "no rows")
            return
        missing = columns - set(rows[0])
        self.check(
            f"{label}: columns present",
            not missing,
            f"missing={sorted(missing)}" if missing else "",
        )

    def check_shards(
        self,
        label: str,
        pattern: str,
        *,
        rows_per_seed: int,
        columns: set[str],
    ) -> list[dict[str, str]]:
        paths, rows = _load_shards(pattern)
        self.check(f"{label}: 20 seed files", len(paths) == 20, f"files={len(paths)}")
        self.check(
            f"{label}: row count",
            len(rows) == 20 * rows_per_seed,
            f"rows={len(rows)}",
        )
        self.require_columns(label, rows, columns)
        if rows:
            seeds = _ints(rows, "seed")
            self.check(
                f"{label}: seeds 42..61",
                seeds == EXPECTED_SEEDS,
                f"got={sorted(seeds)}",
            )
        bad_files: list[str] = []
        for path in paths:
            n_rows = len(_load_csv(path))
            if n_rows != rows_per_seed:
                bad_files.append(f"{path.name}:{n_rows}")
        self.check(
            f"{label}: per-file row counts",
            not bad_files,
            f"bad={bad_files}" if bad_files else "",
        )
        return rows


def _check_solver_provenance(v: Validator, label: str, rows: list[dict[str, str]]) -> None:
    v.check(f"{label}: solver is ECOS", _strings(rows, "solver") == {"ECOS"})
    v.check(
        f"{label}: solvers_used is ECOS",
        _strings(rows, "solvers_used") == {"ECOS"},
    )
    v.check(f"{label}: n_repeats=25", _ints(rows, "n_repeats") == {25})
    hosts = _strings(rows, "host")
    v.check(f"{label}: single timing host", len(hosts) == 1, f"hosts={sorted(hosts)}")


def check_stable_shards(v: Validator) -> None:
    base_timing_cols = {
        "seed",
        "dim_reduced",
        "dim_lifted",
        "n_repeats",
        "solver",
        "solvers_used",
        "host",
    }
    t3 = v.check_shards(
        "Table 3 stable solver shards",
        "paper_posterior_sweep/results_stable/table3_solve_time_seed*.csv",
        rows_per_seed=1,
        columns=base_timing_cols | {"solver_med_8_ms", "solver_med_128_ms", "speedup"},
    )
    if t3:
        v.check("Table 3: reduced dim is 8", _ints(t3, "dim_reduced") == {8})
        v.check("Table 3: lifted dim is 128", _ints(t3, "dim_lifted") == {128})
        _check_solver_provenance(v, "Table 3", t3)

    t4 = v.check_shards(
        "Table 4 stable wall shards",
        "paper_posterior_sweep/results_stable/table4_wall_time_seed*.csv",
        rows_per_seed=1,
        columns=base_timing_cols | {"wall_med_8_ms", "wall_med_128_ms", "speedup_wall"},
    )
    if t4:
        v.check("Table 4: reduced dim is 8", _ints(t4, "dim_reduced") == {8})
        v.check("Table 4: lifted dim is 128", _ints(t4, "dim_lifted") == {128})
        _check_solver_provenance(v, "Table 4", t4)

    t5 = v.check_shards(
        "Table 5 stable stress shards",
        "paper_stress_scaling/results_stable/table5_stress_scaling_seed*.csv",
        rows_per_seed=3,
        columns=base_timing_cols
        | {"fiber", "solver_med_8_ms", "solver_med_lifted_ms", "speedup"},
    )
    if t5:
        expected = {(16, 128, 8), (64, 512, 8), (256, 2048, 8)}
        got = {
            (int(r["fiber"]), int(r["dim_lifted"]), int(r["dim_reduced"]))
            for r in t5
        }
        v.check("Table 5: fiber/dimension grid", got == expected, f"got={sorted(got)}")
        _check_solver_provenance(v, "Table 5", t5)

    fig7 = v.check_shards(
        "Figure 7 agreement shards",
        "paper_posterior_sweep/results_stable/fig7_agreement_per_n_seed*.csv",
        rows_per_seed=8,
        columns={"regime", "n", "seed", "max_abs_q_diff"},
    )
    if fig7:
        grid = {(r["regime"], int(r["n"])) for r in fig7}
        expected_grid = {
            (regime, n) for regime in {"I", "II"} for n in {200, 500, 1000, 5000}
        }
        v.check("Figure 7: regime/sample-size grid", grid == expected_grid)
        max_diff = max(float(r["max_abs_q_diff"]) for r in fig7)
        v.check("Figure 7: bounds agree within 1e-4", max_diff <= 1e-4, f"max={max_diff}")

    named = v.check_shards(
        "Named-fixture stable shards",
        "reb_named_fixtures/results_stable/named_fixtures_stable_seed*.csv",
        rows_per_seed=7,
        columns=(base_timing_cols - {"dim_reduced", "dim_lifted"})
        | {
            "fixture",
            "reduced_dim",
            "lifted_dim",
            "reduced_med_ms",
            "lifted_med_ms",
            "speedup",
            "max_abs_lb_diff",
            "max_abs_ub_diff",
        },
    )
    if named:
        expected_dims = {
            "bow": (8, 6),
            "iv": (32, 12),
            "non_iv": (32, 12),
            "frontdoor": (32, 12),
            "nde": (128, 16),
            "pns": (128, 12),
            "ternary_nde": (4608, 48),
        }
        fixtures = _strings(named, "fixture")
        v.check("Named fixtures: fixture set", fixtures == set(expected_dims))
        got_dims = {
            r["fixture"]: (int(r["lifted_dim"]), int(r["reduced_dim"]))
            for r in named
        }
        v.check(
            "Named fixtures: lifted/reduced dimensions",
            all(got_dims[k] == expected_dims[k] for k in expected_dims),
        )
        max_bound_diff = max(
            max(float(r["max_abs_lb_diff"]), float(r["max_abs_ub_diff"]))
            for r in named
        )
        v.check(
            "Named fixtures: reduced/lifted bounds agree within 1e-4",
            max_bound_diff <= 1e-4,
            f"max={max_bound_diff}",
        )
        _check_solver_provenance(v, "Named fixtures", named)


def check_rebuttal_artifacts(v: Validator) -> None:
    path = HERE / "reb_prop42_tightness" / "prop42_tightness.csv"
    rows = _load_csv(path) if path.exists() else []
    v.check("Prop 4.2 tightness CSV exists", bool(rows), str(path))
    for row in rows:
        case = row["case"]
        d_red = int(row["d_red_sigma_quotient"])
        d_u = int(row["d_U_prop42"])
        v.check(f"Prop 4.2 {case}: d_red <= d_U", d_red <= d_u)


def check_summary_reproducibility(v: Validator) -> None:
    committed = HERE / "paper_posterior_sweep" / "results_stable" / "STABLE_SUMMARY.csv"
    v.check("Committed STABLE_SUMMARY.csv exists", committed.exists(), str(committed))
    if not committed.exists():
        return

    with tempfile.TemporaryDirectory(prefix="canonred-validate-") as tmp:
        out = Path(tmp) / "STABLE_SUMMARY.csv"
        proc = subprocess.run(
            [
                sys.executable,
                str(HERE / "aggregate_stable.py"),
                "--expect-seeds",
                "20",
                "--out",
                str(out),
            ],
            cwd=HERE.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        v.check("aggregate_stable.py exits cleanly", proc.returncode == 0)
        v.check(
            "aggregate_stable.py reports no validation warnings",
            "[VALIDATION WARNINGS]" not in proc.stdout,
        )
        v.check("Recomputed summary was written", out.exists(), str(out))
        if not out.exists():
            return

        committed_rows = {
            (r["table"], r["quantity"]): r for r in _load_csv(committed)
        }
        recomputed_rows = {
            (r["table"], r["quantity"]): r for r in _load_csv(out)
        }
        v.check(
            "Summary row keys match",
            committed_rows.keys() == recomputed_rows.keys(),
            f"committed={len(committed_rows)} recomputed={len(recomputed_rows)}",
        )
        mismatches: list[str] = []
        for key in sorted(committed_rows):
            if key not in recomputed_rows:
                continue
            for col in ("median", "q1", "q3", "min", "max", "n_seeds"):
                lhs = committed_rows[key][col]
                rhs = recomputed_rows[key][col]
                if lhs != rhs:
                    mismatches.append(f"{key}:{col}:{lhs}!={rhs}")
        v.check(
            "Summary values match committed CSV",
            not mismatches,
            f"mismatches={mismatches[:5]}" if mismatches else "",
        )

        expected_medians = {
            ("table3_solver", "speedup"): 9.4235,
            ("table4_wall", "speedup_wall"): 1.2655,
            ("table5_stress", "dim128_speedup"): 10.104,
            ("table5_stress", "dim512_speedup"): 58.607,
            ("table5_stress", "dim2048_speedup"): 300.5625,
            ("named_fixtures", "pns_speedup"): 6.8855,
            ("named_fixtures", "ternary_nde_speedup"): 118.5485,
        }
        for key, expected in expected_medians.items():
            got = float(committed_rows[key]["median"])
            v.check(
                f"Reported median {key}",
                abs(got - expected) <= 5e-5,
                f"got={got} expected={expected}",
            )


def main() -> None:
    v = Validator()
    print("== Stable release shards ==")
    check_stable_shards(v)
    print("\n== Rebuttal artifacts ==")
    check_rebuttal_artifacts(v)
    print("\n== Aggregation reproducibility ==")
    check_summary_reproducibility(v)
    print()

    if v.failures:
        print(f"[validate] FAIL: {len(v.failures)} check(s) failed")
        raise SystemExit(1)
    print("[validate] OK: release artifacts reproduce the committed stable summary")


if __name__ == "__main__":
    main()
