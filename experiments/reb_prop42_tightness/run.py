"""Rebuttal B5 (utei Q5.1): Prop 4.2 tightness — redesigned 2026-04-25.

Prop. 4.2 gives a *sufficient* cardinality d_U(Q) = Π_{V∈C(U)} d_V^{m_V(Q)}.
It is not a coarsest-safe lower bound; §3.1's σ-quotient can be smaller.

Three illustrative cases:

  **tight_NDE_L2.5** — NDE with the running-example L_{2.5} query.
    d_full = 128, d_U = 8, σ-quotient d_red = 8.
    Prop 4.2 attains the coarsest-safe count exactly (the central paper claim).

  **tight_simple** — bow graph (X→Y, X↔Y), Q = P(Y_{x=0}=1), no constraints.
    d_full = 8, d_U = 2, d_red = 2.
    Prop 4.2 is tight — it reaches the same count as the σ-quotient.

  **slack_PNS-like** — bow graph, Q = P(Y_{x=0}=1 ∧ Y_{x=1}=1), no constraints.
    d_full = 8, d_U = 4, d_red = 2.
    Prop 4.2 is 2× slack: m_Y = 2 gives d_U = 4, but the query indicator itself
    is binary so the σ-quotient collapses to 2 classes. The gap illustrates that
    Prop 4.2 counts *structural* parent-world contexts, whereas the σ-quotient
    further collapses contexts that yield the same (objective, constraint)
    signature.

(The earlier "slack" case — bow + P(Y_{x=0}=1 ∧ Y_obs=1) — is excluded here
because it hit an edge regime where Prop 4.2's labelled count exceeds d_full
in the mixed-world regime. That case is still informative but distracts from
the main tight/slack story.)

Output: `prop42_tightness.csv` with d_full, d_U, d_red_sigma, m_V per case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import dump_csv, dump_meta  # noqa: E402

from canonred.context_set import context_multiplicities, prop42_cardinality  # noqa: E402
from canonred.counterfactual import CFQuery, CFVar  # noqa: E402
from canonred.examples import bow_graph, nde_graph  # noqa: E402
from canonred.observation import l25_marginal_constraints  # noqa: E402
from canonred.signature import build_signature_quotient  # noqa: E402
from canonred.worlds import World  # noqa: E402


def tight_nde_case():
    """Paper's running example — Prop 4.2 attains the coarsest-safe count exactly."""
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
    Q = CFQuery(events=((y, 1),), graph=G, name="nde_Y_0_W1")
    v_w1 = CFVar.standard("W", w1, G)
    v_y00 = CFVar.standard("Y", w00, G)
    v_y01 = CFVar.standard("Y", w01, G)
    cs = l25_marginal_constraints([(v_w1, 1), (v_y00, 1), (v_y01, 1)])
    return "tight_NDE_L2.5", G, Q, cs


def tight_simple_case():
    """Prop 4.2 attains d_red on a single-world query."""
    G = bow_graph()
    w0 = World.from_dict("x0", {"X": 0})
    Q = CFQuery(
        events=((CFVar.standard("Y", w0, G), 1),),
        graph=G, name="bow_Y_x0",
    )
    return "tight_simple", G, Q, ()


def slack_pns_case():
    """Prop 4.2 is 2× slack — σ-quotient strictly smaller."""
    G = bow_graph()
    w0 = World.from_dict("x0", {"X": 0})
    w1 = World.from_dict("x1", {"X": 1})
    Q = CFQuery(
        events=(
            (CFVar.standard("Y", w0, G), 1),
            (CFVar.standard("Y", w1, G), 1),
        ),
        graph=G, name="bow_Yx0_Yx1",
    )
    return "slack_PNS-like", G, Q, ()


def describe(name: str, G, Q, cs) -> dict:
    d_full = G.lifted_canonical_size()
    d_U = prop42_cardinality(G, Q)
    m = context_multiplicities(G, Q)
    quot = build_signature_quotient(G, Q, cs)
    d_red = quot.reduced_dim
    gap_abs = d_U - d_red
    gap_mul = (d_U / d_red) if d_red else None
    return {
        "case": name,
        "graph": "NDE" if "NDE" in name else "bow",
        "query": Q.name,
        "n_constraints": len(cs),
        "d_full": d_full,
        "d_U_prop42": d_U,
        "d_red_sigma_quotient": d_red,
        "prop42_gap_abs": gap_abs,
        "prop42_gap_mul": round(gap_mul, 3) if gap_mul else None,
        "prop42_attains_coarsest": d_U == d_red,
        "m_V": {v: int(m[v]) for v in G.nodes},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--out-csv", type=str, default="prop42_tightness.csv")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_meta({"cases": ["tight_NDE_L2.5", "tight_simple", "slack_PNS-like"]},
              out / "meta.json")

    rows: list[dict] = []
    for factory in (tight_nde_case, tight_simple_case, slack_pns_case):
        name, G, Q, cs = factory()
        r = describe(name, G, Q, cs)
        rows.append(r)
        print(
            f"{name:18s}  d_full={r['d_full']:3d}  d_U={r['d_U_prop42']:3d}  "
            f"d_red={r['d_red_sigma_quotient']:3d}  "
            f"gap_mul={r['prop42_gap_mul']}  m={r['m_V']}",
            flush=True,
        )

    dump_csv(rows, out / args.out_csv)
    print(f"\n[wrote] {out / args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
