"""Publication CSV->PDF figures for the posterior sweep (UAI 2026).

Consumes the per-seed shards written by experiments/paper_posterior_sweep/run_stable.py:
  fig3_per_draw_regime{I,II}_n{N}_seed{S}.csv : seed, draw, Q_8, Q_128, Q_diff, L_closed, U_closed
  fig5_per_draw_bounds_seed{S}.csv            : regime, n, seed, draw, L, U
  fig7_agreement_per_n_seed{S}.csv            : regime, n, seed, median_abs_q_diff, p99_abs_q_diff, max_abs_q_diff

Each figure: 2-level uncertainty — inner band = pooled posterior (across all draws &
seeds); outer band = reproducibility (5-95% across per-seed medians). Colorblind-safe
(Okabe-Ito), serif fonts, vector PDF, no plt.show(), no global rc leak.

Run directly to regenerate every figure:
  python canonred/figviz.py --in-dir <results_stable> --out-dir <figures>
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

# ----- constants tied to the NDE L_{2.5} running example -----
P_STAR = (0.6, 0.3, 0.7)
TRUTH_L, TRUTH_U = 0.30, 0.90          # closed-form sharp bounds at p*
Q_STAR = 0.54                          # true query P(Y_{0,W1}=1) = .4*.3 + .6*.7
REGIME_LABEL = {"I": "Binomial summaries", "II": r"Multinomial $t$-data"}

# Okabe-Ito (Wong 2011), colorblind-safe
C = {
    "reduced": "#0072B2",   # blue  (8-state)
    "lifted":  "#D55E00",   # vermillion (128-state)
    "lower":   "#0072B2",
    "upper":   "#D55E00",
    "truth":   "#000000",
    "band":    "#56B4E9",
    "muted":   "#999999",
}


def _apply_style() -> None:
    import matplotlib as mpl
    from cycler import cycler
    OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                 "#56B4E9", "#E69F00", "#F0E442", "#000000"]
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",  # sans-serif math to match Helvetica
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 200, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        "grid.alpha": 0.7, "axes.axisbelow": True,
        "lines.linewidth": 1.0, "lines.markersize": 4, "axes.linewidth": 0.8,
        "legend.frameon": False, "errorbar.capsize": 2,
        "axes.prop_cycle": cycler(color=OKABE_ITO),
    })


class paper_style:
    def __enter__(self):
        import matplotlib as mpl
        self._old = mpl.rcParams.copy()
        _apply_style()
        return self

    def __exit__(self, *a):
        import matplotlib as mpl
        mpl.rcParams.update(self._old)


def _save(fig, out_path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[figviz] wrote {out}")


# ----------------------------------------------------------------------
# loaders (csv + numpy, no pandas dependency)
# ----------------------------------------------------------------------

def _read(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def _fig3(in_dir: Path, regime: str, n: int):
    """Return dict seed -> (Q_8 array, Q_128 array) for one (regime, n)."""
    by_seed: dict[str, dict[str, list]] = defaultdict(lambda: {"q8": [], "q128": []})
    for r in _read(str(in_dir / f"fig3_per_draw_regime{regime}_n{n}_seed*.csv")):
        by_seed[r["seed"]]["q8"].append(float(r["Q_8"]))
        by_seed[r["seed"]]["q128"].append(float(r["Q_128"]))
    return {s: (np.array(d["q8"]), np.array(d["q128"])) for s, d in by_seed.items()}


def _seed_band(per_seed_vals: dict, lo=5, hi=95):
    """Given seed -> 1D array, return (grand_median, inner_lo, inner_hi, outer_lo, outer_hi)
    where inner = percentiles of pooled draws, outer = 5-95% across per-seed medians."""
    pooled = np.concatenate(list(per_seed_vals.values()))
    seed_meds = np.array([np.median(v) for v in per_seed_vals.values()])
    return (
        float(np.median(pooled)),
        float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5)),
        float(np.percentile(seed_meds, lo)), float(np.percentile(seed_meds, hi)),
    )


# ----------------------------------------------------------------------
# FIG 3 — posterior histogram grid (8-state vs lifted 128-state)
# ----------------------------------------------------------------------

def plot_posterior_grid(in_dir, out_path, *, regimes=("I", "II"),
                        ns=(200, 500, 1000), bins=60) -> None:
    import matplotlib.pyplot as plt
    in_dir = Path(in_dir)
    with paper_style():
        # single-column footprint; tight panels; legend goes right of bottom-right panel
        fig, axes = plt.subplots(len(regimes), len(ns), figsize=(3.5, 2.3),
                                 sharex=True, sharey="row", squeeze=False)
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.92,
                            wspace=0.08, hspace=0.18)
        for i, reg in enumerate(regimes):
            for j, n in enumerate(ns):
                ax = axes[i][j]
                per = _fig3(in_dir, reg, n)
                q8 = np.concatenate([v[0] for v in per.values()])
                q128 = np.concatenate([v[1] for v in per.values()])
                ax.hist(q8, bins=bins, range=(0, 1), density=True, color=C["reduced"],
                        alpha=0.35, label="8-state")
                ax.hist(q128, bins=bins, range=(0, 1), density=True, histtype="step",
                        edgecolor=C["lifted"], linewidth=0.7, label="128-state")
                ax.axvline(np.median(q8), color=C["muted"], lw=0.6, ls="-")
                ax.axvline(Q_STAR, color=C["truth"], lw=0.6, ls="--")
                ax.set_xlim(0.1, 0.9)
                ax.set_xticks([0.2, 0.5, 0.8])
                ax.tick_params(labelsize=6.5)
                if i == 0:
                    ax.set_title(f"$n={n}$", fontsize=7.5)
                if j == 0:
                    ax.set_ylabel(REGIME_LABEL[reg].split()[0] + "\ndensity", fontsize=6.5)
                if i == len(regimes) - 1 and j == len(ns) // 2:
                    ax.set_xlabel(r"$Q=P(Y_{0,W_1}{=}1)$", fontsize=7)
        # legend INSIDE the bottom-LEFT subplot, upper-left corner (empty: peak is centered)
        h, lab = axes[-1][0].get_legend_handles_labels()
        h.append(plt.Line2D([0], [0], color=C["truth"], ls="--", lw=0.6))
        lab.append(r"truth $Q^\star$")
        axes[-1][0].legend(h, lab, loc="upper left", fontsize=5.3,
                           handlelength=1.0, handletextpad=0.4, borderaxespad=0.25,
                           labelspacing=0.25, borderpad=0.3,
                           frameon=True, framealpha=0.85, edgecolor="none")
        _save(fig, out_path)


# ----------------------------------------------------------------------
# FIG 5 — posterior bands of sharp bounds [L,U] vs n
# ----------------------------------------------------------------------

def plot_bounds_bands_vs_n(in_dir, out_path, *, regimes=("I", "II")) -> None:
    import matplotlib.pyplot as plt
    rows = _read(str(Path(in_dir) / "fig5_per_draw_bounds_seed*.csv"))
    with paper_style():
        fig, axes = plt.subplots(1, len(regimes), figsize=(3.5, 1.9), sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.22, top=0.86, wspace=0.08)
        for j, reg in enumerate(regimes):
            ax = axes[0][j]
            ns = sorted({int(r["n"]) for r in rows if r["regime"] == reg})
            for endpoint, color, lab in (("L", C["lower"], "lower bound"),
                                         ("U", C["upper"], "upper bound")):
                med, ilo, ihi, olo, ohi = [], [], [], [], []
                for n in ns:
                    per: dict[str, list] = defaultdict(list)
                    for r in rows:
                        if r["regime"] == reg and int(r["n"]) == n:
                            per[r["seed"]].append(float(r[endpoint]))
                    per = {s: np.array(v) for s, v in per.items()}
                    m, il, ih, ol, oh = _seed_band(per)
                    med.append(m); ilo.append(il); ihi.append(ih); olo.append(ol); ohi.append(oh)
                ax.fill_between(ns, ilo, ihi, color=color, alpha=0.15, lw=0)
                ax.fill_between(ns, olo, ohi, color=color, alpha=0.30, lw=0)
                ax.plot(ns, med, color=color, marker="o", label=lab)
            ax.axhline(TRUTH_L, color=C["truth"], ls="--", lw=0.6)
            ax.axhline(TRUTH_U, color=C["truth"], ls="--", lw=0.6)
            ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.set_title(REGIME_LABEL[reg])
            if j == 0:
                ax.set_ylabel("sharp bound")
        axes[0][0].legend(loc="center right", ncol=1)
        _save(fig, out_path)


# ----------------------------------------------------------------------
# FIG 6 — posterior band of Q vs n
# ----------------------------------------------------------------------

def plot_q_bands_vs_n(in_dir, out_path, *, regimes=("I", "II"),
                      ns=(200, 500, 1000, 5000)) -> None:
    import matplotlib.pyplot as plt
    in_dir = Path(in_dir)
    with paper_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for reg, color in zip(regimes, (C["reduced"], C["lifted"])):
            med, ilo, ihi = [], [], []
            xs = []
            for n in ns:
                per = _fig3(in_dir, reg, n)
                if not per:
                    continue
                per_q = {s: v[0] for s, v in per.items()}
                m, il, ih, _, _ = _seed_band(per_q)
                xs.append(n); med.append(m); ilo.append(il); ihi.append(ih)
            ax.fill_between(xs, ilo, ihi, color=color, alpha=0.20, lw=0)
            ax.plot(xs, med, color=color, marker="o", label=REGIME_LABEL[reg])
        ax.axhline(Q_STAR, color=C["truth"], ls="--", lw=0.6, label=r"truth $Q^\star$")
        ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.set_ylabel(r"$Q$")
        ax.legend(loc="best")
        _save(fig, out_path)


# ----------------------------------------------------------------------
# FIG 7 — 8-vs-128 agreement vs n (median + p99, log-y)
# ----------------------------------------------------------------------

def plot_agreement_vs_n(in_dir, out_path, *, regimes=("I", "II"), tol=1e-8) -> None:
    import matplotlib.pyplot as plt
    rows = _read(str(Path(in_dir) / "fig7_agreement_per_n_seed*.csv"))
    with paper_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for reg, color in zip(regimes, (C["reduced"], C["lifted"])):
            ns = sorted({int(r["n"]) for r in rows if r["regime"] == reg})
            for stat, ls, mk in (("median_abs_q_diff", "-", "o"), ("p99_abs_q_diff", ":", "s")):
                med, lo, hi = [], [], []
                for n in ns:
                    vals = np.array([max(float(r[stat]), 1e-18) for r in rows
                                     if r["regime"] == reg and int(r["n"]) == n])
                    med.append(np.median(vals)); lo.append(np.percentile(vals, 5)); hi.append(np.percentile(vals, 95))
                ax.fill_between(ns, lo, hi, color=color, alpha=0.12, lw=0)
                ax.plot(ns, med, color=color, ls=ls, marker=mk, ms=3,
                        label=f"{REGIME_LABEL[reg]} ({'median' if 'median' in stat else 'p99'})")
        ax.axhline(tol, color=C["truth"], ls="--", lw=0.6)
        ax.text(0.02, 0.16, r"theoretical $|\Delta Q|=0$ (Prop. 3.1)", transform=ax.transAxes,
                fontsize=6.5, color=C["muted"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$n$"); ax.set_ylabel(r"$|Q_8-Q_{128}|$")
        ax.legend(loc="best", fontsize=6.5)
        _save(fig, out_path)


def plot_pobs_discrepancy(csv_path, out_path, *, tol=1e-8) -> None:
    """1x2: (left) safe 16-state sigma-quotient agreement with 128 (~machine precision);
    (right) query-only 8-state discrepancy from 128 (nonzero -> query-only loosens)."""
    import matplotlib.pyplot as plt
    rows = _read(str(csv_path))
    safe = np.array([max(float(r["safe_lb_diff"]), float(r["safe_ub_diff"])) for r in rows])
    qonly = np.array([max(float(r["qonly_lb_diff"]), float(r["qonly_ub_diff"])) for r in rows])
    safe = np.clip(safe, 1e-18, None)
    with paper_style():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(3.5, 1.8))
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.26, top=0.86, wspace=0.42)
        # left: safe quotient, log-x
        lo = max(safe.min() / 3, 1e-12)
        axL.hist(safe, bins=np.logspace(np.log10(lo), np.log10(max(tol * 3, safe.max() * 3)), 30),
                 color=C["reduced"], alpha=0.7)
        axL.axvline(tol, color=C["truth"], ls="--", lw=0.6)
        axL.set_xscale("log")
        axL.set_title("16-state (safe)", fontsize=7.5)
        axL.set_xlabel(r"$\max|\Delta\,\mathrm{bound}|$", fontsize=7)
        axL.set_ylabel("count", fontsize=7)
        axL.text(0.95, 0.9, "solver\ntol", transform=axL.transAxes, fontsize=5.5,
                 ha="right", va="top", color=C["muted"])
        axL.tick_params(labelsize=6)
        # right: query-only, linear
        axR.hist(qonly, bins=30, color=C["lifted"], alpha=0.7)
        axR.axvline(np.median(qonly), color=C["truth"], ls="-", lw=0.6)
        axR.set_title("8-state (query-only)", fontsize=7.5)
        axR.set_xlabel(r"$\max|\Delta\,\mathrm{bound}|$", fontsize=7)
        axR.text(0.95, 0.9, f"median\n{np.median(qonly):.3f}", transform=axR.transAxes,
                 fontsize=5.5, ha="right", va="top", color=C["muted"])
        axR.tick_params(labelsize=6)
        _save(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="results_stable dir with fig{3,5,7} shards")
    ap.add_argument("--out-dir", required=True, help="figures/ dir to write PDFs into")
    args = ap.parse_args()
    ind, outd = Path(args.in_dir), Path(args.out_dir)
    plot_posterior_grid(ind, outd / "posteriorQ_hist_grid_2x3_compare.pdf")
    plot_bounds_bands_vs_n(ind, outd / "bounds_bands_vs_n_compare.pdf")
    plot_q_bands_vs_n(ind, outd / "q_bands_vs_n_compare.pdf")
    plot_agreement_vs_n(ind, outd / "bounds_8_vs_128_agreement.pdf")
    print("[figviz] done.")


if __name__ == "__main__":
    main()
