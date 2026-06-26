"""Matplotlib helpers for posterior histograms and runtime comparisons.

All functions save to a file (no ``plt.show()``) and do not leak state between
calls. Matplotlib is imported lazily so the rest of the library works without
it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_posterior_histogram(
    Q_reduced: np.ndarray,
    Q_lifted: np.ndarray,
    *,
    out_path: str | Path,
    title: str = "",
    bins: int = 40,
    bounds_plugin: tuple[float, float] | None = None,
    bounds_truth: tuple[float, float] | None = None,
) -> None:
    """Overlay histogram of Q_reduced and Q_lifted (should coincide by Prop 3.1)."""
    import matplotlib.pyplot as plt

    fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=False)

    ax.hist(Q_reduced, bins=bins, density=True, alpha=0.35, label="Q (reduced)")
    ax.hist(Q_lifted, bins=bins, density=True, histtype="step", linewidth=2, label="Q (lifted)")

    q025, q50, q975 = np.quantile(Q_reduced, [0.025, 0.5, 0.975])
    ax.axvline(q025, linestyle="--", linewidth=1.5, label="Q 2.5%")
    ax.axvline(q975, linestyle="--", linewidth=1.5, label="Q 97.5%")
    ax.axvline(q50, linestyle="-", linewidth=2, label="Q median")
    if bounds_plugin is not None:
        ax.axvline(bounds_plugin[0], linestyle=":", linewidth=2, label="plug-in LB")
        ax.axvline(bounds_plugin[1], linestyle=":", linewidth=2, label="plug-in UB")
    if bounds_truth is not None:
        ax.axvline(bounds_truth[0], linestyle="-.", linewidth=2, label="truth LB")
        ax.axvline(bounds_truth[1], linestyle="-.", linewidth=2, label="truth UB")
    ax.set_title(title)
    ax.set_xlabel("Q")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=8)

    D = Q_lifted - Q_reduced
    ax_diff.hist(D, bins=30, density=True)
    ax_diff.set_title(f"lifted - reduced  (max |D| = {np.max(np.abs(D)):.2e})")
    ax_diff.set_xlabel("difference")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_runtime_bar(
    per_n: list[dict],
    *,
    out_path: str | Path,
    title: str = "LP solve time (median)",
) -> None:
    """Bar chart: 8-state vs lifted solve times per n."""
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in per_n]
    red = [r.get("solve_time_reduced_ms_median", r.get("solve_time_8_ms_median")) for r in per_n]
    lif = [r.get("solve_time_lifted_ms_median", r.get("solve_time_128_ms_median")) for r in per_n]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(ns))
    w = 0.35
    ax.bar(x - w / 2, red, w, label="reduced")
    ax.bar(x + w / 2, lif, w, label="lifted")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylabel("median solver time (ms)")
    ax.set_xlabel("n")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


__all__ = ["plot_posterior_histogram", "plot_runtime_bar"]
