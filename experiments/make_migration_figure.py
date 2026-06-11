"""Lock-in migration figure: wrong answers move into the calm corner.

Plots wrong KG answers in the (SD-UQ, SEU) plane under the adaptive policy
versus the strict entity-first ablation, for RealMedQA and 2WikiMultiHopQA.
Pure log analysis; no reruns.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "paper", "figures"))
import make_figures  # noqa: E402  (rcParams styling)
import matplotlib.pyplot as plt  # noqa: E402

from experiments.trust_analysis import rows_for  # noqa: E402

FLOOR = 1e-12
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "figures")


def wrong_points(slug, policy):
    rows = [r for r in rows_for(slug, policy)
            if not r["correct"] and r["sd"] is not None and r["seu"] is not None]
    x = np.log10(np.array([max(r["sd"], 0.0) for r in rows]) + FLOOR)
    y = np.array([r["seu"] for r in rows])
    return x, y


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    panels = [("realmedqa", "RealMedQA"), ("2wikimultihopqa", "2WikiMHQA")]
    for ax, (slug, label) in zip(axes, panels):
        ax_jitter = np.random.default_rng(0)
        for policy, color, marker, name in [
            ("kg", "#2166AC", "o", "Adaptive KG (wrong)"),
            ("strict", "#B2182B", "^", "Strict KG (wrong)"),
        ]:
            x, y = wrong_points(slug, policy)
            jy = y + ax_jitter.normal(0, 0.012, len(y))
            ax.scatter(x, jy, s=16, alpha=0.65, c=color, marker=marker,
                       label=f"{name}, n={len(x)}", linewidths=0)
        ax.axvline(np.log10(FLOOR) + 0.02, color="#888888", lw=0.7, ls=":")
        ax.text(np.log10(FLOOR) + 0.25, 0.04, "output\nfloor",
                fontsize=6.2, color="#666666")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(r"$\log_{10}(\mathrm{SD\text{-}UQ} + 10^{-12})$", fontsize=8)
        ax.set_xlim(-12.4, -1.0)
        ax.set_ylim(-0.04, 1.04)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("SEU (evidence-support\nuncertainty)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    # one legend without the n counts (panel-specific) - rebuild generic labels
    fig.legend(handles, ["Adaptive KG (wrong answers)", "Strict KG (wrong answers)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2,
               fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.18, wspace=0.08)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"lockin_migration.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print("Saved", path)


if __name__ == "__main__":
    main()
