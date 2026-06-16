"""Lock-in migration figure: wrong answers move into the calm corner.

Plots wrong KG answers in the (SD-UQ, SEU) plane under the adaptive policy
versus the strict entity-first ablation, for 2WikiMultiHopQA and RealMedQA.
Connecting segments trace a sample of questions that are wrong under both
policies, making the migration literal; top marginals show the SD-UQ mass
collapsing onto the floor.  Pure log analysis; no reruns.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "paper", "figures"))
import make_figures  # noqa: E402  (rcParams styling)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

from experiments.trust_analysis import rows_for  # noqa: E402

FLOOR = 1e-12
N_ARROWS = 12
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "figures")
C_AD, C_ST = "#4D4D4D", "#B2182B"


def wrong_rows(slug, policy):
    return {r["qid"]: r for r in rows_for(slug, policy)
            if not r["correct"] and r["sd"] is not None and r["seu"] is not None}


def xy(r):
    return np.log10(max(r["sd"], 0.0) + FLOOR), r["seu"]


def main():
    fig = plt.figure(figsize=(4.6, 3.4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3.6], hspace=0.06)
    panels = [("2wikimultihopqa", "2WikiMHQA")]
    rng = np.random.default_rng(0)
    axes_main = []
    for col, (slug, label) in enumerate(panels):
        ax_m = fig.add_subplot(gs[0, col])
        ax = fig.add_subplot(gs[1, col])
        axes_main.append(ax)
        ad = wrong_rows(slug, "kg")
        st = wrong_rows(slug, "strict")

        # connecting segments for a uniformly random sample of questions
        # wrong under both policies; drawn only on the 2Wiki panel, where
        # the migration is visually strongest
        if col == 0:
            shared = sorted(set(ad) & set(st))
            sample = list(rng.choice(shared, min(N_ARROWS, len(shared)),
                                     replace=False))
            for q in sample:
                x0, y0 = xy(ad[q]); x1, y1 = xy(st[q])
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", lw=0.4,
                                            color="#bbbbbb", alpha=0.45,
                                            shrinkA=2, shrinkB=2), zorder=1)

        for rows, color, marker, name in [
            (ad, C_AD, "o", "Adaptive KG (wrong)"),
            (st, C_ST, "^", "Strict KG (wrong)"),
        ]:
            pts = np.array([xy(r) for r in rows.values()])
            jy = pts[:, 1] + rng.normal(0, 0.012, len(pts))
            ax.scatter(pts[:, 0], jy, s=16, alpha=0.6, c=color, marker=marker,
                       linewidths=0, zorder=3)
            ax_m.hist(pts[:, 0], bins=np.linspace(-12.2, -1.0, 36),
                      color=color, alpha=0.55)
        ax.axvline(np.log10(FLOOR) + 0.02, color="#888888", lw=0.7, ls=":")
        ax.text(np.log10(FLOOR) + 0.25, 0.04, "output\nfloor",
                fontsize=6.2, color="#666666")
        ax_m.set_title(label, fontsize=9, pad=2)
        ax_m.set_xlim(-12.4, -1.0)
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        for s in ("top", "right", "left"):
            ax_m.spines[s].set_visible(False)
        ax.set_xlabel(r"$\log_{10}(\mathrm{SD\text{-}UQ} + 10^{-12})$",
                      fontsize=8)
        ax.set_xlim(-12.4, -1.0)
        ax.set_ylim(-0.04, 1.04)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_main[0].set_ylabel("SEU (evidence-state\nuncertainty)", fontsize=8)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=C_AD,
                   markersize=6, label="Adaptive KG (wrong answers)"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=C_ST,
                   markersize=6, label="Strict KG (wrong answers)"),
        plt.Line2D([0], [0], color="#999999", lw=0.8,
                   label=f"same question, both wrong (random sample of {N_ARROWS})"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=1, fontsize=6.8,
               framealpha=0.9, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.78, bottom=0.15)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"lockin_migration.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print("Saved", path)


if __name__ == "__main__":
    main()
