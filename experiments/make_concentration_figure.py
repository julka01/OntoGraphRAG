"""Figure B: retrieval concentration vs output dispersion, per question.

Uses the archived traces that retained per-question retrieval overlap
(2WikiMultiHopQA n=100, BioASQ n=57, MuSiQue n=66) for both systems.
Pure log analysis; no reruns.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "paper", "figures"))
import make_figures  # noqa: E402  (rcParams styling)
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "figures")
FLOOR = 1e-12
MARKERS = {"2wiki": "o", "bioasq": "s", "musique": "^"}


def load(path, jsonl=False):
    if jsonl:
        return [json.loads(l) for l in open(path) if l.strip()]
    return json.load(open(path))["config_results"][0]["details"]


def main():
    srcs = [("2wiki", load(os.path.join(REPO, "results/mirage_2wikimultihopqa_results.json"))),
            ("bioasq", load(os.path.join(REPO, "results/mirage_bioasq_results.json"))),
            ("musique", load(os.path.join(REPO, "results/checkpoints/musique_thr0.1_k10.jsonl"), jsonl=True))]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (label, pre) in zip(axes, [("Entity-first KG retrieval", "kg"),
                                       ("Dense retrieval", "vanilla")]):
        ov_w, sd_w = [], []
        for name, rows in srcs:
            for r in rows:
                o = r.get(f"{pre}_retrieval_overlap")
                s = r.get(f"{pre}_sd_uq")
                c = r.get(f"{pre}_correct")
                if o is None or s is None or c is None or r.get(f"{pre}_generation_failed"):
                    continue
                y = np.log10(max(float(s), 0.0) + FLOOR)
                if c:
                    ax.scatter(o, y, s=10, c="#BBBBBB", marker=MARKERS[name],
                               alpha=0.45, linewidths=0, zorder=1)
                else:
                    ov_w.append(float(o)); sd_w.append(y)
                    ax.scatter(o, y, s=16, c="#B2182B", marker=MARKERS[name],
                               alpha=0.8, linewidths=0, zorder=2)
        ov_w, sd_w = np.array(ov_w), np.array(sd_w)
        rho = spearmanr(ov_w, sd_w)
        # binned median trend over wrong answers
        bins = np.quantile(ov_w, np.linspace(0, 1, 6))
        mids, meds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ov_w >= lo) & (ov_w <= hi)
            if m.sum() >= 5:
                mids.append(ov_w[m].mean()); meds.append(np.median(sd_w[m]))
        ax.plot(mids, meds, c="#B2182B", lw=1.6, zorder=3)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Per-question retrieval overlap", fontsize=8)
        ax.text(0.03, 0.06,
                rf"wrong answers: $\rho={rho.statistic:.2f}$"
                + (f" (p={rho.pvalue:.3f})" if rho.pvalue >= 0.001 else " (p<0.001)"),
                transform=ax.transAxes, fontsize=7.5, color="#B2182B")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$\log_{10}(\mathrm{SD\text{-}UQ}+10^{-12})$", fontsize=8)
    h = [plt.Line2D([], [], color="#B2182B", marker="o", ls="", ms=5, label="wrong"),
         plt.Line2D([], [], color="#BBBBBB", marker="o", ls="", ms=5, label="correct"),
         plt.Line2D([], [], color="#B2182B", lw=1.6, label="wrong-answer median trend")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3,
               fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.18, wspace=0.07)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"concentration_vs_dispersion.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print("Saved", p)


if __name__ == "__main__":
    main()
