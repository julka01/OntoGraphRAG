"""Figure: retrieval concentration vs answer dispersion, per question.

Uses the archived traces that retained per-question retrieval overlap
(2WikiMultiHopQA n=100, BioASQ n=57, MuSiQue n=66), faceted by dataset with
per-dataset Spearman correlations among wrong answers (pre-empting a
Simpson's-paradox reading of the pooled value), binned means with bootstrap
CIs, and legible correct points.  Pure log analysis; no reruns.
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
RNG = np.random.default_rng(42)
C_WRONG, C_OK = "#B2182B", "#888888"


def load(path, jsonl=False):
    if jsonl:
        return [json.loads(l) for l in open(path) if l.strip()]
    return json.load(open(path))["config_results"][0]["details"]


def points(rows, pre):
    ow, sw, oc, sc = [], [], [], []
    for r in rows:
        o = r.get(f"{pre}_retrieval_overlap")
        s = r.get(f"{pre}_sd_uq")
        c = r.get(f"{pre}_correct")
        if o is None or s is None or c is None or r.get(f"{pre}_generation_failed"):
            continue
        y = np.log10(max(float(s), 0.0) + FLOOR)
        if c:
            oc.append(float(o)); sc.append(y)
        else:
            ow.append(float(o)); sw.append(y)
    return map(np.array, (ow, sw, oc, sc))


def binned_means(x, y, n_bins=4, boot=500):
    if len(x) < 8:
        return [], [], []
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    mids, means, cis = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() < 4:
            continue
        vals = y[m]
        mids.append(float(x[m].mean()))
        means.append(float(vals.mean()))
        bs = [np.mean(RNG.choice(vals, len(vals), replace=True))
              for _ in range(boot)]
        cis.append((float(np.percentile(bs, 2.5)),
                    float(np.percentile(bs, 97.5))))
    return mids, means, cis


def single_panel(srcs):
    """Main-text figure: the 2Wiki KG panel that carries the mechanism."""
    name, rows = srcs[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ow, sw, oc, sc = points(rows, "kg")
    ax.scatter(oc, sc, s=10, c=C_OK, alpha=0.5, linewidths=0, zorder=1,
               label="correct")
    ax.scatter(ow, sw, s=16, c=C_WRONG, alpha=0.75, linewidths=0, zorder=2,
               label="wrong")
    mids, means, cis = binned_means(ow, sw)
    if mids:
        lo = [m - c[0] for m, c in zip(means, cis)]
        hi = [c[1] - m for m, c in zip(means, cis)]
        ax.errorbar(mids, means, yerr=[lo, hi], c=C_WRONG, lw=1.4,
                    capsize=2, marker="o", ms=3, zorder=3,
                    label="wrong: binned mean (95% CI)")
    from scipy.stats import spearmanr as _sp
    rho = _sp(ow, sw)
    ax.text(0.04, 0.05,
            rf"wrong answers: $\rho={rho.statistic:.2f}$ (p={rho.pvalue:.2f}, n={len(ow)})",
            transform=ax.transAxes, fontsize=7, color=C_WRONG)
    ax.set_xlabel("Per-question retrieval overlap", fontsize=8)
    ax.set_ylabel(r"$\log_{10}(\mathrm{SD\text{-}UQ}+10^{-12})$", fontsize=8)
    ax.set_title("Entity-first KG, 2WikiMHQA (archived trace)",
                 fontsize=8.4, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.42), fontsize=5.8,
              framealpha=0.92, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.9, bottom=0.17)
    for ext in ("pdf", "png"):
        out = os.path.join(OUT, f"concentration_vs_dispersion.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print("Saved", out)
    plt.close(fig)


def main():
    srcs = [
        ("2WikiMHQA", load(os.path.join(
            REPO, "results/mirage_2wikimultihopqa_results.json"))),
        ("BioASQ", load(os.path.join(
            REPO, "results/mirage_bioasq_results.json"))),
        ("MuSiQue", load(os.path.join(
            REPO, "results/checkpoints/musique_thr0.1_k10.jsonl"), jsonl=True)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.4),
                             sharey=True, sharex=True)
    for row, (side_label, pre) in enumerate(
            [("Entity-first KG", "kg"), ("Dense", "vanilla")]):
        for col, (name, rows) in enumerate(srcs):
            ax = axes[row, col]
            ow, sw, oc, sc = points(rows, pre)
            ax.scatter(oc, sc, s=9, c=C_OK, alpha=0.55, linewidths=0,
                       zorder=1)
            ax.scatter(ow, sw, s=15, c=C_WRONG, alpha=0.8, linewidths=0,
                       zorder=2)
            mids, means, cis = binned_means(ow, sw)
            if mids:
                lo = [m - c[0] for m, c in zip(means, cis)]
                hi = [c[1] - m for m, c in zip(means, cis)]
                ax.errorbar(mids, means, yerr=[lo, hi], c=C_WRONG, lw=1.4,
                            capsize=2, marker="o", ms=3, zorder=3)
            if len(ow) >= 8 and len(set(ow)) > 1:
                rho = spearmanr(ow, sw)
                ptxt = (f"p={rho.pvalue:.2f}" if rho.pvalue >= 0.01
                        else "p<0.01")
                ax.text(0.03, 0.05,
                        rf"$\rho={rho.statistic:.2f}$ ({ptxt}, n={len(ow)})",
                        transform=ax.transAxes, fontsize=6.6, color=C_WRONG)
            else:
                ax.text(0.03, 0.05, f"n={len(ow)} wrong",
                        transform=ax.transAxes, fontsize=6.6, color=C_WRONG)
            if row == 0:
                ax.set_title(name, fontsize=8.6)
            if col == 0:
                ax.set_ylabel(side_label + "\n"
                              r"$\log_{10}(\mathrm{SD\text{-}UQ}+10^{-12})$",
                              fontsize=7.2)
            if row == 1:
                ax.set_xlabel("Retrieval overlap", fontsize=7.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    h = [plt.Line2D([], [], color=C_WRONG, marker="o", ls="", ms=5,
                    label="wrong"),
         plt.Line2D([], [], color=C_OK, marker="o", ls="", ms=5,
                    label="correct"),
         plt.Line2D([], [], color=C_WRONG, lw=1.4, marker="o", ms=3,
                    label="wrong-answer binned mean (95% CI)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=3, fontsize=7.2, framealpha=0.9, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.88, bottom=0.11,
                        wspace=0.08, hspace=0.14)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"concentration_facets.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print("Saved", p)
    plt.close(fig)
    single_panel(srcs)


if __name__ == "__main__":
    main()
