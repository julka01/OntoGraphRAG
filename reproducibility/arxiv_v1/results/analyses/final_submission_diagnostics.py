#!/usr/bin/env python3
"""Generate final paper diagnostics from saved adaptive KG logs.

The script reads finished JSON artefacts only. It does not call models, rerun
retrieval, or modify experiment outputs.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gps_v3_depth_matched import scores_for as gps_scores_for  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
OUT_JSON = ROOT / "results" / "analyses" / "final_submission_diagnostics.json"

DATASET_SLUGS = {
    "PubMedQA": "pubmedqa",
    "RealMedQA": "realmedqa",
    "HotpotQA": "hotpotqa",
    "HotpotQA FullWiki": "hotpotqa_fullwiki",
    "2WikiMHQA": "2wikimultihopqa",
    "MuSiQue": "musique",
}

KG_CFG = "kg_entity_first_thr0.1_k10_rt0p0"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Palatino", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.6,
    }
)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def latest_result_path(slug: str) -> Path:
    search_roots = [
        ROOT / "results" / "latest_kg_design_final_metrics" / "runs",
        ROOT / "results" / "runs",
    ]
    candidates: list[Path] = []
    for search_root in search_roots:
        candidates.extend(Path(p) for p in glob.glob(str(search_root / "*" / f"mirage_{slug}_results.json")))
    for path in sorted(candidates, key=lambda p: p.parent.name, reverse=True):
        doc = json.loads(path.read_text())
        if any(cfg.get("config", {}).get("name") == KG_CFG and cfg.get("details") for cfg in doc.get("config_results", [])):
            return path
    raise FileNotFoundError(slug)


def load_adaptive_rows() -> dict[str, list[dict[str, object]]]:
    datasets: dict[str, list[dict[str, object]]] = {}
    for label, slug in DATASET_SLUGS.items():
        path = latest_result_path(slug)
        doc = json.loads(path.read_text())
        cfg = next(item for item in doc["config_results"] if item.get("config", {}).get("name") == KG_CFG)
        rows = [
            row
            for row in cfg["details"]
            if not row.get("kg_generation_failed") and not row.get("kg_system_skipped")
        ]
        datasets[label] = rows
    return datasets


def auc(labels: list[int], scores: list[object]) -> float | None:
    pairs = [(float(score), label) for label, score in zip(labels, scores) if finite(score)]
    if len(pairs) < 4:
        return None
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    ranked: list[tuple[int, float]] = []
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + 1 + j) / 2
        ranked.extend((pairs[k][1], rank) for k in range(i, j))
        i = j
    pos_rank_sum = sum(rank for label, rank in ranked if label == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def aurec(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(score, label) for score, label in zip(scores, labels) if finite(score)]
    if len(pairs) < 4 or len({label for _, label in pairs}) < 2:
        return None
    pairs.sort(key=lambda item: item[0])
    coverages = []
    risks = []
    errors = 0
    for idx, (_, label) in enumerate(pairs, start=1):
        errors += label
        coverages.append(idx / len(pairs))
        risks.append(errors / idx)
    return float(np.trapezoid(risks, coverages))


def percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    if len(values) == 1:
        return ranks
    for rank, idx in enumerate(order):
        ranks[idx] = rank / (len(values) - 1)
    return ranks


def final_gps_scores(slug: str) -> dict[str, float | None]:
    try:
        return gps_scores_for(slug, side="kg")
    except FileNotFoundError:
        return {}


def gps_risk(row: dict[str, object], gps_scores: dict[str, float | None]) -> float:
    value = gps_scores.get(str(row.get("question_id")))
    return 1.0 if value is None else float(value)


def composite_scores(rows: list[dict[str, object]], gps_scores: dict[str, float | None]) -> list[float]:
    sd = [float(row.get("kg_sd_uq", 0.0)) if finite(row.get("kg_sd_uq")) else 0.0 for row in rows]
    seu = [
        float(row.get("kg_support_entailment_uncertainty", 0.5))
        if finite(row.get("kg_support_entailment_uncertainty"))
        else 0.5
        for row in rows
    ]
    gps = [gps_risk(row, gps_scores) for row in rows]
    sd_r, seu_r, gps_r = percentile_ranks(sd), percentile_ranks(seu), percentile_ranks(gps)
    return [(a + b + c) / 3.0 for a, b, c in zip(sd_r, seu_r, gps_r)]


def make_family_disagreement(datasets: dict[str, list[dict[str, object]]]) -> None:
    selected = [("RealMedQA", datasets["RealMedQA"]), ("2WikiMHQA", datasets["2WikiMHQA"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (name, rows) in zip(axes, selected):
        xs = np.array([
            math.log10(max(float(row.get("kg_sd_uq", 0.0)), 1e-12))
            if finite(row.get("kg_sd_uq"))
            else -12.0
            for row in rows
        ])
        ys = np.array([
            float(row.get("kg_support_entailment_uncertainty", 0.5))
            if finite(row.get("kg_support_entailment_uncertainty"))
            else 0.5
            for row in rows
        ])
        correct = np.array([bool(row.get("kg_correct")) for row in rows])
        x_med = float(np.median(xs))
        y_med = float(np.median(ys))
        ax.scatter(xs[correct], ys[correct], s=18, c="#1A9641", alpha=0.70, edgecolor="white", linewidth=0.25, label="Correct")
        ax.scatter(xs[~correct], ys[~correct], s=18, c="#D7191C", alpha=0.72, edgecolor="white", linewidth=0.25, label="Wrong")
        ax.axvline(x_med, color="#555555", linestyle="--", linewidth=0.7)
        ax.axhline(y_med, color="#555555", linestyle="--", linewidth=0.7)
        ax.set_title(f"{name} adaptive KG")
        ax.set_xlabel(r"$\log_{10}(\mathrm{SD\text{-}UQ}+10^{-12})$")
        ax.set_ylim(-0.03, 1.03)
        ax.text(0.02, 0.96, "calm + unsupported", transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.text(0.98, 0.05, "variable + supported", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.text(0.02, 0.05, "calm + supported", transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
        ax.text(0.98, 0.96, "variable + unsupported", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    axes[0].set_ylabel("SEU (evidence-support uncertainty)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"family_disagreement_scatter.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_stability_auc_plot() -> None:
    path = ROOT / "results" / "analyses" / "within_adaptive_stability_2wiki.json"
    data = json.loads(path.read_text())["summaries"]
    x = np.arange(len(data))
    labels = [row["bucket"] for row in data]
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.plot(x, [row["DSE_auc"] for row in data], marker="o", color="#2166AC", label="DSE")
    ax.plot(x, [row["SD-UQ_auc"] for row in data], marker="s", color="#D6604D", label="SD-UQ")
    ax.plot(x, [row["VN_auc"] for row in data], marker="^", color="#8856A7", label="VN-Ent.")
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 0.86)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Chunk-overlap band")
    ax.set_title("Adaptive 2Wiki stability split")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"stability_auc_by_overlap.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_composite_table(datasets: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, float | int | None]]:
    summaries: dict[str, dict[str, float | int | None]] = {}
    rows_tex = []
    for name, rows in datasets.items():
        slug = DATASET_SLUGS[name]
        gps_scores = final_gps_scores(slug)
        labels = [0 if row.get("kg_correct") else 1 for row in rows]
        composite = composite_scores(rows, gps_scores)
        sd = [row.get("kg_sd_uq") for row in rows]
        seu = [row.get("kg_support_entailment_uncertainty") for row in rows]
        gps = [gps_risk(row, gps_scores) for row in rows]
        sd_auc = auc(labels, sd)
        seu_auc = auc(labels, seu)
        gps_auc = auc(labels, gps)
        combined_auc = auc(labels, composite)
        base_err = sum(labels) / len(labels) if labels else None
        best_single = max(x for x in (sd_auc, seu_auc, gps_auc) if x is not None)
        summary = {
            "n": len(rows),
            "base_error": base_err,
            "sd_auc": sd_auc,
            "seu_auc": seu_auc,
            "gps_risk_auc": gps_auc,
            "best_single_auc": best_single,
            "combined_auc": combined_auc,
            "combined_aurec": aurec(labels, composite),
        }
        summaries[name] = summary
        best_single_s = fmt(best_single)
        combined_s = fmt(combined_auc)
        if combined_auc is not None and combined_auc > best_single:
            combined_s = r"\textbf{" + combined_s + "}"
        else:
            best_single_s = r"\textbf{" + best_single_s + "}"
        rows_tex.append(
            f"{name} & {summary['n']} & {fmt(base_err)} & {fmt(sd_auc)} & {fmt(seu_auc)} & "
            f"{fmt(gps_auc)} & {best_single_s} & {combined_s} & {fmt(summary['combined_aurec'])} \\\\"
        )

    tex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Adaptive-KG composite audit score.  The combined score is the mean of within-dataset percentile ranks for SD-UQ, SEU, and final GPS-risk, with GPS abstention treated as high risk.  AUROC treats incorrect answers as the positive class.  ``Base err.''\ is the no-abstention error rate; lower AUREC is better within a dataset.  ``Best single'' is the largest per-family AUROC in the row.}",
            r"\label{tab:composite_audit_score}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}lrrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & $n$ & Base err. & SD-UQ AUC & SEU AUC & GPS-risk AUC & Best single & Combined AUC & Combined AUREC \\",
            r"\midrule",
            *rows_tex,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (FIG_DIR / "composite_audit_score.tex").write_text(tex)
    return summaries


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * value:.1f}\\%"


def reliability_cell(items: list[int]) -> str:
    if not items:
        return "--"
    wrong = sum(items)
    return f"{pct(wrong / len(items))} ({wrong}/{len(items)})"


def write_gps_reliability_table(datasets: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    rows_tex = []
    bins = [
        ("low", 0.0, 1.0 / 3.0),
        ("mid", 1.0 / 3.0, 2.0 / 3.0),
        ("high", 2.0 / 3.0, 1.0 + 1e-12),
    ]
    for name, rows in datasets.items():
        slug = DATASET_SLUGS[name]
        gps_scores = final_gps_scores(slug)
        answered = len(rows)
        usable = []
        for row in rows:
            score = gps_scores.get(str(row.get("question_id")))
            if score is not None:
                usable.append((float(score), 0 if row.get("kg_correct") else 1))
        abstained = answered - len(usable)
        binned: dict[str, list[int]] = {label: [] for label, _, _ in bins}
        for score, wrong in usable:
            for label, lower, upper in bins:
                if lower <= score < upper:
                    binned[label].append(wrong)
                    break
        summaries[name] = {
            "answered": answered,
            "usable": len(usable),
            "abstained": abstained,
            "abstention_rate": abstained / answered if answered else None,
            "bins": {
                label: {
                    "n": len(values),
                    "wrong": sum(values),
                    "error_rate": (sum(values) / len(values)) if values else None,
                }
                for label, values in binned.items()
            },
        }
        rows_tex.append(
            f"{name} & {answered} & {abstained} ({pct(abstained / answered if answered else None)}) & "
            f"{len(usable)} & {reliability_cell(binned['low'])} & "
            f"{reliability_cell(binned['mid'])} & {reliability_cell(binned['high'])} \\\\"
        )
    tex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{GPS abstention and fixed-bin empirical error rates on adaptive KG runs.  The three right columns report wrong/total rates within GPS-risk bins; lower GPS means stronger local graph support.  This is a descriptive reliability check, not an ECE calculation, because GPS is a graph-support score rather than a calibrated probability.}",
            r"\label{tab:gps_reliability_bins}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Answered & Abstained & Usable & GPS $<.33$ & $.33$--$.67$ & GPS $>.67$ \\",
            r"\midrule",
            *rows_tex,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (FIG_DIR / "gps_reliability_bins.tex").write_text(tex)
    return summaries


def fmt(value: object) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def main() -> None:
    datasets = load_adaptive_rows()
    make_family_disagreement(datasets)
    make_stability_auc_plot()
    composite = write_composite_table(datasets)
    gps_reliability = write_gps_reliability_table(datasets)
    OUT_JSON.write_text(
        json.dumps(
            {
                "datasets": {name: len(rows) for name, rows in datasets.items()},
                "composite": composite,
                "gps_reliability": gps_reliability,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
