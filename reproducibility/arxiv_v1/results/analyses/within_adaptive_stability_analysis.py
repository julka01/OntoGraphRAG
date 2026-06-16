#!/usr/bin/env python3
"""Build a small within-adaptive retrieval-stability sanity check.

This script reads archived question-level logs that retained per-question
retrieval-overlap scores. It does not rerun retrieval, generation, or metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
TRACE_LOG = ROOT / "results" / "mirage_2wikimultihopqa_results.json"
OUT_JSON = ROOT / "results" / "analyses" / "within_adaptive_stability_2wiki.json"
OUT_TEX = ROOT / "paper" / "figures" / "within_adaptive_stability.tex"

METRICS = [
    ("DSE", "kg_discrete_semantic_entropy"),
    ("SD-UQ", "kg_sd_uq"),
    ("VN", "kg_vn_entropy"),
    ("GPS", "kg_graph_path_support"),
]


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def auc(labels: list[int], scores: list[object]) -> float | None:
    pairs = [(float(score), label) for label, score in zip(labels, scores) if finite_number(score)]
    if not pairs:
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
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranked.append((pairs[k][1], avg_rank))
        i = j

    positive_rank_sum = sum(rank for label, rank in ranked if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def summarise_bucket(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    labels = [0 if row.get("kg_correct") else 1 for row in rows]
    summary: dict[str, object] = {
        "bucket": name,
        "n": len(rows),
        "overlap_min": min(float(row["kg_retrieval_overlap"]) for row in rows),
        "overlap_max": max(float(row["kg_retrieval_overlap"]) for row in rows),
        "accuracy": mean(1.0 if row.get("kg_correct") else 0.0 for row in rows),
    }
    for label, key in METRICS:
        values = [float(row[key]) for row in rows if finite_number(row.get(key))]
        summary[f"{label}_mean"] = mean(values) if values else None
        summary[f"{label}_auc"] = auc(labels, [row.get(key) for row in rows])
    return summary


def fmt(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    data = json.loads(TRACE_LOG.read_text())
    config = data["config_results"][0]
    rows = [
        row
        for row in config["details"]
        if not row.get("kg_generation_failed") and finite_number(row.get("kg_retrieval_overlap"))
    ]
    rows.sort(key=lambda row: float(row["kg_retrieval_overlap"]))
    third = len(rows) // 3
    buckets = [
        ("Low", rows[:third]),
        ("Medium", rows[third : 2 * third]),
        ("High", rows[2 * third :]),
    ]
    summaries = [summarise_bucket(name, bucket_rows) for name, bucket_rows in buckets]

    OUT_JSON.write_text(json.dumps({"source": str(TRACE_LOG), "summaries": summaries}, indent=2))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{%",
        r"  Within-adaptive retrieval-stability sanity check from an archived",
        r"  2WikiMultiHopQA $n=100$ trace that retained per-question chunk-overlap",
        r"  fields.  The newer $n=250$ 2Wiki run is used for the headline results,",
        r"  but it stores only run-level overlap.  This table is therefore a",
        r"  mechanism check, not a replacement for Table~\ref{tab:twowiki_hopwise_diagnostics}.",
        r"  AUROC treats incorrect answers as the positive class.%",
        r"}",
        r"\label{tab:within_adaptive_stability}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Overlap band & $n$ & Range & Acc. & DSE AUC & SD AUC & VN AUC & GPS AUC & DSE mean \\",
        r"\midrule",
    ]
    for summary in summaries:
        overlap_range = f"{fmt(summary['overlap_min'])}--{fmt(summary['overlap_max'])}"
        lines.append(
            " & ".join(
                [
                    str(summary["bucket"]),
                    str(summary["n"]),
                    overlap_range,
                    fmt(summary["accuracy"]),
                    fmt(summary["DSE_auc"]),
                    fmt(summary["SD-UQ_auc"]),
                    fmt(summary["VN_auc"]),
                    fmt(summary["GPS_auc"]),
                    fmt(summary["DSE_mean"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    OUT_TEX.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
