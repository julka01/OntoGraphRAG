"""Current hop-wise analysis for the paper-facing 2Wiki runs.

This script reads the saved per-question logs from the latest 2WikiMultiHopQA
250-question runs and writes:

  - results/analyses/current_hopwise_2wiki.json
  - paper/figures/twowiki_hopwise_diagnostics.tex

It is deliberately separate from the experiment runner. It does not rerun any
model calls or mutate experimental artefacts.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

FINAL_PAIR = ROOT / (
    "results/latest_kg_design_final_metrics/runs/"
    "20260528-231830-2wikimultihopqa-n250-full-metrics-evaluation-subset-rebuildkg/"
    "mirage_2wikimultihopqa_results.json"
)
STRICT = ROOT / (
    "results/latest_kg_design_final_metrics/runs/"
    "20260529-120453-2wikimultihopqa-n250-full-metrics-evaluation-subset/"
    "mirage_2wikimultihopqa_results.json"
)

OUT_JSON = ROOT / "results/analyses/current_hopwise_2wiki.json"
OUT_TEX = ROOT / "paper/figures/twowiki_hopwise_diagnostics.tex"


POLICIES = [
    ("Dense vanilla", FINAL_PAIR, "dense_floor_thr0.1_k10_rt0p0", "vanilla"),
    ("Adaptive KG", FINAL_PAIR, "kg_entity_first_thr0.1_k10_rt0p0", "kg"),
    ("Strict KG", STRICT, "kg_strict_entity_first_thr0.1_k10_rt0p0", "kg"),
]

METRICS = [
    ("semantic_entropy", "SE"),
    ("sd_uq", "SD-UQ"),
    ("vn_entropy", "VN-Ent."),
    ("support_entailment_uncertainty", "SEU"),
    ("graph_path_support", "GPS"),
]


def load_config(path: Path, config_name: str) -> list[dict[str, Any]]:
    with path.open() as f:
        doc = json.load(f)
    for cfg in doc["config_results"]:
        if cfg["config"]["name"] == config_name:
            return cfg["details"]
    raise KeyError(f"{config_name} not found in {path}")


def hop_label(row: dict[str, Any]) -> str | None:
    hop = row.get("hop_count")
    try:
        hop_int = int(hop)
    except (TypeError, ValueError):
        return None
    if hop_int <= 0:
        return None
    return f"{hop_int}-hop" if hop_int <= 4 else "5+-hop"


def metric_rows(rows: list[dict[str, Any]], prefix: str, metric: str) -> list[dict[str, Any]]:
    if metric != "graph_path_support":
        return rows
    null_key = f"{prefix}_graph_path_support_null_reason"
    metric_key = f"{prefix}_{metric}"
    if any(null_key in row for row in rows):
        return [row for row in rows if not str(row.get(null_key, ""))]
    # Older logs used the documented 0.5 sentinel without persisting a null
    # reason, so match the plotting code and exclude the sentinel spike.
    valid = []
    for row in rows:
        try:
            value = float(row.get(metric_key, 0.5))
        except (TypeError, ValueError):
            continue
        if abs(value - 0.5) > 1e-12:
            valid.append(row)
    return valid


def auroc(y_true: np.ndarray, uncertainty: np.ndarray) -> float | None:
    if len(y_true) < 4 or len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, -uncertainty))


def aurec(y_true: np.ndarray, uncertainty: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    order = np.argsort(-uncertainty)
    errors = (1 - y_true[order]).astype(float)
    suffix_errors = np.cumsum(errors[::-1])[::-1]
    n_remaining = np.arange(len(y_true), 0, -1, dtype=float)
    return float(np.mean(suffix_errors / n_remaining))


def clean_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [row for row in rows if not bool(row.get(f"{prefix}_generation_failed", False))]


def summarise_slice(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    clean = clean_rows(rows, prefix)
    y = np.array([1.0 if row.get(f"{prefix}_correct", False) else 0.0 for row in clean])
    out: dict[str, Any] = {
        "n": len(clean),
        "correct": int(np.sum(y)) if len(y) else 0,
        "wrong": int(len(y) - np.sum(y)) if len(y) else 0,
        "accuracy": float(np.mean(y)) if len(y) else None,
        "metrics": {},
    }
    for metric, label in METRICS:
        rows_for_metric = metric_rows(clean, prefix, metric)
        key = f"{prefix}_{metric}"
        values = []
        labels = []
        for row in rows_for_metric:
            value = row.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
            labels.append(1.0 if row.get(f"{prefix}_correct", False) else 0.0)
        arr = np.array(values, dtype=float)
        yy = np.array(labels, dtype=float)
        out["metrics"][metric] = {
            "label": label,
            "usable": int(len(arr)),
            "mean": float(np.mean(arr)) if len(arr) else None,
            "auroc": auroc(yy, arr) if len(arr) else None,
            "aurec": aurec(yy, arr) if len(arr) else None,
        }
    return out


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "--"
    return f"{value:.{digits}f}"


def latex_table(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        hop = item["hop"]
        if hop == "5+-hop":
            # Keep the JSON complete, but avoid encouraging inference from n=4.
            continue
        metrics = item["summary"]["metrics"]
        rows.append(
            " & ".join(
                [
                    item["policy"],
                    hop,
                    str(item["summary"]["n"]),
                    fmt(item["summary"]["accuracy"], 3),
                    fmt(metrics["sd_uq"]["mean"], 3),
                    fmt(metrics["vn_entropy"]["mean"], 3),
                    fmt(metrics["support_entailment_uncertainty"]["mean"], 3),
                    fmt(metrics["sd_uq"]["auroc"], 3),
                    fmt(metrics["vn_entropy"]["auroc"], 3),
                    fmt(metrics["support_entailment_uncertainty"]["auroc"], 3),
                    fmt(metrics["graph_path_support"]["auroc"], 3),
                ]
            )
            + r" \\"
        )
    return r"""\begin{table*}[t]
\centering
\small
\caption{%
  Hop-wise 2WikiMultiHopQA diagnostics on the same fixed $n=250$ subset.
  The table uses clean answered rows. Mean scores show answer-surface or
  evidence-support collapse directly; AUROC is omitted when a hop slice has
  only one correctness class. The 5+-hop bucket contains only four questions
  and is excluded from the table but retained in the released JSON artefact.%
}
\label{tab:twowiki_hopwise_diagnostics}
\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{@{}llrrrrrrrrr@{}}
\toprule
Policy & Hop & $n$ & Acc. & SD mean & VN mean & SEU mean &
SD AUC & VN AUC & SEU AUC & GPS AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def main() -> None:
    output: list[dict[str, Any]] = []
    for policy_name, path, config_name, prefix in POLICIES:
        details = load_config(path, config_name)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in details:
            label = hop_label(row)
            if label:
                buckets[label].append(row)
        for hop in sorted(buckets, key=lambda x: 99 if x == "5+-hop" else int(x.split("-", 1)[0])):
            output.append(
                {
                    "policy": policy_name,
                    "source": str(path.relative_to(ROOT)),
                    "config": config_name,
                    "prefix": prefix,
                    "hop": hop,
                    "summary": summarise_slice(buckets[hop], prefix),
                }
            )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(output, f, indent=2)

    OUT_TEX.write_text(latex_table(output))

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")
    for item in output:
        summary = item["summary"]
        metrics = summary["metrics"]
        print(
            f"{item['policy']:<14} {item['hop']:<6} "
            f"n={summary['n']:<3} acc={fmt(summary['accuracy'])} "
            f"sd_mean={fmt(metrics['sd_uq']['mean'])} "
            f"vn_mean={fmt(metrics['vn_entropy']['mean'])} "
            f"seu_mean={fmt(metrics['support_entailment_uncertainty']['mean'])} "
            f"sd_auc={fmt(metrics['sd_uq']['auroc'])} "
            f"vn_auc={fmt(metrics['vn_entropy']['auroc'])} "
            f"seu_auc={fmt(metrics['support_entailment_uncertainty']['auroc'])} "
            f"gps_auc={fmt(metrics['graph_path_support']['auroc'])} "
            f"gps_usable={metrics['graph_path_support']['usable']}/{summary['n']}"
        )


if __name__ == "__main__":
    main()
