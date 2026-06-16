"""Graph-state diversity analysis for runs with per-sample graph traces.

The current headline run logs predate graph-state tracing, so this script is
also an availability check: it reports which saved runs contain the required
fields and only draws the phase plot when enough rows are present.
"""

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.trust_analysis import KEY, MANIFEST, REPO, STRICT_RUNS  # noqa: E402
from ontographrag.rag.graph_state import graph_state_diversity  # noqa: E402


OUT_JSON = os.path.join(REPO, "results", "analyses", "graph_state_diversity_analysis.json")
FIG_DIR = os.path.join(REPO, "paper", "figures")


def _iter_result_details():
    for slug, manifest_key in KEY.items():
        doc = json.load(open(os.path.join(REPO, MANIFEST[manifest_key]["result_path"])))
        for cfg in doc["config_results"]:
            if not cfg["config"]["name"].startswith("kg_entity_first"):
                continue
            yield slug, "adaptive", cfg["details"]
    for slug, rel_path in STRICT_RUNS.items():
        doc = json.load(open(os.path.join(REPO, rel_path)))
        yield slug, "strict", doc["config_results"][0]["details"]


def _row_diversity(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("kg_graph_state_traces"):
        return graph_state_diversity(row.get("kg_graph_state_traces") or [])
    if row.get("kg_subgraph_entropy") is not None:
        return {
            "sample_count": row.get("kg_graph_state_sample_count", 0),
            "seed_entity_entropy": row.get("kg_seed_entity_entropy", 0.0),
            "path_entropy": row.get("kg_path_entropy", 0.0),
            "subgraph_entropy": row.get("kg_subgraph_entropy", 0.0),
            "chunk_entropy": row.get("kg_chunk_entropy", 0.0),
            "subgraph_jaccard": row.get("kg_subgraph_jaccard", 0.0),
        }
    return {}


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset, policy, details in _iter_result_details():
        for row in details:
            diversity = _row_diversity(row)
            if not diversity:
                continue
            output_entropy = row.get("kg_discrete_semantic_entropy")
            if output_entropy is None:
                continue
            rows.append({
                "dataset": dataset,
                "policy": policy,
                "question_id": str(row.get("question_id")),
                "correct": bool(row.get("kg_correct")),
                "output_discrete_semantic_entropy": float(output_entropy),
                "sd_uq": float(row.get("kg_sd_uq") or 0.0),
                "subgraph_entropy": float(diversity.get("subgraph_entropy") or 0.0),
                "path_entropy": float(diversity.get("path_entropy") or 0.0),
                "seed_entity_entropy": float(diversity.get("seed_entity_entropy") or 0.0),
                "subgraph_jaccard": float(diversity.get("subgraph_jaccard") or 0.0),
                "sample_count": int(diversity.get("sample_count") or 0),
            })
    return rows


def plot(rows: List[Dict[str, Any]]) -> List[str]:
    if plt is None or len(rows) < 5:
        return []
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    x = np.array([r["subgraph_entropy"] for r in rows], dtype=float)
    y = np.array([r["output_discrete_semantic_entropy"] for r in rows], dtype=float)
    wrong = np.array([not r["correct"] for r in rows], dtype=bool)
    ax.scatter(x[~wrong], y[~wrong], s=18, c="#4C78A8", alpha=0.55, label="correct")
    ax.scatter(x[wrong], y[wrong], s=22, c="#E45756", alpha=0.72, label="wrong")
    ax.set_xlabel("Graph-state entropy (subgraph signature, bits)")
    ax.set_ylabel("Output semantic entropy (DSE)")
    ax.set_title("Graph-State Diversity vs Output Dispersion")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    outputs = []
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"graph_state_entropy_vs_output_entropy.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=180)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main():
    rows = collect_rows()
    by_dataset: Dict[str, int] = {}
    for row in rows:
        key = f"{row['dataset']}:{row['policy']}"
        by_dataset[key] = by_dataset.get(key, 0) + 1
    figures = plot(rows)
    out = {
        "status": "ok" if rows else "no_graph_state_trace_fields",
        "note": (
            "Requires per-sample kg_graph_state_traces or derived kg_*_entropy "
            "fields from runs produced after graph-state tracing was added."
        ),
        "row_count": len(rows),
        "rows_by_dataset_policy": by_dataset,
        "figures": figures,
        "rows_preview": rows[:20],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
