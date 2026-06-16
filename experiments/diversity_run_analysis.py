"""Per-sample graph-state diversity analysis for a fresh diversity run.

Reads a run produced with graph-state tracing enabled (current experiment.py
emits per-question kg_seed_entity_jaccard / kg_path_jaccard / kg_subgraph_jaccard
/ kg_chunk_jaccard and the entropy variants).  Computes:

  1. how stable the retrieval STATE is across the N=5 samples (mean Jaccard per
     family) -- the direct grounding of Eq. 2 beyond chunk overlap; and
  2. among WRONG KG answers, the Spearman coupling between retrieval-state
     overlap and answer-state dispersion (SD-UQ) -- higher overlap should
     predict lower dispersion (the lock-in mechanism).

No generation/retrieval rerun here; this only reads the saved run JSON.
Usage: python -m experiments.diversity_run_analysis <glob-or-path-to-results.json>
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "analyses", "diversity_run_analysis.json")

OVERLAPS = [
    ("seed_entity", "kg_seed_entity_jaccard"),
    ("path", "kg_path_jaccard"),
    ("subgraph", "kg_subgraph_jaccard"),
    ("chunk", "kg_chunk_jaccard"),
]


def load_details(path):
    doc = json.load(open(path))
    for cfg in doc["config_results"]:
        if cfg["config"]["name"].startswith("kg_entity_first"):
            return cfg["details"]
    raise SystemExit("no kg_entity_first config in " + path)


def fin(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def main(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        raise SystemExit("no run JSON matched: " + pattern)
    path = paths[-1]
    details = [r for r in load_details(path)
               if not r.get("kg_generation_failed") and not r.get("kg_system_skipped")]
    # rows with a real multi-sample graph trace
    rows = [r for r in details if (r.get("kg_graph_state_sample_count") or 0) >= 2]
    wrong = [r for r in rows if not r.get("kg_correct")]

    out = {"run": os.path.relpath(path, REPO), "n_answered": len(details),
           "n_with_traces": len(rows), "n_wrong_with_traces": len(wrong),
           "stability_mean_jaccard": {}, "coupling_overlap_vs_sduq_amongwrong": {}}

    for name, key in OVERLAPS:
        vals = [float(r[key]) for r in rows if fin(r.get(key))]
        out["stability_mean_jaccard"][name] = {
            "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "n": len(vals),
        }

    for name, key in OVERLAPS:
        xs, ys = [], []
        for r in wrong:
            if fin(r.get(key)) and fin(r.get("kg_sd_uq")):
                xs.append(float(r[key]))
                ys.append(float(r["kg_sd_uq"]))
        if len(xs) >= 5 and len(set(xs)) > 1 and len(set(ys)) > 1:
            rho, p = spearmanr(xs, ys)
            out["coupling_overlap_vs_sduq_amongwrong"][name] = {
                "spearman_rho": float(rho), "p": float(p), "n": len(xs)}
        else:
            out["coupling_overlap_vs_sduq_amongwrong"][name] = {
                "spearman_rho": None, "p": None, "n": len(xs),
                "note": "too few/constant"}

    dsf = [float(r["kg_dominant_seed_entity_fraction"]) for r in rows
           if fin(r.get("kg_dominant_seed_entity_fraction"))]
    out["dominant_seed_entity_fraction_mean"] = float(np.mean(dsf)) if dsf else None

    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1))
    print("wrote", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO, "results", "diversity_run", "runs", "*realmedqa*", "mirage_realmedqa_results.json")
    main(pat)
