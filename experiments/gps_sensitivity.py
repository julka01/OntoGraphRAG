"""GPS hyperparameter sensitivity sweep (offline replay; no rerun, no KG rebuild).

Reuses saved GPS replay stores (per-answer-entity cosine link scores and
shortest path lengths) and the depth-matched GPS scorer, sweeping only the two
calibrated hyperparameters:

    tau   -- soft answer-entity link threshold (frozen at 0.60 in the paper)
    gamma -- distance decay base                (frozen at 0.40 in the paper)

For each reported KG run it recomputes GPS over the usable rows at every
(tau, gamma) cell, then reports the incorrect-as-positive AUROC and the usable
denominator.  This quantifies how much the headline GPS AUROC moves under
plausible hyperparameter perturbations, separating the calibration domain
(RealMedQA) from the held-out suites.  Output: results/analyses/gps_sensitivity.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.gps_v3_depth_matched import (
    REPO,
    expected_hop,
    load_artifact,
    row_map,
    score_from_store_depth_matched,
)

OUT_PATH = os.path.join(REPO, "results", "analyses", "gps_sensitivity.json")

TAUS = [0.50, 0.55, 0.60, 0.65, 0.70]
GAMMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
CAL_TAU, CAL_GAMMA = 0.60, 0.40

# (label, slug, strict); PubMedQA omitted -- yes/no answers expose no entities.
RUNS = [
    ("RealMedQA adaptive", "realmedqa", False),
    ("RealMedQA strict", "realmedqa", True),
    ("HotpotQA", "hotpotqa", False),
    ("HotpotQA FullWiki", "hotpotqa_fullwiki", False),
    ("2WikiMHQA adaptive", "2wikimultihopqa", False),
    ("2WikiMHQA strict", "2wikimultihopqa", True),
    ("MuSiQue", "musique", False),
]


def auroc_at(slug, strict, tau, gamma):
    """Return (auroc, usable_n) for one (tau, gamma) cell using saved stores."""
    artifact = load_artifact(slug, strict=strict)
    rows = row_map(artifact["source_result"])
    scores, wrong = [], []
    for q in artifact["questions"]:
        qid = str(q["question_id"])
        store = (q.get("store") or {}).get("kg")
        if store is None or store.get("null_reason") == "generation_failed":
            continue
        res = score_from_store_depth_matched(
            store, expected_hop(slug, rows.get(qid)), tau=tau, gamma=gamma
        )
        if res["null_reason"]:
            continue
        scores.append(res["score"])
        wrong.append(0 if q.get("kg_correct") else 1)
    if len(scores) == 0 or len(set(wrong)) < 2:
        return None, len(scores)
    return float(roc_auc_score(wrong, scores)), len(scores)


def main():
    out = {
        "method": "GPS tau/gamma sensitivity sweep, offline replay from saved "
        "GPS stores (no generation/retrieval/linking/KG rerun)",
        "calibrated": {"tau": CAL_TAU, "gamma": CAL_GAMMA},
        "grid": {"tau": TAUS, "gamma": GAMMAS},
        "runs": {},
    }
    for label, slug, strict in RUNS:
        grid = {}
        aurocs = []
        usables = []
        for tau in TAUS:
            for gamma in GAMMAS:
                a, n = auroc_at(slug, strict, tau, gamma)
                grid[f"tau{tau}_gamma{gamma}"] = {"auroc": a, "usable": n}
                if a is not None:
                    aurocs.append(a)
                usables.append(n)
        cal = grid[f"tau{CAL_TAU}_gamma{CAL_GAMMA}"]
        out["runs"][label] = {
            "calibrated_auroc": cal["auroc"],
            "calibrated_usable": cal["usable"],
            "auroc_min": min(aurocs) if aurocs else None,
            "auroc_max": max(aurocs) if aurocs else None,
            "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "auroc_std": float(np.std(aurocs)) if aurocs else None,
            "auroc_range": (max(aurocs) - min(aurocs)) if aurocs else None,
            "usable_min": min(usables),
            "usable_max": max(usables),
            "grid": grid,
        }
        print(
            f"{label:22s} cal={cal['auroc']!s:6.6} "
            f"range=[{out['runs'][label]['auroc_min']!s:.5}, "
            f"{out['runs'][label]['auroc_max']!s:.5}] "
            f"std={out['runs'][label]['auroc_std']:.3f} "
            f"usable {min(usables)}-{max(usables)}"
        )
    json.dump(out, open(OUT_PATH, "w"), indent=1, default=float)
    print("wrote", os.path.relpath(OUT_PATH, REPO))


if __name__ == "__main__":
    main()
