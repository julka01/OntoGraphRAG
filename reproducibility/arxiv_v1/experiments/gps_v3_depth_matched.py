"""Depth-matched GPS replay from saved legacy GPS stores.

This script does not rerun retrieval, generation, entity linking, or KG
construction. It reads saved GPS replay artifacts that already store linked
answer entities and shortest graph path lengths, and changes only the scoring
function:

    legacy support:        gamma ** (L - 1)
    depth-matched support: gamma ** abs(L - expected_hop)

The expected hop is taken from logged per-question hop_count where available
(2WikiMultiHopQA and MuSiQue), from the nominal dataset depth for HotpotQA, and
from 1 for RealMedQA.  The link threshold and gamma remain frozen at the
RealMedQA-calibrated values used in the paper.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gps_v2_paper_numbers import percentile_ranks  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "latest_kg_design_final_metrics", "runs")
OUT_PATH = os.path.join(REPO, "results", "analyses", "gps_v3_depth_matched.json")

TAU = 0.60
GAMMA = 0.4
MAX_ENTITIES = 5
B = 2000
RNG = np.random.default_rng(42)

SLUGS = [
    "pubmedqa",
    "realmedqa",
    "hotpotqa",
    "hotpotqa_fullwiki",
    "2wikimultihopqa",
    "musique",
]


def latest_artifact(slug: str, strict: bool = False) -> Optional[str]:
    middle = "_strict_" if strict else "_"
    pattern = os.path.join(RUNS, "*", f"gps_v2_replay_{slug}{middle}[0-9]*.json")
    paths = glob.glob(pattern)
    return sorted(paths)[-1] if paths else None


def load_artifact(slug: str, strict: bool = False) -> Dict[str, Any]:
    path = latest_artifact(slug, strict=strict)
    if not path:
        raise FileNotFoundError(f"missing GPS replay artifact for {slug} strict={strict}")
    with open(path) as f:
        artifact = json.load(f)
    artifact["_artifact_path"] = os.path.relpath(path, REPO)
    return artifact


def row_map(source_result: str) -> Dict[str, Dict[str, Any]]:
    with open(os.path.join(REPO, source_result)) as f:
        doc = json.load(f)
    rows: Dict[str, Dict[str, Any]] = {}
    for cfg in doc.get("config_results", []):
        for row in cfg.get("details", []):
            rows[str(row["question_id"])] = row
    return rows


def expected_hop(slug: str, row: Optional[Dict[str, Any]]) -> int:
    if slug == "realmedqa":
        return 1
    if slug in {"hotpotqa", "hotpotqa_fullwiki"}:
        return 2
    if slug == "musique":
        return int((row or {}).get("hop_count") or 4)
    if slug == "2wikimultihopqa":
        return int((row or {}).get("hop_count") or 2)
    return 1


def score_from_store_depth_matched(
    store_row: Optional[Dict[str, Any]],
    expected_depth: int,
    tau: float = TAU,
    gamma: float = GAMMA,
) -> Dict[str, Any]:
    if not store_row:
        return {"score": 0.5, "null_reason": "missing_store"}
    if store_row.get("null_reason"):
        return {"score": 0.5, "null_reason": store_row["null_reason"]}
    kept = [
        a
        for a in store_row.get("answers", [])
        if a.get("source") == "surface" or float(a.get("cos", 0.0)) >= tau
    ][:MAX_ENTITIES]
    if not kept:
        return {"score": 0.5, "null_reason": "no_a_entities"}
    den = sum(float(a.get("cos", 0.0)) for a in kept)
    if den <= 0:
        return {"score": 0.5, "null_reason": "no_a_entities"}
    num = sum(
        float(a.get("cos", 0.0)) * (gamma ** abs(int(a["L"]) - expected_depth))
        for a in kept
        if a.get("L") is not None
    )
    return {"score": float(1.0 - num / den), "null_reason": None}


def scores_for(slug: str, strict: bool = False, side: str = "kg") -> Dict[str, Optional[float]]:
    artifact = load_artifact(slug, strict=strict)
    rows = row_map(artifact["source_result"])
    out: Dict[str, Optional[float]] = {}
    for q in artifact["questions"]:
        qid = str(q["question_id"])
        store = (q.get("store") or {}).get(side)
        if store is None or store.get("null_reason") == "generation_failed":
            continue
        res = score_from_store_depth_matched(store, expected_hop(slug, rows.get(qid)))
        out[qid] = None if res["null_reason"] else res["score"]
    return out


def scored_rows(slug: str, strict: bool = False, side: str = "kg") -> list[tuple[str, Optional[float], bool]]:
    artifact = load_artifact(slug, strict=strict)
    rows = row_map(artifact["source_result"])
    out = []
    for q in artifact["questions"]:
        qid = str(q["question_id"])
        store = (q.get("store") or {}).get(side)
        if store is None or store.get("null_reason") == "generation_failed":
            continue
        res = score_from_store_depth_matched(store, expected_hop(slug, rows.get(qid)))
        score = None if res["null_reason"] else res["score"]
        out.append((qid, score, bool(q.get(f"{side}_correct"))))
    return out


def auroc_ci(scores: Iterable[float], wrong: Iterable[bool]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    s = np.asarray(list(scores), float)
    w = np.asarray(list(wrong), int)
    if len(s) == 0 or len(set(w)) < 2:
        return None, None, None
    point = float(roc_auc_score(w, s))
    boot = []
    for _ in range(B):
        idx = RNG.integers(0, len(s), len(s))
        if len(set(w[idx])) < 2:
            continue
        boot.append(float(roc_auc_score(w[idx], s[idx])))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def aurec(correct: Iterable[bool], uncertainty: Iterable[float]) -> Optional[float]:
    y = np.asarray(list(correct), float)
    u = np.asarray(list(uncertainty), float)
    if len(y) == 0:
        return None
    order = np.argsort(-u)
    errors = 1.0 - y[order]
    suffix = np.cumsum(errors[::-1])[::-1]
    remaining = np.arange(len(y), 0, -1, dtype=float)
    return float((suffix / remaining).mean())


def summarise(slug: str, strict: bool = False, side: str = "kg") -> Dict[str, Any]:
    rows = scored_rows(slug, strict=strict, side=side)
    defined = [(s, not c) for _, s, c in rows if s is not None]
    scores = [s for s, _ in defined]
    wrong = [w for _, w in defined]
    point, lo, hi = auroc_ci(scores, wrong)
    return {
        "answered": len(rows),
        "usable": len(defined),
        "auroc": point,
        "ci": [lo, hi],
        "aurec": aurec([not w for w in wrong], scores),
    }


def metric_rows(slug: str, policy: str) -> list[Dict[str, Any]]:
    manifest = json.load(open(os.path.join(REPO, "paper", "figures", "latest_results_manifest.json")))
    key = {
        "pubmedqa": "Pubmedqa",
        "realmedqa": "Realmedqa",
        "hotpotqa": "Hotpotqa",
        "hotpotqa_fullwiki": "HotpotqaFullWiki",
        "2wikimultihopqa": "2Wikimultihopqa",
        "musique": "Musique",
    }[slug]
    if policy == "strict":
        artifact = load_artifact(slug, strict=True)
        path = os.path.join(REPO, artifact["source_result"])
        cfg_prefix = "kg_strict_entity_first"
        prefix = "kg"
    else:
        path = os.path.join(REPO, manifest[key]["result_path"])
        cfg_prefix = "kg_entity_first" if policy == "kg" else "dense_floor"
        prefix = "kg" if policy == "kg" else "vanilla"
    doc = json.load(open(path))
    details = next(
        cfg for cfg in doc["config_results"]
        if cfg["config"]["name"].startswith(cfg_prefix)
    )["details"]
    out = []
    gps = scores_for(slug, strict=(policy == "strict"), side=prefix)
    for row in details:
        if row.get(f"{prefix}_generation_failed"):
            continue
        qid = str(row["question_id"])
        out.append({
            "qid": qid,
            "correct": bool(row.get(f"{prefix}_correct")),
            "sd": row.get(f"{prefix}_sd_uq"),
            "seu": row.get(f"{prefix}_support_entailment_uncertainty"),
            "gps": gps.get(qid),
        })
    return out


def spearman_ci(x: list[float], y: list[float]) -> Dict[str, Any]:
    x_arr = np.asarray(x, float)
    y_arr = np.asarray(y, float)
    rho = float(spearmanr(x_arr, y_arr).statistic)
    boot = []
    for _ in range(B):
        idx = RNG.integers(0, len(x_arr), len(x_arr))
        boot.append(float(spearmanr(x_arr[idx], y_arr[idx]).statistic))
    return {
        "rho": rho,
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": int(len(x_arr)),
    }


def main() -> None:
    out: Dict[str, Any] = {
        "method": "Depth-matched GPS replay from saved legacy GPS stores",
        "frozen": {"tau": TAU, "gamma": GAMMA},
        "adaptive": {},
        "strict": {},
        "correlations": {},
        "composite": {},
    }

    pooled = {"sd": [], "seu": [], "gps": []}
    for slug in SLUGS:
        out["adaptive"][slug] = {
            "kg": summarise(slug, side="kg"),
            "vanilla": summarise(slug, side="vanilla"),
            "artifact": load_artifact(slug)["_artifact_path"],
        }
        rows = [
            r for r in metric_rows(slug, "kg")
            if r["sd"] is not None and r["seu"] is not None
        ]
        for r in rows:
            if r["gps"] is not None:
                pooled["sd"].append(float(r["sd"]))
                pooled["seu"].append(float(r["seu"]))
                pooled["gps"].append(float(r["gps"]))
        if len(rows) >= 12:
            sd_r = percentile_ranks([float(r["sd"]) for r in rows])
            seu_r = percentile_ranks([float(r["seu"]) for r in rows])
            gps_r = percentile_ranks([1.0 if r["gps"] is None else float(r["gps"]) for r in rows])
            combined = [(a + b + c) / 3 for a, b, c in zip(sd_r, seu_r, gps_r)]
            wrong = [not r["correct"] for r in rows]
            correct = [r["correct"] for r in rows]
            out["composite"][slug] = {
                "n": len(rows),
                "sd_auc": auroc_ci([float(r["sd"]) for r in rows], wrong)[0],
                "seu_auc": auroc_ci([float(r["seu"]) for r in rows], wrong)[0],
                "gps_risk_auc": auroc_ci([1.0 if r["gps"] is None else float(r["gps"]) for r in rows], wrong)[0],
                "combined_auc": auroc_ci(combined, wrong)[0],
                "combined_aurec": aurec(correct, combined),
            }

    for slug in ("realmedqa", "2wikimultihopqa"):
        out["strict"][slug] = {
            "kg": summarise(slug, strict=True, side="kg"),
            "artifact": load_artifact(slug, strict=True)["_artifact_path"],
        }

    if pooled["gps"]:
        out["correlations"]["sd_vs_seu"] = spearman_ci(pooled["sd"], pooled["seu"])
        out["correlations"]["sd_vs_gps"] = spearman_ci(pooled["sd"], pooled["gps"])
        out["correlations"]["seu_vs_gps"] = spearman_ci(pooled["seu"], pooled["gps"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
