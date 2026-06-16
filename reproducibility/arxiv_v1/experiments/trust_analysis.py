"""Trust-layer analyses from saved logs (no reruns).

Emits results/analyses/trust_analysis.json with:
  A1  silent-failure rates: P(answer-state calm | wrong) per dataset x policy,
      where calm = DSE == 0 and SD-UQ at the numerical floor.
  A2  operating points: selective risk (error rate among accepted) at fixed
      coverage levels for SD-UQ alone vs the composite audit score.
  A4  paired bootstrap for the two headline contrasts:
      (i)  strict RealMedQA: SEU AUROC - SD-UQ AUROC (same rows);
      (ii) RealMedQA SD-UQ AUROC: adaptive vs strict (shared question ids).
"""

import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gps_v2_paper_numbers import percentile_ranks  # noqa: E402
from experiments.gps_v3_depth_matched import scores_for as gps_v3_scores_for  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = json.load(open(os.path.join(REPO, "paper", "figures",
                                       "latest_results_manifest.json")))
KEY = {"pubmedqa": "Pubmedqa", "realmedqa": "Realmedqa", "hotpotqa": "Hotpotqa",
       "hotpotqa_fullwiki": "HotpotqaFullWiki", "2wikimultihopqa": "2Wikimultihopqa",
       "musique": "Musique"}
STRICT_RUNS = {
    "realmedqa": "results/latest_kg_design_final_metrics/runs/"
                 "20260528-174538-realmedqa-n230-full-metrics-evaluation-subset/"
                 "mirage_realmedqa_results.json",
    "2wikimultihopqa": "results/latest_kg_design_final_metrics/runs/"
                       "20260529-120453-2wikimultihopqa-n250-full-metrics-evaluation-subset/"
                       "mirage_2wikimultihopqa_results.json",
}
SD_FLOOR = 1e-9
B = 2000
RNG = np.random.default_rng(42)


def rows_for(slug, policy):
    """policy in {kg, vanilla, strict}; returns list of per-question dicts."""
    if policy == "strict":
        doc = json.load(open(os.path.join(REPO, STRICT_RUNS[slug])))
        details = doc["config_results"][0]["details"]
        prefix = "kg"
    else:
        doc = json.load(open(os.path.join(REPO, MANIFEST[KEY[slug]]["result_path"])))
        cfg_prefix = "kg_entity_first" if policy == "kg" else "dense_floor"
        details = next(c for c in doc["config_results"]
                       if c["config"]["name"].startswith(cfg_prefix))["details"]
        prefix = "kg" if policy == "kg" else "vanilla"
    out = []
    for r in details:
        if r.get(f"{prefix}_generation_failed"):
            continue
        out.append({
            "qid": str(r["question_id"]),
            "correct": bool(r.get(f"{prefix}_correct")),
            "dse": r.get(f"{prefix}_discrete_semantic_entropy"),
            "sd": r.get(f"{prefix}_sd_uq"),
            "seu": r.get(f"{prefix}_support_entailment_uncertainty"),
        })
    return out


def silent_rate(rows):
    wrong = [r for r in rows if not r["correct"]]
    if not wrong:
        return None, 0
    calm = [r for r in wrong
            if r["dse"] is not None and abs(r["dse"]) < 1e-9
            and r["sd"] is not None and r["sd"] <= SD_FLOOR]
    return len(calm) / len(wrong), len(wrong)


def gps_scores(slug, strict=False):
    return gps_v3_scores_for(slug, strict=strict, side="kg")


def selective_risk(scores, correct, coverages=(0.8, 0.9, 0.95, 1.0)):
    order = np.argsort(scores)           # accept most-certain first
    c = np.asarray(correct, float)[order]
    out = {}
    n = len(c)
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        out[f"{int(cov*100)}"] = float(1.0 - c[:k].mean())
    return out


def auroc(scores, wrong):
    if len(set(wrong)) < 2:
        return None
    return float(roc_auc_score([int(w) for w in wrong], scores))


def main():
    out = {"silent_failure": {}, "operating_points": {}, "paired_tests": {}}

    # ── A1: silent-failure rates ────────────────────────────────────────────
    pooled = {"kg": [0, 0], "vanilla": [0, 0], "strict": [0, 0]}
    for slug in KEY:
        out["silent_failure"][slug] = {}
        for policy in ("vanilla", "kg") + (("strict",) if slug in STRICT_RUNS else ()):
            rate, n_wrong = silent_rate(rows_for(slug, policy))
            out["silent_failure"][slug][policy] = {"rate": rate, "n_wrong": n_wrong}
            if rate is not None:
                pooled[policy][0] += round(rate * n_wrong)
                pooled[policy][1] += n_wrong
    out["silent_failure"]["pooled"] = {
        p: {"rate": (v[0] / v[1] if v[1] else None), "n_wrong": v[1]}
        for p, v in pooled.items()
    }

    # ── A2: operating points (KG side, adaptive) ────────────────────────────
    for slug in KEY:
        rows = rows_for(slug, "kg")
        gps = gps_scores(slug)
        usable = [r for r in rows if r["sd"] is not None and r["seu"] is not None]
        if len(usable) < 20:
            continue
        sd = [float(r["sd"]) for r in usable]
        seu = [float(r["seu"]) for r in usable]
        gpsr = [1.0 if gps.get(r["qid"]) is None else float(gps[r["qid"]])
                for r in usable]
        correct = [r["correct"] for r in usable]
        sd_r, seu_r, gps_r = (percentile_ranks(sd), percentile_ranks(seu),
                              percentile_ranks(gpsr))
        combined = [(a + b + c) / 3 for a, b, c in zip(sd_r, seu_r, gps_r)]
        out["operating_points"][slug] = {
            "n": len(usable),
            "sd_only": selective_risk(sd, correct),
            "combined": selective_risk(combined, correct),
        }

    # ── A4(i): strict RealMedQA, paired SEU-vs-SD AUROC difference ─────────
    rows = [r for r in rows_for("realmedqa", "strict")
            if r["sd"] is not None and r["seu"] is not None]
    sd = np.array([r["sd"] for r in rows]); seu = np.array([r["seu"] for r in rows])
    wrong = np.array([not r["correct"] for r in rows], int)
    point = auroc(seu, wrong) - auroc(sd, wrong)
    diffs = []
    for _ in range(B):
        idx = RNG.integers(0, len(rows), len(rows))
        if len(set(wrong[idx])) < 2:
            continue
        diffs.append(roc_auc_score(wrong[idx], seu[idx])
                     - roc_auc_score(wrong[idx], sd[idx]))
    out["paired_tests"]["strict_realmedqa_seu_minus_sd"] = {
        "diff": point, "ci": [float(np.percentile(diffs, 2.5)),
                              float(np.percentile(diffs, 97.5))],
        "n": len(rows),
        "p_leq_0": float(np.mean(np.array(diffs) <= 0)),
    }

    # ── A4(ii): RealMedQA SD-UQ AUROC, adaptive vs strict on shared qids ────
    ad = {r["qid"]: r for r in rows_for("realmedqa", "kg") if r["sd"] is not None}
    st = {r["qid"]: r for r in rows_for("realmedqa", "strict") if r["sd"] is not None}
    shared = sorted(set(ad) & set(st))
    a_sd = np.array([ad[q]["sd"] for q in shared])
    a_w = np.array([not ad[q]["correct"] for q in shared], int)
    s_sd = np.array([st[q]["sd"] for q in shared])
    s_w = np.array([not st[q]["correct"] for q in shared], int)
    point = auroc(a_sd, a_w) - auroc(s_sd, s_w)
    diffs = []
    for _ in range(B):
        idx = RNG.integers(0, len(shared), len(shared))
        if len(set(a_w[idx])) < 2 or len(set(s_w[idx])) < 2:
            continue
        diffs.append(roc_auc_score(a_w[idx], a_sd[idx])
                     - roc_auc_score(s_w[idx], s_sd[idx]))
    out["paired_tests"]["realmedqa_sd_adaptive_minus_strict"] = {
        "diff": point, "ci": [float(np.percentile(diffs, 2.5)),
                              float(np.percentile(diffs, 97.5))],
        "n_shared": len(shared),
        "p_leq_0": float(np.mean(np.array(diffs) <= 0)),
    }

    dest = os.path.join(REPO, "results", "analyses", "trust_analysis.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
