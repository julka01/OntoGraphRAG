"""Reporting robustness checks from saved artefacts only.

This script adds manuscript-facing bookkeeping that reviewers commonly ask
for: answered/failure counts, a small complete-case check, and a
dataset-stratified certificate effect size. It does not rerun retrieval,
generation, judging, or KG construction.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.certificate_eval import certificate_mask, load_all  # noqa: E402
from experiments.trust_analysis import KEY, MANIFEST, REPO, STRICT_RUNS, rows_for  # noqa: E402

RNG = np.random.default_rng(42)
B = 5000


def answered_counts():
    out = {}
    for slug, manifest_key in KEY.items():
        doc = json.load(open(os.path.join(REPO, MANIFEST[manifest_key]["result_path"])))
        out[slug] = {}
        for cfg in doc["config_results"]:
            name = cfg["config"]["name"]
            prefix = "kg" if name.startswith("kg") else "vanilla"
            policy = "kg" if prefix == "kg" else "dense"
            total = len(cfg["details"])
            failed = sum(
                1 for row in cfg["details"]
                if row.get(f"{prefix}_generation_failed")
            )
            out[slug][policy] = {
                "total": total,
                "answered": total - failed,
                "failures": failed,
                "failure_rate": failed / total if total else None,
            }
    for slug, rel_path in STRICT_RUNS.items():
        doc = json.load(open(os.path.join(REPO, rel_path)))
        cfg = doc["config_results"][0]
        total = len(cfg["details"])
        failed = sum(
            1 for row in cfg["details"] if row.get("kg_generation_failed")
        )
        out.setdefault(slug, {})["strict_kg"] = {
            "total": total,
            "answered": total - failed,
            "failures": failed,
            "failure_rate": failed / total if total else None,
        }
    return out


def realmedqa_adaptive_strict_complete_case():
    adaptive_doc = json.load(open(os.path.join(REPO, MANIFEST[KEY["realmedqa"]]["result_path"])))
    strict_doc = json.load(open(os.path.join(REPO, STRICT_RUNS["realmedqa"])))
    adaptive_rows = next(
        cfg["details"] for cfg in adaptive_doc["config_results"]
        if cfg["config"]["name"].startswith("kg")
    )
    strict_rows = strict_doc["config_results"][0]["details"]
    adaptive_answered = {
        str(row["question_id"]) for row in adaptive_rows
        if not row.get("kg_generation_failed")
    }
    strict_answered = {
        str(row["question_id"]) for row in strict_rows
        if not row.get("kg_generation_failed")
    }
    return {
        "adaptive_answered": len(adaptive_answered),
        "strict_answered": len(strict_answered),
        "both_answered": len(adaptive_answered & strict_answered),
    }


def odds_ratio_mh(tables):
    """Mantel-Haenszel common odds ratio over per-dataset 2x2 tables."""
    num = 0.0
    den = 0.0
    for table in tables.values():
        a = table["cert_correct"]
        b = table["cert_wrong"]
        c = table["uncert_correct"]
        d = table["uncert_wrong"]
        n = a + b + c + d
        if n:
            num += (a * d) / n
            den += (b * c) / n
    if den == 0:
        return None
    return num / den


def certificate_effect():
    data = load_all()
    tables = {}
    row_cache = {}
    for slug, rows in data.items():
        sd_med = float(np.median([sd for sd, _, _, _ in rows]))
        selected = certificate_mask(rows, sd_med)
        correct = np.array([c for _, _, _, c in rows], dtype=bool)
        tables[slug] = {
            "n": len(rows),
            "cert_correct": int((selected & correct).sum()),
            "cert_wrong": int((selected & ~correct).sum()),
            "uncert_correct": int((~selected & correct).sum()),
            "uncert_wrong": int((~selected & ~correct).sum()),
        }
        row_cache[slug] = np.array(
            [(bool(selected[i]), bool(correct[i])) for i in range(len(rows))],
            dtype=object,
        )

    pooled = {
        "cert_correct": sum(t["cert_correct"] for t in tables.values()),
        "cert_wrong": sum(t["cert_wrong"] for t in tables.values()),
        "uncert_correct": sum(t["uncert_correct"] for t in tables.values()),
        "uncert_wrong": sum(t["uncert_wrong"] for t in tables.values()),
    }
    pooled["odds_ratio_ha"] = (
        (pooled["cert_correct"] + 0.5) * (pooled["uncert_wrong"] + 0.5)
        / ((pooled["cert_wrong"] + 0.5) * (pooled["uncert_correct"] + 0.5))
    )
    point = odds_ratio_mh(tables)

    boots = []
    for _ in range(B):
        boot_tables = {}
        for slug, rows in row_cache.items():
            idx = RNG.integers(0, len(rows), len(rows))
            sample = rows[idx]
            selected = np.array([x[0] for x in sample], dtype=bool)
            correct = np.array([x[1] for x in sample], dtype=bool)
            boot_tables[slug] = {
                "cert_correct": int((selected & correct).sum()),
                "cert_wrong": int((selected & ~correct).sum()),
                "uncert_correct": int((~selected & correct).sum()),
                "uncert_wrong": int((~selected & ~correct).sum()),
            }
        value = odds_ratio_mh(boot_tables)
        if value is not None and np.isfinite(value):
            boots.append(value)

    return {
        "per_dataset_2x2": tables,
        "pooled_2x2": pooled,
        "mantel_haenszel_or": point,
        "bootstrap_ci": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
        "bootstrap_B": B,
    }


def auroc(scores, wrong):
    return float(roc_auc_score([int(x) for x in wrong], scores))


def headline_effect_sizes():
    adaptive = [
        row for row in rows_for("realmedqa", "kg")
        if row["sd"] is not None
    ]
    strict = [
        row for row in rows_for("realmedqa", "strict")
        if row["sd"] is not None and row["seu"] is not None
    ]
    adaptive_sd = auroc(
        [float(row["sd"]) for row in adaptive],
        [not row["correct"] for row in adaptive],
    )
    strict_sd = auroc(
        [float(row["sd"]) for row in strict],
        [not row["correct"] for row in strict],
    )
    strict_seu = auroc(
        [float(row["seu"]) for row in strict],
        [not row["correct"] for row in strict],
    )

    def rb(value):
        return 2.0 * value - 1.0

    return {
        "note": "Rank-biserial r is reported as 2*AUROC-1, with incorrect answers as the positive class.",
        "realmedqa_adaptive_sd": {
            "auroc": adaptive_sd,
            "rank_biserial_r": rb(adaptive_sd),
            "n": len(adaptive),
        },
        "realmedqa_strict_sd": {
            "auroc": strict_sd,
            "rank_biserial_r": rb(strict_sd),
            "n": len(strict),
        },
        "realmedqa_strict_seu": {
            "auroc": strict_seu,
            "rank_biserial_r": rb(strict_seu),
            "n": len(strict),
        },
    }


def main():
    out = {
        "note": "Saved-artifact reporting checks; no KG/retrieval/generation reruns.",
        "answered_counts": answered_counts(),
        "realmedqa_adaptive_strict_complete_case": realmedqa_adaptive_strict_complete_case(),
        "certificate_effect": certificate_effect(),
        "headline_effect_sizes": headline_effect_sizes(),
    }
    dest = os.path.join(REPO, "results", "analyses", "reporting_robustness.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
