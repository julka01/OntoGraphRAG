"""Audit the 'fully calm' residue of adaptive-KG silent errors (offline, no rerun).

These are wrong, answer-silent rows (DSE=0, SD-UQ at floor) that are also calm
on every non-answer family: not on an empty route, SEU<=0.5, and GPS defined and
<=0.5.  The silent-error accounting reports ~7 such rows pooled; this script
pulls their question, gold answer, and generated answer so they can be
qualitatively characterised (deep presence lock-in vs parametric overconfidence
vs labelling artefact).  Retrieved chunk texts are not archived, so entailment
of the wrong answer cannot be confirmed from logs -- that boundary is reported
honestly.  Output: results/analyses/residue_audit.json
"""

from __future__ import annotations

import json
import os

from experiments.lockin_accounting_analysis import (
    KEY,
    MANIFEST,
    REPO,
    gps_cache,
    gps_flags,
    is_silent,
    route_state,
    seu_flags,
)

OUT_PATH = os.path.join(REPO, "results", "analyses", "residue_audit.json")
SLUGS = ["pubmedqa", "realmedqa", "hotpotqa", "hotpotqa_fullwiki",
         "2wikimultihopqa", "musique"]


def calm_rows_for(slug):
    """Adaptive-KG fully-calm silent errors, with question/answer text."""
    path = os.path.join(REPO, MANIFEST[KEY[slug]]["result_path"])
    doc = json.load(open(path))
    details = next(
        cfg["details"] for cfg in doc["config_results"]
        if cfg["config"]["name"].startswith("kg_entity_first")
    )
    gps = gps_cache(slug, "kg", "kg")
    found = []
    for r in details:
        if r.get("kg_generation_failed") or r.get("kg_system_skipped"):
            continue
        qid = str(r["question_id"])
        row = {
            "qid": qid, "policy": "kg",
            "correct": bool(r.get("kg_correct")),
            "dse": r.get("kg_discrete_semantic_entropy"),
            "sd": r.get("kg_sd_uq"),
            "seu": r.get("kg_support_entailment_uncertainty"),
            "gps": gps.get(qid),
            "route": str(r.get("kg_retrieval_route") or ""),
            "route_reason": str(r.get("kg_route_reason") or ""),
        }
        if row["correct"] or not is_silent(row):
            continue
        if route_state(row) == "empty" or seu_flags(row) or gps_flags(row):
            continue
        found.append({
            "slug": slug,
            "question_id": qid,
            "question": r.get("question"),
            "expected": r.get("expected"),
            "response": r.get("kg_response"),
            "sd_uq": row["sd"], "seu": row["seu"], "gps": row["gps"],
            "route": row["route"], "route_reason": row["route_reason"],
        })
    return found


def main():
    residue = []
    for slug in SLUGS:
        try:
            residue.extend(calm_rows_for(slug))
        except Exception as exc:  # pragma: no cover
            print(f"skip {slug}: {exc}")
    out = {
        "definition": "adaptive-KG wrong & silent (DSE=0, SD-UQ<=1e-9) & route!=empty "
                      "& SEU<=0.5 & GPS defined<=0.5",
        "caveat": "retrieved chunk texts not archived; entailment of the wrong "
                  "answer cannot be confirmed from logs",
        "count": len(residue),
        "cases": residue,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=1, default=float)
    print(f"residue count: {len(residue)}")
    for c in residue:
        print(f"\n[{c['slug']}] {c['question']}")
        print(f"   gold: {c['expected']}")
        print(f"   ans : {str(c['response'])[:160]}")
        print(f"   sd={c['sd_uq']!s:.6} seu={c['seu']} gps={c['gps']} route={c['route']}/{c['route_reason']}")
    print("\nwrote", os.path.relpath(OUT_PATH, REPO))


if __name__ == "__main__":
    main()
