"""GPS-v2 structural replay — recompute Graph Path Support from saved answer logs.

No generation or retrieval is rerun.  For each answered question in the
canonical reported runs, this script re-resolves question/answer entities
against the persistent dataset KG in Neo4j and recomputes structural support
with two upgrades over GPS-v1:

  (a) soft answer-entity linking: in addition to the v1 surface/fuzzy matcher,
      candidate answer spans are embedded (all-MiniLM-L6-v2) and linked to KG
      entities by cosine similarity >= tau.  Soft links carry their cosine as a
      link weight; v1 surface matches carry weight 1.0.
  (b) graded path support: reachability is depth-probed (L = 1..max_hops) and
      a reachable answer entity contributes gamma^(L-1) instead of 1.

      GPS-v2 = 1 - sum_e(w_e * gamma^(L_e - 1) * reach_e) / sum_e(w_e)

Anti-overfitting protocol (paper revision_plan item 12): tau and gamma are
tuned on the RealMedQA replay only (--grid), frozen, then evaluated once on
the remaining datasets (--frozen).  The question side uses the v1 matcher with
a fixed-threshold soft fallback (TAU_Q) that is NOT part of the grid, so the
per-question path-length store is reusable across grid points.

Usage:
  .venv/bin/python experiments/gps_v2_replay.py --dataset realmedqa --grid
  .venv/bin/python experiments/gps_v2_replay.py --dataset all --tau 0.72 --gamma 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.uncertainty_metrics import (  # noqa: E402
    _candidate_structural_spans,
    _load_scoped_entities,
    _match_scoped_entities_to_text,
    _primary_answer_span,
    _question_local_entity_support_predicate,
    _question_local_path_support_predicate,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "paper", "figures", "latest_results_manifest.json")

DATASETS = {
    # slug -> (manifest key, question_scoped, structural max_hops)
    "pubmedqa": ("Pubmedqa", True, 2),
    "realmedqa": ("Realmedqa", False, 2),
    "hotpotqa": ("Hotpotqa", True, 2),
    "hotpotqa_fullwiki": ("HotpotqaFullWiki", False, 2),
    "2wikimultihopqa": ("2Wikimultihopqa", True, 2),
    "musique": ("Musique", True, 4),
}

DENSE_CFG = "dense_floor_thr0.1_k10_rt0p0"
KG_CFG = "kg_entity_first_thr0.1_k10_rt0p0"

MIN_NAME_LENGTH = 4
MAX_ENTITIES = 5          # per side, matching GPS-v1
MAX_SOFT_SPANS = 24       # candidate spans embedded per text
TAU_LOOSE = float(os.environ.get("GPS_TAU_LOOSE", 0.60))          # store floor; grid taus filter above this
TAU_Q = 0.75              # fixed question-side soft-fallback threshold (not tuned)
GRID_TAU = tuple(float(x) for x in os.environ.get("GPS_GRID_TAU", "0.60,0.65,0.72,0.80").split(","))
GRID_GAMMA = tuple(float(x) for x in os.environ.get("GPS_GRID_GAMMA", "0.5,0.7,1.0").split(","))


def _load_env() -> Dict[str, str]:
    env_path = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password"),
        "database": os.environ.get("NEO4J_DATABASE", "neo4j"),
    }


def _select_config(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
    for cfg in doc.get("config_results", []):
        if cfg.get("config", {}).get("name") == name:
            return cfg
    raise ValueError(f"missing config {name}")


def _merged_rows(result_path: str, kg_cfg: str = KG_CFG, with_dense: bool = True) -> List[Dict[str, Any]]:
    """vanilla_* fields from the dense config, kg_* from entity-first (paper convention)."""
    doc = json.load(open(result_path))
    dense = (
        {r["question_id"]: r for r in _select_config(doc, DENSE_CFG)["details"]}
        if with_dense
        else {}
    )
    rows = []
    for r in _select_config(doc, kg_cfg)["details"]:
        row = dict(r)
        d = dense.get(r["question_id"], {})
        for k, v in d.items():
            if k.startswith("vanilla_"):
                row[k] = v
        rows.append(row)
    return rows


class ScopedEntityIndex:
    """Entities (with embeddings) visible in one (kg_name, question_id) scope."""

    def __init__(self, session, kg_name: str, question_id: Optional[str]):
        scope = _question_local_entity_support_predicate(
            "e", kg_name=kg_name, question_id=question_id
        )
        self.rows = _load_scoped_entities(
            session,
            entity_scope=scope,
            kg_name=kg_name,
            question_id=question_id,
            min_name_length=MIN_NAME_LENGTH,
        )
        params = {"min_len": MIN_NAME_LENGTH}
        if kg_name:
            params["kg_name"] = kg_name
        if question_id:
            params["question_id"] = question_id
        emb_rows = session.run(
            f"""
            MATCH (e:__Entity__)
            WHERE size(e.name) >= $min_len AND e.embedding IS NOT NULL AND {scope}
            RETURN DISTINCT e.id AS id, e.name AS name, e.embedding AS emb
            LIMIT 20000
            """,
            params,
        )
        ids, names, embs = [], [], []
        for r in emb_rows:
            ids.append(str(r["id"]))
            names.append(str(r["name"]))
            embs.append(np.asarray(r["emb"], dtype=np.float32))
        self.emb_ids = ids
        self.emb_names = names
        if embs:
            m = np.vstack(embs)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.emb_matrix = m / norms
        else:
            self.emb_matrix = np.zeros((0, 384), dtype=np.float32)

    def surface_match(self, text: str) -> List[Dict[str, Any]]:
        return _match_scoped_entities_to_text(
            text, self.rows, min_name_length=MIN_NAME_LENGTH, max_entities=MAX_ENTITIES
        )

    def soft_match(self, text: str, encoder, tau: float) -> List[Dict[str, Any]]:
        """Top entities by max cosine between any candidate span and the entity name."""
        if self.emb_matrix.shape[0] == 0:
            return []
        spans = _candidate_structural_spans(text, min_name_length=MIN_NAME_LENGTH)[
            :MAX_SOFT_SPANS
        ]
        if not spans:
            return []
        span_embs = encoder.encode(spans, normalize_embeddings=True)
        sims = span_embs @ self.emb_matrix.T          # (spans, entities)
        best = sims.max(axis=0)                       # best span per entity
        order = np.argsort(-best)
        out = []
        for idx in order[: MAX_ENTITIES * 3]:
            cos = float(best[idx])
            if cos < tau:
                break
            out.append({"id": self.emb_ids[idx], "name": self.emb_names[idx], "cos": cos})
        return out


def _min_path_length(
    session,
    q_ids: List[str],
    a_id: str,
    kg_name: str,
    question_id: Optional[str],
    max_hops: int,
) -> Optional[int]:
    """Smallest L in 1..max_hops with a qualifying path, mirroring GPS-v1 filters."""
    path_scope = _question_local_path_support_predicate(
        path_var="p", kg_name=kg_name, question_id=question_id
    )
    params: Dict[str, Any] = {"q_ids": q_ids, "a_id": a_id}
    if kg_name:
        params["kg_name"] = kg_name
    if question_id:
        params["question_id"] = question_id
    for hop in range(1, max_hops + 1):
        query = f"""
        UNWIND $q_ids AS q_id
        MATCH (q_e:__Entity__ {{id: q_id}})
        MATCH (a_e:__Entity__ {{id: $a_id}})
        MATCH p = (q_e)-[*{hop}..{hop}]-(a_e)
        WHERE ALL(r IN relationships(p) WHERE coalesce(r.confidence, 1.0) >= 0.4)
          {path_scope}
        RETURN 1 LIMIT 1
        """
        if list(session.run(query, params, timeout=30)):
            return hop
    return None


def replay_question(
    session,
    encoder,
    index: ScopedEntityIndex,
    question: str,
    answer: str,
    kg_name: str,
    question_id: Optional[str],
    max_hops: int,
) -> Dict[str, Any]:
    """Resolve links + path lengths once, at the loose threshold (store row)."""
    q_matches = index.surface_match(question)
    q_source = "surface"
    if not q_matches:
        q_matches = [
            {"id": m["id"], "name": m["name"]}
            for m in index.soft_match(question, encoder, TAU_Q)[:MAX_ENTITIES]
        ]
        q_source = "soft"
    q_ids = [str(m.get("id")) for m in q_matches if m.get("id")]
    if not q_ids:
        return {"null_reason": "no_q_entities"}

    span = _primary_answer_span(answer)
    a_links: Dict[str, Dict[str, Any]] = {}
    for m in index.surface_match(span):
        a_id = str(m.get("id"))
        if a_id:
            a_links[a_id] = {"id": a_id, "name": m.get("name"), "cos": 1.0, "source": "surface"}
    for m in index.soft_match(span, encoder, TAU_LOOSE):
        if m["id"] not in a_links:
            a_links[m["id"]] = {**m, "source": "soft"}
    # v1 trivial-overlap rule: an answer entity that is also a question entity
    # does not count, and its path sources exclude itself.
    answers = []
    for a in a_links.values():
        sources = [q for q in q_ids if q != a["id"]]
        if not sources:
            continue
        a["q_sources"] = sources
        answers.append(a)
    answers.sort(key=lambda a: (-a["cos"], a["id"]))
    answers = answers[: MAX_ENTITIES * 2]
    if not answers:
        return {"null_reason": "no_a_entities", "q_ids": q_ids, "q_source": q_source}

    for a in answers:
        a["L"] = _min_path_length(
            session, a["q_sources"], a["id"], kg_name, question_id, max_hops
        )
        del a["q_sources"]
    return {"null_reason": None, "q_ids": q_ids, "q_source": q_source, "answers": answers}


def score_from_store(store_row: Dict[str, Any], tau: float, gamma: float) -> Dict[str, Any]:
    """Evaluate one (tau, gamma) grid point from a stored replay row."""
    if store_row.get("null_reason"):
        return {"score": 0.5, "null_reason": store_row["null_reason"]}
    kept = [
        a
        for a in store_row["answers"]
        if a["source"] == "surface" or a["cos"] >= tau
    ][:MAX_ENTITIES]
    if not kept:
        return {"score": 0.5, "null_reason": "no_a_entities"}
    num = sum(a["cos"] * (gamma ** (a["L"] - 1)) for a in kept if a["L"] is not None)
    den = sum(a["cos"] for a in kept)
    return {"score": float(1.0 - num / den), "null_reason": None}


def auroc(scores: List[float], wrong: List[bool]) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(wrong)) < 2:
        return None
    return float(roc_auc_score([int(w) for w in wrong], scores))


def summarize(rows: List[Dict[str, Any]], side: str, tau: float, gamma: float) -> Dict[str, Any]:
    scored, wrong, reasons = [], [], Counter()
    answered = 0
    for r in rows:
        if r.get(f"{side}_generation_failed"):
            continue
        answered += 1
        res = score_from_store(r["store"][side], tau, gamma)
        if res["null_reason"]:
            reasons[res["null_reason"]] += 1
            continue
        scored.append(res["score"])
        wrong.append(not bool(r.get(f"{side}_correct")))
    return {
        "tau": tau,
        "gamma": gamma,
        "answered": answered,
        "usable": len(scored),
        "coverage": round(len(scored) / answered, 3) if answered else None,
        "auroc": auroc(scored, wrong),
        "abstain_reasons": dict(reasons),
    }


def run_dataset(slug: str, env: Dict[str, str], encoder, grid: bool, tau: float, gamma: float):
    from neo4j import GraphDatabase

    manifest_key, scoped, max_hops = DATASETS[slug]
    runfile = os.environ.get("GPS_RUNFILE", "")
    kg_cfg = os.environ.get("GPS_KG_CONFIG", KG_CFG)
    if runfile:
        result_path = os.path.join(REPO_ROOT, runfile)
        rows = _merged_rows(result_path, kg_cfg=kg_cfg, with_dense=False)
    else:
        manifest = json.load(open(MANIFEST))
        result_path = os.path.join(REPO_ROOT, manifest[manifest_key]["result_path"])
        rows = _merged_rows(result_path)
    print(f"[{slug}] {len(rows)} questions from {os.path.basename(result_path)}")

    driver = GraphDatabase.driver(env["uri"], auth=(env["user"], env["password"]))
    shared_index = None
    t0 = time.time()
    with driver.session(database=env["database"]) as session:
        if not scoped:
            shared_index = ScopedEntityIndex(session, slug, None)
            print(f"[{slug}] shared index: {len(shared_index.emb_ids)} embedded entities")
        for i, r in enumerate(rows):
            qid = str(r["question_id"]) if scoped else None
            index = shared_index or ScopedEntityIndex(session, slug, qid)
            r["store"] = {}
            sides = (
                (("kg", "kg_response"),)
                if runfile
                else (("kg", "kg_response"), ("vanilla", "vanilla_response"))
            )
            for side, resp_key in sides:
                if r.get(f"{side}_generation_failed") or not r.get(resp_key):
                    r["store"][side] = {"null_reason": "generation_failed"}
                    continue
                r["store"][side] = replay_question(
                    session, encoder, index, r["question"], r[resp_key], slug, qid, max_hops
                )
            if (i + 1) % 25 == 0:
                print(f"[{slug}] {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    driver.close()

    grid_points = (
        [(t, g) for t in GRID_TAU for g in GRID_GAMMA] if grid else [(tau, gamma)]
    )
    eval_sides = ("kg",) if runfile else ("kg", "vanilla")
    summary = {
        side: [summarize(rows, side, t, g) for (t, g) in grid_points]
        for side in eval_sides
    }
    # v1 comparison from the logged scores (0.5 sentinel rows excluded — the
    # logs do not retain null reasons, so exact-0.5 is used as the abstention
    # marker, matching how the paper's AUROC pipeline drops sentinel rows).
    v1 = {}
    for side in eval_sides:
        s, w = [], []
        for r in rows:
            if r.get(f"{side}_generation_failed"):
                continue
            val = r.get(f"{side}_graph_path_support")
            if val is None or abs(val - 0.5) < 1e-9:
                continue
            s.append(float(val))
            w.append(not bool(r.get(f"{side}_correct")))
        v1[side] = {"usable": len(s), "auroc": auroc(s, w)}

    out = {
        "replay": "gps_v2",
        "date": str(date.today()),
        "dataset": slug,
        "source_result": os.path.relpath(result_path, REPO_ROOT),
        "params": {
            "min_name_length": MIN_NAME_LENGTH,
            "max_entities": MAX_ENTITIES,
            "tau_loose_store": TAU_LOOSE,
            "tau_q_fixed": TAU_Q,
            "max_hops": max_hops,
            "mode": "grid" if grid else "frozen",
        },
        "v1_logged": v1,
        "summary": summary,
        "questions": [
            {
                "question_id": r["question_id"],
                "kg_correct": r.get("kg_correct"),
                "vanilla_correct": r.get("vanilla_correct"),
                "store": r["store"],
            }
            for r in rows
        ],
    }
    suffix = "_strict" if runfile else ""
    out_path = os.path.join(
        os.path.dirname(result_path), f"gps_v2_replay_{slug}{suffix}_{date.today():%Y%m%d}.json"
    )
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[{slug}] wrote {out_path}")
    for side in eval_sides:
        print(f"[{slug}] {side}: v1 usable={v1[side]['usable']} auroc={v1[side]['auroc']}")
        for s in summary[side]:
            print(
                f"[{slug}] {side}: v2 tau={s['tau']} gamma={s['gamma']} "
                f"usable={s['usable']}/{s['answered']} auroc={s['auroc']}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="slug or 'all'")
    ap.add_argument("--grid", action="store_true", help="dev grid (RealMedQA only)")
    ap.add_argument("--tau", type=float, default=0.72)
    ap.add_argument("--gamma", type=float, default=0.7)
    args = ap.parse_args()

    if args.grid and args.dataset != "realmedqa":
        ap.error("--grid is restricted to realmedqa (anti-overfitting protocol)")

    env = _load_env()
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    slugs = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for slug in slugs:
        run_dataset(slug, env, encoder, grid=args.grid, tau=args.tau, gamma=args.gamma)


if __name__ == "__main__":
    main()
