"""Diagnostic-triggered anchor-masking intervention.

This is a targeted follow-up experiment for retrieval-state lock-in. It does
not rebuild any KG. By default it only selects candidate rows from saved
results. Pass ``--run`` to execute the intervention on the existing KG:

1. Select rows with a lock-in signature (low SD-UQ, high SEU by default).
2. Re-run the current KG retriever once to identify the dominant seed anchor.
3. Re-run KG-RAG with that seed entity masked from entity-first expansion.
4. Report whether the answer and graph state changed, and whether correctness
   improved under the existing judge.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.answer_formatting import (  # noqa: E402
    build_answer_instructions,
    normalize_answer_to_contract,
)
from experiments.experiment import MIRAGEEvaluationPipeline  # noqa: E402
from experiments.trust_analysis import KEY, MANIFEST, REPO, STRICT_RUNS  # noqa: E402
from experiments.uncertainty_metrics import compute_all_uncertainty_metrics  # noqa: E402
from ontographrag.rag.graph_state import (  # noqa: E402
    dominant_anchor_id,
    graph_state_diversity,
    summarize_context_graph_state,
)
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem  # noqa: E402


def _result_details(dataset: str, policy: str) -> List[Dict[str, Any]]:
    if policy == "strict":
        path = os.path.join(REPO, STRICT_RUNS[dataset])
        doc = json.load(open(path))
        return doc["config_results"][0]["details"]
    path = os.path.join(REPO, MANIFEST[KEY[dataset]]["result_path"])
    doc = json.load(open(path))
    cfg_prefix = "kg_entity_first" if policy == "kg" else "dense_floor"
    return next(
        cfg["details"] for cfg in doc["config_results"]
        if cfg["config"]["name"].startswith(cfg_prefix)
    )


def select_candidates(
    dataset: str,
    policy: str,
    *,
    sd_threshold: float,
    seu_threshold: float,
    wrong_only: bool,
    require_populated_route: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in _result_details(dataset, policy):
        if row.get("kg_generation_failed"):
            continue
        sd = row.get("kg_sd_uq")
        seu = row.get("kg_support_entailment_uncertainty")
        if sd is None or seu is None:
            continue
        if float(sd) > sd_threshold or float(seu) < seu_threshold:
            continue
        if wrong_only and bool(row.get("kg_correct")):
            continue
        route = str(row.get("kg_retrieval_route", "") or "")
        if require_populated_route and (not route or route.endswith("_empty")):
            continue
        candidates.append({
            "question_id": str(row.get("question_id")),
            "question": row.get("question", ""),
            "expected": row.get("expected", ""),
            "task_type": row.get("task_type", ""),
            "saved_response": row.get("kg_response", ""),
            "saved_correct": bool(row.get("kg_correct")),
            "saved_sd_uq": float(sd),
            "saved_seu": float(seu),
            "saved_route": route,
            "saved_route_reason": row.get("kg_route_reason", ""),
        })
        if len(candidates) >= limit:
            break
    return candidates


def _make_kg_system(policy: str) -> EnhancedRAGSystem:
    if policy == "strict":
        return EnhancedRAGSystem(
            retrieval_mode="entity_first",
            use_rfge=False,
            use_per_entity_ann=False,
            allow_vector_augmentation=False,
            allow_vector_fallback=False,
            embedding_model=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
        )
    return EnhancedRAGSystem(embedding_model=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"))


def _question_id_for_scope(dataset: str, qid: str) -> Optional[str]:
    return qid if dataset in MIRAGEEvaluationPipeline.QUESTION_SCOPED_DATASETS else None


def _generate_once(
    *,
    pipeline: MIRAGEEvaluationPipeline,
    kg_system: EnhancedRAGSystem,
    row: Dict[str, Any],
    dataset: str,
    max_hops: int,
    max_chunks: int,
    allow_decomposition: bool,
    anchor_mask_entity_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    answer_instructions = build_answer_instructions(
        dataset,
        row.get("task_type") or "",
        options=None,
    )
    result = kg_system.generate_response(
        question=row["question"],
        llm=pipeline.accuracy_llm,
        similarity_threshold=0.1,
        max_chunks=max_chunks,
        kg_name=dataset,
        max_hops=max_hops,
        answer_instructions=answer_instructions,
        question_id=_question_id_for_scope(dataset, row["question_id"]),
        allow_decomposition=allow_decomposition,
        retrieval_temperature=0.0,
        retrieval_shortlist_factor=4,
        retrieval_sample_id=0,
        anchor_mask_entity_ids=anchor_mask_entity_ids,
    )
    raw_response = str(result.get("response", ""))
    response = normalize_answer_to_contract(
        dataset,
        row.get("task_type") or "",
        raw_response,
        question=row["question"],
    )
    correct = pipeline._is_answer_correct(
        row.get("expected", ""),
        response,
        question=row["question"],
        task_type=row.get("task_type") or "",
    )
    context = result.get("context", {}) or {}
    state = context.get("graph_state") or summarize_context_graph_state(context)
    return {
        "response": response,
        "raw_response": raw_response,
        "correct": bool(correct),
        "graph_state": state,
        "route": context.get("retrieval_route", ""),
        "route_reason": context.get("route_reason", ""),
        "search_method": context.get("search_method", ""),
    }


def _sample_diversity(
    *,
    pipeline: MIRAGEEvaluationPipeline,
    kg_system: EnhancedRAGSystem,
    row: Dict[str, Any],
    dataset: str,
    max_hops: int,
    max_chunks: int,
    allow_decomposition: bool,
    anchor_mask_entity_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    answer_instructions = build_answer_instructions(
        dataset,
        row.get("task_type") or "",
        options=None,
    )
    samples, _, chunk_jaccard, graph_states = pipeline._collect_sample_responses(
        rag_system=kg_system,
        question=row["question"],
        answer_instructions=answer_instructions,
        similarity_threshold=0.1,
        max_chunks=max_chunks,
        kg_name=dataset,
        question_id=_question_id_for_scope(dataset, row["question_id"]),
        retrieval_temperature=0.0,
        retrieval_shortlist_factor=4,
        generate_kwargs={
            "allow_decomposition": allow_decomposition,
            "anchor_mask_entity_ids": anchor_mask_entity_ids,
        },
        return_graph_state_traces=True,
    )
    uq = compute_all_uncertainty_metrics(
        responses=samples,
        prompt=row["question"],
        context="",
    )
    diversity = graph_state_diversity(graph_states)
    diversity["chunk_text_jaccard"] = float(chunk_jaccard or 0.0)
    diversity["discrete_semantic_entropy"] = float(uq.get("discrete_semantic_entropy", 0.0))
    diversity["sd_uq"] = float(uq.get("sd_uq", 0.0))
    diversity["responses"] = samples
    return diversity


def run_intervention(args, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    pipeline = MIRAGEEvaluationPipeline(
        num_samples=None,
        entropy_samples=args.entropy_samples,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        use_llm_judge=True,
        evaluation_mode=MIRAGEEvaluationPipeline.EVALUATION_MODE_FULL_METRICS,
        rebuild_kg=False,
    )
    kg_system = _make_kg_system(args.policy)
    max_hops = MIRAGEEvaluationPipeline.DATASET_MAX_HOPS.get(
        args.dataset,
        MIRAGEEvaluationPipeline.DEFAULT_MAX_HOPS,
    )
    allow_decomposition = args.policy != "strict"
    results = []
    for row in candidates:
        baseline = _generate_once(
            pipeline=pipeline,
            kg_system=kg_system,
            row=row,
            dataset=args.dataset,
            max_hops=max_hops,
            max_chunks=args.max_chunks,
            allow_decomposition=allow_decomposition,
        )
        baseline_diversity = _sample_diversity(
            pipeline=pipeline,
            kg_system=kg_system,
            row=row,
            dataset=args.dataset,
            max_hops=max_hops,
            max_chunks=args.max_chunks,
            allow_decomposition=allow_decomposition,
        )
        anchor = dominant_anchor_id([baseline["graph_state"]])
        if not anchor:
            anchor = str(baseline_diversity.get("dominant_seed_entity_id") or "")
        masked = None
        masked_diversity = None
        if anchor:
            masked = _generate_once(
                pipeline=pipeline,
                kg_system=kg_system,
                row=row,
                dataset=args.dataset,
                max_hops=max_hops,
                max_chunks=args.max_chunks,
                allow_decomposition=allow_decomposition,
                anchor_mask_entity_ids=[anchor],
            )
            masked_diversity = _sample_diversity(
                pipeline=pipeline,
                kg_system=kg_system,
                row=row,
                dataset=args.dataset,
                max_hops=max_hops,
                max_chunks=args.max_chunks,
                allow_decomposition=allow_decomposition,
                anchor_mask_entity_ids=[anchor],
            )
        results.append({
            **row,
            "intervention_anchor_entity_id": anchor,
            "baseline_rerun": baseline,
            "baseline_diversity": baseline_diversity,
            "masked_rerun": masked,
            "masked_diversity": masked_diversity,
            "answer_changed": bool(masked and masked["response"] != baseline["response"]),
            "state_changed": bool(
                masked
                and masked["graph_state"].get("subgraph_signature")
                != baseline["graph_state"].get("subgraph_signature")
            ),
            "correctness_improved": bool(masked and (not baseline["correct"]) and masked["correct"]),
        })
    return {"results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--policy", choices=["kg", "strict"], default="strict")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--sd-threshold", type=float, default=1e-9)
    parser.add_argument("--seu-threshold", type=float, default=0.5)
    parser.add_argument("--include-correct", action="store_true")
    parser.add_argument(
        "--require-populated-route",
        action="store_true",
        help="Select only rows whose saved KG route was populated; useful for presence lock-in.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--entropy-samples", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--llm-provider", default="openrouter")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--out",
        default=os.path.join(REPO, "results", "analyses", "anchor_masking_intervention.json"),
    )
    args = parser.parse_args()

    candidates = select_candidates(
        args.dataset,
        args.policy,
        sd_threshold=args.sd_threshold,
        seu_threshold=args.seu_threshold,
        wrong_only=not args.include_correct,
        require_populated_route=args.require_populated_route,
        limit=args.limit,
    )
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "run" if args.run else "dry_run",
        "note": "No KG rebuild. Dry-run mode reads saved rows only; --run reruns retrieval/generation on existing KG.",
        "dataset": args.dataset,
        "policy": args.policy,
        "trigger": {
            "sd_uq_lte": args.sd_threshold,
            "seu_gte": args.seu_threshold,
            "wrong_only": not args.include_correct,
            "require_populated_route": args.require_populated_route,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    if args.run and candidates:
        out.update(run_intervention(args, candidates))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
