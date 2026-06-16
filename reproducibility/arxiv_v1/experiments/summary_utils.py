"""
Lightweight helpers for building MIRAGE evaluation summaries.

No heavy runtime imports (no dotenv, wandb, Neo4j, LangChain) so this module
can be imported freely in unit tests without the full experiment stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def compute_accuracy_breakdown(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute raw and generation-failure-adjusted accuracy from per-question rows.

    Expected row keys:
      - vanilla_correct / kg_correct: bool-ish
      - vanilla_generation_failed / kg_generation_failed: bool-ish

    Returns a dict with raw accuracy over all rows plus clean accuracy on:
      - each system's own answered subset
      - the shared subset where neither system failed
    """
    total = len(details)

    vanilla_failures = sum(
        1 for row in details if bool(row.get("vanilla_generation_failed", False))
    )
    kg_failures = sum(
        1 for row in details if bool(row.get("kg_generation_failed", False))
    )

    vanilla_answered = total - vanilla_failures
    kg_answered = total - kg_failures
    shared_answered = sum(
        1
        for row in details
        if not bool(row.get("vanilla_generation_failed", False))
        and not bool(row.get("kg_generation_failed", False))
    )

    vanilla_correct = sum(1 for row in details if bool(row.get("vanilla_correct", False)))
    kg_correct = sum(1 for row in details if bool(row.get("kg_correct", False)))

    vanilla_correct_clean = sum(
        1
        for row in details
        if not bool(row.get("vanilla_generation_failed", False))
        and bool(row.get("vanilla_correct", False))
    )
    kg_correct_clean = sum(
        1
        for row in details
        if not bool(row.get("kg_generation_failed", False))
        and bool(row.get("kg_correct", False))
    )

    vanilla_correct_shared = sum(
        1
        for row in details
        if not bool(row.get("vanilla_generation_failed", False))
        and not bool(row.get("kg_generation_failed", False))
        and bool(row.get("vanilla_correct", False))
    )
    kg_correct_shared = sum(
        1
        for row in details
        if not bool(row.get("vanilla_generation_failed", False))
        and not bool(row.get("kg_generation_failed", False))
        and bool(row.get("kg_correct", False))
    )

    return {
        "total_questions": total,
        "num_generation_failures_vanilla": vanilla_failures,
        "num_generation_failures_kg": kg_failures,
        "vanilla_answered_questions": vanilla_answered,
        "kg_answered_questions": kg_answered,
        "shared_answered_questions": shared_answered,
        "vanilla_accuracy": (vanilla_correct / total) if total else 0.0,
        "kg_accuracy": (kg_correct / total) if total else 0.0,
        "vanilla_accuracy_excluding_errors": (
            vanilla_correct_clean / vanilla_answered if vanilla_answered else 0.0
        ),
        "kg_accuracy_excluding_errors": (
            kg_correct_clean / kg_answered if kg_answered else 0.0
        ),
        "vanilla_accuracy_shared_clean": (
            vanilla_correct_shared / shared_answered if shared_answered else 0.0
        ),
        "kg_accuracy_shared_clean": (
            kg_correct_shared / shared_answered if shared_answered else 0.0
        ),
    }


def compute_hop_accuracy_breakdown(
    details: List[Dict[str, Any]],
    *,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute hop-stratified summary metrics from per-question rows.

    Expected row keys:
      - hop_count: int-like or missing
      - vanilla_correct / kg_correct
      - vanilla_generation_failed / kg_generation_failed
      - vanilla_answer_em / kg_answer_em
      - vanilla_answer_f1 / kg_answer_f1
      - vanilla_<metric> / kg_<metric> for each metric in ``metric_names``

    Returns a dict keyed by hop bucket label such as ``2-hop`` or ``4-hop``.
    Rows without a valid positive hop count are ignored.
    """
    selected_metric_names = (
        metric_names
        if metric_names is not None
        else [
            "sd_uq",
            "vn_entropy",
            "support_entailment_uncertainty",
        ]
    )

    def _as_hop_count(value: Any) -> Optional[int]:
        try:
            hop = int(value)
        except (TypeError, ValueError):
            return None
        return hop if hop > 0 else None

    def _bucket_label(hop: int) -> str:
        return f"{hop}-hop" if hop <= 4 else "5+-hop"

    def _mean(rows: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
        if not rows:
            return default
        return sum(float(row.get(key, default) or default) for row in rows) / len(rows)

    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for row in details:
        hop = _as_hop_count(row.get("hop_count"))
        if hop is None:
            continue
        by_bucket.setdefault(_bucket_label(hop), []).append(row)

    output: Dict[str, Dict[str, Any]] = {}
    for bucket in sorted(by_bucket.keys(), key=lambda b: (999 if b == "5+-hop" else int(b.split("-", 1)[0]))):
        rows = by_bucket[bucket]
        acc = compute_accuracy_breakdown(rows)
        output[bucket] = {
            "n": len(rows),
            "vanilla_accuracy": acc["vanilla_accuracy"],
            "kg_accuracy": acc["kg_accuracy"],
            "vanilla_accuracy_clean": acc["vanilla_accuracy_excluding_errors"],
            "kg_accuracy_clean": acc["kg_accuracy_excluding_errors"],
            "vanilla_accuracy_shared_clean": acc["vanilla_accuracy_shared_clean"],
            "kg_accuracy_shared_clean": acc["kg_accuracy_shared_clean"],
            "vanilla_answer_em": _mean(rows, "vanilla_answer_em"),
            "kg_answer_em": _mean(rows, "kg_answer_em"),
            "vanilla_answer_f1": _mean(rows, "vanilla_answer_f1"),
            "kg_answer_f1": _mean(rows, "kg_answer_f1"),
            "metrics_by_approach": {
                "vanilla_rag": {
                    metric_name: _mean(rows, f"vanilla_{metric_name}")
                    for metric_name in selected_metric_names
                },
                "kg_rag": {
                    metric_name: _mean(rows, f"kg_{metric_name}")
                    for metric_name in selected_metric_names
                },
            },
        }

    return output


def accumulate_track_accuracy(
    dataset_blocks: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute per-track macro accuracy from a list of dataset result blocks.

    Parameters
    ----------
    dataset_blocks:
        List of dicts, each with keys: ``dataset`` (str), ``track`` (str),
        ``config_results`` (list of cfg_res dicts carrying ``vanilla_accuracy``,
        ``kg_accuracy``, and ``config`` → ``name``).

    Returns
    -------
    Dict mapping track name → list of per-config aggregate dicts.
    Each aggregate dict has: config_name, num_datasets, datasets,
    vanilla_macro_accuracy, kg_macro_accuracy.

    One entry per (track, config_name) so multi-config sweeps produce separate
    rows rather than silently collapsing to the first config per dataset.
    """
    track_accumulators: Dict[Tuple[str, str], Dict] = {}

    for block in dataset_blocks:
        dataset_name = block.get("dataset", "")
        track = block.get("track", "unknown")
        for cfg_res in block.get("config_results", []):
            cfg_name = cfg_res.get("config", {}).get("name", "default")
            acc_key = (track, cfg_name)
            ta = track_accumulators.setdefault(
                acc_key, {"vanilla_acc": [], "kg_acc": [], "datasets": []}
            )
            if dataset_name not in ta["datasets"]:
                ta["datasets"].append(dataset_name)
                if cfg_res.get("vanilla_accuracy") is not None:
                    ta["vanilla_acc"].append(float(cfg_res["vanilla_accuracy"]))
                if cfg_res.get("kg_accuracy") is not None:
                    ta["kg_acc"].append(float(cfg_res["kg_accuracy"]))

    result: Dict[str, List[Dict[str, Any]]] = {}
    for (track, cfg_name), ta in track_accumulators.items():
        n = len(ta["datasets"])
        result.setdefault(track, []).append({
            "config_name": cfg_name,
            "num_datasets": n,
            "datasets": ta["datasets"],
            "vanilla_macro_accuracy": sum(ta["vanilla_acc"]) / n if n else None,
            "kg_macro_accuracy": sum(ta["kg_acc"]) / n if n else None,
        })
    return result


def _system_metric_prefix(system_name: str) -> str:
    if system_name == "vanilla_rag":
        return "vanilla"
    if system_name == "kg_rag":
        return "kg"
    raise ValueError(f"Unsupported system_name={system_name!r}")


def _candidate_rank_tuple(cfg_res: Dict[str, Any], system_name: str) -> Tuple[float, float, float, int, float]:
    prefix = _system_metric_prefix(system_name)
    return (
        float(cfg_res.get(f"{prefix}_accuracy", 0.0) or 0.0),
        float(cfg_res.get(f"{prefix}_answer_f1", 0.0) or 0.0),
        float(cfg_res.get(f"{prefix}_answer_em", 0.0) or 0.0),
        int(cfg_res.get(f"{prefix}_answered_questions", 0) or 0),
        float(cfg_res.get(f"{prefix}_accuracy_raw", 0.0) or 0.0),
    )


def select_best_retrieval_configs(
    dataset_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Pick the strongest config per dataset and overall for vanilla and KG-RAG.

    Ranking priority:
      1. clean accuracy
      2. answer F1
      3. answer EM
      4. answered-question count
      5. raw accuracy
    """
    per_dataset: Dict[str, Dict[str, Any]] = {}
    macro_rows: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "vanilla_rag": {},
        "kg_rag": {},
    }

    for block in dataset_blocks:
        dataset_name = str(block.get("dataset", "unknown"))
        config_results = list(block.get("config_results", []))
        if not config_results:
            continue

        per_dataset[dataset_name] = {}
        for system_name in ("vanilla_rag", "kg_rag"):
            prefix = _system_metric_prefix(system_name)
            best_cfg = max(
                config_results,
                key=lambda cfg: (_candidate_rank_tuple(cfg, system_name), str(cfg.get("config", {}).get("name", ""))),
            )
            per_dataset[dataset_name][system_name] = {
                "config_name": best_cfg.get("config", {}).get("name", "default"),
                "accuracy": float(best_cfg.get(f"{prefix}_accuracy", 0.0) or 0.0),
                "answer_f1": float(best_cfg.get(f"{prefix}_answer_f1", 0.0) or 0.0),
                "answer_em": float(best_cfg.get(f"{prefix}_answer_em", 0.0) or 0.0),
                "answered_questions": int(best_cfg.get(f"{prefix}_answered_questions", 0) or 0),
                "accuracy_raw": float(best_cfg.get(f"{prefix}_accuracy_raw", 0.0) or 0.0),
            }

            for cfg_res in config_results:
                cfg_name = str(cfg_res.get("config", {}).get("name", "default"))
                row = macro_rows[system_name].setdefault(
                    cfg_name,
                    {
                        "datasets": [],
                        "accuracy": [],
                        "answer_f1": [],
                        "answer_em": [],
                        "answered_questions": [],
                        "accuracy_raw": [],
                    },
                )
                row["datasets"].append(dataset_name)
                row["accuracy"].append(float(cfg_res.get(f"{prefix}_accuracy", 0.0) or 0.0))
                row["answer_f1"].append(float(cfg_res.get(f"{prefix}_answer_f1", 0.0) or 0.0))
                row["answer_em"].append(float(cfg_res.get(f"{prefix}_answer_em", 0.0) or 0.0))
                row["answered_questions"].append(int(cfg_res.get(f"{prefix}_answered_questions", 0) or 0))
                row["accuracy_raw"].append(float(cfg_res.get(f"{prefix}_accuracy_raw", 0.0) or 0.0))

    overall: Dict[str, Any] = {}
    for system_name, cfg_map in macro_rows.items():
        candidates: List[Dict[str, Any]] = []
        for cfg_name, row in cfg_map.items():
            n = max(1, len(row["datasets"]))
            candidates.append({
                "config_name": cfg_name,
                "num_datasets": len(row["datasets"]),
                "datasets": list(row["datasets"]),
                "macro_accuracy": sum(row["accuracy"]) / n,
                "macro_answer_f1": sum(row["answer_f1"]) / n,
                "macro_answer_em": sum(row["answer_em"]) / n,
                "macro_answered_questions": sum(row["answered_questions"]) / n,
                "macro_accuracy_raw": sum(row["accuracy_raw"]) / n,
            })
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: (
                float(row["macro_accuracy"]),
                float(row["macro_answer_f1"]),
                float(row["macro_answer_em"]),
                float(row["macro_answered_questions"]),
                float(row["macro_accuracy_raw"]),
                str(row["config_name"]),
            ),
        )
        overall[system_name] = {
            "best_config": best,
            "all_configs": sorted(candidates, key=lambda row: str(row["config_name"])),
        }

    return {
        "per_dataset": per_dataset,
        "overall": overall,
    }
