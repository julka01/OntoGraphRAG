"""Utilities for logging symbolic retrieval-state diversity.

The functions here are deliberately dependency-light so they can be used both
inside the live RAG system and by offline experiment scripts.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


def _clean_id(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _set_from(values: Iterable[Any]) -> Set[str]:
    return {text for text in (_clean_id(v) for v in values) if text}


def _signature(values: Iterable[Any]) -> str:
    items = sorted(_set_from(values))
    return "||".join(items) if items else "<empty>"


def _relationship_id(rel: Dict[str, Any]) -> str:
    if not isinstance(rel, dict):
        return ""
    return (
        _clean_id(rel.get("key"))
        or _clean_id(rel.get("element_id"))
        or "::".join(
            part
            for part in (
                _clean_id(rel.get("source")),
                _clean_id(rel.get("type")),
                _clean_id(rel.get("target")),
                f"neg={bool(rel.get('negated', False))}",
            )
            if part
        )
    )


def _path_id(path_entry: Dict[str, Any]) -> str:
    if not isinstance(path_entry, dict):
        return ""
    node_ids = path_entry.get("node_ids") or []
    if node_ids:
        return "->".join(_clean_id(node_id) for node_id in node_ids if _clean_id(node_id))
    return _clean_id(path_entry.get("path"))


def _chunk_id(chunk: Dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""
    return (
        _clean_id(chunk.get("chunk_id"))
        or _clean_id(chunk.get("chunk_element_id"))
        or _clean_id(chunk.get("text"))[:80]
    )


def _entity_ids_from_chunks(chunks: Sequence[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ids.extend(_clean_id(eid) for eid in (chunk.get("linked_entity_ids") or []))
        for entity in chunk.get("entities") or []:
            if isinstance(entity, dict):
                ids.append(_clean_id(entity.get("id")))
    return [eid for eid in ids if eid]


def summarize_context_graph_state(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a compact symbolic state summary for one retrieved context."""
    context = context or {}
    diagnostics = context.get("diagnostics") or {}
    chunks = context.get("chunks") or []
    entities = context.get("entities") or {}
    relationships = context.get("relationships") or []
    traversal_paths = context.get("traversal_paths") or []

    seed_entity_ids = [
        _clean_id(eid)
        for eid in (diagnostics.get("seed_entity_ids") or [])
        if _clean_id(eid)
    ]
    if not seed_entity_ids:
        seed_entity_ids = [
            _clean_id(eid)
            for eid, info in entities.items()
            if isinstance(info, dict) and int(info.get("min_hops", 99) or 99) == 0
        ]

    seed_entity_names = [
        _clean_id(name)
        for name in (
            diagnostics.get("seed_entity_names")
            or diagnostics.get("seed_entities")
            or []
        )
        if _clean_id(name)
    ]
    entity_ids = list(entities.keys()) or _entity_ids_from_chunks(chunks)
    relationship_ids = [_relationship_id(rel) for rel in relationships]
    path_ids = [_path_id(path) for path in traversal_paths]
    chunk_ids = [_chunk_id(chunk) for chunk in chunks]

    subgraph_items = list(entity_ids) + relationship_ids + path_ids
    return {
        "route": _clean_id(context.get("retrieval_route")),
        "search_method": _clean_id(context.get("search_method")),
        "seed_entity_ids": sorted(_set_from(seed_entity_ids)),
        "seed_entity_names": sorted(_set_from(seed_entity_names)),
        "entity_ids": sorted(_set_from(entity_ids)),
        "relationship_ids": sorted(_set_from(relationship_ids)),
        "path_ids": sorted(_set_from(path_ids)),
        "chunk_ids": sorted(_set_from(chunk_ids)),
        "seed_signature": _signature(seed_entity_ids),
        "path_signature": _signature(path_ids),
        "subgraph_signature": _signature(subgraph_items),
        "chunk_signature": _signature(chunk_ids),
        "entity_count": len(_set_from(entity_ids)),
        "relationship_count": len(_set_from(relationship_ids)),
        "path_count": len(_set_from(path_ids)),
        "chunk_count": len(_set_from(chunk_ids)),
    }


def shannon_entropy(labels: Sequence[str]) -> float:
    labels = [str(label) for label in labels]
    if not labels:
        return 0.0
    n = len(labels)
    counts = Counter(labels)
    value = float(-sum((c / n) * math.log(c / n, 2) for c in counts.values()))
    return 0.0 if abs(value) < 1e-12 else value


def normalized_entropy(labels: Sequence[str]) -> float:
    labels = [str(label) for label in labels]
    if len(labels) <= 1:
        return 0.0
    denom = math.log(len(labels), 2)
    return float(shannon_entropy(labels) / denom) if denom > 0 else 0.0


def mean_pairwise_jaccard(sets: Sequence[Iterable[Any]]) -> float:
    clean_sets = [_set_from(values) for values in sets]
    pairs = list(combinations(clean_sets, 2))
    if not pairs:
        return 1.0 if clean_sets else 0.0
    vals = []
    for left, right in pairs:
        union = left | right
        vals.append((len(left & right) / len(union)) if union else 1.0)
    return float(sum(vals) / len(vals)) if vals else 0.0


def graph_state_diversity(sample_states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute entropy/Jaccard diversity across per-sample graph states."""
    states = [state or {} for state in sample_states]
    if not states:
        return {
            "sample_count": 0,
            "seed_entity_entropy": 0.0,
            "path_entropy": 0.0,
            "subgraph_entropy": 0.0,
            "chunk_entropy": 0.0,
            "route_entropy": 0.0,
        }

    def labels(key: str) -> List[str]:
        return [str(state.get(key) or "<empty>") for state in states]

    seed_sets = [state.get("seed_entity_ids") or [] for state in states]
    entity_sets = [state.get("entity_ids") or [] for state in states]
    path_sets = [state.get("path_ids") or [] for state in states]
    subgraph_sets = [
        list(state.get("entity_ids") or [])
        + list(state.get("relationship_ids") or [])
        + list(state.get("path_ids") or [])
        for state in states
    ]
    chunk_sets = [state.get("chunk_ids") or [] for state in states]
    seed_counter = Counter(eid for state in states for eid in (state.get("seed_entity_ids") or []))
    dominant_seed_id, dominant_seed_count = ("", 0)
    if seed_counter:
        dominant_seed_id, dominant_seed_count = seed_counter.most_common(1)[0]

    return {
        "sample_count": len(states),
        "empty_state_count": sum(1 for state in states if not state.get("chunk_ids")),
        "seed_entity_entropy": shannon_entropy(labels("seed_signature")),
        "seed_entity_entropy_norm": normalized_entropy(labels("seed_signature")),
        "path_entropy": shannon_entropy(labels("path_signature")),
        "path_entropy_norm": normalized_entropy(labels("path_signature")),
        "subgraph_entropy": shannon_entropy(labels("subgraph_signature")),
        "subgraph_entropy_norm": normalized_entropy(labels("subgraph_signature")),
        "chunk_entropy": shannon_entropy(labels("chunk_signature")),
        "chunk_entropy_norm": normalized_entropy(labels("chunk_signature")),
        "route_entropy": shannon_entropy(labels("route")),
        "route_entropy_norm": normalized_entropy(labels("route")),
        "seed_entity_jaccard": mean_pairwise_jaccard(seed_sets),
        "entity_jaccard": mean_pairwise_jaccard(entity_sets),
        "path_jaccard": mean_pairwise_jaccard(path_sets),
        "subgraph_jaccard": mean_pairwise_jaccard(subgraph_sets),
        "chunk_jaccard": mean_pairwise_jaccard(chunk_sets),
        "dominant_seed_entity_id": dominant_seed_id,
        "dominant_seed_entity_fraction": (
            float(dominant_seed_count / len(states)) if states else 0.0
        ),
    }


def dominant_anchor_id(sample_states: Sequence[Dict[str, Any]]) -> str:
    """Return the most frequent seed entity id across sample states."""
    diversity = graph_state_diversity(sample_states)
    return str(diversity.get("dominant_seed_entity_id") or "")
