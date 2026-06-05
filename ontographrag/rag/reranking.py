import logging
import os
import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple


_RERANKER_CACHE: Dict[Tuple[str, str, int], Any] = {}
_RERANKER_FAILURES: set = set()
_LATE_INTERACTION_CACHE: Dict[Tuple[str, str, int, int], Any] = {}
_LATE_INTERACTION_FAILURES: set = set()
_LATE_INTERACTION_INDEX_CACHE: "OrderedDict[Tuple[str, str, int, str, str], Dict[str, Any]]" = OrderedDict()


def reranker_enabled() -> bool:
    explicit = os.getenv("ONTOGRAPHRAG_RERANKER", "").strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    profile = str(os.getenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "balanced")).strip().lower()
    return profile in {"accuracy", "quality", "high_accuracy"}


def resolve_reranker_model_name() -> str:
    explicit = os.getenv("ONTOGRAPHRAG_RERANKER_MODEL", "").strip()
    if explicit:
        return explicit
    profile = str(os.getenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "balanced")).strip().lower()
    if profile in {"accuracy", "quality", "high_accuracy"}:
        return "BAAI/bge-reranker-base"
    return ""


def late_interaction_enabled() -> bool:
    explicit = os.getenv("ONTOGRAPHRAG_LATE_INTERACTION", "").strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    profile = str(os.getenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "balanced")).strip().lower()
    return profile in {"accuracy", "quality", "high_accuracy"}


def resolve_late_interaction_model_name() -> str:
    explicit = os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "").strip()
    if explicit:
        return explicit
    embedding_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "").strip()
    if embedding_model:
        return embedding_model
    profile = str(os.getenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "balanced")).strip().lower()
    if profile in {"accuracy", "quality", "high_accuracy"}:
        return "BAAI/bge-base-en-v1.5"
    return ""


def _load_cross_encoder(model_name: str):
    if not model_name:
        return None

    device = os.getenv("ONTOGRAPHRAG_RERANKER_DEVICE", "cpu").strip() or "cpu"
    max_length = int(os.getenv("ONTOGRAPHRAG_RERANKER_MAX_LENGTH", "512").strip() or 512)
    cache_key = (model_name, device, max_length)
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]
    if cache_key in _RERANKER_FAILURES:
        return None

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device=device, max_length=max_length)
    except Exception as exc:
        logging.warning("Could not initialize reranker model '%s': %s", model_name, exc)
        _RERANKER_FAILURES.add(cache_key)
        return None

    _RERANKER_CACHE[cache_key] = model
    return model


def _load_late_interaction_model(model_name: str):
    if not model_name:
        return None

    device = os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_DEVICE", "cpu").strip() or "cpu"
    query_max_length = int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_QUERY_MAX_LENGTH", "48").strip() or 48)
    doc_max_length = int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_DOC_MAX_LENGTH", "256").strip() or 256)
    cache_key = (model_name, device, query_max_length, doc_max_length)
    if cache_key in _LATE_INTERACTION_CACHE:
        return _LATE_INTERACTION_CACHE[cache_key]
    if cache_key in _LATE_INTERACTION_FAILURES:
        return None

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        bundle = {
            "tokenizer": tokenizer,
            "model": model,
            "model_name": model_name,
            "device": device,
            "query_max_length": query_max_length,
            "doc_max_length": doc_max_length,
            "torch": torch,
        }
    except Exception as exc:
        logging.warning("Could not initialize late-interaction model '%s': %s", model_name, exc)
        _LATE_INTERACTION_FAILURES.add(cache_key)
        return None

    _LATE_INTERACTION_CACHE[cache_key] = bundle
    return bundle


def _late_interaction_scores(
    bundle: Dict[str, Any],
    query: str,
    chunks: Sequence[Dict[str, Any]],
) -> List[float]:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    torch = bundle["torch"]
    query_max_length = int(bundle["query_max_length"])
    doc_max_length = int(bundle["doc_max_length"])
    batch_size = int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_BATCH_SIZE", "8").strip() or 8)

    query_inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=query_max_length,
        return_special_tokens_mask=True,
    )
    query_inputs = {key: value.to(device) for key, value in query_inputs.items()}

    with torch.no_grad():
        query_outputs = model(
            input_ids=query_inputs["input_ids"],
            attention_mask=query_inputs["attention_mask"],
        )
    query_hidden = query_outputs.last_hidden_state[0]
    query_special = query_inputs.get("special_tokens_mask")
    if query_special is None:
        query_special = torch.zeros_like(query_inputs["attention_mask"])
    query_mask = (query_inputs["attention_mask"][0] > 0) & (query_special[0] == 0)
    if int(query_mask.sum().item()) == 0:
        query_mask = query_inputs["attention_mask"][0] > 0
    query_tokens = torch.nn.functional.normalize(query_hidden[query_mask], p=2, dim=-1)

    scores: List[float] = []
    texts: List[str] = []
    for chunk in chunks:
        doc = str(chunk.get("document", "")).strip()
        text = str(chunk.get("text", "")).strip()
        texts.append(f"[Document: {doc}]\n{text}" if doc else text)

    for start in range(0, len(texts), batch_size):
        doc_inputs = tokenizer(
            texts[start:start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=doc_max_length,
            return_special_tokens_mask=True,
        )
        doc_inputs = {key: value.to(device) for key, value in doc_inputs.items()}
        with torch.no_grad():
            doc_outputs = model(
                input_ids=doc_inputs["input_ids"],
                attention_mask=doc_inputs["attention_mask"],
            )
        doc_hidden = doc_outputs.last_hidden_state
        doc_special = doc_inputs.get("special_tokens_mask")
        if doc_special is None:
            doc_special = torch.zeros_like(doc_inputs["attention_mask"])
        doc_masks = (doc_inputs["attention_mask"] > 0) & (doc_special == 0)

        for row_idx in range(doc_hidden.shape[0]):
            doc_mask = doc_masks[row_idx]
            if int(doc_mask.sum().item()) == 0:
                doc_mask = doc_inputs["attention_mask"][row_idx] > 0
            doc_tokens = torch.nn.functional.normalize(doc_hidden[row_idx][doc_mask], p=2, dim=-1)
            similarity = torch.matmul(query_tokens, doc_tokens.transpose(0, 1))
            score = float(similarity.max(dim=1).values.mean().item())
            scores.append(score)

    return scores


def _late_interaction_chunk_identity(chunk: Dict[str, Any], idx: int) -> str:
    for key in (
        "chunk_element_id",
        "parent_chunk_element_id",
        "chunk_id",
        "parent_chunk_id",
    ):
        value = chunk.get(key)
        if value:
            return str(value)
    text = str(chunk.get("text", "")).strip()
    return f"row-{idx}:{text[:96]}"


def _late_interaction_chunk_text(chunk: Dict[str, Any]) -> str:
    doc = str(chunk.get("document", "")).strip()
    text = str(chunk.get("text", "")).strip()
    return f"[Document: {doc}]\n{text}" if doc else text


def _late_interaction_index_cache_limit() -> int:
    return max(1, int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_INDEX_CACHE_SIZE", "12").strip() or 12))


def _late_interaction_prefilter_min_docs() -> int:
    return max(16, int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_PREFILTER_MIN_DOCS", "96").strip() or 96))


def _late_interaction_shortlist_cap() -> int:
    return max(16, int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_SHORTLIST_CAP", "384").strip() or 384))


def _late_interaction_default_shortlist_factor() -> int:
    return max(2, int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_SHORTLIST_FACTOR", "8").strip() or 8))


def _late_interaction_index_signature(chunks: Sequence[Dict[str, Any]]) -> str:
    hasher = hashlib.sha1()
    for idx, chunk in enumerate(chunks):
        chunk_id = _late_interaction_chunk_identity(chunk, idx)
        text = _late_interaction_chunk_text(chunk)
        hasher.update(str(idx).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(chunk_id.encode("utf-8", errors="ignore"))
        hasher.update(b"|")
        hasher.update(str(len(text)).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(text[:128].encode("utf-8", errors="ignore"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _late_interaction_index_cache_key(
    bundle: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    index_key: Any,
) -> Tuple[str, str, int, str, str]:
    scope_key = repr(index_key) if index_key is not None else ""
    return (
        str(bundle.get("model_name", "")),
        str(bundle.get("device", "cpu")),
        int(bundle.get("doc_max_length", 256)),
        scope_key,
        _late_interaction_index_signature(chunks),
    )


def _trim_late_interaction_index_cache() -> None:
    while len(_LATE_INTERACTION_INDEX_CACHE) > _late_interaction_index_cache_limit():
        _LATE_INTERACTION_INDEX_CACHE.popitem(last=False)


def _encode_late_interaction_query(bundle: Dict[str, Any], query: str) -> Dict[str, Any]:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    torch = bundle["torch"]
    query_max_length = int(bundle["query_max_length"])

    query_inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=query_max_length,
        return_special_tokens_mask=True,
    )
    query_inputs = {key: value.to(device) for key, value in query_inputs.items()}

    with torch.no_grad():
        query_outputs = model(
            input_ids=query_inputs["input_ids"],
            attention_mask=query_inputs["attention_mask"],
        )

    query_hidden = query_outputs.last_hidden_state[0]
    query_special = query_inputs.get("special_tokens_mask")
    if query_special is None:
        query_special = torch.zeros_like(query_inputs["attention_mask"])
    query_mask = (query_inputs["attention_mask"][0] > 0) & (query_special[0] == 0)
    if int(query_mask.sum().item()) == 0:
        query_mask = query_inputs["attention_mask"][0] > 0

    query_tokens = torch.nn.functional.normalize(query_hidden[query_mask], p=2, dim=-1)
    pooled = torch.nn.functional.normalize(query_tokens.mean(dim=0, keepdim=True), p=2, dim=-1)[0]
    return {
        "tokens": query_tokens,
        "pooled": pooled,
    }


def _build_late_interaction_index(
    bundle: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    *,
    index_key: Any = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    chunk_list = list(chunks)
    if not chunk_list:
        return None, False

    cache_key = _late_interaction_index_cache_key(bundle, chunk_list, index_key)
    cached = _LATE_INTERACTION_INDEX_CACHE.get(cache_key)
    if cached is not None:
        _LATE_INTERACTION_INDEX_CACHE.move_to_end(cache_key)
        return cached, True

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    torch = bundle["torch"]
    doc_max_length = int(bundle["doc_max_length"])
    batch_size = int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_BATCH_SIZE", "8").strip() or 8)

    texts = [_late_interaction_chunk_text(chunk) for chunk in chunk_list]
    doc_tokens: List[Any] = []
    pooled_vectors: List[Any] = []

    for start in range(0, len(texts), batch_size):
        doc_inputs = tokenizer(
            texts[start:start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=doc_max_length,
            return_special_tokens_mask=True,
        )
        doc_inputs = {key: value.to(device) for key, value in doc_inputs.items()}
        with torch.no_grad():
            doc_outputs = model(
                input_ids=doc_inputs["input_ids"],
                attention_mask=doc_inputs["attention_mask"],
            )
        doc_hidden = doc_outputs.last_hidden_state
        doc_special = doc_inputs.get("special_tokens_mask")
        if doc_special is None:
            doc_special = torch.zeros_like(doc_inputs["attention_mask"])
        doc_masks = (doc_inputs["attention_mask"] > 0) & (doc_special == 0)

        for row_idx in range(doc_hidden.shape[0]):
            doc_mask = doc_masks[row_idx]
            if int(doc_mask.sum().item()) == 0:
                doc_mask = doc_inputs["attention_mask"][row_idx] > 0
            token_tensor = torch.nn.functional.normalize(doc_hidden[row_idx][doc_mask], p=2, dim=-1)
            token_tensor = token_tensor.detach().cpu()
            pooled = torch.nn.functional.normalize(token_tensor.mean(dim=0, keepdim=True), p=2, dim=-1)[0]
            doc_tokens.append(token_tensor)
            pooled_vectors.append(pooled)

    if pooled_vectors:
        pooled_matrix = torch.stack(pooled_vectors)
    else:
        hidden_size = int(getattr(model.config, "hidden_size", 1) or 1)
        pooled_matrix = torch.empty((0, hidden_size), dtype=torch.float32)

    index = {
        "doc_tokens": doc_tokens,
        "doc_pooled": pooled_matrix,
        "count": len(doc_tokens),
    }
    _LATE_INTERACTION_INDEX_CACHE[cache_key] = index
    _trim_late_interaction_index_cache()
    return index, False


def _resolve_late_interaction_shortlist_size(
    *,
    total_docs: int,
    max_chunks: Optional[int],
    shortlist_size: Optional[int],
) -> int:
    if total_docs <= 0:
        return 0
    if max_chunks is None:
        return total_docs
    if total_docs < _late_interaction_prefilter_min_docs():
        return total_docs

    requested = shortlist_size
    if requested is None:
        requested = max(1, int(max_chunks)) * _late_interaction_default_shortlist_factor()
    requested = max(int(max_chunks), int(requested))
    requested = min(total_docs, requested, _late_interaction_shortlist_cap())
    return max(int(max_chunks), requested)


def _late_interaction_scores_indexed(
    bundle: Dict[str, Any],
    query: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    max_chunks: Optional[int] = None,
    index_key: Any = None,
    shortlist_size: Optional[int] = None,
) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
    torch = bundle["torch"]
    query_repr = _encode_late_interaction_query(bundle, query)
    index, cache_hit = _build_late_interaction_index(bundle, chunks, index_key=index_key)
    if index is None:
        return [], {
            "index_cached": False,
            "prefiltered": False,
            "shortlist_size": 0,
            "corpus_size": 0,
        }

    total_docs = int(index.get("count", 0) or 0)
    shortlist_n = _resolve_late_interaction_shortlist_size(
        total_docs=total_docs,
        max_chunks=max_chunks,
        shortlist_size=shortlist_size,
    )
    shortlisted_indices = list(range(total_docs))
    prefiltered = shortlist_n < total_docs
    if prefiltered and shortlist_n > 0:
        pooled_scores = torch.matmul(index["doc_pooled"], query_repr["pooled"].detach().cpu())
        topk = torch.topk(pooled_scores, k=shortlist_n)
        shortlisted_indices = [int(idx) for idx in topk.indices.tolist()]

    query_tokens = query_repr["tokens"]
    scored: List[Tuple[int, float]] = []
    for idx in shortlisted_indices:
        doc_tokens = index["doc_tokens"][idx].to(query_tokens.device)
        similarity = torch.matmul(query_tokens, doc_tokens.transpose(0, 1))
        score = float(similarity.max(dim=1).values.mean().item())
        scored.append((idx, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored, {
        "index_cached": bool(cache_hit),
        "prefiltered": bool(prefiltered),
        "shortlist_size": int(len(shortlisted_indices)),
        "corpus_size": int(total_docs),
    }


def late_interaction_rescore_chunks_for_query(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    max_chunks: Optional[int] = None,
    replace_score: bool = False,
    index_key: Any = None,
    shortlist_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reorder chunks with a ColBERT-style token late-interaction scorer."""
    chunk_list = [dict(chunk) for chunk in chunks]
    if not chunk_list:
        return chunk_list, {"enabled": False, "applied": False, "reason": "no_chunks"}
    if not late_interaction_enabled():
        return chunk_list, {"enabled": False, "applied": False, "reason": "disabled"}

    model_name = resolve_late_interaction_model_name()
    if not model_name:
        return chunk_list, {"enabled": True, "applied": False, "reason": "no_model"}

    bundle = _load_late_interaction_model(model_name)
    if bundle is None:
        return chunk_list, {
            "enabled": True,
            "applied": False,
            "reason": "model_unavailable",
            "model": model_name,
        }

    try:
        if index_key is not None:
            indexed_scores, indexed_meta = _late_interaction_scores_indexed(
                bundle,
                query,
                chunk_list,
                max_chunks=max_chunks,
                index_key=index_key,
                shortlist_size=shortlist_size,
            )
            if indexed_scores:
                scored_chunks: List[Tuple[float, int, Dict[str, Any]]] = []
                for idx, score in indexed_scores:
                    chunk = dict(chunk_list[idx])
                    chunk["late_interaction_score"] = float(score)
                    if replace_score:
                        chunk["score"] = float(score)
                    scored_chunks.append((float(score), idx, chunk))

                reranked = [
                    chunk
                    for _score, _idx, chunk in sorted(
                        scored_chunks,
                        key=lambda item: (
                            -item[0],
                            -float(item[2].get("score", 0.0)),
                            item[1],
                        ),
                    )
                ]
                if max_chunks is not None:
                    reranked = reranked[: max(1, int(max_chunks))]
                return reranked, {
                    "enabled": True,
                    "applied": True,
                    "model": model_name,
                    "count": len(reranked),
                    "replace_score": bool(replace_score),
                    "index_keyed": True,
                    **indexed_meta,
                }

        scores = _late_interaction_scores(bundle, query, chunk_list)
    except Exception as exc:
        logging.warning("Late-interaction scoring failed for model '%s': %s", model_name, exc)
        return chunk_list, {
            "enabled": True,
            "applied": False,
            "reason": "scoring_failed",
            "model": model_name,
        }

    scored_chunks: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, (chunk, score) in enumerate(zip(chunk_list, scores)):
        chunk["late_interaction_score"] = float(score)
        if replace_score:
            chunk["score"] = float(score)
        scored_chunks.append((float(score), idx, chunk))

    reranked = [
        chunk
        for _score, _idx, chunk in sorted(
            scored_chunks,
            key=lambda item: (
                -item[0],
                -float(item[2].get("score", 0.0)),
                item[1],
            ),
        )
    ]
    if max_chunks is not None:
        reranked = reranked[: max(1, int(max_chunks))]
    return reranked, {
        "enabled": True,
        "applied": True,
        "model": model_name,
        "count": len(reranked),
        "replace_score": bool(replace_score),
        "index_keyed": False,
    }


def rerank_chunks_for_query(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    max_chunks: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reorder chunks for answer generation with an optional learned reranker.

    Returns `(chunks, metadata)`. If the reranker is disabled or unavailable,
    the input order is preserved.
    """
    chunk_list = [dict(chunk) for chunk in chunks]
    if not chunk_list:
        return chunk_list, {"enabled": False, "applied": False, "reason": "no_chunks"}
    if not reranker_enabled():
        return chunk_list, {"enabled": False, "applied": False, "reason": "disabled"}

    model_name = resolve_reranker_model_name()
    if not model_name:
        return chunk_list, {"enabled": True, "applied": False, "reason": "no_model"}

    model = _load_cross_encoder(model_name)
    if model is None:
        return chunk_list, {
            "enabled": True,
            "applied": False,
            "reason": "model_unavailable",
            "model": model_name,
        }

    pair_texts: List[Tuple[str, str]] = []
    for chunk in chunk_list:
        doc = str(chunk.get("document", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if doc:
            pair_texts.append((query, f"[Document: {doc}]\n{text}"))
        else:
            pair_texts.append((query, text))

    try:
        raw_scores = model.predict(pair_texts, show_progress_bar=False)
    except Exception as exc:
        logging.warning("Reranker scoring failed for model '%s': %s", model_name, exc)
        return chunk_list, {
            "enabled": True,
            "applied": False,
            "reason": "scoring_failed",
            "model": model_name,
        }

    try:
        scores = [float(score) for score in raw_scores]
    except Exception:
        scores = [float(score) for score in list(raw_scores)]

    scored_chunks: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, (chunk, score) in enumerate(zip(chunk_list, scores)):
        chunk["reranker_score"] = score
        scored_chunks.append((score, idx, chunk))

    reranked = [
        chunk
        for _score, _idx, chunk in sorted(
            scored_chunks,
            key=lambda item: (
                -item[0],
                -float(item[2].get("score", 0.0)),
                item[1],
            ),
        )
    ]
    if max_chunks is not None:
        reranked = reranked[: max(1, int(max_chunks))]
    return reranked, {
        "enabled": True,
        "applied": True,
        "model": model_name,
        "count": len(reranked),
    }
