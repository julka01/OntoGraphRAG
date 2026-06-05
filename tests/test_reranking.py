import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.rag import reranking as rerank_mod


def test_resolve_reranker_model_name_uses_accuracy_profile_default(monkeypatch):
    monkeypatch.delenv("ONTOGRAPHRAG_RERANKER_MODEL", raising=False)
    monkeypatch.setenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "accuracy")

    assert rerank_mod.resolve_reranker_model_name() == "BAAI/bge-reranker-base"


def test_rerank_chunks_for_query_sorts_by_model_score(monkeypatch):
    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return [0.1, 0.9]

    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER_MODEL", "fake-reranker")
    monkeypatch.setattr(rerank_mod, "_load_cross_encoder", lambda model_name: FakeModel())

    chunks = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.rerank_chunks_for_query("Question?", chunks)

    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[1]["chunk_id"] == "c1"
    assert meta["applied"] is True
    assert meta["model"] == "fake-reranker"


def test_rerank_chunks_for_query_fails_soft_when_model_unavailable(monkeypatch):
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER_MODEL", "fake-reranker")
    monkeypatch.setattr(rerank_mod, "_load_cross_encoder", lambda model_name: None)

    chunks = [
        {"text": "First.", "chunk_id": "c1", "score": 0.9},
        {"text": "Second.", "chunk_id": "c2", "score": 0.8},
    ]

    reranked, meta = rerank_mod.rerank_chunks_for_query("Question?", chunks)

    assert [chunk["chunk_id"] for chunk in reranked] == ["c1", "c2"]
    assert meta["applied"] is False
    assert meta["reason"] == "model_unavailable"


def test_resolve_late_interaction_model_name_uses_accuracy_profile_default(monkeypatch):
    monkeypatch.delenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", raising=False)
    monkeypatch.delenv("HUGGINGFACE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "accuracy")

    assert rerank_mod.resolve_late_interaction_model_name() == "BAAI/bge-base-en-v1.5"


def test_late_interaction_rescore_chunks_for_query_sorts_by_model_score(monkeypatch):
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")
    monkeypatch.setattr(rerank_mod, "_load_late_interaction_model", lambda model_name: {"fake": True})
    monkeypatch.setattr(rerank_mod, "_late_interaction_scores", lambda bundle, query, chunks: [0.2, 0.8])

    chunks = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query("Question?", chunks)

    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[1]["chunk_id"] == "c1"
    assert meta["applied"] is True
    assert meta["model"] == "fake-late-interaction"


def test_late_interaction_rescore_chunks_can_replace_score(monkeypatch):
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")
    monkeypatch.setattr(rerank_mod, "_load_late_interaction_model", lambda model_name: {"fake": True})
    monkeypatch.setattr(rerank_mod, "_late_interaction_scores", lambda bundle, query, chunks: [0.2, 0.8])

    chunks = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query(
        "Question?",
        chunks,
        replace_score=True,
    )

    assert reranked[0]["score"] == 0.8
    assert reranked[1]["score"] == 0.2
    assert meta["replace_score"] is True


def test_late_interaction_rescore_chunks_for_query_fails_soft_when_model_unavailable(monkeypatch):
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")
    monkeypatch.setattr(rerank_mod, "_load_late_interaction_model", lambda model_name: None)

    chunks = [
        {"text": "First.", "chunk_id": "c1", "score": 0.9},
        {"text": "Second.", "chunk_id": "c2", "score": 0.8},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query("Question?", chunks)

    assert [chunk["chunk_id"] for chunk in reranked] == ["c1", "c2"]
    assert meta["applied"] is False
    assert meta["reason"] == "model_unavailable"


def test_late_interaction_rescore_chunks_for_query_uses_indexed_backend(monkeypatch):
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")
    monkeypatch.setattr(rerank_mod, "_load_late_interaction_model", lambda model_name: {"fake": True})
    monkeypatch.setattr(
        rerank_mod,
        "_late_interaction_scores_indexed",
        lambda bundle, query, chunks, max_chunks=None, index_key=None, shortlist_size=None: (
            [(1, 0.8), (0, 0.2)],
            {
                "index_cached": True,
                "prefiltered": True,
                "shortlist_size": 2,
                "corpus_size": 12,
            },
        ),
    )

    chunks = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query(
        "Question?",
        chunks,
        max_chunks=1,
        replace_score=True,
        index_key=("scope", "q1"),
    )

    assert [chunk["chunk_id"] for chunk in reranked] == ["c2"]
    assert reranked[0]["score"] == 0.8
    assert meta["index_keyed"] is True
    assert meta["index_cached"] is True
    assert meta["prefiltered"] is True


def test_build_late_interaction_index_reuses_cached_doc_embeddings(monkeypatch):
    import torch

    class FakeTokenizer:
        def __call__(
            self,
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_special_tokens_mask=True,
        ):
            items = [texts] if isinstance(texts, str) else list(texts)
            encoded = []
            for idx, text in enumerate(items, 1):
                token_count = max(2, min(4, len(str(text).split()) + 1))
                encoded.append([idx + offset for offset in range(token_count)])
            width = max(len(row) for row in encoded)
            input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row_idx, row in enumerate(encoded):
                input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
                attention_mask[row_idx, : len(row)] = 1
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "special_tokens_mask": torch.zeros_like(input_ids),
            }

    class FakeModel:
        def __init__(self):
            self.calls = 0
            self.config = SimpleNamespace(hidden_size=2)

        def __call__(self, input_ids, attention_mask):
            self.calls += 1
            base = input_ids.unsqueeze(-1).float()
            hidden = torch.cat([base, base + 1.0], dim=-1)
            return SimpleNamespace(last_hidden_state=hidden)

    fake_model = FakeModel()
    bundle = {
        "tokenizer": FakeTokenizer(),
        "model": fake_model,
        "model_name": "fake-late-interaction",
        "device": "cpu",
        "query_max_length": 16,
        "doc_max_length": 16,
        "torch": torch,
    }
    chunks = [
        {"chunk_id": "c1", "document": "doc", "text": "first evidence"},
        {"chunk_id": "c2", "document": "doc", "text": "second evidence"},
    ]

    rerank_mod._LATE_INTERACTION_INDEX_CACHE.clear()
    index_one, cache_hit_one = rerank_mod._build_late_interaction_index(
        bundle,
        chunks,
        index_key=("scope", "q1"),
    )
    index_two, cache_hit_two = rerank_mod._build_late_interaction_index(
        bundle,
        chunks,
        index_key=("scope", "q1"),
    )

    assert cache_hit_one is False
    assert cache_hit_two is True
    assert fake_model.calls == 1
    assert index_one["count"] == 2
    assert index_two["count"] == 2


def test_late_interaction_scores_indexed_prefilters_large_corpus(monkeypatch):
    import torch

    doc_tokens = [
        torch.tensor([[0.1, 0.0]], dtype=torch.float32),
        torch.tensor([[0.9, 0.0]], dtype=torch.float32),
        torch.tensor([[0.8, 0.0]], dtype=torch.float32),
    ]
    doc_pooled = torch.stack([
        torch.tensor([0.1, 0.0], dtype=torch.float32),
        torch.tensor([0.9, 0.0], dtype=torch.float32),
        torch.tensor([0.8, 0.0], dtype=torch.float32),
    ])
    bundle = {"torch": torch}

    monkeypatch.setattr(
        rerank_mod,
        "_encode_late_interaction_query",
        lambda bundle, query: {
            "tokens": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "pooled": torch.tensor([1.0, 0.0], dtype=torch.float32),
        },
    )
    monkeypatch.setattr(
        rerank_mod,
        "_build_late_interaction_index",
        lambda bundle, chunks, index_key=None: (
            {"doc_tokens": doc_tokens, "doc_pooled": doc_pooled, "count": 3},
            False,
        ),
    )
    monkeypatch.setattr(rerank_mod, "_late_interaction_prefilter_min_docs", lambda: 1)
    monkeypatch.setattr(rerank_mod, "_late_interaction_shortlist_cap", lambda: 2)
    monkeypatch.setattr(rerank_mod, "_late_interaction_default_shortlist_factor", lambda: 1)

    scored, meta = rerank_mod._late_interaction_scores_indexed(
        bundle,
        "Question?",
        [{}, {}, {}],
        max_chunks=1,
        index_key=("scope", "q1"),
        shortlist_size=2,
    )

    assert [idx for idx, _score in scored] == [1, 2]
    assert meta["prefiltered"] is True
    assert meta["shortlist_size"] == 2
    assert meta["corpus_size"] == 3
