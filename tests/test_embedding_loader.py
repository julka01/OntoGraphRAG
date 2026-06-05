import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.kg.utils import common_functions as common_mod


def test_openai_loader_uses_current_default_model(monkeypatch):
    captured = {}

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(common_mod, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("OPENAI_EMBEDDING_DIMENSION", raising=False)

    embeddings, dimension = common_mod.load_embedding_model("openai")

    assert isinstance(embeddings, FakeOpenAIEmbeddings)
    assert captured["kwargs"]["model"] == "text-embedding-3-small"
    assert dimension == 1536


def test_openai_loader_accepts_explicit_model_name(monkeypatch):
    captured = {}

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(common_mod, "OpenAIEmbeddings", FakeOpenAIEmbeddings)

    embeddings, dimension = common_mod.load_embedding_model("text-embedding-3-large")

    assert isinstance(embeddings, FakeOpenAIEmbeddings)
    assert captured["kwargs"]["model"] == "text-embedding-3-large"
    assert dimension == 3072


def test_huggingface_loader_accepts_explicit_model_name(monkeypatch):
    captured = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, *, model_name, **kwargs):
            captured["model_name"] = model_name

        def embed_query(self, text):
            return [0.0] * 7

    monkeypatch.setattr(common_mod, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)

    embeddings, dimension = common_mod.load_embedding_model("BAAI/bge-base-en-v1.5")

    assert isinstance(embeddings, FakeHuggingFaceEmbeddings)
    assert captured["model_name"] == "BAAI/bge-base-en-v1.5"
    assert dimension == 7


def test_huggingface_loader_uses_accuracy_profile_default(monkeypatch):
    captured = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, *, model_name, **kwargs):
            captured["model_name"] = model_name

        def embed_query(self, text):
            return [0.0] * 5

    monkeypatch.setattr(common_mod, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)
    monkeypatch.delenv("HUGGINGFACE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "accuracy")

    embeddings, dimension = common_mod.load_embedding_model("sentence_transformers")

    assert isinstance(embeddings, FakeHuggingFaceEmbeddings)
    assert captured["model_name"] == "BAAI/bge-base-en-v1.5"
    assert dimension == 5
