"""
Regression tests for runtime answer guardrail behavior.

Pure unit tests — no live Neo4j or remote LLM calls.
"""

import os
import sys

import pytest

pytest.importorskip("langchain_neo4j", reason="langchain_neo4j not installed — skipping guardrail behavior tests")
pytest.importorskip("langchain", reason="langchain not installed — skipping guardrail behavior tests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.rag.answer_guardrails import RUNTIME_GUARDRAIL_ABSTENTION
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem
from ontographrag.rag.systems.vanilla_rag_system import VanillaRAGSystem


def _base_context() -> dict:
    return {
        "chunks": [{"document": "doc.txt", "text": "supporting evidence"}],
        "documents": ["doc.txt"],
        "total_score": 0.9,
    }


def test_enhanced_retry_then_abstain_sets_final_decision_to_abstain(monkeypatch):
    system = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    verdicts = [
        {
            "decision": "retry",
            "answers_question": False,
            "supported_by_context": True,
            "reason": "off target",
        },
        {
            "decision": "retry",
            "answers_question": False,
            "supported_by_context": True,
            "reason": "still off target",
        },
    ]

    monkeypatch.setattr(
        "ontographrag.rag.systems.enhanced_rag_system.evaluate_runtime_answer_guardrail",
        lambda **_: verdicts.pop(0),
    )

    response, _, metadata = system._apply_runtime_answer_guardrail(
        question="Who is the father-in-law of Helena Palaiologina?",
        llm=object(),
        response="Lazar Brankovic",
        context=_base_context(),
        runtime_guardrail=True,
        runtime_guardrail_mode="retry_then_abstain",
        retry_factory=lambda: ("Still wrong", _base_context()),
    )

    assert response == RUNTIME_GUARDRAIL_ABSTENTION
    assert metadata["retried"] is True
    assert metadata["final_decision"] == "abstain"
    assert metadata["retry_verdict"]["decision"] == "retry"


def test_vanilla_abstention_zeroes_confidence(monkeypatch):
    system = VanillaRAGSystem.__new__(VanillaRAGSystem)
    context = _base_context()

    monkeypatch.setattr(system, "get_vanilla_rag_context", lambda *_, **__: context)
    monkeypatch.setattr(system, "_invoke_answer_chain", lambda **_: "Wrong answer")
    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        lambda **_: (
            RUNTIME_GUARDRAIL_ABSTENTION,
            {"enabled": True, "final_decision": "abstain"},
        ),
    )

    result = system.generate_response(
        "Question?",
        llm=object(),
        runtime_guardrail=True,
    )

    assert result["response"] == RUNTIME_GUARDRAIL_ABSTENTION
    assert result["confidence"] == 0.0


def test_enhanced_abstention_zeroes_confidence(monkeypatch):
    system = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    context = {
        **_base_context(),
        "entities": {},
        "relationships": [],
        "entity_count": 1,
        "relationship_count": 0,
        "search_method": "vector_similarity",
    }

    monkeypatch.setattr(system, "get_rag_context", lambda *_, **__: context)
    monkeypatch.setattr(system, "_invoke_answer_chain", lambda **_: "Wrong answer")
    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        lambda **kwargs: (
            RUNTIME_GUARDRAIL_ABSTENTION,
            kwargs["context"],
            {"enabled": True, "final_decision": "abstain"},
        ),
    )
    monkeypatch.setattr(
        system,
        "_extract_used_entities_and_chunks",
        lambda response, context: {
            "used_entities": [],
            "used_chunks": [],
            "reasoning_edges": [],
        },
    )

    result = system.generate_response(
        "Question?",
        llm=object(),
        runtime_guardrail=True,
        max_hops=1,
    )

    assert result["response"] == RUNTIME_GUARDRAIL_ABSTENTION
    assert result["confidence"] == 0.0


def test_enhanced_retrieval_span_context_does_not_retry_vector_only(monkeypatch):
    system = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    context = {
        **_base_context(),
        "entities": {},
        "relationships": [],
        "entity_count": 0,
        "relationship_count": 0,
        "search_method": "retrieval_span_similarity",
    }

    monkeypatch.setattr(system, "get_rag_context", lambda *_, **__: context)
    monkeypatch.setattr(system, "_invoke_answer_chain", lambda **_: "Insufficient Information")
    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        lambda **kwargs: (
            kwargs["response"],
            kwargs["context"],
            {"enabled": False, "final_decision": "keep"},
        ),
    )
    monkeypatch.setattr(
        system,
        "_vector_similarity_search",
        lambda *_, **__: pytest.fail("pure vector context should not retry vector-only"),
    )
    monkeypatch.setattr(
        system,
        "_extract_used_entities_and_chunks",
        lambda response, context: {
            "used_entities": [],
            "used_chunks": [],
            "reasoning_edges": [],
        },
    )

    result = system.generate_response(
        "Question?",
        llm=object(),
        max_hops=1,
    )

    assert result["response"] == "Insufficient Information"
