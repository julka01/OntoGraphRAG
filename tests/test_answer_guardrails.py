import pytest

pytest.importorskip("langchain_core", reason="langchain_core not installed")

from langchain_core.runnables import RunnableLambda

from ontographrag.rag.answer_guardrails import (
    evaluate_runtime_answer_guardrail,
    parse_guardrail_verdict,
    render_guardrail_context,
)


def test_parse_guardrail_verdict_from_json():
    verdict = parse_guardrail_verdict(
        '{"decision":"retry","answers_question":false,"supported_by_context":true,"reason":"off target"}'
    )
    assert verdict["decision"] == "retry"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is True
    assert verdict["reason"] == "off target"


def test_parse_guardrail_verdict_from_embedded_json():
    verdict = parse_guardrail_verdict(
        'verdict: {"decision":"abstain","answers_question":false,"supported_by_context":false,"reason":"insufficient evidence"}'
    )
    assert verdict["decision"] == "abstain"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is False


def test_render_guardrail_context_limits_chunks():
    chunks = [
        {"document": "a.txt", "text": "first chunk"},
        {"document": "b.txt", "text": "second chunk"},
        {"document": "c.txt", "text": "third chunk"},
    ]
    rendered = render_guardrail_context(chunks, max_chunks=2)
    assert "[1] a.txt: first chunk" in rendered
    assert "[2] b.txt: second chunk" in rendered
    assert "third chunk" not in rendered


def test_parse_guardrail_verdict_ambiguous_defaults_to_abstain():
    # When LLM output contains neither "keep", "retry", nor "abstain", the
    # conservative default must be abstain (not keep).
    verdict = parse_guardrail_verdict("I'm not sure about this one.")
    assert verdict["decision"] == "abstain"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is False


def test_parse_guardrail_verdict_keep_keyword_detected():
    # Prose containing "keep" should resolve to keep when no JSON is present.
    verdict = parse_guardrail_verdict("You should keep this answer.")
    assert verdict["decision"] == "keep"


def test_evaluate_runtime_answer_guardrail_uses_llm_json():
    llm = RunnableLambda(
        lambda _: '{"decision":"retry","answers_question":false,"supported_by_context":true,"reason":"off target"}'
    )
    verdict = evaluate_runtime_answer_guardrail(
        question="Who directed Lost and Delirious?",
        answer="Léa Pool",
        chunks=[{"document": "2wikimultihopqa", "text": "Lost and Delirious is a 2001 Canadian drama film directed by Léa Pool."}],
        llm=llm,
    )
    assert verdict["decision"] == "retry"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is True
    assert verdict["reason"] == "off target"
