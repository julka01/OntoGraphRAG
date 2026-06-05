import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


RUNTIME_GUARDRAIL_ABSTENTION = (
    "I’m not confident the retrieved evidence supports a reliable answer to that question."
)


def render_guardrail_context(chunks: List[Dict[str, Any]], max_chunks: int = 8, max_chars: int = 320) -> str:
    rendered: List[str] = []
    for idx, chunk in enumerate(chunks[:max_chunks], 1):
        text = " ".join(str(chunk.get("text", "")).split())
        text = text[:max_chars]
        document = str(chunk.get("document") or chunk.get("doc_name") or "unknown")
        rendered.append(f"[{idx}] {document}: {text}")
    return "\n".join(rendered)


def parse_guardrail_verdict(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    payload: Dict[str, Any] = {}

    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = {}

    decision = str(payload.get("decision") or "").strip().lower()
    if decision not in {"keep", "retry", "abstain"}:
        lowered = text.lower()
        if "abstain" in lowered:
            decision = "abstain"
        elif "retry" in lowered:
            decision = "retry"
        elif "keep" in lowered:
            decision = "keep"
        else:
            # Ambiguous LLM response — default to abstain rather than keep to
            # avoid surfacing unsupported answers.
            decision = "abstain"

    def _boolish(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return default

    answers_question = _boolish(payload.get("answers_question"), decision == "keep")
    supported_by_context = _boolish(payload.get("supported_by_context"), decision == "keep")
    reason = str(payload.get("reason") or text or "").strip()

    return {
        "decision": decision,
        "answers_question": answers_question,
        "supported_by_context": supported_by_context,
        "reason": reason[:500],
        "raw": text,
    }


def evaluate_runtime_answer_guardrail(
    *,
    question: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    llm: Any,
) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict runtime answer verifier for retrieval-augmented question answering.\n"
            "Given a question, a candidate answer, and retrieved evidence chunks, decide whether the answer should be kept,\n"
            "retried in a safer retrieval mode, or replaced with an abstention.\n\n"
            "Decision rules:\n"
            "- keep: the answer directly addresses the question and is supported by the evidence.\n"
            "- retry: the answer is off-target, incomplete, or mismatched, but the evidence appears to contain useful information.\n"
            "- abstain: the evidence is insufficient or the answer is unsupported.\n\n"
            "Return ONLY valid JSON with this schema:\n"
            "{{\"decision\":\"keep|retry|abstain\",\"answers_question\":true|false,"
            "\"supported_by_context\":true|false,\"reason\":\"short reason\"}}"
        )),
        ("human", (
            "Question: {question}\n\n"
            "Candidate answer: {answer}\n\n"
            "Evidence:\n{evidence}"
        )),
    ])

    evidence = render_guardrail_context(chunks)
    try:
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "answer": answer,
            "evidence": evidence or "[no retrieved evidence]",
        })
        return parse_guardrail_verdict(raw)
    except Exception as exc:
        logging.warning("Runtime answer guardrail failed: %s", exc)
        return {
            "decision": "retry",
            "answers_question": False,
            "supported_by_context": False,
            "reason": f"guardrail_error:{exc}",
            "raw": "",
        }
