"""
Dataset Adapters for MIRAGE-style QA benchmarks.

Provides canonical internal schema and per-dataset adapters to homogenize
different header formats across PubMedQA, BioASQ, MedQA, MedMCQA, MMLU,
HotpotQA, 2WikiMultiHopQA, and MuSiQue.

Canonical Schema:
- InferenceRecord: {id, dataset, question, contexts, options, task_type}
- GoldRecord: {id, short_answer, long_answer, aliases}
"""

import os
import json
import csv
import hashlib
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


QUESTION_CONTEXT_ROLE_SOURCE_DOCUMENT = "source_document"
QUESTION_CONTEXT_ROLE_RETRIEVAL_BUNDLE = "retrieval_bundle"
QUESTION_CONTEXT_ROLE_GOLD_EVIDENCE = "gold_evidence"
QUESTION_CONTEXT_ROLE_NONE = "no_context"


DATASET_CORPUS_PROFILES: Dict[str, Dict[str, Any]] = {
    # Provided contexts are the source abstract/document itself.
    "pubmedqa": {
        "question_context_role": QUESTION_CONTEXT_ROLE_SOURCE_DOCUMENT,
        "requires_shared_corpus_for_fair_retrieval": False,
        "notes": "Question contexts are the source abstract segments, not gold support snippets.",
    },
    "realmedqa": {
        "question_context_role": QUESTION_CONTEXT_ROLE_NONE,
        "requires_shared_corpus_for_fair_retrieval": True,
        "notes": "Questions are paired with gold NICE recommendations; benchmark against the shared recommendation corpus, not per-question answer text.",
    },
    # Provided snippets are expert-selected support evidence from larger PubMed docs.
    "bioasq": {
        "question_context_role": QUESTION_CONTEXT_ROLE_GOLD_EVIDENCE,
        "requires_shared_corpus_for_fair_retrieval": True,
        "notes": "Question contexts are gold support snippets; use a shared abstract corpus for fair retrieval benchmarking.",
    },
    "medhop": {
        "question_context_role": QUESTION_CONTEXT_ROLE_GOLD_EVIDENCE,
        "requires_shared_corpus_for_fair_retrieval": True,
        "notes": "Question contexts are benchmark support abstracts, not an open shared corpus.",
    },
    "multihoprag": {
        "question_context_role": QUESTION_CONTEXT_ROLE_GOLD_EVIDENCE,
        "requires_shared_corpus_for_fair_retrieval": True,
        "notes": "Use corpus.json for fair retrieval; per-question evidence_list items are oracle snippets only.",
    },
    # Closed-corpus question bundles with supporting and distractor passages.
    "hotpotqa": {
        "question_context_role": QUESTION_CONTEXT_ROLE_RETRIEVAL_BUNDLE,
        "requires_shared_corpus_for_fair_retrieval": False,
        "notes": "Question contexts are the benchmark-provided passage bundle, not pure gold snippets.",
    },
    "hotpotqa_fullwiki": {
        "question_context_role": QUESTION_CONTEXT_ROLE_GOLD_EVIDENCE,
        "requires_shared_corpus_for_fair_retrieval": True,
        "notes": (
            "Questions come from HotpotQA fullwiki, but retrieval should use a "
            "shared corpus prepared from FullWiki retrieved paragraphs or the "
            "official processed Wikipedia corpus."
        ),
    },
    "2wikimultihopqa": {
        "question_context_role": QUESTION_CONTEXT_ROLE_RETRIEVAL_BUNDLE,
        "requires_shared_corpus_for_fair_retrieval": False,
        "notes": "Question contexts are the benchmark-provided passage bundle, not pure gold snippets.",
    },
    "musique": {
        "question_context_role": QUESTION_CONTEXT_ROLE_RETRIEVAL_BUNDLE,
        "requires_shared_corpus_for_fair_retrieval": False,
        "notes": "Question contexts are the benchmark-provided paragraph bundle.",
    },
    "medqa": {
        "question_context_role": QUESTION_CONTEXT_ROLE_NONE,
        "requires_shared_corpus_for_fair_retrieval": False,
        "notes": "No retrieval corpus is provided by the question file.",
    },
    "medmcqa": {
        "question_context_role": QUESTION_CONTEXT_ROLE_NONE,
        "requires_shared_corpus_for_fair_retrieval": False,
        "notes": "No retrieval corpus is provided by the question file.",
    },
    "mmlu": {
        "question_context_role": QUESTION_CONTEXT_ROLE_NONE,
        "requires_shared_corpus_for_fair_retrieval": False,
        "notes": "No retrieval corpus is provided by the question file.",
    },
}


class TaskType(Enum):
    BINARY = "binary"      # yes/no/maybe (PubMedQA, BioASQ yesno, HotpotQA comparison)
    MCQ = "mcq"            # multiple choice (MedQA, MedMCQA, MMLU)
    FREE_TEXT = "free_text" # open-ended (HotpotQA bridge, 2WikiMultiHopQA)


@dataclass
class InferenceRecord:
    """Canonical record for model inference (what the model can see)."""
    id: str
    dataset: str
    question: str
    contexts: List[str]
    context_titles: Optional[List[Optional[str]]] = None
    options: Optional[Dict[str, str]] = None
    task_type: str = "binary"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoldRecord:
    """Canonical record for evaluation (ground truth)."""
    id: str
    short_answer: str
    long_answer: Optional[str] = None
    aliases: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Per-Dataset Adapters
# =============================================================================

def adapt_pubmedqa(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt PubMedQA format.
    
    Raw keys: QUESTION, CONTEXTS, LABELS, MESHES, YEAR, 
              final_decision, LONG_ANSWER, reasoning_*
    
    Note: q_id is passed explicitly because PubMedQA uses dict key as PMID.
    """
    # Use explicitly passed q_id (dict key), fallback to PMID field
    if q_id is None:
        q_id = raw_data.get("PMID", "")
    question = raw_data.get("QUESTION", "")
    
    # Extract contexts (list of strings)
    raw_contexts = raw_data.get("CONTEXTS", [])
    contexts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in raw_contexts]
    
    # Gold fields
    short_answer = raw_data.get("final_decision", "").lower()
    long_answer = raw_data.get("LONG_ANSWER", "")
    
    return (
        InferenceRecord(
            id=str(q_id),
            dataset="pubmedqa",
            question=question,
            contexts=contexts,
            options={"A": "yes", "B": "no", "C": "maybe"},
            task_type=TaskType.BINARY.value
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=long_answer,
            aliases=None
        )
    )


def _stable_record_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(str(part or "").strip() for part in parts)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _normalize_realmedqa_verdict(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def _is_realmedqa_ideal_record(raw_data: Dict[str, Any]) -> bool:
    plausible = _normalize_realmedqa_verdict(
        raw_data.get("_normalized_plausible", raw_data.get("Plausible", raw_data.get("plausible", "")))
    )
    answered = _normalize_realmedqa_verdict(
        raw_data.get("_normalized_answered", raw_data.get("Answered", raw_data.get("answered", "")))
    )
    return plausible == "completely" and answered == "completely"


def adapt_realmedqa(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt RealMedQA format.

    Expected raw keys:
      - Question
      - Recommendation
      - Generator
      - Plausible
      - Answered

    We intentionally do not expose Recommendation as an inference-time context,
    because it is the gold answer text. Retrieval should instead operate over
    the shared recommendation corpus built from the dataset rows.
    """
    question = str(raw_data.get("Question", raw_data.get("question", ""))).strip()
    recommendation = str(
        raw_data.get("Recommendation", raw_data.get("recommendation", raw_data.get("Answer", raw_data.get("answer", ""))))
    ).strip()

    if q_id is None:
        q_id = (
            raw_data.get("id")
            or raw_data.get("_id")
            or raw_data.get("row_id")
            or _stable_record_id("realmedqa", question, recommendation)
        )

    return (
        InferenceRecord(
            id=str(q_id),
            dataset="realmedqa",
            question=question,
            contexts=[],
            options=None,
            task_type=TaskType.FREE_TEXT.value,
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=recommendation.lower(),
            long_answer=recommendation,
            aliases=None,
        ),
    )


def adapt_bioasq(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt BioASQ format.
    
    Raw keys: id, body, type, exact_answer, ideal_answer, snippets, documents, ...
    """
    if q_id is None:
        q_id = raw_data.get("id", "")
    question = raw_data.get("body", "")
    q_type = raw_data.get("type", "")
    
    # Extract contexts from snippets
    raw_snippets = raw_data.get("snippets", [])
    contexts = [s.get("text", "") for s in raw_snippets if s.get("text")]
    
    # For yes/no questions
    if q_type == "yesno":
        exact_answer = raw_data.get("exact_answer", "")
        short_answer = exact_answer.lower() if exact_answer else ""
        task_type = TaskType.BINARY.value
        options = {"A": "yes", "B": "no"}
        aliases = None
    elif q_type == "factoid":
        # exact_answer is often [['entity_name', 'alias', ...], ...].
        # Keep the first synonym as the primary answer and preserve the rest
        # as aliases so evaluation doesn't undercount factoid variants.
        exact_answer = raw_data.get("exact_answer", [])
        alias_values: List[str] = []
        if exact_answer and isinstance(exact_answer[0], list) and exact_answer[0]:
            short_answer = exact_answer[0][0].lower().strip()
            for group in exact_answer:
                if isinstance(group, list):
                    alias_values.extend(str(x).lower().strip() for x in group if str(x).strip())
                elif isinstance(group, str):
                    alias_values.append(group.lower().strip())
        elif exact_answer and isinstance(exact_answer[0], str):
            short_answer = exact_answer[0].lower().strip()
            alias_values = [str(x).lower().strip() for x in exact_answer if str(x).strip()]
        else:
            short_answer = ""
        task_type = TaskType.FREE_TEXT.value
        options = None
        aliases = [a for a in alias_values if a and a != short_answer] or None
    else:
        # list/summary — skip (no reliable short answer)
        short_answer = ""
        task_type = TaskType.FREE_TEXT.value
        options = None
        aliases = None
    
    # Ideal answer serves as long answer
    long_answer = raw_data.get("ideal_answer", "")
    
    return (
        InferenceRecord(
            id=str(q_id),
            dataset="bioasq",
            question=question,
            contexts=contexts,
            options=options,
            task_type=task_type
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=long_answer,
            aliases=aliases
        )
    )


def adapt_medqa(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt MedQA (USMLE) format.
    
    Raw keys: question, options, answer_idx, (answer text in options)
    """
    if q_id is None:
        q_id = raw_data.get("id", str(hash(raw_data.get("question", "")))[:10])
    question = raw_data.get("question", "")
    raw_options = raw_data.get("options", {})
    answer_idx = raw_data.get("answer_idx", 0)
    explicit_answer = str(raw_data.get("answer", "")).strip()
    
    # Options are like {"0": "option text", "1": "...", ...} or {"A": "...", "B": "...", ...}
    # Map to A, B, C, D format
    options = {}
    for k, v in raw_options.items():
        # Normalize key to A, B, C, D
        if k.isdigit():
            idx = int(k)
            key = chr(ord('A') + idx)
        else:
            key = k.upper()
        options[key] = v
    
    # Prefer the explicit answer text when the release provides it.
    if explicit_answer:
        short_answer = explicit_answer.lower()
    else:
        correct_key: Optional[str] = None
        answer_keys = list(options.keys())
        if isinstance(answer_idx, int):
            if 0 <= answer_idx < len(answer_keys):
                correct_key = answer_keys[answer_idx]
        elif isinstance(answer_idx, str):
            answer_idx = answer_idx.strip()
            if answer_idx.isdigit():
                idx = int(answer_idx)
                if 0 <= idx < len(answer_keys):
                    correct_key = answer_keys[idx]
            else:
                letter_key = answer_idx.upper()
                if letter_key in options:
                    correct_key = letter_key

        short_answer = options.get(correct_key, "").lower()
    
    # MedQA has no explicit long answer or contexts in the question file
    # (context is the question itself for some formats)
    
    return (
        InferenceRecord(
            id=str(q_id),
            dataset="medqa",
            question=question,
            contexts=[],  # No explicit contexts in MedQA question file
            options=options,
            task_type=TaskType.MCQ.value
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=None,
            aliases=None
        )
    )


def adapt_medmcqa(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt MedMCQA format.
    
    Raw keys: id, question, opa, opb, opc, opd, cop (correct option index)
    """
    if q_id is None:
        q_id = raw_data.get("id", "")
    question = raw_data.get("question", "")
    
    options = {
        "A": raw_data.get("opa", ""),
        "B": raw_data.get("opb", ""),
        "C": raw_data.get("opc", ""),
        "D": raw_data.get("opd", "")
    }
    
    # cop is 1-indexed (1=A, 2=B, etc.)
    cop = raw_data.get("cop", 1)
    correct_key = chr(ord('A') + cop - 1)
    short_answer = options.get(correct_key, "").lower()
    
    return (
        InferenceRecord(
            id=str(q_id),
            dataset="medmcqa",
            question=question,
            contexts=[],
            options=options,
            task_type=TaskType.MCQ.value
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=None,
            aliases=None
        )
    )


def adapt_mmlu(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt MMLU format (single row).
    
    Raw keys: question, A, B, C, D, answer
    """
    if q_id is None:
        q_id = raw_data.get("id", "")
    question = raw_data.get("question", "")
    
    options = {
        "A": raw_data.get("A", ""),
        "B": raw_data.get("B", ""),
        "C": raw_data.get("C", ""),
        "D": raw_data.get("D", "")
    }
    
    # Answer is single letter A, B, C, or D
    answer_letter = raw_data.get("answer", "A").upper()
    short_answer = options.get(answer_letter, "").lower()
    
    return (
        InferenceRecord(
            id=str(q_id),
            dataset="mmlu",
            question=question,
            contexts=[],
            options=options,
            task_type=TaskType.MCQ.value
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=None,
            aliases=None
        )
    )


def adapt_hotpotqa(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt HotpotQA (fullwiki) format.

    Raw keys: _id, question, answer, type, level, supporting_facts,
              context [[title, [sentences]], ...]

    type is either "bridge" (multi-hop factoid) or "comparison" (yes/no).
    Contexts: all passages provided per question; supporting_facts marks
    which sentences are actually relevant, but we include all passages so
    the retriever has to do the work.
    """
    if q_id is None:
        q_id = raw_data.get("_id", "")
    question = raw_data.get("question", "")
    answer = raw_data.get("answer", "").lower().strip()
    q_type = raw_data.get("type", "bridge")

    # Each context entry is [title, [sent0, sent1, ...]]
    # Flatten to one string per passage (title + sentences joined)
    contexts = []
    context_titles: List[Optional[str]] = []
    for title, sentences in raw_data.get("context", []):
        passage = title + ". " + " ".join(sentences)
        contexts.append(passage)
        context_titles.append(str(title).strip() or None)

    if q_type == "comparison" and answer in {"yes", "no"}:
        task_type = TaskType.BINARY.value
        options = {"A": "yes", "B": "no"}
    else:
        task_type = TaskType.FREE_TEXT.value
        options = None

    return (
        InferenceRecord(
            id=str(q_id),
            dataset="hotpotqa",
            question=question,
            contexts=contexts,
            context_titles=context_titles,
            options=options,
            task_type=task_type,
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=answer,
            long_answer=None,
            aliases=None,
        ),
    )


def adapt_hotpotqa_fullwiki(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """Adapt HotpotQA FullWiki while keeping a distinct dataset label."""
    inf_rec, gold_rec = adapt_hotpotqa(raw_data, q_id=q_id)
    inf_rec.dataset = "hotpotqa_fullwiki"
    return inf_rec, gold_rec


def adapt_2wikimultihopqa(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt 2WikiMultiHopQA format.

    Raw keys: _id, question, answer, type, evidences, supporting_facts,
              context [[title, [sentences]], ...]

    Types: bridge, comparison, inference, compositional.
    Answers are short strings (entity names or yes/no).
    """
    if q_id is None:
        q_id = raw_data.get("_id", "")
    question = raw_data.get("question", "")
    answer = raw_data.get("answer", "").lower().strip()
    q_type = raw_data.get("type", "bridge")

    # Same structure as HotpotQA context
    contexts = []
    context_titles: List[Optional[str]] = []
    for title, sentences in raw_data.get("context", []):
        passage = title + ". " + " ".join(sentences)
        contexts.append(passage)
        context_titles.append(str(title).strip() or None)

    # Only use explicit answer aliases if the release provides them.
    aliases = [
        str(a).lower().strip()
        for a in (raw_data.get("answer_aliases", []) or [])
        if str(a).strip()
    ]
    aliases = [a for a in aliases if a and a != answer] or None

    # Types: comparison, bridge_comparison, compositional, inference
    # comparison type may have yes/no or entity answers
    if q_type in {"comparison", "bridge_comparison"} and answer in {"yes", "no"}:
        task_type = TaskType.BINARY.value
        options = {"A": "yes", "B": "no"}
    else:
        task_type = TaskType.FREE_TEXT.value
        options = None

    return (
        InferenceRecord(
            id=str(q_id),
            dataset="2wikimultihopqa",
            question=question,
            contexts=contexts,
            context_titles=context_titles,
            options=options,
            task_type=task_type,
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=answer,
            long_answer=None,
            aliases=aliases,
        ),
    )


def adapt_musique(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt MuSiQue format.

    Typical raw keys:
      - id
      - question
      - answer
      - answer_aliases (optional)
      - paragraphs: [{title, paragraph_text, is_supporting, ...}, ...]

    MuSiQue is a multi-hop free-text QA benchmark.
    """
    if q_id is None:
        q_id = raw_data.get("id", raw_data.get("_id", ""))

    question = str(raw_data.get("question", "")).strip()
    short_answer = str(raw_data.get("answer", "")).lower().strip()

    # Build contexts from paragraph objects when available.
    contexts: List[str] = []
    context_titles: List[Optional[str]] = []
    for p in raw_data.get("paragraphs", []) or []:
        if not isinstance(p, dict):
            txt = str(p).strip()
            if txt:
                contexts.append(txt)
                context_titles.append(None)
            continue

        title = str(p.get("title", "")).strip()
        paragraph_text = str(
            p.get("paragraph_text", p.get("paragraph", p.get("context", "")))
        ).strip()

        if paragraph_text:
            if title and not paragraph_text.lower().startswith(title.lower()):
                contexts.append(f"{title}. {paragraph_text}")
            else:
                contexts.append(paragraph_text)
            context_titles.append(title or None)

    # Fallback to generic context key if paragraphs are absent.
    if not contexts:
        raw_context = raw_data.get("context", [])
        if isinstance(raw_context, list):
            for entry in raw_context:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    title = str(entry[0]).strip()
                    body = " ".join(str(x).strip() for x in (entry[1] or []) if str(x).strip())
                    if body:
                        contexts.append(f"{title}. {body}" if title else body)
                        context_titles.append(title or None)
                elif isinstance(entry, dict):
                    title = str(entry.get("title", "")).strip()
                    body = str(entry.get("text", entry.get("paragraph_text", ""))).strip()
                    if body:
                        contexts.append(f"{title}. {body}" if title else body)
                        context_titles.append(title or None)
                else:
                    txt = str(entry).strip()
                    if txt:
                        contexts.append(txt)
                        context_titles.append(None)

    aliases = [str(a).lower().strip() for a in (raw_data.get("answer_aliases", []) or []) if str(a).strip()]
    aliases = [a for a in aliases if a and a != short_answer]

    return (
        InferenceRecord(
            id=str(q_id),
            dataset="musique",
            question=question,
            contexts=contexts,
            context_titles=context_titles or None,
            options=None,
            task_type=TaskType.FREE_TEXT.value,
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=None,
            aliases=aliases or None,
        ),
    )


def adapt_medhop(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt MedHop format.

    Raw keys:
      - id
      - query
      - answer
      - candidates: [candidate0, candidate1, ...]
      - supports: [abstract0, abstract1, ...]

    MedHop is a biomedical multi-hop multiple-choice QA benchmark induced from
    PubMed abstracts. We keep the provided supporting abstracts as retrieval
    contexts and expose the candidate set as MCQ options.
    """
    if q_id is None:
        q_id = raw_data.get("id", "")

    question = str(raw_data.get("query", "")).strip()
    short_answer = str(raw_data.get("answer", "")).lower().strip()

    contexts = [
        str(ctx).strip()
        for ctx in (raw_data.get("supports", []) or [])
        if str(ctx).strip()
    ]

    candidates = [
        str(candidate).strip()
        for candidate in (raw_data.get("candidates", []) or [])
        if str(candidate).strip()
    ]
    options = {
        chr(ord("A") + idx): candidate
        for idx, candidate in enumerate(candidates)
    } or None

    return (
        InferenceRecord(
            id=str(q_id),
            dataset="medhop",
            question=question,
            contexts=contexts,
            options=options,
            task_type=TaskType.MCQ.value,
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=None,
            aliases=None,
        ),
    )


def _format_multihoprag_evidence_item(item: Dict[str, Any]) -> str:
    """Format one evidence snippet from MultiHop-RAG for compact provenance-rich display."""
    parts: List[str] = []

    def _clean(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() == "none" else text

    title = _clean(item.get("title", ""))
    source = _clean(item.get("source", ""))
    category = _clean(item.get("category", ""))
    published_at = _clean(item.get("published_at", ""))
    fact = _clean(item.get("fact", ""))

    if title:
        parts.append(f"Title: {title}")
    if source:
        parts.append(f"Source: {source}")
    if category:
        parts.append(f"Category: {category}")
    if published_at:
        parts.append(f"Published at: {published_at}")
    if fact:
        parts.append(f"Fact: {fact}")

    return "\n".join(parts).strip()


def _format_multihoprag_corpus_document(doc: Dict[str, Any]) -> str:
    """Format one MultiHop-RAG corpus document for indexing / KG construction."""
    parts: List[str] = []

    def _clean(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() == "none" else text

    title = _clean(doc.get("title", ""))
    author = _clean(doc.get("author", ""))
    source = _clean(doc.get("source", ""))
    category = _clean(doc.get("category", ""))
    published_at = _clean(doc.get("published_at", ""))
    body = _clean(doc.get("body", ""))

    if title:
        parts.append(f"Title: {title}")
    if source:
        parts.append(f"Source: {source}")
    if author:
        parts.append(f"Author: {author}")
    if category:
        parts.append(f"Category: {category}")
    if published_at:
        parts.append(f"Published at: {published_at}")
    if body:
        parts.append(f"Body:\n{body}")

    return "\n".join(parts).strip()


def adapt_multihoprag(raw_data: Dict[str, Any], q_id: str = None) -> Tuple[InferenceRecord, GoldRecord]:
    """
    Adapt MultiHop-RAG format.

    Raw keys:
      - query
      - answer
      - question_type
      - evidence_list: [{title, source, category, published_at, fact, ...}, ...]

    Note:
      The benchmark also ships a shared global corpus in corpus.json. We use the
      per-question evidence snippets only to keep normalized question records
      non-empty and inspectable; the actual KG/index for this dataset is built
      from the full shared corpus via build_global_corpus_passages().
    """
    if q_id is None:
        q_id = raw_data.get("id", f"multihoprag_{abs(hash(raw_data.get('query', '')))}")

    question = str(raw_data.get("query", "")).strip()
    short_answer = str(raw_data.get("answer", "")).lower().strip()

    evidence_list = raw_data.get("evidence_list", []) or []
    contexts = []
    for item in evidence_list:
        if not isinstance(item, dict):
            continue
        formatted = _format_multihoprag_evidence_item(item)
        if formatted:
            contexts.append(formatted)

    if short_answer in {"yes", "no"}:
        task_type = TaskType.BINARY.value
        options = {"A": "yes", "B": "no"}
    else:
        task_type = TaskType.FREE_TEXT.value
        options = None

    return (
        InferenceRecord(
            id=str(q_id),
            dataset="multihoprag",
            question=question,
            contexts=contexts,
            options=options,
            task_type=task_type,
        ),
        GoldRecord(
            id=str(q_id),
            short_answer=short_answer,
            long_answer=None,
            aliases=None,
        ),
    )


# =============================================================================
# Adapter Registry
# =============================================================================

ADAPTERS = {
    "pubmedqa": adapt_pubmedqa,
    "realmedqa": adapt_realmedqa,
    "bioasq": adapt_bioasq,
    "medqa": adapt_medqa,
    "medmcqa": adapt_medmcqa,
    "mmlu": adapt_mmlu,
    "medhop": adapt_medhop,
    "multihoprag": adapt_multihoprag,
    "hotpotqa": adapt_hotpotqa,
    "hotpotqa_fullwiki": adapt_hotpotqa_fullwiki,
    "2wikimultihopqa": adapt_2wikimultihopqa,
    "musique": adapt_musique,
}


def infer_hop_count_from_raw(
    dataset_name: str,
    question_id: str,
    raw_question: Dict[str, Any],
) -> Optional[int]:
    """
    Infer reasoning hop count from raw dataset metadata when available.

    Preference order:
      1. Explicit decomposition length
      2. Dataset-native evidence-chain fields that directly reflect reasoning hops
      3. MuSiQue-style ID prefixes such as ``3hop1__...``
    """
    if isinstance(raw_question, dict):
        for key in (
            "question_decomposition",
            "decomposition",
            "reasoning_steps",
            "sub_questions",
            "supporting_questions",
        ):
            value = raw_question.get(key)
            if isinstance(value, list) and value:
                return len(value)

        if dataset_name == "multihoprag":
            evidence_list = raw_question.get("evidence_list")
            if isinstance(evidence_list, list) and evidence_list:
                return len(evidence_list)

        if dataset_name == "2wikimultihopqa":
            evidences = raw_question.get("evidences")
            if isinstance(evidences, list) and evidences:
                return len(evidences)
            supporting_facts = raw_question.get("supporting_facts")
            if isinstance(supporting_facts, list) and supporting_facts:
                return len(supporting_facts)

        if dataset_name in {"hotpotqa", "hotpotqa_fullwiki"}:
            supporting_facts = raw_question.get("supporting_facts")
            if isinstance(supporting_facts, list) and supporting_facts:
                return len(supporting_facts)

    match = re.match(r"(?i)^(\d+)hop", str(question_id or "").strip())
    if match:
        return int(match.group(1))
    return None


def get_dataset_corpus_profile(dataset_name: str) -> Dict[str, Any]:
    """
    Describe what the per-question contexts mean for a dataset.

    This lets the experiment pipeline distinguish between:
      - source documents / retrieval bundles that can be indexed directly
      - oracle gold-evidence snippets that should not be used as a shared
        retrieval corpus unless the user explicitly opts into a controlled-
        evidence evaluation
    """
    dataset_key = dataset_name.lower()
    profile = DATASET_CORPUS_PROFILES.get(
        dataset_key,
        {
            "question_context_role": QUESTION_CONTEXT_ROLE_RETRIEVAL_BUNDLE,
            "requires_shared_corpus_for_fair_retrieval": False,
            "notes": "No explicit corpus profile registered; treating per-question contexts as retrieval bundles.",
        },
    )
    return {
        "dataset": dataset_key,
        **profile,
    }


def load_raw_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Load raw dataset from MIRAGE rawdata directory.
    
    Returns dict mapping id -> raw record.
    """
    dataset_name = dataset_name.lower()
    
    file_paths = {
        "pubmedqa": "MIRAGE/rawdata/pubmedqa/data/test_set.json",
        "realmedqa": "MIRAGE/rawdata/realmedqa/RealMedQA.json",
        "bioasq": "MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json",
        "medqa": "MIRAGE/rawdata/medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl",
        "medmcqa": "MIRAGE/rawdata/medmcqa/data/test.json",
        "mmlu": "MIRAGE/rawdata/mmlu/data/test",  # Directory with CSV files
        "medhop": "MIRAGE/rawdata/medhop/dev.json",
        "multihoprag": "MIRAGE/rawdata/multihoprag/MultiHopRAG.json",
        # Multi-hop datasets
        # Download: https://hotpotqa.github.io/  (hotpot_dev_fullwiki_v1.json)
        "hotpotqa": "MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json",
        "hotpotqa_fullwiki": "MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json",
        # Download: https://github.com/Alab-NII/2wikimultihop  (dev.json)
        "2wikimultihopqa": "MIRAGE/rawdata/2wikimultihopqa/dev.json",
        # Download: https://github.com/stonybrooknlp/musique
        "musique": "MIRAGE/rawdata/musique/dev.jsonl",
    }
    
    file_path = file_paths.get(dataset_name)
    if not file_path:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "realmedqa":
        candidates = [
            "MIRAGE/rawdata/realmedqa/RealMedQA.json",
            "MIRAGE/rawdata/realmedqa/RealMedQA.jsonl",
            "MIRAGE/rawdata/realmedqa/train.json",
            "MIRAGE/rawdata/realmedqa/train.jsonl",
            "MIRAGE/rawdata/realmedqa/RealMedQA.csv",
            "MIRAGE/rawdata/realmedqa/train.csv",
        ]
        rows = _load_realmedqa_records(candidates)
        if rows is None:
            raise FileNotFoundError(
                "RealMedQA dataset file not found. Looked for: " + ", ".join(candidates)
            )
        ideal_rows = [row for row in rows if _is_realmedqa_ideal_record(row)]
        return {
            str(row.get("id", row.get("_id", row.get("_fallback_id", f"realmedqa_{idx}")))): row
            for idx, row in enumerate(ideal_rows)
        }

    # MuSiQue releases use different file names across mirrors; try common candidates.
    if dataset_name == "musique":
        candidates = [
            "MIRAGE/rawdata/musique/dev.jsonl",
            "MIRAGE/rawdata/musique/dev.json",
            "MIRAGE/rawdata/musique/musique_ans_v1.0_dev.jsonl",
            "MIRAGE/rawdata/musique/musique_full_v1.0_dev.jsonl",
            "MIRAGE/rawdata/musique/musique_ans_v1.0_test.jsonl",
        ]
        existing = next((p for p in candidates if os.path.exists(p)), None)
        if existing:
            file_path = existing
        else:
            raise FileNotFoundError(
                "MuSiQue dataset file not found. Looked for: " + ", ".join(candidates)
            )
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if dataset_name == "pubmedqa":
        with open(file_path, 'r') as f:
            return json.load(f)
    
    elif dataset_name == "bioasq":
        with open(file_path, 'r') as f:
            data = json.load(f)
        # BioASQ has "questions" key with list
        if "questions" in data:
            return {q["id"]: q for q in data["questions"]}
        return data
    
    elif dataset_name == "medqa":
        # JSONL format
        data = {}
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        q_id = item.get("id", f"medqa_{idx}")
                        data[q_id] = item
                    except json.JSONDecodeError:
                        continue
        return data
    
    elif dataset_name == "medmcqa":
        # JSON format with list
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return {item["id"]: item for item in data}
        return data
    
    elif dataset_name == "mmlu":
        # MMLU is special - directory with multiple CSV files
        import pandas as pd
        data = {}
        mmlu_dir = file_path
        domain_files = [
            "anatomy_test.csv", "clinical_knowledge_test.csv", 
            "college_biology_test.csv", "college_medicine_test.csv",
            "medical_genetics_test.csv", "professional_medicine_test.csv"
        ]
        for domain_file in domain_files:
            full_path = os.path.join(mmlu_dir, domain_file)
            if os.path.exists(full_path):
                df = pd.read_csv(full_path, names=["question", "A", "B", "C", "D", "answer"])
                domain = domain_file.replace("_test.csv", "")
                for idx, row in df.iterrows():
                    q_id = f"{domain}_{idx}"
                    data[q_id] = {
                        "id": q_id,
                        "question": row["question"],
                        "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"],
                        "answer": row["answer"]
                    }
        return data
    
    elif dataset_name in {"hotpotqa", "hotpotqa_fullwiki", "2wikimultihopqa", "medhop", "multihoprag"}:
        # Both use JSON list format: [{_id, question, answer, context, ...}, ...]
        with open(file_path, 'r') as f:
            items = json.load(f)
        if isinstance(items, list):
            return {
                str(item.get("_id", item.get("id", f"{dataset_name}_{idx}"))): item
                for idx, item in enumerate(items)
            }
        # Some releases wrap in a dict key
        for key in ("data", "questions"):
            if key in items:
                return {
                    str(item.get("_id", item.get("id", f"{dataset_name}_{idx}"))): item
                    for idx, item in enumerate(items[key])
                }
        return items

    elif dataset_name == "musique":
        data = {}

        if file_path.endswith(".jsonl"):
            with open(file_path, 'r') as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q_id = item.get("id", item.get("_id", f"musique_{idx}"))
                    data[str(q_id)] = item
            return data

        # JSON list/dict fallback
        with open(file_path, 'r') as f:
            items = json.load(f)
        if isinstance(items, list):
            return {
                str(item.get("id", item.get("_id", f"musique_{idx}"))): item
                for idx, item in enumerate(items)
            }
        for key in ("data", "questions"):
            if isinstance(items, dict) and key in items and isinstance(items[key], list):
                return {
                    str(item.get("id", item.get("_id", f"musique_{idx}"))): item
                    for idx, item in enumerate(items[key])
                }
        return items

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def normalize_dataset(dataset_name: str) -> Tuple[List[InferenceRecord], List[GoldRecord]]:
    """
    Load and normalize a dataset to canonical schema.
    
    Returns:
        (inference_records, gold_records) - paired by id
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name not in ADAPTERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(ADAPTERS.keys())}")
    
    adapter = ADAPTERS[dataset_name]
    raw_data = load_raw_dataset(dataset_name)
    
    inference_records = []
    gold_records = []
    
    for q_id, raw_record in raw_data.items():
        try:
            # Pass q_id explicitly for PubMedQA which uses dict key as ID
            inf_rec, gold_rec = adapter(raw_record, q_id=q_id)
            inference_records.append(inf_rec)
            gold_records.append(gold_rec)
        except Exception as e:
            logger.warning(f"Failed to adapt record {q_id}: {e}")
            continue
    
    # Validate ID sets match
    inf_ids = set(r.id for r in inference_records)
    gold_ids = set(r.id for r in gold_records)
    if inf_ids != gold_ids:
        missing_gold = inf_ids - gold_ids
        missing_inf = gold_ids - inf_ids
        if missing_gold:
            logger.warning(f"Missing gold records for {len(missing_gold)} inference records")
        if missing_inf:
            logger.warning(f"Missing inference records for {len(missing_inf)} gold records")
    
    logger.info(f"Normalized {dataset_name}: {len(inference_records)} inference, {len(gold_records)} gold records")
    
    return inference_records, gold_records


@dataclass
class ContextPassage:
    """One source passage from a benchmark dataset, carrying its provenance."""
    text: str
    dataset: str
    question_id: str
    passage_index: int
    source_title: Optional[str] = None


def build_passage_corpus(
    inference_records: List[InferenceRecord],
    *,
    dedupe_across_questions: bool = True,
) -> List[ContextPassage]:
    """
    Build a deduplicated list of ContextPassage objects from inference records.

    Each passage corresponds to exactly one context string from one record.
    Cross-question deduplication is done by text content by default so passages
    shared across questions are stored only once (using whichever question is
    seen first as provenance). For question-scoped corpora, callers should set
    ``dedupe_across_questions=False`` so recurring passages remain available in
    every question-local bundle.

    Passages are NOT joined before being handed to the KG builder; the caller
    is responsible for passing them to a passage-aware extraction method so
    the chunker never slices across passage boundaries.
    """
    seen: set = set()
    passages: List[ContextPassage] = []
    for rec in inference_records:
        rec_titles = list(rec.context_titles or [])
        for idx, ctx in enumerate(rec.contexts):
            ctx_stripped = ctx.strip()
            if not ctx_stripped:
                continue
            dedupe_key = (
                ctx_stripped
                if dedupe_across_questions
                else (str(rec.id), ctx_stripped)
            )
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                passages.append(ContextPassage(
                    text=ctx_stripped,
                    dataset=rec.dataset,
                    question_id=rec.id,
                    passage_index=idx,
                    source_title=rec_titles[idx] if idx < len(rec_titles) else None,
                ))
    return passages


def build_context_corpus(inference_records: List[InferenceRecord]) -> List[str]:
    """Return deduplicated context strings (text only, no provenance).

    Kept for call sites that only need plain strings.  For KG construction
    use build_passage_corpus instead so passage boundaries are preserved.
    """
    return [p.text for p in build_passage_corpus(inference_records)]


def _load_optional_json_records(paths: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Load the first existing JSON/JSONL/CSV corpus file from a list of candidate paths."""
    for path in paths:
        if not os.path.exists(path):
            continue

        if path.endswith(".csv"):
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]

        if path.endswith(".jsonl"):
            records: List[Dict[str, Any]] = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        records.append(item)
            return records

        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("records", "documents", "data", "abstracts"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
    return None


def _load_realmedqa_records(candidate_paths: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Load RealMedQA records from JSON/JSONL/CSV and normalize verifier fields."""
    records = _load_optional_json_records(candidate_paths)
    if records is None:
        return None

    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        question = str(row.get("Question", row.get("question", ""))).strip()
        recommendation = str(
            row.get("Recommendation", row.get("recommendation", row.get("Answer", row.get("answer", ""))))
        ).strip()
        if not question or not recommendation:
            continue

        item = dict(row)
        item["_normalized_plausible"] = _normalize_realmedqa_verdict(
            row.get("Plausible", row.get("plausible", ""))
        )
        item["_normalized_answered"] = _normalize_realmedqa_verdict(
            row.get("Answered", row.get("answered", ""))
        )
        item["_fallback_id"] = _stable_record_id("realmedqa", idx, question, recommendation)
        normalized.append(item)
    return normalized


def _format_realmedqa_recommendation_document(doc: Dict[str, Any]) -> str:
    recommendation = str(
        doc.get("Recommendation", doc.get("recommendation", doc.get("Answer", doc.get("answer", ""))))
    ).strip()
    return f"Recommendation:\n{recommendation}" if recommendation else ""


def _format_biomedical_corpus_document(doc: Dict[str, Any]) -> str:
    """Format a generic biomedical abstract/document record for indexing."""
    title = str(doc.get("title", "")).strip()
    abstract = str(
        doc.get("abstract", doc.get("text", doc.get("body", doc.get("contents", ""))))
    ).strip()
    pmid = str(doc.get("pmid", doc.get("PMID", doc.get("id", "")))).strip()

    parts: List[str] = []
    if pmid:
        parts.append(f"PMID: {pmid}")
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract:\n{abstract}")
    return "\n".join(parts).strip()


def _format_hotpotqa_corpus_document(doc: Dict[str, Any]) -> str:
    """Format a HotpotQA shared-corpus paragraph for indexing."""
    title = str(doc.get("title", doc.get("source_title", ""))).strip()
    text_value = doc.get("text", doc.get("paragraph", doc.get("contents", "")))
    if isinstance(text_value, list):
        text = " ".join(str(part).strip() for part in text_value if str(part).strip())
    else:
        text = str(text_value or "").strip()

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if text:
        parts.append(text)
    return "\n".join(parts).strip()


def _build_optional_shared_corpus_passages(
    dataset_name: str,
    candidate_paths: List[str],
) -> Optional[List[ContextPassage]]:
    """Build passages from an optional local shared-corpus file if one exists."""
    docs = _load_optional_json_records(candidate_paths)
    if docs is None:
        return None

    seen: set = set()
    passages: List[ContextPassage] = []
    for idx, doc in enumerate(docs):
        text = _format_biomedical_corpus_document(doc)
        if text and text not in seen:
            seen.add(text)
            passages.append(
                ContextPassage(
                    text=text,
                    dataset=dataset_name,
                    question_id=str(doc.get("pmid", doc.get("PMID", doc.get("id", f"doc_{idx}")))),
                    passage_index=idx,
                )
            )
    return passages


def build_global_corpus_passages(dataset_name: str) -> Optional[List[ContextPassage]]:
    """
    Return dataset-level corpus passages when a benchmark ships a shared corpus
    separate from the per-question evidence snippets.

    This is used for true corpus-level retrieval/KG construction on datasets
    like MultiHop-RAG, where indexing only the gold evidence would artificially
    simplify retrieval.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "bioasq":
        return _build_optional_shared_corpus_passages(
            "bioasq",
            [
                "MIRAGE/rawdata/bioasq/corpus.jsonl",
                "MIRAGE/rawdata/bioasq/corpus.json",
                "MIRAGE/rawdata/bioasq/abstracts.jsonl",
                "MIRAGE/rawdata/bioasq/pubmed_abstracts.jsonl",
            ],
        )
    if dataset_name == "medhop":
        return _build_optional_shared_corpus_passages(
            "medhop",
            [
                "MIRAGE/rawdata/medhop/corpus.jsonl",
                "MIRAGE/rawdata/medhop/corpus.json",
                "MIRAGE/rawdata/medhop/abstracts.jsonl",
            ],
        )
    if dataset_name == "realmedqa":
        docs = _load_realmedqa_records(
            [
                "MIRAGE/rawdata/realmedqa/RealMedQA.json",
                "MIRAGE/rawdata/realmedqa/RealMedQA.jsonl",
                "MIRAGE/rawdata/realmedqa/train.json",
                "MIRAGE/rawdata/realmedqa/train.jsonl",
                "MIRAGE/rawdata/realmedqa/RealMedQA.csv",
                "MIRAGE/rawdata/realmedqa/train.csv",
            ]
        )
        if docs is None:
            return None

        seen: set = set()
        passages: List[ContextPassage] = []
        for idx, doc in enumerate(d for d in docs if _is_realmedqa_ideal_record(d)):
            text = _format_realmedqa_recommendation_document(doc)
            if text and text not in seen:
                seen.add(text)
                passages.append(
                    ContextPassage(
                        text=text,
                        dataset="realmedqa",
                        question_id=str(doc.get("id", doc.get("_id", doc.get("_fallback_id", f"realmedqa_doc_{idx}")))),
                        passage_index=idx,
                    )
                )
        return passages
    if dataset_name == "hotpotqa_fullwiki":
        docs = _load_optional_json_records(
            [
                "MIRAGE/rawdata/hotpotqa/fullwiki_corpus.jsonl",
                "MIRAGE/rawdata/hotpotqa/hotpotqa_fullwiki_corpus.jsonl",
                "MIRAGE/rawdata/hotpotqa/fullwiki_corpus.json",
                "MIRAGE/rawdata/hotpotqa/hotpotqa_fullwiki_corpus.json",
            ]
        )
        if docs is None:
            return None

        seen: set = set()
        passages: List[ContextPassage] = []
        for idx, doc in enumerate(docs):
            text = _format_hotpotqa_corpus_document(doc)
            if not text or text in seen:
                continue
            seen.add(text)
            passages.append(
                ContextPassage(
                    text=text,
                    dataset="hotpotqa_fullwiki",
                    question_id=str(doc.get("id", doc.get("doc_id", f"hotpot_doc_{idx}"))),
                    passage_index=idx,
                    source_title=str(doc.get("title", doc.get("source_title", ""))).strip() or None,
                )
            )
        return passages
    if dataset_name != "multihoprag":
        return None

    corpus_path = "MIRAGE/rawdata/multihoprag/corpus.json"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"MultiHop-RAG corpus file not found: {corpus_path}")

    with open(corpus_path, "r") as f:
        docs = json.load(f)

    seen: set = set()
    passages: List[ContextPassage] = []
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        text = _format_multihoprag_corpus_document(doc)
        if text and text not in seen:
            seen.add(text)
            passages.append(ContextPassage(
                text=text,
                dataset="multihoprag",
                question_id=f"doc_{idx}",
                passage_index=idx,
            ))
    return passages


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_no_leakage(inference_records: List[InferenceRecord], gold_records: List[GoldRecord]) -> bool:
    """
    Validate that inference records don't contain gold fields and that gold
    answers don't appear verbatim inside the provided context passages.

    Two checks:
    1. Schema leakage: inference records must not carry answer-bearing fields
       (short_answer, long_answer).
    2. Answer leakage: for every inference record, the gold answer string must
       not appear as a verbatim substring of any context passage.  Short binary
       answers (yes/no) are exempt because they appear legitimately in passages.

    Returns True only if both checks pass.
    """
    gold_by_id = {r.id: r for r in gold_records}

    for rec in inference_records:
        if rec.id not in gold_by_id:
            logger.warning(f"Inference record {rec.id} has no matching gold record")
            return False

    # Check 1: schema leakage
    for rec in inference_records:
        if hasattr(rec, 'short_answer') or hasattr(rec, 'long_answer'):
            logger.error(f"Leakage: Inference record {rec.id} has answer fields")
            return False

    # Check 2: answer text leakage into contexts.
    # Collect all meaningful surface forms for each gold record: primary answer,
    # aliases, and any long-answer variant.  Check every surface form so that
    # multi-answer datasets (e.g. 2WikiMultiHopQA) don't slip through.
    _binary = {"yes", "no", "true", "false", "maybe"}
    leakage_count = 0
    for rec in inference_records:
        gold = gold_by_id.get(rec.id)
        if gold is None:
            continue
        # Collect all answer surface forms from the gold record
        primary = str(getattr(gold, 'short_answer', '') or getattr(gold, 'answer', '') or '').strip()
        long_ans = str(getattr(gold, 'long_answer', '') or '').strip()
        aliases = list(getattr(gold, 'aliases', None) or [])
        all_answers = {primary, long_ans, *aliases} - {''}
        # Filter: skip empty, binary, or too-short strings
        candidates = [
            a.lower() for a in all_answers
            if a and a.lower() not in _binary and len(a) > 3
        ]
        if not candidates:
            continue
        contexts = getattr(rec, 'contexts', []) or []
        ctx_texts = [str(c).lower() for c in contexts]
        for surface in candidates:
            if any(surface in ctx for ctx in ctx_texts):
                logger.warning(
                    f"Possible answer leakage in record {rec.id}: "
                    f"surface form '{surface[:60]}' found verbatim in a context passage"
                )
                leakage_count += 1
                break  # one warning per record is enough

    if leakage_count:
        logger.error(
            f"Answer leakage detected in {leakage_count}/{len(inference_records)} records. "
            "Results may over-estimate retrieval quality."
        )
        return False

    logger.info("Validation passed: no schema or answer leakage detected")
    return True
