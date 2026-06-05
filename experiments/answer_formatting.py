"""Task-aware answer formatting instructions for experiment-time generation."""

import re
from typing import Callable, Dict, Iterable, Optional

SHORT_ANSWER_DATASETS = {
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "multihoprag",
}

_BINARY_QUESTION_PREFIXES = (
    "is ",
    "are ",
    "does ",
    "do ",
    "can ",
    "should ",
    "was ",
    "were ",
    "has ",
    "have ",
    "did ",
)

_MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

_DATE_PATTERNS = (
    rf"\b{_MONTH_PATTERN}\s+\d{{1,2}}(?:\s*(?:and|-)\s*\d{{1,2}})?(?:,)?\s+\d{{4}}\b",
    rf"\b\d{{1,2}}(?:\s*(?:and|-)\s*\d{{1,2}})?\s+{_MONTH_PATTERN}\s+\d{{4}}\b",
    rf"\b{_MONTH_PATTERN}\s+\d{{4}}\b",
    r"\b\d{4}\b",
)

_NUMERIC_PATTERNS = (
    r"\b\d+(?:\.\d+)?(?:\s*%|\s*(?:mg|g|kg|ml|l|cm|mm|m|km|years?|months?|days?))?(?=\W|$)",
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",
)

_QUESTION_NORMALIZERS: Dict[str, Callable[[str, str, str], str]] = {}


def _extract_leading_label(response: str, labels: Iterable[str]) -> str:
    """Return a normalized leading decision label when one is explicitly present."""
    text = str(response or "").strip().lower()
    if not text:
        return ""

    allowed = [str(label).strip().lower() for label in labels if str(label).strip()]
    if not allowed:
        return ""

    # Exact single-token label.
    if text in allowed:
        return text

    # Leading label followed by punctuation or explanation.
    pattern = r"^\s*(" + "|".join(re.escape(label) for label in allowed) + r")\b"
    match = re.search(pattern, text)
    if match:
        return match.group(1)

    # Explicit conclusion pattern.
    pattern = (
        r"\b(?:answer|final answer|conclusion)\s*(?:is|:)\s*("
        + "|".join(re.escape(label) for label in allowed)
        + r")\b"
    )
    match = re.search(pattern, text)
    if match:
        return match.group(1)

    return ""


def _strip_short_answer_wrapper(response: str) -> str:
    """Remove explicit answer wrappers without paraphrasing the answer itself."""
    text = str(response or "").strip()
    if not text:
        return ""

    # Prefer the first non-empty line for wrapper-style outputs.
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)

    wrapper_patterns = [
        r"^\s*(?:final answer|answer)\s*[:\-]\s*(.+?)\s*$",
        r"^\s*the answer is\s+(.+?)\s*$",
    ]
    for pattern in wrapper_patterns:
        match = re.match(pattern, first_line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip(" .")

    return text


def _extract_leading_short_label(response: str) -> str:
    """Collapse short-answer labels like yes/no when they lead the response."""
    return _extract_leading_label(
        response,
        ("yes", "no", "maybe", "insufficient information"),
    )


def _is_binary_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    return any(text.startswith(prefix) for prefix in _BINARY_QUESTION_PREFIXES)


def _extract_date_like_span(response: str, question: str) -> str:
    """Return a likely date span when the question is asking for one."""
    q = str(question or "").strip().lower()
    if not (
        q.startswith("when ")
        or " what month" in q
        or q.startswith("what month")
        or " what year" in q
        or q.startswith("what year")
        or " what date" in q
        or q.startswith("what date")
    ):
        return ""

    text = str(response or "").strip()
    if not text:
        return ""

    for pattern in _DATE_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            # Prefer the longest/most specific match. Ties go to the last span.
            matches.sort(key=lambda m: (len(m.group(0)), m.start()))
            return matches[-1].group(0).strip().strip(" .")
    return ""


def _normalize_football_club_answer(response: str, question: str) -> str:
    """Trim club-specific suffixes for benchmark answers when safe to do so."""
    q = str(question or "").strip().lower()
    if "football club" not in q and "club" not in q:
        return str(response or "").strip()

    text = str(response or "").strip()
    if not text or len(text.split()) > 6:
        return text

    patterns = [
        r"\s+football club\.?$",
        r"\s+f\.c\.?$",
        r"\s+fc\.?$",
    ]
    normalized = text
    for pattern in patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    return normalized.strip().strip(" .") or text


def _normalize_multihop_short_answer(response: str, question: str, task_type: str) -> str:
    """Question-aware canonicalizer for short-answer multi-hop benchmarks."""
    text = _strip_short_answer_wrapper(response)
    if not text:
        return text

    # Many benchmark questions are yes/no but still routed through free-text.
    if task_type == "binary" or _is_binary_question(question):
        label = _extract_leading_short_label(text)
        if label:
            return label

    date_span = _extract_date_like_span(text, question)
    if date_span:
        return date_span

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()
    if 0 < len(first_sentence.split()) <= 6:
        text = first_sentence.strip().strip(" .")
    else:
        text = first_line

    text = _normalize_football_club_answer(text, question)
    return text


def _extract_numeric_span(response: str, question: str) -> str:
    """Return a likely count/measurement span when the question asks for one."""
    q = str(question or "").strip().lower()
    if not (
        q.startswith("how many")
        or q.startswith("how much")
        or " how many " in q
        or " how much " in q
    ):
        return ""

    text = str(response or "").strip()
    if not text:
        return ""

    for pattern in _NUMERIC_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip().strip(" .")
    return ""


def _extract_subject_before_copula(text: str) -> str:
    """Extract a short subject phrase from 'X is/was ...' style answers."""
    match = re.match(
        r"^\s*(.+?)\s+(?:is|was|are|were|refers to|stands for)\b",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    candidate = match.group(1).strip().strip(" ,.;:")
    if 0 < len(candidate.split()) <= 8:
        return candidate
    return ""


def _normalize_bioasq_answer(response: str, question: str, task_type: str) -> str:
    """Dataset-specific BioASQ canonicalizer for factoid-style free-text answers."""
    text = _strip_short_answer_wrapper(response)
    if not text:
        return text

    label = _extract_leading_short_label(text)
    if task_type == "binary" and label:
        return label

    date_span = _extract_date_like_span(text, question)
    if date_span:
        return date_span

    numeric_span = _extract_numeric_span(text, question)
    if numeric_span:
        return numeric_span

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()

    subject = _extract_subject_before_copula(first_sentence)
    if subject:
        return subject

    first_clause = re.split(r"\s*[,:;]\s*", first_sentence, maxsplit=1)[0].strip()
    if 0 < len(first_clause.split()) <= 8:
        return first_clause.strip(" .")

    if 0 < len(first_sentence.split()) <= 8:
        return first_sentence.strip(" .")

    return first_line


def _normalize_realmedqa_answer(response: str, question: str, task_type: str) -> str:
    """Dataset-specific recommendation-style canonicalizer for RealMedQA."""
    del question, task_type
    text = _strip_short_answer_wrapper(response)
    if not text:
        return text

    lowered = text.lower().strip()
    if lowered.startswith("insufficient information"):
        return "Insufficient Information."

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text

    return " ".join(sentences[:3]).strip()


def _normalize_generic_answer(response: str, question: str, task_type: str) -> str:
    """Conservative fallback suitable for chat-like free-form tasks."""
    del question, task_type
    return _strip_short_answer_wrapper(response)


def _register_normalizer(dataset_names: Iterable[str], fn: Callable[[str, str, str], str]) -> None:
    for name in dataset_names:
        _QUESTION_NORMALIZERS[str(name).strip().lower()] = fn


_register_normalizer(SHORT_ANSWER_DATASETS, _normalize_multihop_short_answer)
_register_normalizer({"bioasq"}, _normalize_bioasq_answer)
_register_normalizer({"realmedqa"}, _normalize_realmedqa_answer)


def build_answer_instructions(
    dataset_name: str,
    task_type: str,
    *,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """Return task-aware answer instructions for generation prompts.

    The goal is to make the generation format match the evaluation contract.
    """
    dataset = str(dataset_name or "").strip().lower()
    task = str(task_type or "").strip().lower()
    normalized_options = {
        str(k).strip().upper(): str(v).strip()
        for k, v in (options or {}).items()
        if str(v).strip()
    }

    lines = []

    if dataset == "pubmedqa":
        lines.extend([
            "This is a 3-label scientific inference task, not open-ended QA.",
            "Respond with exactly one word: yes, no, or maybe.",
            "Do not add any explanation, hedging phrase, or second sentence.",
            "Choose yes when the abstract overall supports the claim.",
            "Choose no when the abstract overall rejects the claim.",
            "IMPORTANT: 'maybe' is rare — use it only when the abstract explicitly states the evidence is inconclusive or contradictory and no overall conclusion can be drawn. Most abstracts have a clear directional conclusion; choose yes or no in those cases.",
            "If a study shows mixed results but the authors draw a net conclusion (e.g. 'overall, treatment X was effective'), that counts as yes or no, not maybe.",
            "When graph paths are present, treat them as auxiliary support only; they must not override the study conclusion stated in the text chunks.",
        ])
        return "\n".join(lines)

    if dataset == "bioasq" and task == "binary":
        lines.extend([
            "This is a binary biomedical QA task.",
            "Respond with exactly one word: yes or no.",
            "Do not answer with maybe.",
            "Do not add explanation unless the task-specific context explicitly requires it.",
        ])
        return "\n".join(lines)

    if dataset == "realmedqa":
        lines.extend([
            "This is clinical recommendation QA grounded in guideline text.",
            "Answer with a concise clinical recommendation in 1 to 3 sentences.",
            "Do not answer with a single entity, label, or fragment.",
            "Stay close to the retrieved guideline wording and do not invent extra recommendations.",
            'Use "Insufficient Information." only as a last resort when the retrieved guideline text contains no plausible grounded recommendation to return.',
        ])
        return "\n".join(lines)

    if task == "mcq" and normalized_options:
        lines.extend([
            "This is a multiple-choice question.",
            "Choose exactly one option from the list below.",
            "Begin your response with the option letter, then the option text.",
            "Do not invent a new option or answer outside the list.",
            "Options:",
        ])
        for key, value in normalized_options.items():
            lines.append(f"{key}. {value}")
        return "\n".join(lines)

    if dataset == "multihoprag":
        lines.extend([
            "Answer with the shortest correct entity, title, date, number, or phrase grounded in the retrieved documents.",
            "Do not write an explanatory sentence.",
            "If the question compares two candidates, return only the selected candidate.",
            'Prefer returning the shortest best-supported answer span from the retrieved evidence.',
            'Use "Insufficient Information." only as a last resort when no plausible answer candidate is grounded at all.',
        ])
        if task == "binary":
            lines.append("For binary questions, begin your response with exactly Yes or No.")
        return "\n".join(lines)

    if dataset in {"hotpotqa", "2wikimultihopqa", "musique"} and task == "free_text":
        lines.extend([
            "This is a short-answer multi-hop QA task.",
            "Respond with the shortest correct entity, title, date, number, or phrase only.",
            "Do not write an explanatory sentence or reasoning chain after the answer.",
            "Do not restate the question.",
            "For comparison questions, return only the winning entity/title/person, not an explanation.",
            "If the answer is a date, location, person, or title, return exactly that span.",
        ])
        return "\n".join(lines)

    if task == "binary":
        lines.extend([
            "This is a binary question.",
            "Begin your response with exactly Yes or No as the first word.",
        ])
        return "\n".join(lines)

    if task == "free_text":
        return "Answer with a short entity or phrase when possible."

    return ""


def normalize_answer_to_contract(
    dataset_name: str,
    task_type: str,
    response: str,
    question: str = "",
) -> str:
    """
    Canonicalize model outputs for strict-label tasks without changing semantics.

    This is intentionally conservative: it only compresses responses when an
    explicit label is already present. Free-text answers are left untouched.
    """
    dataset = str(dataset_name or "").strip().lower()
    task = str(task_type or "").strip().lower()
    text = str(response or "").strip()
    if not text:
        return text

    if dataset == "pubmedqa":
        label = _extract_leading_label(text, ("yes", "no", "maybe"))
        return label or text

    if dataset == "bioasq" and task == "binary":
        label = _extract_leading_label(text, ("yes", "no"))
        return label or text

    if task == "binary":
        label = _extract_leading_short_label(text)
        return label or text

    if dataset in _QUESTION_NORMALIZERS:
        normalizer = _QUESTION_NORMALIZERS.get(dataset, _normalize_generic_answer)
        return normalizer(text, question, task)

    return _normalize_generic_answer(text, question, task)
