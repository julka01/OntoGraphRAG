"""Shared constants and helpers for the KG creator modules."""

import re
from typing import Optional

# Minimum character length for an entity name (after stripping whitespace).
# Single-character or two-character tokens are almost always stop-words or
# artefacts; named identifiers below this threshold (e.g. "p53", "IL-6") are
# kept via the allowlist below.
_ENTITY_MIN_NAME_LENGTH: int = 3

# Lowercase terms that must not become standalone KG nodes regardless of how
# the LLM classifies them.  These are generic process/state words that appear
# in almost every biomedical chunk and, when extracted as entities, become hub
# nodes fanning out to hundreds of irrelevant chunks during retrieval.
_GENERIC_HUB_ENTITY_BLOCKLIST: frozenset = frozenset({
    "treatment", "treatments", "condition", "conditions", "outcome", "outcomes",
    "model", "models", "effect", "effects", "result", "results", "factor",
    "factors", "mechanism", "mechanisms", "response", "responses", "function",
    "functions", "role", "roles", "process", "processes", "level", "levels",
    "system", "systems", "study", "studies", "analysis", "analyses",
    "approach", "approaches", "method", "methods", "measure", "measures",
    "group", "groups", "sample", "samples", "data", "finding", "findings",
    "evidence", "activity", "activities", "expression", "expressions",
    "production", "change", "changes", "increase", "increases",
    "decrease", "decreases", "type", "types", "form", "forms", "stage",
    "stages", "state", "states", "case", "cases", "patient", "patients",
    "subject", "subjects", "participant", "participants", "control",
    "controls", "target", "targets", "interaction", "interactions",
    "pathway", "pathways", "network", "networks", "signal", "signals",
    "marker", "markers", "indicator", "indicators", "test", "tests",
    "assessment", "assessments", "evaluation", "evaluations", "trial",
    "trials", "experiment", "experiments", "observation", "observations",
    "report", "reports", "review", "reviews", "analysis", "context",
    "information", "knowledge", "concept", "concepts", "feature", "features",
    "aspect", "aspects", "component", "components", "element", "elements",
    "structure", "structures", "property", "properties", "characteristic",
    "characteristics", "parameter", "parameters", "variable", "variables",
    "value", "values", "score", "scores", "rate", "rates", "ratio", "ratios",
    "index", "indices", "index", "mean", "median", "range", "prevalence",
    "incidence", "frequency", "proportion", "percentage", "number", "amount",
    "quantity", "duration", "period", "time", "age", "size", "dose",
    "concentration", "threshold", "limit", "maximum", "minimum",
    # Bare determiners/pronouns that sometimes slip through
    "this", "that", "these", "those", "which", "what", "who", "when",
    "where", "how", "why", "other", "another", "same", "different",
    "various", "several", "many", "few", "all", "both", "each", "every",
    "some", "any", "no", "not", "yes",
})


def _is_valid_entity_name(name: str) -> bool:
    """Return True when a name is specific enough to be a KG node.

    Rejects names that are:
    - shorter than _ENTITY_MIN_NAME_LENGTH (after stripping whitespace), unless
      they match the short-identifier pattern (digits/letters mixed, e.g. p53, IL-6)
    - in the generic hub-entity blocklist (case-insensitive, singular/plural)
    - bare numeric / punctuation fragments without any alphabetic referent
    """
    stripped = name.strip()
    if not stripped:
        return False
    normalized = stripped.lower()
    # Reject punctuation-only fragments or bare numeric tokens. Legitimate short
    # biomedical identifiers such as "p53" or "IL-6" include alphabetic
    # characters and therefore bypass this guard.
    if not re.search(r"[A-Za-z0-9]", stripped):
        return False
    if re.fullmatch(r"[\d\W_]+", stripped):
        return False
    # Blocklist check
    if normalized in _GENERIC_HUB_ENTITY_BLOCKLIST:
        return False
    # Length check: allow short named identifiers (alphanumeric + hyphens/dots)
    if len(stripped) < _ENTITY_MIN_NAME_LENGTH:
        # Allow if it looks like a named identifier: has a digit or a hyphen
        # (e.g. "p53" is 3 chars and passes length anyway; "IL" is 2 and fails
        # unless it has a hyphen such as "IL-6").
        import re as _re
        if not _re.search(r'[\d\-\.]', stripped):
            return False
    return True


def _env_flag(value: Optional[str], default: bool) -> bool:
    """Parse a boolean-like environment value safely."""
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
