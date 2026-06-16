"""
Uncertainty Metrics for Hallucination Detection - 7 Metrics by Family

Entropy-based:
  - Semantic Entropy          (Farquhar et al., Nature 2024 / Kuhn et al., ICLR 2023)
  - Discrete Semantic Entropy (same paper)

Calibration-based:
  - P(True)  (Farquhar et al., self-consistency via NLI clustering — fraction of
               samples in the same semantic cluster as the most probable response.
               Approximates original which reads log P("A") from the model itself.)

Similarity-based:
  - Embedding Consistency / SelfCheckGPT  (Manakul et al., EMNLP 2023)

Perturbation-based:
  - SRE-UQ  (Vipulanandan, Premaratne & Sarkar, arXiv 2601.20026, ICLR 2026)
             Quantum Tensor Network method: treats the empirical token-probability
             distribution as an RKHS wave function and applies first-order quantum
             perturbation theory to measure local sensitivity of the kernel mean
             embedding. Returned under key "sre_uq".

Geometric-based (novel — this work):
  - VN-Entropy  von Neumann entropy of the response Gram matrix.
                Soft, parameter-free analogue of semantic entropy: no NLI model,
                no clustering threshold. Returned under key "vn_entropy".

  - SD-UQ       Differential entropy of the response distribution in the
                question-orthogonal embedding subspace. Conditions out the
                "trivial" question direction, measuring only irreducible
                inter-response spread. Returned under key "sd_uq".

Structural-based (novel — this work):
  - Graph Path  Does the knowledge graph support the generated answer?
    Support     Finds question entities and answer entities in the KG, then
                checks whether any path exists between them (≤ 3 hops).
                Uncertainty = 1 - (fraction of answer entities reachable).
                Does NOT require multiple LLM samples — immune to graph-induced
                overconfidence because it measures graph structure, not LLM
                output variance. Returned under key "graph_path_support".

  - Graph Path  How many distinct candidate answers does the KG suggest?
    Disagreement Walks max_hops from question entities, computes entropy over
                the terminal neighbor entity distribution. Low entropy = KG
                points unambiguously at one answer. High entropy = KG fans out
                to many candidate answers = structurally uncertain. Does NOT
                require the generated answer — purely question-driven structural
                ambiguity. Returned under key "graph_path_disagreement".

  - Competing   How many same-type entities are reachable from question
    Answer       entities via the same relation type used in the generated
    Alternatives answer? Formalises epistemic uncertainty as cardinality of
                the set of plausible alternative correct answers in the KG.
                High count = answer is one of many equally-valid alternatives
                = the KG does not uniquely determine the answer = uncertain.
                Novel: no prior UQ paper uses typed-relation cardinality as
                an uncertainty signal. Returned under key
                "competing_answer_alternatives".
"""

import logging
import re
import threading
import time
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SENTENCE_TRANSFORMER_CACHE: Dict[str, Any] = {}
_SENTENCE_TRANSFORMER_CACHE_LOCK = threading.Lock()

_NLI_MODEL_CACHE: Dict[str, Any] = {}
_NLI_TOKENIZER_CACHE: Dict[str, Any] = {}
_NLI_CACHE_LOCK = threading.Lock()


def _question_local_entity_support_predicate(
    entity_var: str,
    *,
    kg_name: Optional[str],
    question_id: Optional[str],
) -> str:
    """Require an entity to be supported inside the active KG/question scope."""
    conditions: List[str] = []
    if question_id:
        conditions.append("c.questionId = $question_id")
    if kg_name:
        conditions.append("d.kgName = $kg_name")
    where_clause = " AND ".join(conditions) if conditions else "true"
    return f"""EXISTS {{
        MATCH ({entity_var})<-[:MENTIONS|HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
        WHERE {where_clause}
    }}"""


def _question_local_pair_support_predicate(
    left_var: str,
    right_var: str,
    *,
    kg_name: Optional[str],
    question_id: Optional[str],
    relationship_var: Optional[str] = None,
) -> str:
    """Require two entities to be jointly supported inside one question bundle."""
    if not question_id:
        return "true"

    kg_filters = ""
    if kg_name:
        kg_filters = """
          AND d1.kgName = $kg_name
          AND d2.kgName = $kg_name"""

    direct_edge_scope = ""
    if relationship_var:
        direct_edge_scope = (
            f"$question_id IN coalesce({relationship_var}.questionIds, []) OR "
        )

    return f"""{direct_edge_scope}EXISTS {{
        MATCH ({left_var})<-[:MENTIONS|HAS_ENTITY]-(c1:Chunk)-[:PART_OF]->(d1:Document)
        MATCH ({right_var})<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document)
        WHERE c1.questionId = $question_id
          AND c2.questionId = $question_id
          AND coalesce(c1.passageIndex, -1) = coalesce(c2.passageIndex, -1)
          AND abs(
                coalesce(c1.chunkLocalIndex, c1.position)
                - coalesce(c2.chunkLocalIndex, c2.position)
              ) <= 1
              {kg_filters}
    }}"""


def _question_local_path_support_predicate(
    *,
    path_var: str,
    kg_name: Optional[str],
    question_id: Optional[str],
) -> str:
    """Require every hop in a path to stay inside one question bundle."""
    if not question_id:
        return ""

    kg_filters = ""
    if kg_name:
        kg_filters = """
                  AND d1.kgName = $kg_name
                  AND d2.kgName = $kg_name"""

    return f"""
          AND ALL(idx IN range(0, length({path_var}) - 1) WHERE (
                $question_id IN coalesce(relationships({path_var})[idx].questionIds, [])
                OR EXISTS {{
                    WITH nodes({path_var})[idx] AS a, nodes({path_var})[idx + 1] AS b
                    MATCH (a)<-[:MENTIONS|HAS_ENTITY]-(c1:Chunk)-[:PART_OF]->(d1:Document)
                    MATCH (b)<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document)
                    WHERE c1.questionId = $question_id
                      AND c2.questionId = $question_id
                      AND coalesce(c1.passageIndex, -1) = coalesce(c2.passageIndex, -1)
                      AND abs(
                            coalesce(c1.chunkLocalIndex, c1.position)
                            - coalesce(c2.chunkLocalIndex, c2.position)
                          ) <= 1
                          {kg_filters}
                }}
          ))"""


def _normalize_structural_text(value: Any) -> str:
    """Normalize text for conservative entity mention matching."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _candidate_structural_spans(
    text: str,
    *,
    min_name_length: int,
    max_tokens: int = 8,
    max_spans: int = 128,
) -> List[str]:
    """Generate normalized contiguous spans for lightweight entity matching."""
    normalized = _normalize_structural_text(text)
    if not normalized:
        return []

    tokens = normalized.split()
    spans = {normalized} if len(normalized) >= min_name_length else set()
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + max_tokens) + 1):
            span = " ".join(tokens[start:end]).strip()
            if len(span) >= min_name_length:
                spans.add(span)
            if len(spans) >= max_spans:
                break
        if len(spans) >= max_spans:
            break
    return sorted(spans, key=lambda span: (-len(span.split()), -len(span), span))


def _entity_surface_forms(row: Dict[str, Any], *, min_name_length: int) -> List[str]:
    """Return normalized names / aliases / synonyms worth matching."""
    surface_forms = set()
    for key in ("name",):
        normalized = _normalize_structural_text(row.get(key, ""))
        if len(normalized) >= min_name_length:
            surface_forms.add(normalized)
    for key in ("aliases", "original_ids", "synonyms"):
        values = row.get(key) or []
        if not isinstance(values, list):
            continue
        for value in values:
            normalized = _normalize_structural_text(value)
            if len(normalized) >= min_name_length:
                surface_forms.add(normalized)
    return sorted(surface_forms, key=lambda value: (-len(value.split()), -len(value), value))


def _entity_match_score(
    *,
    text_normalized: str,
    candidate_spans: List[str],
    surface_forms: List[str],
) -> float:
    """Score how plausibly an entity is mentioned in text."""
    if not text_normalized or not candidate_spans or not surface_forms:
        return 0.0

    candidate_span_set = set(candidate_spans)
    best_score = 0.0
    for surface_form in surface_forms:
        if surface_form in candidate_span_set:
            return 5.0 + min(len(surface_form.split()), 8) * 0.2 + min(len(surface_form), 64) / 64.0
        if surface_form and surface_form in text_normalized:
            best_score = max(
                best_score,
                4.0 + min(len(surface_form.split()), 8) * 0.2 + min(len(surface_form), 64) / 64.0,
            )
            continue
        for span in candidate_spans:
            if abs(len(span.split()) - len(surface_form.split())) > 1:
                continue
            ratio = SequenceMatcher(None, span, surface_form).ratio()
            if ratio >= 0.92:
                best_score = max(best_score, 3.0 + ratio)
                break
    return best_score


def _load_scoped_entities(
    session: Any,
    *,
    entity_scope: str,
    kg_name: Optional[str],
    question_id: Optional[str],
    min_name_length: int,
    limit: int = 4000,
) -> List[Dict[str, Any]]:
    """Fetch entities visible inside the active KG/question scope."""
    scoped_entity_query = f"""
    MATCH (e:__Entity__)
    WHERE size(e.name) >= $min_len
      AND {entity_scope}
    RETURN DISTINCT e.id AS id,
           e.name AS name,
           coalesce(e.aliases, []) AS aliases,
           coalesce(e.original_ids, []) AS original_ids,
           coalesce(e.synonyms, []) AS synonyms
    LIMIT {int(limit)}
    """
    params = {"min_len": min_name_length}
    if kg_name:
        params["kg_name"] = kg_name
    if question_id:
        params["question_id"] = question_id
    return [dict(row) for row in session.run(scoped_entity_query, params)]


def _match_scoped_entities_to_text(
    text: str,
    scoped_entities: List[Dict[str, Any]],
    *,
    min_name_length: int,
    max_entities: int = 5,
) -> List[Dict[str, Any]]:
    """Select the most plausible KG entities mentioned in text."""
    text_normalized = _normalize_structural_text(text)
    candidate_spans = _candidate_structural_spans(
        text_normalized,
        min_name_length=min_name_length,
    )
    if not candidate_spans:
        return []

    scored: List[tuple] = []
    for row in scoped_entities:
        surface_forms = _entity_surface_forms(row, min_name_length=min_name_length)
        score = _entity_match_score(
            text_normalized=text_normalized,
            candidate_spans=candidate_spans,
            surface_forms=surface_forms,
        )
        if score <= 0.0:
            continue
        scored.append((score, len(surface_forms[0]) if surface_forms else 0, str(row.get("id", "")), row))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    matched: List[Dict[str, Any]] = []
    seen_ids = set()
    for _, _, entity_id, row in scored:
        if entity_id in seen_ids:
            continue
        matched.append(row)
        seen_ids.add(entity_id)
        if len(matched) >= max_entities:
            break
    return matched


def _primary_answer_span(answer: str) -> str:
    """Use the first sentence / first 150 chars to avoid explanatory spillover."""
    first_sentence = str(answer or "").split(".")[0].strip()
    if first_sentence and len(first_sentence) <= 150:
        return first_sentence
    return str(answer or "")[:150]


def _resolve_structural_entities(
    session: Any,
    *,
    question: str,
    answer: str,
    entity_scope: str,
    kg_name: Optional[str],
    question_id: Optional[str],
    min_name_length: int,
) -> Dict[str, Any]:
    """Resolve question / answer entities and retain non-self support sources."""
    scoped_entities = _load_scoped_entities(
        session,
        entity_scope=entity_scope,
        kg_name=kg_name,
        question_id=question_id,
        min_name_length=min_name_length,
    )
    if not scoped_entities:
        return {"null_reason": "no_q_entities"}

    question_entities = _match_scoped_entities_to_text(
        question,
        scoped_entities,
        min_name_length=min_name_length,
    )
    if not question_entities:
        return {"null_reason": "no_q_entities"}

    answer_entities = _match_scoped_entities_to_text(
        _primary_answer_span(answer),
        scoped_entities,
        min_name_length=min_name_length,
    )
    if not answer_entities:
        return {"null_reason": "no_a_entities"}

    question_entity_ids = [str(row.get("id")) for row in question_entities if row.get("id")]
    if not question_entity_ids:
        return {"null_reason": "no_q_entities"}

    effective_answer_ids: List[str] = []
    question_sources_by_answer: Dict[str, List[str]] = {}
    for row in answer_entities:
        answer_id = str(row.get("id"))
        if not answer_id:
            continue
        source_q_ids = [q_id for q_id in question_entity_ids if q_id != answer_id]
        if not source_q_ids:
            continue
        effective_answer_ids.append(answer_id)
        question_sources_by_answer[answer_id] = source_q_ids

    if not effective_answer_ids:
        return {"null_reason": "trivial_overlap"}

    return {
        "null_reason": None,
        "question_entity_ids": question_entity_ids,
        "answer_entity_ids": effective_answer_ids,
        "question_sources_by_answer": question_sources_by_answer,
    }


def _safe_prob_entropy(probabilities: np.ndarray) -> float:
    """Numerically stable entropy in nats — matches Farquhar et al. cluster_assignment_entropy."""
    if probabilities is None:
        return 0.0
    p = np.array(probabilities, dtype=float)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    return float(-np.sum(p * np.log(p)))


def _cluster_responses_by_nli(
    responses: List[str],
    nli_model: str = "microsoft/deberta-large-mnli",
    strict_entailment: bool = False,
    question: Optional[str] = None,
) -> List[int]:
    """
    NLI-based semantic clustering — Farquhar et al. (Nature 2024).

    Default: microsoft/deberta-large-mnli (~900MB, cached after first use).
    Paper-faithful: microsoft/deberta-v2-xlarge-mnli (requires sentencepiece, 1.5GB).
    Both significantly outperform roberta-large-mnli on NLI benchmarks.
    Falls back to roberta-large-mnli if DeBERTa unavailable.

    question (condition_on_question): When provided, each response is prefixed with
        the question text before NLI entailment checking, matching the paper's
        `condition_on_question` flag. This anchors ambiguous short answers ("Yes",
        "No") to the specific question context, significantly improving clustering
        quality for factoid and yes/no questions.

    Equivalence rule (strict_entailment=False, the paper default):
        A ~ B  iff  implication(A→B) ≠ contradiction
                AND implication(B→A) ≠ contradiction
                AND NOT (both neutral)
    i.e. at least one direction entails and neither contradicts.

    strict_entailment=True: both directions must be entailment (class 2).

    Algorithm: greedy — iterate responses; assign to first cluster whose
    representative is equivalent; otherwise open a new cluster.
    Falls back to lexical clustering when transformers unavailable.
    """
    if not responses:
        return []

    # Condition on question: prepend question to each response before NLI
    # (Farquhar et al., compute_uncertainty_measures.py, condition_on_question flag)
    if question:
        nli_inputs = [f"{question} {r}" for r in responses]
    else:
        nli_inputs = responses

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not hasattr(_cluster_responses_by_nli, "_model") or \
                _cluster_responses_by_nli._model_name != nli_model:
            _cluster_responses_by_nli._tokenizer = AutoTokenizer.from_pretrained(nli_model)
            _cluster_responses_by_nli._model = AutoModelForSequenceClassification.from_pretrained(nli_model)
            _cluster_responses_by_nli._model_name = nli_model

        tokenizer = _cluster_responses_by_nli._tokenizer
        model     = _cluster_responses_by_nli._model

        def _implication(text1: str, text2: str) -> int:
            """Return 0=contradiction, 1=neutral, 2=entailment (DeBERTa convention)."""
            inputs = tokenizer(
                text1, text2, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            return int(torch.argmax(F.softmax(logits, dim=1)).item())

        def _are_equivalent(t1: str, t2: str) -> bool:
            imp1 = _implication(t1, t2)
            imp2 = _implication(t2, t1)
            if strict_entailment:
                return imp1 == 2 and imp2 == 2
            # Non-strict (paper default): no contradiction, not both neutral
            return 0 not in (imp1, imp2) and [imp1, imp2] != [1, 1]

        cluster_ids: List[int] = [-1] * len(nli_inputs)
        next_cluster = 0

        for i, inp_i in enumerate(nli_inputs):
            if cluster_ids[i] != -1:
                continue
            cluster_ids[i] = next_cluster
            for j in range(i + 1, len(nli_inputs)):
                if cluster_ids[j] == -1 and _are_equivalent(inp_i, nli_inputs[j]):
                    cluster_ids[j] = next_cluster
            next_cluster += 1

        return cluster_ids

    except Exception as e:
        logger.warning(f"NLI clustering failed ({e}), falling back to lexical clustering")
        return _cluster_responses_by_lexical_similarity(responses)


def _cluster_responses_by_lexical_similarity(
    responses: List[str],
    similarity_threshold: float = 0.45,
) -> List[int]:
    """Jaccard-based clustering — fallback when NLI model unavailable."""
    if not responses:
        return []

    token_sets = []
    for r in responses:
        text = str(r or "").lower()
        tokens = {tok for tok in text.split() if tok}
        token_sets.append(tokens)

    cluster_ids = [-1] * len(responses)
    next_cluster = 0

    for i in range(len(responses)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_cluster

        for j in range(i + 1, len(responses)):
            if cluster_ids[j] != -1:
                continue

            a = token_sets[i]
            b = token_sets[j]
            if not a and not b:
                sim = 1.0
            else:
                union = len(a | b)
                sim = (len(a & b) / union) if union else 0.0

            if sim >= similarity_threshold:
                cluster_ids[j] = next_cluster

        next_cluster += 1

    return cluster_ids

# =============================================================================
# ENTROPY FAMILY
# =============================================================================

def compute_semantic_entropy(
    responses: List[str],
    cluster_ids: List[int],
    log_likelihoods: Optional[np.ndarray] = None,
) -> float:
    """Semantic Entropy — Farquhar et al. (Nature 2024).

    When log_likelihoods are provided (real model token log-probs), computes
    the proper logsumexp-by-cluster probability mass and returns entropy in nats.

    When log_likelihoods are absent, falls back to cluster-count proportions
    (equivalent to discrete_semantic_entropy). This fallback is used when only
    pseudo-probabilities derived from response frequency are available.
    """
    if not responses or not cluster_ids:
        return 0.0

    unique_ids = sorted(set(cluster_ids))

    if log_likelihoods is not None and len(log_likelihoods) == len(cluster_ids):
        ll = np.array(log_likelihoods, dtype=float)
        # Normalise: log p(x_i) = log_ll_i - logsumexp(all log_ll)
        log_total = np.log(np.sum(np.exp(ll - ll.max()))) + ll.max()  # stable logsumexp
        log_cluster_probs = []
        for uid in unique_ids:
            indices = [k for k, x in enumerate(cluster_ids) if x == uid]
            ll_cluster = ll[indices]
            # log p(cluster) = logsumexp(log p(x_i) for i in cluster)
            lse = np.log(np.sum(np.exp(ll_cluster - ll_cluster.max()))) + ll_cluster.max()
            log_cluster_probs.append(lse - log_total)
        log_p = np.array(log_cluster_probs)
        p = np.exp(log_p)
        # H = -Σ p log p  (nats)
        return float(-np.sum(p * log_p))

    # Fallback: uniform weights → same as discrete_semantic_entropy
    n = len(responses)
    counts = np.array([cluster_ids.count(uid) for uid in unique_ids])
    probabilities = counts / n
    return _safe_prob_entropy(probabilities)


def compute_discrete_semantic_entropy(cluster_ids: List[int]) -> float:
    """Discrete Semantic Entropy - Cluster assignment entropy only."""
    if not cluster_ids:
        return 0.0
    n = len(cluster_ids)
    counts = np.bincount(cluster_ids)
    probabilities = counts / n
    return _safe_prob_entropy(probabilities)


# =============================================================================
# CALIBRATION FAMILY
# =============================================================================

def compute_p_true(
    responses: List[str],
    context: str,
    question: str,
    precomputed_cluster_ids: Optional[List[int]] = None,
) -> float:
    """
    P(True) — Farquhar et al. (Nature 2024).

    Measures self-consistency: fraction of sampled responses that are
    semantically consistent with the most probable response (responses[0]).

    Original method uses the model's own log P("A") on a few-shot prompt
    asking "Is this answer true?" — requires white-box log-prob access.

    This implementation uses NLI-based consistency as a faithful approximation:
    fraction of responses in the same NLI cluster as responses[0].
    Higher = more self-consistent = more certain.

    Pass precomputed_cluster_ids (from _cluster_responses_by_nli) to avoid
    running NLI a second time when clusters are already computed for
    semantic_entropy / discrete_semantic_entropy.

    When <2 responses available, returns 0.5 (maximum uncertainty / undefined).
    """
    if not responses or len(responses) < 2:
        return 0.5

    try:
        cluster_ids = (
            precomputed_cluster_ids
            if precomputed_cluster_ids is not None
            else _cluster_responses_by_nli(responses)
        )
        target_cluster = cluster_ids[0]
        consistent = sum(1 for cid in cluster_ids if cid == target_cluster)
        return consistent / len(responses)
    except Exception as e:
        logger.warning(f"P(True) NLI consistency failed: {e}")
        return 0.5


# =============================================================================
# SIMILARITY FAMILY
# =============================================================================

def compute_embedding_consistency(
    responses: List[str],
    nli_model: str = "roberta-large-mnli",
    max_pairs: int = 50
) -> float:
    """Embedding Consistency - Pairwise NLI consistency (SelfCheckGPT)."""
    if len(responses) < 2:
        return 0.0
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        if not hasattr(compute_embedding_consistency, '_model'):
            compute_embedding_consistency._tokenizer = AutoTokenizer.from_pretrained(nli_model)
            compute_embedding_consistency._model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        
        tokenizer = compute_embedding_consistency._tokenizer
        model = compute_embedding_consistency._model
        
        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if len(pairs) >= max_pairs:
                    break
                pairs.append((i, j))
        
        contradiction_count = 0
        for i, j in pairs:
            r1, r2 = responses[i], responses[j]
            for text1, text2 in [(r1, r2), (r2, r1)]:
                inputs = tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    if torch.argmax(probs[0]).item() == 0:  # contradiction
                        contradiction_count += 1
        
        total = len(pairs) * 2
        return contradiction_count / total if total > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Embedding consistency failed: {e}")
        return 0.5


# =============================================================================
# PERTURBATION FAMILY
# =============================================================================

def compute_spuq(
    responses: List[str],
    log_likelihoods: Optional[np.ndarray] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_modes: int = 8,
) -> float:
    """
    SRE-UQ — Vipulanandan, Premaratne & Sarkar (arXiv 2601.20026, ICLR 2026).

    Treats the empirical response distribution as an RKHS wave function via
    kernel mean embedding (KME) and applies first-order quantum perturbation
    theory to measure how sensitive that wave function is to small perturbations
    of the Hamiltonian.  High sensitivity = high uncertainty.

    Implementation:
      1. Encode each response as a d-dimensional embedding.
      2. KME = mean embedding (the "wave function" evaluated at its centre).
      3. Gaussian kernel bandwidth σ = std of distances to the KME.
      4. For each response r:
           κ_r  = exp(−dist_r² / 2σ²)           (kernel value / amplitude)
           E_r  = κ_r − mean(κ)                   (first-order energy correction)
           L_r  = (σ²/2) · (dist_r²/σ⁴ − 1/σ²)  (Laplacian term, curvature of ψ)
           UQ_r = |E_r + L_r|                     (perturbation sensitivity per response)
      5. Average over up to num_modes responses (highest-amplitude modes first).

    When log_likelihoods are available they weight the kernel amplitudes;
    otherwise uniform weights are used (no degradation — uniform is the
    maximum-entropy prior).
    """
    if len(responses) < 2:
        return 0.0

    try:
        model = _get_or_load_sentence_transformer(embedding_model)
        embeddings = model.encode(responses, show_progress_bar=False)  # (R, d)
        embeddings = np.array(embeddings, dtype=float)

        # Kernel mean embedding (wave function centre)
        if log_likelihoods is not None:
            probs = np.exp(log_likelihoods)
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(responses)) / len(responses)

        kme = (embeddings * probs[:, None]).sum(axis=0)  # weighted mean, (d,)

        # Gaussian kernel bandwidth
        dists = np.linalg.norm(embeddings - kme, axis=1)  # (R,)
        sigma = np.std(dists)
        if sigma < 1e-6:
            return 0.0  # all responses identical → no uncertainty
        sigma = sigma + 1e-8

        # Kernel amplitudes
        k_vals = np.exp(-dists ** 2 / (2 * sigma ** 2))  # (R,)

        # First-order energy corrections
        energy_correction = k_vals - (k_vals * probs).sum()  # (R,)

        # Laplacian of Gaussian kernel (second-order curvature of wave function)
        laplacian = (sigma ** 2 / 2) * (dists ** 2 / sigma ** 4 - 1.0 / sigma ** 2)  # (R,)

        # Per-response perturbation sensitivity
        uq_per_response = np.abs(energy_correction + laplacian)  # (R,)

        # Average over the top num_modes modes (highest amplitude first)
        order = np.argsort(-k_vals)
        top_modes = order[:num_modes]
        score = float(uq_per_response[top_modes].mean())

        return score

    except ImportError:
        return 0.0
    except Exception as e:
        logger.warning(f"SRE-UQ failed: {e}")
        return 0.0


# =============================================================================
# GEOMETRIC FAMILY (novel — this work)
# =============================================================================

def compute_von_neumann_entropy(
    responses: List[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """
    VN-Entropy — Von Neumann entropy of the response Gram matrix.

    FORMULATION
    -----------
    Given N response embeddings V ∈ ℝ^{N×d} (L2-normalised), the Gram matrix
    G = VVᵀ ∈ ℝ^{N×N} has G_{ij} = cos(v_i, v_j) ∈ [−1, 1].

    Normalise to a density matrix:  ρ = G / tr(G) = G / N  (tr = N since ‖vᵢ‖=1).

    Von Neumann entropy:
        S(ρ) = −tr(ρ log ρ) = −∑ᵢ λᵢ log λᵢ

    where λᵢ = eigenvalue(G)/N ≥ 0 and ∑ λᵢ = 1.

    INTERPRETATION
    --------------
    S(ρ) = 0   → all responses identical (rank-1 Gram matrix, λ₁ = 1).
    S(ρ) → log N → responses mutually orthogonal (maximally diverse / uncertain).

    This is a soft, parameter-free analogue of semantic entropy:
    - No NLI clustering step → no threshold hyperparameter
    - Captures continuous similarity structure instead of hard cluster boundaries
    - Related to SRE-UQ (Vipulanandan): both operate on quantum-style density matrices
      of response embeddings; VN-Entropy uses the response Gram matrix directly whereas
      SRE-UQ applies perturbation theory to the kernel mean embedding.

    PROGRESSION (this work)
    -----------------------
    semantic_entropy  — hard NLI clusters in meaning space
    vn_entropy        — soft Gram-matrix density, full embedding space
    sd_uq             — soft Gram-matrix density, question-orthogonal subspace only

    REFERENCES
    ----------
    - Nielsen & Chuang (2010) — Quantum Computation and Quantum Information, §11.3.
    - Vipulanandan et al. (ICLR 2026) — SRE-UQ uses quantum density matrices for UQ.
    - Farquhar et al. (Nature 2024) — semantic entropy as the discrete predecessor.
    """
    if not responses or len(responses) < 2:
        return 0.0

    try:
        model = _get_or_load_sentence_transformer(embedding_model)
        V = model.encode(list(responses), show_progress_bar=False, normalize_embeddings=True)
        N = len(V)

        # Gram matrix of cosine similarities (all positive because normalised)
        G = V @ V.T  # (N, N)

        # Density matrix: ρ = G / N  (tr(G) = N since ‖vᵢ‖ = 1)
        eigenvalues = np.linalg.eigvalsh(G) / N  # (N,) ascending

        # Clip numerical noise (eigenvalues should be ≥ 0)
        eigenvalues = np.clip(eigenvalues, 0, None)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]  # drop zero modes

        if len(eigenvalues) == 0:
            return 0.0

        # Von Neumann entropy: S = -∑ λ log λ  (nats)
        s = -float(np.sum(eigenvalues * np.log(eigenvalues)))
        return max(s, 0.0)

    except ImportError:
        return 0.0
    except Exception as e:
        logger.warning(f"VN-Entropy failed: {e}")
        return 0.0


def compute_sre_uq(
    prompt: str,
    responses: List[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_components: int = 8,
) -> float:
    """
    SD-UQ — Semantic Drift UQ (novel — this work).

    Information-theoretic uncertainty measure based on the differential entropy
    of the response distribution in the question-orthogonal embedding subspace.

    FORMULATION
    -----------
    For a fixed question with unit embedding q, decompose each response embedding v_i as:

        v_i = (v_i · q) q  +  e_i                  # parallel + orthogonal components
        e_i = v_i − (v_i · q) q                     # question-orthogonal residual

    The residuals {e_i} represent semantic content each response adds *beyond*
    the question direction.  Model them as Gaussian:

        P_E ≈ N(μ_E, Σ_E)

    The differential entropy of this Gaussian is:

        H(E) = ½ log det(2πe Σ_E)

    which equals, up to constants, the log-volume of semantic space explored by
    the residual distribution.  We estimate this via the entropy power — the
    geometric mean of the top-k eigenvalues of the empirical residual covariance:

        SD-UQ = exp( (1/k) Σᵢ log(λᵢ + ε) )   ∝  exp( H(E) / k )

    where λ₁ ≥ … ≥ λ_k come from a thin SVD of the centred residual matrix
    (k = min(N−1, n_components) to avoid rank deficiency with small N).

    INFORMATION-THEORETIC INTERPRETATION
    -------------------------------------
    SD-UQ estimates H(v | q-direction) — the conditional entropy of the response
    distribution given the question direction.  This is the complement of the
    mutual information between the question and the response:

        I(q; v) = H(v) − H(v | q-direction) = H(v) − H(E)

    A certain model produces responses that share a consistent question-orthogonal
    signature → low H(E) → low SD-UQ → high MI(q; v).
    An uncertain / hallucinating model produces responses that scatter in the
    orthogonal subspace → high H(E) → high SD-UQ → low MI(q; v).

    ANCHOR MODES
    ------------
    When prompt is provided:   anchor = embed(prompt)        — QA / conditional setting.
    When prompt is None/empty: anchor = normalize(mean(V))   — anchor-free setting.
      In the anchor-free mode the metric reduces to the entropy of the response
      distribution in the subspace orthogonal to their own mean direction, i.e. a
      pure inter-response consistency measure that works in any generation setting
      (summarisation, dialogue, code generation, translation, …).

    PROPERTIES
    ----------
    - Length-invariant: embeddings are L2-normalised before projection.
    - No concatenation: anchor and responses are encoded as independent inputs.
    - Returns 0 when all responses are identical (Σ_E = 0 → H = −∞ → exp = 0).
    - Numerically stable via thin SVD; robust to N ≪ d via rank-k truncation.
    - Task-agnostic: prompt may be None for anchor-free uncertainty estimation.

    REFERENCES
    ----------
    - Reimers & Gurevych (EMNLP 2019) — Sentence-BERT: embedding space geometry.
    - Cover & Thomas (2006) — Elements of Information Theory, §8: differential entropy.
    - Kuhn et al. (ICLR 2023) — inter-response consistency as an uncertainty signal.
    """
    if not responses or len(responses) < 2:
        return 0.0

    try:
        model = _get_or_load_sentence_transformer(embedding_model)

        # Encode responses; optionally encode the prompt as anchor
        V = model.encode(list(responses), show_progress_bar=False, normalize_embeddings=True)
        N = len(V)

        if prompt:
            # QA / conditional mode: anchor = question embedding
            q = model.encode([prompt], show_progress_bar=False, normalize_embeddings=True)[0]
        else:
            # Anchor-free mode: anchor = normalised mean response embedding
            mean_v = V.mean(axis=0)
            norm = np.linalg.norm(mean_v)
            q = mean_v / norm if norm > 1e-10 else V[0]

        # Gram-Schmidt: project out the anchor direction from each response
        proj_coeffs = V @ q                       # (N,)
        E = V - np.outer(proj_coeffs, q)          # (N, d) orthogonal residuals

        # Centre residuals (removes the mean residual direction)
        E_centred = E - E.mean(axis=0)            # (N, d)

        # Thin SVD of centred residual matrix: E_centred = U S Vᵀ
        # Covariance eigenvalues: λᵢ = sᵢ² / N
        k = min(N - 1, n_components)
        if k < 1:
            return 0.0

        _, s, _ = np.linalg.svd(E_centred / np.sqrt(N), full_matrices=False)
        eigenvalues = s[:k] ** 2                  # top-k eigenvalues of Σ_E

        # Entropy power: geometric mean of eigenvalues ∝ exp(H(E) / k)
        eps = 1e-12
        log_geomean = float(np.mean(np.log(eigenvalues + eps)))
        return float(np.exp(log_geomean))

    except ImportError:
        return 0.0
    except Exception as e:
        logger.warning(f"SD-UQ failed: {e}")
        return 0.0


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compute_all_uncertainty_metrics(
    responses: List[str],
    prompt: str,
    context: Optional[str] = None,
    log_likelihoods: Optional[np.ndarray] = None,
    hidden_states: Optional[List[np.ndarray]] = None,
    cluster_ids: Optional[List[int]] = None,
    train_embeddings: Optional[List[np.ndarray]] = None,
    train_labels: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute all 7 uncertainty metrics organized by family.

    Returns:
        Dictionary with metrics organized by family:
        - entropy_family: semantic_entropy, discrete_semantic_entropy
        - calibration_family: p_true
        - similarity_family: selfcheckgpt
        - perturbation_family: sre_uq
        - geometric_family: vn_entropy, sd_uq
    """
    results = {}

    # Cluster responses using NLI-based bidirectional entailment (Farquhar et al.)
    # condition_on_question: prepend the question to each response before NLI
    # (matches paper's condition_on_question flag — anchors ambiguous short answers).
    # Falls back to lexical Jaccard clustering when transformers unavailable.
    effective_cluster_ids = cluster_ids
    if not effective_cluster_ids and responses and len(responses) >= 2:
        effective_cluster_ids = _cluster_responses_by_nli(responses, question=prompt)

    # Build a pseudo log-likelihood vector from response frequency as fallback.
    # Used by sre_uq to weight kernel amplitudes when true log-probs are unavailable.
    effective_log_likelihoods = log_likelihoods
    if effective_log_likelihoods is None and responses:
        counts = {}
        for r in responses:
            key = str(r or "").strip().lower()
            counts[key] = counts.get(key, 0) + 1
        pseudo_probs = np.array([counts.get(str(r or "").strip().lower(), 1) for r in responses], dtype=float)
        pseudo_probs = pseudo_probs / pseudo_probs.sum() if pseudo_probs.sum() > 0 else np.ones(len(responses)) / max(1, len(responses))
        effective_log_likelihoods = np.log(np.clip(pseudo_probs, 1e-12, 1.0))
    
    compute_times: Dict[str, float] = {}

    # ENTROPY FAMILY
    _t0 = time.perf_counter()
    results["semantic_entropy"] = compute_semantic_entropy(responses, effective_cluster_ids, effective_log_likelihoods) if effective_cluster_ids else 0.0
    compute_times["semantic_entropy"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    results["discrete_semantic_entropy"] = compute_discrete_semantic_entropy(effective_cluster_ids) if effective_cluster_ids else 0.0
    compute_times["discrete_semantic_entropy"] = time.perf_counter() - _t0

    # CALIBRATION FAMILY
    _t0 = time.perf_counter()
    results["p_true"] = compute_p_true(
        responses, context, prompt,
        precomputed_cluster_ids=effective_cluster_ids,
    ) if len(responses) >= 2 else 0.5
    compute_times["p_true"] = time.perf_counter() - _t0

    # SIMILARITY FAMILY
    _t0 = time.perf_counter()
    results["selfcheckgpt"] = compute_embedding_consistency(responses) if len(responses) >= 2 else 1.0
    compute_times["selfcheckgpt"] = time.perf_counter() - _t0

    # PERTURBATION FAMILY (Vipulanandan et al., ICLR 2026)
    _t0 = time.perf_counter()
    results["sre_uq"] = compute_spuq(responses, effective_log_likelihoods) if len(responses) >= 2 else 0.0
    compute_times["sre_uq"] = time.perf_counter() - _t0

    # GEOMETRIC FAMILY — novel (this work)
    _t0 = time.perf_counter()
    results["vn_entropy"] = compute_von_neumann_entropy(responses) if len(responses) >= 2 else 0.0
    compute_times["vn_entropy"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    results["sd_uq"] = compute_sre_uq(prompt, responses) if responses else 0.0
    compute_times["sd_uq"] = time.perf_counter() - _t0

    results["compute_times"] = compute_times
    return results


class UncertaintyEvaluator:
    """Unified interface for all 8 uncertainty metrics."""
    
    def evaluate(
        self,
        responses: List[str],
        prompt: str,
        context: Optional[str] = None,
        log_likelihoods: Optional[np.ndarray] = None,
        hidden_states: Optional[List[np.ndarray]] = None,
        cluster_ids: Optional[List[int]] = None,
        train_embeddings: Optional[List[np.ndarray]] = None,
        train_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Evaluate all metrics organized by family."""
        metrics = compute_all_uncertainty_metrics(
            responses=responses,
            prompt=prompt,
            context=context,
            log_likelihoods=log_likelihoods,
            hidden_states=hidden_states,
            cluster_ids=cluster_ids,
            train_embeddings=train_embeddings,
            train_labels=train_labels
        )
        
        return {
            "entropy_family": {
                "semantic_entropy": metrics["semantic_entropy"],
                "discrete_semantic_entropy": metrics["discrete_semantic_entropy"]
            },
            "calibration_family": {
                "p_true": metrics["p_true"],
            },
            "similarity_family": {
                "selfcheckgpt": metrics["selfcheckgpt"]
            },
            "perturbation_family": {
                "sre_uq": metrics["sre_uq"]
            },
            "geometric_family": {
                "vn_entropy": metrics["vn_entropy"],
                "sd_uq": metrics["sd_uq"],
            },
            "raw_metrics": metrics
        }


# Backward-compatible alias for older call sites.
compute_spux = compute_spuq


# =============================================================================
# STRUCTURAL FAMILY (novel — this work)
# =============================================================================

def compute_graph_path_support(
    question: str,
    answer: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    max_hops: int = 3,
    min_name_length: int = 4,
) -> float:
    """
    Graph Path Support — Structural uncertainty metric (novel — this work).

    Unlike all other metrics in this file, this metric does NOT require multiple
    LLM samples. It operates entirely on the knowledge graph structure.

    ALGORITHM
    ---------
    1. Find question entities: KG entities whose names appear in the question text
    2. Find answer entities: KG entities whose names appear in the generated answer
    3. For each answer entity, check if any path exists from a question entity
       within max_hops hops in the KG
    4. support   = |reachable answer entities| / |all answer entities|
    5. uncertainty = 1 - support

    INTUITION
    ---------
    If the KG can trace a path from what was asked to what was answered, the graph
    structurally supports the answer → low uncertainty.
    If the KG has no path connecting question entities to answer entities, the answer
    is not grounded in the graph structure → high uncertainty → likely wrong.

    IMMUNITY TO GRAPH-INDUCED OVERCONFIDENCE
    -----------------------------------------
    Variance-based metrics (semantic entropy, VN-entropy, SD-UQ) collapse for
    KG-RAG because identical graph context → identical LLM samples → zero variance.
    This metric bypasses LLM sampling entirely — it measures uncertainty inside
    the graph structure, which is unaffected by LLM sampling consistency.

    RETURN VALUES
    -------------
    0.0  — full graph support (all answer entities reachable) → low uncertainty
    1.0  — no graph support (no answer entities reachable)   → high uncertainty
    0.5  — undefined (no entities found in question or answer)

    NOTES
    -----
    - Scoped to kg_name when provided (avoids cross-KG contamination)
    - Filters entity names shorter than min_name_length to avoid stop-word matches
    - Question entities that also appear in the answer are excluded from the
      answer entity set (trivial self-paths don't count as evidence)
    """
    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        logger.warning("neo4j driver not available — skipping graph path support")
        return 0.5

    if not question or not answer:
        return 0.5

    return compute_graph_path_support_detailed(
        question=question,
        answer=answer,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        kg_name=kg_name,
        question_id=question_id,
        max_hops=max_hops,
        min_name_length=min_name_length,
    )["score"]


def compute_graph_path_support_detailed(
    question: str,
    answer: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    max_hops: int = 3,
    min_name_length: int = 4,
) -> Dict[str, Any]:
    """Like compute_graph_path_support but returns a dict with score AND null_reason.

    Keys:
      score       float  — the GPS uncertainty score (0.0–1.0, or 0.5 when undefined)
      null_reason str|None — why 0.5 was returned, or None when a real score was computed
                   "no_q_entities"  : no KG entities matched the question text
                   "no_a_entities"  : no KG entities matched the answer text
                   "trivial_overlap": all answer entities also appeared in the question
                   "neo4j_unavailable" : driver import failed
                   "error"          : unexpected exception

    The null_reason field lets callers exclude undefined rows from AUROC instead of
    treating them as a real 0.5 score, which would inflate discrimination estimates.
    """
    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        logger.warning("neo4j driver not available — skipping graph path support")
        return {"score": 0.5, "null_reason": "neo4j_unavailable"}

    if not question or not answer:
        return {"score": 0.5, "null_reason": "no_input"}

    driver = None
    try:
        driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session(database=neo4j_database) as session:
            entity_scope = _question_local_entity_support_predicate(
                "e",
                kg_name=kg_name,
                question_id=question_id,
            )
            path_scope = _question_local_path_support_predicate(
                path_var="p",
                kg_name=kg_name,
                question_id=question_id,
            )
            resolved = _resolve_structural_entities(
                session,
                question=question,
                answer=answer,
                entity_scope=entity_scope,
                kg_name=kg_name,
                question_id=question_id,
                min_name_length=min_name_length,
            )
            null_reason = str(resolved.get("null_reason") or "")
            if null_reason:
                driver.close()
                return {"score": 0.5, "null_reason": null_reason}
            question_entity_ids = list(resolved["question_entity_ids"])
            answer_entity_ids = list(resolved["answer_entity_ids"])
            question_sources_by_answer = dict(resolved["question_sources_by_answer"])

            # Step 3: Check reachability per answer entity with LIMIT 1 each.
            # Running one query per answer entity (capped at 5) lets us use
            # LIMIT 1 correctly — stop at the first qualifying path — while
            # keeping the confidence filter that shortestPath cannot support.
            reachable_ids = set()
            per_pair_query = f"""
            UNWIND $q_ids AS q_id
            MATCH (q_e:__Entity__ {{id: q_id}})
            MATCH (a_e:__Entity__ {{id: $a_id}})
            MATCH p = (q_e)-[*1..{max_hops}]-(a_e)
            WHERE ALL(r IN relationships(p) WHERE coalesce(r.confidence, 1.0) >= 0.4)
              {path_scope}
            RETURN a_e.id AS reachable_id
            LIMIT 1
            """
            for a_id in answer_entity_ids:
                params = {
                    "q_ids": question_sources_by_answer.get(a_id, question_entity_ids),
                    "a_id": a_id,
                }
                if kg_name:
                    params["kg_name"] = kg_name
                if question_id:
                    params["question_id"] = question_id
                res = list(session.run(per_pair_query, params, timeout=20))
                if res:
                    reachable_ids.add(a_id)

        driver.close()

        support = len(reachable_ids) / len(answer_entity_ids)
        return {"score": float(1.0 - support), "null_reason": None}

    except Exception as e:
        logger.warning(f"Graph path support failed: {e}")
        if driver is not None:
            try: driver.close()
            except Exception: pass
        return {"score": 0.5, "null_reason": "error"}


def compute_graph_path_disagreement(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    max_hops: int = 2,
    min_name_length: int = 4,
) -> float:
    """
    Graph Path Disagreement — Structural uncertainty metric (novel — this work).

    Stronger than graph_path_support because it does NOT require the generated
    answer. It measures structural ambiguity in the knowledge graph given only
    the question: how many distinct candidate answers does the graph suggest?

    ALGORITHM
    ---------
    1. Find question entities: KG entities whose names appear in the question
    2. Walk max_hops directed hops from each question entity → collect terminal
       neighbor entities and their path-frequency (how many paths reach them)
    3. Compute entropy over the terminal entity frequency distribution

    INTUITION
    ---------
    Low entropy  → KG points toward few distinct entities → graph is unambiguous
                   → low structural uncertainty → answer is likely well-grounded
    High entropy → KG fans out to many distinct entities → graph is ambiguous
                   → high structural uncertainty → model may be confidently wrong

    Example:
      "What film did Christopher Nolan direct?"
      KG: Nolan → directed → [Inception only]          → entropy ≈ 0, certain
      KG: Nolan → directed → [Inception, TDK, Interstellar] → high entropy, uncertain

    ADVANTAGE OVER graph_path_support
    ----------------------------------
    Path support is binary (reachable or not) and vulnerable to small-KG density
    where everything is reachable within 3 hops. Disagreement is continuous and
    measures the actual shape of the graph around the question — not just
    whether a path exists, but how many different paths exist.

    RETURN VALUES
    -------------
    0.0  — one terminal entity (KG unambiguous) → low uncertainty
    high — many terminal entities (KG ambiguous) → high uncertainty
    0.5  — undefined (no question entities found in KG)
    """
    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        logger.warning("neo4j driver not available — skipping graph path disagreement")
        return 0.5

    if not question:
        return 0.5

    driver = None
    try:
        driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        question_lower = question.lower()

        with driver.session(database=neo4j_database) as session:
            entity_scope = _question_local_entity_support_predicate(
                "e",
                kg_name=kg_name,
                question_id=question_id,
            )
            path_scope = _question_local_path_support_predicate(
                path_var="path",
                kg_name=kg_name,
                question_id=question_id,
            )
            # Step 1: Find question entities
            q_entity_query = f"""
            MATCH (e:__Entity__)
            WHERE size(e.name) >= $min_len
              AND $question_lower CONTAINS toLower(e.name)
              AND {entity_scope}
            RETURN DISTINCT e.id AS id
            LIMIT 20
            """
            q_params = {"question_lower": question_lower, "min_len": min_name_length}
            if kg_name:
                q_params["kg_name"] = kg_name
            if question_id:
                q_params["question_id"] = question_id
            q_result = session.run(q_entity_query, q_params)
            question_entity_ids = [r["id"] for r in q_result]

            if not question_entity_ids:
                driver.close()
                return 0.5

            # Step 2: Walk max_hops from question entities, count neighbor frequencies
            walk_query = f"""
            MATCH (q_e:__Entity__)
            WHERE q_e.id IN $q_ids
            MATCH path = (q_e)-[*1..{max_hops}]->(neighbor:__Entity__)
            WHERE NOT q_e.id = neighbor.id
              AND NOT neighbor.id IN $q_ids
              AND ALL(r IN relationships(path) WHERE coalesce(r.confidence, 1.0) >= 0.4)
              {path_scope}
            RETURN neighbor.id AS nid, count(path) AS freq
            """
            walk_params = {"q_ids": question_entity_ids}
            if kg_name:
                walk_params["kg_name"] = kg_name
            if question_id:
                walk_params["question_id"] = question_id
            walk_result = session.run(walk_query, walk_params)
            rows = [(r["nid"], r["freq"]) for r in walk_result]

        driver.close()

        if not rows:
            return 0.5

        # Step 3: Entropy over terminal entity frequency distribution
        freqs = np.array([float(f) for _, f in rows])
        total = freqs.sum()
        if total <= 0:
            return 0.5
        probs = freqs / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        # Normalise by log(N) so score ∈ [0, 1]
        n = len(rows)
        max_entropy = np.log(n) if n > 1 else 1.0
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    except Exception as e:
        logger.warning(f"Graph path disagreement failed: {e}")
        if driver is not None:
            try: driver.close()
            except Exception: pass
        return 0.5


def compute_evidence_vn_entropy(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    max_hops: int = 2,
    min_name_length: int = 4,
    n_triples: int = 20,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """
    Evidence VN-Entropy — Input-side structural uncertainty metric (novel — this work).

    MOTIVATION
    ----------
    All output-side UQ methods (semantic entropy, VN-entropy, SD-UQ, selfcheckgpt)
    collapse for KG-RAG because deterministic graph retrieval forces identical LLM
    inputs → identical outputs → zero sampling variance. Evidence VN-Entropy bypasses
    LLM sampling entirely by measuring the semantic geometry of the *retrieved KG
    evidence*, not the generated responses.

    FORMULATION
    -----------
    Combined uncertainty from two complementary signals:

        U = 1 − A × (1 − S_e)

    where:
        S_e = VN-entropy of the retrieved-triple Gram matrix, normalised ∈ [0, 1]
              (measures coverage: how semantically spread is the retrieved evidence?)
        A   = cosine_similarity(mean_triple_embedding, question_embedding), ∈ [0, 1]
              (measures alignment: does the evidence actually address the question?)

    CASE ANALYSIS
    -------------
    | Case                           | S_e  | A    | U    |
    |--------------------------------|------|------|------|
    | Correct, confident             | low  | high | ≈ 0  | ← correctly certain
    | Coverage gap (diffuse KG)      | high | any  | ≈ 1  | ← correctly uncertain
    | Abstraction loss (off-target)  | low  | low  | ≈ 1  | ← correctly uncertain
    | Genuinely ambiguous KG         | high | mod  | ≈ 1  | ← correctly uncertain

    U is low only when the evidence is both semantically focused AND well-aligned
    with the question — the one regime where confidence is warranted.

    LIMITATION
    ----------
    Abstraction loss where evidence is on-topic but at the wrong granularity
    (e.g. LOCATED_IN → Istanbul when Laleli is needed) is partially detectable
    via alignment but not fully — sentence embeddings cannot easily distinguish
    city-level from neighbourhood-level answers to the same location question.
    This remains an open problem identified as future work.

    NOVELTY
    -------
    KLE (Nikitin et al., NeurIPS 2024) applies VN-entropy to the Gram matrix of
    LLM *output samples*. This metric applies the same spectral framework to the
    *retrieved evidence embeddings* — the input side. No prior paper does this.
    The combined U = 1 − A·(1 − S_e) formula integrating entropy with question
    alignment is new.

    ALGORITHM
    ---------
    1. Find question entities (KG entities whose names appear in the question)
    2. Retrieve all triples in their max_hops neighbourhood
    3. Format each triple as "head RELATION tail" text
    4. Embed all triple texts with a sentence encoder
    5. Build Gram matrix G = VV^T (L2-normalised embeddings)
    6. Compute S_e = −∑ λ_i log λ_i / log(N)  (normalised VN-entropy)
    7. Compute A = cosine(mean_triple_embedding, question_embedding)
    8. Return U = 1 − A × (1 − S_e)

    RETURN VALUES
    -------------
    0.0  — no evidence retrieved (fallback)
    ≈ 0  — focused, well-aligned evidence → low uncertainty
    → 1  — diffuse or misaligned evidence → high uncertainty
    """
    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        logger.warning("neo4j not available — skipping evidence VN-entropy")
        return 0.0

    if not question:
        return 0.0

    driver = None
    try:
        driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        question_lower = question.lower()

        with driver.session(database=neo4j_database) as session:
            entity_scope = _question_local_entity_support_predicate(
                "e",
                kg_name=kg_name,
                question_id=question_id,
            )
            pair_scope = _question_local_pair_support_predicate(
                "h",
                "t",
                kg_name=kg_name,
                question_id=question_id,
                relationship_var="r",
            )
            path_scope = _question_local_path_support_predicate(
                path_var="path",
                kg_name=kg_name,
                question_id=question_id,
            )

            # Step 1: Find question entities
            q_entity_query = f"""
            MATCH (e:__Entity__)
            WHERE size(e.name) >= $min_len
              AND $question_lower CONTAINS toLower(e.name)
              AND {entity_scope}
            RETURN DISTINCT e.id AS id
            LIMIT 20
            """
            q_params = {"question_lower": question_lower, "min_len": min_name_length}
            if kg_name:
                q_params["kg_name"] = kg_name
            if question_id:
                q_params["question_id"] = question_id
            q_result = session.run(q_entity_query, q_params)
            question_entity_ids = [r["id"] for r in q_result]

            if not question_entity_ids:
                driver.close()
                return 0.0

            # Step 2: Retrieve triples in max_hops neighbourhood
            triple_query = f"""
            MATCH (h:__Entity__)-[r]->(t:__Entity__)
            WHERE (
                  h.id IN $q_ids
               OR t.id IN $q_ids
               OR EXISTS {{
                   MATCH path = (q_e:__Entity__)-[*1..{max_hops}]-(h)
                   WHERE q_e.id IN $q_ids
                     AND ALL(rel IN relationships(path) WHERE coalesce(rel.confidence, 1.0) >= 0.4)
                     {path_scope}
                }}
               OR EXISTS {{
                   MATCH path = (q_e:__Entity__)-[*1..{max_hops}]-(t)
                   WHERE q_e.id IN $q_ids
                     AND ALL(rel IN relationships(path) WHERE coalesce(rel.confidence, 1.0) >= 0.4)
                     {path_scope}
               }}
            )
              AND coalesce(r.confidence, 1.0) >= 0.4
              AND {pair_scope}
            RETURN h.name AS head, type(r) AS rel, t.name AS tail
            LIMIT {n_triples}
            """
            triple_params = {"q_ids": question_entity_ids}
            if kg_name:
                triple_params["kg_name"] = kg_name
            if question_id:
                triple_params["question_id"] = question_id
            triple_result = session.run(triple_query, triple_params)
            triples = [(r["head"], r["rel"], r["tail"]) for r in triple_result
                       if r["head"] and r["rel"] and r["tail"]]

        driver.close()

        if not triples:
            return 0.0

        # Step 3: Format triples as text
        triple_texts = [f"{h} {rel.replace('_', ' ').lower()} {t}" for h, rel, t in triples]

        # Step 4: Embed triples and question
        model = _get_or_load_sentence_transformer(embedding_model)

        V = model.encode(triple_texts, show_progress_bar=False, normalize_embeddings=True)
        q_vec = model.encode([question], show_progress_bar=False, normalize_embeddings=True)[0]
        N = len(V)

        # Step 5: VN-entropy of Gram matrix
        G = V @ V.T
        eigenvalues = np.linalg.eigvalsh(G) / N
        eigenvalues = np.clip(eigenvalues, 0, None)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0:
            return 0.0
        raw_entropy = float(-np.sum(eigenvalues * np.log(eigenvalues)))
        # Normalise to [0, 1]
        max_entropy = np.log(N) if N > 1 else 1.0
        S_e = float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))

        # Step 6: Question-evidence alignment
        mean_v = V.mean(axis=0)
        norm = np.linalg.norm(mean_v)
        if norm > 1e-10:
            mean_v = mean_v / norm
        A = float(np.clip(float(mean_v @ q_vec), 0.0, 1.0))

        # Step 7: Combined uncertainty U = 1 - A * (1 - S_e)
        U = 1.0 - A * (1.0 - S_e)
        return float(np.clip(U, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"Evidence VN-entropy failed: {e}")
        if driver is not None:
            try: driver.close()
            except Exception: pass
        return 0.0


def compute_competing_answer_alternatives(
    question: str,
    answer: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    min_name_length: int = 4,
) -> float:
    """
    Competing Answer Alternatives — Structural uncertainty metric (novel — this work).

    MOTIVATION
    ----------
    A core operationalisation of epistemic uncertainty: if there are many same-type
    entities reachable via the same relation type that the answer uses, then the
    answer is one of multiple plausible alternatives the graph supports — the KG
    does not uniquely determine the answer. High cardinality of alternatives = high
    structural uncertainty.

    ALGORITHM
    ---------
    1. Find question entities: KG entities whose names appear in the question text
    2. Find answer entity: the KG entity best matching the generated answer text
    3. Find the relation type(s) that connect question entities to the answer entity
    4. For each such relation type, count all distinct entities reachable from any
       question entity via that same relation type → "competing alternatives"
    5. Return a monotone transform of the count:  1 - 1/(1 + N_alternatives)
       so score ∈ (0, 1), approaches 1 as alternatives → ∞

    EXAMPLE
    -------
    Q: "What country is Paris the capital of?"
    KG: Paris -[CAPITAL_OF]-> France
    Via CAPITAL_OF from Paris: just France → alternatives = 1 → uncertainty ≈ 0.5

    Q: "What city is a major financial hub in the UK?"
    KG: London -[MAJOR_FINANCIAL_HUB_IN]-> UK
        Edinburgh -[MAJOR_FINANCIAL_HUB_IN]-> UK
        Manchester -[MAJOR_FINANCIAL_HUB_IN]-> UK
    Answered "London". Via MAJOR_FINANCIAL_HUB_IN from question entities: 3
    alternatives → uncertainty ≈ 0.75 (answer could have been any of the 3)

    WHY THIS IS DIFFERENT FROM graph_path_disagreement
    ---------------------------------------------------
    graph_path_disagreement measures the *breadth* of the graph reachable from
    question entities (entropy over all neighbour entity types).
    competing_answer_alternatives is *typed* — it specifically looks at the
    relation type the answer used and asks: how many other entities could have
    been the answer via that same typed relation? This is a direct count of
    competing candidates, not a general fan-out entropy.

    NOVEL CONTRIBUTION
    ------------------
    No published paper uses typed-relation cardinality as a UQ signal. The closest
    related work (arXiv:2512.22318) uses entity-relation co-occurrence frequency for
    OOD detection in KGE, not for answer uncertainty. This metric is:
    - Immune to graph-induced overconfidence (no LLM sampling)
    - KG-specific (uses typed relation from answer)
    - Theoretically grounded: models the set of equally-plausible competing answers

    RETURN VALUES
    -------------
    0.0  — fallback (no entities found, no matching relation)
    ~0.5 — one alternative (expected for unambiguous facts)
    → 1  — many alternatives (highly ambiguous in the KG)

    Formula:  uncertainty = 1 - 1 / (1 + N_competing)
    """
    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        logger.warning("neo4j driver not available — skipping competing answer alternatives")
        return 0.0

    if not question or not answer:
        return 0.0

    driver = None
    try:
        driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        question_lower = question.lower()
        answer_lower = answer.lower()

        with driver.session(database=neo4j_database) as session:
            entity_scope = _question_local_entity_support_predicate(
                "e",
                kg_name=kg_name,
                question_id=question_id,
            )
            pair_scope_rel = _question_local_pair_support_predicate(
                "q_e",
                "a_e",
                kg_name=kg_name,
                question_id=question_id,
                relationship_var="r",
            )
            pair_scope_alt = _question_local_pair_support_predicate(
                "q_e",
                "alt",
                kg_name=kg_name,
                question_id=question_id,
                relationship_var="r",
            )

            # Step 1: Find question entities
            q_entity_query = f"""
            MATCH (e:__Entity__)
            WHERE size(e.name) >= $min_len
              AND $question_lower CONTAINS toLower(e.name)
              AND {entity_scope}
            RETURN DISTINCT e.id AS id, e.name AS name
            LIMIT 20
            """
            q_params = {"question_lower": question_lower, "min_len": min_name_length}
            if kg_name:
                q_params["kg_name"] = kg_name
            if question_id:
                q_params["question_id"] = question_id
            q_result = session.run(q_entity_query, q_params)
            question_entity_ids = [r["id"] for r in q_result]

            if not question_entity_ids:
                driver.close()
                return 0.0

            # Step 2: Find answer entity (best substring match)
            a_entity_query = f"""
            MATCH (e:__Entity__)
            WHERE size(e.name) >= $min_len
              AND $answer_lower CONTAINS toLower(e.name)
              AND {entity_scope}
            RETURN DISTINCT e.id AS id, e.name AS name, size(e.name) AS name_len
            ORDER BY name_len DESC
            LIMIT 5
            """
            a_params = {"answer_lower": answer_lower, "min_len": min_name_length}
            if kg_name:
                a_params["kg_name"] = kg_name
            if question_id:
                a_params["question_id"] = question_id
            a_result = session.run(a_entity_query, a_params)
            answer_entity_rows = [(r["id"], r["name"]) for r in a_result]

            if not answer_entity_rows:
                driver.close()
                return 0.0

            answer_entity_ids = [row[0] for row in answer_entity_rows]
            # Remove question entities from answer entities
            q_id_set = set(question_entity_ids)
            answer_entity_ids = [eid for eid in answer_entity_ids if eid not in q_id_set]

            if not answer_entity_ids:
                driver.close()
                return 0.0

            # Step 3: Find relation types connecting question entities → answer entity
            rel_type_query = f"""
            MATCH (q_e:__Entity__)-[r]->(a_e:__Entity__)
            WHERE q_e.id IN $q_ids
              AND a_e.id IN $a_ids
              AND coalesce(r.confidence, 1.0) >= 0.4
              AND {pair_scope_rel}
            RETURN DISTINCT type(r) AS rel_type
            LIMIT 10
            """
            rel_params = {"q_ids": question_entity_ids, "a_ids": answer_entity_ids}
            if kg_name:
                rel_params["kg_name"] = kg_name
            if question_id:
                rel_params["question_id"] = question_id
            rel_result = session.run(rel_type_query, rel_params)
            relation_patterns = [(r["rel_type"], "forward") for r in rel_result]

            if not relation_patterns:
                # Also try reverse direction (answer → question)
                rel_type_query_rev = f"""
                MATCH (a_e:__Entity__)-[r]->(q_e:__Entity__)
                WHERE q_e.id IN $q_ids
                  AND a_e.id IN $a_ids
                  AND coalesce(r.confidence, 1.0) >= 0.4
                  AND {pair_scope_rel}
                RETURN DISTINCT type(r) AS rel_type
                LIMIT 10
                """
                rel_result_rev = session.run(rel_type_query_rev, rel_params)
                relation_patterns = [(r["rel_type"], "reverse") for r in rel_result_rev]

            if not relation_patterns:
                driver.close()
                return 0.0

            # Step 4: Count all entities reachable from question entities via those
            # relation types — these are the "competing answer alternatives"
            competing_count = 0
            for rel_type, direction in relation_patterns:
                if direction == "reverse":
                    alt_query = f"""
                    MATCH (alt:__Entity__)-[r:{rel_type}]->(q_e:__Entity__)
                    WHERE q_e.id IN $q_ids
                      AND NOT alt.id IN $q_ids
                      AND coalesce(r.confidence, 1.0) >= 0.4
                      AND {pair_scope_alt}
                    RETURN count(DISTINCT alt) AS n
                    """
                else:
                    alt_query = f"""
                    MATCH (q_e:__Entity__)-[r:{rel_type}]->(alt:__Entity__)
                    WHERE q_e.id IN $q_ids
                      AND NOT alt.id IN $q_ids
                      AND coalesce(r.confidence, 1.0) >= 0.4
                      AND {pair_scope_alt}
                    RETURN count(DISTINCT alt) AS n
                    """
                alt_params = {"q_ids": question_entity_ids}
                if kg_name:
                    alt_params["kg_name"] = kg_name
                if question_id:
                    alt_params["question_id"] = question_id
                alt_result = session.run(alt_query, alt_params)
                row = alt_result.single()
                if row:
                    competing_count = max(competing_count, row["n"])

        driver.close()

        if competing_count == 0:
            return 0.0

        # Monotone transform: 1 - 1/(1 + N) so ∈ (0, 1), asymptotes to 1
        return float(1.0 - 1.0 / (1.0 + competing_count))

    except Exception as e:
        logger.warning(f"Competing answer alternatives failed: {e}")
        if driver is not None:
            try: driver.close()
            except Exception: pass
        return 0.0


def compute_subgraph_perturbation_stability(
    question: str,
    answer: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    max_hops: int = 3,
    min_name_length: int = 4,
    n_perturbations: int = 16,
    min_dropout: float = 0.10,
    max_dropout: float = 0.40,
    max_paths: int = 500,
    seed: int = 42,
) -> float:
    """Backward-compatible scalar SPS-UQ wrapper."""
    return compute_subgraph_perturbation_stability_detailed(
        question=question,
        answer=answer,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        kg_name=kg_name,
        question_id=question_id,
        max_hops=max_hops,
        min_name_length=min_name_length,
        n_perturbations=n_perturbations,
        min_dropout=min_dropout,
        max_dropout=max_dropout,
        max_paths=max_paths,
        seed=seed,
    )["score"]


def compute_subgraph_perturbation_stability_detailed(
    question: str,
    answer: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    max_hops: int = 3,
    min_name_length: int = 4,
    n_perturbations: int = 16,
    min_dropout: float = 0.10,
    max_dropout: float = 0.40,
    max_paths: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Subgraph Perturbation Stability (SPS-UQ) — Structural uncertainty (novel).

    Core idea:
      If tiny perturbations of the supporting KG subgraph quickly destroy support
      for the generated answer, confidence should be low.

    Algorithm:
      1) Extract question entities and answer entities from KG mentions.
      2) Enumerate answer-supporting paths (<= max_hops) from question -> answer entities.
      3) Build a relation-id universe from those paths.
      4) Simulate n_perturbations counterfactual KGs by randomly dropping a subset
         of relations (dropout in [min_dropout, max_dropout]).
      5) Recompute answer support under each perturbation.
      6) Stability = mean( support_cf / support_base ) clipped to [0, 1].
      7) Uncertainty = 1 - Stability.

    Return values:
      0.0  -> highly stable support under perturbations (low uncertainty)
      1.0  -> highly fragile support (high uncertainty)
      {"score": 0.5, "null_reason": "..."} -> undefined/abstained

    Null reasons mirror GPS where possible:
      no_input, no_q_entities, no_a_entities, trivial_overlap,
      neo4j_unavailable, error.
    """
    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        logger.warning("neo4j driver not available — skipping sps_uq")
        return {"score": 0.5, "null_reason": "neo4j_unavailable"}

    if not question or not answer:
        return {"score": 0.5, "null_reason": "no_input"}

    driver = None
    try:
        import zlib

        driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session(database=neo4j_database) as session:
            entity_scope = _question_local_entity_support_predicate(
                "e",
                kg_name=kg_name,
                question_id=question_id,
            )
            path_scope = _question_local_path_support_predicate(
                path_var="p",
                kg_name=kg_name,
                question_id=question_id,
            )
            resolved = _resolve_structural_entities(
                session,
                question=question,
                answer=answer,
                entity_scope=entity_scope,
                kg_name=kg_name,
                question_id=question_id,
                min_name_length=min_name_length,
            )
            null_reason = str(resolved.get("null_reason") or "")
            if null_reason:
                driver.close()
                return {"score": 0.5, "null_reason": null_reason}
            q_ids = list(resolved["question_entity_ids"])
            a_ids = list(resolved["answer_entity_ids"])
            question_sources_by_answer = dict(resolved["question_sources_by_answer"])

            # 3) supporting paths with relationship IDs — capped at max_paths
            path_query = f"""
            UNWIND $q_ids AS q_id
            MATCH (q:__Entity__ {{id: q_id}})
            MATCH p = (q)-[*1..{max_hops}]-(a:__Entity__)
            WHERE a.id IN $a_ids
              AND ALL(r IN relationships(p) WHERE coalesce(r.confidence, 1.0) >= 0.4)
              {path_scope}
            RETURN a.id AS a_id,
                   [rel IN relationships(p) | id(rel)] AS rel_ids
            LIMIT {max_paths}
            """
            path_rows: List[Dict[str, Any]] = []
            for a_id in a_ids:
                path_params = {
                    "q_ids": question_sources_by_answer.get(a_id, q_ids),
                    "a_ids": [a_id],
                }
                if kg_name:
                    path_params["kg_name"] = kg_name
                if question_id:
                    path_params["question_id"] = question_id
                path_rows.extend(
                    dict(r) for r in session.run(path_query, path_params, timeout=20)
                )

        driver.close()

        # No support paths => maximally uncertain for this answer wrt KG
        if not path_rows:
            return {"score": 1.0, "null_reason": None}

        answer_count = max(1, len(a_ids))
        baseline_reachable = len({row["a_id"] for row in path_rows}) / answer_count
        if baseline_reachable <= 0:
            return {"score": 1.0, "null_reason": None}

        rel_universe = sorted({rid for row in path_rows for rid in (row.get("rel_ids") or [])})
        if not rel_universe:
            # Degenerate case: treat as unsupported if we cannot identify edges.
            return {"score": 1.0, "null_reason": None}

        # Deterministic RNG per (question, answer)
        pair_seed = seed ^ zlib.crc32(f"{question}||{answer}".encode("utf-8"))
        rng = np.random.default_rng(pair_seed)

        # Guardrails
        n_perturbations = max(1, int(n_perturbations))
        min_dropout = float(np.clip(min_dropout, 0.0, 1.0))
        max_dropout = float(np.clip(max_dropout, min_dropout, 1.0))

        stability_terms: List[float] = []
        rel_universe_arr = np.array(rel_universe, dtype=np.int64)

        for _ in range(n_perturbations):
            dropout = float(rng.uniform(min_dropout, max_dropout))
            k = int(round(dropout * len(rel_universe_arr)))
            if len(rel_universe_arr) > 0:
                k = min(len(rel_universe_arr), max(1, k))
                dropped = set(rng.choice(rel_universe_arr, size=k, replace=False).tolist())
            else:
                dropped = set()

            reachable_cf = set()
            for row in path_rows:
                rel_ids = row.get("rel_ids") or []
                if rel_ids and not any(rid in dropped for rid in rel_ids):
                    reachable_cf.add(row["a_id"])

            support_cf = len(reachable_cf) / answer_count
            stability_terms.append(float(np.clip(support_cf / baseline_reachable, 0.0, 1.0)))

        stability = float(np.mean(stability_terms)) if stability_terms else float(baseline_reachable)
        uncertainty = 1.0 - stability
        return {"score": float(np.clip(uncertainty, 0.0, 1.0)), "null_reason": None}

    except Exception as e:
        logger.warning(f"Counterfactual subgraph stability failed: {e}")
        if driver is not None:
            try: driver.close()
            except Exception: pass
        return {"score": 0.5, "null_reason": "error"}


# =============================================================================
# SELECTIVE-PREDICTION METRICS: AUROC & AUREC
# =============================================================================

# Metrics where HIGHER value = MORE uncertain (standard "uncertainty score").
# For AUROC we predict correctness as -uncertainty, so we negate these.
_HIGHER_IS_MORE_UNCERTAIN = {
    "semantic_entropy", "discrete_semantic_entropy",
    "sre_uq", "selfcheckgpt", "vn_entropy", "sd_uq",
    "graph_path_support",              # structural: 1 - support = uncertainty
    "graph_path_disagreement",         # structural: entropy over graph neighbor distribution
    "competing_answer_alternatives",   # structural: typed-relation cardinality of competing answers
    "evidence_vn_entropy",             # structural: combined evidence entropy + question alignment
    "subgraph_informativeness",        # structural: pre-generation answer-space concentration (novel — this work)
    "subgraph_perturbation_stability",  # structural: perturbation fragility of answer-supporting subgraph
    "support_entailment_uncertainty",  # grounding: evidence-answer NLI entailment deficit (novel — this work)
    "evidence_conflict_uncertainty",   # grounding: fraction of E-C conflicting chunk pairs (novel — this work)
}
# Metrics where HIGHER value = MORE certain (confidence score).
# For AUROC we use the value directly; for AUREC we sort descending on -score.
_HIGHER_IS_MORE_CERTAIN = {"p_true"}
_GPS_NULL_REASON_ORDER = (
    "no_q_entities",
    "no_a_entities",
    "trivial_overlap",
    "neo4j_unavailable",
    "no_input",
    "error",
)
_SPS_NULL_REASON_ORDER = _GPS_NULL_REASON_ORDER


def compute_subgraph_informativeness(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    kg_name: str = None,
    question_id: Optional[str] = None,
    max_hops: int = 2,
    min_name_length: int = 4,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """
    Subgraph Informativeness (SI) — Pre-generation structural uncertainty metric
    (novel — this work).

    MOTIVATION
    ----------
    All existing UQ metrics are post-hoc: they generate first, then measure output
    variance or self-consistency. For KG-RAG this is problematic because deterministic
    graph retrieval collapses output variance regardless of correctness.

    SI operates entirely on the retrieved subgraph BEFORE generation. It asks:
    "How much does this subgraph constrain the answer space?" — modelling the KG as
    an information source and measuring whether it concentrates probability mass on
    a small set of answer candidates.

    FORMULATION
    -----------
        SI = (1 − H_normalized) × A

    where:
        H_normalized = H(P_candidates) / log(N)
            Entropy of the weighted distribution over answer candidate entities,
            normalised to [0, 1]. Low = subgraph concentrates on few candidates.

        A = cosine(question_embedding, mean_relation_embedding)
            Semantic alignment between the question and the relation types in the
            subgraph. Low = the graph paths are topically unrelated to the question.

    SI is high (→ 1) only when the subgraph simultaneously:
      (a) concentrates on a small set of answer candidates, AND
      (b) does so via relations semantically aligned with the question.

    Uncertainty score: U = 1 − SI  (returned, so higher = more uncertain)

    CASE ANALYSIS
    -------------
    | Case                            | H     | A    | SI   | U    |
    |---------------------------------|-------|------|------|------|
    | Focused, on-topic subgraph      | low   | high | high | low  | ← confident, correct
    | Diffuse subgraph (many answers) | high  | any  | low  | high | ← uncertain
    | Off-topic subgraph              | any   | low  | low  | high | ← uncertain
    | Abstraction loss                | low   | mod  | mod  | mod  | ← partially detected
    | No entities found               | —     | —    | —    | 0.5  | ← fallback

    NOVELTY
    -------
    - graph_path_disagreement: entropy over ALL neighbour types, unweighted,
      not combined with alignment. SI weights candidates by path support and
      filters by relation relevance.
    - evidence_vn_entropy: spectral entropy of triple TEXT embeddings (coverage
      of the evidence space). SI models the ANSWER CANDIDATE distribution
      (concentration of the answer space). Complementary, not redundant.
    - competing_answer_alternatives: requires the generated answer (post-hoc).
      SI is fully pre-generation — no answer needed.

    No prior paper models KG retrieval as a pre-generation distribution over
    answer candidates and measures its entropy as an uncertainty signal.

    ALGORITHM
    ---------
    1. Find question entities (KG entities whose names appear in the question)
    2. Walk max_hops from each question entity; collect (candidate_entity, relation_type,
       path_length) tuples
    3. Weight each candidate: w_i = sum over all paths to i of (1 / path_length)
       — longer paths get lower weight (less direct support)
    4. Normalise weights to a probability distribution P_candidates
    5. Compute H_normalized = H(P_candidates) / log(N_candidates)
    6. Embed each distinct relation type as text; compute mean relation embedding
    7. Embed the question; compute A = cosine(question_emb, mean_relation_emb)
    8. Return U = 1 − (1 − H_normalized) × A

    RETURN VALUES
    -------------
    0.5  — fallback (no entities found or single candidate — uninformative)
    → 0  — highly focused, on-topic subgraph → low uncertainty
    → 1  — diffuse or off-topic subgraph → high uncertainty
    """
    try:
        from sentence_transformers import SentenceTransformer
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        logger.warning("sentence_transformers or neo4j not available — skipping subgraph informativeness")
        return 0.5

    if not question:
        return 0.5

    driver = None
    try:
        driver = _GDB.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        question_lower = question.lower()

        with driver.session(database=neo4j_database) as session:
            entity_scope = _question_local_entity_support_predicate(
                "e",
                kg_name=kg_name,
                question_id=question_id,
            )
            path_scope = _question_local_path_support_predicate(
                path_var="path",
                kg_name=kg_name,
                question_id=question_id,
            )

            # Step 1: find question entities
            q_entity_query = f"""
            MATCH (e:__Entity__)
            WHERE size(e.name) >= $min_len
              AND $question_lower CONTAINS toLower(e.name)
              AND {entity_scope}
            RETURN DISTINCT e.id AS id, e.name AS name
            LIMIT 20
            """
            q_params: dict = {"question_lower": question_lower, "min_len": min_name_length}
            if kg_name:
                q_params["kg_name"] = kg_name
            if question_id:
                q_params["question_id"] = question_id

            q_result = session.run(q_entity_query, q_params)
            question_entity_ids = [r["id"] for r in q_result]

            if not question_entity_ids:
                driver.close()
                return 0.5

            # Step 2: walk max_hops from question entities, collecting
            # (candidate_id, candidate_name, relation_type, path_length)
            walk_query = f"""
            MATCH (seed:__Entity__)
            WHERE seed.id IN $seed_ids
            MATCH path = (seed)-[*1..{max_hops}]->(candidate:__Entity__)
            WHERE NOT candidate.id IN $seed_ids
              AND ALL(r IN relationships(path) WHERE coalesce(r.confidence, 1.0) >= 0.4)
              {path_scope}
            UNWIND relationships(path) AS rel
            RETURN DISTINCT
                candidate.id   AS candidate_id,
                candidate.name AS candidate_name,
                type(rel)      AS rel_type,
                length(path)   AS path_len
            LIMIT 500
            """
            walk_params: dict = {"seed_ids": question_entity_ids}
            if kg_name:
                walk_params["kg_name"] = kg_name
            if question_id:
                walk_params["question_id"] = question_id

            walk_result = session.run(walk_query, walk_params)
            rows = [dict(r) for r in walk_result]

        driver.close()

        if not rows:
            return 0.5

        # Step 3: weight each candidate by sum of 1/path_length across all paths
        from collections import defaultdict
        candidate_weight: dict = defaultdict(float)
        candidate_name: dict = {}
        relation_types: set = set()

        for row in rows:
            cid = row["candidate_id"]
            candidate_weight[cid] += 1.0 / max(row["path_len"], 1)
            candidate_name[cid] = row["candidate_name"]
            relation_types.add(row["rel_type"])

        if len(candidate_weight) <= 1:
            # Only one candidate — perfectly concentrated but trivially so
            # Return moderate uncertainty rather than 0
            return 0.2

        # Step 4: normalise to probability distribution
        weights = np.array(list(candidate_weight.values()), dtype=float)
        weights /= weights.sum()

        # Step 5: normalised entropy of candidate distribution
        H = _safe_prob_entropy(weights)
        H_max = np.log(len(weights))
        H_normalized = H / H_max if H_max > 0 else 0.0

        # Step 6-7: question-relation alignment
        # Embed distinct relation types + question; compute cosine alignment
        model = _get_or_load_sentence_transformer(embedding_model)
        relation_texts = [rt.replace("_", " ").lower() for rt in relation_types]
        all_texts = [question] + relation_texts
        embeddings = model.encode(all_texts, normalize_embeddings=True)

        question_emb = embeddings[0]
        relation_embs = embeddings[1:]
        mean_relation_emb = relation_embs.mean(axis=0)
        norm = np.linalg.norm(mean_relation_emb)
        if norm > 1e-8:
            mean_relation_emb = mean_relation_emb / norm

        A = float(np.dot(question_emb, mean_relation_emb))
        A = max(0.0, A)  # clamp negative cosine to 0

        # Step 8: SI = (1 - H_normalized) * A;  U = 1 - SI
        SI = (1.0 - H_normalized) * A
        U = 1.0 - SI

        logger.debug(
            f"[subgraph_informativeness] candidates={len(candidate_weight)} "
            f"H_norm={H_normalized:.3f} A={A:.3f} SI={SI:.3f} U={U:.3f}"
        )
        return float(np.clip(U, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"[subgraph_informativeness] error: {e}")
        if driver is not None:
            try: driver.close()
            except Exception: pass
        return 0.5


def _get_or_load_sentence_transformer(model_name: str):
    """Cache sentence transformer models to avoid reloading on every call."""
    normalized_name = model_name.replace("sentence-transformers/", "")
    with _SENTENCE_TRANSFORMER_CACHE_LOCK:
        if normalized_name not in _SENTENCE_TRANSFORMER_CACHE:
            from sentence_transformers import SentenceTransformer
            _SENTENCE_TRANSFORMER_CACHE[normalized_name] = SentenceTransformer(normalized_name)
    return _SENTENCE_TRANSFORMER_CACHE[normalized_name]


def _get_or_load_nli_model(model_name: str):
    """Return (tokenizer, model) for an NLI sequence-classification model, cached."""
    with _NLI_CACHE_LOCK:
        if model_name not in _NLI_MODEL_CACHE:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            _NLI_TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
            _NLI_MODEL_CACHE[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name)
    return _NLI_TOKENIZER_CACHE[model_name], _NLI_MODEL_CACHE[model_name]


def _classify_chunks_against_answer(
    chunks: List[str],
    answer: str,
    nli_model: str = "microsoft/deberta-large-mnli",
    max_chunks: int = 10,
) -> List[int]:
    """NLI-classify each chunk as entailment/neutral/contradiction of the answer.

    Returns a list of int labels using DeBERTa convention:
      0 = contradiction  (chunk contradicts the answer)
      1 = neutral        (chunk does not address the answer)
      2 = entailment     (chunk supports / entails the answer)

    Premise = chunk text; hypothesis = answer.
    Returns an empty list on any failure so callers can return their fallback.
    """
    if not chunks or not answer:
        return []
    active = [c.strip() for c in chunks if c.strip()][:max_chunks]
    if not active:
        return []

    try:
        import torch
        import torch.nn.functional as F
        tokenizer, model = _get_or_load_nli_model(nli_model)
        labels: List[int] = []
        for chunk in active:
            inputs = tokenizer(
                chunk, answer,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            labels.append(int(torch.argmax(F.softmax(logits, dim=1)).item()))
        return labels
    except Exception as e:
        logger.warning("NLI chunk classification failed: %s", e)
        return []


# =============================================================================
# GROUNDING FAMILY (novel — this work)
# =============================================================================

def compute_support_entailment_uncertainty(
    chunks: List[str],
    answer: str,
    nli_model: str = "microsoft/deberta-large-mnli",
    max_chunks: int = 10,
) -> float:
    """
    Support Entailment Uncertainty (SEU) — Grounding family (novel — this work).

    MOTIVATION
    ----------
    Output-uncertainty metrics (SE, SelfCheckGPT, VN-Entropy) measure disagreement
    across N LLM samples given fixed context.  Under KG-RAG context determinism,
    all N samples receive identical context and these metrics collapse to measuring
    decoder stochasticity alone.

    SEU bypasses sampling entirely: it evaluates the evidence-answer relationship
    directly.  Even when all N samples say the same thing, SEU can signal high
    uncertainty if no retrieved chunk entails the answer — the canonical failure
    mode for abstentions on answerable questions ("I cannot determine from the
    provided context") and for wrong-hop answers on multi-hop benchmarks.

    ALGORITHM
    ---------
    For each retrieved chunk, run NLI(chunk → answer):
      entailment    (2): evidence supports the answer
      neutral       (1): evidence does not address the answer
      contradiction (0): evidence contradicts the answer

    support_score = (n_entail − n_contradict) / n_chunks  ∈ [−1, 1]
    uncertainty   = (1 − support_score) / 2               ∈  [0, 1]

    RETURN VALUES
    -------------
    0.0  — all chunks entail the answer (fully grounded, low uncertainty)
    0.5  — all chunks neutral OR equal entailment and contradiction (undefined)
    1.0  — all chunks contradict the answer (maximally ungrounded)
    0.5  — fallback: no chunks, empty answer, or NLI failure
    """
    labels = _classify_chunks_against_answer(chunks, answer, nli_model, max_chunks)
    if not labels:
        return 0.5

    n = len(labels)
    n_entail = sum(1 for lb in labels if lb == 2)
    n_contradict = sum(1 for lb in labels if lb == 0)
    support_score = (n_entail - n_contradict) / n
    return float((1.0 - support_score) / 2.0)


def compute_evidence_conflict_uncertainty(
    chunks: List[str],
    answer: str,
    nli_model: str = "microsoft/deberta-large-mnli",
    max_chunks: int = 10,
) -> float:
    """
    Evidence Conflict Uncertainty (ECU) — Grounding family (novel — this work).

    MOTIVATION
    ----------
    SEU reports the aggregate entailment signal; ECU reports its variance.
    A high ECU score means some chunks support the answer while others contradict
    it — genuine evidentiary conflict, not neutral silence.

    This is the grounding analogue of SelfCheckGPT: SelfCheckGPT measures whether
    LLM output samples contradict each other; ECU measures whether evidence chunks
    contradict each other in their support for the answer.

    ECU is particularly diagnostic for multi-hop questions: hop-1 evidence and
    hop-2 evidence may independently support different intermediate conclusions,
    producing entail-contradict pairs even when the overall answer is wrong.

    ALGORITHM
    ---------
    1. Compute per-chunk NLI labels against the answer (same pass as SEU).
    2. Count entail-contradict (E-C) conflict pairs: pairs (i, j) where one
       label is entailment (2) and the other is contradiction (0).
    3. ECU = conflict_pairs / total_pairs.

    RETURN VALUES
    -------------
    0.0  — all chunks agree (uniformly entailing, neutral, or contradicting)
    1.0  — maximum conflict (half entail, half contradict)
    0.0  — fallback: fewer than 2 chunks or NLI failure
    """
    labels = _classify_chunks_against_answer(chunks, answer, nli_model, max_chunks)
    if len(labels) < 2:
        return 0.0

    conflict = 0
    total = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            total += 1
            if (labels[i] == 2 and labels[j] == 0) or (labels[i] == 0 and labels[j] == 2):
                conflict += 1

    return float(conflict / total) if total > 0 else 0.0


def compute_precision_at_k(
    details: List[Dict[str, Any]],
    metric_names: Optional[List[str]] = None,
    k_fractions: tuple = (0.1, 0.2, 0.3),
    exclude_generation_failures: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Precision@k (PPV@k) — fraction of top-k% most uncertain questions that are wrong.

    AUROC measures global ranking quality. PPV@k measures what happens at the
    operating point you actually care about: when you abstain on the most uncertain
    k% of predictions, how often are those abstained questions actually wrong?

    A good uncertainty metric has PPV@k >> baseline_error_rate:
    e.g. baseline error 40%, PPV@20% = 0.70 means the flagged 20% contains 75%
    of errors — a strong filter.

    ALGORITHM
    ---------
    For each metric and k-fraction:
      1. Sort questions by descending uncertainty score.
      2. Take the top ceil(n * k) questions.
      3. PPV@k = fraction of those top-k that are incorrect.

    Returns
    -------
    {
      "vanilla_rag": {
          "semantic_entropy_ppv@10": float,
          "semantic_entropy_ppv@20": float,
          "semantic_entropy_ppv@30": float,
          ...
      },
      "kg_rag": { ... },
      "baseline": {"error_rate_vanilla": float, "error_rate_kg": float},
    }
    NaN when fewer than 4 questions available.
    """
    if not details:
        return {}

    if metric_names is None:
        metric_names = [
            "semantic_entropy", "discrete_semantic_entropy",
            "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq",
            "graph_path_support", "graph_path_disagreement",
            "competing_answer_alternatives", "evidence_vn_entropy",
            "subgraph_informativeness", "subgraph_perturbation_stability",
            "support_entailment_uncertainty", "evidence_conflict_uncertainty",
        ]

    output: Dict[str, Dict[str, float]] = {"vanilla_rag": {}, "kg_rag": {}, "baseline": {}}

    for system, prefix in (("vanilla_rag", "vanilla"), ("kg_rag", "kg")):
        system_details = details
        if exclude_generation_failures:
            system_details = [
                d for d in details
                if not bool(d.get(f"{prefix}_generation_failed", False))
            ]
        if len(system_details) < 4:
            continue

        y_true = np.array([
            1.0 if d.get(f"{prefix}_correct", False) else 0.0
            for d in system_details
        ])
        n = len(y_true)
        output["baseline"][f"error_rate_{prefix}"] = float(1.0 - y_true.mean())

        for metric in metric_names:
            metric_details = system_details
            if metric == "graph_path_support":
                metric_details = [
                    d for d in system_details
                    if not str(d.get(f"{prefix}_graph_path_support_null_reason", ""))
                ]
            elif metric == "subgraph_perturbation_stability":
                metric_details = [
                    d for d in system_details
                    if not str(d.get(f"{prefix}_subgraph_perturbation_stability_null_reason", ""))
                ]
            if len(metric_details) < 4:
                for frac in k_fractions:
                    pct = int(round(frac * 100))
                    output[system][f"{metric}_ppv@{pct}"] = float("nan")
                continue

            y_metric = np.array([
                1.0 if d.get(f"{prefix}_correct", False) else 0.0
                for d in metric_details
            ])
            raw = np.array([float(d.get(f"{prefix}_{metric}", 0.0)) for d in metric_details])
            if metric in _HIGHER_IS_MORE_CERTAIN:
                uncertainty = -raw
            else:
                uncertainty = raw

            order = np.argsort(-uncertainty)  # most uncertain first

            for frac in k_fractions:
                k = max(1, int(np.ceil(len(y_metric) * frac)))
                top_k_correct = y_metric[order[:k]]
                ppv = float(np.mean(top_k_correct == 0))  # fraction wrong in top-k
                pct = int(round(frac * 100))
                output[system][f"{metric}_ppv@{pct}"] = ppv

    return output


def compute_ece(
    details: List[Dict[str, Any]],
    metric_names: Optional[List[str]] = None,
    n_bins: int = 10,
    exclude_generation_failures: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Expected Calibration Error proxy for each uncertainty metric and RAG system.

    This computes a probability-like calibration proxy after converting each metric
    into a confidence score with within-run min-max scaling. It is not a
    train/validation-calibrated confidence model; it is a descriptive check of
    whether higher-confidence regions are empirically more accurate.

    ALGORITHM
    ---------
    For each metric:
      1. Convert the metric to uncertainty (higher = less certain).
      2. Min-max scale uncertainty to [0, 1], then map to confidence = 1 - scaled_uncertainty.
      3. Bin questions into n_bins equal-width bins by confidence score.
      4. In each bin: compute mean_confidence and empirical accuracy.
      5. ECE = weighted average of |mean_confidence - accuracy| over bins,
         weighted by bin size (# questions in bin / total questions).

    Metrics where HIGHER = MORE CERTAIN (p_true) are flipped before scaling so
    that the score always represents uncertainty (higher = less certain).

    Returns
    -------
    {
      "vanilla_rag": {"semantic_entropy_ece": float, "p_true_ece": float, ...},
      "kg_rag":      {"semantic_entropy_ece": float, ...},
    }
    NaN when too few bins are populated or all scores are identical.
    """
    if not details:
        return {}

    if metric_names is None:
        metric_names = [
            "semantic_entropy", "discrete_semantic_entropy",
            "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq",
            "graph_path_support", "graph_path_disagreement",
            "competing_answer_alternatives", "evidence_vn_entropy",
            "subgraph_informativeness", "subgraph_perturbation_stability",
            "support_entailment_uncertainty", "evidence_conflict_uncertainty",
        ]

    output: Dict[str, Dict[str, float]] = {"vanilla_rag": {}, "kg_rag": {}}

    for system, prefix in (("vanilla_rag", "vanilla"), ("kg_rag", "kg")):
        system_details = details
        if exclude_generation_failures:
            system_details = [
                d for d in details
                if not bool(d.get(f"{prefix}_generation_failed", False))
            ]
        if not system_details:
            continue

        for metric in metric_names:
            metric_details = system_details
            if metric == "graph_path_support":
                metric_details = [
                    d for d in system_details
                    if not str(d.get(f"{prefix}_graph_path_support_null_reason", ""))
                ]
            elif metric == "subgraph_perturbation_stability":
                metric_details = [
                    d for d in system_details
                    if not str(d.get(f"{prefix}_subgraph_perturbation_stability_null_reason", ""))
                ]
            if len(metric_details) < 4:
                output[system][f"{metric}_ece"] = float("nan")
                continue

            y_true = np.array([
                1.0 if d.get(f"{prefix}_correct", False) else 0.0
                for d in metric_details
            ])
            key = f"{prefix}_{metric}"
            raw = np.array([float(d.get(key, 0.0)) for d in metric_details])

            # Flip certainty-scored metrics so higher always = more uncertain
            if metric in _HIGHER_IS_MORE_CERTAIN:
                raw = -raw

            # Min-max scale to [0, 1]; skip if all values identical
            r_min, r_max = raw.min(), raw.max()
            if r_max - r_min < 1e-12:
                output[system][f"{metric}_ece"] = float("nan")
                continue
            scaled_uncertainty = (raw - r_min) / (r_max - r_min)
            confidence = 1.0 - scaled_uncertainty

            # Equal-width binning
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_indices = np.digitize(confidence, bin_edges[1:-1])  # 0-based bucket

            total_n = len(y_true)
            ece = 0.0
            populated = 0
            for b in range(n_bins):
                mask = bin_indices == b
                n_b = int(mask.sum())
                if n_b == 0:
                    continue
                populated += 1
                mean_conf = float(confidence[mask].mean())
                accuracy = float(y_true[mask].mean())
                ece += (n_b / total_n) * abs(mean_conf - accuracy)

            output[system][f"{metric}_ece"] = ece if populated >= max(2, n_bins // 2) else float("nan")

    return output


def compute_auroc_aurec(
    details: List[Dict[str, Any]],
    metric_names: Optional[List[str]] = None,
    exclude_generation_failures: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute AUROC and AUREC for each uncertainty metric and each RAG system.

    Parameters
    ----------
    details : list of per-question dicts containing {vanilla_correct, kg_correct,
              vanilla_<metric>, kg_<metric>} entries.
    metric_names : metrics to evaluate. Defaults to all 8 standard metrics.
    exclude_generation_failures : when True, selective-prediction metrics are
              computed on the same clean-evaluation population as headline
              accuracy, excluding rows where generation failed.

    Returns
    -------
    {
      "vanilla_rag": {"semantic_entropy_auroc": float, "semantic_entropy_aurec": float, ...},
      "kg_rag":      {"semantic_entropy_auroc": float, ...},
    }

    Notes
    -----
    AUROC  — Area Under the ROC Curve. Higher = metric better predicts correctness.
             0.5 = random; 1.0 = perfect discrimination.
             Requires at least one correct AND one incorrect question; returns NaN otherwise.

    AUREC  — Area Under the Rejection-Error Curve (selective prediction / abstention).
             Sort questions by DESCENDING uncertainty (reject most uncertain first).
             At rejection level k/N (k questions abstained on): compute error rate
             on the remaining N-k accepted (most confident) questions.
             error(0) = overall error rate; error(N-1) = error on single most-certain Q.
             AUREC = mean of error(k) across k = 0..N-1. Lower = better.
             Interpretation: a perfect uncertainty metric scores 0 AUREC (always abstains
             on wrong answers first); random scoring = overall_error_rate.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("scikit-learn not available — skipping AUROC/AUREC computation")
        return {}

    if not details:
        return {}

    if metric_names is None:
        metric_names = [
            "semantic_entropy", "discrete_semantic_entropy",
            "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq",
            "graph_path_support", "graph_path_disagreement",
            "competing_answer_alternatives", "evidence_vn_entropy",
            "subgraph_informativeness", "subgraph_perturbation_stability",
            "support_entailment_uncertainty", "evidence_conflict_uncertainty",
        ]

    def _auroc(y_true: np.ndarray, uncertainty: np.ndarray) -> float:
        """AUROC: higher uncertainty should predict incorrectness (y_true=0)."""
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, -uncertainty))

    def _aurec(y_true: np.ndarray, uncertainty: np.ndarray) -> float:
        """AUREC: Area Under the Rejection-Error Curve.

        Sort by descending uncertainty (reject most uncertain first).
        At each rejection level k, compute error rate on the remaining N-k questions.
        Return the mean error across all rejection levels.
        """
        n = len(y_true)
        if n == 0:
            return float("nan")
        # Sort descending by uncertainty → most uncertain rejected first
        order = np.argsort(-uncertainty)
        errors = (1 - y_true[order]).astype(float)  # 1 = incorrect

        # At rejection level k: keep questions [k:], i.e. the N-k most certain ones
        # error(k) = sum(errors[k:]) / (N - k)   for k = 0 .. N-1
        # At k = N-1: one question left; at k = N: all rejected → skip (undefined)
        suffix_errors = np.cumsum(errors[::-1])[::-1]  # suffix sums
        n_remaining = np.arange(n, 0, -1, dtype=float)  # N, N-1, ..., 1
        rejection_errors = suffix_errors / n_remaining   # error rate at each rejection level
        return float(rejection_errors.mean())

    output: Dict[str, Dict[str, float]] = {"vanilla_rag": {}, "kg_rag": {}}

    for system, prefix in (("vanilla_rag", "vanilla"), ("kg_rag", "kg")):
        system_details = details
        if exclude_generation_failures:
            system_details = [
                d for d in details
                if not bool(d.get(f"{prefix}_generation_failed", False))
            ]

        for metric in metric_names:
            metric_details = system_details
            if metric == "graph_path_support":
                metric_details = [
                    d for d in system_details
                    if not str(d.get(f"{prefix}_graph_path_support_null_reason", ""))
                ]
            elif metric == "subgraph_perturbation_stability":
                metric_details = [
                    d for d in system_details
                    if not str(d.get(f"{prefix}_subgraph_perturbation_stability_null_reason", ""))
                ]
            if len(metric_details) < 2:
                output[system][f"{metric}_auroc"] = float("nan")
                output[system][f"{metric}_aurec"] = float("nan")
                continue

            y_true = np.array([
                1.0 if d.get(f"{prefix}_correct", False) else 0.0
                for d in metric_details
            ])
            key = f"{prefix}_{metric}"
            raw_scores = np.array([float(d.get(key, 0.0)) for d in metric_details])

            # Normalise to an "uncertainty" score (higher = less certain)
            if metric in _HIGHER_IS_MORE_CERTAIN:
                uncertainty = -raw_scores
            else:
                uncertainty = raw_scores

            output[system][f"{metric}_auroc"] = _auroc(y_true, uncertainty)
            output[system][f"{metric}_aurec"] = _aurec(y_true, uncertainty)

        # GPS null-rate: fraction of rows where GPS returned an undefined 0.5.
        # Track every observed reason explicitly so per-reason totals stay in sync
        # with the overall null-rate even as new categories are introduced.
        # A high null-rate means AUROC for GPS is measured on a small effective sample.
        gps_null_key = f"{prefix}_graph_path_support_null_reason"
        null_reasons = [str(d.get(gps_null_key, "")) for d in system_details]
        n_total = len(null_reasons)
        if n_total > 0:
            observed_reasons = [
                reason
                for reason in _GPS_NULL_REASON_ORDER
                if any(r == reason for r in null_reasons)
            ]
            for reason in sorted({r for r in null_reasons if r} - set(observed_reasons)):
                observed_reasons.append(reason)
            for reason in observed_reasons:
                count = sum(1 for r in null_reasons if r == reason)
                output[system][f"graph_path_support_null_{reason}"] = count / n_total
            total_null = sum(1 for r in null_reasons if r)
            output[system]["graph_path_support_null_rate"] = total_null / n_total
            # AUROC computed on non-null GPS rows only (more honest estimate)
            non_null_details = [
                d for d in system_details
                if not str(d.get(gps_null_key, ""))
            ]
            if len(non_null_details) >= 2:
                y_nn = np.array([
                    1.0 if d.get(f"{prefix}_correct", False) else 0.0
                    for d in non_null_details
                ])
                gps_nn = np.array([
                    float(d.get(f"{prefix}_graph_path_support", 0.0))
                    for d in non_null_details
                ])
                output[system]["graph_path_support_auroc_non_null"] = _auroc(y_nn, gps_nn)
                output[system]["graph_path_support_aurec_non_null"] = _aurec(y_nn, gps_nn)

        # SPS-UQ null-rate: same abstention accounting as GPS.  The default
        # subgraph_perturbation_stability_auroc above is already conditional
        # on these non-null rows; the *_non_null aliases keep old analysis
        # notebooks explicit.
        sps_null_key = f"{prefix}_subgraph_perturbation_stability_null_reason"
        null_reasons = [str(d.get(sps_null_key, "")) for d in system_details]
        n_total = len(null_reasons)
        if n_total > 0:
            observed_reasons = [
                reason
                for reason in _SPS_NULL_REASON_ORDER
                if any(r == reason for r in null_reasons)
            ]
            for reason in sorted({r for r in null_reasons if r} - set(observed_reasons)):
                observed_reasons.append(reason)
            for reason in observed_reasons:
                count = sum(1 for r in null_reasons if r == reason)
                output[system][f"subgraph_perturbation_stability_null_{reason}"] = count / n_total
            total_null = sum(1 for r in null_reasons if r)
            output[system]["subgraph_perturbation_stability_null_rate"] = total_null / n_total
            non_null_details = [
                d for d in system_details
                if not str(d.get(sps_null_key, ""))
            ]
            if len(non_null_details) >= 2:
                y_nn = np.array([
                    1.0 if d.get(f"{prefix}_correct", False) else 0.0
                    for d in non_null_details
                ])
                sps_nn = np.array([
                    float(d.get(f"{prefix}_subgraph_perturbation_stability", 0.0))
                    for d in non_null_details
                ])
                output[system]["subgraph_perturbation_stability_auroc_non_null"] = _auroc(y_nn, sps_nn)
                output[system]["subgraph_perturbation_stability_aurec_non_null"] = _aurec(y_nn, sps_nn)

    return output


# Backward-compatible alias
compute_auroc_aurc = compute_auroc_aurec
