import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ontographrag.rag.systems._constants import RAG_CONFIG

class QueryPlanningMixin:
    """Question classification, query shaping, fusion, and decomposition.

    Mixin for :class:`EnhancedRAGSystem`; method bodies are unchanged
    from the original monolithic implementation.
    """

    @staticmethod
    def _comparison_branches(query: str) -> List[str]:
        """Extract simple left/right comparison branches from `A or B` style questions."""
        query = str(query or "").strip().rstrip("?")
        if not query or " or " not in query.lower():
            return []
        parts = re.split(r"\bor\b", query, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            return []
        left, right = parts[0].strip(), parts[1].strip()
        if "," in left:
            left = left.split(",")[-1].strip()
        left = re.sub(
            r"^(which|what|who|where|when|whose|is|are|was|were|has|have|had|does|do|did)\s+",
            "",
            left,
            flags=re.IGNORECASE,
        ).strip()
        branches = [branch.strip(" ,.;:") for branch in (left, right)]
        return [branch for branch in branches if len(branch) >= 4]

    @classmethod
    def _comparison_branch_coverage(cls, query: str, chunks: List[Dict[str, Any]]) -> int:
        branches = cls._comparison_branches(query)
        if not branches:
            return 0
        covered = 0
        for branch in branches:
            branch_norm = branch.lower()
            if any(
                branch_norm in str(chunk.get("text", "")).lower()
                or branch_norm in str(chunk.get("document", "")).lower()
                for chunk in chunks
            ):
                covered += 1
        return covered

    @classmethod
    def _comparison_branch_match_count(cls, query: str, chunk: Dict[str, Any]) -> int:
        branches = cls._comparison_branches(query)
        if not branches:
            return 0
        haystack = " ".join(
            filter(
                None,
                [
                    str(chunk.get("document", "")).lower(),
                    str(chunk.get("text", "")).lower(),
                ],
            )
        )
        return sum(1 for branch in branches if branch.lower() in haystack)

    @classmethod
    def _lexical_query_overlap_count(cls, query: str, chunk: Dict[str, Any]) -> int:
        query_tokens = cls._content_query_tokens(query)
        if not query_tokens:
            return 0
        haystack = " ".join(
            filter(
                None,
                [
                    str(chunk.get("document", "")).lower(),
                    str(chunk.get("text", "")).lower(),
                ],
            )
        )
        return sum(1 for token in query_tokens if token in haystack)

    @staticmethod
    def _normalize_retrieval_query(query: str) -> str:
        return re.sub(r"\s+", " ", str(query or "")).strip()

    @classmethod
    def _query_fusion_enabled(cls) -> bool:
        return str(os.getenv("ONTOGRAPHRAG_QUERY_FUSION", "1")).strip().lower() not in {
            "0", "false", "off", "no",
        }

    @classmethod
    def _should_run_query_fusion(
        cls,
        question: str,
        context: Optional[Dict[str, Any]],
        *,
        max_hops: int,
    ) -> bool:
        if not cls._query_fusion_enabled():
            return False

        branches = cls._comparison_branches(question)
        if branches:
            coverage = cls._comparison_branch_coverage(question, list((context or {}).get("chunks", [])))
            if coverage < len(branches):
                return True

        if max_hops >= 2 and str((context or {}).get("search_method") or "") != "iterative_hop":
            return True

        content_tokens = cls._content_query_tokens(question)
        if len(content_tokens) >= 8 and len(list((context or {}).get("chunks", []))) < 4:
            return True
        return False

    def _generate_query_variants(
        self,
        question: str,
        llm,
        *,
        max_hops: int,
    ) -> List[str]:
        variants: List[str] = []
        seen: Set[str] = {self._normalize_retrieval_query(question).lower()}

        for branch in self._comparison_branches(question):
            branch_query = self._normalize_retrieval_query(
                f"Focus on {branch}. Original question: {question}"
            )
            branch_key = branch_query.lower()
            if branch_query and branch_key not in seen:
                seen.add(branch_key)
                variants.append(branch_query)

        if llm is None:
            return variants[: self._QUERY_FUSION_MAX_VARIANTS]

        remaining = self._QUERY_FUSION_MAX_VARIANTS - len(variants)
        if remaining <= 0:
            return variants[: self._QUERY_FUSION_MAX_VARIANTS]

        is_complex = bool(self._comparison_branches(question)) or max_hops >= 2 or len(self._content_query_tokens(question)) >= 8
        if not is_complex:
            return variants[: self._QUERY_FUSION_MAX_VARIANTS]

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are optimizing retrieval queries for RAG. Produce up to {n} alternative "
                "search queries that preserve the original question's constraints exactly while "
                "surfacing missing evidence. Return ONLY a JSON array of strings.\n\n"
                "Rules:\n"
                "1. Do not answer the question.\n"
                "2. Keep names, labels, comparison targets, and temporal constraints intact.\n"
                "3. Prefer short evidence-seeking queries.\n"
                "4. If the question compares two targets, at least one query should foreground each target.\n"
                "5. If no useful reformulation exists, return []."
            )),
            ("human", "{question}"),
        ])
        try:
            chain = prompt | llm | StrOutputParser()
            raw = chain.invoke({"question": question, "n": remaining})
            match = re.search(r"\[.*\]", str(raw), re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    for item in parsed:
                        candidate = self._normalize_retrieval_query(str(item))
                        candidate_key = candidate.lower()
                        if not candidate or candidate_key in seen or len(candidate) > 220:
                            continue
                        seen.add(candidate_key)
                        variants.append(candidate)
                        if len(variants) >= self._QUERY_FUSION_MAX_VARIANTS:
                            break
        except Exception as exc:
            logging.debug("Query fusion reformulation failed (non-fatal): %s", exc)

        return variants[: self._QUERY_FUSION_MAX_VARIANTS]

    def classify_question_type(self, query: str) -> str:
        """
        Classify the question type to determine retrieval strategy.

        Returns one of: "comparison", "bridge", "statistical", "semantic", "generic".
        "comparison" and "bridge" are checked first because they drive hard routing
        decisions (comparison suppresses graph traversal; bridge enables it).
        """
        query_lower = query.lower().strip().rstrip("?")

        # --- Comparison: parallel attribute lookup across two named entities ---
        comparison_patterns = [
            # "are/do/did/were/is/have both ..."
            r"\bare both\b", r"\bdo both\b", r"\bdid both\b", r"\bwere both\b",
            r"\bis both\b", r"\bhave both\b",
            # "both X and Y ..." at start or after wh-word
            r"\bboth\b.{1,60}\band\b",
            # "are/is X and Y both ..." — subject before "both"
            r"\band\b.{1,60}\bboth\b",
            # "X and Y share", "X and Y are the same", "X and Y both"
            r"\band\b.{1,80}\bshare\b",
            r"\bsame\b.{1,40}\b(breed|species|profession|nationality|type|genre|band|group)\b",
            # "between X and Y, who/which" — explicit comparison framing
            r"^between\b",
            # "in between X and Y"
            r"^in between\b",
        ]
        if any(re.search(p, query_lower) for p in comparison_patterns):
            return "comparison"
        # "X or Y" style with a named entity on each side (handled by _comparison_branches)
        if len(self._comparison_branches(query)) == 2:
            return "comparison"

        # --- Bridge: multi-hop entity chain (A→B→answer) ---
        # HotpotQA bridge questions typically ask about a property of an entity
        # that is itself defined by a chain through another entity.
        # Heuristic: wh-question that doesn't match comparison and contains
        # at least two named-entity-like tokens (capitalised mid-sentence tokens).
        bridge_starters = ["who", "what", "which", "where", "when", "whose"]
        if any(query_lower.startswith(s) for s in bridge_starters):
            # Count plausible named entity tokens (title-case words not at start)
            words = query.split()
            ne_count = sum(
                1 for w in words[1:]
                if w and w[0].isupper() and w.isalpha() and len(w) >= 3
            )
            if ne_count >= 2:
                return "bridge"

        # --- Statistical ---
        statistical_terms = [
            "statistic", "tendencies", "trend", "correlation", "rate", "incidence",
            "prevalence", "distribution", "frequency", "proportion", "percentage",
            "average", "mean", "median", "variance", "standard deviation",
            "regression", "p-value", "significance", "confidence interval",
            "sample size", "cohort", "meta-analysis", "epidemiology",
            "how many", "how much", "what percentage", "what proportion",
            "quantity", "quantity of", "number of", "count", "total", "sum"
        ]
        if any(term in query_lower for term in statistical_terms):
            return "statistical"
        quantitative_starters = ["how many", "how much", "what percentage", "what proportion"]
        if any(query_lower.startswith(s) for s in quantitative_starters):
            return "statistical"

        # --- Semantic ---
        semantic_terms = [
            "explain", "describe", "what is", "how does", "define", "meaning",
            "concept", "principle", "theory", "framework", "model", "interpretation",
            "understanding", "overview", "context", "background", "history",
            "development", "evolution", "mechanism", "process", "function"
        ]
        if any(term in query_lower for term in semantic_terms):
            return "semantic"
        semantic_starters = ["what is", "how does", "explain", "describe"]
        if any(query_lower.startswith(s) for s in semantic_starters):
            return "semantic"

        return "generic"

    def calculate_dynamic_threshold(self, query: str, entity_count: int = 0) -> float:
        """
        Calculate dynamic similarity threshold based on question type and context
        """
        question_type = self.classify_question_type(query)
        config = RAG_CONFIG.get(question_type, RAG_CONFIG["generic"])

        if question_type == "statistical":
            # Lower threshold for statistical queries to catch more data
            base_threshold = max(config["threshold_floor"], 0.08 - (entity_count * config["threshold_factor"]))
            return min(base_threshold, 0.15)

        elif question_type == "semantic":
            # Slightly higher threshold for focused semantic questions
            base_threshold = min(config["threshold_ceiling"], 0.08 + config["threshold_boost"])
            return max(base_threshold, 0.06)

        else:  # generic
            return config["default_threshold"]

    def get_adaptive_retrieval_params(self, query: str) -> Dict[str, Any]:
        """
        Get adaptive retrieval parameters based on question classification
        """
        question_type = self.classify_question_type(query)

        # For statistical questions, use reasonable chunk limit to avoid timeouts
        if question_type == "statistical":
            max_chunks = 200  # Aggressive limit to prevent timeouts - focus on quality over quantity
        else:
            max_chunks = RAG_CONFIG.get(question_type, RAG_CONFIG["generic"])["default_max_chunks"]

        params = {
            "question_type": question_type,
            "similarity_threshold": self.calculate_dynamic_threshold(query, entity_count=0),
            "max_chunks": max_chunks
        }

        logging.info(f"Question '{query[:50]}...' classified as '{question_type}': threshold={params['similarity_threshold']:.3f}, max_chunks={params['max_chunks']} (total available: all for stats)")
        return params

    @classmethod
    def _content_query_tokens(cls, query: str) -> List[str]:
        """Return de-duplicated content-bearing query tokens for entity grounding."""
        seen: Set[str] = set()
        tokens: List[str] = []
        for token in re.findall(r"[A-Za-z0-9]+", query.lower()):
            if len(token) < 4 or token in cls._ENTITY_MATCH_STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        return tokens

    @classmethod
    def _count_matched_query_tokens(cls, query: str, entity_names: List[str]) -> int:
        """Count how many distinct content tokens from the query are covered by entity names."""
        query_tokens = cls._content_query_tokens(query)
        if not query_tokens:
            return 0

        covered: Set[str] = set()
        for entity_name in entity_names:
            entity_name_norm = re.sub(r"\s+", " ", str(entity_name or "").lower()).strip()
            if not entity_name_norm:
                continue
            entity_tokens = {
                token
                for token in re.findall(r"[A-Za-z0-9]+", entity_name_norm)
                if len(token) >= 4
            }
            for token in query_tokens:
                if token in entity_tokens or re.search(rf"(?<!\w){re.escape(token)}(?!\w)", entity_name_norm):
                    covered.add(token)
        return len(covered)

    @classmethod
    def _grounding_quality(cls, query: str, matched_query_token_count: int) -> float:
        """
        Fraction of content-bearing query tokens that were grounded in matched entity names.
        Used as a routing meta-signal: high grounding → structural metrics are reliable;
        low grounding → fall back to generative metrics.
        """
        content_words = cls._content_query_tokens(query)
        if not content_words:
            return 0.0
        return min(1.0, matched_query_token_count / len(content_words))

    def _extract_query_entities(self, query: str, llm) -> List[str]:
        """
        Extract named entity mentions from a question using the LLM.

        Returns a list of short entity strings (e.g. ["TBK1", "IRF3"]).
        Results are cached by query text.  On any failure returns an empty list
        so the caller falls back to the raw-query-embedding ANN pass.
        """
        model_key = type(llm).__name__ + getattr(llm, "model_name", getattr(llm, "model", ""))
        cache_key = (query, model_key)
        if cache_key in self._entity_extraction_cache:
            return self._entity_extraction_cache[cache_key]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract all named entities (people, organisations, places, concepts, "
             "medical terms, genes, chemicals, events) mentioned in the question. "
             "Return ONLY a JSON array of strings, nothing else. "
             'Example: ["Marie Curie", "Poland", "radioactivity"]'),
            ("human", "{question}"),
        ])
        try:
            chain = prompt | llm | StrOutputParser()
            raw = chain.invoke({"question": query})
            raw = raw.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw.strip())
            entities = json.loads(raw)
            if not isinstance(entities, list):
                entities = []
            cleaned_entities: List[str] = []
            seen_entities: Set[str] = set()
            for entity in entities:
                normalized = re.sub(r"\s+", " ", str(entity).strip())
                if not normalized:
                    continue
                entity_key = normalized.casefold()
                if entity_key in seen_entities:
                    continue
                seen_entities.add(entity_key)
                cleaned_entities.append(normalized)
                if len(cleaned_entities) >= self._MAX_EXTRACTED_QUERY_ENTITIES:
                    break
            entities = cleaned_entities
        except Exception as exc:
            logging.debug("Query entity extraction failed (%s); falling back to raw query embedding.", exc)
            entities = []

        self._entity_extraction_cache[cache_key] = entities
        return entities

    def _decompose_question(self, question: str, llm, max_hops: int = 2) -> List[str]:
        """
        Decompose a multi-hop question into an ordered list of sub-questions.

        Each sub-question targets one reasoning hop; the bridge answer from hop N
        is substituted into hop N+1's sub-question before retrieval.  Falls back
        to [question] on any failure so the caller always receives a usable list.
        """
        n_hops = max(2, min(max_hops, 4))
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a reasoning assistant. Decompose the following multi-hop question "
                "into exactly {n} ordered sub-questions, each targeting exactly one reasoning step. "
                "Return ONLY a JSON array of strings, with no prose.\n\n"
                "Rules:\n"
                "1. Resolve nested references from the inside out.\n"
                "2. Preserve the original semantics exactly; do not broaden or rewrite predicates.\n"
                "3. When a later hop depends on an earlier answer, use the literal token [BRIDGE].\n"
                "4. Keep each sub-question locally answerable in one hop.\n\n"
                "Example 1:\n"
                "Question: Where is the headquarters of the Radio Television of the country whose co-official language is the same as the one Politika is written in?\n"
                "Output: [\"What language is Politika written in?\", \"Which country has [BRIDGE] as a co-official language?\", \"Where is the headquarters of the Radio Television of [BRIDGE]?\"]\n\n"
                "Example 2:\n"
                "Question: Who is the father-in-law of Helena Palaiologina, Despotess of Serbia?\n"
                "Output: [\"Who is Helena Palaiologina, Despotess of Serbia married to?\", \"Who is [BRIDGE]'s father?\"]"
            )),
            ("human", "{question}"),
        ])
        try:
            chain = prompt | llm | StrOutputParser()
            raw = chain.invoke({"question": question, "n": n_hops})
            # Extract JSON array from response (tolerate surrounding text)
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                sub_qs = json.loads(m.group(0))
                if isinstance(sub_qs, list) and len(sub_qs) >= 2:
                    # Hard-cap to n_hops so a verbose LLM can't silently exceed max_hops
                    sub_qs = sub_qs[:n_hops]
                    sub_qs = [str(q).strip() for q in sub_qs if str(q).strip()]
                    logging.info("Decomposed into %d sub-questions: %s", len(sub_qs), sub_qs)
                    return sub_qs
        except Exception as e:
            logging.warning("Question decomposition failed: %s", e)
        return [question]
