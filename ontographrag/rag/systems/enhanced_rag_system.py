import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from ontographrag.kg.utils.common_functions import load_embedding_model
from ontographrag.rag.answer_guardrails import (
    RUNTIME_GUARDRAIL_ABSTENTION,
    evaluate_runtime_answer_guardrail,
)
from ontographrag.rag.retrieval_sampling import (
    compute_candidate_limit,
    select_ranked_subset,
)
from ontographrag.rag.reranking import (
    late_interaction_enabled,
    late_interaction_rescore_chunks_for_query,
    rerank_chunks_for_query,
)

# Configurable parameters for different question types
RAG_CONFIG = {
    "statistical": {
        "default_max_chunks": 100,  # More chunks for statistical analysis
        "threshold_floor": 0.05,
        "threshold_factor": 0.03
    },
    "semantic": {
        "default_max_chunks": 15,  # Fewer chunks for focused semantic questions
        "threshold_floor": 0.08,
        "threshold_ceiling": 0.15,
        "threshold_boost": 0.02
    },
    "generic": {
        "default_max_chunks": 20,  # Default chunk count
        "default_threshold": 0.08
    }
}

class EnhancedRAGSystem:
    """
    Enhanced RAG System that properly connects to the knowledge graph with embeddings
    """
    _ENTITY_MATCH_STOPWORDS: Set[str] = {
        "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
        "does", "do", "did", "done", "have", "has", "had", "with", "from",
        "that", "this", "these", "those", "into", "about", "after", "before",
        "through", "there", "their", "them", "they", "would", "could", "should",
        "question", "answer", "answers", "company", "county", "state", "city",
        "country", "province", "league", "team", "film", "band", "album",
        "series", "season", "where", "owner", "located", "founded", "born",
        "spouse", "husband", "wife", "plays", "performed", "performer",
    }
    _MAX_ENTITY_SEEDS = 12
    _ENTITY_LOOKUP_LIMIT = 40
    _MAX_EXTRACTED_QUERY_ENTITIES = 8
    _ITERATIVE_SUBQUESTION_MAX_HOPS = 3
    _HYBRID_SUPPLEMENT_LIMIT = 5
    _WEAK_GRAPH_GROUNDING_THRESHOLD = 0.5
    _ANCHOR_LATER_HOP_RETRIEVAL_MIN = 12
    _ANCHOR_LATER_HOP_BRIDGE_PASSAGE_LIMIT = 8
    _GRAPH_TRAVERSAL_SEED_LIMIT = 24
    _GRAPH_TRAVERSAL_NEIGHBOR_LIMIT = 30
    _GRAPH_TRAVERSAL_MAX_HOPS_BY_KG: Dict[str, int] = {
        # Shared-corpus MultihopRAG graphs get dense very quickly under open-ended
        # expansion; keep graph traversal supplemental rather than exhaustive.
        "multihoprag": 2,
    }
    _GRAPH_TRAVERSAL_SEED_LIMIT_BY_KG: Dict[str, int] = {
        "multihoprag": 12,
    }
    _GRAPH_TRAVERSAL_NEIGHBOR_LIMIT_BY_KG: Dict[str, int] = {
        "multihoprag": 12,
    }
    _ITERATIVE_SUBQUESTION_MAX_HOPS_BY_KG: Dict[str, int] = {
        # 2Wiki sub-questions are usually already local bridge steps; allowing
        # deeper local traversals increases path drift more than it helps.
        "2wikimultihopqa": 1,
    }
    _ORIGINAL_QUESTION_ANCHOR_KGS: Set[str] = {
        "musique",
        "multihoprag",
    }
    # Pre-traversal disambiguation confidence gate.
    # If grounding_q < threshold AND no matched entity has a symbolic signal
    # (exact substring or ≥2 token overlaps), skip graph traversal entirely
    # and fall back to vector search.  An embedding-only anchor on a noisy
    # open-domain graph is more likely to start traversal from the wrong node
    # than to retrieve useful evidence.
    # Minimum grounding fraction required before graph traversal runs.
    # If the matched entity tokens cover less than this fraction of the
    # meaningful query words AND no entity has a symbolic match (exact
    # substring or ≥2 token overlap), traversal is skipped in favour of
    # vector search — an embedding-only anchor on a noisy graph is more
    # likely to start from the wrong node than to retrieve useful evidence.
    #
    # Default 0.25: skip traversal when fewer than 1-in-4 query content words
    # are matched by any KG entity.  Per-KG overrides tune this for each
    # dataset's vocabulary density.  Biomedical datasets (bioasq, pubmedqa,
    # realmedqa) use a slightly higher bar because their entity names are
    # highly specific and a low-grounding anchor almost always indicates a
    # failed extraction rather than a legitimate sparse question.
    _ENTITY_MATCH_MIN_GROUNDING: float = 0.25
    _ENTITY_MATCH_MIN_GROUNDING_BY_KG: Dict[str, float] = {
        "hotpotqa":        0.30,
        "2wikimultihopqa": 0.25,
        "musique":         0.25,
        "multihoprag":     0.20,
        "bioasq":          0.30,
        "pubmedqa":        0.30,
        "realmedqa":       0.30,
    }
    # On source-document biomedical tasks, KG structure is useful for finding
    # and linking passages, but relation extraction can still compress away
    # negation or study-level hedging. Keep final answer generation passage-led
    # here so KG-RAG cannot underperform vanilla purely because of graph
    # serialization noise.
    _PASSAGE_ONLY_ANSWER_KGS = frozenset({
        "pubmedqa",
        "realmedqa",
    })
    _QUERY_FUSION_MAX_VARIANTS = 3
    _QUERY_FUSION_RRF_K = 60

    # Valid retrieval modes for the retrieval_mode constructor param.
    RETRIEVAL_MODES = frozenset({"hybrid_auto", "entity_first", "rfge", "vector_only"})

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "neo4j",
        embedding_model: str = None,  # Use environment variable if not provided
        # --- Retrieval mode ---
        # hybrid_auto : entity_first → RFGE → vector (default, best recall)
        # entity_first: question-entity ANN + symbolic only, no RFGE fallback
        # rfge        : retriever-first graph expansion only, skip entity_first
        # vector_only : pure dense retrieval, no graph expansion
        retrieval_mode: str = "hybrid_auto",
        # --- Component ablation toggles ---
        # Set any to False to disable a specific component for ablation experiments.
        use_per_entity_ann: bool = True,    # NER-based query entity extraction before ANN
        use_node_specificity: bool = True,  # hub-entity down-weighting in ANN scores
        use_ppr_scoring: bool = True,       # PPR power iteration (vs fixed hop scores)
        use_rfge: bool = True,              # retriever-first graph expansion fallback
        use_evidence_block: bool = True,    # KG2RAG-style chain-grouped evidence format
        use_rog_path_planning: bool = True, # RoG-style relation-path-guided traversal
        allow_vector_augmentation: bool = True,  # merge vector evidence after graph retrieval
        allow_vector_fallback: bool = True,      # retry/fallback to vector when graph is thin
        max_chunks_per_passage: int = None, # per-passage chunk cap (None → class default)
        traversal_chunk_min_sim: float = None,  # min query-chunk cosine for hop-1+ chunks
    ):
        # Load Neo4j credentials from environment variables if not provided
        self.neo4j_uri = neo4j_uri if neo4j_uri is not None else os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user if neo4j_user is not None else os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password if neo4j_password is not None else os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise ValueError("Neo4j connection parameters not found. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")

        self._vector_index_available: Optional[bool] = None  # cached after first check
        self._active_vector_index_name: Optional[str] = None
        self._graph: Optional[object] = None               # shared Neo4j connection
        self._embedding_cache: dict = {}                    # query text → embedding vector
        self._entity_extraction_cache: dict = {}            # (query, model_key) → entity strings
        self._late_interaction_corpus_cache: dict = {}      # scoped corpus rows for LI retrieval

        if retrieval_mode not in self.RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {sorted(self.RETRIEVAL_MODES)}, got {retrieval_mode!r}"
            )
        self.retrieval_mode = retrieval_mode
        self.use_per_entity_ann = use_per_entity_ann
        self.use_node_specificity = use_node_specificity
        self.use_ppr_scoring = use_ppr_scoring
        self.use_rfge = use_rfge
        self.use_evidence_block = use_evidence_block
        self.use_rog_path_planning = use_rog_path_planning
        self.allow_vector_augmentation = allow_vector_augmentation
        self.allow_vector_fallback = allow_vector_fallback
        if max_chunks_per_passage is not None:
            self._MAX_CHUNKS_PER_PASSAGE = max_chunks_per_passage
        if traversal_chunk_min_sim is not None:
            self._TRAVERSAL_CHUNK_MIN_SIM = traversal_chunk_min_sim

        # Initialize embeddings through the shared loader so KG construction and
        # query-time retrieval use the same backend/model defaults.
        embedding_provider = (
            embedding_model
            or os.getenv("EMBEDDING_PROVIDER")
            or os.getenv("EMBEDDING_MODEL", "sentence_transformers")
        )
        self.embedding_model, self.embedding_dimension = load_embedding_model(embedding_provider)
        logging.info(
            "Initialized retrieval embedding backend '%s' with dimension %d",
            embedding_provider,
            self.embedding_dimension,
        )

        # KG-RAG prompt: KG2RAG-style evidence organization.
        # Evidence is pre-organized into reasoning chains (graph paths + their
        # supporting passages) before generation, making multi-hop evidence explicit
        # and reducing the chance the LLM invents bridge steps.
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable AI assistant. Answer the question using the provided evidence.

The evidence is organized in up to three sections:

1. REASONING CHAINS — each chain is a multi-hop graph path with the text passages that support it.
   Format:  Chain N (K hops): A --RELATION--> B --RELATION--> C
            [P1] passage text ...
   Passages listed under a chain directly mention the entities in that chain.
   Prefer evidence from chains where BOTH the path and the supporting passages confirm the same fact.

2. ADDITIONAL PASSAGES — passages retrieved by semantic similarity not on any chain.
   Use these for context and corroboration.

3. STRUCTURAL HINTS — graph paths with no retrieved passage support.
   Treat these as weak background structure only. Do NOT use them as evidence unless
   sections 1 and 2 contain no relevant passages at all.

HOW TO REASON:
- For source-document tasks (biomedical classification, yes/no questions backed by an abstract):
  let the study conclusion or finding in the PASSAGES govern the answer.
  Graph paths are auxiliary; never let a bare structural hint override a clear passage answer.
- Follow the task-specific answer instructions below when they are provided.
- If the task-specific instructions require an exact label-only answer, obey them exactly and do not add explanation.
- If the task-specific instructions require a short answer only, give only that short answer and do not add explanation.
- For multi-hop questions: trace the reasoning chain step by step (e.g. "Chain 1 shows A→B; Chain 2 shows B→C; therefore A→C").
- Evidence that appears in BOTH a chain path AND its supporting passages is the strongest signal.

IMPORTANT:
- Unless task-specific instructions say otherwise, for yes/no questions begin with "Yes" or "No" followed by your explanation.
- Ground every claim in the provided evidence; do not invent facts.
- If the answer requires connecting two pieces of evidence, make the inference explicitly.
- Only say the context is insufficient if you genuinely cannot find any relevant evidence.

Task-Specific Answer Instructions:
{answer_instructions}

{evidence_block}

Question: {question}"""),
            ("human", "{question}")
        ])

    def _iterative_subquestion_max_hops_for_kg(
        self,
        kg_name: Optional[str],
        dataset_max_hops: int,
    ) -> int:
        """Return the per-subquestion traversal cap for iterative retrieval."""
        kg_key = str(kg_name or "").strip().lower()
        local_cap = self._ITERATIVE_SUBQUESTION_MAX_HOPS_BY_KG.get(
            kg_key,
            self._ITERATIVE_SUBQUESTION_MAX_HOPS,
        )
        return max(1, min(dataset_max_hops, local_cap))

    @classmethod
    def _graph_traversal_seed_limit_for_kg(cls, kg_name: Optional[str]) -> int:
        kg_key = str(kg_name or "").strip().lower()
        return cls._GRAPH_TRAVERSAL_SEED_LIMIT_BY_KG.get(
            kg_key,
            cls._GRAPH_TRAVERSAL_SEED_LIMIT,
        )

    @classmethod
    def _graph_traversal_neighbor_limit_for_kg(
        cls,
        kg_name: Optional[str],
        requested_limit: int,
    ) -> int:
        kg_key = str(kg_name or "").strip().lower()
        dataset_limit = cls._GRAPH_TRAVERSAL_NEIGHBOR_LIMIT_BY_KG.get(
            kg_key,
            cls._GRAPH_TRAVERSAL_NEIGHBOR_LIMIT,
        )
        return max(1, min(requested_limit, dataset_limit))

    @classmethod
    def _graph_traversal_max_hops_for_kg(
        cls,
        kg_name: Optional[str],
        requested_max_hops: int,
    ) -> int:
        kg_key = str(kg_name or "").strip().lower()
        dataset_limit = cls._GRAPH_TRAVERSAL_MAX_HOPS_BY_KG.get(
            kg_key,
            requested_max_hops,
        )
        return max(1, min(requested_max_hops, dataset_limit))

    @staticmethod
    def _chunk_identity(chunk: Dict[str, Any]) -> str:
        return (
            chunk.get("chunk_id")
            or chunk.get("chunk_element_id")
            or str(chunk.get("text", ""))[:80]
        )

    @staticmethod
    def _relationship_identity(rel: Dict[str, Any]) -> str:
        return (
            rel.get("key")
            or rel.get("element_id")
            or (
                f"{rel.get('source')}-{rel.get('type')}-{rel.get('target')}"
                f"-negated={bool(rel.get('negated', False))}"
                f"-condition={str(rel.get('condition') or '').strip()}"
                f"-quantitative={str(rel.get('quantitative') or '').strip()}"
            )
        )

    @staticmethod
    def _format_relationship_label(rel: Optional[Dict[str, Any]]) -> str:
        if not isinstance(rel, dict):
            return "RELATED"
        label = str(rel.get("type") or "RELATED").strip() or "RELATED"
        if rel.get("negated"):
            label = f"NOT {label}"
        condition = str(rel.get("condition") or "").strip()
        if condition:
            label = f"{label} [{condition}]"
        quantitative = str(rel.get("quantitative") or "").strip()
        if quantitative and quantitative.lower() not in condition.lower():
            label = f"{label} ({quantitative})"
        return label

    @classmethod
    def _should_use_passage_only_answer_prompt(cls, kg_name: Optional[str]) -> bool:
        return str(kg_name or "").strip().lower() in cls._PASSAGE_ONLY_ANSWER_KGS

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

    @classmethod
    def _iterative_retrieval_query(
        cls,
        question: str,
        sub_question: str,
        kg_name: Optional[str],
        hop_idx: int,
        next_sub_question: Optional[str] = None,
    ) -> str:
        if hop_idx <= 0 or kg_name not in cls._ORIGINAL_QUESTION_ANCHOR_KGS:
            return sub_question

        question_clean = " ".join(str(question or "").split())
        sub_question_clean = " ".join(str(sub_question or "").split())
        if not question_clean or not sub_question_clean:
            return sub_question
        if question_clean in sub_question_clean:
            return sub_question
        parts = [sub_question_clean, f"Original question: {question_clean}"]
        next_sub_question_clean = " ".join(str(next_sub_question or "").split())
        if next_sub_question_clean:
            parts.append(f"Next hop target: {next_sub_question_clean}")
        return "\n".join(parts)

    @classmethod
    def _iterative_hop_retrieval_budget(
        cls,
        kg_name: Optional[str],
        hop_idx: int,
        base_hop_max: int,
    ) -> int:
        if hop_idx > 0 and kg_name in cls._ORIGINAL_QUESTION_ANCHOR_KGS:
            return max(base_hop_max, cls._ANCHOR_LATER_HOP_RETRIEVAL_MIN)
        return base_hop_max

    @classmethod
    def _sort_chunks_for_query(cls, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        branches = cls._comparison_branches(query)
        return sorted(
            chunks,
            key=lambda c: (
                -cls._comparison_branch_match_count(query, c) if branches else 0,
                -cls._lexical_query_overlap_count(query, c),
                -int(c.get("linked_entity_count", 0)),
                -float(c.get("score", 0.0)),
                1 if c.get("adjacent") else 0,
                int(c.get("position") or 0),
            ),
        )

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

    @classmethod
    def _retained_entity_ids_from_chunks(cls, chunks: List[Dict[str, Any]]) -> Set[str]:
        retained_eids: Set[str] = set()
        for chunk in chunks:
            for eid in chunk.get("linked_entity_ids") or []:
                if eid:
                    retained_eids.add(str(eid))
            for entity in chunk.get("entities") or []:
                eid = entity.get("id") if isinstance(entity, dict) else None
                if eid:
                    retained_eids.add(str(eid))
        return retained_eids

    @staticmethod
    def _selected_chunk_local_scope(
        chunks: List[Dict[str, Any]],
    ) -> Tuple[Set[int], Set[str], Set[str]]:
        positions: Set[int] = set()
        question_ids: Set[str] = set()
        passage_keys: Set[str] = set()
        for chunk in chunks:
            pos = chunk.get("position")
            if isinstance(pos, (int, float)):
                positions.add(int(pos))
            qid = chunk.get("question_id")
            if qid is not None and str(qid).strip():
                qid_str = str(qid)
                question_ids.add(qid_str)
                passage_keys.add(f"{qid_str}::p{chunk.get('passage_index', -1)}")
        return positions, question_ids, passage_keys

    @staticmethod
    def _relationship_supported_by_selected_chunks(
        rel: Dict[str, Any],
        *,
        selected_positions: Set[int],
        selected_question_ids: Set[str],
        selected_passage_keys: Set[str],
    ) -> bool:
        rel_positions = {
            int(pos)
            for pos in (rel.get("provenance_positions") or [])
            if isinstance(pos, (int, float))
        }
        if rel_positions:
            return bool(rel_positions & selected_positions)

        rel_passage_keys = {
            str(key)
            for key in (rel.get("passage_keys") or [])
            if str(key).strip()
        }
        if rel_passage_keys:
            return bool(rel_passage_keys & selected_passage_keys)

        rel_question_ids = {
            str(qid)
            for qid in (rel.get("question_ids") or [])
            if str(qid).strip()
        }
        if rel_question_ids:
            return bool(rel_question_ids & selected_question_ids)

        return True

    @classmethod
    def _path_supported_by_selected_chunks(
        cls,
        path_entry: Dict[str, Any],
        *,
        selected_positions: Set[int],
        selected_question_ids: Set[str],
        selected_passage_keys: Set[str],
    ) -> bool:
        return cls._relationship_supported_by_selected_chunks(
            path_entry,
            selected_positions=selected_positions,
            selected_question_ids=selected_question_ids,
            selected_passage_keys=selected_passage_keys,
        )

    @classmethod
    def _apply_final_chunk_selection(
        cls,
        *,
        query: str,
        context: Dict[str, Any],
        max_chunks: int,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        kg_name: Optional[str],
    ) -> Dict[str, Any]:
        chunks = cls._sort_chunks_for_query(query, list(context.get("chunks", [])))
        selected_chunks = select_ranked_subset(
            chunks,
            max_items=max_chunks,
            retrieval_temperature=retrieval_temperature,
            shortlist_factor=retrieval_shortlist_factor,
            sample_id=retrieval_sample_id,
            seed_parts=("kg", kg_name, query),
            score_getter=lambda chunk: float(chunk.get("score", 0.0)),
        )

        pruned = dict(context)
        pruned["chunks"] = selected_chunks
        retained_eids = cls._retained_entity_ids_from_chunks(selected_chunks)

        entities = context.get("entities", {}) or {}
        graph_neighbors = context.get("graph_neighbors", {}) or {}
        relationships = context.get("relationships", []) or []
        traversal_paths = context.get("traversal_paths", []) or []

        if retained_eids:
            selected_positions, selected_question_ids, selected_passage_keys = cls._selected_chunk_local_scope(
                selected_chunks
            )
            pruned["entities"] = {
                entity_id: info
                for entity_id, info in entities.items()
                if entity_id in retained_eids
            }
            pruned["graph_neighbors"] = {
                entity_id: info
                for entity_id, info in graph_neighbors.items()
                if entity_id in retained_eids
            }
            pruned["relationships"] = [
                rel
                for rel in relationships
                if rel.get("source") in retained_eids
                and rel.get("target") in retained_eids
                and cls._relationship_supported_by_selected_chunks(
                    rel,
                    selected_positions=selected_positions,
                    selected_question_ids=selected_question_ids,
                    selected_passage_keys=selected_passage_keys,
                )
            ]
            filtered_paths: List[Dict[str, Any]] = []
            seen_paths: Set[str] = set()
            for path_entry in traversal_paths:
                path_str = path_entry.get("path", "")
                if path_str in seen_paths:
                    continue
                node_ids = path_entry.get("node_ids")
                if node_ids and not all(node_id in retained_eids for node_id in node_ids):
                    continue
                if not cls._path_supported_by_selected_chunks(
                    path_entry,
                    selected_positions=selected_positions,
                    selected_question_ids=selected_question_ids,
                    selected_passage_keys=selected_passage_keys,
                ):
                    continue
                seen_paths.add(path_str)
                filtered_paths.append(path_entry)
            pruned["traversal_paths"] = filtered_paths
        else:
            pruned["entities"] = entities
            pruned["graph_neighbors"] = graph_neighbors
            pruned["relationships"] = relationships
            pruned["traversal_paths"] = traversal_paths

        pruned["documents"] = list({
            chunk.get("document")
            for chunk in selected_chunks
            if chunk.get("document")
        })
        pruned["total_score"] = float(
            sum(float(chunk.get("score", 0.0)) for chunk in selected_chunks)
        )
        pruned["entity_count"] = len(pruned.get("entities", {}))
        pruned["relationship_count"] = len(pruned.get("relationships", []))
        pruned["retrieval_sampling"] = {
            "temperature": float(retrieval_temperature or 0.0),
            "shortlist_factor": int(retrieval_shortlist_factor or 1),
            "sample_id": int(retrieval_sample_id or 0),
            "candidate_limit": int(
                compute_candidate_limit(
                    max_chunks,
                    retrieval_temperature,
                    retrieval_shortlist_factor,
                )
            ),
        }
        return pruned

    @classmethod
    def _graph_context_is_meaningful(cls, query: str, context: Optional[Dict[str, Any]]) -> bool:
        """
        Decide whether the graph-centric retrieval signal is strong enough to stay primary.

        The goal is not to prove correctness, only to detect obviously weak graph
        retrieval so we can switch to a vector-first hybrid context instead of
        over-trusting whatever nodes/paths were found.
        """
        if not context or not context.get("chunks"):
            return False

        grounding = context.get("grounding_quality")
        has_structure = bool(context.get("relationships")) or bool(context.get("traversal_paths"))
        if grounding is not None and grounding < cls._WEAK_GRAPH_GROUNDING_THRESHOLD and not has_structure:
            return False

        branches = cls._comparison_branches(query)
        if branches:
            return cls._comparison_branch_coverage(query, context.get("chunks", [])) >= len(branches)

        return has_structure or (grounding is not None and grounding >= cls._WEAK_GRAPH_GROUNDING_THRESHOLD)

    @classmethod
    def _append_adjacent_chunks_to_context(
        cls,
        graph,
        context: Dict[str, Any],
        *,
        kg_name: Optional[str],
        question_id: Optional[str],
        document_names: Optional[List[str]],
        max_adjacent: int,
    ) -> Dict[str, Any]:
        """Append position-adjacent chunks so answer spans split across boundaries survive.

        Vanilla retrieval already recovers adjacent chunks from the same local passage.
        Apply the same recovery here so graph-first and vector-first KG retrieval paths
        do not lose answer-bearing sentences simply because the retriever hit a nearby span.
        """
        chunks = list(context.get("chunks") or [])
        if not chunks:
            return context

        seed_element_ids = [
            chunk.get("parent_chunk_element_id") or chunk.get("chunk_element_id")
            for chunk in chunks
            if chunk.get("parent_chunk_element_id") or chunk.get("chunk_element_id")
        ]
        if not seed_element_ids:
            return context

        seen_ids = {
            str(chunk_id)
            for chunk in chunks
            for chunk_id in [chunk.get("chunk_element_id")]
            if chunk_id
        }
        if not seen_ids:
            return context

        params: Dict[str, Any] = {
            "element_ids": seed_element_ids,
            "max_adjacent": max(1, int(max_adjacent or 1)),
        }
        scope_conditions: List[str] = []
        if kg_name:
            scope_conditions.append("d.kgName = $kg_name")
            params["kg_name"] = kg_name
        if document_names:
            scope_conditions.append("d.fileName IN $document_names")
            params["document_names"] = list(document_names)
        if question_id:
            scope_conditions.append("seed.questionId = $question_id")
            scope_conditions.append("adj.questionId = $question_id")
            params["question_id"] = question_id
        scope_clause = ""
        if scope_conditions:
            scope_clause = "\n                  AND " + "\n                  AND ".join(scope_conditions)

        adj_query = f"""
        UNWIND $element_ids AS eid
        MATCH (seed:Chunk)-[:PART_OF]->(d:Document)
        WHERE elementId(seed) = eid
        MATCH (adj:Chunk)-[:PART_OF]->(d)
        WHERE ((
            seed.questionId IS NOT NULL
            AND adj.questionId = seed.questionId
            AND coalesce(adj.passageIndex, -1) = coalesce(seed.passageIndex, -1)
            AND abs(
                coalesce(adj.chunkLocalIndex, adj.position)
                - coalesce(seed.chunkLocalIndex, seed.position)
            ) = 1
        ) OR (
            seed.questionId IS NULL
            AND adj.questionId IS NULL
            AND abs(adj.position - seed.position) = 1
        )){scope_clause}
        OPTIONAL MATCH (adj)-[:HAS_ENTITY]->(entity:__Entity__)
        WITH adj, d, collect(DISTINCT entity) AS chunk_entities
        RETURN DISTINCT
            adj.text AS text,
            adj.id AS chunk_id,
            elementId(adj) AS chunk_element_id,
            adj.position AS position,
            adj.source AS source,
            adj.questionId AS question_id,
            adj.passageIndex AS passage_index,
            adj.chunkLocalIndex AS chunk_local_index,
            0.0 AS score,
            d.fileName AS document,
            d.kgName AS kg_name,
            [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                id: coalesce(entity.id, entity.name),
                element_id: elementId(entity),
                type: coalesce(entity.type, 'Entity'),
                description: coalesce(entity.name, '')
            }}] AS entities
        ORDER BY d.fileName ASC,
                 coalesce(adj.questionId, ''),
                 coalesce(adj.passageIndex, -1),
                 coalesce(adj.chunkLocalIndex, adj.position),
                 adj.position ASC
        LIMIT $max_adjacent
        """
        try:
            adj_results = graph.query(adj_query, params) or []
        except Exception as adj_err:
            logging.debug("Enhanced adjacent chunk expansion failed (non-fatal): %s", adj_err)
            return context

        if not adj_results:
            return context

        updated = dict(context)
        updated_chunks = list(chunks)
        documents = set(context.get("documents") or [])
        for result in adj_results:
            chunk_element_id = result.get("chunk_element_id")
            if not chunk_element_id or str(chunk_element_id) in seen_ids:
                continue
            entities = result.get("entities") or []
            updated_chunks.append({
                "text": result["text"],
                "chunk_id": result["chunk_id"],
                "chunk_element_id": chunk_element_id,
                "position": result.get("position"),
                "source": result.get("source"),
                "question_id": result.get("question_id"),
                "passage_index": result.get("passage_index"),
                "chunk_local_index": result.get("chunk_local_index"),
                "score": 0.0,
                "document": result["document"],
                "kg_name": result.get("kg_name"),
                "entities": entities,
                "linked_entity_ids": [
                    entity.get("id")
                    for entity in entities
                    if isinstance(entity, dict) and entity.get("id")
                ],
                "linked_entity_count": len(entities),
                "adjacent": True,
            })
            documents.add(result["document"])
            seen_ids.add(str(chunk_element_id))

        updated["chunks"] = updated_chunks
        updated["documents"] = list(documents)
        return updated

    @classmethod
    def _merge_retrieval_contexts(
        cls,
        primary: Dict[str, Any],
        secondary: Optional[Dict[str, Any]],
        *,
        max_chunks: int,
        search_method: str,
        secondary_limit: Optional[int] = None,
        min_secondary_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Merge two retrieval contexts while keeping primary chunks dominant."""
        if not secondary or not secondary.get("chunks"):
            merged = dict(primary)
            merged["search_method"] = search_method
            return merged

        merged: Dict[str, Any] = dict(primary)
        merged["chunks"] = [dict(chunk) for chunk in primary.get("chunks", [])]
        merged["entities"] = dict(primary.get("entities", {}))
        merged["relationships"] = list(primary.get("relationships", []))
        merged["graph_neighbors"] = dict(primary.get("graph_neighbors", {}))
        merged["traversal_paths"] = list(primary.get("traversal_paths", []))

        existing_chunk_ids: Set[str] = set()
        for chunk in merged["chunks"]:
            chunk_key = cls._chunk_identity(chunk)
            if chunk_key:
                existing_chunk_ids.add(str(chunk_key))
            parent_chunk_key = chunk.get("parent_chunk_id")
            if parent_chunk_key:
                existing_chunk_ids.add(str(parent_chunk_key))
        new_chunks: List[Dict[str, Any]] = []
        for chunk in secondary.get("chunks", []):
            cid = cls._chunk_identity(chunk)
            parent_cid = chunk.get("parent_chunk_id")
            if cid in existing_chunk_ids or (parent_cid and str(parent_cid) in existing_chunk_ids):
                continue
            existing_chunk_ids.add(str(cid))
            if parent_cid:
                existing_chunk_ids.add(str(parent_cid))
            candidate = dict(chunk)
            if min_secondary_score is not None:
                candidate["score"] = max(float(candidate.get("score", 0.0)), min_secondary_score)
            new_chunks.append(candidate)
            if secondary_limit is not None and len(new_chunks) >= secondary_limit:
                break

        merged["chunks"].extend(new_chunks)
        merged["chunks"] = cls._sort_chunks_for_query(primary.get("query", ""), merged["chunks"])
        merged["chunks"] = merged["chunks"][:max_chunks]

        for entity_id, entity_info in secondary.get("entities", {}).items():
            merged["entities"].setdefault(entity_id, entity_info)
        for entity_id, entity_info in secondary.get("graph_neighbors", {}).items():
            merged["graph_neighbors"].setdefault(entity_id, entity_info)

        seen_rel_keys = {cls._relationship_identity(rel) for rel in merged["relationships"]}
        for rel in secondary.get("relationships", []):
            rel_key = cls._relationship_identity(rel)
            if rel_key in seen_rel_keys:
                continue
            seen_rel_keys.add(rel_key)
            merged["relationships"].append(rel)

        seen_paths = {path.get("path", "") for path in merged["traversal_paths"]}
        for path_entry in secondary.get("traversal_paths", []):
            path_str = path_entry.get("path", "")
            if not path_str or path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            merged["traversal_paths"].append(path_entry)

        merged["documents"] = list({
            *(primary.get("documents") or []),
            *(secondary.get("documents") or []),
            *(chunk.get("document") for chunk in merged["chunks"] if chunk.get("document")),
        })
        merged["total_score"] = float(sum(float(chunk.get("score", 0.0)) for chunk in merged["chunks"]))
        merged["entity_count"] = len(merged["entities"])
        merged["relationship_count"] = len(merged["relationships"])
        merged["search_method"] = search_method
        # Always preserve the highest grounding_quality seen across both contexts
        # so that a weak-graph fallback to vector doesn't erase genuine entity grounding.
        gq = max(
            primary.get("grounding_quality") or 0.0,
            (secondary.get("grounding_quality") if secondary else None) or 0.0,
        )
        merged["grounding_quality"] = gq
        merged["seed_entity_count"] = max(
            int(primary.get("seed_entity_count") or 0),
            int((secondary or {}).get("seed_entity_count") or 0),
        )
        return merged

    @classmethod
    def _fuse_contexts_with_rrf(
        cls,
        *,
        query: str,
        contexts: List[Dict[str, Any]],
        max_chunks: int,
        kg_name: Optional[str],
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        search_method: str,
    ) -> Optional[Dict[str, Any]]:
        usable_contexts = [ctx for ctx in contexts if ctx and ctx.get("chunks")]
        if not usable_contexts:
            return None
        if len(usable_contexts) == 1:
            return usable_contexts[0]

        merged: Dict[str, Any] = {
            "query": query,
            "chunks": [],
            "entities": {},
            "relationships": [],
            "graph_neighbors": {},
            "traversal_paths": [],
            "documents": [],
            "total_score": 0.0,
            "entity_count": 0,
            "relationship_count": 0,
            "search_method": search_method,
            "kg_name": kg_name,
            "grounding_quality": 0.0,
            "seed_entity_count": 0,
        }
        chunk_records: Dict[str, Dict[str, Any]] = {}
        seen_rel_keys: Set[str] = set()
        seen_paths: Set[str] = set()

        for context in usable_contexts:
            ranked_chunks = cls._sort_chunks_for_query(query, list(context.get("chunks", [])))
            for rank, chunk in enumerate(ranked_chunks, 1):
                chunk_key = str(
                    chunk.get("parent_chunk_id")
                    or chunk.get("chunk_id")
                    or chunk.get("chunk_element_id")
                    or str(chunk.get("text", ""))[:120]
                )
                record = chunk_records.get(chunk_key)
                if record is None:
                    record = {
                        "chunk": dict(chunk),
                        "rrf": 0.0,
                        "support_count": 0,
                        "best_score": float(chunk.get("score", 0.0)),
                    }
                    chunk_records[chunk_key] = record
                record["rrf"] += 1.0 / float(cls._QUERY_FUSION_RRF_K + rank)
                record["support_count"] += 1
                current_score = float(chunk.get("score", 0.0))
                if current_score >= record["best_score"]:
                    record["chunk"] = dict(chunk)
                    record["best_score"] = current_score

            for entity_id, entity_info in (context.get("entities") or {}).items():
                merged["entities"].setdefault(entity_id, entity_info)
            for entity_id, entity_info in (context.get("graph_neighbors") or {}).items():
                merged["graph_neighbors"].setdefault(entity_id, entity_info)
            for rel in context.get("relationships", []) or []:
                rel_key = cls._relationship_identity(rel)
                if rel_key in seen_rel_keys:
                    continue
                seen_rel_keys.add(rel_key)
                merged["relationships"].append(rel)
            for path_entry in context.get("traversal_paths", []) or []:
                path_str = path_entry.get("path", "")
                if not path_str or path_str in seen_paths:
                    continue
                seen_paths.add(path_str)
                merged["traversal_paths"].append(path_entry)
            merged["grounding_quality"] = max(
                float(merged.get("grounding_quality") or 0.0),
                float(context.get("grounding_quality") or 0.0),
            )
            merged["seed_entity_count"] = max(
                int(merged.get("seed_entity_count") or 0),
                int(context.get("seed_entity_count") or 0),
            )

        fused_chunks: List[Dict[str, Any]] = []
        for record in chunk_records.values():
            chunk = dict(record["chunk"])
            chunk["rrf_score"] = record["rrf"]
            chunk["retrieval_support_count"] = record["support_count"]
            chunk["score"] = (
                record["rrf"]
                + 0.05 * min(record["support_count"], 3)
                + 0.01 * float(record["best_score"])
            )
            fused_chunks.append(chunk)

        merged["chunks"] = cls._sort_chunks_for_query(query, fused_chunks)
        merged["documents"] = list({
            chunk.get("document")
            for chunk in merged["chunks"]
            if chunk.get("document")
        })
        merged["total_score"] = float(
            sum(float(chunk.get("score", 0.0)) for chunk in merged["chunks"])
        )
        merged["entity_count"] = len(merged["entities"])
        merged["relationship_count"] = len(merged["relationships"])
        return cls._apply_final_chunk_selection(
            query=query,
            context=merged,
            max_chunks=max_chunks,
            retrieval_temperature=retrieval_temperature,
            retrieval_shortlist_factor=retrieval_shortlist_factor,
            retrieval_sample_id=retrieval_sample_id,
            kg_name=kg_name,
        )

    @staticmethod
    def _is_pure_vector_search_method(search_method: Optional[str]) -> bool:
        """Return True when the context already came from a pure vector-only route."""
        return str(search_method or "").strip().lower() in {
            "vector_similarity",
            "retrieval_span_similarity",
        }

    def _create_neo4j_connection(self):
        """Return the shared Neo4j connection, creating it once on first call."""
        if self._graph is None:
            self._graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database,
                refresh_schema=False,
                sanitize=True,
            )
        return self._graph

    def clear_retrieval_caches(self) -> None:
        """Drop cached retrieval state after KG mutations."""
        self._vector_index_available = None
        self._active_vector_index_name = None
        self._graph = None
        self._embedding_cache.clear()
        self._entity_extraction_cache.clear()
        self._late_interaction_corpus_cache.clear()

    @staticmethod
    def _late_interaction_corpus_cap() -> int:
        return max(200, int(os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_CORPUS_CAP", "5000").strip() or 5000))

    def _first_stage_late_interaction_enabled(self) -> bool:
        explicit = str(os.getenv("ONTOGRAPHRAG_FIRST_STAGE_LATE_INTERACTION", "")).strip().lower()
        if explicit:
            return explicit in {"1", "true", "yes", "on"}
        return late_interaction_enabled()

    @staticmethod
    def _late_interaction_scope_key(
        *,
        kg_name: Optional[str],
        document_names: Optional[List[str]],
        question_id: Optional[str],
        include_entities: bool,
    ) -> tuple:
        return (
            kg_name or "",
            tuple(sorted(document_names or [])),
            question_id or "",
            bool(include_entities),
        )

    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate query embedding, caching results to avoid redundant calls."""
        if query in self._embedding_cache:
            return self._embedding_cache[query]

        if hasattr(self.embedding_model, "embed_query"):
            embedding = self.embedding_model.embed_query(query)
        elif hasattr(self.embedding_model, "encode"):
            raw = self.embedding_model.encode(query, convert_to_numpy=True)
            embedding = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        else:
            raise ValueError("Unsupported embedding model interface. Expected embed_query or encode.")

        # Cap cache size to avoid unbounded memory growth.
        if len(self._embedding_cache) >= 2048:
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        self._embedding_cache[query] = embedding
        return embedding

    def _late_interaction_corpus_rows(
        self,
        graph,
        *,
        kg_name: Optional[str],
        document_names: Optional[List[str]],
        question_id: Optional[str],
        include_entities: bool,
    ) -> List[Dict[str, Any]]:
        cache = getattr(self, "_late_interaction_corpus_cache", None)
        if cache is None:
            cache = {}
            self._late_interaction_corpus_cache = cache

        scope_key = self._late_interaction_scope_key(
            kg_name=kg_name,
            document_names=document_names,
            question_id=question_id,
            include_entities=include_entities,
        )
        if scope_key in cache:
            return [dict(row) for row in cache[scope_key]]

        params: Dict[str, Any] = {"corpus_cap": self._late_interaction_corpus_cap()}
        filters: List[str] = []
        if kg_name:
            filters.append("d.kgName = $kg_name")
            params["kg_name"] = kg_name
        if document_names:
            filters.append("d.fileName IN $document_names")
            params["document_names"] = list(document_names)
        if question_id:
            filters.append("coalesce(retrieval.questionId, chunk.questionId) = $question_id")
            params["question_id"] = question_id
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        entity_match = ""
        with_clause = "WITH retrieval, chunk, d"
        entity_return = "[] AS entities"
        if include_entities:
            entity_match = "OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(entity:__Entity__)"
            with_clause = "WITH retrieval, chunk, d, collect(DISTINCT entity) AS chunk_entities"
            entity_return = """[entity IN chunk_entities WHERE entity IS NOT NULL | {
                id: coalesce(entity.id, entity.name),
                element_id: elementId(entity),
                type: coalesce(entity.type, 'Entity'),
                description: coalesce(entity.name, '')
            }] AS entities"""

        retrieval_query = f"""
        MATCH (retrieval:RetrievalChunk)-[:RETRIEVES_FROM]->(chunk:Chunk)-[:PART_OF]->(d:Document)
        {where_clause}
        {entity_match}
        {with_clause}
        RETURN
            retrieval.text AS text,
            retrieval.id AS chunk_id,
            elementId(retrieval) AS chunk_element_id,
            chunk.id AS parent_chunk_id,
            elementId(chunk) AS parent_chunk_element_id,
            coalesce(retrieval.questionId, chunk.questionId) AS question_id,
            coalesce(retrieval.passageIndex, chunk.passageIndex) AS passage_index,
            coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex) AS chunk_local_index,
            retrieval.retrievalLocalIndex AS retrieval_local_index,
            chunk.position AS position,
            chunk.source AS source,
            d.fileName AS document,
            d.kgName AS kg_name,
            {entity_return}
        ORDER BY d.fileName ASC,
                 coalesce(retrieval.questionId, ''),
                 coalesce(retrieval.passageIndex, -1),
                 coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex, chunk.position),
                 chunk.position ASC
        LIMIT $corpus_cap
        """
        rows = graph.query(retrieval_query, params) or []

        if not rows:
            chunk_filters: List[str] = []
            if kg_name:
                chunk_filters.append("d.kgName = $kg_name")
            if document_names:
                chunk_filters.append("d.fileName IN $document_names")
            if question_id:
                chunk_filters.append("chunk.questionId = $question_id")
            chunk_where = "WHERE " + " AND ".join(chunk_filters) if chunk_filters else ""

            entity_match = ""
            with_clause = "WITH chunk, d"
            entity_return = "[] AS entities"
            if include_entities:
                entity_match = "OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(entity:__Entity__)"
                with_clause = "WITH chunk, d, collect(DISTINCT entity) AS chunk_entities"
                entity_return = """[entity IN chunk_entities WHERE entity IS NOT NULL | {
                    id: coalesce(entity.id, entity.name),
                    element_id: elementId(entity),
                    type: coalesce(entity.type, 'Entity'),
                    description: coalesce(entity.name, '')
                }] AS entities"""

            chunk_query = f"""
            MATCH (chunk:Chunk)-[:PART_OF]->(d:Document)
            {chunk_where}
            {entity_match}
            {with_clause}
            RETURN
                chunk.text AS text,
                chunk.id AS chunk_id,
                elementId(chunk) AS chunk_element_id,
                chunk.questionId AS question_id,
                chunk.passageIndex AS passage_index,
                chunk.chunkLocalIndex AS chunk_local_index,
                chunk.position AS position,
                chunk.source AS source,
                d.fileName AS document,
                d.kgName AS kg_name,
                {entity_return}
            ORDER BY d.fileName ASC,
                     coalesce(chunk.questionId, ''),
                     coalesce(chunk.passageIndex, -1),
                     coalesce(chunk.chunkLocalIndex, chunk.position),
                     chunk.position ASC
            LIMIT $corpus_cap
            """
            rows = graph.query(chunk_query, params) or []

        cache[scope_key] = [dict(row) for row in rows]
        return [dict(row) for row in rows]

    def _late_interaction_search(
        self,
        graph,
        query: str,
        document_names: List[str] = None,
        max_chunks: int = 20,
        kg_name: str = None,
        max_hops: int = 2,
        question_id: str = None,
    ) -> Dict[str, Any]:
        try:
            scope_key = self._late_interaction_scope_key(
                kg_name=kg_name,
                document_names=document_names,
                question_id=question_id,
                include_entities=True,
            )
            rows = self._late_interaction_corpus_rows(
                graph,
                kg_name=kg_name,
                document_names=document_names,
                question_id=question_id,
                include_entities=True,
            )
            if not rows:
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "late_interaction_unavailable",
                }

            candidate_rows: List[Dict[str, Any]] = []
            for row in rows:
                entities = row.get("entities") or []
                candidate_rows.append({
                    "text": row["text"],
                    "chunk_id": row["chunk_id"],
                    "chunk_element_id": row["chunk_element_id"],
                    "parent_chunk_id": row.get("parent_chunk_id"),
                    "parent_chunk_element_id": row.get("parent_chunk_element_id"),
                    "question_id": row.get("question_id"),
                    "passage_index": row.get("passage_index"),
                    "chunk_local_index": row.get("chunk_local_index"),
                    "retrieval_local_index": row.get("retrieval_local_index"),
                    "position": row.get("position"),
                    "source": row.get("source"),
                    "score": 0.0,
                    "document": row["document"],
                    "kg_name": row.get("kg_name"),
                    "entities": entities,
                    "linked_entity_ids": [
                        entity.get("id")
                        for entity in entities
                        if isinstance(entity, dict) and entity.get("id")
                    ],
                    "linked_entity_count": len(entities),
                })

            reranked_rows, li_meta = late_interaction_rescore_chunks_for_query(
                query,
                candidate_rows,
                max_chunks=max_chunks,
                replace_score=True,
                index_key=scope_key,
            )
            selected_rows = reranked_rows[:max_chunks]
            if not li_meta.get("applied"):
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "late_interaction_unavailable",
                    "late_interaction_stage": li_meta,
                }

            score_values = [
                abs(float(row.get("late_interaction_score", row.get("score", 0.0)) or 0.0))
                for row in selected_rows
            ]
            if not selected_rows or max(score_values, default=0.0) <= 1e-9:
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "late_interaction_unavailable",
                    "late_interaction_stage": {
                        **li_meta,
                        "applied": False,
                        "reason": "no_score_signal",
                    },
                }

            context = {
                "query": query,
                "chunks": selected_rows,
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": list({
                    row.get("document")
                    for row in selected_rows
                    if row.get("document")
                }),
                "total_score": float(sum(float(row.get("score", 0.0)) for row in selected_rows)),
                "search_method": "late_interaction_similarity",
                "late_interaction_stage": li_meta,
            }

            all_entity_ids: List[str] = []
            for row in selected_rows:
                for entity in row.get("entities", []):
                    entity_id = entity.get("id")
                    if not entity_id:
                        continue
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity.get("element_id"),
                            "type": entity.get("type"),
                            "description": entity.get("description"),
                            "mentioned_in_chunks": [row["chunk_id"]],
                            "source": "late_interaction",
                        }
                        all_entity_ids.append(entity_id)
                    else:
                        context["entities"][entity_id].setdefault("mentioned_in_chunks", []).append(row["chunk_id"])

            if all_entity_ids:
                expansion = self._expand_entities_via_graph(
                    graph,
                    all_entity_ids[: self._GRAPH_TRAVERSAL_SEED_LIMIT],
                    kg_name=kg_name,
                    max_hops=max_hops,
                    question_id=question_id,
                    document_names=document_names,
                )
                context["graph_neighbors"] = expansion["neighbors"]
                context["traversal_paths"] = expansion["paths"]
                rel_results = self._fetch_relationships_for_entity_ids(
                    graph,
                    all_entity_ids[:40],
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                for rel in rel_results:
                    rel_key = (
                        rel.get("relationship_element_id")
                        or f"{rel['source']}-{rel['relationship_type']}-{rel['target']}"
                        f"-negated={bool(rel.get('negated', False))}"
                    )
                    context["relationships"].append({
                        "key": rel_key,
                        "source": rel["source"],
                        "source_element_id": rel["source_element_id"],
                        "target": rel["target"],
                        "target_element_id": rel["target_element_id"],
                        "type": rel["relationship_type"],
                        "element_id": rel["relationship_element_id"],
                        "negated": bool(rel.get("negated", False)),
                        "condition": rel.get("condition"),
                        "quantitative": rel.get("quantitative"),
                        "confidence": rel.get("confidence"),
                        "question_ids": rel.get("question_ids") or [],
                        "passage_keys": rel.get("passage_keys") or [],
                        "provenance_positions": rel.get("provenance_positions") or [],
                    })

            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])
            context.setdefault("grounding_quality", 0.0)
            context.setdefault("seed_entity_count", 0)
            return context
        except Exception as exc:
            logging.warning("Late-interaction search failed: %s", exc)
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0,
                "search_method": "late_interaction_error",
                "error": str(exc),
            }

    def _semantic_similarity_search(
        self,
        graph,
        query: str,
        document_names: List[str] = None,
        similarity_threshold: float = 0.08,
        max_chunks: int = 20,
        kg_name: str = None,
        max_hops: int = 2,
        question_id: str = None,
        allow_first_stage_late_interaction: bool = True,
    ) -> Dict[str, Any]:
        if allow_first_stage_late_interaction and self._first_stage_late_interaction_enabled():
            context = self._late_interaction_search(
                graph,
                query,
                document_names=document_names,
                max_chunks=max_chunks,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
            )
            if context.get("chunks"):
                return context
        return self._vector_similarity_search(
            graph,
            query,
            document_names=document_names,
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks,
            kg_name=kg_name,
            max_hops=max_hops,
            question_id=question_id,
        )

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

    def check_vector_index(self) -> bool:
        """
        Check if vector index exists and has correct dimensions. Result is cached after first call.
        """
        if self._vector_index_available is not None:
            return self._vector_index_available
        try:
            graph = self._create_neo4j_connection()
            result = graph.query(
                "SHOW INDEXES YIELD name, type, state, options "
                "WHERE type = 'VECTOR'"
            )
            if not result:
                logging.warning("No vector index found - falling back to text search")
                self._vector_index_available = False
                return False
            probe_embedding = self._generate_query_embedding("test")
            online_indexes = {
                str(row.get("name")): row
                for row in result
                if str(row.get("state") or "").upper() in {"", "ONLINE"}
            }
            for candidate in ("retrieval_vector", "vector"):
                if candidate not in online_indexes:
                    continue
                try:
                    graph.query(
                        f"""
                        CALL db.index.vector.queryNodes('{candidate}', 1, $query_vector)
                        YIELD node, score
                        RETURN count(node) AS n
                        """,
                        {"query_vector": probe_embedding},
                    )
                    logging.info("Vector index probe succeeded for %s — result cached", candidate)
                    self._active_vector_index_name = candidate
                    self._vector_index_available = True
                    return True
                except Exception as probe_error:
                    logging.warning(
                        "Vector index %s exists but probe failed: %s",
                        candidate,
                        probe_error,
                    )
            logging.warning("No compatible vector index found - falling back to text search")
            self._vector_index_available = False
            return False
        except Exception as e:
            logging.error(f"Error checking vector index: {e}")
            self._vector_index_available = False
            return False

    @staticmethod
    def _question_local_pair_support_clause(
        left_var: str,
        right_var: str,
        *,
        kg_name: Optional[str],
        question_id: Optional[str],
        document_names: Optional[List[str]] = None,
        relationship_var: Optional[str] = None,
    ) -> str:
        """
        Build a Cypher predicate requiring local support for an entity pair.

        On question-scoped datasets, entity nodes are still shared within the
        dataset KG. This clause keeps relation lookup tied to the current
        question bundle by requiring both entities to be supported in the same
        passage, or in adjacent chunks of that passage.
        """
        if not question_id and not document_names:
            return "true"

        left_filters = ""
        right_filters = ""
        if kg_name:
            left_filters += "\n              AND d1.kgName = $kg_name"
            right_filters += "\n              AND d2.kgName = $kg_name"
        if document_names:
            left_filters += "\n              AND d1.fileName IN $document_names"
            right_filters += "\n              AND d2.fileName IN $document_names"

        if not question_id:
            return f"""EXISTS {{
            MATCH ({left_var})<-[:MENTIONS|HAS_ENTITY]-(c1:Chunk)-[:PART_OF]->(d1:Document)
            WHERE true{left_filters}
        }} AND EXISTS {{
            MATCH ({right_var})<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document)
            WHERE true{right_filters}
        }}"""

        direct_edge_scope = ""
        if relationship_var:
            direct_edge_scope = f"$question_id IN coalesce({relationship_var}.questionIds, []) OR "

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
                      {left_filters}
                      {right_filters}
        }}"""

    @staticmethod
    def _question_local_entity_support_clause(
        entity_var: str,
        *,
        kg_name: Optional[str],
        question_id: Optional[str],
        document_names: Optional[List[str]] = None,
    ) -> str:
        """Build a Cypher predicate requiring an entity seed to be locally supported."""
        conditions: List[str] = []
        if question_id:
            conditions.append("c.questionId = $question_id")
        if kg_name:
            conditions.append("d.kgName = $kg_name")
        if document_names:
            conditions.append("d.fileName IN $document_names")
        where_clause = " AND ".join(conditions) if conditions else "true"
        return f"""EXISTS {{
            MATCH ({entity_var})<-[:MENTIONS|HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
            WHERE {where_clause}
        }}"""

    @staticmethod
    def _question_local_path_support_clause(
        *,
        kg_name: Optional[str],
        question_id: Optional[str],
        document_names: Optional[List[str]] = None,
    ) -> str:
        """Build a Cypher predicate requiring question-local support for every path hop."""
        if not question_id and not document_names:
            return ""

        path_filters = ""
        if kg_name:
            path_filters += """
                      AND d1.kgName = $kg_name
                      AND d2.kgName = $kg_name"""
        if document_names:
            path_filters += """
                      AND d1.fileName IN $document_names
                      AND d2.fileName IN $document_names"""

        if not question_id:
            node_conditions: List[str] = []
            if kg_name:
                node_conditions.append("d.kgName = $kg_name")
            if document_names:
                node_conditions.append("d.fileName IN $document_names")
            node_where = " AND ".join(node_conditions) if node_conditions else "true"
            return f"""
              AND ALL(pathNode IN nodes(path) WHERE EXISTS {{
                    MATCH (pathNode)<-[:MENTIONS|HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
                    WHERE {node_where}
              }})"""

        return f"""
              AND ALL(idx IN range(0, length(path) - 1) WHERE (
                    $question_id IN coalesce(relationships(path)[idx].questionIds, [])
                    OR EXISTS {{
                    WITH nodes(path)[idx] AS a, nodes(path)[idx + 1] AS b
                    MATCH (a)<-[:MENTIONS|HAS_ENTITY]-(c1:Chunk)-[:PART_OF]->(d1:Document)
                    MATCH (b)<-[:MENTIONS|HAS_ENTITY]-(c2:Chunk)-[:PART_OF]->(d2:Document)
                    WHERE c1.questionId = $question_id
                      AND c2.questionId = $question_id
                      AND coalesce(c1.passageIndex, -1) = coalesce(c2.passageIndex, -1)
                      AND abs(
                            coalesce(c1.chunkLocalIndex, c1.position)
                            - coalesce(c2.chunkLocalIndex, c2.position)
                          ) <= 1
                      {path_filters}
              }}))"""

    def _fetch_relationships_for_entity_ids(
        self,
        graph,
        entity_ids: List[str],
        *,
        kg_name: Optional[str] = None,
        question_id: Optional[str] = None,
        document_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch entity-to-entity relationships, optionally scoped to one question bundle."""
        if not entity_ids:
            return []

        pair_scope = self._question_local_pair_support_clause(
            "e1",
            "e2",
            kg_name=kg_name,
            question_id=question_id,
            document_names=document_names,
            relationship_var="r",
        )
        relationship_query = f"""
        MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
        WHERE e1.id IN $entity_ids
          AND e2.id IN $entity_ids
          AND coalesce(r.confidence, 1.0) >= 0.4
          AND {pair_scope}
        RETURN DISTINCT
            e1.id AS source,
            e1.name AS source_name,
            elementId(e1) AS source_element_id,
            e2.id AS target,
            e2.name AS target_name,
            elementId(e2) AS target_element_id,
            type(r) AS relationship_type,
            elementId(r) AS relationship_element_id,
            coalesce(r.negated, false) AS negated,
            r.condition AS condition,
            r.quantitative AS quantitative,
            coalesce(r.confidence, 1.0) AS confidence,
            coalesce(r.questionIds, []) AS question_ids,
            coalesce(r.passageKeys, []) AS passage_keys,
            coalesce(r.provenancePositions, []) AS provenance_positions
        """
        return graph.query(
            relationship_query,
            {
                "entity_ids": entity_ids,
                "kg_name": kg_name,
                "question_id": question_id,
                "document_names": document_names,
            },
        ) or []

    # Relation types that are structural meta-edges and should always be excluded
    # from RoG path planning (same list used in traversal WHERE clause).
    _STRUCTURAL_REL_TYPES: frozenset = frozenset({
        "HAS_ENTITY", "FROM_CHUNK", "PART_OF", "MENTIONED_IN", "MENTIONS", "QUALIFIES",
    })

    def _plan_relation_paths(
        self,
        graph,
        seed_entity_ids: List[str],
        query: str,
        *,
        kg_name: str = None,
        question_id: str = None,
        document_names: List[str] = None,
        top_k: int = 10,
        min_sim: float = 0.20,
    ) -> Optional[List[str]]:
        """RoG-style relation path planning.

        Enumerate all relation types reachable in 1-2 hops from seed entities,
        score each by cosine similarity between its label embedding and the query
        embedding, and return the top-K types whose similarity exceeds min_sim.

        Returns None if planning should be skipped (too few relation types found
        or all types are structural), which signals the caller to use open traversal.
        """
        if not seed_entity_ids:
            return None

        scope_filters = ""
        params: Dict[str, Any] = {"seed_ids": seed_entity_ids}
        if kg_name:
            params["kg_name"] = kg_name
            scope_filters += "\n  AND (a.kgName IS NULL OR a.kgName = $kg_name)"
        if question_id:
            params["question_id"] = question_id
            scope_filters += "\n  AND (a.questionId IS NULL OR a.questionId = $question_id)"

        rel_query = f"""
        MATCH (a:__Entity__)-[r]-(b:__Entity__)
        WHERE a.id IN $seed_ids
          {scope_filters}
          AND NOT type(r) IN {list(self._STRUCTURAL_REL_TYPES)}
        RETURN DISTINCT type(r) AS rel_type, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 40
        """
        try:
            rows = graph.query(rel_query, params) or []
        except Exception as _e:
            logging.debug("RoG relation planning query failed: %s", _e)
            return None

        rel_types = [row["rel_type"] for row in rows if row.get("rel_type")]
        if not rel_types:
            return None

        # Score each relation type label against the query by cosine similarity.
        query_emb = np.array(self._generate_query_embedding(query), dtype=np.float32)
        qnorm = np.linalg.norm(query_emb)
        if qnorm == 0:
            return None

        scored: List[Tuple[float, str]] = []
        for rel_type in rel_types:
            # Convert snake_case / ALL_CAPS label to readable words for embedding
            readable = rel_type.replace("_", " ").lower()
            try:
                rel_emb = np.array(self._generate_query_embedding(readable), dtype=np.float32)
            except Exception:
                continue
            rnorm = np.linalg.norm(rel_emb)
            if rnorm == 0:
                continue
            sim = float(np.dot(query_emb, rel_emb) / (qnorm * rnorm))
            scored.append((sim, rel_type))

        scored.sort(reverse=True)
        allowed = [rt for sim, rt in scored[:top_k] if sim >= min_sim]

        if not allowed:
            logging.debug("RoG planning: no relation types above sim threshold %.2f — using open traversal", min_sim)
            return None

        logging.info(
            "RoG relation planning: %d/%d types selected (top sim=%.3f): %s",
            len(allowed), len(rel_types),
            scored[0][0] if scored else 0.0,
            allowed[:5],
        )
        return allowed

    def _expand_entities_via_graph(
        self,
        graph,
        seed_entity_ids: List[str],
        kg_name: str = None,
        max_hops: int = 2,
        max_neighbors: int = 30,
        question_id: str = None,
        document_names: List[str] = None,
        allowed_rel_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-hop graph traversal from seed entities.

        Starting from the entities found in the initially retrieved chunks,
        this method walks the knowledge graph up to ``max_hops`` relationship
        hops and returns:
          - ``neighbors``: newly discovered entities (not in the seed set)
          - ``paths``: human-readable traversal paths for the LLM prompt,
              e.g. "Metformin --TREATS--> Type2Diabetes --HAS_COMPLICATION--> Nephropathy"

        Neighbors are ranked by:
          1. How many distinct seed entities connect to them (higher = more central)
          2. Minimum hop distance (closer = higher priority)
        """
        if not seed_entity_ids:
            return {"neighbors": {}, "paths": []}

        try:
            effective_max_hops = self._graph_traversal_max_hops_for_kg(kg_name, max_hops)
            effective_max_neighbors = self._graph_traversal_neighbor_limit_for_kg(
                kg_name,
                max_neighbors,
            )
            seed_limit = self._graph_traversal_seed_limit_for_kg(kg_name)
            if len(seed_entity_ids) > seed_limit:
                logging.info(
                    "Capping graph traversal seeds for kg '%s' from %d to %d",
                    kg_name,
                    len(seed_entity_ids),
                    seed_limit,
                )
                seed_entity_ids = seed_entity_ids[:seed_limit]

            params: Dict[str, Any] = {
                "seed_ids": seed_entity_ids,
                "max_neighbors": effective_max_neighbors,
            }

            seed_scope = "true"
            neighbor_scope = ""
            if kg_name or question_id or document_names:
                params["kg_name"] = kg_name
                params["question_id"] = question_id
                params["document_names"] = document_names
                seed_scope = self._question_local_entity_support_clause(
                    "seed",
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                neighbor_scope = self._question_local_entity_support_clause(
                    "neighbor",
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                neighbor_scope = f"\n              AND {neighbor_scope}"

            path_scope = self._question_local_path_support_clause(
                kg_name=kg_name,
                question_id=question_id,
                document_names=document_names,
            )

            # RoG-style: when an allowed relation type list is provided, constrain
            # traversal to only follow those relation types (in addition to the
            # standard structural-edge exclusion).
            rog_rel_filter = ""
            if allowed_rel_types:
                params["rog_allowed_rel_types"] = allowed_rel_types
                rog_rel_filter = "\n                AND type(r) IN $rog_allowed_rel_types"

            traversal_query = f"""
            MATCH (seed:__Entity__)
            WHERE seed.id IN $seed_ids
              AND {seed_scope}
            MATCH path = (seed)-[*1..{effective_max_hops}]-(neighbor:__Entity__)
            WHERE NOT neighbor.id IN $seed_ids
              AND neighbor.id IS NOT NULL
              AND ALL(r IN relationships(path) WHERE NOT type(r) IN ['HAS_ENTITY', 'FROM_CHUNK', 'PART_OF', 'MENTIONED_IN', 'MENTIONS', 'QUALIFIES']
                AND coalesce(r.confidence, 1.0) >= 0.4{rog_rel_filter})
              {neighbor_scope}
              {path_scope}
            WITH neighbor,
                 length(path) AS hops,
                 [n IN nodes(path) | coalesce(n.name, n.id)] AS node_names,
                 [n IN nodes(path) | n.id] AS node_ids,
                [r IN relationships(path) | {{
                     type: type(r),
                     negated: coalesce(r.negated, false),
                     condition: r.condition,
                     quantitative: r.quantitative,
                     confidence: coalesce(r.confidence, 1.0),
                     question_ids: coalesce(r.questionIds, []),
                     passage_keys: coalesce(r.passageKeys, []),
                     provenance_positions: coalesce(r.provenancePositions, [])
                 }}] AS rel_data,
                 seed.id AS seed_id
            WITH neighbor,
                 min(hops) AS min_hops,
                 count(DISTINCT seed_id) AS seed_connections,
                 collect(DISTINCT {{nodes: node_names, node_ids: node_ids, rels: rel_data}})[0..3] AS sample_paths
            RETURN
                 neighbor.id AS id,
                 neighbor.name AS name,
                 coalesce(neighbor.type, 'Entity') AS type,
                 elementId(neighbor) AS element_id,
                 min_hops,
                 seed_connections,
                 sample_paths
            ORDER BY seed_connections DESC, min_hops ASC
            LIMIT $max_neighbors
            """

            results = graph.query(traversal_query, params)

            neighbors: Dict[str, Any] = {}
            seen_paths: set = set()
            paths: List[Dict[str, Any]] = []

            for row in results:
                neighbor_id = row["id"]
                if not neighbor_id:
                    continue

                neighbors[neighbor_id] = {
                    "id": neighbor_id,
                    "name": row["name"] or neighbor_id,
                    "type": row["type"],
                    "element_id": row["element_id"],
                    "min_hops": row["min_hops"],
                    "seed_connections": row["seed_connections"],
                    "source": "graph_traversal",
                }

                # Build human-readable path strings for the prompt.
                # Store node_ids alongside the path string so iterative retrieval
                # can prune paths whose nodes are not in the retained entity set.
                for path_data in (row["sample_paths"] or []):
                    node_names = path_data.get("nodes", [])
                    node_ids = path_data.get("node_ids", [])
                    rel_data = path_data.get("rels", [])
                    if node_names and rel_data and len(node_names) == len(rel_data) + 1:
                        parts = [node_names[0]]
                        for rel, node in zip(rel_data, node_names[1:]):
                            if rel is None:
                                rel_label = "RELATED"
                            elif isinstance(rel, dict):
                                rel_label = self._format_relationship_label(rel)
                            else:
                                rel_label = str(rel) or "RELATED"
                            parts.extend([f"--{rel_label}-->", node])
                        path_str = " ".join(parts)
                        if path_str not in seen_paths:
                            path_positions: Set[int] = set()
                            path_question_ids: Set[str] = set()
                            path_passage_keys: Set[str] = set()
                            for rel in rel_data:
                                if not isinstance(rel, dict):
                                    continue
                                path_positions.update(
                                    int(pos)
                                    for pos in (rel.get("provenance_positions") or [])
                                    if isinstance(pos, (int, float))
                                )
                                path_question_ids.update(
                                    str(qid)
                                    for qid in (rel.get("question_ids") or [])
                                    if str(qid).strip()
                                )
                                path_passage_keys.update(
                                    str(key)
                                    for key in (rel.get("passage_keys") or [])
                                    if str(key).strip()
                                )
                            seen_paths.add(path_str)
                            paths.append({
                                "path": path_str,
                                "hops": row["min_hops"],
                                "node_ids": [nid for nid in node_ids if nid],
                                "provenance_positions": sorted(path_positions),
                                "question_ids": sorted(path_question_ids),
                                "passage_keys": sorted(path_passage_keys),
                            })

            logging.info(
                "Graph traversal: %d neighbors, %d paths from %d seeds (max_hops=%d)",
                len(neighbors), len(paths), len(seed_entity_ids), effective_max_hops,
            )
            return {"neighbors": neighbors, "paths": paths}

        except Exception as e:
            logging.warning("Graph traversal failed: %s", e)
            return {"neighbors": {}, "paths": []}

    def get_rag_context(
        self,
        query: str,
        document_names: List[str] = None,
        similarity_threshold: float = 0.08,
        max_chunks: int = 20,
        kg_name: str = None,
        max_hops: int = 2,
        question_id: str = None,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        retrieval_sample_id: int = 0,
        llm=None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive RAG context including chunks, entities, and relationships using vector search.

        Args:
            query: The search query
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            kg_name: Optional KG name to filter retrieval to a specific named KG
            question_id: Optional question ID to scope chunk retrieval to a single
                question's passages, preventing cross-question entity contamination.
                Use for datasets where each question has its own closed passage set
                (source_document and retrieval_bundle corpus contracts).
        """
        try:
            graph = self._create_neo4j_connection()
            candidate_chunk_limit = compute_candidate_limit(
                max_chunks,
                retrieval_temperature,
                retrieval_shortlist_factor,
            )
            hybrid_secondary_limit = max(
                3,
                min(
                    self._HYBRID_SUPPLEMENT_LIMIT * max(1, int(retrieval_shortlist_factor or 1)),
                    max(1, candidate_chunk_limit // 2),
                ),
            )

            # First check if we have any data in the knowledge graph
            # If kg_name is specified, check only for that KG
            if kg_name:
                check_query = """
                MATCH (d:Document {kgName: $kg_name})<-[:PART_OF]-(c:Chunk)
                RETURN count(c) as chunk_count
                """
                check_result = graph.query(check_query, {"kg_name": kg_name})
                
                if not check_result or check_result[0]['chunk_count'] == 0:
                    logging.warning(f"No chunks found in KG '{kg_name}'")
                    return {
                        "query": query,
                        "chunks": [],
                        "entities": {},
                        "relationships": [],
                        "graph_neighbors": {},
                        "traversal_paths": [],
                        "documents": [],
                        "total_score": 0,
                        "entity_count": 0,
                        "relationship_count": 0,
                        "kg_name": kg_name,
                        "error": f"No data found in KG '{kg_name}'. Please process documents to this KG first."
                    }
            else:
                check_query = "MATCH (c:Chunk) RETURN count(c) as chunk_count LIMIT 1"
                check_result = graph.query(check_query)

                if not check_result or check_result[0]['chunk_count'] == 0:
                    logging.warning("No chunks found in knowledge graph")
                    return {
                        "query": query,
                        "chunks": [],
                        "entities": {},
                        "relationships": [],
                        "graph_neighbors": {},
                        "traversal_paths": [],
                        "documents": [],
                        "total_score": 0,
                        "entity_count": 0,
                        "relationship_count": 0,
                        "error": "No data found in knowledge graph. Please upload and process a document first."
                    }

            # Primary path: entity-first retrieval.
            # Skipped when retrieval_mode is 'rfge' or 'vector_only', and also
            # for comparison questions — those need parallel branch vector retrieval,
            # not graph traversal which tends to pollute context via shared entities.
            _qtype = self.classify_question_type(query)
            _skip_graph_for_comparison = (_qtype == "comparison")
            if _skip_graph_for_comparison:
                logging.info("Comparison question detected — bypassing entity-first graph traversal")
            entity_context = None
            if getattr(self, "retrieval_mode", "hybrid_auto") not in ("rfge", "vector_only") and not _skip_graph_for_comparison:
                entity_context = self._entity_first_search(
                    graph,
                    query,
                    candidate_chunk_limit,
                    kg_name,
                    max_hops=max_hops,
                    question_id=question_id,
                    llm=llm,
                    document_names=document_names,
                )
            if entity_context and entity_context["chunks"]:
                logging.info(
                    "Entity-first search succeeded: %d chunks, %d entities, %d relationships",
                    len(entity_context["chunks"]),
                    entity_context["entity_count"],
                    entity_context["relationship_count"],
                )
                if getattr(self, "allow_vector_augmentation", True) and (
                    self.check_vector_index() or self._first_stage_late_interaction_enabled()
                ):
                    try:
                        vec_ctx = self._semantic_similarity_search(
                            graph, query, document_names, similarity_threshold,
                            candidate_chunk_limit, kg_name, max_hops=max_hops, question_id=question_id,
                            allow_first_stage_late_interaction=getattr(self, "retrieval_mode", "hybrid_auto") != "entity_first",
                        )
                        if vec_ctx and vec_ctx.get("chunks"):
                            if self._graph_context_is_meaningful(query, entity_context):
                                entity_context = self._merge_retrieval_contexts(
                                    entity_context,
                                    vec_ctx,
                                    max_chunks=candidate_chunk_limit,
                                    search_method="hybrid",
                                    secondary_limit=hybrid_secondary_limit,
                                    min_secondary_score=0.55,
                                )
                            else:
                                logging.info(
                                    "Graph signal looked weak for query '%s'; using vector-primary hybrid fallback",
                                    query,
                                )
                                entity_context = self._merge_retrieval_contexts(
                                    vec_ctx,
                                    entity_context,
                                    max_chunks=candidate_chunk_limit,
                                    search_method="hybrid_vector_primary",
                                    secondary_limit=hybrid_secondary_limit,
                                    min_secondary_score=0.55,
                                )
                    except Exception as _ve:
                        logging.debug("Hybrid vector augmentation failed (non-fatal): %s", _ve)
                entity_context["chunks"] = self._sort_chunks_for_query(
                    query,
                    entity_context.get("chunks", []),
                )[:candidate_chunk_limit]
                return self._apply_final_chunk_selection(
                    query=query,
                    context=entity_context,
                    max_chunks=max_chunks,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=retrieval_sample_id,
                    kg_name=kg_name,
                )
            logging.info("Entity-first search found no chunks, trying retriever-first graph expansion")

            # Second path: retriever-first graph expansion.
            # Skipped when retrieval_mode is 'entity_first' or 'vector_only',
            # use_rfge toggle is False, or for comparison questions (same reason as above).
            rfge_ctx = None
            if getattr(self, "retrieval_mode", "hybrid_auto") not in ("entity_first", "vector_only") and self.use_rfge and not _skip_graph_for_comparison:
                rfge_ctx = self._retriever_first_graph_expansion(
                    graph,
                    query,
                    candidate_chunk_limit,
                    kg_name,
                    max_hops=max_hops,
                    question_id=question_id,
                    document_names=document_names,
                )
            if rfge_ctx and rfge_ctx.get("chunks"):
                logging.info(
                    "RFGE succeeded: %d chunks, grounding=%.2f",
                    len(rfge_ctx["chunks"]),
                    rfge_ctx.get("grounding_quality", 0.0),
                )
                rfge_ctx["chunks"] = self._sort_chunks_for_query(
                    query, rfge_ctx["chunks"]
                )[:candidate_chunk_limit]
                return self._apply_final_chunk_selection(
                    query=query,
                    context=rfge_ctx,
                    max_chunks=max_chunks,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=retrieval_sample_id,
                    kg_name=kg_name,
                )
            logging.info("RFGE found no chunks, falling back to semantic search")

            # Final fallback: pure vector similarity (no graph).
            # grounding_quality=0 signals to callers that structural metrics are unreliable.
            has_semantic_backend = self.check_vector_index() or self._first_stage_late_interaction_enabled()

            if getattr(self, "retrieval_mode", "hybrid_auto") == "entity_first" and not getattr(self, "allow_vector_fallback", True):
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "entity_count": 0,
                    "relationship_count": 0,
                    "search_method": "entity_first",
                    "kg_name": kg_name,
                    "seed_entity_count": 0,
                    "grounding_quality": 0.0,
                    "retrieval_route": "entity_first_empty",
                    "route_reason": "strict_no_graph_signal",
                    "diagnostics": {
                        "rfge_fired": False,
                        "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto"),
                    },
                }

            if has_semantic_backend:
                logging.info(f"Attempting semantic similarity search (kg_name: {kg_name})")
                try:
                    if getattr(self, "retrieval_mode", "hybrid_auto") == "entity_first" and self.check_vector_index():
                        context = self._vector_similarity_search(
                            graph,
                            query,
                            document_names,
                            similarity_threshold,
                            candidate_chunk_limit,
                            kg_name,
                            max_hops=max_hops,
                            question_id=question_id,
                        )
                    else:
                        context = self._semantic_similarity_search(
                            graph,
                            query,
                            document_names,
                            similarity_threshold,
                            candidate_chunk_limit,
                            kg_name,
                            max_hops=max_hops,
                            question_id=question_id,
                        )
                    if not context["chunks"]:
                        logging.warning("Semantic search returned no results, falling back to text search")
                        ctx = self._text_similarity_search(
                            graph,
                            query,
                            document_names,
                            candidate_chunk_limit,
                            kg_name,
                            max_hops=max_hops,
                            question_id=question_id,
                        )
                        ctx.setdefault("grounding_quality", 0.0)
                        ctx.setdefault("retrieval_route", "semantic_only")
                        ctx.setdefault("route_reason", "no_graph_signal")
                        ctx.setdefault(
                            "diagnostics",
                            {"rfge_fired": False, "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto")},
                        )
                        ctx.setdefault("seed_entity_count", 0)
                        ctx["chunks"] = self._sort_chunks_for_query(
                            query,
                            ctx.get("chunks", []),
                        )[:candidate_chunk_limit]
                        return self._apply_final_chunk_selection(
                            query=query,
                            context=ctx,
                            max_chunks=max_chunks,
                            retrieval_temperature=retrieval_temperature,
                            retrieval_shortlist_factor=retrieval_shortlist_factor,
                            retrieval_sample_id=retrieval_sample_id,
                            kg_name=kg_name,
                        )
                    context.setdefault("grounding_quality", 0.0)
                    context.setdefault("retrieval_route", "semantic_only")
                    context.setdefault("route_reason", "no_graph_signal")
                    context.setdefault(
                        "diagnostics",
                        {"rfge_fired": False, "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto")},
                    )
                    context.setdefault("seed_entity_count", 0)
                    context["chunks"] = self._sort_chunks_for_query(
                        query,
                        context.get("chunks", []),
                    )[:candidate_chunk_limit]
                    return self._apply_final_chunk_selection(
                        query=query,
                        context=context,
                        max_chunks=max_chunks,
                        retrieval_temperature=retrieval_temperature,
                        retrieval_shortlist_factor=retrieval_shortlist_factor,
                        retrieval_sample_id=retrieval_sample_id,
                        kg_name=kg_name,
                    )
                except Exception as e:
                    logging.error(f"Semantic search failed: {e}, falling back to text search")
                    ctx = self._text_similarity_search(
                        graph,
                        query,
                        document_names,
                        candidate_chunk_limit,
                        kg_name,
                        max_hops=max_hops,
                        question_id=question_id,
                    )
                    ctx.setdefault("grounding_quality", 0.0)
                    ctx.setdefault("retrieval_route", "semantic_only")
                    ctx.setdefault("route_reason", "no_graph_signal")
                    ctx.setdefault(
                        "diagnostics",
                        {"rfge_fired": False, "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto")},
                    )
                    ctx.setdefault("seed_entity_count", 0)
                    ctx["chunks"] = self._sort_chunks_for_query(
                        query,
                        ctx.get("chunks", []),
                    )[:candidate_chunk_limit]
                    return self._apply_final_chunk_selection(
                        query=query,
                        context=ctx,
                        max_chunks=max_chunks,
                        retrieval_temperature=retrieval_temperature,
                        retrieval_shortlist_factor=retrieval_shortlist_factor,
                        retrieval_sample_id=retrieval_sample_id,
                        kg_name=kg_name,
                    )
            else:
                logging.info("No vector index available, using text similarity search")
                ctx = self._text_similarity_search(
                    graph,
                    query,
                    document_names,
                    candidate_chunk_limit,
                    kg_name,
                    max_hops=max_hops,
                    question_id=question_id,
                )
                ctx.setdefault("grounding_quality", 0.0)
                ctx.setdefault("seed_entity_count", 0)
                ctx["chunks"] = self._sort_chunks_for_query(
                    query,
                    ctx.get("chunks", []),
                )[:candidate_chunk_limit]
                return self._apply_final_chunk_selection(
                    query=query,
                    context=ctx,
                    max_chunks=max_chunks,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=retrieval_sample_id,
                    kg_name=kg_name,
                )

        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0,
                "entity_count": 0,
                "relationship_count": 0,
                "error": str(e)
            }

    # Hop-distance → retrieval score mapping (used as fallback when PPR not available).
    # Chunks reached via seed entities (hop 0) are most relevant; each additional
    # hop attenuates the score.  This mirrors C2RAG's insight that evidence quality
    # degrades with path length and lets downstream re-ranking favour closer support.
    _HOP_SCORE = {0: 1.0, 1: 0.8, 2: 0.6}
    _HOP_SCORE_DEFAULT = 0.5  # for hops > 2

    # Maximum number of chunks that may come from any single source passage
    # (same questionId + passageIndex).  Caps graph traversal from flooding
    # the context with many chunks from one entity-dense but answer-irrelevant
    # passage while crowding out the single chunk that holds the actual answer.
    _MAX_CHUNKS_PER_PASSAGE: int = 2

    # Minimum cosine similarity between a traversal-discovered chunk and the
    # query before it is admitted to the context.  Chunks from entity neighbours
    # that are topically unrelated to the question are dropped.  Applied only to
    # hop-1+ chunks (seed chunks are always kept).
    # Set to 0.0 to disable.
    _TRAVERSAL_CHUNK_MIN_SIM: float = 0.30

    @staticmethod
    def _chunk_rank_key(chunk: Dict[str, Any]) -> Tuple[float, float, int]:
        return (
            -float(chunk.get("score", 0.0)),
            -int(chunk.get("linked_entity_count", 0)),
            int(chunk.get("position") or 0),
        )

    def _apply_per_passage_chunk_cap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep the highest-ranked chunks from each source passage."""
        _passage_cap = self._MAX_CHUNKS_PER_PASSAGE
        if _passage_cap <= 0:
            return list(chunks)

        ranked_chunks = sorted(chunks, key=self._chunk_rank_key)
        passage_counts: Dict[str, int] = {}
        capped_chunks: List[Dict[str, Any]] = []
        for chunk in ranked_chunks:
            qid_val = chunk.get("question_id") or ""
            pidx_val = chunk.get("passage_index")
            if pidx_val is not None and qid_val:
                pkey = f"{qid_val}::p{pidx_val}"
            else:
                pkey = chunk.get("chunk_id") or id(chunk)
            count = passage_counts.get(pkey, 0)
            if count < _passage_cap:
                capped_chunks.append(chunk)
                passage_counts[pkey] = count + 1
            else:
                logging.debug(
                    "Entity-first: capped passage %s at %d chunks",
                    pkey, _passage_cap,
                )
        return capped_chunks or ranked_chunks

    # Personalized PageRank parameters.
    # alpha: restart probability (fraction of walk mass that teleports back to seeds each step).
    # steps: power-iteration depth; 5 is sufficient for typical subgraphs (<500 entities).
    _PPR_ALPHA = 0.85
    _PPR_STEPS = 5

    # Retriever-first graph expansion: minimum cosine similarity for a chunk to seed
    # the graph expansion pass.  Lower than entity-first ANN threshold because passages
    # are already domain-matched; we want breadth not precision here.
    _RFGE_VECTOR_THRESHOLD = 0.20
    # Minimum PPR entity score before a chunk is considered graph-supported in RFGE.
    _RFGE_MIN_PPR_SCORE = 0.005

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

    @staticmethod
    def _build_symbolic_entity_lookup_query(entity_support_scope: str) -> str:
        """Build the Cypher query used by the symbolic entity matcher.

        Neo4j aliases defined in a WITH projection are not visible to other
        expressions in the same projection list. We therefore normalize
        ``entity_name`` in one WITH clause and compute token overlap in the
        next one; otherwise the symbolic matcher fails with a syntax error and
        the system unnecessarily falls back to vector search.
        """
        return f"""
        MATCH (e:__Entity__)
        WITH e,
             toLower(coalesce(e.name, '')) AS entity_name,
             toLower(coalesce(e.source_title, '')) AS source_title_lower
        WITH e, entity_name, source_title_lower,
             [tok IN $query_tokens WHERE entity_name CONTAINS tok] AS matched_tokens,
             CASE WHEN $query_text CONTAINS entity_name THEN 1 ELSE 0 END AS exact_in_query,
             CASE WHEN entity_name CONTAINS $query_text THEN 1 ELSE 0 END AS query_in_entity,
             CASE WHEN size(source_title_lower) >= 4
                       AND ($query_text CONTAINS source_title_lower
                            OR source_title_lower CONTAINS $query_text)
                  THEN 1 ELSE 0 END AS title_match
        WHERE size(entity_name) >= 4
          AND (
            exact_in_query = 1
            OR query_in_entity = 1
            OR size(matched_tokens) >= 2
            OR title_match = 1
            OR ANY(alias IN coalesce(e.all_names, [])
                   WHERE $query_text CONTAINS toLower(alias)
                      OR toLower(alias) CONTAINS $query_text)
            OR ANY(syn IN coalesce(e.synonyms, [])
                   WHERE $query_text CONTAINS toLower(syn)
                      OR toLower(syn) CONTAINS $query_text)
          )
          AND {entity_support_scope}
        RETURN e.id AS id, e.name AS name,
               coalesce(e.type, 'Entity') AS type,
               elementId(e) AS element_id,
               exact_in_query,
               query_in_entity,
               size(matched_tokens) AS token_overlap
        ORDER BY exact_in_query DESC,
                 query_in_entity DESC,
                 token_overlap DESC,
                 title_match DESC,
                 size(entity_name) DESC
        LIMIT $entity_lookup_limit
        """

    def _fetch_subgraph_edges(
        self,
        graph,
        entity_ids: List[str],
        kg_name: str = None,
        question_id: str = None,
        document_names: List[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Return all directed edges (src_id, tgt_id) between a set of entity IDs.

        Used to build the adjacency structure for PPR power iteration.
        Structural meta-edges (HAS_ENTITY, PART_OF, etc.) are excluded so only
        domain-semantic relationships are included in the random walk.
        """
        if not entity_ids:
            return []
        params: Dict[str, Any] = {
            "entity_ids": entity_ids,
            "kg_name": kg_name,
            "question_id": question_id,
            "document_names": document_names,
        }
        source_support = self._question_local_entity_support_clause(
            "a",
            kg_name=kg_name,
            question_id=question_id,
            document_names=document_names,
        )
        target_support = self._question_local_entity_support_clause(
            "b",
            kg_name=kg_name,
            question_id=question_id,
            document_names=document_names,
        )
        local_filters = ""
        if kg_name or question_id or document_names:
            local_filters = f"""
                  AND {source_support}
                  AND {target_support}"""
        qid_filter = ""
        if question_id:
            qid_filter = """
            AND ANY(pid IN coalesce(r.questionIds, []) WHERE pid = $question_id)"""
        try:
            rows = graph.query(
                f"""
                MATCH (a:__Entity__)-[r]->(b:__Entity__)
                WHERE a.id IN $entity_ids
                  AND b.id IN $entity_ids
                  AND NOT type(r) IN ['HAS_ENTITY', 'FROM_CHUNK', 'PART_OF',
                                      'MENTIONED_IN', 'MENTIONS', 'QUALIFIES']
                  AND coalesce(r.confidence, 1.0) >= 0.4
                  {local_filters}
                  {qid_filter}
                RETURN a.id AS src, b.id AS tgt, coalesce(r.confidence, 1.0) AS conf
                LIMIT 2000
                """,
                params,
            ) or []
            return [(row["src"], row["tgt"], float(row["conf"])) for row in rows if row["src"] and row["tgt"]]
        except Exception as exc:
            logging.debug("_fetch_subgraph_edges failed (%s); PPR will fall back to hop scores.", exc)
            return []

    def _ppr_entity_scores(
        self,
        seed_ids: List[str],
        all_entity_ids: List[str],
        edges: List[Tuple],
    ) -> Dict[str, float]:
        """
        Approximate Personalized PageRank via confidence-weighted power iteration.

        Each edge carries a confidence weight (0.4–1.0 from the KG).  The random
        surfer splits probability mass across neighbours proportional to edge weight
        rather than uniformly, so high-confidence relation paths receive stronger
        activation than low-confidence ones.

        Returns {entity_id: ppr_score}.  Scores sum to 1.0 over all entities.
        """
        n = len(all_entity_ids)
        if n == 0:
            return {}

        idx = {eid: i for i, eid in enumerate(all_entity_ids)}

        # Build weighted undirected adjacency: adj[i] = {j: weight, ...}
        adj: List[Dict[int, float]] = [{} for _ in range(n)]
        for edge in edges:
            src, tgt = edge[0], edge[1]
            weight = float(edge[2]) if len(edge) > 2 else 1.0
            i, j = idx.get(src), idx.get(tgt)
            if i is not None and j is not None and i != j:
                adj[i][j] = max(adj[i].get(j, 0.0), weight)
                adj[j][i] = max(adj[j].get(i, 0.0), weight)

        seed_indices = [idx[s] for s in seed_ids if s in idx]
        e = np.zeros(n)
        if seed_indices:
            for si in seed_indices:
                e[si] = 1.0 / len(seed_indices)

        v = e.copy()
        alpha = self._PPR_ALPHA
        for _ in range(self._PPR_STEPS):
            new_v = np.zeros(n)
            for i in range(n):
                if adj[i]:
                    total_weight = sum(adj[i].values())
                    for j, w in adj[i].items():
                        new_v[j] += v[i] * (w / total_weight)
                else:
                    # Dangling nodes redistribute through the restart distribution
                    # so probability mass is preserved and scores stay comparable
                    # across sparse and dense local subgraphs.
                    new_v += v[i] * e
            v = (1.0 - alpha) * e + alpha * new_v

        return {eid: float(v[idx[eid]]) for eid in all_entity_ids}

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

    def _entity_first_search(self, graph, query: str, max_chunks: int = 20, kg_name: str = None, max_hops: int = 2, question_id: str = None, llm=None, document_names: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Entity-first retrieval: find KG entities whose names appear in the question,
        walk the graph from those seeds, then retrieve the chunks those entities came from.

        Improvements over baseline:
        - Hop-weighted chunk scores: seed chunks score 1.0, 1-hop 0.8, 2-hop 0.6.
        - Grounding quality is computed and returned with every result.
        - Sufficiency check: if < 2 chunks are recovered, returns None so the caller
          falls back to vector search rather than serving thin, low-recall context.

        Returns a context dict on success, or None to signal the caller should fall back
        to vector/text similarity search.
        """
        try:
            # Step 1a: embedding-based entity linking — find KG entities whose
            # embedding is closest to the question embedding.  This catches
            # partial-name matches (e.g. "TBK1" → "TBK1 kinase") that substring
            # search misses.
            entity_support_scope = self._question_local_entity_support_clause(
                "e",
                kg_name=kg_name,
                question_id=question_id,
                document_names=document_names,
            )
            kg_scope_exist = f"\n                  AND {entity_support_scope}"

            # Step 1a: ANN entity linking.
            #
            # HippoRAG-style: when an LLM is available, extract named entity
            # mentions from the question first, then embed each short entity
            # string individually and look it up in the entity_vector index.
            # entity-string → entity-embedding similarity is much tighter than
            # full-question → entity-embedding similarity, so we get fewer
            # hub-entity false positives and can use a higher threshold (0.72).
            #
            # Fallback (no llm, or extraction returns nothing): embed the full
            # query and use the original 0.55 threshold, same as before.
            query_embedding = self._generate_query_embedding(query)

            # Check entity_vector index dimension before querying — mismatches cause
            # a hard exception that aborts the entire entity-first search.
            emb_matched = []
            try:
                _idx_info = graph.query(
                    "SHOW INDEXES YIELD name, options WHERE name = 'entity_vector' RETURN options"
                )
                _idx_dim = None
                if _idx_info:
                    _opts = _idx_info[0].get("options") or {}
                    _idx_dim = (_opts.get("indexConfig") or {}).get("vector.dimensions")
                if _idx_dim is not None and int(_idx_dim) != self.embedding_dimension:
                    logging.warning(
                        "Skipping embedding entity lookup: index has %d dims, model has %d dims. "
                        "Rebuild the KG with the current embedding model to enable this.",
                        _idx_dim, self.embedding_dimension,
                    )
                else:
                    emb_lookup_query = f"""
                    CALL db.index.vector.queryNodes('entity_vector', 15, $query_vector)
                    YIELD node AS e, score
                    WHERE score >= $ann_threshold
                      {kg_scope_exist}
                    RETURN e.id AS id, e.name AS name,
                           coalesce(e.type, 'Entity') AS type,
                           elementId(e) AS element_id,
                           score AS ann_score,
                           coalesce(e.node_specificity, 1.0) AS node_specificity,
                           coalesce(e.passage_count, 1) AS passage_count,
                           CASE WHEN $use_node_specificity
                                THEN score * coalesce(e.node_specificity, 1.0)
                                ELSE score
                           END AS weighted_score
                    ORDER BY weighted_score DESC
                    """
                    ann_scope_params: Dict[str, Any] = {
                        "kg_name": kg_name,
                        "use_node_specificity": self.use_node_specificity,
                    }
                    if question_id:
                        ann_scope_params["question_id"] = question_id
                    if document_names:
                        ann_scope_params["document_names"] = document_names

                    # --- Per-entity embedding pass (HippoRAG-style) ---
                    query_entities: List[str] = []
                    if llm is not None and self.use_per_entity_ann:
                        query_entities = self._extract_query_entities(query, llm)

                    if query_entities:
                        # One ANN call per extracted entity; union results, keep best score.
                        seen_by_id: Dict[str, Any] = {}
                        for entity_mention in query_entities:
                            entity_vec = self._generate_query_embedding(entity_mention)
                            rows = graph.query(
                                emb_lookup_query,
                                {
                                    **ann_scope_params,
                                    "query_vector": entity_vec,
                                    "ann_threshold": 0.72,
                                },
                            ) or []
                            for row in rows:
                                row = dict(row)
                                eid = row["id"]
                                if eid not in seen_by_id or row["weighted_score"] > seen_by_id[eid]["weighted_score"]:
                                    seen_by_id[eid] = row
                        emb_matched = list(seen_by_id.values())
                        logging.info(
                            "Entity-first ANN (per-entity): extracted %d query entities → %d KG candidates",
                            len(query_entities), len(emb_matched),
                        )
                    else:
                        # Fallback: embed the full query (original behaviour)
                        emb_matched = graph.query(
                            emb_lookup_query,
                            {
                                **ann_scope_params,
                                "query_vector": query_embedding,
                                "ann_threshold": 0.55,
                            },
                        ) or []
            except Exception as _emb_err:
                logging.warning("Embedding entity lookup skipped: %s", _emb_err)

            # Step 1b: substring / token matching as a complementary pass
            params: Dict[str, Any] = {"query_text": query.lower()}
            if kg_name:
                params["kg_name"] = kg_name
            if question_id:
                params["question_id"] = question_id
            if document_names:
                params["document_names"] = document_names

            # Build a conservative content-token set for symbolic lookup.
            # Requiring either an exact phrase or at least two content-token overlaps
            # sharply reduces spurious open-domain matches on dataset KGs.
            query_tokens = self._content_query_tokens(query)
            params["query_tokens"] = query_tokens
            params["entity_lookup_limit"] = self._ENTITY_LOOKUP_LIMIT

            entity_lookup_query = self._build_symbolic_entity_lookup_query(entity_support_scope)
            sub_matched = graph.query(entity_lookup_query, params) or []

            # Merge both passes, deduplicating by entity id and keeping the best-ranked
            # candidate features. Exact phrase matches outrank overlap matches, which
            # outrank embedding-only matches.
            merged_by_id: Dict[str, Dict[str, Any]] = {}
            for row in emb_matched:
                row = dict(row)
                row.setdefault("exact_in_query", 0)
                row.setdefault("query_in_entity", 0)
                row.setdefault("token_overlap", 0)
                row["_embedding_score"] = float(row.get("ann_score", row.get("score", 0.0)))
                row["_source_kind"] = "embedding"
                merged_by_id[row["id"]] = row

            for row in sub_matched:
                row = dict(row)
                row["_embedding_score"] = float(
                    merged_by_id.get(row["id"], {}).get("_embedding_score", 0.0)
                )
                row["_source_kind"] = "symbolic"
                existing = merged_by_id.get(row["id"])
                if existing is None:
                    merged_by_id[row["id"]] = row
                    continue
                existing_rank = (
                    int(existing.get("exact_in_query", 0)),
                    int(existing.get("query_in_entity", 0)),
                    int(existing.get("token_overlap", 0)),
                    float(existing.get("_embedding_score", 0.0)),
                )
                new_rank = (
                    int(row.get("exact_in_query", 0)),
                    int(row.get("query_in_entity", 0)),
                    int(row.get("token_overlap", 0)),
                    float(row.get("_embedding_score", 0.0)),
                )
                if new_rank > existing_rank:
                    merged_by_id[row["id"]] = row

            matched = sorted(
                merged_by_id.values(),
                key=lambda row: (
                    int(row.get("exact_in_query", 0)),
                    int(row.get("query_in_entity", 0)),
                    int(row.get("token_overlap", 0)),
                    float(row.get("_embedding_score", 0.0)),
                    len(str(row.get("name") or row.get("id") or "")),
                ),
                reverse=True,
            )[:self._MAX_ENTITY_SEEDS]

            matched_query_token_count = self._count_matched_query_tokens(
                query,
                [row.get("name") or row.get("id") for row in matched],
            )
            grounding_q = self._grounding_quality(query, matched_query_token_count)

            if not matched:
                logging.info("Entity-first: no entities found in question (grounding=0)")
                return None

            # Disambiguation confidence gate: if grounding is weak AND no entity
            # has a symbolic signal (exact substring or ≥2 token overlaps), the
            # anchor is embedding-only on a noisy graph — skip traversal and fall
            # back to vector search rather than confidently traversing from the
            # wrong node.
            _min_grounding = self._ENTITY_MATCH_MIN_GROUNDING_BY_KG.get(
                (kg_name or "").lower(),
                self._ENTITY_MATCH_MIN_GROUNDING,
            )
            if _min_grounding > 0.0:
                _has_symbolic = any(
                    int(r.get("exact_in_query", 0))
                    or int(r.get("query_in_entity", 0))
                    or int(r.get("token_overlap", 0)) >= 2
                    for r in matched
                )
                if grounding_q < _min_grounding and not _has_symbolic:
                    logging.info(
                        "Entity-first: low-confidence anchor "
                        "(grounding=%.2f < %.2f, no symbolic match) — "
                        "skipping graph traversal, falling back to vector",
                        grounding_q, _min_grounding,
                    )
                    return None

            logging.info(
                "Entity-first: matched %d entities covering %d query tokens (grounding=%.2f): %s",
                len(matched), matched_query_token_count, grounding_q, [r["name"] for r in matched]
            )

            # Step 2: build seed entity dict (hop 0)
            entities: Dict[str, Any] = {}
            for row in matched:
                eid = row["id"]
                if not eid:
                    continue
                entities[eid] = {
                    "id": eid,
                    "element_id": row["element_id"],
                    "type": row["type"],
                    "description": row["name"],
                    "mentioned_in_chunks": [],
                    "source": "entity_lookup",
                    "min_hops": 0,
                }

            if not entities:
                return None

            # Step 3: walk the graph from seed entities to discover neighbors.
            # RoG-style: plan which relation types are query-relevant before
            # expanding, so traversal is constrained to semantically useful edges.
            seed_ids = list(entities.keys())
            allowed_rel_types = None
            if self.use_rog_path_planning:
                allowed_rel_types = self._plan_relation_paths(
                    graph,
                    seed_ids,
                    query,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
            expansion = self._expand_entities_via_graph(
                graph,
                seed_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
                allowed_rel_types=allowed_rel_types,
            )
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in entities:
                    entities[nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Step 4: retrieve chunks, tracking which entity (and therefore which hop
            # depth) each chunk was reached through so we can assign hop-weighted scores.
            all_entity_ids = list(entities.keys())
            candidate_chunk_limit = min(max_chunks * 10, 200)
            chunk_params: Dict[str, Any] = {
                "entity_ids": all_entity_ids,
                "candidate_chunk_limit": candidate_chunk_limit,
            }
            kg_chunk_filter = ""
            if kg_name:
                kg_chunk_filter = "AND d.kgName = $kg_name"
                chunk_params["kg_name"] = kg_name
            if document_names:
                kg_chunk_filter += " AND d.fileName IN $document_names"
                chunk_params["document_names"] = document_names
            # Scope to the current question's passages when question_id is provided.
            # This prevents cross-question entity contamination on datasets where each
            # question has its own closed passage set (source_document / retrieval_bundle).
            if question_id:
                kg_chunk_filter += " AND c.questionId = $question_id"
                chunk_params["question_id"] = question_id

            # Collect the minimum hop depth of any entity linked to each chunk so
            # we can assign the most favourable score when a chunk is reachable via
            # multiple entities at different depths.
            chunk_query = f"""
            MATCH (c:Chunk)-[:HAS_ENTITY]->(e:__Entity__)
            WHERE e.id IN $entity_ids
            MATCH (c)-[:PART_OF]->(d:Document)
            WHERE true {kg_chunk_filter}
            WITH c, d,
                 collect(DISTINCT e.id) AS linked_entity_ids,
                 count(DISTINCT e) AS linked_entity_count
            RETURN DISTINCT
                c.text AS text,
                c.id AS chunk_id,
                elementId(c) AS chunk_element_id,
                d.fileName AS document,
                d.kgName AS kg_name,
                c.position AS position,
                c.source AS source,
                c.questionId AS question_id,
                c.passageIndex AS passage_index,
                c.chunkLocalIndex AS chunk_local_index,
                linked_entity_ids,
                linked_entity_count
            ORDER BY linked_entity_count DESC, c.position ASC
            LIMIT $candidate_chunk_limit
            """
            chunk_results = graph.query(chunk_query, chunk_params)

            # Sufficiency check: fewer than 2 chunks usually means entity matching
            # found only a stray surface-form hit and the context will be too thin to
            # be useful.  Return None so the caller falls back to vector search.
            if not chunk_results or len(chunk_results) < 2:
                logging.info(
                    "Entity-first: only %d chunk(s) — below sufficiency threshold, falling back",
                    len(chunk_results) if chunk_results else 0,
                )
                return None

            # PPR scoring: fetch edges in the entity subgraph and run power iteration.
            # This replaces the fixed hop-score table with a principled random-walk
            # score that accounts for the full graph topology — entities connected to
            # more seeds via shorter, denser paths receive higher scores.
            all_entity_ids_for_ppr = list(entities.keys())
            seed_ids_for_ppr = [eid for eid, info in entities.items() if info.get("min_hops", 99) == 0]
            ppr_scores: Dict[str, float] = {}
            subgraph_edges: List[Tuple[str, str]] = []
            if self.use_ppr_scoring and seed_ids_for_ppr and len(all_entity_ids_for_ppr) > 1:
                subgraph_edges = self._fetch_subgraph_edges(
                    graph,
                    all_entity_ids_for_ppr,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                if subgraph_edges:
                    ppr_scores = self._ppr_entity_scores(
                        seed_ids_for_ppr, all_entity_ids_for_ppr, subgraph_edges
                    )
                    logging.info(
                        "Entity-first PPR: %d entities, %d edges, top score=%.4f",
                        len(all_entity_ids_for_ppr), len(subgraph_edges),
                        max(ppr_scores.values()) if ppr_scores else 0.0,
                    )

            # Normalise PPR scores to [0, 1] so they're comparable to hop scores.
            max_ppr = max(ppr_scores.values()) if ppr_scores else 0.0

            chunks = []
            documents: set = set()
            for row in chunk_results:
                linked = row.get("linked_entity_ids") or []
                if ppr_scores and max_ppr > 0:
                    # PPR path: score = avg normalised PPR of covering entities
                    # + coverage bonus (n'·P from HippoRAG).
                    entity_ppr = [
                        ppr_scores.get(eid, 0.0) / max_ppr
                        for eid in linked if eid in entities
                    ]
                    if entity_ppr:
                        base_score = sum(entity_ppr) / len(entity_ppr)
                        coverage_bonus = 0.05 * (len(entity_ppr) - 1)
                        hop_score = min(1.0, base_score + coverage_bonus)
                    else:
                        hop_score = self._HOP_SCORE_DEFAULT
                else:
                    # Fallback: fixed hop-score table (used when subgraph has no edges).
                    entity_hop_scores = [
                        self._HOP_SCORE.get(entities[eid]["min_hops"], self._HOP_SCORE_DEFAULT)
                        for eid in linked if eid in entities
                    ]
                    if entity_hop_scores:
                        base_score = sum(entity_hop_scores) / len(entity_hop_scores)
                        coverage_bonus = 0.05 * (len(entity_hop_scores) - 1)
                        hop_score = min(1.0, base_score + coverage_bonus)
                    else:
                        hop_score = self._HOP_SCORE_DEFAULT

                min_hop = min(
                    (entities[eid]["min_hops"] for eid in linked if eid in entities),
                    default=2,
                )
                chunks.append({
                    "text": row["text"],
                    "chunk_id": row["chunk_id"],
                    "chunk_element_id": row["chunk_element_id"],
                    "score": hop_score,
                    "min_hop": min_hop,
                    "document": row["document"],
                    "kg_name": row.get("kg_name"),
                    "position": row.get("position"),
                    "source": row.get("source"),
                    "question_id": row.get("question_id"),
                    "passage_index": row.get("passage_index"),
                    "chunk_local_index": row.get("chunk_local_index"),
                    "entities": [],
                    "linked_entity_count": int(row.get("linked_entity_count") or 0),
                    # Retained for post-truncation entity pruning in iterative retrieval
                    "linked_entity_ids": linked,
                })
                documents.add(row["document"])

            # --- Traversal relevance gate ---
            # Drop hop-1+ chunks whose text is not similar enough to the query.
            # This prevents entity neighbours from flooding the context with
            # passages that share an entity name but are topically off-target
            # (e.g. a historical "Vicente García González" pulled in because the
            # graph connected him to "Vicente García" the musician by name).
            # Seed chunks (min_hop == 0) are always kept.
            # Embeddings are fetched in a single batched query to avoid N+1 overhead.
            _min_sim = self._TRAVERSAL_CHUNK_MIN_SIM
            if _min_sim > 0.0 and query_embedding:
                import numpy as np
                qvec = np.array(query_embedding, dtype=float)
                qnorm = np.linalg.norm(qvec)
                # Collect element IDs of hop-1+ chunks that need similarity checks
                traversal_eids = [
                    c["chunk_element_id"]
                    for c in chunks
                    if c.get("min_hop", 1) > 0 and c.get("chunk_element_id")
                ]
                eid_to_sim: Dict[str, float] = {}
                if traversal_eids and qnorm > 0:
                    try:
                        _emb_rows = graph.query(
                            "MATCH (c:Chunk) WHERE elementId(c) IN $eids "
                            "RETURN elementId(c) AS eid, c.embedding AS emb",
                            {"eids": traversal_eids},
                        )
                        for _row in (_emb_rows or []):
                            _emb = _row.get("emb")
                            if not _emb:
                                continue
                            cvec = np.array(_emb, dtype=float)
                            cnorm = np.linalg.norm(cvec)
                            sim = float(np.dot(qvec, cvec) / (qnorm * cnorm)) if cnorm > 0 else 0.0
                            eid_to_sim[_row["eid"]] = sim
                    except Exception as _ge:
                        logging.debug("Traversal relevance gate: batch embedding fetch failed: %s", _ge)
                filtered_chunks = []
                for c in chunks:
                    if c.get("min_hop", 1) == 0:
                        filtered_chunks.append(c)
                        continue
                    eid = c.get("chunk_element_id")
                    sim = eid_to_sim.get(eid)
                    if sim is None or sim >= _min_sim:
                        # Keep if similarity unknown (no embedding stored) or above threshold
                        filtered_chunks.append(c)
                    else:
                        logging.debug(
                            "Entity-first: dropped traversal chunk (passage_index=%s, sim=%.3f < %.3f)",
                            c.get("passage_index"), sim, _min_sim,
                        )
                n_before = len(chunks)
                if filtered_chunks:
                    chunks = filtered_chunks
                logging.info(
                    "Traversal relevance gate: kept %d/%d chunks (min_sim=%.2f)",
                    len(chunks), n_before, _min_sim,
                )

            # --- Per-passage chunk cap ---
            # Limit how many chunks from any single source passage (same
            # questionId + passageIndex) enter the context.  A single passage
            # may produce many entity-linked chunks; taking them all starves
            # other passages that hold complementary evidence.
            if self._MAX_CHUNKS_PER_PASSAGE > 0:
                chunks = self._apply_per_passage_chunk_cap(chunks)

            # Sort by hop score descending so the LLM sees the most directly supported
            # chunks first (mirrors C2RAG's constraint-aligned evidence ordering).
            chunks.sort(key=self._chunk_rank_key)
            chunks = chunks[:max_chunks]

            # Step 5: relationships between all entities
            relationships = []
            if all_entity_ids:
                rel_results = self._fetch_relationships_for_entity_ids(
                    graph,
                    all_entity_ids,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                for rel in rel_results:
                    rel_key = (
                        rel.get("relationship_element_id")
                        or f"{rel['source']}-{rel['relationship_type']}-{rel['target']}"
                        f"-negated={bool(rel.get('negated', False))}"
                    )
                    relationships.append({
                        "key": rel_key,
                        "source": rel["source"],
                        "source_element_id": rel["source_element_id"],
                        "target": rel["target"],
                        "target_element_id": rel["target_element_id"],
                        "type": rel["relationship_type"],
                        "element_id": rel["relationship_element_id"],
                        "negated": bool(rel.get("negated", False)),
                        "condition": rel.get("condition"),
                        "quantitative": rel.get("quantitative"),
                        "confidence": rel.get("confidence"),
                        "question_ids": rel.get("question_ids") or [],
                        "passage_keys": rel.get("passage_keys") or [],
                        "provenance_positions": rel.get("provenance_positions") or [],
                    })

            return {
                "query": query,
                "chunks": chunks,
                "entities": entities,
                "relationships": relationships,
                "graph_neighbors": expansion["neighbors"],
                "traversal_paths": expansion["paths"],
                "documents": list(documents),
                "total_score": float(sum(c["score"] for c in chunks)),
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "search_method": "entity_first",
                "kg_name": kg_name,
                "seed_entity_count": len(seed_ids),
                "grounding_quality": grounding_q,
                "retrieval_route": "entity_first",
                "route_reason": "success",
                "diagnostics": {
                    "seed_entities": [entities[eid].get("name", eid) for eid in seed_ids[:10]],
                    "ann_match_count": len([e for e in matched if e.get("_source_kind") == "embedding"]),
                    "symbolic_match_count": len([e for e in matched if e.get("_source_kind") == "symbolic"]),
                    "rfge_fired": False,
                    "subgraph_edge_count": len(subgraph_edges),
                    "top_ppr_entities": sorted(
                        ((eid, score) for eid, score in ppr_scores.items()),
                        key=lambda x: -x[1],
                    )[:5] if ppr_scores else [],
                    "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto"),
                },
            }

        except Exception as e:
            logging.warning("Entity-first search failed: %s", e)
            return None

    def _retriever_first_graph_expansion(
        self,
        graph,
        query: str,
        max_chunks: int = 20,
        kg_name: str = None,
        max_hops: int = 2,
        question_id: str = None,
        document_names: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retriever-first graph expansion (RFGE).

        Complements entity-first by starting from dense vector retrieval rather
        than from question entities.  Entity-first excels when the question contains
        explicit named entities that exist in the KG; RFGE excels when the question
        is abstract or paraphrased but semantically-close passages are retrievable.

        Flow:
          1. Vector search → top-k chunks (with their linked entities).
          2. Collect entity IDs from those chunks as graph seeds.
          3. Expand the graph from those seeds (_expand_entities_via_graph).
          4. Fetch subgraph edges → run PPR to score all entities.
          5. Re-score chunks: combined = 0.5 * cosine + 0.5 * ppr_chunk_score.
          6. Fetch additional chunks for graph-neighbor entities not already retrieved.
          7. Return merged context with grounding_quality derived from PPR coverage.
        """
        try:
            # Step 1: vector retrieval — overfetch so we have enough seeds.
            vec_ctx = self._semantic_similarity_search(
                graph,
                query,
                document_names=document_names,
                similarity_threshold=self._RFGE_VECTOR_THRESHOLD,
                max_chunks=min(max_chunks * 3, 60),
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
            )
            if not vec_ctx or not vec_ctx.get("chunks"):
                return None

            # Step 2: collect entity seeds from returned chunks.
            vec_entity_ids: List[str] = []
            cosine_by_chunk: Dict[str, float] = {}
            _MAX_ENTITIES_PER_CHUNK = 4  # cap seeds per chunk to avoid generic-entity flooding
            for chunk in vec_ctx["chunks"]:
                cosine_by_chunk[chunk["chunk_id"]] = float(chunk.get("score", 0.0))
                chunk_ents = chunk.get("entities", [])
                added = 0
                for ent in chunk_ents:
                    if added >= _MAX_ENTITIES_PER_CHUNK:
                        break
                    eid = ent.get("id")
                    if not eid:
                        continue
                    eid_lower = eid.lower().strip()
                    if len(eid_lower) <= 2 or eid_lower in self._ENTITY_MATCH_STOPWORDS:
                        continue
                    if eid not in vec_entity_ids:
                        vec_entity_ids.append(eid)
                        added += 1

            if not vec_entity_ids:
                # No entities tagged on any retrieved chunk — RFGE can't expand.
                return None

            # Step 3: graph expansion from passage-derived seeds.
            expansion = self._expand_entities_via_graph(
                graph,
                vec_entity_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
            )

            # Build the combined entity pool: passage seeds + graph neighbors.
            all_entities: Dict[str, Any] = {}
            for eid in vec_entity_ids:
                all_entities[eid] = {"id": eid, "min_hops": 0, "source": "retriever_seed"}
            for nid, ninfo in expansion["neighbors"].items():
                all_entities[nid] = {**ninfo, "min_hops": ninfo.get("min_hops", 1)}

            # Step 4: PPR scoring over the full entity pool.
            all_eid_list = list(all_entities.keys())
            subgraph_edges = self._fetch_subgraph_edges(
                graph,
                all_eid_list,
                kg_name=kg_name,
                question_id=question_id,
                document_names=document_names,
            )
            ppr_scores: Dict[str, float] = {}
            max_ppr = 0.0
            if subgraph_edges:
                ppr_scores = self._ppr_entity_scores(vec_entity_ids, all_eid_list, subgraph_edges)
                max_ppr = max(ppr_scores.values()) if ppr_scores else 0.0

            # Step 5: re-score existing chunks with combined cosine + PPR signal.
            chunks = []
            for chunk in vec_ctx["chunks"]:
                cid = chunk["chunk_id"]
                cosine = cosine_by_chunk.get(cid, float(chunk.get("score", 0.0)))
                linked = [e.get("id") for e in chunk.get("entities", []) if e.get("id")]
                if ppr_scores and max_ppr > 0:
                    ppr_vals = [ppr_scores.get(eid, 0.0) / max_ppr for eid in linked if eid in ppr_scores]
                    if ppr_vals:
                        ppr_chunk = sum(ppr_vals) / len(ppr_vals) + 0.05 * (len(ppr_vals) - 1)
                        combined = 0.5 * cosine + 0.5 * min(1.0, ppr_chunk)
                    else:
                        combined = 0.5 * cosine  # no PPR signal for this chunk
                else:
                    combined = cosine
                chunks.append({**chunk, "score": combined, "linked_entity_ids": linked})

            # Step 6: fetch chunks for graph-neighbor entities not yet retrieved.
            neighbor_eids = [
                eid for eid in expansion["neighbors"]
                if ppr_scores.get(eid, 0.0) >= self._RFGE_MIN_PPR_SCORE
            ]
            if neighbor_eids:
                existing_cids = {c["chunk_id"] for c in chunks}
                extra_params: Dict[str, Any] = {
                    "entity_ids": neighbor_eids[:20],
                    "limit": max(4, max_chunks // 3),
                }
                scope = ""
                if kg_name:
                    scope += " AND d.kgName = $kg_name"
                    extra_params["kg_name"] = kg_name
                if document_names:
                    scope += " AND d.fileName IN $document_names"
                    extra_params["document_names"] = document_names
                if question_id:
                    scope += " AND c.questionId = $question_id"
                    extra_params["question_id"] = question_id
                extra_rows = graph.query(
                    f"""
                    MATCH (e:__Entity__)<-[:HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
                    WHERE e.id IN $entity_ids {scope}
                    RETURN DISTINCT c.text AS text, c.id AS chunk_id,
                           elementId(c) AS chunk_element_id,
                           d.fileName AS document, d.kgName AS kg_name,
                           c.questionId AS question_id,
                           c.position AS position,
                           c.passageIndex AS passage_index,
                           c.chunkLocalIndex AS chunk_local_index,
                           e.id AS seed_entity_id
                    LIMIT $limit
                    """,
                    extra_params,
                ) or []
                for row in extra_rows:
                    cid = row["chunk_id"]
                    if not cid or cid in existing_cids:
                        continue
                    eid = row.get("seed_entity_id", "")
                    ppr_val = ppr_scores.get(eid, 0.0) / max_ppr if max_ppr > 0 else 0.0
                    chunks.append({
                        "text": row["text"],
                        "chunk_id": cid,
                        "chunk_element_id": row["chunk_element_id"],
                        "score": 0.3 * ppr_val,  # graph-only: lower base score
                        "min_hop": 1,
                        "document": row["document"],
                        "kg_name": row.get("kg_name"),
                        "question_id": row.get("question_id"),
                        "position": row.get("position"),
                        "passage_index": row.get("passage_index"),
                        "chunk_local_index": row.get("chunk_local_index"),
                        "entities": [],
                        "linked_entity_ids": [eid] if eid else [],
                        "linked_entity_count": 1 if eid else 0,
                    })
                    existing_cids.add(cid)

            chunks.sort(key=lambda c: -float(c.get("score", 0.0)))
            chunks = chunks[:max_chunks]

            # Grounding quality: fraction of retrieved entities with PPR > threshold.
            supported = sum(1 for eid in vec_entity_ids if ppr_scores.get(eid, 0.0) >= self._RFGE_MIN_PPR_SCORE)
            grounding_q = supported / len(vec_entity_ids) if vec_entity_ids else 0.0

            relationships = []
            if all_eid_list:
                rel_results = self._fetch_relationships_for_entity_ids(
                    graph,
                    all_eid_list[:40],
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                for rel in rel_results:
                    relationships.append({
                        "key": (
                            rel.get("relationship_element_id")
                            or f"{rel['source']}-{rel['relationship_type']}-{rel['target']}"
                            f"-negated={bool(rel.get('negated', False))}"
                        ),
                        "source": rel["source"],
                        "source_element_id": rel["source_element_id"],
                        "target": rel["target"],
                        "target_element_id": rel["target_element_id"],
                        "type": rel["relationship_type"],
                        "element_id": rel["relationship_element_id"],
                        "negated": bool(rel.get("negated", False)),
                        "condition": rel.get("condition"),
                        "quantitative": rel.get("quantitative"),
                        "confidence": rel.get("confidence"),
                        "question_ids": rel.get("question_ids") or [],
                        "passage_keys": rel.get("passage_keys") or [],
                        "provenance_positions": rel.get("provenance_positions") or [],
                    })

            logging.info(
                "RFGE: %d seeds, %d total entities, %d edges, %d chunks, grounding=%.2f",
                len(vec_entity_ids), len(all_eid_list), len(subgraph_edges),
                len(chunks), grounding_q,
            )
            return {
                "query": query,
                "chunks": chunks,
                "entities": all_entities,
                "relationships": relationships,
                "graph_neighbors": expansion["neighbors"],
                "traversal_paths": expansion["paths"],
                "documents": list({c["document"] for c in chunks if c.get("document")}),
                "total_score": sum(c["score"] for c in chunks),
                "entity_count": len(all_entities),
                "relationship_count": len(relationships),
                "search_method": "retriever_first_graph_expansion",
                "kg_name": kg_name,
                "seed_entity_count": len(vec_entity_ids),
                "grounding_quality": grounding_q,
                "retrieval_route": "rfge",
                "route_reason": "entity_first_failed",
                "diagnostics": {
                    "seed_entities": vec_entity_ids[:10],
                    "ann_match_count": 0,
                    "symbolic_match_count": 0,
                    "rfge_fired": True,
                    "subgraph_edge_count": len(subgraph_edges),
                    "top_ppr_entities": sorted(
                        ((eid, score) for eid, score in ppr_scores.items()),
                        key=lambda x: -x[1],
                    )[:5] if ppr_scores else [],
                    "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto"),
                },
            }

        except Exception as exc:
            logging.warning("Retriever-first graph expansion failed: %s", exc)
            return None

    def _text_similarity_search(self, graph, query: str, document_names: List[str] = None, max_chunks: int = 20, kg_name: str = None, max_hops: int = 2, question_id: str = None) -> Dict[str, Any]:
        """
        Text-based similarity search as fallback when vector search fails
        """
        try:
            logging.info(f"Using text similarity search as fallback (kg_name: {kg_name})")

            # Build WHERE clause based on filters and text match
            where_conditions = ["ANY(term IN $search_terms WHERE toLower(c.text) CONTAINS term)"]
            params = {}

            if kg_name:
                where_conditions.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name

            if question_id:
                where_conditions.append("c.questionId = $question_id")
                params["question_id"] = question_id

            if document_names:
                where_conditions.append("d.fileName IN $document_names")
                params["document_names"] = document_names
            
            # Build a single WHERE clause to avoid duplicate WHERE errors
            where_clause = "WHERE " + " AND ".join(where_conditions)

            # Text similarity search: score by number of query terms matched in chunk text
            search_query = f"""
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
            {where_clause}
            WITH c, d,
                 size([term IN $search_terms WHERE toLower(c.text) CONTAINS term]) AS matched_terms
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            WITH c, d, matched_terms, collect(DISTINCT e) AS chunk_entities
            RETURN
                c.text AS text,
                c.id AS chunk_id,
                elementId(c) AS chunk_element_id,
                c.questionId AS question_id,
                c.passageIndex AS passage_index,
                c.chunkLocalIndex AS chunk_local_index,
                toFloat(matched_terms) / size($search_terms) AS score,
                d.fileName AS document,
                d.kgName AS kg_name,
                [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                    id: coalesce(entity.id, entity.name),
                    element_id: elementId(entity),
                    type: coalesce(entity.type, 'Entity'),
                    description: coalesce(entity.name, '')
                }}] AS entities
            ORDER BY score DESC
            LIMIT $max_chunks
            """

            # Tokenized fallback matching is more robust than a single 3-word phrase.
            raw_terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
            search_terms = [t for t in raw_terms if len(t) >= 3][:8]
            if not search_terms:
                search_terms = [query.lower().strip()]

            params.update({
                "search_terms": search_terms,
                "max_chunks": max_chunks
            })

            results = graph.query(search_query, params)

            context = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "documents": set(),
                "total_score": 0,
                "search_method": "text_similarity_fallback"
            }

            for result in results:
                result_entities = result.get("entities") or []
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "question_id": result.get("question_id"),
                    "passage_index": result.get("passage_index"),
                    "chunk_local_index": result.get("chunk_local_index"),
                    "score": result["score"],
                    "document": result["document"],
                    "kg_name": result.get("kg_name"),
                    "entities": result.get("entities") or []
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]

                # Collect unique entities
                for entity in (result.get("entities") or []):
                    entity_id = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": []
                        }
                    if result.get("chunk_id"):
                        context["entities"][entity_id]["mentioned_in_chunks"].append(result["chunk_id"])

            # Multi-hop graph traversal from seed entities
            seed_ids = list(context["entities"].keys())
            expansion = self._expand_entities_via_graph(
                graph,
                seed_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
            )
            context["graph_neighbors"] = expansion["neighbors"]
            context["traversal_paths"] = expansion["paths"]

            # Merge neighbor entities into the entity dict (tagged so they're
            # distinguishable from directly-retrieved seed entities)
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in context["entities"]:
                    context["entities"][nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Relationships between ALL entities (seeds + neighbors)
            all_entity_ids = list(context["entities"].keys())
            if all_entity_ids:
                relationship_results = self._fetch_relationships_for_entity_ids(
                    graph,
                    all_entity_ids,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                for rel_result in relationship_results:
                    rel_key = (
                        rel_result.get("relationship_element_id")
                        or f"{rel_result['source']}-{rel_result['relationship_type']}-{rel_result['target']}"
                        f"-negated={bool(rel_result.get('negated', False))}"
                    )
                    if not any(r.get('key') == rel_key for r in context["relationships"]):
                        context["relationships"].append({
                            "key": rel_key,
                            "source": rel_result["source"],
                            "source_element_id": rel_result["source_element_id"],
                            "target": rel_result["target"],
                            "target_element_id": rel_result["target_element_id"],
                            "type": rel_result["relationship_type"],
                            "element_id": rel_result["relationship_element_id"],
                            "negated": bool(rel_result.get("negated", False)),
                            "condition": rel_result.get("condition"),
                            "quantitative": rel_result.get("quantitative"),
                            "confidence": rel_result.get("confidence"),
                            "question_ids": rel_result.get("question_ids") or [],
                            "passage_keys": rel_result.get("passage_keys") or [],
                            "provenance_positions": rel_result.get("provenance_positions") or [],
                        })

            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])

            return context

        except Exception as e:
            logging.error(f"Error in text similarity search: {e}")
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0,
                "search_method": "text_similarity_fallback",
                "error": str(e)
            }

    def _build_evidence_block(self, context: Dict[str, Any]) -> str:
        """
        KG2RAG-style evidence organization.

        Groups graph traversal paths with the text passages that support them
        (overlap in linked entity IDs), then lists remaining passages as
        "Additional Evidence".  This makes multi-hop reasoning chains explicit
        to the LLM and reduces the chance of hallucinated bridge steps.
        """
        chunks = context.get("chunks", [])
        traversal_paths = context.get("traversal_paths", [])
        relationships = context.get("relationships", [])
        entities = context.get("entities", {})

        def _chunk_passage_key(chunk: Dict[str, Any]) -> Optional[str]:
            qid = chunk.get("question_id")
            if qid is None or not str(qid).strip():
                return None
            return f"{qid}::p{chunk.get('passage_index', -1)}"

        def _chunk_supports_path(chunk: Dict[str, Any], path_info: Dict[str, Any]) -> bool:
            chunk_positions = set()
            pos = chunk.get("position")
            if isinstance(pos, (int, float)):
                chunk_positions.add(int(pos))
            path_positions = {
                int(p)
                for p in (path_info.get("provenance_positions") or [])
                if isinstance(p, (int, float))
            }
            if path_positions:
                return bool(chunk_positions & path_positions)

            chunk_passage_key = _chunk_passage_key(chunk)
            path_passage_keys = {
                str(key)
                for key in (path_info.get("passage_keys") or [])
                if str(key).strip()
            }
            if path_passage_keys and chunk_passage_key is not None:
                return chunk_passage_key in path_passage_keys

            chunk_entity_ids = set(chunk.get("linked_entity_ids") or [])
            path_nodes = set(path_info.get("node_ids") or [])
            if not chunk_entity_ids or not path_nodes:
                return False

            overlap = chunk_entity_ids & path_nodes
            # When provenance metadata is absent, require at least two path
            # entities for a chunk to count as chain support. This avoids
            # treating a passage that merely mentions one endpoint as evidence
            # for the whole relation/path.
            required_overlap = min(2, len(path_nodes))
            return len(overlap) >= required_overlap

        # Index chunks by their linked entities for fast lookup.
        chunk_by_id: Dict[str, Any] = {c["chunk_id"]: c for c in chunks if c.get("chunk_id")}

        # Build a unified path list from both traversal paths and direct relationships.
        # Each entry: {path_str, node_ids, hops}
        all_paths: List[Dict[str, Any]] = []
        for p in sorted(traversal_paths, key=lambda x: x.get("hops", 99)):
            all_paths.append({
                "path_str": p.get("path", ""),
                "node_ids": set(p.get("node_ids") or []),
                "hops": p.get("hops", 1),
                "provenance_positions": list(p.get("provenance_positions") or []),
                "passage_keys": list(p.get("passage_keys") or []),
            })
        entity_name_lookup = {eid: (info.get("name") or info.get("description") or eid)
                              for eid, info in entities.items()}
        MAX_REL_PATHS = max(0, 25 - len(all_paths))
        for rel in relationships[:MAX_REL_PATHS]:
            src = entity_name_lookup.get(rel["source"], rel["source"])[:50]
            tgt = entity_name_lookup.get(rel["target"], rel["target"])[:50]
            rel_label = self._format_relationship_label(rel)
            all_paths.append({
                "path_str": f"{src} --{rel_label}--> {tgt}",
                "node_ids": {rel["source"], rel["target"]},
                "hops": 1,
                "provenance_positions": list(rel.get("provenance_positions") or []),
                "passage_keys": list(rel.get("passage_keys") or []),
            })

        # Associate each chunk with up to two supporting chains so shared bridge
        # evidence can appear in multiple reasoning chains without flooding all of them.
        chunk_chain_count: Dict[str, int] = {}
        _MAX_CHAINS_PER_CHUNK = 2
        chain_blocks: List[str] = []
        ungrounded_paths: List[str] = []  # paths with no supporting passage
        for chain_idx, path_info in enumerate(all_paths, 1):
            path_str = path_info["path_str"]
            if not path_str:
                continue
            supporting: List[Any] = []
            for cid, chunk in chunk_by_id.items():
                if chunk_chain_count.get(cid, 0) >= _MAX_CHAINS_PER_CHUNK:
                    continue
                if _chunk_supports_path(chunk, path_info):
                    supporting.append(chunk)
            if not supporting:
                # Collect passage-unsupported paths separately; they are emitted
                # in a clearly-labelled structural hints section so the LLM knows
                # not to treat them as confirmed textual evidence.
                ungrounded_paths.append(
                    f"  {path_info['hops']} hop: {path_str}"
                )
            else:
                lines = [
                    f"Chain {chain_idx} ({path_info['hops']} hop{'s' if path_info['hops'] != 1 else ''}): "
                    f"{path_str}"
                ]
                for i, chunk in enumerate(supporting[:3], 1):
                    text = (chunk.get("text") or "").strip()[:400]
                    lines.append(f"  [P{i}] {text}")
                    chunk_chain_count[chunk["chunk_id"]] = chunk_chain_count.get(chunk["chunk_id"], 0) + 1
                chain_blocks.append("\n".join(lines))

        # Remaining chunks become "Additional Evidence".
        additional: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            if chunk_chain_count.get(chunk.get("chunk_id"), 0) == 0:
                text = (chunk.get("text") or "").strip()
                additional.append(f"[{i}] {text}")

        sections: List[str] = []
        if chain_blocks:
            sections.append("REASONING CHAINS:\n" + "\n\n".join(chain_blocks))
        if additional:
            sections.append("ADDITIONAL PASSAGES:\n" + "\n\n".join(additional))
        if ungrounded_paths:
            # Clearly labelled so the LLM treats these as weak structural hints
            # only, not as textual evidence.  The prompt instructs the LLM to
            # ignore this section if passage evidence is available.
            sections.append(
                "STRUCTURAL HINTS (graph paths — no passage retrieved; "
                "use only if no passage evidence addresses the question):\n"
                + "\n".join(ungrounded_paths)
            )
        if not sections:
            sections.append("(No evidence retrieved)")

        return "\n\n".join(sections)

    def _build_passages_only_evidence_block(self, context: Dict[str, Any]) -> str:
        chunks = context.get("chunks", []) or []
        if not chunks:
            return "(No passages retrieved)"
        lines: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            text = (chunk.get("text") or "").strip()
            if text:
                lines.append(f"[{idx}] {text}")
        if not lines:
            return "(No passages retrieved)"
        return "PASSAGES:\n" + "\n\n".join(lines)

    def format_context_for_llm(self, context: Dict[str, Any]) -> tuple:
        """
        Format the context for the LLM prompt.

        Returns:
            (evidence_block, formatted_entities, formatted_paths)

        evidence_block is the KG2RAG-style organized evidence string passed to
        {evidence_block} in the prompt template.  formatted_entities and
        formatted_paths are retained for backward-compatible callers.
        """
        evidence_block = self._build_evidence_block(context)

        # --- legacy: flat chunk list (for callers that use formatted_context directly) ---
        chunk_texts = []
        for i, chunk in enumerate(context["chunks"], 1):
            doc = chunk.get('document', 'unknown')
            chunk_text = f"[Source: {doc}]\n{chunk['text']}\n"
            chunk_texts.append(chunk_text)
        formatted_context = "\n".join(chunk_texts)

        # --- graph paths (flat, for backward compat) ---
        path_lines = []
        traversal_paths = context.get("traversal_paths", [])
        if traversal_paths:
            sorted_paths = sorted(traversal_paths, key=lambda p: p.get("hops", 99))
            path_lines.extend([f"  {p['path']}" for p in sorted_paths])
        entity_name_lookup = {
            eid: (info.get("description") or eid)
            for eid, info in context.get("entities", {}).items()
        }
        MAX_RELATIONSHIPS = 25
        for rel in context.get("relationships", [])[:MAX_RELATIONSHIPS]:
            src_name = entity_name_lookup.get(rel["source"], rel["source"])[:50]
            tgt_name = entity_name_lookup.get(rel["target"], rel["target"])[:50]
            rel_type = self._format_relationship_label(rel)
            path_lines.append(f"  {src_name} --{rel_type}--> {tgt_name}")
        formatted_paths = "\n".join(path_lines) if path_lines else "(No graph relationships found)"

        # --- entities ---
        entity_texts = []
        seed_entities = {
            eid: info for eid, info in context["entities"].items()
            if info.get("source") != "graph_traversal"
        }
        neighbor_entities = {
            eid: info for eid, info in context["entities"].items()
            if info.get("source") == "graph_traversal"
        }

        MAX_SEED_ENTITIES = 15
        MAX_NEIGHBOR_ENTITIES = 10

        if seed_entities:
            entity_texts.append("Seed entities (directly from retrieved chunks):")
            for entity_id, info in list(seed_entities.items())[:MAX_SEED_ENTITIES]:
                eid = info.get('id') or entity_id or 'Unknown'
                etype = info.get('type') or 'Unknown'
                line = f"  - {eid} | Type: {etype}"
                desc = info.get("description")
                if desc and desc != eid:
                    line += f" | {desc}"
                entity_texts.append(line)
            if len(seed_entities) > MAX_SEED_ENTITIES:
                entity_texts.append(f"  ... ({len(seed_entities) - MAX_SEED_ENTITIES} more omitted)")

        if neighbor_entities:
            shown = list(neighbor_entities.items())[:MAX_NEIGHBOR_ENTITIES]
            entity_texts.append(f"\nGraph-traversal neighbors ({len(neighbor_entities)} discovered, showing top {len(shown)}):")
            for entity_id, info in shown:
                eid = info.get('id') or entity_id or 'Unknown'
                etype = info.get('type') or 'Unknown'
                hops = info.get("min_hops", "?")
                line = f"  - {eid} | Type: {etype} | {hops}-hop neighbor"
                desc = info.get("description")
                if desc and desc != eid:
                    line += f" | {desc}"
                entity_texts.append(line)

        formatted_entities = "\n".join(entity_texts) if entity_texts else "(No entities found)"

        return evidence_block, formatted_entities, formatted_paths

    def _runtime_answer_guardrail_enabled(self, runtime_guardrail: Optional[bool]) -> bool:
        if runtime_guardrail is not None:
            return bool(runtime_guardrail)
        return str(os.getenv("ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL", "0")).strip().lower() in {
            "1", "true", "yes", "on",
        }

    def _runtime_answer_guardrail_mode(self, runtime_guardrail_mode: Optional[str]) -> str:
        mode = str(
            runtime_guardrail_mode
            or os.getenv("ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL_MODE", "retry_then_abstain")
        ).strip().lower()
        if mode not in {"retry_then_abstain", "abstain_only"}:
            return "retry_then_abstain"
        return mode

    def _invoke_answer_chain(
        self,
        *,
        question: str,
        llm,
        context: Dict[str, Any],
        answer_instructions: str = "",
        extra_context_texts: Optional[List[str]] = None,
        kg_name: Optional[str] = None,
    ) -> str:
        evidence_block, formatted_entities, formatted_paths = self.format_context_for_llm(context)
        if self._should_use_passage_only_answer_prompt(kg_name):
            evidence_block = self._build_passages_only_evidence_block(context)
        elif not self.use_evidence_block:
            # Flat-format fallback for ablation: just concatenated chunk texts,
            # optionally followed by graph paths for lightweight structure.
            chunk_texts = []
            for i, chunk in enumerate(context.get("chunks", []), 1):
                chunk_texts.append(f"[{i}] {chunk.get('text', '').strip()}")
            flat_passages = "\n\n".join(chunk_texts) or "(No passages retrieved)"
            evidence_block = f"PASSAGES:\n{flat_passages}"
            if formatted_paths and "No graph" not in formatted_paths:
                evidence_block += f"\n\nGRAPH PATHS:\n{formatted_paths}"

        if extra_context_texts:
            extra_block = "\n\n".join(
                f"Provided Context {i+1}:\n{t}" for i, t in enumerate(extra_context_texts) if t.strip()
            )
            evidence_block = extra_block + "\n\n" + evidence_block if evidence_block else extra_block
            logging.info(f"Prepended {len(extra_context_texts)} extra context(s) to prompt")

        chain = self.rag_prompt | llm | StrOutputParser()
        return chain.invoke({
            "evidence_block": evidence_block,
            "question": question,
            "answer_instructions": answer_instructions or "No additional formatting constraints.",
        })

    def _apply_runtime_answer_guardrail(
        self,
        *,
        question: str,
        llm,
        response: str,
        context: Dict[str, Any],
        runtime_guardrail: Optional[bool],
        runtime_guardrail_mode: Optional[str],
        retry_factory=None,
    ) -> tuple:
        if not self._runtime_answer_guardrail_enabled(runtime_guardrail):
            return response, context, {
                "enabled": False,
                "mode": self._runtime_answer_guardrail_mode(runtime_guardrail_mode),
            }

        mode = self._runtime_answer_guardrail_mode(runtime_guardrail_mode)
        verdict = evaluate_runtime_answer_guardrail(
            question=question,
            answer=response,
            chunks=context.get("chunks", []),
            llm=llm,
        )
        metadata: Dict[str, Any] = {
            "enabled": True,
            "mode": mode,
            "initial_verdict": verdict,
            "final_decision": verdict["decision"],
            "retried": False,
        }
        if verdict["decision"] == "keep":
            return response, context, metadata

        if mode == "retry_then_abstain" and verdict["decision"] == "retry" and retry_factory is not None:
            retry_result = retry_factory()
            if retry_result:
                retry_response, retry_context = retry_result
                retry_verdict = evaluate_runtime_answer_guardrail(
                    question=question,
                    answer=retry_response,
                    chunks=retry_context.get("chunks", []),
                    llm=llm,
                )
                metadata["retried"] = True
                metadata["retry_verdict"] = retry_verdict
                if retry_verdict["decision"] == "keep":
                    metadata["final_decision"] = "retry_keep"
                    return retry_response, retry_context, metadata
                context = retry_context
                metadata["final_decision"] = "abstain"

        metadata["final_decision"] = "abstain"
        return RUNTIME_GUARDRAIL_ABSTENTION, context, metadata

    # ------------------------------------------------------------------
    # IRCoT / StepChain-style iterative multi-hop retrieval
    # ------------------------------------------------------------------

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

    def _iterative_hop_retrieval(
        self,
        graph,
        question: str,
        sub_questions: List[str],
        max_chunks: int,
        kg_name: str,
        max_hops: int,
        llm,
        similarity_threshold: float,
        document_names: List[str],
        question_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Per-hop retrieval with bridge answer extraction between hops (IRCoT / StepChain).

        For each sub-question:
          1. Run entity-first search (or vector fallback) for that sub-question.
          2. If not the last hop, ask the LLM for a concise bridge answer from the
             retrieved chunks; substitute it into the next sub-question.
          3. Accumulate unique chunks, entities, and relationships across all hops.

        Returns a merged context dict, or None if every hop returned empty results.
        """
        bridge_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are given a sub-question and supporting text passages. "
                "Answer the sub-question in ONE short phrase (a name, date, or short clause). "
                "If multiple candidates are plausible, prefer the one that is explicitly "
                "supported by the passages and best enables the next sub-question. "
                "Return ONLY the answer phrase with no explanation."
            )),
            ("human", "Sub-question: {sub_q}\nNext sub-question: {next_sub_q}\n\nPassages:\n{passages}"),
        ])
        bridge_chain = bridge_prompt | llm | StrOutputParser()

        # Per-hop buckets so we can prune graph facts after chunk truncation.
        # Each entry: {"chunks": [...], "entities": {...}, "relationships": [...],
        #              "graph_neighbors": {...}, "traversal_paths": [...]}
        hop_buckets: List[Dict[str, Any]] = []

        # grounding / seed metrics are anchored to hop 0 (the original question).
        # Captured once after the first hop completes; never overwritten.
        _hop0_seed_count: int = 0
        _hop0_grounding: float = 0.0

        seen_chunk_ids: set = set()

        current_sub_qs = list(sub_questions)
        iterative_subquestion_max_hops = self._iterative_subquestion_max_hops_for_kg(
            kg_name,
            max_hops,
        )

        for hop_idx, sub_q in enumerate(current_sub_qs):
            logging.info("Iterative hop %d/%d — sub-question: %s", hop_idx + 1, len(current_sub_qs), sub_q)
            hop_max = max(6, max_chunks // len(current_sub_qs))
            hop_retrieval_k = self._iterative_hop_retrieval_budget(kg_name, hop_idx, hop_max)
            comparison_query = bool(self._comparison_branches(sub_q))
            next_sub_q = current_sub_qs[hop_idx + 1] if hop_idx < len(current_sub_qs) - 1 else ""
            retrieval_query = self._iterative_retrieval_query(
                question,
                sub_q,
                kg_name,
                hop_idx,
                next_sub_question=next_sub_q,
            )
            if retrieval_query != sub_q:
                logging.info(
                    "Iterative hop %d retrieval anchored with original question context",
                    hop_idx + 1,
                )

            # Primary: entity-first search with a bounded local graph walk per
            # sub-question. This is more permissive than the old 1-hop cap but
            # still constrained enough to avoid uncontrolled graph explosion.
            hop_ctx = self._entity_first_search(
                graph,
                retrieval_query,
                max_chunks=hop_retrieval_k,
                kg_name=kg_name,
                max_hops=iterative_subquestion_max_hops,
                question_id=question_id,
                llm=llm,
                document_names=document_names,
            )

            # Hybrid retrieval for multi-hop KG-RAG:
            # even when entity-first finds chunks, we still probe vector search so
            # comparison questions and weakly grounded hops can recover attribute-
            # bearing passages the graph path missed.
            if hop_ctx and hop_ctx.get("chunks") and (self.check_vector_index() or self._first_stage_late_interaction_enabled()):
                try:
                    vec_ctx = self._semantic_similarity_search(
                        graph, retrieval_query, document_names, similarity_threshold,
                        hop_retrieval_k, kg_name, max_hops=iterative_subquestion_max_hops,
                        question_id=question_id,
                        allow_first_stage_late_interaction=getattr(self, "retrieval_mode", "hybrid_auto") != "entity_first",
                    )
                    if vec_ctx and vec_ctx.get("chunks"):
                        secondary_limit = max(2, min(self._HYBRID_SUPPLEMENT_LIMIT, hop_retrieval_k // 2 or 1))
                        if self._graph_context_is_meaningful(retrieval_query, hop_ctx) and not comparison_query:
                            hop_ctx = self._merge_retrieval_contexts(
                                hop_ctx,
                                vec_ctx,
                                max_chunks=hop_retrieval_k,
                                search_method="hybrid",
                                secondary_limit=secondary_limit,
                                min_secondary_score=0.55,
                            )
                        else:
                            # Comparison-style hops and weak graph hops should let
                            # vector evidence lead, while retaining whatever graph
                            # facts were retrieved as secondary support.
                            hop_ctx = self._merge_retrieval_contexts(
                                vec_ctx,
                                hop_ctx,
                                max_chunks=hop_retrieval_k,
                                search_method="hybrid_vector_primary",
                                secondary_limit=secondary_limit,
                                min_secondary_score=0.55,
                            )
                except Exception as _ve:
                    logging.debug("Hybrid vector augmentation failed at hop %d: %s", hop_idx + 1, _ve)

            # Fallback: go straight to vector search, bypassing get_rag_context so we
            # don't re-run _entity_first_search a second time for the same sub-question.
            if not hop_ctx or not hop_ctx.get("chunks"):
                has_vec = self.check_vector_index() or self._first_stage_late_interaction_enabled()
                try:
                    if has_vec:
                        hop_ctx = self._semantic_similarity_search(
                            graph, retrieval_query, document_names, similarity_threshold,
                            hop_retrieval_k, kg_name, max_hops=iterative_subquestion_max_hops,
                            question_id=question_id,
                            allow_first_stage_late_interaction=getattr(self, "retrieval_mode", "hybrid_auto") != "entity_first",
                        )
                    if not hop_ctx or not hop_ctx.get("chunks"):
                        hop_ctx = self._text_similarity_search(
                            graph, retrieval_query, document_names, hop_retrieval_k, kg_name, max_hops=iterative_subquestion_max_hops,
                            question_id=question_id,
                        )
                except Exception as _fe:
                    logging.warning("Hop %d fallback search failed: %s", hop_idx + 1, _fe)
                    hop_ctx = None

            if not hop_ctx or not hop_ctx.get("chunks"):
                logging.info("Hop %d: no chunks found, continuing", hop_idx + 1)
                hop_buckets.append({"all_chunk_ids": set(), "new_chunks": [],
                                    "entities": {}, "relationships": [],
                                    "graph_neighbors": {}, "traversal_paths": []})
                continue

            # Separate "all chunk IDs retrieved this hop" (for activation checking, pre-dedup)
            # from "new chunks not seen in prior hops" (what gets added to the pool).
            # Using all_chunk_ids for activation ensures that a hop's graph facts are kept
            # whenever ANY of its chunks survive truncation, even if they were first seen in
            # an earlier hop and therefore not counted as new here.
            all_hop_cids: set = {c.get("chunk_id") or c.get("text", "")[:60]
                                  for c in hop_ctx["chunks"]}
            new_chunks = []
            for chunk in hop_ctx["chunks"]:
                cid = chunk.get("chunk_id") or chunk.get("text", "")[:60]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    new_chunks.append(chunk)

            hop_buckets.append({
                "all_chunk_ids": all_hop_cids,   # pre-dedup, used for activation
                "new_chunks": new_chunks,          # only newly seen chunks added to pool
                "entities": hop_ctx.get("entities", {}),
                "relationships": hop_ctx.get("relationships", []),
                "graph_neighbors": hop_ctx.get("graph_neighbors", {}),
                "traversal_paths": hop_ctx.get("traversal_paths", []),
            })

            # Anchor grounding/seed metrics to hop 0 only (direct question entities).
            # When entity-first gated out and the hop fell back to vector/text search,
            # those contexts carry no grounding_quality, so compute it from question vs
            # the entity IDs that appeared in the returned context.
            if hop_idx == 0:
                if "grounding_quality" in hop_ctx and "seed_entity_count" in hop_ctx:
                    _hop0_grounding = hop_ctx["grounding_quality"]
                    _hop0_seed_count = hop_ctx["seed_entity_count"]
                else:
                    # Vector/text fallback: entity keys are opaque IDs, not readable names.
                    # Estimate grounding from the coverage of content-bearing question tokens.
                    surfaces: List[str] = []
                    for einfo in hop_ctx.get("entities", {}).values():
                        surface = (
                            einfo.get("description") or
                            einfo.get("name") or
                            einfo.get("id") or ""
                        )
                        if surface and len(surface) >= 4:
                            surfaces.append(surface)
                    matched_token_count = self._count_matched_query_tokens(question, surfaces)
                    _hop0_seed_count = len(surfaces)
                    _hop0_grounding = self._grounding_quality(question, matched_token_count)

            # Extract bridge answer to refine the next sub-question
            if hop_idx < len(current_sub_qs) - 1 and new_chunks:
                bridge_passage_limit = 4
                if hop_idx > 0 and kg_name in self._ORIGINAL_QUESTION_ANCHOR_KGS:
                    bridge_passage_limit = self._ANCHOR_LATER_HOP_BRIDGE_PASSAGE_LIMIT
                passages = "\n\n".join(c["text"] for c in new_chunks[:bridge_passage_limit])
                try:
                    raw_bridge = bridge_chain.invoke(
                        {
                            "sub_q": sub_q,
                            "next_sub_q": next_sub_q,
                            "passages": passages,
                        }
                    ).strip()
                    _bad_phrases = ("i don't know", "i do not know", "cannot determine",
                                    "not mentioned", "no information", "unknown")
                    if (raw_bridge
                            and len(raw_bridge) <= 200
                            and not any(p in raw_bridge.lower() for p in _bad_phrases)):
                        logging.info("Bridge answer for hop %d: %s", hop_idx + 1, raw_bridge)
                        next_q = current_sub_qs[hop_idx + 1]
                        if "[BRIDGE]" in next_q:
                            current_sub_qs[hop_idx + 1] = next_q.replace("[BRIDGE]", raw_bridge)
                        else:
                            current_sub_qs[hop_idx + 1] = f"{next_q} (context: {raw_bridge})"

                        # Entity re-seeding: extract named entities from the bridge
                        # answer and prepend them to the next sub-question so that
                        # _entity_first_search can seed the graph walk from the
                        # intermediate entity rather than re-embedding the full text.
                        if llm:
                            try:
                                bridge_entities = self._extract_query_entities(raw_bridge, llm)
                                if bridge_entities:
                                    entity_hint = ", ".join(bridge_entities[:3])
                                    current_sub_qs[hop_idx + 1] = (
                                        f"{entity_hint}: {current_sub_qs[hop_idx + 1]}"
                                    )
                                    logging.info(
                                        "Hop %d entity re-seed: %s",
                                        hop_idx + 1, entity_hint,
                                    )
                            except Exception as _ee:
                                logging.debug("Bridge entity extraction failed: %s", _ee)
                    else:
                        logging.info(
                            "Bridge answer for hop %d rejected (empty/unhelpful): %r",
                            hop_idx + 1, raw_bridge,
                        )
                except Exception as e:
                    logging.warning("Bridge answer extraction failed at hop %d: %s", hop_idx + 1, e)

        # --- Flatten all hop chunks (new-only), sort, and truncate ---
        all_chunks: List[Dict] = []
        for bucket in hop_buckets:
            all_chunks.extend(bucket["new_chunks"])
        if not all_chunks:
            return None

        all_chunks = self._sort_chunks_for_query(question, all_chunks)
        all_chunks = all_chunks[:max_chunks]

        # --- Build chunk-level entity set from retained chunks ---
        # Use all_chunk_ids (pre-dedup) for hop activation so that a hop with a
        # deduplicated chunk still becomes active when that chunk is in the final set.
        retained_cids = {c.get("chunk_id") or c.get("text", "")[:60] for c in all_chunks}
        active_hops: set = set()
        for h_idx, bucket in enumerate(hop_buckets):
            if bucket["all_chunk_ids"] & retained_cids:
                active_hops.add(h_idx)

        # Derive the entity set directly from retained chunks (chunk-level granularity):
        # - entity-first chunks carry "linked_entity_ids" (set at construction time)
        # - vector/text chunks carry "entities" list of dicts with "id" keys
        retained_eids: set = set()
        for chunk in all_chunks:
            for eid in chunk.get("linked_entity_ids") or []:
                if eid:
                    retained_eids.add(eid)
            for e in chunk.get("entities") or []:
                eid = e.get("id") if isinstance(e, dict) else None
                if eid:
                    retained_eids.add(eid)

        # --- Merge graph facts from active hops, then filter to retained entity set ---
        merged_entities: Dict[str, Any] = {}
        merged_relationships: List[Dict] = []
        merged_graph_neighbors: Dict[str, Any] = {}
        merged_traversal_paths: List[Dict] = []
        seen_rel_keys: set = set()
        seen_path_strs: set = set()

        for h_idx in active_hops:
            bucket = hop_buckets[h_idx]
            for eid, einfo in bucket["entities"].items():
                if eid in retained_eids:
                    merged_entities.setdefault(eid, einfo)
            for rel in bucket["relationships"]:
                src, tgt = rel.get("source"), rel.get("target")
                if src not in retained_eids or tgt not in retained_eids:
                    continue
                rkey = rel.get("key", f"{src}-{rel.get('type')}-{tgt}")
                if rkey not in seen_rel_keys:
                    seen_rel_keys.add(rkey)
                    merged_relationships.append(rel)
            for nid, ninfo in bucket["graph_neighbors"].items():
                if nid in retained_eids:
                    merged_graph_neighbors.setdefault(nid, ninfo)
            for path_entry in bucket["traversal_paths"]:
                pstr = path_entry.get("path", "")
                if not pstr or pstr in seen_path_strs:
                    continue
                # Prune paths that mention nodes outside the retained entity set.
                # Paths without node_ids (built before this fix) are kept as-is.
                path_nids = path_entry.get("node_ids")
                if path_nids and not all(nid in retained_eids for nid in path_nids):
                    continue
                seen_path_strs.add(pstr)
                merged_traversal_paths.append(path_entry)

        # --- Compute final provenance fields from the actual retained chunk set ---
        final_total_score = sum(c.get("score", 0.0) for c in all_chunks)
        final_docs = {c["document"] for c in all_chunks if c.get("document")}

        return {
            "query": question,
            "chunks": all_chunks,
            "entities": merged_entities,
            "relationships": merged_relationships,
            "graph_neighbors": merged_graph_neighbors,
            "traversal_paths": merged_traversal_paths,
            "documents": list(final_docs),
            "total_score": final_total_score,
            "entity_count": len(merged_entities),
            "relationship_count": len(merged_relationships),
            "search_method": "iterative_hop",
            "kg_name": kg_name,
            "retrieval_route": "iterative_hop",
            "route_reason": "success",
            "diagnostics": {
                "rfge_fired": False,
                "retrieval_mode_config": getattr(self, "retrieval_mode", "hybrid_auto"),
                "subquestion_count": len(sub_questions),
                "active_hop_count": len(active_hops),
            },
            # Anchored to hop-0 question-entity matches only, not bridge-induced later hops
            "seed_entity_count": _hop0_seed_count,
            "grounding_quality": _hop0_grounding,
        }

    def generate_response(
        self,
        question: str,
        llm,
        document_names: List[str] = None,
        similarity_threshold: float = None,
        max_chunks: int = None,
        timeout: float = None,
        extra_context_texts: Optional[List[str]] = None,
        kg_name: str = None,
        max_hops: int = 2,
        answer_instructions: str = "",
        runtime_guardrail: Optional[bool] = None,
        runtime_guardrail_mode: Optional[str] = None,
        question_id: str = None,
        allow_decomposition: bool = True,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        retrieval_sample_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate a RAG response using the knowledge graph with adaptive retrieval.

        Args:
            extra_context_texts: Optional list of additional context strings to prepend
                to the retrieved chunks (e.g. ground-truth question contexts for MIRAGE eval).
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            logging.info(f"Starting generate_response for question: {question}")

            # Use adaptive retrieval parameters if not explicitly provided
            if similarity_threshold is None or max_chunks is None:
                retrieval_params = self.get_adaptive_retrieval_params(question)
                similarity_threshold = similarity_threshold or retrieval_params["similarity_threshold"]
                max_chunks = max_chunks or retrieval_params["max_chunks"]

            # IRCoT / StepChain-style iterative multi-hop retrieval when max_hops >= 2.
            # Decompose first so we can skip the redundant full baseline retrieval when
            # the question genuinely decomposes into multiple hops.
            context = None
            if max_hops >= 2 and allow_decomposition:
                sub_questions = self._decompose_question(question, llm, max_hops)
                if len(sub_questions) > 1:
                    graph = self._create_neo4j_connection()
                    mh_ctx = self._iterative_hop_retrieval(
                        graph=graph,
                        question=question,
                        sub_questions=sub_questions,
                        max_chunks=compute_candidate_limit(
                            max_chunks,
                            retrieval_temperature,
                            retrieval_shortlist_factor,
                        ),
                        kg_name=kg_name,
                        max_hops=max_hops,
                        llm=llm,
                        similarity_threshold=similarity_threshold,
                        document_names=document_names or [],
                        question_id=question_id,
                    )
                    if mh_ctx and mh_ctx.get("chunks"):
                        context = mh_ctx
                        logging.info(
                            "Iterative multi-hop retrieval: %d chunks, %d entities, %d rels",
                            len(context["chunks"]), context["entity_count"], context["relationship_count"],
                        )

            # Fall back to the standard single-pass retrieval when:
            #   - max_hops < 2, or
            #   - decomposition returned a single sub-question, or
            #   - iterative retrieval found no chunks.
            if context is None:
                context = self.get_rag_context(
                    question,
                    document_names=document_names,
                    similarity_threshold=similarity_threshold,
                    max_chunks=max_chunks,
                    kg_name=kg_name,
                    max_hops=max_hops,
                    question_id=question_id,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=retrieval_sample_id,
                    llm=llm,
                )

            if self._should_run_query_fusion(question, context, max_hops=max_hops):
                variant_queries = self._generate_query_variants(question, llm, max_hops=max_hops)
                variant_contexts: List[Dict[str, Any]] = [context]
                for variant_query in variant_queries:
                    try:
                        alt_context = self.get_rag_context(
                            variant_query,
                            document_names=document_names,
                            similarity_threshold=similarity_threshold,
                            max_chunks=max_chunks,
                            kg_name=kg_name,
                            max_hops=max_hops,
                            question_id=question_id,
                            retrieval_temperature=retrieval_temperature,
                            retrieval_shortlist_factor=retrieval_shortlist_factor,
                            retrieval_sample_id=retrieval_sample_id,
                            llm=llm,
                        )
                    except Exception as fusion_exc:
                        logging.debug(
                            "Query-fusion retrieval failed for variant %r (non-fatal): %s",
                            variant_query,
                            fusion_exc,
                        )
                        continue
                    if alt_context and alt_context.get("chunks"):
                        variant_contexts.append(alt_context)
                fused_context = self._fuse_contexts_with_rrf(
                    query=question,
                    contexts=variant_contexts,
                    max_chunks=max_chunks,
                    kg_name=kg_name,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=retrieval_sample_id,
                    search_method="query_fusion_" + str(context.get("search_method") or "hybrid"),
                )
                if fused_context and fused_context.get("chunks"):
                    context = fused_context

            if context.get("chunks"):
                try:
                    context = self._append_adjacent_chunks_to_context(
                        self._create_neo4j_connection(),
                        context,
                        kg_name=kg_name,
                        question_id=question_id,
                        document_names=document_names,
                        max_adjacent=max_chunks,
                    )
                except Exception as adj_err:
                    logging.debug("Enhanced adjacent recovery skipped (non-fatal): %s", adj_err)

            li_chunks, li_meta = late_interaction_rescore_chunks_for_query(
                question,
                context.get("chunks", []),
                max_chunks=max_chunks,
            )
            if li_chunks:
                context["chunks"] = li_chunks
            context["late_interaction"] = li_meta

            reranked_chunks, reranker_meta = rerank_chunks_for_query(
                question,
                context.get("chunks", []),
                max_chunks=max_chunks,
            )
            if reranked_chunks:
                context["chunks"] = reranked_chunks
            context["reranker"] = reranker_meta

            logging.info(f"Got context with {context.get('entity_count', 0)} entities")

            if not context["chunks"] and not extra_context_texts:
                return {
                    "response": "I couldn't find any relevant information in the knowledge graph to answer your question.",
                    "context": context,
                    "sources": [],
                    "entities": [],
                    "confidence": 0.0
                }

            # Debug: Check for Version in context
            if "Version" in context.get("entities", {}):
                logging.warning(f"Found Version entity in context: {context['entities']['Version']}")

            response = self._invoke_answer_chain(
                question=question,
                llm=llm,
                context=context,
                answer_instructions=answer_instructions,
                extra_context_texts=extra_context_texts,
                kg_name=kg_name,
            )

            def _retry_with_vector_only():
                if self._is_pure_vector_search_method(context.get("search_method")):
                    return None
                retry_graph = self._create_neo4j_connection()
                retry_context = self._vector_similarity_search(
                    retry_graph,
                    question,
                    document_names or [],
                    similarity_threshold,
                    compute_candidate_limit(
                        max_chunks,
                        retrieval_temperature,
                        retrieval_shortlist_factor,
                    ),
                    kg_name,
                    max_hops=max_hops,
                    question_id=question_id,
                )
                retry_context = self._apply_final_chunk_selection(
                    query=question,
                    context=retry_context,
                    max_chunks=max_chunks,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=retrieval_sample_id,
                    kg_name=kg_name,
                )
                if retry_context.get("chunks"):
                    try:
                        retry_context = self._append_adjacent_chunks_to_context(
                            retry_graph,
                            retry_context,
                            kg_name=kg_name,
                            question_id=question_id,
                            document_names=document_names,
                            max_adjacent=max_chunks,
                        )
                    except Exception as adj_err:
                        logging.debug("Vector-only adjacent recovery skipped (non-fatal): %s", adj_err)
                retry_li_chunks, retry_li_meta = late_interaction_rescore_chunks_for_query(
                    question,
                    retry_context.get("chunks", []),
                    max_chunks=max_chunks,
                )
                if retry_li_chunks:
                    retry_context["chunks"] = retry_li_chunks
                retry_context["late_interaction"] = retry_li_meta
                retry_reranked_chunks, retry_reranker_meta = rerank_chunks_for_query(
                    question,
                    retry_context.get("chunks", []),
                    max_chunks=max_chunks,
                )
                if retry_reranked_chunks:
                    retry_context["chunks"] = retry_reranked_chunks
                retry_context["reranker"] = retry_reranker_meta
                if not retry_context.get("chunks") and not extra_context_texts:
                    return None
                retry_response = self._invoke_answer_chain(
                    question=question,
                    llm=llm,
                    context=retry_context,
                    answer_instructions=answer_instructions,
                    extra_context_texts=extra_context_texts,
                    kg_name=kg_name,
                )
                return retry_response, retry_context

            response, context, guardrail = self._apply_runtime_answer_guardrail(
                question=question,
                llm=llm,
                response=response,
                context=context,
                runtime_guardrail=runtime_guardrail,
                runtime_guardrail_mode=runtime_guardrail_mode,
                retry_factory=_retry_with_vector_only,
            )

            # If entity-first retrieval produced an "Insufficient Information" response,
            # retry with pure vector search as a floor — this ensures KG-RAG is never
            # worse than vanilla RAG on questions where graph anchoring fails.
            if (getattr(self, "allow_vector_fallback", True)
                    and "insufficient information" in response.lower()
                    and not self._is_pure_vector_search_method(context.get("search_method"))):
                logging.info("Insufficient-info response from graph-first path; retrying with vector-only")
                _vec_retry = _retry_with_vector_only()
                if _vec_retry:
                    _retry_response, _retry_context = _vec_retry
                    if "insufficient information" not in _retry_response.lower():
                        response = _retry_response
                        context = _retry_context
                        guardrail["retried_insufficient_info"] = True

            # Extract entities and chunks actually mentioned in the response, plus reasoning edges
            extracted_info = self._extract_used_entities_and_chunks(response, context)
            used_entities = extracted_info["used_entities"]
            used_chunks = extracted_info["used_chunks"]
            reasoning_edges = extracted_info["reasoning_edges"]

            # Calculate confidence based on similarity scores
            avg_score = context["total_score"] / len(context["chunks"]) if context["chunks"] else 0
            confidence = max(0.0, min((avg_score - 0.05) / 0.95, 1.0))  # Scale similarity score to 0-1
            # Penalize when multiple entities were retrieved but no graph relationships connected them
            # — this suggests the KG graph path wasn't useful, answer relies on text chunks only
            if context.get("relationship_count", 0) == 0 and context.get("entity_count", 0) > 1:
                confidence = min(confidence, 0.6)
            if guardrail.get("enabled") and guardrail.get("final_decision") not in {"keep", "retry_keep"}:
                confidence = 0.0

            return {
                "response": response,
                "context": context,
                "sources": context["documents"],
                "entities": list(context["entities"].keys()),
                "used_entities": used_entities,  # Entities actually used in the answer
                "used_chunks": used_chunks,  # Chunks actually used in the answer
                "reasoning_edges": reasoning_edges,  # Edges that form the reasoning path
                "relationships": context["relationships"],
                "confidence": confidence,
                "chunk_count": len(context["chunks"]),
                "entity_count": context["entity_count"],
                "relationship_count": context["relationship_count"],
                "guardrail": guardrail,
                "retrieval_params": {
                    "question_type": self.classify_question_type(question),
                    "similarity_threshold": similarity_threshold,
                    "max_chunks": max_chunks,
                    "timeout": timeout,
                    "retrieval_temperature": float(retrieval_temperature or 0.0),
                    "retrieval_shortlist_factor": int(retrieval_shortlist_factor or 1),
                    "retrieval_sample_id": int(retrieval_sample_id or 0),
                },
                "late_interaction_stage": context.get("late_interaction_stage", {}),
                "late_interaction": context.get("late_interaction", {}),
                "reranker": context.get("reranker", {}),
            }
            
        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            return {
                "response": f"An error occurred while generating the response: {str(e)}",
                "context": {},
                "sources": [],
                "entities": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific entity
        """
        try:
            graph = self._create_neo4j_connection()
            
            query = """
            MATCH (e:__Entity__ {id: $entity_id})
            OPTIONAL MATCH (e)-[r]-(related:__Entity__)
            OPTIONAL MATCH (e)<-[:HAS_ENTITY|MENTIONS]-(c:Chunk)-[:PART_OF]->(d:Document)
            RETURN 
                e.id AS id,
                e.type AS type,
                e.description AS description,
                elementId(e) AS element_id,
                collect(DISTINCT {
                    related_id: related.id,
                    related_type: related.type,
                    relationship_type: type(r),
                    relationship_element_id: elementId(r)
                }) AS relationships,
                collect(DISTINCT {
                    chunk_id: c.id,
                    document: d.fileName
                }) AS mentions
            """
            
            results = graph.query(query, {"entity_id": entity_id})
            
            if not results:
                return {"error": f"Entity {entity_id} not found"}
            
            result = results[0]
            return {
                "id": result["id"],
                "type": result["type"],
                "description": result["description"],
                "element_id": result["element_id"],
                "relationships": result["relationships"],
                "mentions": result["mentions"]
            }
            
        except Exception as e:
            logging.error(f"Error getting entity details: {e}")
            return {"error": str(e)}

    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        """
        try:
            graph = self._create_neo4j_connection()
            
            stats_query = """
            MATCH (d:Document)
            OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            OPTIONAL MATCH (e)-[r]-(e2:__Entity__)
            RETURN 
                count(DISTINCT d) AS documents,
                count(DISTINCT c) AS chunks,
                count(DISTINCT e) AS entities,
                count(DISTINCT r) AS relationships,
                collect(DISTINCT d.fileName) AS document_names
            """
            
            results = graph.query(stats_query)
            
            if results:
                result = results[0]
                return {
                    "documents": result["documents"],
                    "chunks": result["chunks"],
                    "entities": result.get("entities") or [],
                    "relationships": result["relationships"],
                    "document_names": result["document_names"],
                    "has_embeddings": False
                }
            else:
                return {
                    "documents": 0,
                    "chunks": 0,
                    "entities": 0,
                    "relationships": 0,
                    "document_names": [],
                    "has_embeddings": False
                }
                
        except Exception as e:
            logging.error(f"Error getting KG stats: {e}")
            return {"error": str(e)}

    def search_entities(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities using text-based matching
        """
        try:
            graph = self._create_neo4j_connection()

            search_query = """
            MATCH (e:__Entity__)
            WHERE toLower(e.name) CONTAINS toLower($query)
            OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(c:Chunk)-[:PART_OF]->(d:Document)
            RETURN
                e.id AS id,
                e.type AS type,
                e.name AS name,
                elementId(e) AS element_id,
                count(DISTINCT c) AS chunk_mentions,
                collect(DISTINCT d.fileName) AS documents
            ORDER BY chunk_mentions DESC
            LIMIT $top_k
            """

            results = graph.query(search_query, {
                "query": query,
                "top_k": top_k
            })

            return [
                {
                    "id": result["id"],
                    "element_id": result["element_id"],
                    "type": result["type"],
                    "description": result["name"],
                    "score": result["chunk_mentions"] * 0.1,  # Simple scoring based on mentions
                    "chunk_mentions": result["chunk_mentions"],
                    "documents": result["documents"]
                }
                for result in results
            ]

        except Exception as e:
            logging.error(f"Error searching entities: {e}")
            return []

    def _vector_similarity_search(self, graph, query: str, document_names: List[str] = None, similarity_threshold: float = 0.08, max_chunks: int = 20, kg_name: str = None, max_hops: int = 2, question_id: str = None) -> Dict[str, Any]:
        """
        Vector similarity search using real sentence transformer embeddings.
        
        Args:
            graph: Neo4j graph connection
            query: The search query
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            logging.info(f"Using vector similarity search with real embeddings (kg_name: {kg_name})")

            query_embedding = self._generate_query_embedding(query)
            logging.info(f"Generated query embedding with shape: {len(query_embedding)}")
            if not self.check_vector_index():
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "vector_similarity_unavailable",
                }
            vector_index_name = self._active_vector_index_name or "vector"

            # Build the vector search query with optional filtering by kg_name and/or document_names
            # Relationship is: Chunk -[PART_OF]-> Document
            # We need to filter by either document_names or kg_name (or both)
            
            # Build one WHERE clause that always includes score threshold
            where_conditions = ["score >= $similarity_threshold"]
            params = {
                "query_vector": query_embedding,
                "similarity_threshold": similarity_threshold
            }
            
            if kg_name:
                where_conditions.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name
                
            if document_names:
                where_conditions.append("d.fileName IN $document_names")
                params["document_names"] = document_names

            if question_id:
                where_conditions.append("chunk.questionId = $question_id")
                params["question_id"] = question_id

            where_clause = "WHERE " + " AND ".join(where_conditions)

            # When filtering by kg_name or document_names the vector index is
            # queried globally first, then filtered.  Overfetch so that the
            # target KG's chunks are not accidentally excluded because they
            # didn't rank in the global top-N.
            # Overfetch so KG-filtered chunks aren't excluded by global top-N,
            # but cap at 500 to avoid timeouts on large statistical queries.
            retrieval_count = min(max_chunks * 20, 500) if (kg_name or document_names or question_id) else max_chunks

            if vector_index_name == "retrieval_vector":
                if question_id:
                    where_clause = where_clause.replace(
                        "chunk.questionId = $question_id",
                        "coalesce(retrieval.questionId, chunk.questionId) = $question_id",
                    )
                search_query = f"""
                CALL db.index.vector.queryNodes('{vector_index_name}', {retrieval_count}, $query_vector)
                YIELD node AS retrieval, score
                MATCH (retrieval:RetrievalChunk)-[:RETRIEVES_FROM]->(chunk:Chunk)-[:PART_OF]->(d:Document)
                {where_clause}
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH retrieval, chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
                RETURN
                    retrieval.text AS text,
                    retrieval.id AS chunk_id,
                    elementId(retrieval) AS chunk_element_id,
                    chunk.id AS parent_chunk_id,
                    elementId(chunk) AS parent_chunk_element_id,
                    coalesce(retrieval.questionId, chunk.questionId) AS question_id,
                    coalesce(retrieval.passageIndex, chunk.passageIndex) AS passage_index,
                    coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex) AS chunk_local_index,
                    retrieval.retrievalLocalIndex AS retrieval_local_index,
                    chunk.position AS position,
                    chunk.source AS source,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                        id: coalesce(entity.id, entity.name),
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Entity'),
                        description: coalesce(entity.name, '')
                    }}] AS entities
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            else:
                search_query = f"""
                CALL db.index.vector.queryNodes('{vector_index_name}', {retrieval_count}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                {where_clause}
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    chunk.questionId AS question_id,
                    chunk.passageIndex AS passage_index,
                    chunk.chunkLocalIndex AS chunk_local_index,
                    chunk.position AS position,
                    chunk.source AS source,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                        id: coalesce(entity.id, entity.name),
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Entity'),
                        description: coalesce(entity.name, '')
                    }}] AS entities
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            params["max_chunks"] = max_chunks

            results = graph.query(search_query, params)
            if vector_index_name == "retrieval_vector" and question_id and not results:
                logging.info(
                    "No question-scoped retrieval spans found for kg=%s question_id=%s; "
                    "falling back to parent Chunk vector index.",
                    kg_name,
                    question_id,
                )
                fallback_where_clause = where_clause.replace(
                    "coalesce(retrieval.questionId, chunk.questionId) = $question_id",
                    "chunk.questionId = $question_id",
                )
                fallback_query = f"""
                CALL db.index.vector.queryNodes('vector', {retrieval_count}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                {fallback_where_clause}
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    chunk.questionId AS question_id,
                    chunk.passageIndex AS passage_index,
                    chunk.chunkLocalIndex AS chunk_local_index,
                    chunk.position AS position,
                    chunk.source AS source,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                        id: coalesce(entity.id, entity.name),
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Entity'),
                        description: coalesce(entity.name, '')
                    }}] AS entities
                ORDER BY score DESC
                LIMIT $max_chunks
                """
                try:
                    fallback_results = graph.query(fallback_query, params) or []
                except Exception as fallback_exc:
                    logging.warning(
                        "Chunk-vector fallback failed for kg=%s question_id=%s: %s",
                        kg_name,
                        question_id,
                        fallback_exc,
                    )
                else:
                    if fallback_results:
                        results = fallback_results
                        vector_index_name = "vector"

            context = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": set(),
                "total_score": 0,
                "search_method": (
                    "retrieval_span_similarity"
                    if vector_index_name == "retrieval_vector"
                    else "vector_similarity"
                )
            }

            for result in results:
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "parent_chunk_id": result.get("parent_chunk_id"),
                    "parent_chunk_element_id": result.get("parent_chunk_element_id"),
                    "question_id": result.get("question_id"),
                    "passage_index": result.get("passage_index"),
                    "chunk_local_index": result.get("chunk_local_index"),
                    "retrieval_local_index": result.get("retrieval_local_index"),
                    "position": result.get("position"),
                    "source": result.get("source"),
                    "score": result["score"],
                    "document": result["document"],
                    "kg_name": result.get("kg_name"),
                    "entities": result.get("entities") or [],
                    "linked_entity_ids": [
                        entity.get("id")
                        for entity in (result.get("entities") or [])
                        if entity.get("id")
                    ],
                    "linked_entity_count": len(result.get("entities") or []),
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]

                # Collect unique entities
                for entity in (result.get("entities") or []):
                    entity_id = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": []
                        }
                    if result.get("chunk_id"):
                        context["entities"][entity_id]["mentioned_in_chunks"].append(result["chunk_id"])

            # Multi-hop graph traversal from seed entities found in chunks
            seed_ids = list(context["entities"].keys())
            expansion = self._expand_entities_via_graph(
                graph,
                seed_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
            )
            context["graph_neighbors"] = expansion["neighbors"]
            context["traversal_paths"] = expansion["paths"]

            # Merge neighbor entities into the entity dict
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in context["entities"]:
                    context["entities"][nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Relationships between ALL entities (seeds + neighbors)
            all_entity_ids = list(context["entities"].keys())
            if all_entity_ids:
                relationship_results = self._fetch_relationships_for_entity_ids(
                    graph,
                    all_entity_ids,
                    kg_name=kg_name,
                    question_id=question_id,
                    document_names=document_names,
                )
                for rel_result in relationship_results:
                    rel_key = (
                        rel_result.get("relationship_element_id")
                        or f"{rel_result['source']}-{rel_result['relationship_type']}-{rel_result['target']}"
                        f"-negated={bool(rel_result.get('negated', False))}"
                    )
                    if not any(r.get('key') == rel_key for r in context["relationships"]):
                        context["relationships"].append({
                            "key": rel_key,
                            "source": rel_result["source"],
                            "source_element_id": rel_result["source_element_id"],
                            "target": rel_result["target"],
                            "target_element_id": rel_result["target_element_id"],
                            "type": rel_result["relationship_type"],
                            "element_id": rel_result["relationship_element_id"],
                            "negated": bool(rel_result.get("negated", False)),
                            "condition": rel_result.get("condition"),
                            "quantitative": rel_result.get("quantitative"),
                            "confidence": rel_result.get("confidence"),
                            "question_ids": rel_result.get("question_ids") or [],
                            "passage_keys": rel_result.get("passage_keys") or [],
                            "provenance_positions": rel_result.get("provenance_positions") or [],
                        })

            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])

            return context

        except Exception as e:
            logging.error(f"Error in vector similarity search: {e}")
            # Don't return error — let caller fall back to text search
            raise Exception(f"Vector search failed: {str(e)}")

    def _extract_used_entities_and_chunks(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and chunks that are actually mentioned in the RAG answer,
        using multiple fallback strategies to ensure filtering works
        """
        used_entities = []
        used_chunks = []
        reasoning_edges = []

        try:
            context_entities = context.get("entities", {})
            context_chunks = context.get("chunks", [])
            context_relationships = context.get("relationships", [])

            response_lower = response.lower()
            mentioned_entity_names = set()
            mentioned_chunk_ids = set()

            # Match entities by name: use description (= n.name, human-readable) as primary
            # key, falling back to id only if description is absent.
            # entity_info["id"] is a UUID-prefixed key, not a readable name — don't use it
            # for text matching against the LLM response.
            for entity_key, entity_info in context_entities.items():
                # description = coalesce(entity.name, '') from Cypher — the readable name
                entity_name = (entity_info.get("description") or entity_info.get("id") or "").lower().replace("_", " ").strip()

                if entity_name and len(entity_name) > 2:
                    if entity_name in response_lower:
                        mentioned_entity_names.add(entity_key)
                        continue
                    # Prefix match (handles abbreviations like "BRCA" matching "BRCA2")
                    if len(entity_name) > 4 and any(
                        word.startswith(entity_name[:4]) for word in response_lower.split()
                    ):
                        mentioned_entity_names.add(entity_key)
                        continue

                # Also match on individual significant words of the entity name
                for word in entity_name.split():
                    if len(word) > 4 and word in response_lower:
                        mentioned_entity_names.add(entity_key)
                        break

            # Map chunk ordinal references ("chunk 2") back to actual chunk IDs
            for chunk_num in re.findall(r'\bchunk\s*(\d+)\b', response, re.IGNORECASE):
                try:
                    chunk_index = int(chunk_num) - 1
                    if 0 <= chunk_index < len(context_chunks):
                        matched_chunk = context_chunks[chunk_index]
                        if matched_chunk.get("chunk_id"):
                            mentioned_chunk_ids.add(matched_chunk["chunk_id"])
                        if matched_chunk.get("chunk_element_id"):
                            mentioned_chunk_ids.add(matched_chunk["chunk_element_id"])
                except (ValueError, TypeError):
                    continue

            logging.info(f"Name-based entity matches: {len(mentioned_entity_names)}")

            for entity_id, entity_info in context_entities.items():
                if entity_id in mentioned_entity_names:
                    used_entities.append({
                        "id": entity_id,
                        "element_id": entity_info.get("element_id", ""),
                        "type": entity_info.get("type", "Unknown"),
                        "description": entity_info.get("description", ""),
                        "reasoning_context": "mentioned by name"
                    })

            # Strategy 4: Include chunks that contain mentioned entities (semantic linking)
            relevant_chunk_ids = set()
            for entity_id in {e['id'] for e in used_entities}:
                for chunk in context_chunks:
                    chunk_entities = chunk.get('entities', [])
                    if any(ce.get('id') == entity_id for ce in chunk_entities):
                        relevant_chunk_ids.add(chunk.get('chunk_id'))
                        relevant_chunk_ids.add(chunk.get('chunk_element_id'))

            # Include explicitly mentioned chunks
            for chunk in context_chunks:
                chunk_id = chunk.get("chunk_id", "")
                chunk_element_id = chunk.get("chunk_element_id", "")
                is_direct = chunk_id in mentioned_chunk_ids or chunk_element_id in mentioned_chunk_ids
                has_relevant_entity = chunk_id in relevant_chunk_ids or chunk_element_id in relevant_chunk_ids
                if is_direct or has_relevant_entity:
                    text = chunk.get("text", "")
                    used_chunks.append({
                        "id": chunk_id,
                        "element_id": chunk_element_id,
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "reasoning_context": "directly referenced" if is_direct else "contains relevant entities"
                    })

            # Strategy 5: Find relationships between selected entities
            if used_entities:  # Only look for relationships if we have filtered entities
                # Build lookup: element_id or id → human-readable name
                entity_name_lookup = {}
                for e in used_entities:
                    name = e.get("description") or e.get("id", "")
                    if e.get("element_id"):
                        entity_name_lookup[e["element_id"]] = name
                    if e.get("id"):
                        entity_name_lookup[e["id"]] = name

                used_entity_element_ids = {e['element_id'] for e in used_entities if e['element_id']}
                used_entity_ids = {e['id'] for e in used_entities}

                for rel in context_relationships:
                    source_id = rel.get("source", "")
                    target_id = rel.get("target", "")
                    source_element_id = rel.get("source_element_id", "")
                    target_element_id = rel.get("target_element_id", "")

                    # Include edge if both connected entities are in our filtered set
                    source_in_set = (source_id in used_entity_ids or
                                   source_element_id in used_entity_element_ids)
                    target_in_set = (target_id in used_entity_ids or
                                   target_element_id in used_entity_element_ids)

                    if source_in_set and target_in_set:
                        src_key = source_element_id or source_id
                        tgt_key = target_element_id or target_id
                        reasoning_edges.append({
                            "from": src_key,
                            "to": tgt_key,
                            "from_name": entity_name_lookup.get(src_key) or entity_name_lookup.get(source_id) or src_key,
                            "to_name": entity_name_lookup.get(tgt_key) or entity_name_lookup.get(target_id) or tgt_key,
                            "relationship": rel.get("type", "CONNECTED_TO"),
                            "reasoning_context": "connects relevant entities"
                        })

            logging.info(f"Filtered to {len(used_entities)} entities ({len([e for e in used_entities if 'mentioned by name' in e['reasoning_context']])} by name) and {len(used_chunks)} chunks")
            logging.info(f"Found {len(reasoning_edges)} reasoning edges")

            return {
                "used_entities": used_entities,
                "used_chunks": used_chunks,
                "reasoning_edges": reasoning_edges,
                "total_filtered_entities": len(used_entities),
                "total_filtered_chunks": len(used_chunks),
                "total_reasoning_edges": len(reasoning_edges)
            }

        except Exception as e:
            logging.error(f"Error extracting used entities and chunks: {e}")
            return {
                "used_entities": [],
                "used_chunks": [],
                "reasoning_edges": [],
                "error": str(e)
            }
