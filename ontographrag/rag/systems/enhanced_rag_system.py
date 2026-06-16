import os
import logging
import re
from typing import Dict, Any, List, Optional, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from ontographrag.kg.utils.common_functions import load_embedding_model
from ontographrag.rag.answer_guardrails import (
    RUNTIME_GUARDRAIL_ABSTENTION,
    evaluate_runtime_answer_guardrail,
)
from ontographrag.rag.graph_state import summarize_context_graph_state
from ontographrag.rag.retrieval_sampling import compute_candidate_limit
from ontographrag.rag.reranking import (
    late_interaction_rescore_chunks_for_query,
    rerank_chunks_for_query,
)

from ontographrag.rag.systems._constants import RAG_CONFIG  # noqa: F401  (re-exported)
from ontographrag.rag.systems._context_assembly import ContextAssemblyMixin
from ontographrag.rag.systems._graph_search import GraphSearchMixin
from ontographrag.rag.systems._iterative import IterativeRetrievalMixin
from ontographrag.rag.systems._query_planning import QueryPlanningMixin
from ontographrag.rag.systems._vector_search import VectorSearchMixin

class EnhancedRAGSystem(
    ContextAssemblyMixin,
    GraphSearchMixin,
    IterativeRetrievalMixin,
    QueryPlanningMixin,
    VectorSearchMixin,
):
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
    _PASSAGE_ONLY_ANSWER_KGS = frozenset({
        "pubmedqa",
        "realmedqa",
    })
    _QUERY_FUSION_MAX_VARIANTS = 3
    _QUERY_FUSION_RRF_K = 60
    RETRIEVAL_MODES = frozenset({"hybrid_auto", "entity_first", "rfge", "vector_only"})
    _STRUCTURAL_REL_TYPES: frozenset = frozenset({
        "HAS_ENTITY", "FROM_CHUNK", "PART_OF", "MENTIONED_IN", "MENTIONS", "QUALIFIES",
    })
    _HOP_SCORE = {0: 1.0, 1: 0.8, 2: 0.6}
    _HOP_SCORE_DEFAULT = 0.5  # for hops > 2
    _MAX_CHUNKS_PER_PASSAGE: int = 2
    _TRAVERSAL_CHUNK_MIN_SIM: float = 0.30
    _PPR_ALPHA = 0.85
    _PPR_STEPS = 5
    _RFGE_VECTOR_THRESHOLD = 0.20
    _RFGE_MIN_PPR_SCORE = 0.005

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
        anchor_mask_entity_ids: Optional[List[str]] = None,
        anchor_mask_entity_names: Optional[List[str]] = None,
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
                    anchor_mask_entity_ids=anchor_mask_entity_ids,
                    anchor_mask_entity_names=anchor_mask_entity_names,
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
                    anchor_mask_entity_ids=anchor_mask_entity_ids,
                    anchor_mask_entity_names=anchor_mask_entity_names,
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
        anchor_mask_entity_ids: Optional[List[str]] = None,
        anchor_mask_entity_names: Optional[List[str]] = None,
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
                        anchor_mask_entity_ids=anchor_mask_entity_ids,
                        anchor_mask_entity_names=anchor_mask_entity_names,
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
                    anchor_mask_entity_ids=anchor_mask_entity_ids,
                    anchor_mask_entity_names=anchor_mask_entity_names,
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
                            anchor_mask_entity_ids=anchor_mask_entity_ids,
                            anchor_mask_entity_names=anchor_mask_entity_names,
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
            context["graph_state"] = summarize_context_graph_state(context)

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
