import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Set
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

class VanillaRAGSystem:
    """
    Vanilla RAG system that retrieves chunks directly using vector similarity search
    without knowledge graph augmentation
    """
    _QUERY_STOPWORDS: Set[str] = {
        "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
        "does", "do", "did", "done", "have", "has", "had", "with", "from",
        "that", "this", "these", "those", "into", "about", "after", "before",
        "through", "there", "their", "them", "they", "would", "could", "should",
        "question", "answer", "answers",
    }
    _QUERY_FUSION_MAX_VARIANTS = 3
    _QUERY_FUSION_RRF_K = 60

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "neo4j",
        embedding_model: str = None  # Use environment variable if not provided
    ):
        # Load Neo4j credentials from environment variables if not provided
        self.neo4j_uri = neo4j_uri if neo4j_uri is not None else os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user if neo4j_user is not None else os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password if neo4j_password is not None else os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.neo4j_uri or not self.neo4j_user or not self.neo4j_password:
            raise ValueError("Neo4j connection parameters not found. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")

        # Initialize embeddings through the shared loader so KG construction and
        # vanilla retrieval use the same backend/model defaults.
        embedding_provider = (
            embedding_model
            or os.getenv("EMBEDDING_PROVIDER")
            or os.getenv("EMBEDDING_MODEL", "sentence_transformers")
        )
        self.embedding_model, self.embedding_dimension = load_embedding_model(embedding_provider)
        logging.info(
            "Initialized vanilla retrieval embedding backend '%s' with dimension %d",
            embedding_provider,
            self.embedding_dimension,
        )
        self._vector_index_name: Optional[str] = None
        self._late_interaction_corpus_cache: dict = {}

        # Simple RAG prompt template (no KG traversal instructions)
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that provides accurate, factual answers based on the provided context.

Context Information:
{context}

Guidelines:
- Follow the task-specific answer instructions below when they are provided.
- If the task-specific instructions require an exact label-only answer, obey them exactly and do not add explanation.
- If the task-specific instructions require a short answer only, give only that short answer and do not add explanation.
- Read all context passages carefully — the answer is often present but may require connecting two passages.
- For multi-hop questions: explicitly chain your reasoning step by step (e.g. "The film starred X → X later held position Y").
- Base your answer on the provided context; do not invent facts.
- If the answer is not directly stated but can be inferred by connecting two pieces of evidence, make the inference explicitly and state your reasoning chain.
- Only say the context is insufficient if you genuinely cannot find any relevant evidence after carefully reading all passages.
- Be concise but comprehensive; include specific facts to support your answer.
- For source-document biomedical classification tasks, let the study conclusion in the text govern the final label.
- IMPORTANT: Unless task-specific instructions say otherwise, for yes/no questions
  (questions starting with: Is, Are, Does, Do, Can, Should, Was, Were, Has, Have),
  begin your response with either "Yes" or "No" as the very first word.

Task-Specific Answer Instructions:
{answer_instructions}

User Query: {question}"""),
            ("human", "{question}")
        ])

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    def clear_retrieval_caches(self) -> None:
        """Drop cached retrieval state after KG mutations."""
        self._vector_index_name = None
        self._late_interaction_corpus_cache.clear()

    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query across supported embedding backends."""
        if hasattr(self.embedding_model, "embed_query"):
            return self.embedding_model.embed_query(query)

        if hasattr(self.embedding_model, "encode"):
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        raise ValueError("Unsupported embedding model interface. Expected embed_query or encode.")

    def _resolve_vector_index_name(self, graph) -> Optional[str]:
        """Prefer retrieval-span vectors when available, else fall back to Chunk vectors."""
        if self._vector_index_name is not None:
            return self._vector_index_name

        try:
            index_rows = graph.query(
                "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR' RETURN name, state"
            ) or []
        except Exception as exc:
            logging.warning("Could not inspect Neo4j vector indexes: %s", exc)
            return None

        online_indexes = {
            str(row.get("name")): str(row.get("state") or "").upper()
            for row in index_rows
            if str(row.get("state") or "").upper() in {"", "ONLINE"}
        }
        probe_embedding = self._generate_query_embedding("test")
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
                self._vector_index_name = candidate
                return candidate
            except Exception as exc:
                logging.warning("Vector index %s probe failed: %s", candidate, exc)
        return None

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
    ) -> tuple:
        return (
            kg_name or "",
            tuple(sorted(document_names or [])),
            question_id or "",
        )

    @staticmethod
    def _normalize_retrieval_query(query: str) -> str:
        return re.sub(r"\s+", " ", str(query or "")).strip()

    @classmethod
    def _content_query_tokens(cls, query: str) -> List[str]:
        seen: Set[str] = set()
        tokens: List[str] = []
        for token in re.findall(r"[A-Za-z0-9]+", str(query or "").lower()):
            if len(token) < 4 or token in cls._QUERY_STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        return tokens

    @classmethod
    def _comparison_branches(cls, query: str) -> List[str]:
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
    def _lexical_query_overlap_count(cls, query: str, chunk: Dict[str, Any]) -> int:
        tokens = cls._content_query_tokens(query)
        if not tokens:
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
        return sum(1 for token in tokens if token in haystack)

    @classmethod
    def _sort_chunks_for_query(cls, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        branches = cls._comparison_branches(query)
        return sorted(
            chunks,
            key=lambda chunk: (
                -cls._comparison_branch_match_count(query, chunk) if branches else 0,
                -cls._lexical_query_overlap_count(query, chunk),
                -float(chunk.get("score", 0.0)),
                1 if chunk.get("adjacent") else 0,
                int(chunk.get("position") or 0),
            ),
        )

    @classmethod
    def _query_fusion_enabled(cls) -> bool:
        return str(os.getenv("ONTOGRAPHRAG_QUERY_FUSION", "1")).strip().lower() not in {
            "0", "false", "off", "no",
        }

    @classmethod
    def _should_run_query_fusion(cls, question: str, context: Optional[Dict[str, Any]]) -> bool:
        if not cls._query_fusion_enabled():
            return False
        branches = cls._comparison_branches(question)
        if branches:
            coverage = cls._comparison_branch_coverage(question, list((context or {}).get("chunks", [])))
            if coverage < len(branches):
                return True
        content_tokens = cls._content_query_tokens(question)
        return len(content_tokens) >= 8 and len(list((context or {}).get("chunks", []))) < 4

    def _generate_query_variants(self, question: str, llm) -> List[str]:
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

        is_complex = bool(self._comparison_branches(question)) or len(self._content_query_tokens(question)) >= 8
        if not is_complex:
            return variants[: self._QUERY_FUSION_MAX_VARIANTS]

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are optimizing retrieval queries for vanilla RAG. Produce up to {n} alternative "
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
            logging.debug("Vanilla query fusion reformulation failed (non-fatal): %s", exc)

        return variants[: self._QUERY_FUSION_MAX_VARIANTS]

    @classmethod
    def _fuse_contexts_with_rrf(
        cls,
        *,
        query: str,
        contexts: List[Dict[str, Any]],
        max_chunks: int,
        search_method: str,
    ) -> Optional[Dict[str, Any]]:
        usable_contexts = [ctx for ctx in contexts if ctx and ctx.get("chunks")]
        if not usable_contexts:
            return None
        if len(usable_contexts) == 1:
            return usable_contexts[0]

        chunk_records: Dict[str, Dict[str, Any]] = {}
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

        fused_chunks = cls._sort_chunks_for_query(query, fused_chunks)[:max_chunks]
        return {
            "query": query,
            "chunks": fused_chunks,
            "documents": list({
                chunk.get("document")
                for chunk in fused_chunks
                if chunk.get("document")
            }),
            "total_score": float(sum(float(chunk.get("score", 0.0)) for chunk in fused_chunks)),
            "search_method": search_method,
        }

    def _late_interaction_corpus_rows(
        self,
        graph,
        *,
        kg_name: Optional[str],
        document_names: Optional[List[str]],
        question_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        cache = getattr(self, "_late_interaction_corpus_cache", None)
        if cache is None:
            cache = {}
            self._late_interaction_corpus_cache = cache

        scope_key = self._late_interaction_scope_key(
            kg_name=kg_name,
            document_names=document_names,
            question_id=question_id,
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

        retrieval_query = f"""
        MATCH (retrieval:RetrievalChunk)-[:RETRIEVES_FROM]->(chunk:Chunk)-[:PART_OF]->(d:Document)
        {where_clause}
        RETURN
            retrieval.text AS text,
            retrieval.id AS chunk_id,
            elementId(retrieval) AS chunk_element_id,
            chunk.id AS parent_chunk_id,
            elementId(chunk) AS parent_chunk_element_id,
            chunk.position AS position,
            chunk.source AS source,
            coalesce(retrieval.questionId, chunk.questionId) AS question_id,
            coalesce(retrieval.passageIndex, chunk.passageIndex) AS passage_index,
            coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex) AS chunk_local_index,
            retrieval.retrievalLocalIndex AS retrieval_local_index,
            d.fileName AS document,
            d.kgName AS kg_name
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
            chunk_query = f"""
            MATCH (chunk:Chunk)-[:PART_OF]->(d:Document)
            {chunk_where}
            RETURN
                chunk.text AS text,
                chunk.id AS chunk_id,
                elementId(chunk) AS chunk_element_id,
                chunk.position AS position,
                chunk.source AS source,
                chunk.questionId AS question_id,
                chunk.passageIndex AS passage_index,
                chunk.chunkLocalIndex AS chunk_local_index,
                d.fileName AS document,
                d.kgName AS kg_name
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
        question_id: str = None,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        retrieval_sample_id: int = 0,
    ) -> Dict[str, Any]:
        scope_key = self._late_interaction_scope_key(
            kg_name=kg_name,
            document_names=document_names,
            question_id=question_id,
        )
        rows = self._late_interaction_corpus_rows(
            graph,
            kg_name=kg_name,
            document_names=document_names,
            question_id=question_id,
        )
        if not rows:
            return {
                "query": query,
                "chunks": [],
                "documents": [],
                "total_score": 0,
                "search_method": "vanilla_late_interaction_unavailable",
            }

        candidate_limit = compute_candidate_limit(
            max_chunks,
            retrieval_temperature,
            retrieval_shortlist_factor,
            hard_cap=500,
        )
        candidate_rows = []
        for row in rows:
            candidate_rows.append({
                "text": row["text"],
                "chunk_id": row["chunk_id"],
                "chunk_element_id": row["chunk_element_id"],
                "position": row.get("position"),
                "source": row.get("source"),
                "question_id": row.get("question_id"),
                "passage_index": row.get("passage_index"),
                "chunk_local_index": row.get("chunk_local_index"),
                "retrieval_local_index": row.get("retrieval_local_index"),
                "parent_chunk_id": row.get("parent_chunk_id"),
                "parent_chunk_element_id": row.get("parent_chunk_element_id"),
                "score": 0.0,
                "document": row["document"],
                "kg_name": row.get("kg_name"),
            })
        reranked_rows, li_meta = late_interaction_rescore_chunks_for_query(
            query,
            candidate_rows,
            max_chunks=candidate_limit,
            replace_score=True,
            index_key=scope_key,
        )
        if not li_meta.get("applied"):
            return {
                "query": query,
                "chunks": [],
                "documents": [],
                "total_score": 0.0,
                "search_method": "vanilla_late_interaction_unavailable",
                "late_interaction_stage": li_meta,
            }

        score_values = [
            abs(float(row.get("late_interaction_score", row.get("score", 0.0)) or 0.0))
            for row in reranked_rows[:candidate_limit]
        ]
        if not reranked_rows or max(score_values, default=0.0) <= 1e-9:
            return {
                "query": query,
                "chunks": [],
                "documents": [],
                "total_score": 0.0,
                "search_method": "vanilla_late_interaction_unavailable",
                "late_interaction_stage": {
                    **li_meta,
                    "applied": False,
                    "reason": "no_score_signal",
                },
            }
        selected_results = select_ranked_subset(
            reranked_rows,
            max_items=max_chunks,
            retrieval_temperature=retrieval_temperature,
            shortlist_factor=retrieval_shortlist_factor,
            sample_id=retrieval_sample_id,
            seed_parts=(
                "vanilla_late_interaction",
                kg_name,
                tuple(document_names or []),
                query,
            ),
            score_getter=lambda row: float(row.get("score", 0.0)),
        )

        context = {
            "query": query,
            "chunks": [],
            "documents": set(),
            "total_score": 0.0,
            "search_method": "vanilla_late_interaction_similarity",
            "retrieval_sampling": {
                "temperature": float(retrieval_temperature or 0.0),
                "shortlist_factor": int(retrieval_shortlist_factor or 1),
                "sample_id": int(retrieval_sample_id or 0),
                "candidate_limit": int(candidate_limit),
            },
            "late_interaction_stage": li_meta,
        }

        seen_ids = set()
        for result in selected_results:
            chunk_info = {
                "text": result["text"],
                "chunk_id": result["chunk_id"],
                "chunk_element_id": result["chunk_element_id"],
                "position": result.get("position"),
                "source": result.get("source"),
                "question_id": result.get("question_id"),
                "passage_index": result.get("passage_index"),
                "chunk_local_index": result.get("chunk_local_index"),
                "retrieval_local_index": result.get("retrieval_local_index"),
                "parent_chunk_id": result.get("parent_chunk_id"),
                "parent_chunk_element_id": result.get("parent_chunk_element_id"),
                "score": result["score"],
                "document": result["document"],
                "kg_name": result.get("kg_name"),
                "late_interaction_score": result.get("late_interaction_score"),
            }
            context["chunks"].append(chunk_info)
            context["documents"].add(result["document"])
            context["total_score"] += float(result["score"])
            seen_ids.add(result["chunk_element_id"])

        if selected_results:
            seed_element_ids = [
                r.get("parent_chunk_element_id") or r["chunk_element_id"]
                for r in selected_results
            ]
            adj_params = {
                "element_ids": seed_element_ids,
                "kg_name": kg_name,
                "max_adjacent": max_chunks,
            }
            kg_filter = "AND d.kgName = $kg_name" if kg_name else ""
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
            ))
              {kg_filter}
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
                d.kgName AS kg_name
            ORDER BY d.fileName ASC,
                     coalesce(adj.questionId, ''),
                     coalesce(adj.passageIndex, -1),
                     coalesce(adj.chunkLocalIndex, adj.position),
                     adj.position ASC
            LIMIT $max_adjacent
            """
            try:
                adj_results = graph.query(adj_query, adj_params)
                for result in adj_results:
                    if result["chunk_element_id"] not in seen_ids:
                        context["chunks"].append({
                            "text": result["text"],
                            "chunk_id": result["chunk_id"],
                            "chunk_element_id": result["chunk_element_id"],
                            "position": result.get("position"),
                            "source": result.get("source"),
                            "question_id": result.get("question_id"),
                            "passage_index": result.get("passage_index"),
                            "chunk_local_index": result.get("chunk_local_index"),
                            "score": 0.0,
                            "document": result["document"],
                            "kg_name": result.get("kg_name"),
                            "adjacent": True,
                        })
                        context["documents"].add(result["document"])
                        seen_ids.add(result["chunk_element_id"])
            except Exception as adj_err:
                logging.debug("Adjacent chunk expansion failed (non-fatal): %s", adj_err)

        context["chunks"] = self._sort_chunks_for_query(query, context["chunks"])
        context["documents"] = list(context["documents"])
        return context

    def get_vanilla_rag_context(
        self,
        query: str,
        document_names: List[str] = None,
        similarity_threshold: float = 0.1,
        max_chunks: int = 20,
        kg_name: str = None,
        question_id: str = None,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        retrieval_sample_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Get vanilla RAG context using direct vector similarity search on chunks.
        
        Args:
            query: The search query
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            graph = self._create_neo4j_connection()

            # Check if we have data - optionally filter by kg_name
            if kg_name:
                check_query = """
                MATCH (c:Chunk)-[:PART_OF]->(d:Document {kgName: $kg_name})
                RETURN count(c) as chunk_count LIMIT 1
                """
                check_result = graph.query(check_query, {"kg_name": kg_name})
            else:
                check_query = "MATCH (c:Chunk) RETURN count(c) as chunk_count LIMIT 1"
                check_result = graph.query(check_query)

            if not check_result or check_result[0]['chunk_count'] == 0:
                logging.warning(f"No chunks found in database (kg_name: {kg_name})")
                return {
                    "query": query,
                    "chunks": [],
                    "documents": [],
                    "total_score": 0,
                    "error": f"No data found in database (kg: {kg_name}). Please upload and process a document first."
                }

            if self._first_stage_late_interaction_enabled():
                li_context = self._late_interaction_search(
                    graph,
                    query,
                    document_names=document_names,
                    max_chunks=max_chunks,
                    kg_name=kg_name,
                    question_id=question_id,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=retrieval_sample_id,
                )
                if li_context.get("chunks"):
                    return li_context

            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            vector_index_name = self._resolve_vector_index_name(graph)
            if not vector_index_name:
                logging.warning("No usable vector index found for vanilla retrieval")
                return {
                    "query": query,
                    "chunks": [],
                    "documents": [],
                    "total_score": 0,
                    "error": "No usable vector index found in Neo4j.",
                }

            # Vector search query - direct retrieval from chunks with optional filtering
            where_clauses = []
            params = {
                "query_vector": query_embedding,
                "similarity_threshold": similarity_threshold
            }
            
            if kg_name:
                where_clauses.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name
                
            if document_names:
                where_clauses.append("d.fileName IN $document_names")
                params["document_names"] = document_names

            if question_id:
                where_clauses.append("chunk.questionId = $question_id")
                params["question_id"] = question_id
            
            # Build the complete WHERE clause
            if where_clauses:
                where_clause = "WHERE " + " AND ".join(where_clauses) + " AND score >= $similarity_threshold"
            else:
                where_clause = "WHERE score >= $similarity_threshold"
            
            candidate_limit = compute_candidate_limit(
                max_chunks,
                retrieval_temperature,
                retrieval_shortlist_factor,
                hard_cap=500,
            )
            retrieval_count = min(candidate_limit * 20, 500) if (kg_name or document_names or question_id) else candidate_limit

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
                RETURN
                    retrieval.text AS text,
                    retrieval.id AS chunk_id,
                    elementId(retrieval) AS chunk_element_id,
                    chunk.id AS parent_chunk_id,
                    elementId(chunk) AS parent_chunk_element_id,
                    chunk.position AS position,
                    chunk.source AS source,
                    coalesce(retrieval.questionId, chunk.questionId) AS question_id,
                    coalesce(retrieval.passageIndex, chunk.passageIndex) AS passage_index,
                    coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex) AS chunk_local_index,
                    retrieval.retrievalLocalIndex AS retrieval_local_index,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            else:
                search_query = f"""
                CALL db.index.vector.queryNodes('{vector_index_name}', {retrieval_count}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                {where_clause}
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    chunk.position AS position,
                    chunk.source AS source,
                    chunk.questionId AS question_id,
                    chunk.passageIndex AS passage_index,
                    chunk.chunkLocalIndex AS chunk_local_index,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            params["max_chunks"] = candidate_limit

            results = graph.query(search_query, params)
            selected_results = select_ranked_subset(
                results,
                max_items=max_chunks,
                retrieval_temperature=retrieval_temperature,
                shortlist_factor=retrieval_shortlist_factor,
                sample_id=retrieval_sample_id,
                seed_parts=(
                    "vanilla",
                    kg_name,
                    tuple(document_names or []),
                    query,
                ),
                score_getter=lambda row: float(row.get("score", 0.0)),
            )

            if vector_index_name == "retrieval_vector" and question_id and not selected_results:
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
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    chunk.position AS position,
                    chunk.source AS source,
                    chunk.questionId AS question_id,
                    chunk.passageIndex AS passage_index,
                    chunk.chunkLocalIndex AS chunk_local_index,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name
                ORDER BY score DESC
                LIMIT $max_chunks
                """
                try:
                    fallback_results = graph.query(fallback_query, params)
                except Exception as fallback_exc:
                    logging.warning(
                        "Chunk-vector fallback failed for kg=%s question_id=%s: %s",
                        kg_name,
                        question_id,
                        fallback_exc,
                    )
                else:
                    fallback_selected = select_ranked_subset(
                        fallback_results,
                        max_items=max_chunks,
                        retrieval_temperature=retrieval_temperature,
                        shortlist_factor=retrieval_shortlist_factor,
                        sample_id=retrieval_sample_id,
                        seed_parts=(
                            "vanilla",
                            kg_name,
                            tuple(document_names or []),
                            query,
                            "fallback_vector",
                        ),
                        score_getter=lambda row: float(row.get("score", 0.0)),
                    )
                    if fallback_selected:
                        selected_results = fallback_selected
                        vector_index_name = "vector"

            context = {
                "query": query,
                "chunks": [],
                "documents": set(),
                "total_score": 0,
                "search_method": (
                    "vanilla_retrieval_span_similarity"
                    if vector_index_name == "retrieval_vector"
                    else "vanilla_vector_similarity"
                ),
                "retrieval_sampling": {
                    "temperature": float(retrieval_temperature or 0.0),
                    "shortlist_factor": int(retrieval_shortlist_factor or 1),
                    "sample_id": int(retrieval_sample_id or 0),
                    "candidate_limit": int(candidate_limit),
                },
            }

            seen_ids = set()
            for result in selected_results:
                chunk_info = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "position": result.get("position"),
                    "source": result.get("source"),
                    "question_id": result.get("question_id"),
                    "passage_index": result.get("passage_index"),
                    "chunk_local_index": result.get("chunk_local_index"),
                    "retrieval_local_index": result.get("retrieval_local_index"),
                    "parent_chunk_id": result.get("parent_chunk_id"),
                    "parent_chunk_element_id": result.get("parent_chunk_element_id"),
                    "score": result["score"],
                    "document": result["document"],
                    "kg_name": result.get("kg_name"),
                }
                context["chunks"].append(chunk_info)
                context["documents"].add(result["document"])
                context["total_score"] += result["score"]
                seen_ids.add(result["chunk_element_id"])

            # Adjacent chunk expansion: fetch position±1 neighbours of every
            # retrieved chunk to capture answers split across chunk boundaries.
            # When retrieval_vector is active, chunk_element_id is a RetrievalChunk
            # element id; use parent_chunk_element_id (the Chunk node) instead so
            # the adjacency query correctly matches Chunk nodes.
            if selected_results:
                seed_element_ids = [
                    r.get("parent_chunk_element_id") or r["chunk_element_id"]
                    for r in selected_results
                ]
                adj_params = {
                    "element_ids": seed_element_ids,
                    "kg_name": kg_name,
                    "max_adjacent": max_chunks,
                }
                kg_filter = "AND d.kgName = $kg_name" if kg_name else ""
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
                ))
                  {kg_filter}
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
                    d.kgName AS kg_name
                ORDER BY d.fileName ASC,
                         coalesce(adj.questionId, ''),
                         coalesce(adj.passageIndex, -1),
                         coalesce(adj.chunkLocalIndex, adj.position),
                         adj.position ASC
                LIMIT $max_adjacent
                """
                try:
                    adj_results = graph.query(adj_query, adj_params)
                    for result in adj_results:
                        if result["chunk_element_id"] not in seen_ids:
                            context["chunks"].append({
                                "text": result["text"],
                                "chunk_id": result["chunk_id"],
                                "chunk_element_id": result["chunk_element_id"],
                                "position": result.get("position"),
                                "source": result.get("source"),
                                "question_id": result.get("question_id"),
                                "passage_index": result.get("passage_index"),
                                "chunk_local_index": result.get("chunk_local_index"),
                                "score": 0.0,
                                "document": result["document"],
                                "kg_name": result.get("kg_name"),
                                "adjacent": True,
                            })
                            context["documents"].add(result["document"])
                            seen_ids.add(result["chunk_element_id"])
                except Exception as adj_err:
                    logging.debug("Adjacent chunk expansion failed (non-fatal): %s", adj_err)

            context["chunks"] = self._sort_chunks_for_query(query, context["chunks"])
            context["documents"] = list(context["documents"])

            return context

        except Exception as e:
            logging.error(f"Error getting vanilla RAG context: {e}")
            return {
                "query": query,
                "chunks": [],
                "documents": [],
                "total_score": 0,
                "error": str(e)
            }

    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Format the context for the LLM prompt (simplified, no entities)
        """
        # Format chunks with scores
        chunk_texts = []
        for i, chunk in enumerate(context["chunks"], 1):
            chunk_text = f"Chunk {i} (Similarity Score: {chunk['score']:.3f}):\n{chunk['text']}\n"
            chunk_texts.append(chunk_text)

        formatted_context = "\n".join(chunk_texts)
        return formatted_context

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
    ) -> str:
        formatted_context = self.format_context_for_llm(context)

        if extra_context_texts:
            extra_block = "\n\n".join(
                f"Provided Context {i+1}:\n{t}" for i, t in enumerate(extra_context_texts) if t.strip()
            )
            formatted_context = extra_block + "\n\n" + formatted_context if formatted_context else extra_block
            logging.info(f"Prepended {len(extra_context_texts)} extra context(s) to prompt")

        chain = self.rag_prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": formatted_context,
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
    ) -> tuple:
        if not self._runtime_answer_guardrail_enabled(runtime_guardrail):
            return response, {
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
            return response, metadata
        metadata["final_decision"] = "abstain"
        return RUNTIME_GUARDRAIL_ABSTENTION, metadata

    def generate_response(
        self,
        question: str,
        llm,
        document_names: List[str] = None,
        similarity_threshold: float = 0.1,
        max_chunks: int = 20,
        extra_context_texts: Optional[List[str]] = None,
        kg_name: str = None,
        answer_instructions: str = "",
        question_id: str = None,
        runtime_guardrail: Optional[bool] = None,
        runtime_guardrail_mode: Optional[str] = None,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        retrieval_sample_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate a vanilla RAG response using direct vector retrieval.

        Args:
            question: The question to answer
            llm: The LLM to use for response generation
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            extra_context_texts: Optional list of additional context strings to prepend
                to the retrieved chunks (e.g. ground-truth question contexts for MIRAGE eval).
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            logging.info(f"Starting vanilla RAG generate_response for question: {question}")

            # Get context using vanilla vector retrieval
            context = self.get_vanilla_rag_context(
                question,
                document_names=document_names,
                similarity_threshold=similarity_threshold,
                max_chunks=max_chunks,
                kg_name=kg_name,
                question_id=question_id,
                retrieval_temperature=retrieval_temperature,
                retrieval_shortlist_factor=retrieval_shortlist_factor,
                retrieval_sample_id=retrieval_sample_id,
            )
            logging.info(f"Got context with {len(context.get('chunks', []))} chunks")

            if self._should_run_query_fusion(question, context):
                variant_queries = self._generate_query_variants(question, llm)
                variant_contexts: List[Dict[str, Any]] = [context]
                for variant_query in variant_queries:
                    try:
                        alt_context = self.get_vanilla_rag_context(
                            variant_query,
                            document_names=document_names,
                            similarity_threshold=similarity_threshold,
                            max_chunks=max_chunks,
                            kg_name=kg_name,
                            question_id=question_id,
                            retrieval_temperature=retrieval_temperature,
                            retrieval_shortlist_factor=retrieval_shortlist_factor,
                            retrieval_sample_id=retrieval_sample_id,
                        )
                    except Exception as fusion_exc:
                        logging.debug(
                            "Vanilla query-fusion retrieval failed for variant %r (non-fatal): %s",
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
                    search_method="query_fusion_" + str(context.get("search_method") or "vanilla_vector_similarity"),
                )
                if fused_context and fused_context.get("chunks"):
                    context = fused_context

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

            if not context["chunks"] and not extra_context_texts:
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "context": context,
                    "sources": [],
                    "confidence": 0.0
                }

            response = self._invoke_answer_chain(
                question=question,
                llm=llm,
                context=context,
                answer_instructions=answer_instructions,
                extra_context_texts=extra_context_texts,
            )
            response, guardrail = self._apply_runtime_answer_guardrail(
                question=question,
                llm=llm,
                response=response,
                context=context,
                runtime_guardrail=runtime_guardrail,
                runtime_guardrail_mode=runtime_guardrail_mode,
            )

            # Retrieval scores are already cosine similarities in [0, 1].
            avg_score = context["total_score"] / len(context["chunks"]) if context["chunks"] else 0
            confidence = avg_score
            if guardrail.get("enabled") and guardrail.get("final_decision") != "keep":
                confidence = 0.0

            return {
                "response": response,
                "context": context,
                "sources": context["documents"],
                "confidence": confidence,
                "guardrail": guardrail,
                "chunk_count": len(context["chunks"]),
                "retrieval_params": {
                    "similarity_threshold": similarity_threshold,
                    "max_chunks": max_chunks,
                    "retrieval_temperature": float(retrieval_temperature or 0.0),
                    "retrieval_shortlist_factor": int(retrieval_shortlist_factor or 1),
                    "retrieval_sample_id": int(retrieval_sample_id or 0),
                },
                "late_interaction_stage": context.get("late_interaction_stage", {}),
                "late_interaction": context.get("late_interaction", {}),
                "reranker": context.get("reranker", {}),
            }

        except Exception as e:
            logging.error(f"Error generating vanilla RAG response: {e}")
            return {
                "response": f"An error occurred while generating the response: {str(e)}",
                "context": {},
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
