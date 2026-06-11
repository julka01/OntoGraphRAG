import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from ontographrag.rag.retrieval_sampling import (
    compute_candidate_limit,
    select_ranked_subset,
)

class ContextAssemblyMixin:
    """Chunk selection, context merging, fusion, and prompt formatting.

    Mixin for :class:`EnhancedRAGSystem`; method bodies are unchanged
    from the original monolithic implementation.
    """

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
