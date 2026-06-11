import logging
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class IterativeRetrievalMixin:
    """Iterative multi-hop sub-question retrieval.

    Mixin for :class:`EnhancedRAGSystem`; method bodies are unchanged
    from the original monolithic implementation.
    """

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
