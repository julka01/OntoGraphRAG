import json
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import logging

class EnrichmentMixin:
    """Anchor grounding, relationship recovery, verification, and graph repair.

    Mixin for :class:`OntologyGuidedKGCreator`; method bodies are
    unchanged from the original monolithic implementation.
    """

    @classmethod
    def _merge_anchor_spans(cls, *span_groups: Any) -> List[Dict[str, Any]]:
        """Merge and deduplicate anchor span dicts."""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for group in span_groups:
            if not group:
                continue
            if isinstance(group, str):
                try:
                    group = json.loads(group)
                except Exception:
                    continue
            if not isinstance(group, list):
                continue
            for span in group:
                if not isinstance(span, dict):
                    continue
                start = span.get("start")
                end = span.get("end")
                text = str(span.get("text") or "").strip()
                chunk_position = span.get("chunk_position")
                local_start = span.get("local_start")
                local_end = span.get("local_end")
                key = (start, end, text, chunk_position, local_start, local_end)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "text": text,
                        "start": start,
                        "end": end,
                        "chunk_position": chunk_position,
                        "local_start": local_start,
                        "local_end": local_end,
                    }
                )
        merged.sort(
            key=lambda span: (
                float("inf") if span.get("start") is None else int(span.get("start")),
                float("inf") if span.get("end") is None else int(span.get("end")),
                str(span.get("text") or ""),
            )
        )
        return merged

    @classmethod
    def _merge_anchor_grounding(cls, *groundings: Any) -> Dict[str, List[Dict[str, Any]]]:
        """Merge nested anchor grounding dicts component-wise."""
        merged: Dict[str, List[Dict[str, Any]]] = {}
        for grounding in groundings:
            if not grounding:
                continue
            if isinstance(grounding, str):
                try:
                    grounding = json.loads(grounding)
                except Exception:
                    continue
            if not isinstance(grounding, dict):
                continue
            for key, spans in grounding.items():
                merged[key] = cls._merge_anchor_spans(merged.get(key), spans)
        return merged

    def _relationship_anchor_candidates(self, rel: Dict[str, Any]) -> List[str]:
        """Best-effort exact text candidates for relation-phrase restoration."""
        properties = dict(rel.get("properties") or {})
        relation_type = str(rel.get("type") or "").strip()
        candidates: List[str] = []
        for value in (
            properties.get("anchor_text"),
            properties.get("description"),
            relation_type.replace("_", " ").replace("-", " "),
            relation_type,
        ):
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        return list(dict.fromkeys(candidates))

    def _restoration_from_anchor_grounding(
        self,
        grounding: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Summarize restoration quality from grounded subject/relation/object anchors."""
        core_components = ("source", "relation", "target")
        grounded_components = [
            component
            for component in core_components
            if self._merge_anchor_spans((grounding or {}).get(component))
        ]
        grounded_count = len(grounded_components)
        if grounded_count == 3:
            status = "full"
        elif grounded_count >= 2 and {"source", "target"}.issubset(set(grounded_components)):
            status = "partial"
        elif grounded_count >= 1:
            status = "minimal"
        else:
            status = "failed"
        return {
            "status": status,
            "verified": status == "full",
            "grounded_components": grounded_components,
            "grounded_count": grounded_count,
        }

    def _verify_relationship_restoration(
        self,
        rel: Dict[str, Any],
        verification_chunks: List[Dict[str, Any]],
        *,
        source_name: str,
        target_name: str,
        relation_type: str,
        source_aliases: Optional[List[str]] = None,
        target_aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Restoration-style verification using exact spans across scoped evidence chunks."""
        properties = dict(rel.get("properties") or {})
        source_candidates = [source_name] + [alias for alias in (source_aliases or []) if isinstance(alias, str)]
        target_candidates = [target_name] + [alias for alias in (target_aliases or []) if isinstance(alias, str)]
        relation_candidates = self._relationship_anchor_candidates(
            {"type": relation_type, "properties": properties}
        )
        condition_candidates = [str(properties.get("condition") or "")]
        quantitative_candidates = [str(properties.get("quantitative") or "")]

        source_spans = self._merge_anchor_spans(
            properties.get("source_anchor_spans"),
            *[
                self._find_exact_text_spans(
                    str(chunk.get("text") or ""),
                    source_candidates,
                    start_offset=int(chunk.get("start_pos") or 0),
                    chunk_position=int(chunk.get("position", 0) or 0),
                )
                for chunk in (verification_chunks or [])
            ],
        )
        target_spans = self._merge_anchor_spans(
            properties.get("target_anchor_spans"),
            *[
                self._find_exact_text_spans(
                    str(chunk.get("text") or ""),
                    target_candidates,
                    start_offset=int(chunk.get("start_pos") or 0),
                    chunk_position=int(chunk.get("position", 0) or 0),
                )
                for chunk in (verification_chunks or [])
            ],
        )
        relation_spans = self._merge_anchor_spans(
            properties.get("relation_anchor_spans"),
            *[
                self._find_exact_text_spans(
                    str(chunk.get("text") or ""),
                    relation_candidates,
                    start_offset=int(chunk.get("start_pos") or 0),
                    chunk_position=int(chunk.get("position", 0) or 0),
                )
                for chunk in (verification_chunks or [])
            ],
        )
        condition_spans = self._merge_anchor_spans(
            properties.get("condition_anchor_spans"),
            *[
                self._find_exact_text_spans(
                    str(chunk.get("text") or ""),
                    condition_candidates,
                    start_offset=int(chunk.get("start_pos") or 0),
                    chunk_position=int(chunk.get("position", 0) or 0),
                )
                for chunk in (verification_chunks or [])
            ],
        )
        quantitative_spans = self._merge_anchor_spans(
            properties.get("quantitative_anchor_spans"),
            *[
                self._find_exact_text_spans(
                    str(chunk.get("text") or ""),
                    quantitative_candidates,
                    start_offset=int(chunk.get("start_pos") or 0),
                    chunk_position=int(chunk.get("position", 0) or 0),
                )
                for chunk in (verification_chunks or [])
            ],
        )

        anchor_grounding = self._merge_anchor_grounding(
            properties.get("anchor_grounding"),
            {
                "source": source_spans,
                "target": target_spans,
                "relation": relation_spans,
                "condition": condition_spans,
                "quantitative": quantitative_spans,
            },
        )
        restoration = self._restoration_from_anchor_grounding(anchor_grounding)
        restoration["anchor_grounding"] = anchor_grounding
        return restoration

    def _augment_result_with_anchor_entities(
        self,
        result: Dict[str, Any],
        anchor_inventory: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Ensure anchor-discovered entities survive even if the extraction omits them."""
        merged_entities: Dict[str, Dict[str, Any]] = {}
        for anchor in anchor_inventory.get("entity_anchors", []):
            merged_entities[self._normalize_entity_text(anchor["text"])] = {
                "id": anchor["text"],
                "type": anchor["type"],
                "properties": {
                    "name": anchor["text"],
                    "description": f"{anchor['type']}: {anchor['text']}",
                    "anchor_spans": anchor.get("anchor_spans") or [],
                    "anchor_discovered": True,
                },
            }

        for entity in result.get("entities", []) or []:
            if isinstance(entity, str):
                entity = {"id": entity}
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("id") or "").strip()
            if not entity_id:
                continue
            key = self._normalize_entity_text(entity_id)
            merged = dict(merged_entities.get(key) or {})
            merged["id"] = entity_id
            merged["type"] = entity.get("type") or merged.get("type")
            merged_props = dict(merged.get("properties") or {})
            merged_props.update(dict(entity.get("properties") or {}))
            merged["properties"] = merged_props
            merged_entities[key] = merged

        result["entities"] = list(merged_entities.values())
        return result

    def _supplement_relationships_from_unused_anchors(
        self,
        result: Dict[str, Any],
        *,
        chunk_text: str,
        llm,
        model_name: str,
        context_preamble: str,
        anchor_inventory: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Coverage-aware supplement for entity anchors unused by the first extraction pass."""
        if not getattr(self, "enable_anchor_coverage_supplement", True):
            return result
        entity_anchors = anchor_inventory.get("entity_anchors", [])
        if llm is None or len(entity_anchors) < 2:
            return result

        used_anchor_keys = {
            self._normalize_entity_text(rel.get("source", ""))
            for rel in result.get("relationships", []) or []
            if isinstance(rel, dict)
        } | {
            self._normalize_entity_text(rel.get("target", ""))
            for rel in result.get("relationships", []) or []
            if isinstance(rel, dict)
        }
        unused = [
            anchor["text"]
            for anchor in entity_anchors
            if self._normalize_entity_text(anchor["text"]) not in used_anchor_keys
        ]
        if not unused:
            return result

        all_entity_texts = [anchor["text"] for anchor in entity_anchors]
        relation_texts = [anchor["text"] for anchor in anchor_inventory.get("relation_anchors", [])]
        attribute_lines = [
            f"- {anchor['text']} (role: {anchor['role']})"
            for anchor in anchor_inventory.get("attribute_anchors", [])
        ]

        supplement_prompt = f"""
You are filling coverage gaps in a grounded knowledge graph extraction.

KNOWN ENTITY ANCHORS:
{json.dumps(all_entity_texts, ensure_ascii=False, indent=2)}

RELATION ANCHORS:
{json.dumps(relation_texts, ensure_ascii=False, indent=2)}

ATTRIBUTE ANCHORS:
{chr(10).join(attribute_lines) if attribute_lines else "- none"}

UNUSED ENTITY ANCHORS:
{json.dumps(unused[:8], ensure_ascii=False, indent=2)}

TASK:
Find any additional supported relationships in the text that involve at least one UNUSED ENTITY ANCHOR.

Return ONLY a JSON object:
{{
  "relationships": [
    {{
      "source": "exact entity anchor text",
      "target": "exact entity anchor text",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "anchor_text": "exact relation anchor text, or null",
        "description": "brief grounded relation description",
        "condition": "exact condition attribute anchor text, or null",
        "quantitative": "exact quantitative attribute anchor text, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ]
}}

Rules:
- Every source and target MUST come from KNOWN ENTITY ANCHORS.
- At least one endpoint of each relationship MUST come from UNUSED ENTITY ANCHORS.
- Do not invent anchor text.
- If there are no additional relationships, return {{"relationships": []}}.

{context_preamble}TEXT TO ANALYZE:
{chunk_text}

Return ONLY the JSON object.
"""
        supplement_response = llm.generate(supplement_prompt, "", model_name)
        supplement_data = self._parse_llm_json_response(
            supplement_response,
            allow_partial_extraction=True,
            empty_result={"relationships": []},
        )
        extra_relationships = supplement_data.get("relationships")
        if isinstance(extra_relationships, list) and extra_relationships:
            result.setdefault("relationships", [])
            result["relationships"].extend(extra_relationships)
        return result

    def _recover_cross_section_relationships(
        self,
        *,
        text: str,
        chunks: List[Dict[str, Any]],
        entities_per_chunk: Dict[int, List[Dict[str, Any]]],
        section_headers: List[Tuple[int, str]],
        llm,
        model_name: str,
        scope_label: str,
    ) -> List[Dict[str, Any]]:
        """Recover relations across section boundaries after section-aware chunking."""
        if llm is None:
            return []

        section_segments = self._build_section_segments(
            text,
            section_headers=section_headers,
        )
        if len(section_segments) < 2:
            return []

        section_texts: List[str] = []
        section_entities: Dict[int, List[Dict[str, Any]]] = {}
        section_positions: Dict[int, List[int]] = {}

        for section_idx, segment in enumerate(section_segments):
            section_texts.append(segment["text"])
            positions: List[int] = []
            entities: List[Dict[str, Any]] = []
            for chunk_idx, chunk in enumerate(chunks):
                chunk_start = int(chunk.get("start_pos", 0) or 0)
                if not (segment["start_pos"] <= chunk_start < segment["end_pos"]):
                    continue
                positions.append(int(chunk.get("position", chunk_idx)))
                entities.extend(entities_per_chunk.get(chunk_idx, []))
            section_positions[section_idx] = sorted(set(positions))
            section_entities[section_idx] = entities

        if sum(1 for entities in section_entities.values() if len(entities) >= 1) < 2:
            return []

        return self._extract_relationships_for_segment_windows(
            section_texts,
            section_entities,
            section_positions,
            llm,
            model_name,
            max_window_size=getattr(self, "cross_section_relation_window", 2),
            scope_label=scope_label,
            relationship_scope_metadata={
                "dataset": chunks[0].get("dataset") if chunks else None,
                "question_id": str(chunks[0].get("question_id")) if chunks and chunks[0].get("question_id") is not None else None,
                "passage_index": int(chunks[0].get("passage_index")) if chunks and chunks[0].get("passage_index") is not None else None,
                "source": chunks[0].get("source") if chunks else None,
                "source_title": chunks[0].get("source_title") if chunks else None,
                "source_scope_key": chunks[0].get("source_scope_key") if chunks else None,
            },
        )

    def _group_passages_for_cross_passage_recovery(
        self,
        passages,
        passage_entities: Dict[int, List[Dict[str, Any]]],
        passage_positions: Dict[int, List[int]],
    ) -> List[Dict[str, Any]]:
        """Return contiguous passage groups that may participate in cross-passage recovery.

        Question-scoped bundle datasets should only recover relations within a
        single question bundle. Shared-corpus datasets typically expose unique
        passage question ids, so this grouping naturally degenerates to
        singletons and skips cross-passage recovery altogether.
        """
        groups: List[Dict[str, Any]] = []
        current_group: Optional[Dict[str, Any]] = None

        for idx, passage in enumerate(passages):
            group_key = (
                getattr(passage, "dataset", ""),
                str(getattr(passage, "question_id", "") or ""),
            )
            if current_group is None or current_group["group_key"] != group_key:
                current_group = {
                    "group_key": group_key,
                    "texts": [],
                    "entities": {},
                    "positions": {},
                    "source_label": f"Cross-passage[{group_key[0]}/{group_key[1]}]",
                }
                groups.append(current_group)

            local_idx = len(current_group["texts"])
            current_group["texts"].append(getattr(passage, "text", "") or "")
            current_group["entities"][local_idx] = passage_entities.get(idx, [])
            current_group["positions"][local_idx] = passage_positions.get(idx, [])

        return groups

    def _verify_triple_confidence(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        chunks: List[Dict],
        source_aliases: List[str] = None,
        target_aliases: List[str] = None,
    ) -> float:
        """
        Evidence-grounded verification for an extracted triple (inspired by MOSAICX verify).

        Searches all chunks for co-occurrence of the source and target entity names.
        Returns a confidence score in [0.0, 1.0]:
          - 1.0  both names found in the same sentence
          - 0.7  both names found in the same chunk (not same sentence)
          - 0.4  only one name found in any chunk
          - 0.1  neither name found (LLM may have hallucinated this triple)

        This is intentionally cheap (string matching only) so it doesn't add
        meaningful latency. An LLM-based re-verification pass can be added later
        for low-confidence triples as an opt-in upgrade.
        """
        if not chunks or not source_name or not target_name:
            return 0.5  # neutral when no evidence to check

        src_lower = source_name.strip().lower()
        tgt_lower = target_name.strip().lower()

        # Build boundary-aware patterns.
        # Use (?<!\w)/(?!\w) lookarounds instead of \b so that entity names that
        # start or end with non-word characters (parentheses, hyphens, dots) are
        # still matched correctly.  e.g. "gleason score (gs)" ends with ')' which
        # is not a word char — \b would fail there, but (?!\w) does not.
        def _boundary_pattern(name: str) -> re.Pattern:
            prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
            suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
            return re.compile(prefix + re.escape(name) + suffix)

        def _surface_forms(name: str) -> set:
            """Return the name plus underscore↔space variants so LLM IDs like
            'United_States' match chunk text 'United States' and vice versa."""
            forms = {name}
            forms.add(name.replace('_', ' '))
            forms.add(name.replace(' ', '_'))
            return {f for f in forms if f}

        # Build a list of patterns for each entity — canonical name plus all aliases,
        # including underscore/space variants to survive LLM ID normalisation differences.
        _src_forms = list(
            _surface_forms(src_lower)
            | {v for a in (source_aliases or []) if a for v in _surface_forms(a.strip().lower())}
        )
        _tgt_forms = list(
            _surface_forms(tgt_lower)
            | {v for a in (target_aliases or []) if a for v in _surface_forms(a.strip().lower())}
        )
        src_pats = [_boundary_pattern(f) for f in _src_forms if f]
        tgt_pats = [_boundary_pattern(f) for f in _tgt_forms if f]

        # Keep single-pattern aliases for backwards compatibility
        src_pat = src_pats[0] if src_pats else _boundary_pattern(src_lower)
        tgt_pat = tgt_pats[0] if tgt_pats else _boundary_pattern(tgt_lower)

        found_src, found_tgt, same_chunk, same_sentence = False, False, False, False

        for chunk in chunks:
            text = chunk.get('text', '').lower()
            has_src = any(p.search(text) for p in src_pats)
            has_tgt = any(p.search(text) for p in tgt_pats)

            if has_src:
                found_src = True
            if has_tgt:
                found_tgt = True

            if has_src and has_tgt:
                # Both entities present in this single chunk — check for sentence co-occurrence
                same_chunk = True
                sentences = re.split(r'(?<=[.!?])\s+', text)
                if any(
                    any(sp.search(s) for sp in src_pats) and any(tp.search(s) for tp in tgt_pats)
                    for s in sentences
                ):
                    same_sentence = True
                    break  # best possible score — stop scanning

        if same_sentence:
            return 1.0
        if same_chunk:
            # Both found within the same chunk (but not the same sentence)
            return 0.7
        if found_src and found_tgt:
            # Found in separate chunks of the same document — weaker evidence
            return 0.4
        if found_src or found_tgt:
            return 0.3
        return 0.1

    def _reverify_low_confidence_triple(
        self,
        *,
        source_name: str,
        target_name: str,
        relationship_type: str,
        verification_chunks: List[Dict[str, Any]],
        llm,
        model_name: str,
    ) -> Optional[bool]:
        """Ask the LLM to verify a low-confidence triple against local text only."""
        if llm is None:
            return None

        evidence = "\n\n".join(
            str(chunk.get("text") or "").strip()
            for chunk in verification_chunks[:4]
            if str(chunk.get("text") or "").strip()
        )
        if not evidence:
            return None

        cache_key = hashlib.sha1(
            f"{source_name}|{relationship_type}|{target_name}|{evidence}".encode("utf-8")
        ).hexdigest()
        if cache_key in self._triple_reverification_cache:
            return self._triple_reverification_cache[cache_key]

        prompt = f"""You are verifying whether a knowledge-graph relation is directly supported by text.

CLAIM:
- Source: {source_name}
- Relationship: {relationship_type}
- Target: {target_name}

EVIDENCE TEXT:
{evidence}

Rules:
- Answer YES only if the text directly supports this relation.
- Answer NO if the relation is unsupported, contradicted, or only weakly implied.
- Use only the evidence text above.
- Return exactly YES or NO.
"""
        try:
            response = str(llm.generate(prompt, "", model_name)).strip().lower()
        except Exception as exc:
            logging.debug("Low-confidence triple reverification failed: %s", exc)
            return None

        if response.startswith("yes"):
            self._triple_reverification_cache[cache_key] = True
            return True
        if response.startswith("no"):
            self._triple_reverification_cache[cache_key] = False
            return False
        return None

    def _heuristic_component_summary_text(
        self,
        component: Dict[str, Any],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        """Produce a deterministic prose summary for one connected component."""
        entity_labels = [
            str((node_lookup.get(entity_id, {}).get("properties") or {}).get("name") or entity_id)
            for entity_id in component.get("top_entities") or []
        ]
        rel_phrases: List[str] = []
        for rel in component.get("relationships") or []:
            source_id = str(rel.get("source") or rel.get("from") or "").strip()
            target_id = str(rel.get("target") or rel.get("to") or "").strip()
            source_name = str((node_lookup.get(source_id, {}).get("properties") or {}).get("name") or source_id)
            target_name = str((node_lookup.get(target_id, {}).get("properties") or {}).get("name") or target_id)
            relation = str(rel.get("type") or "RELATED_TO").replace("_", " ").lower()
            if rel.get("negated"):
                rel_phrases.append(f"{source_name} is not linked by {relation} to {target_name}")
            else:
                rel_phrases.append(f"{source_name} {relation} {target_name}")

        summary_lines = [
            f"Component with {component.get('entity_count', 0)} entities and {component.get('relationship_count', 0)} relationships.",
        ]
        if entity_labels:
            summary_lines.append("Central entities: " + ", ".join(entity_labels[: getattr(self, "max_summary_entities", 6)]) + ".")
        if rel_phrases:
            summary_lines.append("Key relations: " + "; ".join(rel_phrases[: getattr(self, "max_summary_relationships", 6)]) + ".")
        return " ".join(summary_lines)

    def _claim_records_from_component(
        self,
        component: Dict[str, Any],
        node_lookup: Dict[str, Dict[str, Any]],
        *,
        component_index: int,
    ) -> List[Dict[str, Any]]:
        """Synthesize simple claim records from component relations."""
        if not getattr(self, "enable_claim_extraction", False):
            return []

        claims: List[Dict[str, Any]] = []
        for rel_index, rel in enumerate(component.get("relationships") or []):
            source_id = str(rel.get("source") or rel.get("from") or "").strip()
            target_id = str(rel.get("target") or rel.get("to") or "").strip()
            source_name = str((node_lookup.get(source_id, {}).get("properties") or {}).get("name") or source_id)
            target_name = str((node_lookup.get(target_id, {}).get("properties") or {}).get("name") or target_id)
            relation = str(rel.get("type") or "RELATED_TO").replace("_", " ").lower()
            if rel.get("negated"):
                text = f"{source_name} is not supported to {relation} {target_name}."
                polarity = "negated"
            else:
                text = f"{source_name} {relation} {target_name}."
                polarity = "affirmed"
            claim_id = hashlib.sha1(
                f"{component_index}:{source_id}:{relation}:{target_id}:{polarity}".encode("utf-8")
            ).hexdigest()
            claims.append(
                {
                    "id": claim_id,
                    "text": text,
                    "polarity": polarity,
                    "confidence": self._confidence_score(
                        (rel.get("properties") or {}).get("confidence")
                    ),
                    "entity_ids": [entity_id for entity_id in [source_id, target_id] if entity_id],
                    "provenance_positions": list(rel.get("provenance_positions") or []),
                    "component_index": component_index,
                    "relationship_index": rel_index,
                }
            )
        return claims

    def _propose_fragmentation_bridges(
        self,
        components: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Propose conservative synthetic bridge edges across disconnected components."""
        if not getattr(self, "enable_fragmentation_repair", False):
            return []
        if len(components) < 2:
            return []

        candidate_bridges: List[Dict[str, Any]] = []
        generic_types = {"concept", "entity", "unknown", "other", ""}

        for left_index, left_component in enumerate(components):
            for right_index, right_component in enumerate(components[left_index + 1:], start=left_index + 1):
                best_bridge = None
                for left_entity_id in left_component.get("top_entities") or []:
                    left_node = node_lookup.get(left_entity_id, {})
                    left_name = str((left_node.get("properties") or {}).get("name") or left_entity_id)
                    left_type = str((left_node.get("properties") or {}).get("type") or left_node.get("label") or "").strip().lower()
                    left_alias_keys = {
                        key
                        for candidate in self._entity_candidate_names(left_node)
                        for key in self._soft_entity_alias_keys(candidate)
                    }
                    for right_entity_id in right_component.get("top_entities") or []:
                        right_node = node_lookup.get(right_entity_id, {})
                        right_name = str((right_node.get("properties") or {}).get("name") or right_entity_id)
                        right_type = str((right_node.get("properties") or {}).get("type") or right_node.get("label") or "").strip().lower()
                        right_alias_keys = {
                            key
                            for candidate in self._entity_candidate_names(right_node)
                            for key in self._soft_entity_alias_keys(candidate)
                        }
                        alias_overlap = sorted(left_alias_keys & right_alias_keys)
                        lexical_guard = bool(alias_overlap) or self._names_pass_synonym_guard(
                            left_name,
                            right_name,
                        )
                        if not lexical_guard:
                            continue
                        if (
                            left_type not in generic_types
                            and right_type not in generic_types
                            and left_type == right_type
                            and alias_overlap
                        ):
                            # Same-type exact alias matches should have been merged by soft linking,
                            # so remaining matches are more useful as bridge hints than duplicate merges.
                            pass

                        similarity = self._cosine_similarity(
                            left_node.get("embedding"),
                            right_node.get("embedding"),
                        )
                        if similarity and similarity < getattr(
                            self,
                            "fragmentation_bridge_similarity_threshold",
                            0.92,
                        ):
                            continue
                        confidence = max(similarity, 0.95 if alias_overlap else 0.75)
                        bridge = {
                            "source_id": left_entity_id,
                            "target_id": right_entity_id,
                            "source_name": left_name,
                            "target_name": right_name,
                            "confidence": round(float(confidence), 4),
                            "reason": "alias_overlap" if alias_overlap else "lexical_guard",
                            "shared_alias_keys": alias_overlap,
                            "left_component": left_index,
                            "right_component": right_index,
                        }
                        if best_bridge is None or bridge["confidence"] > best_bridge["confidence"]:
                            best_bridge = bridge
                if best_bridge:
                    candidate_bridges.append(best_bridge)

        candidate_bridges.sort(
            key=lambda bridge: (
                -float(bridge.get("confidence") or 0.0),
                str(bridge.get("source_name") or ""),
                str(bridge.get("target_name") or ""),
            )
        )
        return candidate_bridges[: max(0, getattr(self, "max_fragmentation_bridges", 8))]
