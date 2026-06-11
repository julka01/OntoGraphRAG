import json
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from ontographrag.kg.chunking import chunk_text as _chunk_text_fn
from collections import defaultdict, Counter
import logging
import time
from ontographrag.schemas.models import (
    EntityType as OntEntityType,
    RelationshipType as OntRelType,
    PropertyType,
)

class ExtractionMixin:
    """LLM entity/relationship extraction, prompt construction, and chunk preparation.

    Mixin for :class:`OntologyGuidedKGCreator`; method bodies are
    unchanged from the original monolithic implementation.
    """

    def _select_relationship_types_for_prompt(
        self,
        chunk_text: str,
        selected_entity_types: List[OntEntityType],
        max_rel_types: int,
    ) -> List[OntRelType]:
        """Prefer relationship types that are lexically or structurally relevant."""
        schema = self._ontology_schema
        if not schema or not schema.relationship_types:
            return []

        chunk_lower = chunk_text.lower()
        selected_ids = {et.id for et in selected_entity_types}
        scored: List[Tuple[int, int, OntRelType]] = []
        for idx, rt in enumerate(schema.relationship_types):
            score = 0
            id_form = rt.id.lower()
            label_form = rt.label.lower()
            if id_form in chunk_lower or label_form in chunk_lower:
                score += 4
            if rt.domain in selected_ids:
                score += 2
            if rt.range in selected_ids:
                score += 2
            if rt.domain in selected_ids and rt.range in selected_ids:
                score += 2
            scored.append((score, idx, rt))

        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
        selected = [rt for score, _, rt in ranked if score > 0][:max_rel_types]
        if len(selected) < max_rel_types:
            selected_ids_seen = {rt.id for rt in selected}
            for _, _, rt in ranked:
                if rt.id in selected_ids_seen:
                    continue
                selected.append(rt)
                selected_ids_seen.add(rt.id)
                if len(selected) >= max_rel_types:
                    break
        return selected

    def _ontology_prompt_keyword_set(self, text: str) -> set:
        """Return normalized keywords for schema relevance scoring."""
        if not isinstance(text, str):
            return set()
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {
            token for token in tokens
            if len(token) >= 4 and token not in {"with", "from", "that", "this", "were", "which"}
        }

    def _rank_entity_types_for_prompt(
        self,
        chunk_text: str,
        entity_types: List[OntEntityType],
    ) -> List[Tuple[int, int, OntEntityType]]:
        """Stage 1 of ontology prompting: score entity types by local relevance."""
        chunk_lower = chunk_text.lower()
        chunk_tokens = self._ontology_prompt_keyword_set(chunk_text)
        ranked: List[Tuple[int, int, OntEntityType]] = []
        for idx, et in enumerate(entity_types):
            score = 0
            label_text = et.label.lower()
            id_text = et.id.lower()
            if label_text in chunk_lower or id_text in chunk_lower:
                score += 8
            candidate_text = " ".join(
                [
                    et.id,
                    et.label,
                    et.description or "",
                    " ".join(p.name for p in et.properties),
                ]
            )
            overlap = len(chunk_tokens & self._ontology_prompt_keyword_set(candidate_text))
            score += min(overlap, 4)
            ranked.append((score, idx, et))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    @staticmethod
    def _adaptive_prompt_item_cap(
        ranked_items: List[Tuple[int, int, Any]],
        *,
        base_cap: int,
        absolute_cap: int,
    ) -> int:
        """Grow the cap when many items are relevant, while keeping a hard ceiling."""
        positive = sum(1 for score, _, _ in ranked_items if score > 0)
        return min(len(ranked_items), max(base_cap, min(base_cap + positive, absolute_cap)))

    @staticmethod
    def _render_prompt_lines_with_budget(
        lines: List[str],
        *,
        char_budget: int,
        min_items: int,
        hard_cap: int,
    ) -> List[str]:
        """Stage 2 of ontology prompting: keep relevant lines until the prompt budget is full."""
        selected: List[str] = []
        total_chars = 0
        for line in lines[:hard_cap]:
            projected = total_chars + len(line) + 1
            if selected and len(selected) >= min_items and projected > char_budget:
                break
            selected.append(line)
            total_chars = projected
        return selected

    def _ground_chunk_extraction(
        self,
        chunk_kg: Dict[str, Any],
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach exact-span anchors and restoration metadata to one chunk extraction."""
        if not isinstance(chunk_kg, dict):
            return {"entities": [], "relationships": []}

        chunk_text = str(chunk.get("text") or "")
        chunk_start = int(chunk.get("start_pos") or 0)
        chunk_position = int(chunk.get("position", 0) or 0)
        dataset = chunk.get("dataset")
        question_id = chunk.get("question_id")
        passage_index = chunk.get("passage_index")
        source_label = chunk.get("source")
        source_title = chunk.get("source_title")
        source_scope_key = (
            chunk.get("source_scope_key")
            or self._build_passage_scope_key(
                dataset=dataset,
                question_id=question_id,
                passage_index=passage_index,
            )
        )

        grounded_entities: List[Dict[str, Any]] = []
        entity_by_id: Dict[str, Dict[str, Any]] = {}

        for entity in chunk_kg.get("entities", []) or []:
            if not isinstance(entity, dict):
                continue
            entity_copy = dict(entity)
            properties = dict(entity_copy.get("properties") or {})
            anchor_spans = self._find_exact_text_spans(
                chunk_text,
                self._entity_candidate_names(entity_copy),
                start_offset=chunk_start,
                chunk_position=chunk_position,
            )
            if anchor_spans:
                properties["anchor_spans"] = self._merge_anchor_spans(
                    properties.get("anchor_spans"),
                    anchor_spans,
                )
                properties["anchor_mention_count"] = len(properties["anchor_spans"])
            if dataset:
                properties.setdefault("dataset", dataset)
            if question_id is not None:
                properties.setdefault("question_id", str(question_id))
            if passage_index is not None:
                properties.setdefault("passage_index", int(passage_index))
            if source_label:
                properties.setdefault("source", str(source_label))
            if source_title:
                properties.setdefault("source_title", str(source_title))
            if source_scope_key:
                properties.setdefault("source_scope_key", str(source_scope_key))
            entity_copy["properties"] = properties
            grounded_entities.append(entity_copy)
            entity_id = entity_copy.get("id")
            if isinstance(entity_id, str) and entity_id.strip():
                entity_by_id[entity_id] = entity_copy

        grounded_relationships: List[Dict[str, Any]] = []
        for rel in chunk_kg.get("relationships", []) or []:
            if not isinstance(rel, dict):
                continue
            rel_copy = dict(rel)
            properties = dict(rel_copy.get("properties") or {})

            source_entity = entity_by_id.get(str(rel_copy.get("source") or ""))
            target_entity = entity_by_id.get(str(rel_copy.get("target") or ""))
            source_spans = self._merge_anchor_spans(
                properties.get("source_anchor_spans"),
                (source_entity.get("properties") or {}).get("anchor_spans") if source_entity else None,
                self._find_exact_text_spans(
                    chunk_text,
                    [str(rel_copy.get("source") or "")],
                    start_offset=chunk_start,
                    chunk_position=chunk_position,
                ),
            )
            target_spans = self._merge_anchor_spans(
                properties.get("target_anchor_spans"),
                (target_entity.get("properties") or {}).get("anchor_spans") if target_entity else None,
                self._find_exact_text_spans(
                    chunk_text,
                    [str(rel_copy.get("target") or "")],
                    start_offset=chunk_start,
                    chunk_position=chunk_position,
                ),
            )
            relation_spans = self._merge_anchor_spans(
                properties.get("relation_anchor_spans"),
                self._find_exact_text_spans(
                    chunk_text,
                    self._relationship_anchor_candidates(rel_copy),
                    start_offset=chunk_start,
                    chunk_position=chunk_position,
                ),
            )
            condition_spans = self._merge_anchor_spans(
                properties.get("condition_anchor_spans"),
                self._find_exact_text_spans(
                    chunk_text,
                    [str(properties.get("condition") or "")],
                    start_offset=chunk_start,
                    chunk_position=chunk_position,
                ),
            )
            quantitative_spans = self._merge_anchor_spans(
                properties.get("quantitative_anchor_spans"),
                self._find_exact_text_spans(
                    chunk_text,
                    [str(properties.get("quantitative") or "")],
                    start_offset=chunk_start,
                    chunk_position=chunk_position,
                ),
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
            properties["anchor_grounding"] = anchor_grounding
            properties["source_anchor_spans"] = source_spans
            properties["target_anchor_spans"] = target_spans
            properties["relation_anchor_spans"] = relation_spans
            if condition_spans:
                properties["condition_anchor_spans"] = condition_spans
            if quantitative_spans:
                properties["quantitative_anchor_spans"] = quantitative_spans
            properties["restoration_status"] = restoration["status"]
            properties["restoration_verified"] = restoration["verified"]
            properties["restoration_grounded_components"] = restoration["grounded_components"]
            properties["restoration_grounded_count"] = restoration["grounded_count"]
            if dataset:
                properties.setdefault("dataset", dataset)
            if question_id is not None:
                properties.setdefault("question_id", str(question_id))
            if passage_index is not None:
                properties.setdefault("passage_index", int(passage_index))
            if source_label:
                properties.setdefault("source", str(source_label))
            if source_title:
                properties.setdefault("source_title", str(source_title))
            if source_scope_key:
                properties.setdefault("source_scope_key", str(source_scope_key))
            rel_copy["properties"] = properties
            grounded_relationships.append(rel_copy)

        return {
            "entities": grounded_entities,
            "relationships": grounded_relationships,
        }

    def _extract_qualifier_sentences(self, text: str, max_sentences: int = 4) -> str:
        """Return up to *max_sentences* sentences from *text* that contain
        qualifier / experimental-context keywords.  Used to build the
        "context from previous chunk" header.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        selected = [s.strip() for s in sentences if self._QUALIFIER_KEYWORDS.search(s)]
        # Prefer sentences near the end of the chunk (most recent context)
        selected = selected[-max_sentences:]
        return " ".join(selected)

    def _resolve_coreferences_llm(
        self,
        chunk_text: str,
        context_text: str,
        llm,
        model_name: str,
    ) -> str:
        """Use the LLM to resolve demonstrative coreferences in *chunk_text*
        using *context_text* (content from previous chunks) as the lookup source.

        Returns a rewritten version of *chunk_text* with all resolvable references
        replaced by their full referents.  Falls back to the original text on any
        error.

        This is the cross-chunk qualifier coreference step: phrases like
        "these conditions", "this model", "the treated cells" are replaced with
        the specific experimental setup described in *context_text*, so that the
        downstream extraction LLM can attach the correct qualifier to each claim.
        """
        if not context_text.strip():
            return chunk_text

        coref_prompt = f"""You are resolving ambiguous references in a biomedical research text.

CONTEXT (from earlier in the paper — use this to identify what demonstrative phrases refer to):
{context_text}

CURRENT PASSAGE (rewrite this passage, replacing every ambiguous demonstrative reference with its full referent from the CONTEXT):
{chunk_text}

Rules:
- Replace "these conditions" / "this model" / "the treated cells" / "such circumstances" etc. with their specific referents from the CONTEXT.
- If a reference cannot be resolved from the CONTEXT, leave it unchanged.
- Do NOT change any scientific claims, entity names, or factual content.
- Return ONLY the rewritten passage text, nothing else."""

        try:
            resolved = llm.generate(coref_prompt, "", model_name)
            resolved = resolved.strip()
            if len(resolved) < 20 or len(resolved) > len(chunk_text) * 3:
                # Sanity check: resolved text must be plausible
                return chunk_text
            return resolved
        except Exception as e:
            logging.warning("Coreference resolution failed, using original chunk: %s", e)
            return chunk_text

    def _chunk_text_with_section_boundaries(
        self,
        text: str,
        *,
        section_headers: Optional[List[Tuple[int, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk text without crossing detected section boundaries when possible."""
        headers = list(section_headers or self._detect_section_headers(text))
        if not headers:
            return self._chunk_text(text)

        boundaries = [0]
        for pos, _ in headers:
            if pos not in boundaries and 0 < pos < len(text):
                boundaries.append(pos)
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))

        if len(boundaries) <= 2:
            return self._chunk_text(text)

        section_chunks: List[Dict[str, Any]] = []
        for start, end in zip(boundaries, boundaries[1:]):
            segment_text = text[start:end]
            if not segment_text.strip():
                continue
            local_chunks = _chunk_text_fn(
                text=segment_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedding_fn=self.embedding_function.embed_query,
            )
            for local_chunk in local_chunks:
                chunk_copy = dict(local_chunk)
                chunk_copy["start_pos"] = start + int(local_chunk.get("start_pos") or 0)
                chunk_copy["end_pos"] = start + int(local_chunk.get("end_pos") or 0)
                chunk_copy["position"] = len(section_chunks)
                section_chunks.append(chunk_copy)

        return section_chunks or self._chunk_text(text)

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using RecursiveCharacterTextSplitter with tiktoken encoding.

        Delegates to ontographrag.kg.chunking.chunk_text so the logic is
        testable in isolation without instantiating the full creator.
        """
        return _chunk_text_fn(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_fn=self.embedding_function.embed_query,
        )

    def _prepare_chunk_text_for_extraction(
        self,
        chunk_text: str,
        *,
        previous_entities: Optional[List[Dict[str, Any]]] = None,
        previous_texts: Optional[List[str]] = None,
        llm=None,
        model_name: str = "openai/gpt-oss-120b:free",
    ) -> str:
        """Resolve cheap coreference first, then optional LLM coref on leftovers."""
        extraction_text = chunk_text
        if not extraction_text or not self._has_coreference_markers(extraction_text):
            return extraction_text

        if (
            getattr(self, "enable_heuristic_coreference_resolution", True)
            and previous_entities
        ):
            heuristic_text = self._resolve_coreferences_heuristic(
                extraction_text,
                previous_entities,
            )
            if heuristic_text != extraction_text:
                logging.info("Applied heuristic coreference rewrite before extraction")
                extraction_text = heuristic_text

        if (
            self.enable_coreference_resolution
            and llm is not None
            and self._has_coreference_markers(extraction_text)
        ):
            coref_context = "\n\n".join(
                text for text in (previous_texts or []) if isinstance(text, str) and text.strip()
            )
            if coref_context:
                logging.info("Running LLM coreference resolution on remaining markers")
                extraction_text = self._resolve_coreferences_llm(
                    extraction_text,
                    coref_context,
                    llm,
                    model_name,
                )
                time.sleep(0.5)

        return extraction_text

    def _build_retrieval_subchunks(
        self,
        chunk: Dict[str, Any],
        *,
        parent_chunk_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Build smaller retrieval-only spans from a larger KG extraction chunk.

        KG extraction keeps broad context, but vector retrieval should operate on
        spans sized for the embedding model. These RetrievalChunk nodes map back
        to their parent Chunk so graph provenance stays intact.
        """
        chunk_text = str(chunk.get("text") or "")
        if not chunk_text.strip():
            return []

        retrieval_chunk_size = max(64, int(getattr(self, "retrieval_chunk_size", 256) or 256))
        retrieval_chunk_overlap = max(
            0,
            min(
                int(getattr(self, "retrieval_chunk_overlap", 64) or 64),
                retrieval_chunk_size - 1,
            ),
        )

        raw_subchunks = _chunk_text_fn(
            text=chunk_text,
            chunk_size=retrieval_chunk_size,
            chunk_overlap=retrieval_chunk_overlap,
            embedding_fn=self.embedding_function.embed_query,
        )
        retrieval_subchunks: List[Dict[str, Any]] = []
        parent_start = int(chunk.get("start_pos") or 0)
        for subchunk in raw_subchunks:
            retrieval_subchunks.append({
                "id": hashlib.sha1(
                    f"{parent_chunk_id}:{subchunk['position']}:{subchunk['text']}".encode()
                ).hexdigest(),
                "text": subchunk["text"],
                "embedding": subchunk.get("embedding"),
                "retrieval_local_index": subchunk["position"],
                "start_pos": parent_start + int(subchunk.get("start_pos") or 0),
                "end_pos": parent_start + int(subchunk.get("end_pos") or 0),
                "position": int(chunk.get("position") or 0),
                "source": chunk.get("source"),
                "dataset": chunk.get("dataset"),
                "question_id": chunk.get("question_id"),
                "passage_index": chunk.get("passage_index"),
                "chunk_local_index": chunk.get("chunk_local_index", chunk.get("chunk_id", 0)),
                "parent_chunk_id": parent_chunk_id,
            })
        return retrieval_subchunks

    def _build_ontology_prompt_section(self, chunk_text: str, max_entity_types: int = 30, max_rel_types: int = 25) -> str:
        """Build the ontology section of the extraction prompt.

        When a typed OntologySchema is available, emits:
          - entity types with description + typed property list
          - relationship types with description + domain/range/cardinality + attributes

        Uses a two-stage strategy on typed ontologies:
          1. rank entity/relationship types by chunk-local relevance
          2. render the highest-ranked lines until a prompt-size budget is reached

        Falls back to the flat list format when _ontology_schema is not available.
        """
        schema = self._ontology_schema

        # ---- entity types ----
        if schema and schema.entity_types:
            ranked_entity_types = self._rank_entity_types_for_prompt(
                chunk_text,
                schema.entity_types,
            )
            ranked_entities = [et for _, _, et in ranked_entity_types]
            entity_hard_cap = self._adaptive_prompt_item_cap(
                ranked_entity_types,
                base_cap=max_entity_types,
                absolute_cap=80,
            )

            entity_line_pairs = []
            for et in ranked_entities:
                desc = f" — {et.description}" if et.description else ""
                line = f"  {et.id}{desc}"
                if et.properties:
                    prop_parts = []
                    for p in et.properties:
                        pt = p.type.value
                        extra = ""
                        if p.type == PropertyType.ENUM and p.enum_values:
                            extra = f" [{', '.join(p.enum_values)}]"
                        if p.unit:
                            extra += f" ({p.unit})"
                        flag = " [identifier]" if p.identifier else (" [required]" if p.required else "")
                        prop_parts.append(f"{p.name}:{pt}{extra}{flag}")
                    line += f"\n      properties: {', '.join(prop_parts)}"
                entity_line_pairs.append((et, line))
            et_lines = self._render_prompt_lines_with_budget(
                [line for _, line in entity_line_pairs],
                char_budget=4200,
                min_items=min(12, len(entity_line_pairs)),
                hard_cap=entity_hard_cap,
            )
            selected_et = [et for et, line in entity_line_pairs if line in et_lines]
            entity_section = "ENTITY TYPES (id — description; properties with types):\n" + "\n".join(et_lines)
        else:
            lines = [
                f"  {c['id']} — {c.get('description') or c.get('label', '')}"
                for c in self.ontology_classes[:max_entity_types]
                if isinstance(c, dict)
            ]
            entity_section = "ENTITY TYPES:\n" + "\n".join(lines)

        # ---- relationship types ----
        if schema and schema.relationship_types:
            selected_relationship_types = self._select_relationship_types_for_prompt(
                chunk_text,
                selected_et if schema and schema.entity_types else [],
                self._adaptive_prompt_item_cap(
                    [
                        (
                            (4 if (rt.id.lower() in chunk_text.lower() or rt.label.lower() in chunk_text.lower()) else 0)
                            + (2 if rt.domain and any(rt.domain == et.id for et in selected_et) else 0)
                            + (2 if rt.range and any(rt.range == et.id for et in selected_et) else 0),
                            idx,
                            rt,
                        )
                        for idx, rt in enumerate(schema.relationship_types)
                    ],
                    base_cap=max_rel_types,
                    absolute_cap=60,
                ),
            )
            rel_line_pairs = []
            for rt in selected_relationship_types:
                desc = f" — {rt.description}" if rt.description else ""
                dom_rng = ""
                if rt.domain or rt.range:
                    dom_rng = f" ({rt.domain or '?'} → {rt.range or '?'})"
                    if rt.cardinality:
                        dom_rng += f" [{rt.cardinality}]"
                line = f"  {rt.id}{desc}{dom_rng}"
                if rt.attributes:
                    attr_parts = [
                        f"{a.name}:{a.type.value}" + (f" ({a.unit})" if a.unit else "")
                        for a in rt.attributes
                    ]
                    line += f"\n      attributes: {', '.join(attr_parts)}"
                rel_line_pairs.append(line)
            rel_lines = self._render_prompt_lines_with_budget(
                rel_line_pairs,
                char_budget=3200,
                min_items=min(10, len(rel_line_pairs)),
                hard_cap=len(rel_line_pairs),
            )
            rel_section = "RELATIONSHIP TYPES (id — description; domain→range [cardinality]):\n" + "\n".join(rel_lines)
        else:
            lines = [
                f"  {r['id']} — {r.get('description') or r.get('label', '')}"
                + (f" ({r.get('domain', '')} → {r.get('range', '')})" if r.get('domain') or r.get('range') else "")
                for r in self.ontology_relationships[:max_rel_types]
                if isinstance(r, dict)
            ]
            rel_section = "RELATIONSHIP TYPES:\n" + "\n".join(lines)

        return entity_section + "\n\n" + rel_section

    def _extract_entities_and_relationships_with_self_consistency(
        self,
        chunk_text: str,
        llm,
        *,
        model_name: str,
        context_header: Optional[str],
        section_header: Optional[str],
    ) -> Dict[str, Any]:
        """Run multiple independent extractions and keep majority-supported items."""
        num_samples = max(1, int(getattr(self, "self_consistency_n", 1) or 1))
        samples: List[Dict[str, Any]] = []
        for _ in range(num_samples):
            samples.append(
                self._extract_entities_and_relationships_with_llm(
                    chunk_text,
                    llm,
                    model_name=model_name,
                    context_header=context_header,
                    section_header=section_header,
                    _self_consistency_depth=1,
                )
            )

        threshold = (num_samples // 2) + 1
        entity_votes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            seen_entity_keys = set()
            for entity in sample.get("entities", []) or []:
                if not isinstance(entity, dict):
                    continue
                key = self._normalize_entity_text(entity.get("id") or entity.get("name") or "")
                if not key or key in seen_entity_keys:
                    continue
                entity_votes[key].append(entity)
                seen_entity_keys.add(key)

        kept_entities: List[Dict[str, Any]] = []
        entity_alias_map: Dict[str, str] = {}
        for key, voted_entities in entity_votes.items():
            if len(voted_entities) < threshold:
                continue
            representative = max(
                voted_entities,
                key=lambda e: (
                    len(str((e.get("properties") or {}).get("description") or "")),
                    len(str(e.get("id") or "")),
                ),
            ).copy()
            type_counts = Counter(str(e.get("type") or "Concept") for e in voted_entities)
            representative["type"] = type_counts.most_common(1)[0][0]
            properties = dict(representative.get("properties") or {})
            all_names = {
                str(e.get("id") or "").strip()
                for e in voted_entities
                if str(e.get("id") or "").strip()
            }
            if len(all_names) > 1:
                properties["all_names"] = sorted(all_names)
            merged_anchor_spans = self._merge_anchor_spans(
                *[
                    (entity.get("properties") or {}).get("anchor_spans")
                    for entity in voted_entities
                ]
            )
            if merged_anchor_spans:
                properties["anchor_spans"] = merged_anchor_spans
                properties["anchor_mention_count"] = len(merged_anchor_spans)
            representative["properties"] = properties
            kept_entities.append(representative)
            for alias in all_names or {str(representative.get("id") or "").strip()}:
                alias_key = self._normalize_entity_text(alias)
                if alias_key:
                    entity_alias_map[alias_key] = str(representative.get("id") or alias)

        entity_type_by_name = {
            str(entity.get("id") or ""): str(entity.get("type") or "")
            for entity in kept_entities
        }
        relationship_votes: Dict[Tuple[str, str, str, bool], List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            seen_relationship_keys = set()
            for rel in sample.get("relationships", []) or []:
                if not isinstance(rel, dict):
                    continue
                src_key = self._normalize_entity_text(rel.get("source", ""))
                tgt_key = self._normalize_entity_text(rel.get("target", ""))
                if src_key not in entity_alias_map or tgt_key not in entity_alias_map:
                    continue
                canonical_src = entity_alias_map[src_key]
                canonical_tgt = entity_alias_map[tgt_key]
                canonical_type = self._canonicalize_relationship_type(
                    rel.get("type", ""),
                    source_type=entity_type_by_name.get(canonical_src),
                    target_type=entity_type_by_name.get(canonical_tgt),
                ) or str(rel.get("type") or "").strip().replace(" ", "_").replace("-", "_").upper()
                vote_key = (
                    canonical_src,
                    canonical_type,
                    canonical_tgt,
                    bool(rel.get("negated", False)),
                )
                if vote_key in seen_relationship_keys:
                    continue
                rel_copy = dict(rel)
                rel_copy["source"] = canonical_src
                rel_copy["target"] = canonical_tgt
                rel_copy["type"] = canonical_type
                relationship_votes[vote_key].append(rel_copy)
                seen_relationship_keys.add(vote_key)

        kept_relationships: List[Dict[str, Any]] = []
        for vote_key, voted_relationships in relationship_votes.items():
            if len(voted_relationships) < threshold:
                continue
            representative = max(
                voted_relationships,
                key=lambda rel: len(str((rel.get("properties") or {}).get("description") or "")),
            ).copy()
            representative_properties = dict(representative.get("properties") or {})
            provenance = sorted(
                {
                    int(pos)
                    for rel in voted_relationships
                    for pos in (rel.get("provenance_positions") or [])
                    if isinstance(pos, (int, float))
                }
            )
            if provenance:
                representative["provenance_positions"] = provenance
            merged_anchor_grounding = self._merge_anchor_grounding(
                *[
                    (rel.get("properties") or {}).get("anchor_grounding")
                    for rel in voted_relationships
                ]
            )
            if merged_anchor_grounding:
                representative_properties["anchor_grounding"] = merged_anchor_grounding
                restoration = self._restoration_from_anchor_grounding(merged_anchor_grounding)
                representative_properties["restoration_status"] = restoration["status"]
                representative_properties["restoration_verified"] = restoration["verified"]
                representative_properties["restoration_grounded_components"] = restoration["grounded_components"]
                representative_properties["restoration_grounded_count"] = restoration["grounded_count"]
            representative["properties"] = representative_properties
            kept_relationships.append(representative)

        return {
            "entities": kept_entities,
            "relationships": kept_relationships,
        }

    def _extract_entities_and_relationships_with_llm(
        self,
        chunk_text: str,
        llm,
        model_name: str = "openai/gpt-oss-120b:free",
        context_header: str = None,
        section_header: str = None,
        _self_consistency_depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM with ontology guidance (if ontology
        available) or natural LLM detection.

        Args:
            chunk_text:      The text chunk to extract from.
            llm:             LLM provider instance.
            model_name:      Model identifier.
            context_header:  Qualifier sentences from the previous chunk, injected
                             before the main text so the LLM can resolve cross-chunk
                             experimental conditions.
            section_header:  Paper section the chunk belongs to (e.g. "Methods",
                             "Results").  Helps the LLM interpret claims correctly.
        """
        if llm is None:
            logging.warning("_extract_entities_and_relationships_with_llm called with llm=None; returning empty.")
            return {"entities": [], "relationships": []}

        if getattr(self, "self_consistency_n", 1) > 1 and _self_consistency_depth == 0:
            return self._extract_entities_and_relationships_with_self_consistency(
                chunk_text,
                llm,
                model_name=model_name,
                context_header=context_header,
                section_header=section_header,
            )

        # Build context preamble from section header + qualifier context.
        # This is injected before the main chunk text so the LLM can resolve
        # cross-chunk experimental conditions and interpret claims correctly.
        context_preamble = ""
        if section_header:
            context_preamble += (
                f"[DOCUMENT SECTION: {section_header}]\n"
                f"Interpret claims in the context of a {section_header} section "
                f"(e.g. experimental setups in Methods, findings reported in Results).\n\n"
            )
        if context_header:
            context_preamble += (
                f"[QUALIFIER CONTEXT FROM PREVIOUS CHUNK]\n"
                f"The following experimental conditions were established earlier in the paper. "
                f"Use them to resolve any ambiguous references (e.g. 'these conditions', "
                f"'this model') in the text below and attach them as qualifiers where relevant:\n"
                f"{context_header}\n\n"
            )

        # ------------------------------------------------------------------
        # Build the ontology section of the system prompt.
        # When a typed OntologySchema is available we emit schema-aware text
        # (entity descriptions + allowed properties, relationship domain/range/
        # cardinality/attributes).  We limit to the most relevant subset so the
        # prompt stays compact: entity types whose label appears in the chunk
        # text are always included; the rest are included up to a cap.
        # ------------------------------------------------------------------
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)
        ontology_section = ""

        if has_ontology:
            few_shot_examples = self._build_ontology_few_shot_examples()
            few_shot_block = ""
            if few_shot_examples:
                few_shot_block = f"FEW-SHOT EXAMPLES:\n{few_shot_examples}\n"
            try:
                ontology_section = self._build_ontology_prompt_section(chunk_text)
            except Exception as _oe:
                logging.warning("Failed to build schema-aware ontology prompt section: %s", _oe)
                ontology_section = (
                    "ENTITY TYPES:\n"
                    + "\n".join(f"- {c['label']} ({c['id']})" for c in self.ontology_classes[:100] if isinstance(c, dict))
                    + "\n\nRELATIONSHIP TYPES:\n"
                    + "\n".join(f"- {r['label']} ({r['id']})" for r in self.ontology_relationships[:50] if isinstance(r, dict))
                )

            system_message = f"""
You are an expert ontology-guided knowledge graph extraction system.
Extract entities and relationships from text using the schema below.

{ontology_section}

{few_shot_block}

INSTRUCTIONS:
1. Extract specific named entities only — proper nouns, technical terms, gene/protein/drug/disease names,
   defined concepts with a precise referent. Do NOT extract generic process words used as nouns
   (e.g. "treatment", "condition", "outcome", "model", "effect", "result", "factor", "mechanism",
   "response", "function", "role", "process", "level", "system", "study", "analysis", "approach",
   "method", "measure", "group", "sample", "data", "finding", "evidence", "activity", "expression",
   "production", "change", "increase", "decrease", "type", "form", "stage", "state", "case").
   These become hub nodes that fan out to irrelevant chunks during retrieval.
2. Entity names must be at least 3 characters long after stripping whitespace; skip single letters or
   bare numbers unless they are a named identifier (e.g. "p53", "IL-6", "5-HT").
3. For each entity, populate the typed properties defined in the schema (dates, quantities, enums, IDs)
4. Create relationships ONLY between entities that actually interact in the text; use schema relationship types when they match
5. Prefer relationship types whose domain/range match the source/target entity types
6. Include specific named entities and technical terms — these are often the answer to downstream questions
7. Ignore pure function words, filler phrases, generic pronouns, and the generic process words listed in rule 1

Return ONLY a valid JSON object.
IMPORTANT: Output "relationships" FIRST, then "entities".
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "anchor_text": "exact supporting relation phrase from the text, or null",
        "description": "how they are related in the text",
        "condition": "condition constraining this claim, or null",
        "quantitative": "numerical finding attached to this relationship, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ],
  "entities": [
    {{
      "id": "canonical_entity_name",
      "type": "EntityType",
      "properties": {{
        "name": "canonical_entity_name",
        "description": "brief grounded description",
        "aliases": ["alternate name or abbreviation from text if present, else omit"]
      }}
    }}
  ]
}}

NEGATION RULE: If the text states a relationship does NOT hold, set "negated": true and keep the positive form of the type.
GROUNDING RULE: When possible, set "anchor_text" to the shortest exact phrase from the text that expresses the relationship.
QUALIFIER RULE: Conditions (e.g. "in ALS patients") go in "condition"; numerical findings (e.g. "3-fold increase") go in "quantitative".
ALIAS RULE: If the text introduces a shorter name, acronym, or alternate form for an entity (e.g. "United States (US)", "TBK1 kinase (TBK1)"), list it in aliases. If no alias is present, omit the aliases field entirely.

{context_preamble}TEXT TO ANALYZE:
{chunk_text}

IMPORTANT: Return ONLY the JSON object, no additional text."""
        else:
            system_message = f"""
You are an expert knowledge graph extraction system.
Extract entities and relationships from text naturally and comprehensively.

INSTRUCTIONS:
1. Extract specific named entities only — proper nouns, technical terms, defined concepts with a
   precise referent. Do NOT extract generic process words used as nouns such as: treatment,
   condition, outcome, model, effect, result, factor, mechanism, response, function, role,
   process, level, system, study, analysis, approach, method, measure, group, sample, data,
   finding, evidence, activity, expression, change, increase, decrease, type, form, stage, state.
   These become hub nodes that pollute graph traversal with irrelevant results.
2. Entity names must be at least 3 characters; skip bare letters or plain numbers unless they are
   named identifiers (e.g. "p53", "IL-6").
3. Create relationships between any meaningfully related entities
4. Use descriptive relationship types that capture how entities interact
5. Be precise over comprehensive; one well-grounded entity is better than five vague ones

Return ONLY a valid JSON object.
IMPORTANT: Output "relationships" FIRST, then "entities".
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "anchor_text": "exact supporting relation phrase from the text, or null",
        "description": "how they are related in the text",
        "condition": "condition constraining this claim, or null",
        "quantitative": "any numerical finding, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ],
  "entities": [
    {{
      "id": "canonical_entity_name",
      "type": "EntityType",
      "properties": {{
        "name": "canonical_entity_name",
        "description": "contextual description",
        "aliases": ["alternate name or abbreviation from text if present, else omit"]
      }}
    }}
  ]
}}

NEGATION RULE: If the text states a relationship does NOT hold, set "negated": true and keep the positive form.
GROUNDING RULE: When possible, set "anchor_text" to the shortest exact phrase from the text that expresses the relationship.
QUALIFIER RULE: Conditions go in "condition"; numerical findings go in "quantitative".
ALIAS RULE: If the text introduces a shorter name, acronym, or alternate form for an entity (e.g. "United States (US)"), list it in aliases. If no alias is present, omit the aliases field.

{context_preamble}TEXT TO ANALYZE:
{chunk_text}

IMPORTANT: Return ONLY the JSON object, no additional text."""

        try:
            if self.enable_anchor_constrained_extraction:
                anchored_result = self._extract_entities_and_relationships_with_anchor_constraints(
                    chunk_text,
                    llm,
                    model_name=model_name,
                    context_preamble=context_preamble,
                    ontology_section=ontology_section,
                    has_ontology=has_ontology,
                )
                if anchored_result is not None:
                    return anchored_result

            try:
                response = llm.generate(system_message, "", model_name)
            except Exception as timeout_error:
                if "timeout" in str(timeout_error).lower() or "read operation timed out" in str(timeout_error).lower():
                    logging.warning("LLM request timed out for chunk. Returning empty result to continue processing.")
                    return {'entities': [], 'relationships': []}
                raise timeout_error

            logging.info("Raw LLM response length: %d", len(str(response)))
            logging.info("Raw LLM response preview: %s...", str(response)[:200])
            result = self._parse_llm_json_response(
                response,
                allow_partial_extraction=True,
                empty_result={"entities": [], "relationships": []},
            )
            return self._postprocess_extraction_result(
                result,
                chunk_text=chunk_text,
                llm=llm,
                model_name=model_name,
            )
        except Exception as e:
            logging.error(f"LLM extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def _partial_json_extract(self, text: str) -> Dict[str, Any]:
        """
        Best-effort recovery when json.loads() fails on an LLM response.

        Tries three strategies in order:
        1. Extract the 'entities' array via regex and parse it independently.
        2. Extract the 'relationships' array via regex and parse it independently.
        3. Return empty dict (caller will fall back to empty entities/relationships).

        This prevents silently discarding an entire chunk just because a trailing
        comma or stray character made the top-level JSON invalid.
        """
        def _extract_array(key: str, src: str):
            """Regex-extract the JSON array for a given key from a potentially broken JSON string."""
            pattern = rf'"{key}"\s*:\s*(\[.*?\])'
            match = re.search(pattern, src, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                # Try to clean common issues: trailing commas before ]
                cleaned = re.sub(r',\s*]', ']', match.group(1))
                cleaned = re.sub(r',\s*}', '}', cleaned)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None

        entities = _extract_array('entities', text)
        relationships = _extract_array('relationships', text)

        if entities is not None or relationships is not None:
            recovered = {
                'entities': entities if isinstance(entities, list) else [],
                'relationships': relationships if isinstance(relationships, list) else [],
            }
            logging.info(
                f"Partial JSON recovery succeeded: {len(recovered['entities'])} entities, "
                f"{len(recovered['relationships'])} relationships"
            )
            return recovered

        logging.warning("Partial JSON recovery failed; returning empty extraction for this chunk.")
        return {'entities': [], 'relationships': []}

    def _parse_llm_json_response(
        self,
        response: Any,
        *,
        allow_partial_extraction: bool = False,
        empty_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse a JSON object from an LLM response with fence/brace recovery."""
        if empty_result is None:
            empty_result = {}
        if response is None:
            return dict(empty_result)

        response_text = str(response).strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        json_start = response_text.find('{')
        if json_start == -1:
            return dict(empty_result)

        json_content = ""
        brace_count = 0
        json_end = json_start
        in_string = False
        escape_next = False

        for i, char in enumerate(response_text[json_start:], json_start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

        if brace_count == 0 and json_end > json_start:
            json_content = response_text[json_start:json_end]
        else:
            json_end = response_text.rfind('}') + 1
            if json_end > json_start:
                json_content = response_text[json_start:json_end]
            else:
                return dict(empty_result)

        try:
            parsed = json.loads(json_content)
        except json.JSONDecodeError:
            if allow_partial_extraction:
                parsed = self._partial_json_extract(json_content)
            else:
                return dict(empty_result)

        if not isinstance(parsed, dict):
            return dict(empty_result)
        return parsed

    def _build_anchor_discovery_prompt(
        self,
        *,
        chunk_text: str,
        context_preamble: str,
        ontology_section: str,
        has_ontology: bool,
    ) -> str:
        """Stage 1 of anchor-constrained extraction: discover exact text anchors."""
        schema_block = f"{ontology_section}\n\n" if ontology_section else ""
        schema_rules = ""
        if has_ontology:
            schema_rules = (
                "Use the schema to type entity anchors and to propose relation type hints.\n"
                "If unsure about an entity type, choose the closest schema type rather than inventing one.\n"
            )

        return f"""
You are discovering grounded knowledge anchors for a knowledge graph extraction pipeline.

{schema_block}{schema_rules}
TASK:
Identify exact text-grounded anchors from the text below.

Return ONLY a JSON object with three arrays:
{{
  "entity_anchors": [
    {{"text": "exact entity span from text", "type": "EntityType"}}
  ],
  "relation_anchors": [
    {{"text": "exact relation phrase from text", "type_hint": "RELATIONSHIP_TYPE"}}
  ],
  "attribute_anchors": [
    {{"text": "exact attribute/value span from text", "role": "condition|quantitative|other"}}
  ]
}}

Rules:
- Every anchor text MUST be an exact substring of the source text.
- Do NOT paraphrase or normalize anchor text.
- Prefer precise named entities and technical terms.
- Skip generic hub terms such as treatment, condition, result, factor, process, study, analysis, method, group, sample, data, evidence, activity, change, increase, decrease, state.
- Relation anchors should be the shortest exact phrase that expresses the relation.
- Attribute anchors should capture exact qualifiers such as patient groups, experimental conditions, dates, doses, and numerical findings.
- If an array is empty, return [] for it.

{context_preamble}TEXT TO ANALYZE:
{chunk_text}

Return ONLY the JSON object.
"""

    def _build_anchor_constrained_extraction_prompt(
        self,
        *,
        chunk_text: str,
        context_preamble: str,
        ontology_section: str,
        has_ontology: bool,
        anchor_inventory: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """Stage 2 of anchor-constrained extraction: compose triples only from anchors."""
        schema_block = f"{ontology_section}\n\n" if ontology_section else ""
        entity_lines = "\n".join(
            f"- {anchor['text']} (type: {anchor['type']})"
            for anchor in anchor_inventory.get("entity_anchors", [])
        ) or "- none"
        relation_lines = "\n".join(
            f"- {anchor['text']} (type hint: {anchor['type_hint']})"
            for anchor in anchor_inventory.get("relation_anchors", [])
        ) or "- none"
        attribute_lines = "\n".join(
            f"- {anchor['text']} (role: {anchor['role']})"
            for anchor in anchor_inventory.get("attribute_anchors", [])
        ) or "- none"

        schema_rules = ""
        if has_ontology:
            schema_rules = (
                "Use schema-compatible relationship types and prefer domain/range-consistent choices.\n"
            )

        return f"""
You are performing grounded knowledge graph extraction from a closed anchor inventory.

{schema_block}{schema_rules}
AVAILABLE ENTITY ANCHORS:
{entity_lines}

AVAILABLE RELATION ANCHORS:
{relation_lines}

AVAILABLE ATTRIBUTE ANCHORS:
{attribute_lines}

TASK:
Extract grounded entities and relationships using ONLY the anchors listed above.

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
        "condition": "exact attribute anchor text with role=condition, or null",
        "quantitative": "exact attribute anchor text with role=quantitative, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ],
  "entities": [
    {{
      "id": "exact entity anchor text",
      "type": "EntityType",
      "properties": {{
        "name": "exact entity anchor text",
        "description": "brief grounded description"
      }}
    }}
  ]
}}

Rules:
- source and target MUST be exact entity anchor texts from the inventory.
- anchor_text MUST be either an exact relation anchor text or null if no suitable relation anchor exists.
- condition and quantitative MUST be exact attribute anchor texts from the inventory or null.
- Do NOT invent any entity text, relation phrase, or attribute text not present in the anchor inventory.
- It is okay to omit weak or unsupported triples.
- Keep entities that participate in relations and any other anchors that are independently important.

{context_preamble}TEXT TO ANALYZE:
{chunk_text}

Return ONLY the JSON object.
"""

    def _extract_entities_and_relationships_with_anchor_constraints(
        self,
        chunk_text: str,
        llm,
        *,
        model_name: str,
        context_preamble: str,
        ontology_section: str,
        has_ontology: bool,
    ) -> Optional[Dict[str, Any]]:
        """Anchor-first KG extraction with constrained relation composition."""
        if llm is None:
            return None

        try:
            anchor_prompt = self._build_anchor_discovery_prompt(
                chunk_text=chunk_text,
                context_preamble=context_preamble,
                ontology_section=ontology_section,
                has_ontology=has_ontology,
            )
            anchor_response = llm.generate(anchor_prompt, "", model_name)
            raw_anchor_inventory = self._parse_llm_json_response(
                anchor_response,
                allow_partial_extraction=False,
                empty_result={
                    "entity_anchors": [],
                    "relation_anchors": [],
                    "attribute_anchors": [],
                },
            )
            anchor_inventory = self._normalize_anchor_inventory(
                raw_anchor_inventory,
                chunk_text=chunk_text,
            )
            if not anchor_inventory.get("entity_anchors"):
                return None

            grounded_prompt = self._build_anchor_constrained_extraction_prompt(
                chunk_text=chunk_text,
                context_preamble=context_preamble,
                ontology_section=ontology_section,
                has_ontology=has_ontology,
                anchor_inventory=anchor_inventory,
            )
            grounded_response = llm.generate(grounded_prompt, "", model_name)
            grounded_result = self._parse_llm_json_response(
                grounded_response,
                allow_partial_extraction=True,
                empty_result={"entities": [], "relationships": []},
            )
            grounded_result = self._augment_result_with_anchor_entities(
                grounded_result,
                anchor_inventory,
            )
            grounded_result = self._supplement_relationships_from_unused_anchors(
                grounded_result,
                chunk_text=chunk_text,
                llm=llm,
                model_name=model_name,
                context_preamble=context_preamble,
                anchor_inventory=anchor_inventory,
            )
            grounded_result = self._postprocess_extraction_result(
                grounded_result,
                chunk_text=chunk_text,
                llm=llm,
                model_name=model_name,
            )
            for entity in grounded_result.get("entities", []) or []:
                entity_props = dict(entity.get("properties") or {})
                anchor_key = self._normalize_entity_text(entity.get("id") or "")
                anchor_match = next(
                    (
                        anchor
                        for anchor in anchor_inventory.get("entity_anchors", [])
                        if self._normalize_entity_text(anchor["text"]) == anchor_key
                    ),
                    None,
                )
                if anchor_match:
                    entity_props["anchor_spans"] = self._merge_anchor_spans(
                        entity_props.get("anchor_spans"),
                        anchor_match.get("anchor_spans"),
                    )
                    entity_props["anchor_discovered"] = True
                    entity["properties"] = entity_props
            logging.info(
                "Anchor-constrained extraction succeeded: %d entity anchors, %d relation anchors, %d relationships",
                len(anchor_inventory.get("entity_anchors", [])),
                len(anchor_inventory.get("relation_anchors", [])),
                len(grounded_result.get("relationships", [])),
            )
            return grounded_result
        except Exception as exc:
            logging.warning("Anchor-constrained extraction failed; falling back to direct extraction: %s", exc)
            return None

    def _postprocess_extraction_result(
        self,
        result: Dict[str, Any],
        *,
        chunk_text: str,
        llm,
        model_name: str,
    ) -> Dict[str, Any]:
        """Normalize parsed extraction JSON into entity/relationship records."""
        if not isinstance(result, dict) or ('entities' not in result and 'relationships' not in result):
            logging.warning("Invalid response structure. Returning empty result.")
            return {'entities': [], 'relationships': []}
        result = dict(result)
        result.setdefault('entities', [])
        result.setdefault('relationships', [])

        medical_entities = []
        entities_raw = result.get('entities', [])
        if not isinstance(entities_raw, list):
            logging.warning(
                "LLM returned entities as %s instead of list: %s. Skipping chunk.",
                type(entities_raw),
                entities_raw,
            )
            return {'entities': [], 'relationships': []}

        for entity in entities_raw:
            if isinstance(entity, str):
                try:
                    entity_type = self._coerce_entity_type_with_ontology(None, entity)
                    if not entity_type:
                        logging.info("Skipping string entity with no schema-compatible type: %s", entity)
                        continue
                    entity = {
                        "id": entity,
                        "type": entity_type,
                        "properties": {
                            "name": entity,
                            "description": f"{entity_type}: {entity}"
                        }
                    }
                except Exception as e:
                    logging.warning("Error processing string entity '%s': %s. Skipping.", entity, e)
                    continue
            elif isinstance(entity, dict):
                if 'id' not in entity:
                    logging.warning("Entity missing 'id' field: %s. Skipping.", entity)
                    continue
                if not isinstance(entity['id'], str):
                    logging.warning(
                        "Entity id is not a string: %s = %s. Converting to string.",
                        type(entity['id']),
                        entity['id'],
                    )
                    entity['id'] = str(entity['id'])

                try:
                    entity_type = self._coerce_entity_type_with_ontology(
                        entity.get('type'),
                        entity.get('id'),
                    )
                    if not entity_type:
                        logging.info("Skipping entity with no schema-compatible type: %s", entity.get("id"))
                        continue
                    entity['type'] = entity_type
                    if 'properties' not in entity:
                        entity['properties'] = {
                            "name": entity['id'],
                            "description": f"{entity['type']}: {entity['id']}"
                        }
                except Exception as e:
                    logging.warning("Error processing dict entity %s: %s. Skipping.", entity, e)
                    continue
            else:
                logging.warning("Entity is neither string nor dict: %s = %s. Skipping.", type(entity), entity)
                continue

            medical_entities.append(entity)

        medical_relationships = []
        relationship_drop_count = 0
        relationship_fuzzy_resolutions = 0
        relationships_raw = result.get('relationships', [])

        if not isinstance(relationships_raw, list):
            logging.warning(
                "LLM returned relationships as %s instead of list: %s. Skipping relationships.",
                type(relationships_raw),
                relationships_raw,
            )
            medical_relationships = []
        else:
            for rel in relationships_raw:
                if isinstance(rel, str):
                    logging.warning("Relationship is string: %s. Skipping.", rel)
                    continue
                if not isinstance(rel, dict):
                    logging.warning("Relationship is neither string nor dict: %s = %s. Skipping.", type(rel), rel)
                    continue
                raw_src = rel.get('source', '')
                raw_tgt = rel.get('target', '')
                if not isinstance(raw_src, str) or not isinstance(raw_tgt, str):
                    logging.warning("Relationship source/target not strings: %s. Skipping.", rel)
                    continue
                canonical_src = self._resolve_relationship_endpoint(raw_src, medical_entities)
                canonical_tgt = self._resolve_relationship_endpoint(raw_tgt, medical_entities)
                if canonical_src and canonical_tgt:
                    rel_copy = dict(rel)
                    rel_copy['source'] = canonical_src
                    rel_copy['target'] = canonical_tgt
                    rel_props = dict(rel_copy.get('properties') or {})
                    rel_props.setdefault('source_name', raw_src)
                    rel_props.setdefault('target_name', raw_tgt)
                    rel_copy['properties'] = rel_props
                    if canonical_src != raw_src or canonical_tgt != raw_tgt:
                        relationship_fuzzy_resolutions += 1
                    medical_relationships.append(rel_copy)
                else:
                    relationship_drop_count += 1
                    logging.warning("Invalid relationship (source/target not in entities): %s. Skipping.", rel)

        if relationships_raw:
            logging.info(
                "Relationship endpoint resolution kept=%d dropped=%d fuzzy_resolved=%d",
                len(medical_relationships),
                relationship_drop_count,
                relationship_fuzzy_resolutions,
            )

        if not getattr(self, "enable_self_reflection", True):
            return {
                'entities': medical_entities,
                'relationships': medical_relationships
            }

        try:
            existing_entity_names = [e['id'] for e in medical_entities]
            reflection_prompt = f"""You are a quality-control agent for a knowledge graph extraction pipeline.

ORIGINAL TEXT:
{chunk_text}

ALREADY-EXTRACTED ENTITIES:
{json.dumps(existing_entity_names, indent=2)}

TASK: Read the original text carefully. Identify any named entities (people, organizations, locations, works, events, concepts, domain-specific terms, quantitative measurements) that are clearly present in the text but MISSING from the already-extracted list above.

Return ONLY a JSON object with the NEW missing entities (do not repeat already-extracted ones):
{{
  "new_entities": [
    {{
      "id": "exact_name_from_text",
      "type": "EntityType",
      "properties": {{
        "name": "exact_name_from_text",
        "description": "brief description"
      }}
    }}
  ]
}}

If there are no missing entities, return: {{"new_entities": []}}
Return ONLY the JSON object."""

            reflection_data = self._parse_llm_json_response(
                llm.generate(reflection_prompt, "", model_name),
                allow_partial_extraction=False,
                empty_result={"new_entities": []},
            )
            new_entities_raw = reflection_data.get('new_entities', [])
            existing_ids_lower = {e['id'].lower() for e in medical_entities}
            chunk_text_lower = chunk_text.lower()
            added = 0
            added_entity_ids = []
            for entity in new_entities_raw:
                if not isinstance(entity, dict) or 'id' not in entity:
                    continue
                if not isinstance(entity['id'], str):
                    entity['id'] = str(entity['id'])
                if entity['id'].lower() in existing_ids_lower:
                    continue
                if entity['id'].lower() not in chunk_text_lower:
                    logging.info(
                        "Self-reflection: skipping hallucinated entity not in chunk text: %s",
                        entity['id'],
                    )
                    continue
                entity_type = self._coerce_entity_type_with_ontology(
                    entity.get('type'),
                    entity['id'],
                )
                if not entity_type:
                    logging.info("Skipping reflected entity with no schema-compatible type: %s", entity['id'])
                    continue
                entity['type'] = entity_type
                entity.setdefault('properties', {'name': entity['id'], 'description': entity['type']})
                medical_entities.append(entity)
                existing_ids_lower.add(entity['id'].lower())
                added_entity_ids.append(entity['id'])
                added += 1
            if added:
                logging.info("Self-reflection added %d new entities", added)

            if added_entity_ids:
                all_known_ids = [e['id'] for e in medical_entities]
                recon_prompt = f"""You are extracting relationships for a knowledge graph.

TEXT:
{chunk_text}

NEW ENTITIES (just discovered — need relationships):
{json.dumps(added_entity_ids, indent=2)}

ALL KNOWN ENTITIES IN THIS CHUNK:
{json.dumps(all_known_ids, indent=2)}

TASK: Find ALL relationships in the text that involve at least one NEW ENTITY and any other known entity.

Return ONLY a JSON object:
{{
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "anchor_text": "exact supporting relation phrase from the text, or null",
        "description": "how they are related in the text",
        "condition": "condition constraining this claim, or null",
        "quantitative": "any numerical finding, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ]
}}

Rules:
- source and target MUST be entity ids from the ALL KNOWN ENTITIES list above
- At least one of source or target MUST be from the NEW ENTITIES list
- When possible, set "anchor_text" to the shortest exact phrase from the text that expresses the relationship
- If no relationships found, return {{"relationships": []}}
- Return ONLY the JSON object."""

                recon_data = self._parse_llm_json_response(
                    llm.generate(recon_prompt, "", model_name),
                    allow_partial_extraction=True,
                    empty_result={"relationships": []},
                )
                new_rels = recon_data.get('relationships', [])
                new_rels_added = 0
                for rel in new_rels:
                    if isinstance(rel, dict) and rel.get('source') and rel.get('target') and rel.get('type'):
                        medical_relationships.append(rel)
                        new_rels_added += 1
                if new_rels_added:
                    logging.info(
                        "Reflection reconciliation added %d relationships for new entities",
                        new_rels_added,
                    )
        except Exception as refl_err:
            logging.debug("Self-reflection pass failed (non-fatal): %s", refl_err)

        return {
            'entities': medical_entities,
            'relationships': medical_relationships
        }

    def _extract_relationships_only(self, combined_text: str, known_entities: List[Dict], llm, model_name: str) -> List[Dict]:
        """
        Given a combined text (two adjacent chunks) and entities already known from those chunks,
        ask the LLM only for relationships between those entities — no new entity extraction.

        Returns a list of raw relationship dicts {source, target, type, properties}.
        """
        if llm is None or len(known_entities) < 2:
            return []

        entity_list = "\n".join(
            f"- {e['id']} (type: {e.get('type', 'Unknown')})"
            for e in known_entities[:60]  # cap to avoid over-long prompts
        )

        prompt = f"""You are a knowledge graph extraction expert.
Given the following text and list of known entities, identify ONLY relationships between these entities that are explicitly supported by the text.

KNOWN ENTITIES:
{entity_list}

TEXT:
{combined_text}

Return ONLY a JSON object with a "relationships" array (no "entities" key needed):
{{
  "relationships": [
    {{
      "source": "source_entity_id_from_list",
      "target": "target_entity_id_from_list",
      "type": "RELATIONSHIP_TYPE",
      "negated": false,
      "properties": {{
        "anchor_text": "exact supporting relation phrase from the text, or null",
        "description": "how they relate in the text",
        "condition": "biological/experimental condition constraining this claim, or null",
        "quantitative": "any numerical finding attached to this relationship, or null",
        "confidence": "demonstrated|suggested|hypothesized"
      }}
    }}
  ]
}}

Rules:
- source and target MUST be entity ids from the KNOWN ENTITIES list above
- Only include relationships explicitly supported by the text
- Prefer specific relationship types (TREATS, CAUSES, INDICATES, etc.) over generic ones
- NEGATION RULE: set "negated": true if the text states the relationship does NOT hold
- GROUNDING RULE: set "anchor_text" to the shortest exact phrase from the text that expresses the relationship when possible
- QUALIFIER RULE: put experimental conditions in "condition", numerical findings in "quantitative"
- Return ONLY the JSON object, no other text"""

        try:
            response = llm.generate(prompt, "", model_name)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                return []
            result = json.loads(response[json_start:json_end])
            return result.get('relationships', [])
        except Exception as e:
            logging.warning(f"Cross-chunk relationship extraction failed: {e}")
            return []

    def _select_entities_for_relation_prompt(
        self,
        entities: List[Dict[str, Any]],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Keep only entities grounded in text, with a safe prompt-size cap."""
        if not isinstance(text, str) or not text.strip():
            return []
        deduped: List[Dict[str, Any]] = []
        seen_entity_ids = set()
        for entity in entities:
            entity_id = entity.get("id")
            dedupe_key = entity_id if isinstance(entity_id, str) and entity_id else id(entity)
            if dedupe_key in seen_entity_ids:
                continue
            if self._entity_appears_in_text(entity, text):
                deduped.append(entity)
                seen_entity_ids.add(dedupe_key)
        if len(deduped) <= getattr(self, "max_relationship_prompt_entities", 40):
            return deduped

        def _support_score(entity: Dict[str, Any]) -> Tuple[int, int]:
            names = self._entity_candidate_names(entity)
            longest_name = max((len(name) for name in names), default=0)
            mentions = sum(1 for name in names if isinstance(name, str) and name.lower() in text.lower())
            return (mentions, longest_name)

        deduped.sort(key=_support_score, reverse=True)
        return deduped[:getattr(self, "max_relationship_prompt_entities", 40)]

    def _extract_relationships_for_segment_windows(
        self,
        segment_texts: List[str],
        segment_entities: Dict[int, List[Dict[str, Any]]],
        segment_positions: Dict[int, List[int]],
        llm,
        model_name: str,
        *,
        max_window_size: int,
        scope_label: str,
        relationship_scope_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Recover relationships that span multiple adjacent chunks or passages."""
        if llm is None or len(segment_texts) < 2:
            return []

        n_segments = len(segment_texts)
        max_window_size = max(2, min(int(max_window_size or 2), n_segments))
        recovered: List[Dict[str, Any]] = []
        total_rels = 0

        for window_size in range(2, max_window_size + 1):
            for start in range(0, n_segments - window_size + 1):
                indices = range(start, start + window_size)
                combined_text = "\n\n".join(
                    str(segment_texts[idx] or "")
                    for idx in indices
                    if str(segment_texts[idx] or "").strip()
                )
                if not combined_text.strip():
                    continue

                candidate_entities: List[Dict[str, Any]] = []
                for idx in indices:
                    candidate_entities.extend(segment_entities.get(idx, []))
                candidate_entities = self._select_entities_for_relation_prompt(
                    candidate_entities,
                    combined_text,
                )
                if len(candidate_entities) < 2:
                    continue

                new_rels = self._extract_relationships_only(
                    combined_text,
                    candidate_entities,
                    llm,
                    model_name,
                )
                if new_rels:
                    if relationship_scope_metadata:
                        for rel in new_rels:
                            if not isinstance(rel, dict):
                                continue
                            rel_properties = dict(rel.get("properties") or {})
                            for key, value in relationship_scope_metadata.items():
                                if value is not None:
                                    rel_properties.setdefault(key, value)
                            rel["properties"] = rel_properties
                    provenance_positions: List[int] = []
                    for idx in indices:
                        provenance_positions.extend(segment_positions.get(idx, []))
                    recovered.extend(
                        self._attach_relationship_provenance(
                            new_rels,
                            provenance_positions,
                        )
                    )
                    total_rels += len(new_rels)
                    logging.info(
                        "%s pass (%d-%d, window=%d): %d relationships found",
                        scope_label,
                        start + 1,
                        start + window_size,
                        window_size,
                        len(new_rels),
                    )
                time.sleep(0.5)

        if total_rels:
            logging.info("%s extraction complete: %d additional relationships found", scope_label, total_rels)
        return recovered
