import re
import hashlib
import difflib
import importlib
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import os
import logging
import time
import math
from ontographrag.kg.utils.common_functions import load_embedding_model
from ontographrag.schemas.models import (
    OntologySchema,
)

# Shared helpers re-exported for backward compatibility (tests and callers
# access them through this module).
from ontographrag.kg.builders._creator_shared import (  # noqa: F401
    _ENTITY_MIN_NAME_LENGTH,
    _GENERIC_HUB_ENTITY_BLOCKLIST,
    _env_flag,
    _is_valid_entity_name,
)
from ontographrag.kg.builders._enrichment import EnrichmentMixin
from ontographrag.kg.builders._evaluation import EvaluationMixin
from ontographrag.kg.builders._extraction import ExtractionMixin
from ontographrag.kg.builders._ontology import OntologySchemaMixin
from ontographrag.kg.builders._storage import StorageMixin

class OntologyGuidedKGCreator(
    EnrichmentMixin,
    EvaluationMixin,
    ExtractionMixin,
    OntologySchemaMixin,
    StorageMixin,
):
    """
    Ontology-Guided Knowledge Graph Creator that properly extracts entities from PDF content
    using LLM with ontology guidance for better entity classification and relationships
    """
    _SECTION_HEADER_RE = re.compile(
        r"(?m)^(?:"
        r"\d+[\.\d]*\s*"                              # optional numbering: "2.", "2.1."
        r")?"
        r"(?P<header>"
        r"abstract|introduction|background|methods?|materials?\s+and\s+methods?|"
        r"experimental\s+(?:procedures?|design)|study\s+design|"
        r"results?(?:\s+and\s+discussion)?|"
        r"discussion|conclusions?|summary|"
        r"statistical\s+analysis|data\s+analysis|"
        r"supplementary|acknowledgements?|references?"
        r")"
        r"[:\s]*$",
        re.IGNORECASE,
    )
    _QUALIFIER_KEYWORDS = re.compile(
        r"\b(?:condition|experiment|treat(?:ed|ment)|knockout|knock[- ]?out|"
        r"mutant|patient|cohort|model|culture|in\s+vitro|in\s+vivo|"
        r"hypox|normox|baseline|control|express(?:ed|ion)|stimulat|"
        r"inhibit|activat|induc|depleted|overexpress|transfect|"
        r"under\s+these|such\s+conditions?|this\s+(?:model|system|context|protocol)|"
        r"these\s+(?:cells?|conditions?|animals?|patients?|mice|rats?))\b",
        re.IGNORECASE,
    )
    _COREF_MARKERS = re.compile(
        r"\b(?:"
        r"these\s+(?:conditions?|cells?|animals?|mice|rats?|patients?|results?|findings?|data)|"
        r"this\s+(?:model|system|treatment|context|approach|protocol|setup|condition|disease|disorder|syndrome|gene|protein|enzyme|receptor|biomarker)|"
        r"such\s+conditions?|"
        r"under\s+these\s+(?:conditions?|circumstances?)|"
        r"the\s+(?:treated|knockout|mutant|control)\s+(?:group|cells?|mice|animals?)|"
        r"the\s+above[-\s](?:mentioned\s+)?(?:conditions?|treatment|model|protocol)|"
        r"as\s+(?:described|mentioned|stated)\s+(?:above|previously|earlier)"
        r")\b",
        re.IGNORECASE,
    )
    _TITLE_AWARE_BUNDLE_DATASETS = {"hotpotqa", "2wikimultihopqa", "musique"}

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        embedding_model: str = "sentence_transformers",
        ontology_path: str = None,
        enable_coreference_resolution: bool = False,
        enable_heuristic_coreference_resolution: bool = True,
        retrieval_chunk_size: Optional[int] = None,
        retrieval_chunk_overlap: Optional[int] = None,
        strict_ontology: bool = True,
        self_consistency_n: int = 1,
        few_shot_example_count: int = 2,
        min_triple_confidence: float = 0.15,
        relationship_type_similarity_threshold: float = 0.62,
        enable_low_confidence_triple_reverification: bool = False,
        low_confidence_reverify_threshold: float = 0.4,
        enable_umls_linking: bool = False,
        umls_spacy_model: Optional[str] = None,
        enable_anchor_constrained_extraction: bool = True,
        enable_self_reflection: bool = True,
        enable_anchor_coverage_supplement: bool = True,
        enable_cross_passage_relation_recovery: bool = True,
        enable_soft_entity_linking: bool = False,
        soft_entity_similarity_threshold: float = 0.88,
        enable_fragmentation_repair: bool = False,
        fragmentation_bridge_similarity_threshold: float = 0.92,
        max_fragmentation_bridges: int = 8,
        enable_graph_summaries: bool = False,
        enable_claim_extraction: bool = False,
        max_summary_entities: int = 6,
        max_summary_relationships: int = 6,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        env_retrieval_chunk_size = os.getenv("RETRIEVAL_CHUNK_SIZE")
        env_retrieval_chunk_overlap = os.getenv("RETRIEVAL_CHUNK_OVERLAP")
        resolved_retrieval_chunk_size = retrieval_chunk_size
        if resolved_retrieval_chunk_size is None and env_retrieval_chunk_size:
            try:
                resolved_retrieval_chunk_size = int(env_retrieval_chunk_size)
            except ValueError:
                logging.warning(
                    "Invalid RETRIEVAL_CHUNK_SIZE=%r; falling back to 256",
                    env_retrieval_chunk_size,
                )
        resolved_retrieval_chunk_overlap = retrieval_chunk_overlap
        if resolved_retrieval_chunk_overlap is None and env_retrieval_chunk_overlap:
            try:
                resolved_retrieval_chunk_overlap = int(env_retrieval_chunk_overlap)
            except ValueError:
                logging.warning(
                    "Invalid RETRIEVAL_CHUNK_OVERLAP=%r; falling back to 64",
                    env_retrieval_chunk_overlap,
                )
        self.retrieval_chunk_size = max(64, int(resolved_retrieval_chunk_size or 256))
        default_retrieval_overlap = resolved_retrieval_chunk_overlap
        if default_retrieval_overlap is None:
            default_retrieval_overlap = min(64, max(16, self.retrieval_chunk_size // 4))
        self.retrieval_chunk_overlap = max(
            0,
            min(int(default_retrieval_overlap), self.retrieval_chunk_size - 1),
        )
        self.cross_chunk_relation_window = max(
            2,
            int(os.getenv("KG_CROSS_CHUNK_RELATION_WINDOW", "3") or 3),
        )
        self.cross_section_relation_window = max(
            2,
            int(os.getenv("KG_CROSS_SECTION_RELATION_WINDOW", "2") or 2),
        )
        self.cross_passage_relation_window = max(
            1,
            int(os.getenv("KG_CROSS_PASSAGE_RELATION_WINDOW", "2") or 2),
        )
        self.max_relationship_prompt_entities = max(
            8,
            int(os.getenv("KG_RELATION_PROMPT_ENTITY_CAP", "40") or 40),
        )
        self.enable_coreference_resolution = enable_coreference_resolution
        self.enable_heuristic_coreference_resolution = enable_heuristic_coreference_resolution
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.embedding_model = embedding_model
        self.ontology_path = ontology_path
        self.strict_ontology = strict_ontology
        self.self_consistency_n = max(
            1,
            int(os.getenv("KG_SELF_CONSISTENCY_N", str(self_consistency_n)) or self_consistency_n),
        )
        self.few_shot_example_count = max(
            0,
            int(os.getenv("KG_FEW_SHOT_EXAMPLE_COUNT", str(few_shot_example_count)) or few_shot_example_count),
        )
        try:
            self.min_triple_confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_MIN_TRIPLE_CONFIDENCE",
                            str(min_triple_confidence),
                        )
                        or min_triple_confidence
                    ),
                ),
            )
        except ValueError:
            self.min_triple_confidence = min_triple_confidence
        try:
            self.relationship_type_similarity_threshold = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_RELATIONSHIP_TYPE_SIMILARITY_THRESHOLD",
                            str(relationship_type_similarity_threshold),
                        )
                        or relationship_type_similarity_threshold
                    ),
                ),
            )
        except ValueError:
            self.relationship_type_similarity_threshold = relationship_type_similarity_threshold
        self.enable_low_confidence_triple_reverification = _env_flag(
            os.getenv("KG_ENABLE_LOW_CONFIDENCE_TRIPLE_REVERIFY"),
            enable_low_confidence_triple_reverification,
        )
        try:
            self.low_confidence_reverify_threshold = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_LOW_CONFIDENCE_REVERIFY_THRESHOLD",
                            str(low_confidence_reverify_threshold),
                        )
                        or low_confidence_reverify_threshold
                    ),
                ),
            )
        except ValueError:
            self.low_confidence_reverify_threshold = low_confidence_reverify_threshold
        self.enable_umls_linking = _env_flag(
            os.getenv("KG_ENABLE_UMLS_LINKING"),
            enable_umls_linking,
        )
        self.umls_spacy_model = (
            os.getenv("KG_UMLS_SPACY_MODEL", umls_spacy_model or "en_core_sci_sm") or "en_core_sci_sm"
        )
        self.enable_anchor_constrained_extraction = _env_flag(
            os.getenv("KG_ENABLE_ANCHOR_CONSTRAINED_EXTRACTION"),
            enable_anchor_constrained_extraction,
        )
        self.enable_self_reflection = _env_flag(
            os.getenv("KG_ENABLE_SELF_REFLECTION"),
            enable_self_reflection,
        )
        self.enable_anchor_coverage_supplement = _env_flag(
            os.getenv("KG_ENABLE_ANCHOR_COVERAGE_SUPPLEMENT"),
            enable_anchor_coverage_supplement,
        )
        self.enable_cross_passage_relation_recovery = _env_flag(
            os.getenv("KG_ENABLE_CROSS_PASSAGE_RELATION_RECOVERY"),
            enable_cross_passage_relation_recovery,
        )
        self.enable_soft_entity_linking = _env_flag(
            os.getenv("KG_ENABLE_SOFT_ENTITY_LINKING"),
            enable_soft_entity_linking,
        )
        try:
            self.soft_entity_similarity_threshold = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_SOFT_ENTITY_SIMILARITY_THRESHOLD",
                            str(soft_entity_similarity_threshold),
                        )
                        or soft_entity_similarity_threshold
                    ),
                ),
            )
        except ValueError:
            self.soft_entity_similarity_threshold = soft_entity_similarity_threshold
        self.enable_fragmentation_repair = _env_flag(
            os.getenv("KG_ENABLE_FRAGMENTATION_REPAIR"),
            enable_fragmentation_repair,
        )
        try:
            self.fragmentation_bridge_similarity_threshold = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_FRAGMENTATION_BRIDGE_SIMILARITY_THRESHOLD",
                            str(fragmentation_bridge_similarity_threshold),
                        )
                        or fragmentation_bridge_similarity_threshold
                    ),
                ),
            )
        except ValueError:
            self.fragmentation_bridge_similarity_threshold = (
                fragmentation_bridge_similarity_threshold
            )
        try:
            self.max_fragmentation_bridges = max(
                0,
                int(
                    os.getenv(
                        "KG_MAX_FRAGMENTATION_BRIDGES",
                        str(max_fragmentation_bridges),
                    )
                    or max_fragmentation_bridges
                ),
            )
        except ValueError:
            self.max_fragmentation_bridges = max_fragmentation_bridges
        self.enable_graph_summaries = _env_flag(
            os.getenv("KG_ENABLE_GRAPH_SUMMARIES"),
            enable_graph_summaries,
        )
        self.enable_claim_extraction = _env_flag(
            os.getenv("KG_ENABLE_CLAIM_EXTRACTION"),
            enable_claim_extraction,
        )
        try:
            self.max_summary_entities = max(
                1,
                int(
                    os.getenv(
                        "KG_MAX_SUMMARY_ENTITIES",
                        str(max_summary_entities),
                    )
                    or max_summary_entities
                ),
            )
        except ValueError:
            self.max_summary_entities = max_summary_entities
        try:
            self.max_summary_relationships = max(
                1,
                int(
                    os.getenv(
                        "KG_MAX_SUMMARY_RELATIONSHIPS",
                        str(max_summary_relationships),
                    )
                    or max_summary_relationships
                ),
            )
        except ValueError:
            self.max_summary_relationships = max_summary_relationships
        self._umls_linker_state = "disabled"
        self._umls_nlp = None
        self._triple_reverification_cache: Dict[str, bool] = {}
        self._last_schema_enforcement_stats = {
            "dropped_entities": 0,
            "dropped_relationships": 0,
            "kept_entities": 0,
            "kept_relationships": 0,
        }
        self._last_relationship_harmonization_stats = {
            "kept": 0,
            "dropped_unmapped": 0,
            "dropped_schema_mismatch": 0,
            "deduped": 0,
        }
        self._last_relationship_contradiction_stats = {
            "contradiction_groups": 0,
            "contradiction_edges": 0,
        }

        # Initialize embedding model
        self.embedding_function, self.embedding_dimension = load_embedding_model(embedding_model)
        logging.info(f"Initialized embedding model: {embedding_model}, dimension: {self.embedding_dimension}")

        # Load ontology if provided
        self.ontology_classes = []
        self.ontology_relationships = []
        self._ontology_schema: Optional[OntologySchema] = None
        if ontology_path and os.path.exists(ontology_path):
            logging.info(f"Ontology file exists at: {ontology_path} (size: {os.path.getsize(ontology_path)} bytes)")
            try:
                self._load_ontology(ontology_path)
                logging.info(f"✅ Successfully loaded ontology: {len(self.ontology_classes)} classes, {len(self.ontology_relationships)} relationships")

                # Validate ontology structure to prevent "string indices must be integers" errors
                self._validate_ontology_structure()

            except Exception as e:
                logging.error(f"❌ Failed to load ontology: {e}")
                if self.strict_ontology:
                    raise
                # Continue with empty ontology - LLM extraction will still work
                self.ontology_classes = []
                self.ontology_relationships = []
        else:
            if ontology_path:
                logging.warning(f"Ontology file not found: {ontology_path}")
                logging.info(f"Available files in temp dir: {os.listdir(os.path.dirname(ontology_path)) if ontology_path else 'N/A'}")
                if self.strict_ontology:
                    raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
            else:
                logging.info("No ontology provided - using basic LLM entity extraction")
            # No ontology - use empty lists (will fall back to pattern matching)

        # Pre-compute ontology class label embeddings for semantic classification
        self._ontology_class_embeddings: List[Tuple[str, Any]] = []  # [(class_id, embedding), ...]
        self._ontology_relationship_embeddings: List[Tuple[str, str, Any]] = []  # [(id, label, embedding), ...]
        if self.ontology_classes and self.embedding_function:
            try:
                labels = [cls['label'] for cls in self.ontology_classes]
                embeddings = self.embedding_function.embed_documents(labels)
                self._ontology_class_embeddings = [
                    (cls['id'], emb)
                    for cls, emb in zip(self.ontology_classes, embeddings)
                ]
                logging.info(f"Pre-computed embeddings for {len(self._ontology_class_embeddings)} ontology classes")
            except Exception as e:
                logging.warning(f"Could not pre-compute ontology class embeddings: {e}. Falling back to keyword matching.")
        if self.embedding_function:
            try:
                if self._ontology_schema and self._ontology_schema.relationship_types:
                    rel_labels = [rt.label for rt in self._ontology_schema.relationship_types]
                    embeddings = self.embedding_function.embed_documents(rel_labels)
                    self._ontology_relationship_embeddings = [
                        (rt.id, rt.label, emb)
                        for rt, emb in zip(self._ontology_schema.relationship_types, embeddings)
                    ]
                elif self.ontology_relationships:
                    rel_labels = [rel.get("label", rel.get("id", "")) for rel in self.ontology_relationships]
                    embeddings = self.embedding_function.embed_documents(rel_labels)
                    self._ontology_relationship_embeddings = [
                        (rel.get("id", ""), rel.get("label", rel.get("id", "")), emb)
                        for rel, emb in zip(self.ontology_relationships, embeddings)
                    ]
                if self._ontology_relationship_embeddings:
                    logging.info(
                        "Pre-computed embeddings for %d ontology relationship types",
                        len(self._ontology_relationship_embeddings),
                    )
            except Exception as e:
                logging.warning(
                    "Could not pre-compute ontology relationship embeddings: %s. Falling back to string matching.",
                    e,
                )

    @staticmethod
    def _boundary_pattern(name: str, *, ignore_case: bool = False) -> re.Pattern:
        """Boundary-aware regex for entity/relation span restoration."""
        flags = re.IGNORECASE if ignore_case else 0
        prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
        suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
        return re.compile(prefix + re.escape(name) + suffix, flags)

    @staticmethod
    def _surface_forms(name: str) -> set:
        """Return underscore/space variants for exact-text restoration."""
        forms = {name}
        forms.add(name.replace('_', ' '))
        forms.add(name.replace(' ', '_'))
        return {f for f in forms if isinstance(f, str) and f.strip()}

    def _find_exact_text_spans(
        self,
        text: str,
        candidates: List[str],
        *,
        start_offset: int = 0,
        chunk_position: Optional[int] = None,
        max_matches_per_candidate: int = 12,
    ) -> List[Dict[str, Any]]:
        """Locate exact candidate text spans with absolute character offsets."""
        if not isinstance(text, str) or not text.strip():
            return []

        unique_candidates: List[str] = []
        seen_candidates = set()
        for candidate in candidates or []:
            if not isinstance(candidate, str):
                continue
            cleaned = candidate.strip()
            if not cleaned or cleaned.lower() in seen_candidates:
                continue
            seen_candidates.add(cleaned.lower())
            unique_candidates.append(cleaned)

        spans: List[Dict[str, Any]] = []
        for candidate in sorted(unique_candidates, key=len, reverse=True):
            forms = sorted(self._surface_forms(candidate), key=len, reverse=True)
            candidate_matches = 0
            for form in forms:
                pattern = self._boundary_pattern(form, ignore_case=True)
                for match in pattern.finditer(text):
                    candidate_matches += 1
                    spans.append(
                        {
                            "text": text[match.start():match.end()],
                            "start": start_offset + match.start(),
                            "end": start_offset + match.end(),
                            "chunk_position": chunk_position,
                            "local_start": match.start(),
                            "local_end": match.end(),
                        }
                    )
                    if candidate_matches >= max_matches_per_candidate:
                        break
                if candidate_matches >= max_matches_per_candidate:
                    break
        return self._merge_anchor_spans(spans)

    def _detect_section_headers(self, text: str) -> List[Tuple[int, str]]:
        """Return list of (char_position, normalised_section_name) for every section
        header found in *text*, in document order.

        e.g. [(0, 'Abstract'), (412, 'Introduction'), (2105, 'Methods'), ...]
        """
        headers: List[Tuple[int, str]] = []
        for m in self._SECTION_HEADER_RE.finditer(text):
            raw = m.group("header").strip()
            # Normalise to title case; collapse "materials and methods" variants
            if re.match(r"materials?\s+and\s+methods?", raw, re.I):
                normalised = "Methods"
            elif re.match(r"experimental\s+(?:procedures?|design)|study\s+design", raw, re.I):
                normalised = "Methods"
            elif re.match(r"results?\s+and\s+discussion", raw, re.I):
                normalised = "Results"
            else:
                normalised = raw.title()
            headers.append((m.start(), normalised))
        return headers

    def _get_section_for_position(
        self, pos: int, section_headers: List[Tuple[int, str]]
    ) -> Optional[str]:
        """Return the section name that covers character position *pos*."""
        current = None
        for header_pos, name in section_headers:
            if header_pos <= pos:
                current = name
            else:
                break
        return current

    def _has_coreference_markers(self, text: str) -> bool:
        """Return True if *text* contains demonstrative coreference markers."""
        return bool(self._COREF_MARKERS.search(text))

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Return cosine similarity for two dense vectors."""
        denom = (
            (sum(a * a for a in vec_a) ** 0.5 + 1e-9)
            * (sum(b * b for b in vec_b) ** 0.5 + 1e-9)
        )
        return float(sum(a * b for a, b in zip(vec_a, vec_b)) / denom)

    def _resolve_coreferences_heuristic(
        self,
        chunk_text: str,
        previous_entities: List[Dict[str, Any]],
    ) -> str:
        """Cheap pronoun/coreference resolver using previously seen entities.

        This is intentionally conservative. It only rewrites typed demonstrative
        phrases such as "this disease" or "these patients" when a recent entity
        of the matching coarse type exists in the preceding context.
        """
        if not previous_entities or not self._has_coreference_markers(chunk_text):
            return chunk_text

        type_aliases = {
            "disease": {"disease", "condition", "disorder", "illness", "syndrome"},
            "treatment": {"treatment", "therapy", "drug", "medication", "procedure"},
            "gene": {"gene", "mutation", "variant", "allele"},
            "protein": {"protein", "enzyme", "receptor", "biomarker"},
            "patient": {"patient", "patients", "subject", "subjects", "cohort"},
            "cell": {"cell", "cells", "line", "lines"},
            "animal": {"animal", "animals", "mouse", "mice", "rat", "rats"},
            "model": {"model", "system", "context", "protocol", "setup"},
            "person": {"person", "people", "actor", "director", "scientist"},
            "organization": {"organization", "company", "museum", "institute", "university"},
            "location": {"city", "country", "state", "province", "location"},
            "work": {"film", "movie", "book", "song", "album", "series"},
        }
        type_preferences = {
            "disease": ("Disease", "MedicalCondition", "Condition", "Disorder"),
            "treatment": ("Treatment", "Drug", "Medication", "MedicalProcedure"),
            "gene": ("Gene", "Genetic Mutation", "Genetic Variant", "Genetic Disorder"),
            "protein": ("Protein", "Biomarker", "Enzyme", "Receptor"),
            "patient": ("Patient", "PopulationGroup", "Person"),
            "cell": ("Cell", "Cell Type", "CellType", "Cell Line"),
            "animal": ("Organism", "PopulationGroup"),
            "model": ("Model", "Concept", "Biological System"),
            "person": ("Person", "Physician"),
            "organization": ("Organization", "Hospital", "Institute"),
            "location": ("Location", "Anatomical Structure", "AnatomicalStructure"),
            "work": ("Work", "CreativeWork", "Film"),
        }

        def _choose_antecedent(category: str) -> Optional[str]:
            preferred_types = type_preferences.get(category, ())
            category_aliases = type_aliases.get(category, {category})
            for entity in reversed(previous_entities):
                entity_name = str(entity.get("id") or "").strip()
                entity_type = str(entity.get("type") or "")
                if not entity_name:
                    continue
                if entity_type in preferred_types:
                    return entity_name
                entity_desc = str((entity.get("properties") or {}).get("description") or "").lower()
                if any(alias in entity_desc for alias in category_aliases):
                    return entity_name
            for entity in reversed(previous_entities):
                entity_name = str(entity.get("id") or "").strip()
                if entity_name:
                    return entity_name
            return None

        resolved = chunk_text
        replacement_patterns = [
            ("disease", re.compile(r"\b(?:this|the)\s+(?:disease|condition|disorder|syndrome)\b", re.IGNORECASE)),
            ("treatment", re.compile(r"\b(?:this|the)\s+(?:treatment|therapy|drug|medication|procedure)\b", re.IGNORECASE)),
            ("gene", re.compile(r"\b(?:this|the)\s+(?:gene|mutation|variant|allele)\b", re.IGNORECASE)),
            ("protein", re.compile(r"\b(?:this|the)\s+(?:protein|enzyme|receptor|biomarker)\b", re.IGNORECASE)),
            ("patient", re.compile(r"\b(?:these|the)\s+(?:patients|subjects|cases|participants)\b", re.IGNORECASE)),
            ("cell", re.compile(r"\b(?:these|the)\s+(?:cells|cell lines)\b", re.IGNORECASE)),
            ("animal", re.compile(r"\b(?:these|the)\s+(?:animals|mice|rats)\b", re.IGNORECASE)),
            ("model", re.compile(r"\b(?:this|the)\s+(?:model|system|protocol|setup|context)\b", re.IGNORECASE)),
            ("person", re.compile(r"\b(?:this|the)\s+(?:person|actor|director|scientist)\b", re.IGNORECASE)),
            ("organization", re.compile(r"\b(?:this|the)\s+(?:organization|company|museum|institute|university)\b", re.IGNORECASE)),
            ("location", re.compile(r"\b(?:this|the)\s+(?:city|country|state|province|location)\b", re.IGNORECASE)),
            ("work", re.compile(r"\b(?:this|the)\s+(?:film|movie|book|song|album|series)\b", re.IGNORECASE)),
        ]

        for category, pattern in replacement_patterns:
            antecedent = _choose_antecedent(category)
            if antecedent:
                resolved = pattern.sub(antecedent, resolved)

        return resolved

    def _build_section_segments(
        self,
        text: str,
        *,
        section_headers: Optional[List[Tuple[int, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return ordered section-aligned text segments for upstream relation recovery."""
        headers = list(section_headers or self._detect_section_headers(text))
        if not text:
            return []

        boundaries = [0]
        for pos, _ in headers:
            if pos not in boundaries and 0 < pos < len(text):
                boundaries.append(pos)
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))

        segments: List[Dict[str, Any]] = []
        for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            segment_text = text[start:end]
            if not segment_text.strip():
                continue
            section_name = self._get_section_for_position(start, headers)
            segments.append({
                "index": idx,
                "name": section_name or "Preamble",
                "start_pos": start,
                "end_pos": end,
                "text": segment_text,
            })
        return segments

    def merge_synonym_entities(self, graph, similarity_threshold: float = 0.82, kg_name: str = None) -> int:
        """
        HippoRAG-style synonym merging: cluster entity nodes that share nearly-identical
        embeddings (surface-form variants like "TBK1" / "TBK1 kinase" / "TBK1 protein")
        and merge them into a single canonical node.

        Uses the existing entity_vector index.  Merges the lower-degree node into the
        higher-degree one so that the richer node survives.

        Returns the number of merges performed.
        """
        try:
            import numpy as np

            # Fetch all entity nodes with their embeddings
            params = {"kg_name": kg_name} if kg_name else {}
            kg_filter = "AND e.kgName = $kg_name" if kg_name else ""
            fetch_q = f"""
            MATCH (e:__Entity__)
            WHERE e.embedding IS NOT NULL {kg_filter}
            RETURN elementId(e) AS eid, coalesce(e.name, e.original_id, e.id) AS name, e.embedding AS emb,
                   COUNT {{ (e)--() }} AS degree,
                   coalesce(e.type, e.ontology_class, '') AS etype
            """
            rows = graph.query(fetch_q, params)
            if not rows:
                logging.info("No entity embeddings found; skipping synonym merging.")
                return 0

            rows = [r for r in rows if r.get('emb') is not None]
            if not rows:
                logging.info("No entity embeddings found after filtering; skipping synonym merging.")
                return 0
            eids = [r['eid'] for r in rows]
            names = [r['name'] for r in rows]
            degrees = [r['degree'] for r in rows]
            etypes = [r.get('etype', '') or '' for r in rows]
            embeddings = np.array([r['emb'] for r in rows], dtype=float)

            # Generic types that should not block merging (the LLM falls back to these
            # when it cannot classify an entity, so two nodes might genuinely represent
            # the same concept even though one was labelled "Entity" and the other "Concept").
            _generic_types = {'entity', 'concept', 'unknown', 'other', ''}

            # Normalise for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            normed = embeddings / norms

            # Build equivalence clusters via union-find so that synonym chains
            # (A≈B, B≈C) are merged atomically rather than pairwise.
            # Pairwise sequential merges leave dangling references when later pairs
            # reference nodes already merged away.
            sim_matrix = normed @ normed.T
            n = len(eids)

            parent = list(range(n))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # path compression
                    x = parent[x]
                return x

            def _union(x: int, y: int) -> None:
                px, py = _find(x), _find(y)
                if px != py:
                    parent[px] = py

            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= similarity_threshold:
                        ti, tj = etypes[i].lower(), etypes[j].lower()
                        if ti not in _generic_types and tj not in _generic_types and ti != tj:
                            continue
                        if not self._names_pass_synonym_guard(names[i], names[j]):
                            continue
                        _union(i, j)

            # Group indices by cluster root
            from collections import defaultdict as _dd
            clusters: dict = _dd(list)
            for i in range(n):
                clusters[_find(i)].append(i)

            # Build one (canonical, [duplicates]) tuple per multi-node cluster.
            # Canonical = highest-degree node in the cluster.
            merge_tasks = []
            for members in clusters.values():
                if len(members) < 2:
                    continue
                canonical_idx = max(members, key=lambda i: degrees[i])
                dups = [i for i in members if i != canonical_idx]
                merge_tasks.append((canonical_idx, dups))

            if not merge_tasks:
                logging.info("No synonym clusters found above threshold %.2f", similarity_threshold)
                return 0

            merged = 0
            for canonical_idx, dup_indices in merge_tasks:
                canonical_name = names[canonical_idx]
                for dup_idx in dup_indices:
                    duplicate_name = names[dup_idx]
                    try:
                        # Stamp synonym alias on canonical BEFORE merge (dup ceases to exist after).
                        # Then use apoc.refactor.mergeNodes to atomically rewire all relationships
                        # and delete the duplicate.  {properties: 'discard'} means the first
                        # node in the list (canonical) wins all property conflicts.
                        merge_q = """
                        MATCH (can:__Entity__) WHERE elementId(can) = $can_eid
                        MATCH (dup:__Entity__) WHERE elementId(dup) = $dup_eid
                        SET can.synonyms = coalesce(can.synonyms, []) + [dup.id]
                        WITH [can, dup] AS nodes
                        CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true})
                        YIELD node
                        RETURN node.id AS merged_id
                        """
                        graph.query(
                            merge_q,
                            {
                                "dup_eid": eids[dup_idx],
                                "can_eid": eids[canonical_idx],
                            },
                        )
                        logging.info("Merged synonym '%s' → '%s'", duplicate_name, canonical_name)
                        merged += 1
                    except Exception as merge_err:
                        logging.warning("Failed to merge '%s' → '%s': %s",
                                        duplicate_name, canonical_name, merge_err)

            logging.info(f"Synonym merging complete: {merged}/{len(merge_tasks)} pairs merged")
            return merged

        except Exception as e:
            logging.error(f"Synonym merging failed: {e}")
            return 0

    def _semantic_relationship_match(
        self,
        raw_type: str,
        candidate_ids: List[str],
    ) -> Optional[Tuple[str, float]]:
        """Return the best semantic relationship-type match if embeddings exist."""
        embeddings = getattr(self, "_ontology_relationship_embeddings", []) or []
        if not embeddings or not self.embedding_function or not raw_type or not candidate_ids:
            return None

        try:
            raw_embedding = self.embedding_function.embed_query(raw_type)
        except Exception as exc:
            logging.debug("Relationship embedding lookup failed for '%s': %s", raw_type, exc)
            return None

        candidate_id_set = set(candidate_ids)
        best_id = None
        best_score = 0.0
        for rel_id, _, rel_embedding in embeddings:
            if rel_id not in candidate_id_set:
                continue
            score = self._cosine_similarity(raw_embedding, rel_embedding)
            if score > best_score:
                best_id = rel_id
                best_score = score

        if best_id and best_score >= getattr(self, "relationship_type_similarity_threshold", 0.62):
            return best_id, best_score
        return None

    def _generate_entity_id(self, entity: Dict[str, Any]) -> str:
        """Generate a deterministic UUID scoped to both normalized text AND type.

        Including the type means same-surface/different-type entities (e.g. "depression"
        as Disease vs GeologicalFeature) get distinct UUIDs after the harmonization split.
        For merged entities (LLM type-drift already resolved to one specific type by
        _harmonize_entities) the representative carries a single resolved type, so
        chunk-level variants still converge to the same UUID.
        """
        norm_text = self._normalize_entity_text(entity['id'])
        entity_type = (entity.get('type') or '').strip()
        properties = dict(entity.get("properties") or {})
        scope_key = str(
            properties.get("disambiguation_scope")
            or properties.get("title_scope_key")
            or ""
        ).strip()
        unique_seed = f"{norm_text}:{entity_type}" if entity_type else norm_text
        if scope_key:
            unique_seed = f"{unique_seed}:{scope_key}"
        return str(uuid.uuid5(uuid.NAMESPACE_OID, unique_seed))

    def _bundle_title_variants(self, raw_title: str) -> List[str]:
        """Return normalized title variants for bundle-style article titles."""
        if not isinstance(raw_title, str):
            return []
        cleaned = raw_title.strip()
        if not cleaned:
            return []
        variants = {cleaned}
        # Hotpot/2Wiki titles often carry parenthetical disambiguators.
        variants.add(re.sub(r"\s+\([^)]*\)\s*$", "", cleaned).strip())
        normalized = []
        seen = set()
        for variant in variants:
            norm = self._normalize_entity_text(variant)
            if norm and norm not in seen:
                normalized.append(norm)
                seen.add(norm)
        return normalized

    def _infer_passage_title(self, passage_text: str, dataset: str) -> Optional[str]:
        """Infer a source/article title from a benchmark passage string when present."""
        if not isinstance(passage_text, str):
            return None
        text = passage_text.strip()
        if not text:
            return None
        if text.lower().startswith("title:"):
            first_line = text.splitlines()[0]
            title = first_line.split(":", 1)[1].strip()
            return title or None
        if dataset in self._TITLE_AWARE_BUNDLE_DATASETS:
            prefix, sep, _ = text.partition(". ")
            if sep and prefix and len(prefix) <= 160:
                return prefix.strip()
        return None

    def _build_passage_scope_key(
        self,
        *,
        dataset: Any,
        question_id: Any,
        passage_index: Any,
    ) -> Optional[str]:
        dataset_str = str(dataset or "").strip()
        question_str = str(question_id or "").strip()
        if not dataset_str or not question_str or passage_index is None:
            return None
        return f"{dataset_str}/{question_str}/p{passage_index}"

    def _entity_matches_own_source_title(
        self,
        normalized_text: str,
        entity: Dict[str, Any],
    ) -> bool:
        properties = dict(entity.get("properties") or {})
        source_title = properties.get("source_title")
        if not isinstance(source_title, str) or not source_title.strip():
            return False
        return normalized_text in self._bundle_title_variants(source_title)

    def _names_pass_synonym_guard(self, left_name: str, right_name: str) -> bool:
        """Require some lexical agreement before merging embedding-near entities."""
        left_norm = self._normalize_entity_text(left_name or "")
        right_norm = self._normalize_entity_text(right_name or "")
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        if len(left_norm) >= 4 and left_norm in right_norm:
            return True
        if len(right_norm) >= 4 and right_norm in left_norm:
            return True

        left_tokens = [tok for tok in re.split(r"[_\s]+", left_norm) if tok]
        right_tokens = [tok for tok in re.split(r"[_\s]+", right_norm) if tok]
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        shared = left_set & right_set
        if shared:
            longest_shared = max(len(token) for token in shared)
            union = left_set | right_set
            if longest_shared >= 4 and (
                left_set.issubset(right_set)
                or right_set.issubset(left_set)
                or (len(union) > 0 and len(shared) / len(union) >= 0.5)
            ):
                return True
            if any(any(ch.isdigit() for ch in token) for token in shared):
                return True

        def _initialism(tokens: List[str]) -> str:
            return "".join(token[0] for token in tokens if token)

        if len(left_tokens) == 1 and left_tokens[0] == _initialism(right_tokens):
            return True
        if len(right_tokens) == 1 and right_tokens[0] == _initialism(left_tokens):
            return True

        return difflib.SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.92

    def _entity_name_variants(self, name: str) -> List[str]:
        """Generate robust surface-form variants for entity resolution."""
        if not isinstance(name, str):
            return []
        base = name.strip()
        if not base:
            return []

        variants = {
            base,
            base.lower(),
            base.replace('_', ' '),
            base.replace(' ', '_'),
            self._normalize_entity_text(base),
        }
        return [v for v in variants if isinstance(v, str) and v.strip()]

    def _entity_candidate_names(self, entity: Dict[str, Any]) -> List[str]:
        """Collect plausible surface forms for an extracted entity."""
        if not isinstance(entity, dict):
            return []

        candidates: List[str] = []
        for candidate in (
            entity.get("id"),
            entity.get("name"),
            (entity.get("properties") or {}).get("name"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                candidates.append(candidate)

        props = entity.get("properties") or {}
        all_names = props.get("all_names") or []
        if isinstance(all_names, list):
            for alias in all_names:
                if isinstance(alias, str) and alias.strip():
                    candidates.append(alias)

        # Include LLM-extracted aliases (alternate surface forms / abbreviations)
        llm_aliases = props.get("aliases") or []
        if isinstance(llm_aliases, list):
            for alias in llm_aliases:
                if isinstance(alias, str) and alias.strip():
                    candidates.append(alias)

        seen = set()
        deduped: List[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    def _entity_appears_in_text(self, entity: Dict[str, Any], text: str) -> bool:
        """Check whether an entity or any of its aliases appears in text."""
        if not isinstance(text, str) or not text.strip():
            return False

        text_lower = text.lower()
        seen_variants = set()
        for candidate in self._entity_candidate_names(entity):
            for variant in self._entity_name_variants(candidate):
                normalized = variant.strip().lower()
                if len(normalized) < 3 or normalized in seen_variants:
                    continue
                seen_variants.add(normalized)
                prefix = r'(?<!\w)' if not normalized[:1].isalnum() and normalized[:1] != '_' else r'\b'
                suffix = r'(?!\w)' if not normalized[-1:].isalnum() and normalized[-1:] != '_' else r'\b'
                pattern = re.compile(prefix + re.escape(normalized) + suffix)
                if pattern.search(text_lower):
                    return True
        return False

    def _resolve_relationship_endpoint(
        self,
        raw_name: str,
        entities: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Resolve a relationship endpoint to the best matching extracted entity ID.

        The LLM often uses short forms in relationship triples while the entity
        list carries a fuller mention, so we allow safe fuzzy matching here.
        """
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None

        raw_name = raw_name.strip()
        raw_variants = self._entity_name_variants(raw_name)
        raw_norm = self._normalize_entity_text(raw_name)

        direct_matches: List[str] = []
        best_entity_id: Optional[str] = None
        best_score = 0.0
        second_best = 0.0

        for entity in entities:
            entity_id = entity.get("id")
            if not isinstance(entity_id, str) or not entity_id.strip():
                continue

            candidate_variants = set()
            for candidate in self._entity_candidate_names(entity):
                candidate_variants.update(self._entity_name_variants(candidate))

            if any(rv.lower() == cv.lower() for rv in raw_variants for cv in candidate_variants):
                direct_matches.append(entity_id)
                continue

            entity_best = 0.0
            for candidate in candidate_variants:
                candidate_norm = self._normalize_entity_text(candidate)
                if not candidate_norm or not raw_norm:
                    continue
                if candidate_norm == raw_norm:
                    entity_best = 1.0
                    break

                score = 0.0
                if raw_norm in candidate_norm or candidate_norm in raw_norm:
                    if min(len(raw_norm), len(candidate_norm)) >= 4:
                        score = 0.94
                else:
                    score = difflib.SequenceMatcher(None, raw_norm, candidate_norm).ratio()
                    raw_tokens = set(raw_norm.split('_'))
                    candidate_tokens = set(candidate_norm.split('_'))
                    if raw_tokens and candidate_tokens:
                        overlap = len(raw_tokens & candidate_tokens) / max(
                            1, min(len(raw_tokens), len(candidate_tokens))
                        )
                        score = max(score, overlap * 0.92)

                entity_best = max(entity_best, score)

            if entity_best > best_score:
                second_best = best_score
                best_score = entity_best
                best_entity_id = entity_id
            elif entity_best > second_best:
                second_best = entity_best

        unique_direct = list(dict.fromkeys(direct_matches))
        if len(unique_direct) == 1:
            return unique_direct[0]
        if len(unique_direct) > 1:
            for entity_id in unique_direct:
                if entity_id.lower() == raw_name.lower():
                    return entity_id
            return None

        if best_entity_id and best_score >= 0.90 and (best_score - second_best) >= 0.03:
            return best_entity_id
        return None

    @staticmethod
    def _relationship_evidence_scope(
        verification_chunks: List[Dict[str, Any]],
        triple_confidence: float,
    ) -> str:
        """Categorize how directly a relation is text-supported.

        This intentionally does not collapse weak evidence into cross-section.
        In the current confidence scheme:
        - 1.0 means same sentence
        - 0.7 means same chunk
        - 0.4 means both entities found across multiple chunks
        - 0.3 means only one side grounded
        - 0.1 means unsupported
        """
        if not verification_chunks:
            return "unknown"
        if triple_confidence >= 0.95:
            return "sentence"
        if triple_confidence >= 0.65:
            return "chunk"
        if triple_confidence >= 0.35:
            section_names = {
                str(chunk.get("section_name")).strip()
                for chunk in (verification_chunks or [])
                if str(chunk.get("section_name") or "").strip()
            }
            if len(section_names) > 1:
                return "cross_section"
            return "cross_chunk"
        if triple_confidence >= 0.25:
            return "partial_grounding"
        return "unsupported"

    def _ensure_umls_linker(self):
        """Lazily initialize an optional SciSpaCy UMLS linker."""
        if not getattr(self, "enable_umls_linking", False):
            return None
        if getattr(self, "_umls_linker_state", "disabled") == "ready":
            return self._umls_nlp
        if getattr(self, "_umls_linker_state", "disabled") == "unavailable":
            return None

        try:
            spacy = importlib.import_module("spacy")
            importlib.import_module("scispacy")
            nlp = spacy.load(self.umls_spacy_model)
            if "scispacy_linker" not in getattr(nlp, "pipe_names", []):
                nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                    },
                )
            self._umls_nlp = nlp
            self._umls_linker_state = "ready"
            logging.info("UMLS linker enabled using spaCy model '%s'", self.umls_spacy_model)
            return nlp
        except Exception as exc:
            self._umls_linker_state = "unavailable"
            self._umls_nlp = None
            logging.warning("UMLS linker unavailable, continuing without it: %s", exc)
            return None

    def _link_entity_to_umls(
        self,
        entity_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Return top UMLS link metadata for an entity name when available."""
        nlp = self._ensure_umls_linker()
        if nlp is None or not entity_name.strip():
            return None
        try:
            doc = nlp(entity_name)
        except Exception as exc:
            logging.debug("UMLS linking failed for '%s': %s", entity_name, exc)
            return None

        kb_ents = []
        for ent in getattr(doc, "ents", []):
            kb_ents.extend(list(getattr(getattr(ent, "_", None), "kb_ents", []) or []))
        if not kb_ents:
            kb_ents = list(getattr(getattr(doc, "_", None), "kb_ents", []) or [])
        if not kb_ents:
            return None

        cui, score = kb_ents[0]
        if score < 0.75:
            return None

        kb_name = None
        try:
            linker = nlp.get_pipe("scispacy_linker")
            kb_entry = linker.kb.cui_to_entity.get(cui)
            if kb_entry:
                kb_name = kb_entry[0]
        except Exception:
            kb_name = None

        return {
            "cui": cui,
            "score": float(score),
            "name": kb_name,
        }

    def _apply_optional_umls_linking(
        self,
        harmonized_entities: List[Dict[str, Any]],
        entity_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Attach optional UMLS metadata and merge entities sharing a confident CUI."""
        if not getattr(self, "enable_umls_linking", False):
            return harmonized_entities, entity_map
        if self._ensure_umls_linker() is None:
            return harmonized_entities, entity_map

        generic_types = {"Concept", "Entity", "Unknown", "Other"}
        entities_with_links: List[Dict[str, Any]] = []
        for entity in harmonized_entities:
            entity_copy = dict(entity)
            properties = dict(entity_copy.get("properties") or {})
            link = self._link_entity_to_umls(str(entity_copy.get("id") or ""))
            if link:
                properties["umls_cui"] = link["cui"]
                properties["umls_score"] = round(link["score"], 4)
                if link.get("name"):
                    properties["umls_name"] = link["name"]
            entity_copy["properties"] = properties
            entities_with_links.append(entity_copy)

        grouped_by_cui: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        ungrouped: List[Dict[str, Any]] = []
        for entity in entities_with_links:
            cui = (entity.get("properties") or {}).get("umls_cui")
            if cui:
                grouped_by_cui[str(cui)].append(entity)
            else:
                ungrouped.append(entity)

        merged_entities: List[Dict[str, Any]] = []
        remapped_entities_by_uuid: Dict[str, Dict[str, Any]] = {}

        for cui, cui_group in grouped_by_cui.items():
            specific_types = {
                entity.get("type")
                for entity in cui_group
                if entity.get("type") not in generic_types
            }
            if len(cui_group) == 1 or len(specific_types) > 1:
                for entity in cui_group:
                    merged_entities.append(entity)
                    if entity.get("uuid"):
                        remapped_entities_by_uuid[entity["uuid"]] = entity
                continue

            representative = max(
                cui_group,
                key=lambda e: (
                    0 if e.get("type") in generic_types else 1,
                    len(str((e.get("properties") or {}).get("description") or "")),
                    len(str(e.get("id") or "")),
                ),
            ).copy()
            properties = dict(representative.get("properties") or {})
            all_names = set(properties.get("all_names") or [representative.get("id")])
            all_descriptions = set(properties.get("all_descriptions") or [])
            original_ids = set(properties.get("original_ids") or [representative.get("id")])
            merged_anchor_spans = self._merge_anchor_spans(properties.get("anchor_spans"))

            for entity in cui_group:
                entity_props = dict(entity.get("properties") or {})
                all_names.update(entity_props.get("all_names") or [entity.get("id")])
                all_descriptions.update(entity_props.get("all_descriptions") or [])
                description = entity_props.get("description")
                if description:
                    all_descriptions.add(description)
                original_ids.update(entity_props.get("original_ids") or [entity.get("id")])
                merged_anchor_spans = self._merge_anchor_spans(
                    merged_anchor_spans,
                    entity_props.get("anchor_spans"),
                )
                if entity.get("embedding") and representative.get("embedding") is None:
                    representative["embedding"] = entity.get("embedding")
                if entity.get("uuid"):
                    remapped_entities_by_uuid[entity["uuid"]] = representative

            properties["all_names"] = sorted(
                {name for name in all_names if isinstance(name, str) and name and str(name).strip()}
            )
            properties["all_descriptions"] = sorted(
                {desc for desc in all_descriptions if isinstance(desc, str) and desc.strip()},
                key=len,
                reverse=True,
            )
            properties["original_ids"] = sorted(
                {value for value in original_ids if isinstance(value, str) and value and str(value).strip()}
            )
            if merged_anchor_spans:
                properties["anchor_spans"] = merged_anchor_spans
                properties["anchor_mention_count"] = len(merged_anchor_spans)
            representative["properties"] = properties
            merged_entities.append(representative)
            logging.info(
                "Merged %d entities through shared UMLS CUI %s",
                len(cui_group),
                cui,
            )

        merged_entities.extend(ungrouped)
        updated_entity_map: Dict[str, Dict[str, Any]] = {}
        for variant_name, representative in entity_map.items():
            uuid_value = representative.get("uuid") if isinstance(representative, dict) else None
            updated_entity_map[variant_name] = remapped_entities_by_uuid.get(
                uuid_value,
                representative,
            )

        return merged_entities, updated_entity_map

    def _soft_entity_alias_keys(self, name: str) -> List[str]:
        """Return conservative soft-canonical keys for a surface form."""
        if not isinstance(name, str):
            return []
        cleaned = name.strip()
        if not cleaned:
            return []

        keys = {self._normalize_entity_text(cleaned)}
        stripped_parenthetical = re.sub(r"\s+\([^)]*\)\s*$", "", cleaned).strip()
        if stripped_parenthetical and stripped_parenthetical != cleaned:
            keys.add(self._normalize_entity_text(stripped_parenthetical))

        tokens = [
            tok for tok in re.split(r"[_\s]+", self._normalize_entity_text(cleaned)) if tok
        ]
        if len(tokens) >= 2:
            keys.add("".join(token[0] for token in tokens if token))

        removable_suffixes = {
            "protein",
            "proteins",
            "gene",
            "genes",
            "kinase",
            "kinases",
            "receptor",
            "receptors",
            "complex",
            "complexes",
            "subunit",
            "subunits",
            "pathway",
            "pathways",
            "signaling",
            "signalling",
            "isoform",
            "isoforms",
        }
        stripped_tokens = list(tokens)
        while len(stripped_tokens) > 1 and stripped_tokens[-1] in removable_suffixes:
            stripped_tokens = stripped_tokens[:-1]
            keys.add("_".join(stripped_tokens))
            if len(stripped_tokens) >= 2:
                keys.add("".join(token[0] for token in stripped_tokens if token))

        return [key for key in keys if isinstance(key, str) and key.strip()]

    @staticmethod
    def _cosine_similarity(left: Any, right: Any) -> float:
        """Lightweight cosine similarity without a hard NumPy dependency."""
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return 0.0
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for l_val, r_val in zip(left, right):
            try:
                l_float = float(l_val)
                r_float = float(r_val)
            except (TypeError, ValueError):
                return 0.0
            dot += l_float * r_float
            left_norm += l_float * l_float
            right_norm += r_float * r_float
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return dot / math.sqrt(left_norm * right_norm)

    @staticmethod
    def _confidence_score(value: Any) -> float:
        """Normalize numeric or verbal confidence labels onto a sortable [0, 1] scale."""
        if isinstance(value, (int, float)):
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        text = str(value or "").strip().lower()
        if not text or text in {"none", "null", "nan", "unknown", "n/a"}:
            return 0.0

        try:
            return max(0.0, min(1.0, float(text)))
        except (TypeError, ValueError):
            pass

        verbal_scale = {
            "demonstrated": 0.95,
            "supported": 0.85,
            "high": 0.8,
            "strong": 0.8,
            "suggested": 0.6,
            "moderate": 0.55,
            "hypothesized": 0.35,
            "hypothesis": 0.35,
            "speculative": 0.25,
            "weak": 0.2,
            "low": 0.2,
        }
        return verbal_scale.get(text, 0.0)

    def _merge_entity_cluster(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a soft-linked entity cluster into one representative entity."""
        representative = max(
            entities,
            key=lambda entity: (
                self._entity_type_specificity(entity.get("type")),
                len(str((entity.get("properties") or {}).get("description") or "")),
                len(str(entity.get("id") or "")),
            ),
        ).copy()

        properties = dict(representative.get("properties") or {})
        merged_anchor_spans = self._merge_anchor_spans(
            *[(entity.get("properties") or {}).get("anchor_spans") for entity in entities]
        )

        all_names: set = set()
        all_descriptions: set = set()
        aliases: set = set()
        source_scope_keys: set = set()
        source_titles: set = set()
        question_ids: set = set()
        passage_indexes: set = set()
        umls_cuis: set = set()
        umls_names: set = set()
        best_embedding = representative.get("embedding")

        for entity in entities:
            entity_properties = dict(entity.get("properties") or {})
            if not best_embedding and entity.get("embedding") is not None:
                best_embedding = entity.get("embedding")

            for name in self._entity_candidate_names(entity):
                if isinstance(name, str) and name.strip():
                    all_names.add(name.strip())
            description = str(entity_properties.get("description") or "").strip()
            if description:
                all_descriptions.add(description)
            for alias in entity_properties.get("aliases") or []:
                if isinstance(alias, str) and alias.strip():
                    aliases.add(alias.strip())
            for key in entity_properties.get("source_scope_keys") or []:
                if isinstance(key, str) and key.strip():
                    source_scope_keys.add(key.strip())
            source_scope_key = str(entity_properties.get("source_scope_key") or "").strip()
            if source_scope_key:
                source_scope_keys.add(source_scope_key)
            for title in entity_properties.get("source_titles") or []:
                if isinstance(title, str) and title.strip():
                    source_titles.add(title.strip())
            source_title = str(entity_properties.get("source_title") or "").strip()
            if source_title:
                source_titles.add(source_title)
            for question_id in entity_properties.get("question_ids") or []:
                if isinstance(question_id, str) and question_id.strip():
                    question_ids.add(question_id.strip())
            question_id = str(entity_properties.get("question_id") or "").strip()
            if question_id:
                question_ids.add(question_id)
            for passage_index in entity_properties.get("passage_indexes") or []:
                if isinstance(passage_index, int):
                    passage_indexes.add(int(passage_index))
            if isinstance(entity_properties.get("passage_index"), int):
                passage_indexes.add(int(entity_properties["passage_index"]))
            cui = str(entity_properties.get("umls_cui") or "").strip()
            if cui:
                umls_cuis.add(cui)
            umls_name = str(entity_properties.get("umls_name") or "").strip()
            if umls_name:
                umls_names.add(umls_name)

            for key, value in entity_properties.items():
                if key in {
                    "description",
                    "all_names",
                    "aliases",
                    "anchor_spans",
                    "source_scope_keys",
                    "source_titles",
                    "question_ids",
                    "passage_indexes",
                    "umls_cui",
                    "umls_name",
                }:
                    continue
                properties.setdefault(key, value)

        if all_names:
            properties["all_names"] = sorted(all_names)
            properties["original_ids"] = sorted(all_names)
        if all_descriptions:
            properties["all_descriptions"] = sorted(all_descriptions, key=len, reverse=True)
            properties.setdefault("description", max(all_descriptions, key=len))
        if aliases:
            properties["aliases"] = sorted(aliases)
        if merged_anchor_spans:
            properties["anchor_spans"] = merged_anchor_spans
            properties["anchor_mention_count"] = len(merged_anchor_spans)
        if source_scope_keys:
            properties["source_scope_keys"] = sorted(source_scope_keys)
        if source_titles:
            properties["source_titles"] = sorted(source_titles)
        if question_ids:
            properties["question_ids"] = sorted(question_ids)
        if passage_indexes:
            properties["passage_indexes"] = sorted(passage_indexes)
        if len(umls_cuis) == 1:
            properties["umls_cui"] = next(iter(umls_cuis))
        if len(umls_names) == 1:
            properties["umls_name"] = next(iter(umls_names))
        properties["soft_linked"] = True
        properties["soft_link_cluster_size"] = len(entities)

        representative["properties"] = properties
        representative["embedding"] = best_embedding
        representative.setdefault("uuid", entities[0].get("uuid"))
        return representative

    def _apply_soft_entity_linking(
        self,
        harmonized_entities: List[Dict[str, Any]],
        entity_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Apply conservative soft canonicalisation before UUID endpoints freeze."""
        if not getattr(self, "enable_soft_entity_linking", False):
            return harmonized_entities, entity_map
        if len(harmonized_entities) < 2:
            return harmonized_entities, entity_map

        generic_types = {"concept", "entity", "unknown", "other", ""}
        candidate_names = {
            entity.get("uuid"): self._entity_candidate_names(entity)
            for entity in harmonized_entities
        }
        alias_keys = {
            entity.get("uuid"): {
                key
                for name in candidate_names.get(entity.get("uuid"), [])
                for key in self._soft_entity_alias_keys(name)
            }
            for entity in harmonized_entities
        }
        embeddings = {
            entity.get("uuid"): entity.get("embedding")
            for entity in harmonized_entities
        }
        missing_embeddings = [
            entity
            for entity in harmonized_entities
            if entity.get("uuid") and embeddings.get(entity.get("uuid")) is None
        ]
        if missing_embeddings and self.embedding_function:
            try:
                generated = self.embedding_function.embed_documents(
                    [str(entity.get("id") or "") for entity in missing_embeddings]
                )
                for entity, embedding in zip(missing_embeddings, generated):
                    entity_uuid = entity.get("uuid")
                    if entity_uuid:
                        embeddings[entity_uuid] = embedding
                        entity["embedding"] = embedding
            except Exception as exc:
                logging.debug("Soft entity linking embeddings unavailable: %s", exc)

        entities = [entity for entity in harmonized_entities if entity.get("uuid")]
        parent = {entity["uuid"]: entity["uuid"] for entity in entities}

        def _find(item: str) -> str:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def _union(left_uuid: str, right_uuid: str) -> None:
            left_root = _find(left_uuid)
            right_root = _find(right_uuid)
            if left_root != right_root:
                parent[left_root] = right_root

        for idx, left_entity in enumerate(entities):
            left_uuid = left_entity["uuid"]
            left_type = str(left_entity.get("type") or "").strip().lower()
            for right_entity in entities[idx + 1:]:
                right_uuid = right_entity["uuid"]
                right_type = str(right_entity.get("type") or "").strip().lower()

                alias_overlap = alias_keys.get(left_uuid, set()) & alias_keys.get(right_uuid, set())
                lexical_guard = bool(alias_overlap)
                if not lexical_guard:
                    lexical_guard = any(
                        self._names_pass_synonym_guard(left_name, right_name)
                        for left_name in candidate_names.get(left_uuid, [])
                        for right_name in candidate_names.get(right_uuid, [])
                    )

                if not lexical_guard:
                    similarity = self._cosine_similarity(
                        embeddings.get(left_uuid),
                        embeddings.get(right_uuid),
                    )
                    if similarity < getattr(self, "soft_entity_similarity_threshold", 0.88):
                        continue
                elif alias_overlap:
                    similarity = 1.0
                else:
                    similarity = self._cosine_similarity(
                        embeddings.get(left_uuid),
                        embeddings.get(right_uuid),
                    )

                if (
                    left_type not in generic_types
                    and right_type not in generic_types
                    and left_type != right_type
                    and not alias_overlap
                ):
                    continue

                if (
                    not alias_overlap
                    and similarity > 0.0
                    and similarity < getattr(self, "soft_entity_similarity_threshold", 0.88)
                ):
                    continue

                _union(left_uuid, right_uuid)

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            grouped[_find(entity["uuid"])].append(entity)

        if not any(len(group) > 1 for group in grouped.values()):
            return harmonized_entities, entity_map

        merged_entities: List[Dict[str, Any]] = []
        remapped_by_uuid: Dict[str, Dict[str, Any]] = {}
        merge_count = 0
        for group in grouped.values():
            if len(group) == 1:
                merged_entities.append(group[0])
                remapped_by_uuid[group[0]["uuid"]] = group[0]
                continue
            representative = self._merge_entity_cluster(group)
            merged_entities.append(representative)
            for entity in group:
                remapped_by_uuid[entity["uuid"]] = representative
            merge_count += len(group) - 1

        updated_entity_map = {
            key: remapped_by_uuid.get(value.get("uuid"), value)
            for key, value in entity_map.items()
        }
        for entity in merged_entities:
            for candidate in self._entity_candidate_names(entity):
                updated_entity_map.setdefault(candidate, entity)
                for variant in self._entity_name_variants(candidate):
                    updated_entity_map.setdefault(variant, entity)
                for scope_key in (entity.get("properties") or {}).get("source_scope_keys") or []:
                    if isinstance(scope_key, str) and scope_key.strip():
                        updated_entity_map.setdefault((scope_key, candidate), entity)

        logging.info(
            "Soft entity linking merged %d entities into %d representatives",
            merge_count,
            len(merged_entities),
        )
        return merged_entities, updated_entity_map

    def _harmonize_entities(self, all_entities: List[Dict], return_id_map: bool = False):
        """
        Harmonize entities across chunks to avoid duplicates using improved normalization
        """
        logging.info(f"Starting harmonization of {len(all_entities)} raw entities")

        # Step 0: Drop generic hub-entity names before any deduplication or
        # grouping.  This prevents high-frequency vague terms from becoming
        # hub nodes that fan out to hundreds of irrelevant chunks at retrieval
        # time.  The filter is conservative: it only removes names that are
        # explicitly blocklisted or shorter than the minimum length threshold.
        filtered_entities = [e for e in all_entities if _is_valid_entity_name(e.get('id', '') or e.get('name', ''))]
        if len(filtered_entities) < len(all_entities):
            logging.info(
                "Hub-entity filter removed %d/%d entities (generic names or too short)",
                len(all_entities) - len(filtered_entities),
                len(all_entities),
            )
        all_entities = filtered_entities

        # Step 1: Build grouping by normalized text, then refine to avoid cross-type collapse.
        #
        # Why text-first, not (text, type):
        #   Using (text, type) directly splits "Prostate Cancer" typed as Disease in chunk 1
        #   and Concept in chunk 2 into two nodes — that is LLM drift, not a real distinction.
        #   Grouping by text first and picking the most specific type handles this correctly.
        #
        # Why we then split by specific type:
        #   Pure text grouping collapses genuinely different entities that happen to share a
        #   surface form — e.g. "depression" (Disease) vs "depression" (GeologicalFeature).
        #   If a text group contains more than one *distinct specific type*, we split it into
        #   per-specific-type buckets; generic-typed occurrences (Concept/Entity/Unknown/Other)
        #   are assigned to the dominant (largest) specific-type bucket so that LLM drift
        #   toward generic labels still merges into the right node rather than floating free.
        _generic_types = {'Concept', 'Entity', 'Unknown', 'Other'}

        text_groups: Dict[str, list] = defaultdict(list)
        for entity in all_entities:
            normalized_text = self._normalize_entity_text(entity['id'])
            text_groups[normalized_text].append(entity)

        # Refine:
        #   1. for bundle-style multihop datasets, preserve article-local title
        #      entities (same surface form, different passage/article scope)
        #   2. split remaining groups that contain multiple distinct specific types
        # dominant_for_surface records which specific type is the largest bucket for
        # each surface form that was split — used below to make entity_map writes
        # deterministic (dominant type always wins, regardless of iteration order).
        entity_groups: Dict = defaultdict(list)
        dominant_for_surface: Dict[str, str] = {}  # norm_text → dominant specific type

        for norm_text, entities in text_groups.items():
            pending_groups: List[Tuple[Any, List[Dict[str, Any]]]] = [(norm_text, list(entities))]
            title_matched_by_scope: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            residual_entities: List[Dict[str, Any]] = []

            for entity in entities:
                properties = dict(entity.get("properties") or {})
                dataset = str(properties.get("dataset") or "").strip().lower()
                scope_key = str(properties.get("source_scope_key") or "").strip()
                if (
                    dataset in self._TITLE_AWARE_BUNDLE_DATASETS
                    and scope_key
                    and self._entity_matches_own_source_title(norm_text, entity)
                ):
                    title_matched_by_scope[scope_key].append(entity)
                else:
                    residual_entities.append(entity)

            if len(title_matched_by_scope) > 1:
                pending_groups = [
                    ((norm_text, "__title_scope__", scope_key), scoped_entities)
                    for scope_key, scoped_entities in sorted(title_matched_by_scope.items())
                ]
                if residual_entities:
                    pending_groups.append((norm_text, residual_entities))
                logging.info(
                    "Surface form '%s' split by title/article scope across %d passage scopes",
                    norm_text,
                    len(title_matched_by_scope),
                )

            for group_key, group_entities in pending_groups:
                specific_types = {
                    e.get('type')
                    for e in group_entities
                    if e.get('type') not in _generic_types
                }
                if len(specific_types) <= 1:
                    entity_groups[group_key].extend(group_entities)
                    continue

                # Multiple distinct specific types → split, generics go to largest bucket.
                type_buckets: Dict[str, list] = defaultdict(list)
                generics = []
                for e in group_entities:
                    if e.get('type') in _generic_types:
                        generics.append(e)
                    else:
                        type_buckets[e['type']].append(e)
                dominant = max(type_buckets, key=lambda t: len(type_buckets[t]))
                if generics:
                    type_buckets[dominant].extend(generics)
                dominant_for_surface[str(group_key)] = dominant
                logging.info(
                    "Surface form '%s' has %d distinct specific types %s; "
                    "relationships will resolve to dominant type '%s' (%d occurrences). "
                    "Add source/target type fields to relationship dicts for per-type resolution.",
                    norm_text, len(type_buckets), sorted(type_buckets),
                    dominant, len(type_buckets[dominant]),
                )
                for stype, bucket in type_buckets.items():
                    entity_groups[(group_key, stype)].extend(bucket)

        # Step 2: Log entity distribution before harmonization
        type_distribution = Counter()
        for entities in entity_groups.values():
            for entity in entities:
                type_distribution[entity['type']] += 1

        logging.info(f"Entity type distribution before harmonization: {dict(type_distribution)}")

        # Step 3: Create harmonized entities
        harmonized_entities = []
        entity_map = {}  # For mapping original IDs to harmonized versions
        total_duplicates_removed = 0

        for normalized_key, entities in entity_groups.items():
            if not entities:
                continue

            dominant_lookup_key = (
                normalized_key[0]
                if isinstance(normalized_key, tuple) and len(normalized_key) == 2
                else normalized_key
            )
            title_scope_key = None
            if (
                isinstance(normalized_key, tuple)
                and len(normalized_key) >= 3
                and normalized_key[1] == "__title_scope__"
            ):
                title_scope_key = str(normalized_key[2])
            elif (
                isinstance(normalized_key, tuple)
                and len(normalized_key) == 2
                and isinstance(normalized_key[0], tuple)
                and len(normalized_key[0]) >= 3
                and normalized_key[0][1] == "__title_scope__"
            ):
                title_scope_key = str(normalized_key[0][2])

            # --- Deterministic deduplication rules ---
            # Rule 1: prefer specific ontology types over generic fallbacks (Concept, Entity, Unknown)
            # Rule 2: among equally specific types, prefer the longest description
            # Rule 3: if still tied, prefer the longest entity name (most fully-qualified)
            # (_generic_types defined at Step 1 above)

            def _type_specificity(e):
                return 0 if e.get('type', 'Concept') in _generic_types else 1

            def _desc_len(e):
                return len(e.get('properties', {}).get('description') or '')

            def _name_len(e):
                return len(e.get('id') or '')

            representative_entity = max(
                entities, key=lambda e: (_type_specificity(e), _desc_len(e), _name_len(e))
            ).copy()

            # Merge information from all occurrences
            all_names = set()
            all_descriptions = set()
            all_anchor_spans: List[Dict[str, Any]] = []
            all_source_scope_keys = set()
            all_source_titles = set()
            all_question_ids = set()
            all_passage_indexes = set()

            for entity in entities:
                # Rule 3: accumulate all surface forms as synonyms
                all_names.add(entity['id'])
                desc = entity.get('properties', {}).get('description')
                if desc:
                    all_descriptions.add(desc if isinstance(desc, str) else str(desc))
                all_anchor_spans = self._merge_anchor_spans(
                    all_anchor_spans,
                    (entity.get("properties") or {}).get("anchor_spans"),
                )
                props = dict(entity.get("properties") or {})
                scope_key = props.get("source_scope_key")
                if scope_key:
                    all_source_scope_keys.add(str(scope_key))
                source_title = props.get("source_title")
                if source_title:
                    all_source_titles.add(str(source_title))
                question_id = props.get("question_id")
                if question_id is not None:
                    all_question_ids.add(str(question_id))
                passage_index = props.get("passage_index")
                if passage_index is not None:
                    all_passage_indexes.add(int(passage_index))

                # Keep the best embedding (prefer any available embedding)
                if not representative_entity.get('embedding') and entity.get('embedding'):
                    representative_entity['embedding'] = entity['embedding']

            # Merge all non-description properties from variants into representative
            for entity in entities:
                if entity.get('properties'):
                    for k, v in entity['properties'].items():
                        if k not in ('description',):  # description already resolved above
                            representative_entity.setdefault('properties', {}).setdefault(k, v)

            # Update representative entity with merged information
            representative_entity.setdefault('properties', {})
            representative_entity['properties'].setdefault('original_ids', [representative_entity['id']])
            if all_anchor_spans:
                representative_entity["properties"]["anchor_spans"] = all_anchor_spans
                representative_entity["properties"]["anchor_mention_count"] = len(all_anchor_spans)
            if all_source_scope_keys:
                representative_entity["properties"]["source_scope_keys"] = sorted(all_source_scope_keys)
            if all_source_titles:
                representative_entity["properties"]["source_titles"] = sorted(all_source_titles)
            if all_question_ids:
                representative_entity["properties"]["question_ids"] = sorted(all_question_ids)
            if all_passage_indexes:
                representative_entity["properties"]["passage_indexes"] = sorted(all_passage_indexes)
            if title_scope_key:
                representative_entity["properties"]["title_entity_scoped"] = True
                representative_entity["properties"]["title_scope_key"] = title_scope_key
                representative_entity["properties"]["disambiguation_scope"] = title_scope_key
            if len(all_names) > 1 or len(all_descriptions) > 1:
                representative_entity['properties']['all_names'] = sorted(all_names)
                representative_entity['properties']['all_descriptions'] = sorted(all_descriptions, key=len, reverse=True)
                representative_entity['properties']['original_ids'] = sorted(all_names)
                total_duplicates_removed += len(entities) - 1

            # Generate deterministic UUID
            entity_uuid = self._generate_entity_id(representative_entity)
            representative_entity['uuid'] = entity_uuid

            harmonized_entities.append(representative_entity)

            # Map all original name variations to the harmonized entity.
            # For split surface forms, entity_map[surface_form] must always point to the
            # dominant-type representative so relationship resolution is deterministic
            # regardless of which order entity_groups iterates the split buckets.
            # The dominant type was pre-computed in the split phase above.
            for entity in entities:
                norm = self._normalize_entity_text(entity['id'])
                is_dominant = (
                    dominant_for_surface.get(str(dominant_lookup_key)) == representative_entity.get('type')
                    if str(dominant_lookup_key) in dominant_for_surface
                    else True  # not a split surface form — always write
                )
                if is_dominant or entity['id'] not in entity_map:
                    entity_map[entity['id']] = representative_entity
                scope_key = str((entity.get("properties") or {}).get("source_scope_key") or "").strip()
                if scope_key:
                    entity_map[(scope_key, entity['id'])] = representative_entity

        logging.info(f"Harmonization complete: {len(harmonized_entities)} entities (removed {total_duplicates_removed} duplicates)")

        # Log final distribution
        final_type_distribution = Counter(e['type'] for e in harmonized_entities)
        logging.info(f"Entity type distribution after harmonization: {dict(final_type_distribution)}")

        harmonized_entities, entity_map = self._apply_soft_entity_linking(
            harmonized_entities,
            entity_map,
        )
        harmonized_entities, entity_map = self._apply_optional_umls_linking(
            harmonized_entities,
            entity_map,
        )

        if return_id_map:
            return harmonized_entities, entity_map  # entity_map: original_id → representative
        return harmonized_entities

    def _harmonize_relationships(self, all_relationships: List[Dict], entity_map: Dict) -> List[Dict]:
        """
        Harmonize relationships across chunks and map to UUID-based entity IDs.

        entity_map may be either:
          - original_id → representative entity  (from _harmonize_entities with return_id_map=True)
          - uuid → entity  (legacy call sites)
        In both cases we build original_to_uuid by walking .values().
        """
        harmonized_relationships = []
        seen_relationships: Dict[str, Dict[str, Any]] = {}

        # Build variant_name → uuid from entity_map.
        # entity_map keys are ALL original variant IDs (every surface form seen during
        # extraction), values are the representative entities with uuid set.
        # Iterating .items() — not .values() — ensures every variant name is covered,
        # so relationships that used a non-canonical spelling are not silently dropped.
        original_to_uuid = {}
        scoped_to_uuid = {}
        candidate_names_by_uuid: Dict[str, set] = defaultdict(set)
        entity_type_by_uuid: Dict[str, str] = {}
        for variant_name, representative in entity_map.items():
            if 'uuid' not in representative:
                logging.warning(f"Entity '{representative.get('id', '?')}' missing uuid in entity_map — skipping in relationship mapping")
                continue
            uuid_value = representative['uuid']
            if representative.get('type'):
                entity_type_by_uuid.setdefault(uuid_value, representative.get('type'))
            if (
                isinstance(variant_name, tuple)
                and len(variant_name) == 2
                and all(isinstance(v, str) for v in variant_name)
            ):
                scope_key, scoped_name = variant_name
                for candidate in self._entity_name_variants(scoped_name):
                    scoped_to_uuid.setdefault((scope_key, candidate), uuid_value)
                    scoped_to_uuid.setdefault((scope_key, candidate.lower()), uuid_value)
                    candidate_names_by_uuid[uuid_value].add(candidate)
                continue
            for candidate in self._entity_name_variants(variant_name) if isinstance(variant_name, str) else []:
                original_to_uuid.setdefault(candidate, uuid_value)
                original_to_uuid.setdefault(candidate.lower(), uuid_value)
                candidate_names_by_uuid[uuid_value].add(candidate)
            for candidate in self._entity_candidate_names(representative):
                for variant in self._entity_name_variants(candidate):
                    original_to_uuid.setdefault(variant, uuid_value)
                    original_to_uuid.setdefault(variant.lower(), uuid_value)
                    candidate_names_by_uuid[uuid_value].add(variant)

        def _lookup_uuid(name: str, preferred_scope: Optional[str] = None):
            """Try increasingly fuzzy lookups before giving up."""
            if not isinstance(name, str) or not name.strip():
                return None
            if preferred_scope:
                for candidate in self._entity_name_variants(name):
                    scoped = (
                        scoped_to_uuid.get((preferred_scope, candidate))
                        or scoped_to_uuid.get((preferred_scope, candidate.lower()))
                    )
                    if scoped:
                        return scoped
            exact = (
                original_to_uuid.get(name)
                or original_to_uuid.get(name.lower())
                or original_to_uuid.get(name.replace('_', ' '))
                or original_to_uuid.get(name.replace('_', ' ').lower())
                or original_to_uuid.get(name.replace(' ', '_'))
                or original_to_uuid.get(name.replace(' ', '_').lower())
                or original_to_uuid.get(self._normalize_entity_text(name))
            )
            if exact:
                return exact

            raw_norm = self._normalize_entity_text(name)
            best_uuid = None
            best_score = 0.0
            second_best = 0.0

            for uuid_value, candidates in candidate_names_by_uuid.items():
                entity_best = 0.0
                for candidate in candidates:
                    cand_norm = self._normalize_entity_text(candidate)
                    if not cand_norm or not raw_norm:
                        continue
                    if cand_norm == raw_norm:
                        entity_best = 1.0
                        break
                    if raw_norm in cand_norm or cand_norm in raw_norm:
                        if min(len(raw_norm), len(cand_norm)) >= 4:
                            entity_best = max(entity_best, 0.94)
                    else:
                        entity_best = max(
                            entity_best,
                            difflib.SequenceMatcher(None, raw_norm, cand_norm).ratio(),
                        )

                if entity_best > best_score:
                    second_best = best_score
                    best_score = entity_best
                    best_uuid = uuid_value
                elif entity_best > second_best:
                    second_best = entity_best

            if best_uuid and best_score >= 0.90 and (best_score - second_best) >= 0.03:
                return best_uuid
            return None

        dropped_unmapped = 0
        dropped_schema_mismatch = 0
        deduped_relationships = 0
        for rel in all_relationships:
            src = rel.get('source')
            tgt = rel.get('target')
            rel_props = dict(rel.get("properties") or {})
            preferred_scope = str(rel_props.get("source_scope_key") or "").strip() or None
            source_uuid = _lookup_uuid(src, preferred_scope=preferred_scope)
            target_uuid = _lookup_uuid(tgt, preferred_scope=preferred_scope)

            if not source_uuid or not target_uuid:
                dropped_unmapped += 1
                logging.warning(
                    "Dropping relationship — entity not found in map: '%s' -[%s]-> '%s'",
                    src, rel.get('type', '?'), tgt,
                )
                continue

            if source_uuid and target_uuid:
                # Create new relationship with UUID-based IDs
                uuid_rel = rel.copy()
                rel_properties = dict(uuid_rel.get('properties') or {})
                if isinstance(src, str):
                    rel_properties.setdefault("source_name", src)
                if isinstance(tgt, str):
                    rel_properties.setdefault("target_name", tgt)
                uuid_rel['properties'] = rel_properties
                uuid_rel['source'] = source_uuid
                uuid_rel['target'] = target_uuid

                # Include negated in the key: A-INHIBITS->B (negated=True) and
                # A-INHIBITS->B (negated=False) are opposite claims and must not collapse.
                _neg = bool(uuid_rel.get('negated', False))
                condition = uuid_rel.get('condition') or rel_properties.get('condition') or ''
                quantitative = uuid_rel.get('quantitative') or rel_properties.get('quantitative') or ''
                rel_type_key = self._canonicalize_relationship_type(
                    uuid_rel.get('type', ''),
                    source_type=entity_type_by_uuid.get(source_uuid),
                    target_type=entity_type_by_uuid.get(target_uuid),
                )
                if not rel_type_key:
                    dropped_schema_mismatch += 1
                    logging.info(
                        "Dropping relationship with no schema-compatible type: '%s' -[%s]-> '%s'",
                        src,
                        rel.get('type', '?'),
                        tgt,
                    )
                    continue
                uuid_rel['type'] = rel_type_key
                rel_key = (
                    f"{source_uuid}:{rel_type_key}:{target_uuid}:{_neg}:"
                    f"{str(condition).strip().lower()}:{str(quantitative).strip().lower()}"
                )

                if rel_key not in seen_relationships:
                    harmonized_relationships.append(uuid_rel)
                    seen_relationships[rel_key] = uuid_rel
                else:
                    deduped_relationships += 1
                    existing_rel = seen_relationships[rel_key]
                    existing_props = dict(existing_rel.get("properties") or {})
                    new_props = dict(uuid_rel.get("properties") or {})
                    merged_anchor_grounding = self._merge_anchor_grounding(
                        existing_props.get("anchor_grounding"),
                        new_props.get("anchor_grounding"),
                    )
                    if merged_anchor_grounding:
                        existing_props["anchor_grounding"] = merged_anchor_grounding
                        restoration = self._restoration_from_anchor_grounding(
                            merged_anchor_grounding
                        )
                        existing_props["restoration_status"] = restoration["status"]
                        existing_props["restoration_verified"] = restoration["verified"]
                        existing_props["restoration_grounded_components"] = restoration["grounded_components"]
                        existing_props["restoration_grounded_count"] = restoration["grounded_count"]
                    for key, value in new_props.items():
                        existing_props.setdefault(key, value)
                    existing_rel["properties"] = existing_props
                    merged_provenance = sorted(
                        {
                            int(pos)
                            for pos in (existing_rel.get("provenance_positions") or [])
                            + (uuid_rel.get("provenance_positions") or [])
                            if isinstance(pos, (int, float))
                        }
                    )
                    if merged_provenance:
                        existing_rel["provenance_positions"] = merged_provenance

        logging.info(
            "Relationship harmonization kept=%d dropped_unmapped=%d dropped_schema_mismatch=%d deduped=%d",
            len(harmonized_relationships),
            dropped_unmapped,
            dropped_schema_mismatch,
            deduped_relationships,
        )
        self._last_relationship_harmonization_stats = {
            "kept": len(harmonized_relationships),
            "dropped_unmapped": dropped_unmapped,
            "dropped_schema_mismatch": dropped_schema_mismatch,
            "deduped": deduped_relationships,
        }

        harmonized_relationships = self._mark_relationship_contradictions(
            harmonized_relationships
        )

        return harmonized_relationships

    def _mark_relationship_contradictions(
        self,
        harmonized_relationships: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Flag relation pairs that appear with both positive and negated polarity."""
        grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for rel in harmonized_relationships:
            properties = dict(rel.get("properties") or {})
            condition = rel.get("condition")
            if condition is None:
                condition = properties.get("condition")
            quantitative = rel.get("quantitative")
            if quantitative is None:
                quantitative = properties.get("quantitative")
            key = (
                str(rel.get("source") or ""),
                str(rel.get("type") or ""),
                str(rel.get("target") or ""),
                str(condition or "").strip().lower(),
                str(quantitative or "").strip().lower(),
            )
            grouped[key].append(rel)

        contradiction_groups = 0
        contradiction_edges = 0
        for rels in grouped.values():
            polarities = {bool(rel.get("negated", False)) for rel in rels}
            if len(polarities) < 2:
                continue
            contradiction_groups += 1
            contradiction_edges += len(rels)
            for rel in rels:
                rel_props = dict(rel.get("properties") or {})
                rel_props["contradiction_detected"] = True
                rel["properties"] = rel_props
                rel["contradiction_detected"] = True

        self._last_relationship_contradiction_stats = {
            "contradiction_groups": contradiction_groups,
            "contradiction_edges": contradiction_edges,
        }
        return harmonized_relationships

    def _graph_components_from_kg(self, kg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return connected components over the entity-relation graph."""
        nodes = list(kg.get("nodes") or [])
        relationships = list(kg.get("relationships") or [])
        node_lookup = {str(node.get("id") or ""): node for node in nodes if node.get("id")}
        entity_ids = [entity_id for entity_id in node_lookup.keys() if entity_id]
        adjacency: Dict[str, set] = {entity_id: set() for entity_id in entity_ids}

        for rel in relationships:
            source_id = str(rel.get("source") or rel.get("from") or "").strip()
            target_id = str(rel.get("target") or rel.get("to") or "").strip()
            if source_id in adjacency and target_id in adjacency:
                adjacency[source_id].add(target_id)
                adjacency[target_id].add(source_id)

        visited = set()
        components: List[Dict[str, Any]] = []
        for entity_id in entity_ids:
            if entity_id in visited:
                continue
            stack = [entity_id]
            component_ids: List[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_ids.append(current)
                stack.extend(sorted(adjacency.get(current, set()) - visited))

            component_set = set(component_ids)
            component_relationships = [
                rel for rel in relationships
                if str(rel.get("source") or rel.get("from") or "").strip() in component_set
                and str(rel.get("target") or rel.get("to") or "").strip() in component_set
            ]
            degree = Counter()
            for rel in component_relationships:
                source_id = str(rel.get("source") or rel.get("from") or "").strip()
                target_id = str(rel.get("target") or rel.get("to") or "").strip()
                if source_id:
                    degree[source_id] += 1
                if target_id:
                    degree[target_id] += 1

            top_entities = sorted(
                component_ids,
                key=lambda entity_uuid: (
                    -degree.get(entity_uuid, 0),
                    str((node_lookup.get(entity_uuid, {}).get("properties") or {}).get("name") or entity_uuid),
                ),
            )[: max(1, getattr(self, "max_summary_entities", 6))]
            top_relationships = sorted(
                component_relationships,
                key=lambda rel: (
                    -self._confidence_score((rel.get("properties") or {}).get("confidence")),
                    str(rel.get("type") or ""),
                ),
            )[: max(1, getattr(self, "max_summary_relationships", 6))]
            components.append(
                {
                    "entity_ids": sorted(component_ids),
                    "relationships": top_relationships,
                    "entity_count": len(component_ids),
                    "relationship_count": len(component_relationships),
                    "degree": degree,
                    "top_entities": top_entities,
                }
            )

        return components

    def _build_graph_enrichment_records(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """Create optional component summaries, claims, and fragmentation bridges."""
        if not (
            getattr(self, "enable_graph_summaries", False)
            or getattr(self, "enable_claim_extraction", False)
            or getattr(self, "enable_fragmentation_repair", False)
        ):
            return {}

        node_lookup = {
            str(node.get("id") or ""): node
            for node in (kg.get("nodes") or [])
            if node.get("id")
        }
        components = self._graph_components_from_kg(kg)
        include_component_summaries = bool(
            getattr(self, "enable_graph_summaries", False)
        )
        component_summaries: List[Dict[str, Any]] = []
        all_claims: List[Dict[str, Any]] = []

        for component_index, component in enumerate(components):
            if include_component_summaries:
                summary_id = hashlib.sha1(
                    f"{kg.get('metadata', {}).get('kg_name', '')}:component:{component_index}".encode("utf-8")
                ).hexdigest()
                summary_text = self._heuristic_component_summary_text(component, node_lookup)
                component_record = {
                    "id": summary_id,
                    "component_index": component_index,
                    "entity_ids": list(component.get("entity_ids") or []),
                    "top_entities": list(component.get("top_entities") or []),
                    "entity_count": int(component.get("entity_count") or 0),
                    "relationship_count": int(component.get("relationship_count") or 0),
                    "text": summary_text,
                }
                component_summaries.append(component_record)
            all_claims.extend(
                self._claim_records_from_component(
                    component,
                    node_lookup,
                    component_index=component_index,
                )
            )

        graph_summary = None
        if getattr(self, "enable_graph_summaries", False):
            preview = [
                f"Component {record['component_index'] + 1}: {record['entity_count']} entities, {record['relationship_count']} relationships"
                for record in component_summaries[:3]
            ]
            graph_summary = {
                "id": hashlib.sha1(
                    f"{kg.get('metadata', {}).get('kg_name', '')}:graph_summary".encode("utf-8")
                ).hexdigest(),
                "text": (
                    f"Graph contains {len(component_summaries)} connected components. "
                    + " ".join(preview)
                ).strip(),
                "component_count": len(component_summaries),
            }

        return {
            "component_summaries": component_summaries,
            "claims": all_claims,
            "graph_summary": graph_summary,
            "fragmentation_bridges": self._propose_fragmentation_bridges(
                components,
                node_lookup,
            ),
        }

    @staticmethod
    def _attach_relationship_provenance(
        relationships: List[Dict],
        chunk_positions: List[int],
    ) -> List[Dict]:
        """Attach local chunk provenance so triple verification can stay passage-local."""
        annotated: List[Dict] = []
        normalized_positions = [
            int(pos)
            for pos in chunk_positions
            if isinstance(pos, (int, float))
        ]
        for rel in relationships or []:
            if not isinstance(rel, dict):
                continue
            rel_copy = dict(rel)
            existing_positions = rel_copy.get("provenance_positions") or []
            merged_positions = {
                int(pos)
                for pos in existing_positions
                if isinstance(pos, (int, float))
            }
            merged_positions.update(normalized_positions)
            rel_copy["provenance_positions"] = sorted(merged_positions)
            annotated.append(rel_copy)
        return annotated

    @staticmethod
    def _relationship_local_provenance(
        rel: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, List[Any]]:
        """
        Resolve question-local provenance for a relationship from chunk positions.

        Older KGs only expose chunk positions on relationships. Newer builds also
        stamp question and passage provenance onto the edge so question-scoped
        retrieval can be enforced exactly at query time.
        """
        positions = {
            int(pos)
            for pos in (rel.get("provenance_positions") or [])
            if isinstance(pos, (int, float))
        }
        if not positions:
            return {
                "provenance_positions": [],
                "question_ids": [],
                "passage_keys": [],
            }

        question_ids = set()
        passage_keys = set()
        for chunk in chunks or []:
            pos = chunk.get("position")
            if not isinstance(pos, (int, float)) or int(pos) not in positions:
                continue
            qid = chunk.get("question_id")
            if qid is not None and str(qid).strip():
                question_ids.add(str(qid))
            passage_key = f"{chunk.get('question_id', '')}::p{chunk.get('passage_index', -1)}"
            passage_keys.add(passage_key)

        return {
            "provenance_positions": sorted(positions),
            "question_ids": sorted(question_ids),
            "passage_keys": sorted(passage_keys),
        }

    def _compute_node_specificity_weights(self, graph, kg_name: Optional[str]) -> None:
        """Compute HippoRAG-style node specificity within the active KG scope."""
        graph.query(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
            WHERE $kg_name IS NULL OR d.kgName = $kg_name
            MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            WITH e, count(DISTINCT c) AS passage_count
            SET e.passage_count = passage_count,
                e.node_specificity = CASE
                    WHEN passage_count > 0 THEN 1.0 / toFloat(passage_count)
                    ELSE 1.0
                END
            """,
            {"kg_name": kg_name},
        )

    def generate_knowledge_graph(self, text: str, llm, file_name: str = None, model_name: str = "openai/gpt-oss-120b:free", max_chunks: int = None, kg_name: str = None, doc_metadata: dict = None, doc_hash: str = None) -> Dict[str, Any]:
        """
        Generate knowledge graph from text with ontology-guided entity extraction

        Args:
            text: Input text to process
            llm: LLM provider instance
            file_name: Optional filename for storage
            model_name: LLM model name
            max_chunks: Maximum number of chunks to process (for large documents)
        """
        logging.info("Starting ontology-guided knowledge graph generation")

        # Determine if ontology is available
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)
        extraction_method = "ontology_guided_llm" if has_ontology else "natural_llm"
        logging.info(f"Extraction method: {extraction_method}")

        # Step 1b: Detect section headers once across the full document.
        # Each chunk is tagged with its section (e.g. "Methods", "Results") so
        # the extraction LLM can interpret claims in the correct context.
        section_headers = self._detect_section_headers(text)
        logging.info("Detected %d section headers: %s", len(section_headers),
                     [name for _, name in section_headers])

        # Step 1: Chunk the text, keeping detected document sections separate when
        # possible so extraction never straddles unrelated headers.
        chunks = self._chunk_text_with_section_boundaries(
            text,
            section_headers=section_headers,
        )
        logging.info(f"Created {len(chunks)} chunks")

        # Limit chunks if specified (for very large documents)
        if max_chunks and len(chunks) > max_chunks:
            logging.warning(f"Limiting processing to {max_chunks} chunks out of {len(chunks)} total")
            chunks = chunks[:max_chunks]

        # Step 2: Extract entities and relationships from each chunk
        all_entities = []
        all_relationships = []
        processed_chunks = 0
        failed_chunks = 0
        entities_per_chunk: Dict[int, List[Dict]] = {}  # chunk index → entities (for cross-chunk pass)

        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {i+1}/{len(chunks)}")

                # --- Context enrichment ---
                # 1. Section header for this chunk
                chunk_section = self._get_section_for_position(
                    chunk.get("start_pos", 0), section_headers
                )
                chunk["section_name"] = chunk_section or "Preamble"

                # 2. Qualifier sentences from the previous chunk
                qualifier_ctx = ""
                if i > 0:
                    qualifier_ctx = self._extract_qualifier_sentences(chunks[i - 1]["text"])

                previous_chunk_indexes = list(range(max(0, i - 2), i))
                previous_entities = [
                    entity
                    for prev_idx in previous_chunk_indexes
                    for entity in entities_per_chunk.get(prev_idx, [])
                ]
                previous_texts = [
                    chunks[prev_idx]["text"]
                    for prev_idx in previous_chunk_indexes
                ]
                extraction_text = self._prepare_chunk_text_for_extraction(
                    chunk["text"],
                    previous_entities=previous_entities,
                    previous_texts=previous_texts,
                    llm=llm,
                    model_name=model_name,
                )

                chunk_kg = self._extract_entities_and_relationships_with_llm(
                    extraction_text,
                    llm,
                    model_name,
                    context_header=qualifier_ctx or None,
                    section_header=chunk_section,
                )
                chunk_kg = self._ground_chunk_extraction(chunk_kg, chunk)

                # Ensure chunk_kg is a dictionary with expected keys
                try:
                    if isinstance(chunk_kg, dict) and (chunk_kg.get('entities', []) or chunk_kg.get('relationships', [])):
                        chunk_entities = chunk_kg.get('entities', [])
                        all_entities.extend(chunk_entities)
                        all_relationships.extend(
                            self._attach_relationship_provenance(
                                chunk_kg.get('relationships', []),
                                [chunk.get("position", i)],
                            )
                        )
                        entities_per_chunk[i] = chunk_entities
                        processed_chunks += 1
                        logging.info(f"✓ Chunk {i+1} processed: {len(chunk_entities)} entities, {len(chunk_kg.get('relationships', []))} relationships")
                    else:
                        logging.warning(f"⚠ Chunk {i+1} returned invalid or empty format: {type(chunk_kg)}")
                        failed_chunks += 1
                except Exception as inner_e:
                    logging.error(f"❌ Error processing chunk result: {inner_e}")
                    failed_chunks += 1

            except Exception as e:
                logging.error(f"❌ Failed to process chunk {i+1}: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
                failed_chunks += 1
                continue

            # Add small delay between chunks only when an external LLM is in use
            if llm is not None and i < len(chunks) - 1:  # Don't delay after the last chunk
                time.sleep(1.0)  # 1 second delay between API calls

        logging.info(f"Processing complete: {processed_chunks} successful, {failed_chunks} failed")

        # Step 2b: Cross-chunk relationship extraction across wider adjacent windows.
        # The per-chunk LLM pass can only see entities within one chunk at a time, so relationships
        # that span chunk boundaries are otherwise lost. We recover them with a relation-only pass
        # over adjacent windows (default size 2-3), which improves recall without re-running
        # full entity extraction.
        if llm is not None and len(chunks) > 1:
            chunk_positions = {
                i: [int(chunks[i].get("position", i))]
                for i in range(len(chunks))
            }
            all_relationships.extend(
                self._extract_relationships_for_segment_windows(
                    [chunk.get("text", "") for chunk in chunks],
                    entities_per_chunk,
                    chunk_positions,
                    llm,
                    model_name,
                    max_window_size=getattr(self, "cross_chunk_relation_window", 3),
                    scope_label="Cross-chunk",
                )
            )

            all_relationships.extend(
                self._recover_cross_section_relationships(
                    text=text,
                    chunks=chunks,
                    entities_per_chunk=entities_per_chunk,
                    section_headers=section_headers,
                    llm=llm,
                    model_name=model_name,
                    scope_label="Cross-section",
                )
            )

        if processed_chunks == 0 and failed_chunks > 0:
            raise RuntimeError(
                f"KG extraction failed: all {failed_chunks} chunk(s) returned errors. "
                "Check LLM connectivity and rate limits."
            )

        # Step 3: Harmonize entities and relationships.
        # _harmonize_entities returns the representative entities AND builds the
        # full original-ID → representative mapping internally (stored on the
        # entities as entity['_all_ids'] is NOT available, so we rebuild it here).
        # We need every original variant ID to map to its representative so that
        # relationships whose source/target used a non-canonical name aren't dropped.
        harmonized_entities, id_to_representative = self._harmonize_entities(all_entities, return_id_map=True)
        harmonized_relationships = self._harmonize_relationships(all_relationships, id_to_representative)
        harmonized_entities, harmonized_relationships = self._coerce_harmonized_entities_to_schema(
            harmonized_entities,
            harmonized_relationships,
        )
        harmonized_relationships = self._mark_relationship_contradictions(
            harmonized_relationships
        )

        logging.info(f"Harmonized to {len(harmonized_entities)} entities and {len(harmonized_relationships)} relationships")

        # Step 4: Format the final knowledge graph
        # Use UUID-based IDs to prevent duplicates
        kg_prefix = f"{kg_name}_" if kg_name else ""

        kg = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['uuid']}",
                    "label": entity['type'],
                    "properties": {
                        "name": entity['id'],
                        "type": entity['type'],
                        "original_id": entity['id'],  # Keep original ID for reference
                        **entity.get('properties', {})
                    },
                    "embedding": entity.get('embedding'),
                    "color": self._get_node_color(entity['type']),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": f"Entity: {entity['id']}\nType: {entity['type']}\nKG: {kg_name or 'default'}\nClick for details"
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{rel['source']}_{rel['type']}_{rel['target']}_{idx}",
                    "from": f"{kg_prefix}{rel['source']}",
                    "to": f"{kg_prefix}{rel['target']}",
                    "source": f"{kg_prefix}{rel['source']}",
                    "target": f"{kg_prefix}{rel['target']}",
                    "type": rel['type'],
                    "label": rel['type'],
                    "negated": rel.get('negated', False),
                    "properties": rel.get('properties', {}),
                    "arrows": "to",
                    "color": {"color": "#444444"},
                    "font": {"size": 12, "align": "middle"}
                }
                for idx, rel in enumerate(harmonized_relationships)
            ],
            "chunks": chunks,
            "metadata": {
                "total_chunks": len(chunks),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": "ontology_guided_llm" if has_ontology else "natural_llm",
                "kg_name": kg_name,
                "provider": provider,
                "model": model_name,
                "max_chunks_setting": max_chunks,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name,
                "doc_hash": doc_hash,
                "schema_card": self._build_schema_card()
            }
        }
        kg["metadata"].update({
            "schema_enforcement_dropped_entities": self._last_schema_enforcement_stats.get("dropped_entities", 0),
            "schema_enforcement_dropped_relationships": self._last_schema_enforcement_stats.get("dropped_relationships", 0),
            "harmonization_relationships_dropped_unmapped": self._last_relationship_harmonization_stats.get("dropped_unmapped", 0),
            "harmonization_relationships_dropped_schema_mismatch": self._last_relationship_harmonization_stats.get("dropped_schema_mismatch", 0),
            "harmonization_relationships_deduped": self._last_relationship_harmonization_stats.get("deduped", 0),
            "harmonization_relationship_contradiction_groups": self._last_relationship_contradiction_stats.get("contradiction_groups", 0),
            "harmonization_relationship_contradiction_edges": self._last_relationship_contradiction_stats.get("contradiction_edges", 0),
            "anchor_grounded_entities": sum(
                1 for entity in harmonized_entities
                if (entity.get("properties") or {}).get("anchor_spans")
            ),
            "restoration_full_relationships": sum(
                1 for rel in harmonized_relationships
                if ((rel.get("properties") or {}).get("restoration_status") == "full")
            ),
            "restoration_partial_relationships": sum(
                1 for rel in harmonized_relationships
                if ((rel.get("properties") or {}).get("restoration_status") == "partial")
            ),
        })
        enrichment = self._build_graph_enrichment_records(kg)
        if enrichment:
            kg["metadata"]["graph_enrichment"] = enrichment
            kg["metadata"]["component_summary_count"] = len(enrichment.get("component_summaries") or [])
            kg["metadata"]["claim_count"] = len(enrichment.get("claims") or [])
            kg["metadata"]["fragmentation_bridge_count"] = len(enrichment.get("fragmentation_bridges") or [])

        # Attach any document-level metadata from the source (e.g. CSV columns)
        if doc_metadata:
            kg['metadata']['doc_metadata'] = doc_metadata

        # Step 5: Store in Neo4j if requested
        if file_name:
            success = self.store_knowledge_graph_with_embeddings(
                kg,
                file_name,
                doc_metadata=doc_metadata,
                doc_hash=doc_hash,
                llm=llm,
                model_name=model_name,
            )
            kg['metadata']['stored_in_neo4j'] = success

            # Step 5b: HippoRAG-style synonym merging.
            # After embeddings are in Neo4j, cluster near-duplicate entity nodes
            # (e.g. "TBK1" / "TBK1 kinase") and merge them so entity-first search
            # can find all surface-form variants via a single canonical node.
            if success:
                try:
                    graph_for_merge = self._create_neo4j_connection()
                    merges = self.merge_synonym_entities(graph_for_merge, kg_name=kg_name)
                    kg['metadata']['synonym_merges'] = merges
                    logging.info(f"Synonym merging complete: {merges} pairs merged")
                except Exception as syn_err:
                    logging.warning(f"Synonym merging failed (non-fatal): {syn_err}")

                # Compute node specificity (HippoRAG-style IDF weight):
                # s(e) = 1 / |passages containing e|.  Stored on each entity node
                # so that retrieval can down-weight ubiquitous hub entities as seeds.
                try:
                    graph_for_merge = self._create_neo4j_connection()
                    self._compute_node_specificity_weights(graph_for_merge, kg_name)
                    logging.info("Node specificity weights computed for kg_name=%s", kg_name)
                except Exception as spec_err:
                    logging.warning("Node specificity computation failed (non-fatal): %s", spec_err)

        return kg

    def generate_knowledge_graph_from_passages(
        self,
        passages,           # List[ContextPassage] — avoids circular import; duck-typed
        llm,
        file_name: str = None,
        model_name: str = "openai/gpt-oss-120b:free",
        kg_name: str = None,
        doc_metadata: dict = None,
        doc_hash: str = None,
    ) -> Dict[str, Any]:
        """Generate a KG from a list of ContextPassage objects.

        Unlike generate_knowledge_graph, this method never concatenates passages
        from different records before chunking.  Each passage is chunked
        independently; sub-splitting only occurs when a single passage exceeds
        chunk_size.  Cross-chunk relationship extraction is scoped within each
        passage so the LLM never sees entity pairs that are only co-located
        because two unrelated passages were glued together.

        Harmonisation and Neo4j storage happen once after all passages are
        processed, so the result is equivalent in structure to the single-call
        path but with clean passage-level extraction boundaries.
        """
        logging.info(
            "Starting passage-aware KG generation: %d passages", len(passages)
        )
        has_ontology = bool(self.ontology_classes) or bool(self.ontology_relationships)
        extraction_method = "ontology_guided_llm" if has_ontology else "natural_llm"

        all_chunks: List[Dict] = []
        all_entities: List[Dict] = []
        all_relationships: List[Dict] = []
        passage_entities: Dict[int, List[Dict[str, Any]]] = {}
        passage_positions: Dict[int, List[int]] = {}

        global_chunk_offset = 0  # running count for position uniqueness across passages

        for p_idx, passage in enumerate(passages):
            passage_text = passage.text
            source_label = f"{passage.dataset}/{passage.question_id}/p{passage.passage_index}"
            source_title = (
                getattr(passage, "source_title", None)
                or self._infer_passage_title(passage_text, passage.dataset)
            )
            source_scope_key = self._build_passage_scope_key(
                dataset=passage.dataset,
                question_id=passage.question_id,
                passage_index=passage.passage_index,
            )

            # Shift position fields so they are globally unique across passages.
            section_headers = self._detect_section_headers(passage_text)
            p_chunks = self._chunk_text_with_section_boundaries(
                passage_text,
                section_headers=section_headers,
            )
            for local_i, ch in enumerate(p_chunks):
                ch["position"] = global_chunk_offset + local_i
                ch["source"] = source_label
                ch["dataset"] = passage.dataset
                ch["question_id"] = passage.question_id
                ch["passage_index"] = passage.passage_index
                ch["chunk_local_index"] = ch.get("chunk_id", local_i)
                ch["source_title"] = source_title
                ch["source_scope_key"] = source_scope_key

            entities_this_passage: Dict[int, List[Dict]] = {}
            processed = 0
            failed = 0

            for local_i, chunk in enumerate(p_chunks):
                global_i = global_chunk_offset + local_i
                try:
                    chunk_section = self._get_section_for_position(
                        chunk.get("start_pos", 0), section_headers
                    )
                    chunk["section_name"] = chunk_section or "Preamble"
                    qualifier_ctx = ""
                    if local_i > 0:
                        qualifier_ctx = self._extract_qualifier_sentences(
                            p_chunks[local_i - 1]["text"]
                        )
                    previous_local_indexes = list(range(max(0, local_i - 2), local_i))
                    previous_entities = [
                        entity
                        for prev_idx in previous_local_indexes
                        for entity in entities_this_passage.get(prev_idx, [])
                    ]
                    previous_texts = [
                        p_chunks[prev_idx]["text"]
                        for prev_idx in previous_local_indexes
                    ]
                    extraction_text = self._prepare_chunk_text_for_extraction(
                        chunk["text"],
                        previous_entities=previous_entities,
                        previous_texts=previous_texts,
                        llm=llm,
                        model_name=model_name,
                    )

                    chunk_kg = self._extract_entities_and_relationships_with_llm(
                        extraction_text, llm, model_name,
                        context_header=qualifier_ctx or None,
                        section_header=chunk_section,
                    )
                    chunk_kg = self._ground_chunk_extraction(chunk_kg, chunk)
                    if isinstance(chunk_kg, dict) and (
                        chunk_kg.get("entities") or chunk_kg.get("relationships")
                    ):
                        chunk_entities = chunk_kg.get("entities", [])
                        all_entities.extend(chunk_entities)
                        all_relationships.extend(
                            self._attach_relationship_provenance(
                                chunk_kg.get("relationships", []),
                                [chunk.get("position", global_i)],
                            )
                        )
                        entities_this_passage[local_i] = chunk_entities
                        processed += 1
                        logging.info(
                            "Passage %d/%d chunk %d/%d: %d entities, %d rels [%s]",
                            p_idx + 1, len(passages),
                            local_i + 1, len(p_chunks),
                            len(chunk_entities),
                            len(chunk_kg.get("relationships", [])),
                            source_label,
                        )
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(
                        "Passage %d chunk %d failed: %s", p_idx + 1, local_i + 1, e
                    )
                    failed += 1
                    continue

                if llm is not None and local_i < len(p_chunks) - 1:
                    time.sleep(1.0)

            # Cross-chunk relationship extraction — scoped within this passage only.
            if llm is not None and len(p_chunks) > 1:
                chunk_positions_this_passage = {
                    local_i: [int(p_chunks[local_i].get("position", global_chunk_offset + local_i))]
                    for local_i in range(len(p_chunks))
                }
                all_relationships.extend(
                    self._extract_relationships_for_segment_windows(
                        [chunk.get("text", "") for chunk in p_chunks],
                        entities_this_passage,
                        chunk_positions_this_passage,
                        llm,
                        model_name,
                        max_window_size=getattr(self, "cross_chunk_relation_window", 3),
                        scope_label=f"Cross-chunk[{source_label}]",
                        relationship_scope_metadata={
                            "dataset": passage.dataset,
                            "question_id": str(passage.question_id),
                            "passage_index": int(passage.passage_index),
                            "source": source_label,
                            "source_title": source_title,
                            "source_scope_key": source_scope_key,
                        },
                    )
                )
                all_relationships.extend(
                    self._recover_cross_section_relationships(
                        text=passage_text,
                        chunks=p_chunks,
                        entities_per_chunk=entities_this_passage,
                        section_headers=section_headers,
                        llm=llm,
                        model_name=model_name,
                        scope_label=f"Cross-section[{source_label}]",
                    )
                )

            passage_entities[p_idx] = [
                entity
                for entity_list in entities_this_passage.values()
                for entity in entity_list
            ]
            passage_positions[p_idx] = [
                int(chunk.get("position", global_chunk_offset + local_i))
                for local_i, chunk in enumerate(p_chunks)
            ]

            all_chunks.extend(p_chunks)
            global_chunk_offset += len(p_chunks)

            logging.info(
                "Passage %d/%d complete: processed=%d failed=%d [%s]",
                p_idx + 1, len(passages), processed, failed, source_label,
            )

        logging.info(
            "All passages processed: %d chunks, %d raw entities, %d raw relationships",
            len(all_chunks), len(all_entities), len(all_relationships),
        )

        if not all_entities and not all_relationships:
            raise RuntimeError(
                "Passage-aware KG extraction yielded no entities or relationships. "
                "Check LLM connectivity and passage content."
            )

        if (
            llm is not None
            and len(passages) > 1
            and getattr(self, "enable_cross_passage_relation_recovery", True)
        ):
            cross_passage_groups = self._group_passages_for_cross_passage_recovery(
                passages,
                passage_entities,
                passage_positions,
            )
            for group in cross_passage_groups:
                if len(group["texts"]) < 2:
                    continue
                all_relationships.extend(
                    self._extract_relationships_for_segment_windows(
                        group["texts"],
                        group["entities"],
                        group["positions"],
                        llm,
                        model_name,
                        max_window_size=getattr(self, "cross_passage_relation_window", 2),
                        scope_label=group["source_label"],
                    )
                )

        # Harmonise globally across all passages.
        harmonized_entities, id_to_representative = self._harmonize_entities(
            all_entities, return_id_map=True
        )
        harmonized_relationships = self._harmonize_relationships(
            all_relationships, id_to_representative
        )
        harmonized_entities, harmonized_relationships = self._coerce_harmonized_entities_to_schema(
            harmonized_entities,
            harmonized_relationships,
        )
        harmonized_relationships = self._mark_relationship_contradictions(
            harmonized_relationships
        )
        logging.info(
            "Harmonized: %d entities, %d relationships",
            len(harmonized_entities), len(harmonized_relationships),
        )

        kg_prefix = f"{kg_name}_" if kg_name else ""
        kg = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['uuid']}",
                    "label": entity["type"],
                    "properties": {
                        "name": entity["id"],
                        "type": entity["type"],
                        "original_id": entity["id"],
                        **entity.get("properties", {}),
                    },
                    "embedding": entity.get("embedding"),
                    "color": self._get_node_color(entity["type"]),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": (
                        f"Entity: {entity['id']}\nType: {entity['type']}"
                        f"\nKG: {kg_name or 'default'}\nClick for details"
                    ),
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{rel['source']}_{rel['type']}_{rel['target']}_{idx}",
                    "from": f"{kg_prefix}{rel['source']}",
                    "to": f"{kg_prefix}{rel['target']}",
                    "source": f"{kg_prefix}{rel['source']}",
                    "target": f"{kg_prefix}{rel['target']}",
                    "type": rel["type"],
                    "label": rel["type"],
                    "negated": rel.get("negated", False),
                    "properties": rel.get("properties", {}),
                    "provenance_positions": rel.get("provenance_positions") or [],
                    "arrows": "to",
                    "color": {"color": "#444444"},
                    "font": {"size": 12, "align": "middle"},
                }
                for idx, rel in enumerate(harmonized_relationships)
            ],
            "chunks": all_chunks,
            "metadata": {
                "total_chunks": len(all_chunks),
                "total_passages": len(passages),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": extraction_method,
                "kg_name": kg_name,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name,
                "doc_hash": doc_hash,
            },
        }
        kg["metadata"].update({
            "schema_enforcement_dropped_entities": self._last_schema_enforcement_stats.get("dropped_entities", 0),
            "schema_enforcement_dropped_relationships": self._last_schema_enforcement_stats.get("dropped_relationships", 0),
            "harmonization_relationships_dropped_unmapped": self._last_relationship_harmonization_stats.get("dropped_unmapped", 0),
            "harmonization_relationships_dropped_schema_mismatch": self._last_relationship_harmonization_stats.get("dropped_schema_mismatch", 0),
            "harmonization_relationships_deduped": self._last_relationship_harmonization_stats.get("deduped", 0),
            "harmonization_relationship_contradiction_groups": self._last_relationship_contradiction_stats.get("contradiction_groups", 0),
            "harmonization_relationship_contradiction_edges": self._last_relationship_contradiction_stats.get("contradiction_edges", 0),
            "anchor_grounded_entities": sum(
                1 for entity in harmonized_entities
                if (entity.get("properties") or {}).get("anchor_spans")
            ),
            "restoration_full_relationships": sum(
                1 for rel in harmonized_relationships
                if ((rel.get("properties") or {}).get("restoration_status") == "full")
            ),
            "restoration_partial_relationships": sum(
                1 for rel in harmonized_relationships
                if ((rel.get("properties") or {}).get("restoration_status") == "partial")
            ),
        })
        enrichment = self._build_graph_enrichment_records(kg)
        if enrichment:
            kg["metadata"]["graph_enrichment"] = enrichment
            kg["metadata"]["component_summary_count"] = len(enrichment.get("component_summaries") or [])
            kg["metadata"]["claim_count"] = len(enrichment.get("claims") or [])
            kg["metadata"]["fragmentation_bridge_count"] = len(enrichment.get("fragmentation_bridges") or [])

        if doc_metadata:
            kg["metadata"]["doc_metadata"] = doc_metadata

        if file_name:
            success = self.store_knowledge_graph_with_embeddings(
                kg,
                file_name,
                doc_metadata=doc_metadata,
                doc_hash=doc_hash,
                llm=llm,
                model_name=model_name,
            )
            kg["metadata"]["stored_in_neo4j"] = success
            if success:
                try:
                    graph_for_merge = self._create_neo4j_connection()
                    merges = self.merge_synonym_entities(
                        graph_for_merge, kg_name=kg_name
                    )
                    kg["metadata"]["synonym_merges"] = merges
                    logging.info("Synonym merging complete: %d pairs merged", merges)
                except Exception as syn_err:
                    logging.warning("Synonym merging failed (non-fatal): %s", syn_err)

                try:
                    graph_for_merge = self._create_neo4j_connection()
                    self._compute_node_specificity_weights(graph_for_merge, kg_name)
                    logging.info("Node specificity weights computed for kg_name=%s", kg_name)
                except Exception as spec_err:
                    logging.warning("Node specificity computation failed (non-fatal): %s", spec_err)

        return kg

    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            "Disease": "#ff6666",
            "Treatment": "#66ff66",
            "Drug": "#66ffff",
            "Symptom": "#ffff66",
            "Patient": "#ff9999",
            "Physician": "#ffcc99",
            "Hospital": "#99ccff",
            "MedicalProcedure": "#cc99ff",
            "MedicalDevice": "#ff99cc",
            "Anatomy": "#99ff99",
            "Concept": "#a6cee3"
        }
        return color_map.get(entity_type, "#a6cee3")

    def get_rag_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get RAG context for a query using vector similarity search
        """
        try:
            graph = self._create_neo4j_connection()

            # Generate query embedding
            query_embedding = self.embedding_function.embed_query(query)

            # Vector search query
            search_query = """
            CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
            YIELD node AS chunk, score
            MATCH (chunk)-[:PART_OF]->(d:Document)
            OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
            WITH chunk, score, d, collect(e) AS entities
            RETURN
                chunk.text AS text,
                chunk.id AS chunk_id,
                score,
                d.fileName AS document,
                [entity IN entities | {id: entity.id, type: entity.type, description: entity.description}] AS entities
            ORDER BY score DESC
            """

            results = graph.query(search_query, {
                "top_k": top_k,
                "query_vector": query_embedding
            })

            context = {
                "query": query,
                "chunks": [],
                "entities": set(),
                "documents": set()
            }

            for result in results:
                context["chunks"].append({
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "score": result["score"],
                    "document": result["document"],
                    "entities": result["entities"]
                })

                context["documents"].add(result["document"])
                for entity in result["entities"]:
                    context["entities"].add(entity["id"])

            context["entities"] = list(context["entities"])
            context["documents"] = list(context["documents"])

            return context

        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {"query": query, "chunks": [], "entities": [], "documents": []}
