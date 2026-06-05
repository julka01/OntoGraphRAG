import json
import re
import hashlib
import difflib
import importlib
import xml.etree.ElementTree as ET
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ontographrag.kg.chunking import chunk_text as _chunk_text_fn
from collections import defaultdict, Counter
import os
import sys
import logging
import time
import math

# Import from local kg_utils
from ontographrag.kg.utils.common_functions import load_embedding_model
from ontographrag.schemas.models import (
    OntologySchema, EntityType as OntEntityType, RelationshipType as OntRelType,
    DataBinding, RelationshipAttribute, PropertyType,
)

# Minimum character length for an entity name (after stripping whitespace).
# Single-character or two-character tokens are almost always stop-words or
# artefacts; named identifiers below this threshold (e.g. "p53", "IL-6") are
# kept via the allowlist below.
_ENTITY_MIN_NAME_LENGTH: int = 3

# Lowercase terms that must not become standalone KG nodes regardless of how
# the LLM classifies them.  These are generic process/state words that appear
# in almost every biomedical chunk and, when extracted as entities, become hub
# nodes fanning out to hundreds of irrelevant chunks during retrieval.
_GENERIC_HUB_ENTITY_BLOCKLIST: frozenset = frozenset({
    "treatment", "treatments", "condition", "conditions", "outcome", "outcomes",
    "model", "models", "effect", "effects", "result", "results", "factor",
    "factors", "mechanism", "mechanisms", "response", "responses", "function",
    "functions", "role", "roles", "process", "processes", "level", "levels",
    "system", "systems", "study", "studies", "analysis", "analyses",
    "approach", "approaches", "method", "methods", "measure", "measures",
    "group", "groups", "sample", "samples", "data", "finding", "findings",
    "evidence", "activity", "activities", "expression", "expressions",
    "production", "change", "changes", "increase", "increases",
    "decrease", "decreases", "type", "types", "form", "forms", "stage",
    "stages", "state", "states", "case", "cases", "patient", "patients",
    "subject", "subjects", "participant", "participants", "control",
    "controls", "target", "targets", "interaction", "interactions",
    "pathway", "pathways", "network", "networks", "signal", "signals",
    "marker", "markers", "indicator", "indicators", "test", "tests",
    "assessment", "assessments", "evaluation", "evaluations", "trial",
    "trials", "experiment", "experiments", "observation", "observations",
    "report", "reports", "review", "reviews", "analysis", "context",
    "information", "knowledge", "concept", "concepts", "feature", "features",
    "aspect", "aspects", "component", "components", "element", "elements",
    "structure", "structures", "property", "properties", "characteristic",
    "characteristics", "parameter", "parameters", "variable", "variables",
    "value", "values", "score", "scores", "rate", "rates", "ratio", "ratios",
    "index", "indices", "index", "mean", "median", "range", "prevalence",
    "incidence", "frequency", "proportion", "percentage", "number", "amount",
    "quantity", "duration", "period", "time", "age", "size", "dose",
    "concentration", "threshold", "limit", "maximum", "minimum",
    # Bare determiners/pronouns that sometimes slip through
    "this", "that", "these", "those", "which", "what", "who", "when",
    "where", "how", "why", "other", "another", "same", "different",
    "various", "several", "many", "few", "all", "both", "each", "every",
    "some", "any", "no", "not", "yes",
})


def _is_valid_entity_name(name: str) -> bool:
    """Return True when a name is specific enough to be a KG node.

    Rejects names that are:
    - shorter than _ENTITY_MIN_NAME_LENGTH (after stripping whitespace), unless
      they match the short-identifier pattern (digits/letters mixed, e.g. p53, IL-6)
    - in the generic hub-entity blocklist (case-insensitive, singular/plural)
    - bare numeric / punctuation fragments without any alphabetic referent
    """
    stripped = name.strip()
    if not stripped:
        return False
    normalized = stripped.lower()
    # Reject punctuation-only fragments or bare numeric tokens. Legitimate short
    # biomedical identifiers such as "p53" or "IL-6" include alphabetic
    # characters and therefore bypass this guard.
    if not re.search(r"[A-Za-z0-9]", stripped):
        return False
    if re.fullmatch(r"[\d\W_]+", stripped):
        return False
    # Blocklist check
    if normalized in _GENERIC_HUB_ENTITY_BLOCKLIST:
        return False
    # Length check: allow short named identifiers (alphanumeric + hyphens/dots)
    if len(stripped) < _ENTITY_MIN_NAME_LENGTH:
        # Allow if it looks like a named identifier: has a digit or a hyphen
        # (e.g. "p53" is 3 chars and passes length anyway; "IL" is 2 and fails
        # unless it has a hyphen such as "IL-6").
        import re as _re
        if not _re.search(r'[\d\-\.]', stripped):
            return False
    return True


def _env_flag(value: Optional[str], default: bool) -> bool:
    """Parse a boolean-like environment value safely."""
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


class OntologyGuidedKGCreator:
    """
    Ontology-Guided Knowledge Graph Creator that properly extracts entities from PDF content
    using LLM with ontology guidance for better entity classification and relationships
    """

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

    # ------------------------------------------------------------------
    # Ontology loading — supports OWL/RDF and JSON
    # ------------------------------------------------------------------

    @staticmethod
    def _prop_type(raw: str) -> PropertyType:
        """Map a raw type string from JSON to a PropertyType enum value."""
        _map = {
            "string": PropertyType.STRING, "str": PropertyType.STRING,
            "integer": PropertyType.INTEGER, "int": PropertyType.INTEGER,
            "decimal": PropertyType.DECIMAL, "numeric": PropertyType.DECIMAL,
            "double": PropertyType.DOUBLE,
            "float": PropertyType.FLOAT, "number": PropertyType.FLOAT,
            "boolean": PropertyType.BOOLEAN, "bool": PropertyType.BOOLEAN,
            "date": PropertyType.DATE, "datetime": PropertyType.DATETIME,
            "enum": PropertyType.ENUM,
            "id": PropertyType.ID, "identifier": PropertyType.ID,
        }
        return _map.get((raw or "string").strip().lower(), PropertyType.STRING)

    def _load_ontology(self, ontology_path: str):
        """Load ontology from OWL/RDF (XML) or Ontology Playground-style JSON.

        Both paths normalise into:
          self._ontology_schema     — OntologySchema (full typed model)
          self.ontology_classes     — List[dict]  (legacy flat list)
          self.ontology_relationships — List[dict]  (legacy flat list)
        """
        ext = os.path.splitext(ontology_path)[1].lower()
        is_json = ext == '.json'
        if not is_json and ext not in ('.owl', '.rdf', '.ttl', '.xml'):
            # Peek at first byte to detect JSON
            try:
                with open(ontology_path, 'r', encoding='utf-8') as _f:
                    _peek = _f.read(3).lstrip()
                is_json = _peek.startswith('{') or _peek.startswith('[')
            except OSError:
                pass

        try:
            if is_json:
                self._ontology_schema = self._load_ontology_json(ontology_path)
            else:
                self._ontology_schema = self._load_ontology_owl(ontology_path)
        except Exception as e:
            logging.error("Error loading ontology: %s", e)
            raise

        # Populate legacy flat lists for backwards compatibility
        self.ontology_classes = [
            {'id': et.id, 'uri': et.uri or '', 'label': et.label,
             'description': et.description or ''}
            for et in self._ontology_schema.entity_types
        ]
        self.ontology_relationships = [
            {'id': rt.id, 'uri': rt.uri or '', 'label': rt.label,
             'description': rt.description or '',
             'domain': rt.domain or '', 'range': rt.range or '',
             'cardinality': rt.cardinality or ''}
            for rt in self._ontology_schema.relationship_types
        ]
        logging.info(
            "Loaded ontology (%s): %d entity types, %d relationship types",
            self._ontology_schema.source_format,
            len(self.ontology_classes), len(self.ontology_relationships),
        )

    def _load_ontology_json(self, ontology_path: str) -> OntologySchema:
        """Parse an Ontology Playground-style JSON file.

        Accepts layout A: {"classes": [...], "relationships": [...]}
        and layout B:     {"entity_types": [...], "relationship_types": [...]}
        """
        with open(ontology_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        raw_classes = raw.get('classes') or raw.get('entity_types') or []
        raw_rels = raw.get('relationships') or raw.get('relationship_types') or []

        entity_types: List[OntEntityType] = []
        for cls in raw_classes:
            if not isinstance(cls, dict):
                continue
            eid = cls.get('id') or cls.get('name') or ''
            if not eid:
                continue
            props = []
            for p in cls.get('properties') or []:
                pname = (p.get('name') or p.get('id') or '') if isinstance(p, dict) else ''
                if not pname:
                    continue
                props.append(DataBinding(
                    name=pname,
                    type=self._prop_type(p.get('type', 'string')),
                    description=p.get('description') or None,
                    identifier=bool(p.get('identifier', False)),
                    required=bool(p.get('required', False)),
                    enum_values=list(p.get('enum_values') or p.get('values') or []),
                    unit=p.get('unit') or None,
                ))
            entity_types.append(OntEntityType(
                id=eid,
                label=cls.get('label') or eid.replace('_', ' ').title(),
                description=cls.get('description') or None,
                uri=cls.get('uri') or None,
                properties=props,
            ))

        relationship_types: List[OntRelType] = []
        for rel in raw_rels:
            if not isinstance(rel, dict):
                continue
            rid = rel.get('id') or rel.get('name') or rel.get('type') or ''
            if not rid:
                continue
            attrs = []
            for a in rel.get('attributes') or rel.get('properties') or []:
                aname = (a.get('name') or a.get('id') or '') if isinstance(a, dict) else ''
                if not aname:
                    continue
                attrs.append(RelationshipAttribute(
                    name=aname,
                    type=self._prop_type(a.get('type', 'string')),
                    description=a.get('description') or None,
                    unit=a.get('unit') or None,
                ))
            relationship_types.append(OntRelType(
                id=rid,
                label=rel.get('label') or rid.replace('_', ' ').title(),
                description=rel.get('description') or None,
                uri=rel.get('uri') or None,
                domain=rel.get('from') or rel.get('domain') or None,
                range=rel.get('to') or rel.get('range') or None,
                cardinality=rel.get('cardinality') or None,
                attributes=attrs,
            ))

        return OntologySchema(
            entity_types=entity_types, relationship_types=relationship_types,
            source_format='json', source_path=ontology_path,
        )

    def _load_ontology_owl(self, ontology_path: str) -> OntologySchema:
        """Parse an OWL/RDF XML ontology into OntologySchema."""
        tree = ET.parse(ontology_path)
        root = tree.getroot()

        ns = {
            'owl':  'http://www.w3.org/2002/07/owl#',
            'rdf':  'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        }
        _rdf_about  = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about'
        _rdfs_label = '{http://www.w3.org/2000/01/rdf-schema#}label'
        _rdfs_cmt   = '{http://www.w3.org/2000/01/rdf-schema#}comment'
        _rdfs_dom   = '{http://www.w3.org/2000/01/rdf-schema#}domain'
        _rdfs_rng   = '{http://www.w3.org/2000/01/rdf-schema#}range'
        _rdf_rsrc   = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'

        def _local(uri: str) -> str:
            return uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]

        def _res_local(elem):
            if elem is None:
                return None
            r = elem.get(_rdf_rsrc, '')
            return _local(r) if r else None

        def _child_text_by_local_name(parent, local_name: str) -> Optional[str]:
            if parent is None:
                return None
            for child in list(parent):
                tag = child.tag
                if isinstance(tag, str):
                    child_local = tag.split('}')[-1] if '}' in tag else tag.split(':')[-1]
                    if child_local == local_name and child.text:
                        return child.text.strip()
            return None

        def _bool_text(value: Optional[str]) -> bool:
            return str(value or "").strip().lower() in {"true", "1", "yes"}

        def _xsd_to_prop_type(range_uri: Optional[str], explicit_type: Optional[str]) -> PropertyType:
            if explicit_type:
                return self._prop_type(explicit_type)
            local = _local(range_uri) if range_uri else ""
            return {
                "string": PropertyType.STRING,
                "integer": PropertyType.INTEGER,
                "int": PropertyType.INTEGER,
                "long": PropertyType.INTEGER,
                "decimal": PropertyType.DECIMAL,
                "float": PropertyType.FLOAT,
                "double": PropertyType.DOUBLE,
                "date": PropertyType.DATE,
                "dateTime": PropertyType.DATETIME,
                "boolean": PropertyType.BOOLEAN,
            }.get(local, PropertyType.STRING)

        entity_types: List[OntEntityType] = []
        for cls_elem in root.findall('.//owl:Class', ns):
            uri = cls_elem.get(_rdf_about, '')
            if not uri:
                continue
            local = _local(uri)
            if not local:
                continue
            lbl_el = cls_elem.find(_rdfs_label)
            cmt_el = cls_elem.find(_rdfs_cmt)
            entity_types.append(OntEntityType(
                id=local, uri=uri,
                label=(lbl_el.text.strip() if lbl_el is not None and lbl_el.text else local.replace('_', ' ').title()),
                description=(cmt_el.text.strip() if cmt_el is not None and cmt_el.text else None),
            ))

        entity_by_id = {et.id: et for et in entity_types}

        relationship_attribute_map: Dict[str, List[RelationshipAttribute]] = defaultdict(list)

        for dt_elem in root.findall('.//owl:DatatypeProperty', ns):
            uri = dt_elem.get(_rdf_about, '')
            if not uri:
                continue

            local = _local(uri)
            label = _child_text_by_local_name(dt_elem, 'label') or local
            description = _child_text_by_local_name(dt_elem, 'comment')
            domain = _res_local(dt_elem.find('.//' + _rdfs_dom))
            range_uri = dt_elem.find('.//' + _rdfs_rng)
            range_local = _res_local(range_uri)
            explicit_type = _child_text_by_local_name(dt_elem, 'propertyType') or _child_text_by_local_name(dt_elem, 'attributeType')
            prop_type = _xsd_to_prop_type(range_local, explicit_type)
            enum_values_text = _child_text_by_local_name(dt_elem, 'enumValues')
            enum_values = [v.strip() for v in (enum_values_text or '').split(',') if v.strip()]
            unit = _child_text_by_local_name(dt_elem, 'unit')
            identifier = _bool_text(_child_text_by_local_name(dt_elem, 'isIdentifier'))
            relationship_attr_of = _child_text_by_local_name(dt_elem, 'relationshipAttributeOf')

            if relationship_attr_of:
                relationship_attribute_map[relationship_attr_of].append(
                    RelationshipAttribute(
                        name=label,
                        type=prop_type,
                        description=description,
                        unit=unit,
                    )
                )
                continue

            if not domain or domain not in entity_by_id:
                continue

            entity_by_id[domain].properties.append(
                DataBinding(
                    name=label,
                    type=prop_type,
                    description=description,
                    identifier=identifier or prop_type == PropertyType.ID,
                    required=False,
                    enum_values=enum_values,
                    unit=unit,
                )
            )

        relationship_types: List[OntRelType] = []
        for prop_elem in root.findall('.//owl:ObjectProperty', ns):
            uri = prop_elem.get(_rdf_about, '')
            if not uri:
                continue
            local = _local(uri)
            if not local:
                continue
            lbl_el = prop_elem.find(_rdfs_label)
            cmt_el = prop_elem.find(_rdfs_cmt)
            dom_el = prop_elem.find('.//' + _rdfs_dom)
            rng_el = prop_elem.find('.//' + _rdfs_rng)
            relationship_types.append(OntRelType(
                id=local, uri=uri,
                label=(lbl_el.text.strip() if lbl_el is not None and lbl_el.text else local.replace('_', ' ').title()),
                description=(cmt_el.text.strip() if cmt_el is not None and cmt_el.text else None),
                domain=_res_local(dom_el),
                range=_res_local(rng_el),
                cardinality=_child_text_by_local_name(prop_elem, 'cardinality'),
                attributes=relationship_attribute_map.get(local, []),
            ))

        return OntologySchema(
            entity_types=entity_types, relationship_types=relationship_types,
            source_format='owl', source_path=ontology_path,
        )

    def _validate_ontology_structure(self):
        """
        Validate and clean ontology class and relationship structures to prevent "string indices must be integers" errors
        """
        # Clean ontology classes
        valid_classes = []
        for cls in self.ontology_classes:
            if isinstance(cls, dict) and 'id' in cls and 'label' in cls:
                valid_classes.append(cls)
            else:
                logging.warning(f"Removing invalid ontology class entry: {cls}")

        self.ontology_classes = valid_classes
        logging.info(f"Validated ontology classes: {len(self.ontology_classes)} valid entries")

        # Clean ontology relationships
        valid_relationships = []
        for rel in self.ontology_relationships:
            if isinstance(rel, dict) and 'id' in rel and 'label' in rel:
                valid_relationships.append(rel)
            else:
                logging.warning(f"Removing invalid ontology relationship entry: {rel}")

        self.ontology_relationships = valid_relationships
        logging.info(f"Validated ontology relationships: {len(self.ontology_relationships)} valid entries")

    @staticmethod
    def _normalize_ontology_identifier(value: Optional[str]) -> str:
        """Normalize ontology ids/labels for exact and fuzzy matching."""
        if not isinstance(value, str):
            return ""
        normalized = value.strip().lower()
        normalized = re.sub(r"[\s\-]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def _schema_generic_entity_type(self) -> Optional[str]:
        """Return a generic ontology entity class when the schema defines one."""
        schema = self._ontology_schema
        if schema and schema.entity_types:
            preferred = {"concept", "entity", "thing", "unknown", "other"}
            for et in schema.entity_types:
                if (
                    self._normalize_ontology_identifier(et.id) in preferred
                    or self._normalize_ontology_identifier(et.label) in preferred
                ):
                    return et.id
        return None

    def _match_ontology_entity_type(
        self,
        raw_type: Optional[str],
        *,
        allow_fuzzy: bool = True,
        min_score: float = 0.80,
    ) -> Optional[str]:
        """Map a raw entity type string onto a known ontology class id."""
        normalized_raw = self._normalize_ontology_identifier(raw_type)
        if not normalized_raw:
            return None

        schema = self._ontology_schema
        if schema and schema.entity_types:
            best_match = None
            best_score = 0.0
            for et in schema.entity_types:
                candidates = [
                    self._normalize_ontology_identifier(et.id),
                    self._normalize_ontology_identifier(et.label),
                ]
                if normalized_raw in candidates:
                    return et.id
                if allow_fuzzy:
                    score = max(
                        difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                        for candidate in candidates
                        if candidate
                    )
                    if score > best_score:
                        best_score = score
                        best_match = et
            if allow_fuzzy and best_match and best_score >= min_score:
                return best_match.id

        best_match = None
        best_score = 0.0
        for cls in self.ontology_classes:
            cls_id = self._normalize_ontology_identifier(cls.get("id"))
            cls_label = self._normalize_ontology_identifier(cls.get("label"))
            if normalized_raw in {cls_id, cls_label}:
                return cls.get("id")
            if allow_fuzzy:
                score = max(
                    difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                    for candidate in (cls_id, cls_label)
                    if candidate
                )
                if score > best_score:
                    best_score = score
                    best_match = cls
        if allow_fuzzy and best_match and best_score >= min_score:
            return best_match.get("id")
        return None

    def _coerce_entity_type_with_ontology(
        self,
        raw_type: Optional[str],
        entity_text: Optional[str] = None,
    ) -> Optional[str]:
        """Coerce extracted entity types onto the active ontology when present."""
        has_ontology = bool(self._ontology_schema and self._ontology_schema.entity_types) or bool(self.ontology_classes)
        if not has_ontology:
            if raw_type:
                return str(raw_type)
            if entity_text:
                return self._classify_entity_with_ontology(entity_text)
            return None

        matched = self._match_ontology_entity_type(raw_type)
        if matched:
            return matched

        if entity_text:
            matched = self._match_ontology_entity_type(entity_text, allow_fuzzy=False)
            if matched:
                return matched
            classified = self._classify_entity_with_ontology(entity_text)
            matched = self._match_ontology_entity_type(classified, allow_fuzzy=False) or self._match_ontology_entity_type(classified)
            if matched:
                return matched

        generic_type = self._schema_generic_entity_type()
        if generic_type:
            if raw_type and self._normalize_ontology_identifier(raw_type) != self._normalize_ontology_identifier(generic_type):
                logging.info(
                    "Coercing off-schema entity type '%s' for '%s' to generic ontology type '%s'",
                    raw_type,
                    entity_text or "",
                    generic_type,
                )
            return generic_type

        logging.warning(
            "Dropping entity '%s' with off-schema type '%s' (no compatible ontology class found)",
            entity_text or "",
            raw_type or "",
        )
        return None

    def _schema_generic_relationship_type(self) -> Optional[str]:
        """Return a schema-defined generic relationship type when available."""
        preferred = {"related_to", "associated_with", "connects_to", "linked_to"}
        schema = self._ontology_schema
        if schema and schema.relationship_types:
            for rt in schema.relationship_types:
                if (
                    self._normalize_ontology_identifier(rt.id) in preferred
                    or self._normalize_ontology_identifier(rt.label) in preferred
                ):
                    return rt.id.replace(" ", "_").replace("-", "_").upper()
        for rel in self.ontology_relationships:
            rel_id = rel.get("id", "")
            rel_label = rel.get("label", "")
            if (
                self._normalize_ontology_identifier(rel_id) in preferred
                or self._normalize_ontology_identifier(rel_label) in preferred
            ):
                return rel_id.replace(" ", "_").replace("-", "_").upper()
        return None

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

    def _build_schema_card(self) -> dict:
        """Build a versioned snapshot of the ontology for this KG build.

        Stored on the Document node so future queries can detect ontology drift.
        Includes full property signatures, domain/range, cardinalities, and
        attribute schemas when a typed OntologySchema is available.
        """
        classes = [c.get('id', '') for c in self.ontology_classes if isinstance(c, dict)]
        rels    = [r.get('id', '') for r in self.ontology_relationships if isinstance(r, dict)]

        ontology_file_hash = None
        if self.ontology_path and os.path.exists(self.ontology_path):
            try:
                with open(self.ontology_path, 'rb') as f:
                    ontology_file_hash = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                pass

        card: dict = {
            "ontologyFileHash": ontology_file_hash or "",
            "ontologyPath":     os.path.basename(self.ontology_path) if self.ontology_path else "",
            "sourceFormat":     (self._ontology_schema.source_format if self._ontology_schema else "unknown"),
            "classes":          sorted(classes),
            "relationships":    sorted(rels),
            "classCount":       len(classes),
            "relationshipCount": len(rels),
            "builtAt":          datetime.now().isoformat(),
        }

        # Enrich with typed property signatures and domain/range when available
        schema = self._ontology_schema
        if schema:
            card["entityTypes"] = [
                {
                    "id": et.id,
                    "label": et.label,
                    "description": et.description,
                    "properties": [
                        {
                            "name": p.name, "type": p.type.value,
                            "identifier": p.identifier, "required": p.required,
                            "enum_values": p.enum_values, "unit": p.unit,
                        }
                        for p in et.properties
                    ],
                }
                for et in schema.entity_types
            ]
            card["relationshipTypes"] = [
                {
                    "id": rt.id, "label": rt.label, "description": rt.description,
                    "domain": rt.domain, "range": rt.range, "cardinality": rt.cardinality,
                    "attributes": [
                        {"name": a.name, "type": a.type.value, "unit": a.unit}
                        for a in rt.attributes
                    ],
                }
                for rt in schema.relationship_types
            ]

        fingerprint_payload = {
            "sourceFormat": card.get("sourceFormat", "unknown"),
            "classes": card.get("classes", []),
            "relationships": card.get("relationships", []),
            "entityTypes": card.get("entityTypes", []),
            "relationshipTypes": card.get("relationshipTypes", []),
        }
        fingerprint_str = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
        )
        schema_hash = hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
        card["schemaVersion"] = schema_hash[:16]
        card["schemaHash"] = schema_hash

        return card

    def _create_neo4j_connection(self):
        """Create Neo4j graph connection"""
        # Ensure password is not None - use environment variable as fallback
        password = self.neo4j_password
        if password is None or password == "":
            password = os.getenv("NEO4J_PASSWORD", "password")

        # Set environment variables to ensure LangChain Neo4jGraph can read them
        os.environ["NEO4J_URI"] = self.neo4j_uri
        os.environ["NEO4J_USERNAME"] = self.neo4j_user
        os.environ["NEO4J_PASSWORD"] = password
        os.environ["NEO4J_DATABASE"] = self.neo4j_database

        return Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=password,
            database=self.neo4j_database,
            refresh_schema=False,
            sanitize=True
        )

    @staticmethod
    def _is_neo4j_primitive(value: Any) -> bool:
        """Return True for Neo4j-safe primitive property values."""
        if isinstance(value, bool):
            return True
        if isinstance(value, (str, int)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        return False

    @classmethod
    def _coerce_neo4j_property_value(cls, value: Any) -> Any:
        """
        Convert arbitrary Python values into Neo4j-safe property values.

        Neo4j only accepts primitives or arrays of primitives as property values.
        Nested dicts/lists are serialized to JSON so richer metadata can still be
        preserved without breaking writes.
        """
        if value is None:
            return None
        if cls._is_neo4j_primitive(value):
            return value
        if isinstance(value, (list, tuple, set)):
            sequence = list(value)
            if all(cls._is_neo4j_primitive(item) for item in sequence):
                return list(sequence)
            return json.dumps(sequence, ensure_ascii=False, default=str)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)

    @classmethod
    def _sanitize_neo4j_properties(cls, properties: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Drop null/NaN values and coerce the rest to Neo4j-safe property values."""
        if not isinstance(properties, dict):
            return {}

        safe_properties: Dict[str, Any] = {}
        for key, value in properties.items():
            if value is None:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            coerced = cls._coerce_neo4j_property_value(value)
            if coerced is None:
                continue
            safe_properties[str(key)] = coerced
        return safe_properties

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

    # ------------------------------------------------------------------
    # Context enrichment helpers (section headers, qualifier sentences,
    # and cross-chunk coreference resolution)
    # ------------------------------------------------------------------

    # Patterns that mark standard biomedical paper sections
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

    # Keywords that mark qualifier-bearing sentences
    _QUALIFIER_KEYWORDS = re.compile(
        r"\b(?:condition|experiment|treat(?:ed|ment)|knockout|knock[- ]?out|"
        r"mutant|patient|cohort|model|culture|in\s+vitro|in\s+vivo|"
        r"hypox|normox|baseline|control|express(?:ed|ion)|stimulat|"
        r"inhibit|activat|induc|depleted|overexpress|transfect|"
        r"under\s+these|such\s+conditions?|this\s+(?:model|system|context|protocol)|"
        r"these\s+(?:cells?|conditions?|animals?|patients?|mice|rats?))\b",
        re.IGNORECASE,
    )

    # Demonstrative coreference markers that indicate cross-chunk references
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

    def _has_coreference_markers(self, text: str) -> bool:
        """Return True if *text* contains demonstrative coreference markers."""
        return bool(self._COREF_MARKERS.search(text))

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

    # ------------------------------------------------------------------
    # Schema-aware prompt helpers (step 4)
    # ------------------------------------------------------------------

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

    @staticmethod
    def _example_surface_form_for_type(type_label: str) -> str:
        label = str(type_label or "").lower()
        if "drug" in label or "medication" in label or "treatment" in label:
            return "Metformin"
        if "disease" in label or "condition" in label or "disorder" in label or "syndrome" in label:
            return "type 2 diabetes"
        if "gene" in label or "variant" in label or "mutation" in label:
            return "BRCA1"
        if "protein" in label or "enzyme" in label or "receptor" in label or "biomarker" in label:
            return "TP53 protein"
        if "person" in label or "patient" in label:
            return "Marie Curie"
        if "organization" in label or "hospital" in label or "institute" in label or "university" in label:
            return "World Health Organization"
        if "location" in label or "city" in label or "country" in label:
            return "Paris"
        if "film" in label or "work" in label or "book" in label or "album" in label:
            return "Inception"
        return type_label.replace("_", " ") or "Example entity"

    def _build_ontology_few_shot_examples(self) -> str:
        """Generate a few compact schema-aware examples for the extraction prompt."""
        if getattr(self, "few_shot_example_count", 0) <= 0:
            return ""

        examples: List[str] = []
        schema = self._ontology_schema
        if schema and schema.relationship_types and schema.entity_types:
            for rt in schema.relationship_types[: max(1, self.few_shot_example_count)]:
                source_type = next((et for et in schema.entity_types if et.id == rt.domain), None)
                target_type = next((et for et in schema.entity_types if et.id == rt.range), None)
                if not source_type or not target_type:
                    continue
                source_name = self._example_surface_form_for_type(source_type.label)
                target_name = self._example_surface_form_for_type(target_type.label)
                examples.append(
                    (
                        f"Example {len(examples) + 1}\n"
                        f"TEXT: \"{source_name} {rt.label.lower().replace('_', ' ')} {target_name}.\"\n"
                        "JSON:\n"
                        "{\n"
                        '  "relationships": [\n'
                        f'    {{"source": "{source_name}", "target": "{target_name}", "type": "{rt.id}", "negated": false, "properties": {{"description": "{source_name} {rt.label.lower().replace("_", " ")} {target_name}", "condition": null, "quantitative": null, "confidence": "demonstrated"}}}}\n'
                        "  ],\n"
                        '  "entities": [\n'
                        f'    {{"id": "{source_name}", "type": "{source_type.id}", "properties": {{"name": "{source_name}", "description": "{source_type.label}: {source_name}"}}}},\n'
                        f'    {{"id": "{target_name}", "type": "{target_type.id}", "properties": {{"name": "{target_name}", "description": "{target_type.label}: {target_name}"}}}}\n'
                        "  ]\n"
                        "}"
                    )
                )
                if len(examples) >= self.few_shot_example_count:
                    break
        elif self.ontology_classes:
            for cls in self.ontology_classes[: self.few_shot_example_count]:
                entity_name = self._example_surface_form_for_type(cls.get("label", cls.get("id", "Entity")))
                examples.append(
                    (
                        f"Example {len(examples) + 1}\n"
                        f'TEXT: "{entity_name} is mentioned in the document."\n'
                        "JSON:\n"
                        "{\n"
                        '  "relationships": [],\n'
                        '  "entities": [\n'
                        f'    {{"id": "{entity_name}", "type": "{cls.get("id", "Concept")}", "properties": {{"name": "{entity_name}", "description": "{cls.get("label", cls.get("id", "Concept"))}: {entity_name}"}}}}\n'
                        "  ]\n"
                        "}"
                    )
                )

        return "\n\n".join(examples)

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

    def _normalize_anchor_inventory(
        self,
        raw_inventory: Dict[str, Any],
        *,
        chunk_text: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize discovered anchors and attach exact local spans."""
        inventory = {
            "entity_anchors": [],
            "relation_anchors": [],
            "attribute_anchors": [],
        }
        if not isinstance(raw_inventory, dict):
            return inventory

        seen_entities = set()
        for anchor in raw_inventory.get("entity_anchors", []) or []:
            text = None
            anchor_type = None
            if isinstance(anchor, str):
                text = anchor
            elif isinstance(anchor, dict):
                text = anchor.get("text") or anchor.get("id")
                anchor_type = anchor.get("type")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if not _is_valid_entity_name(text):
                continue
            spans = self._find_exact_text_spans(chunk_text, [text])
            if not spans:
                continue
            exact_text = spans[0]["text"]
            key = self._normalize_entity_text(exact_text)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            coerced_type = self._coerce_entity_type_with_ontology(anchor_type, exact_text)
            if not coerced_type:
                continue
            inventory["entity_anchors"].append(
                {
                    "text": exact_text,
                    "type": coerced_type,
                    "anchor_spans": spans,
                }
            )

        seen_relations = set()
        for anchor in raw_inventory.get("relation_anchors", []) or []:
            if isinstance(anchor, str):
                text = anchor
                type_hint = anchor
            elif isinstance(anchor, dict):
                text = anchor.get("text")
                type_hint = anchor.get("type_hint") or anchor.get("type") or text
            else:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            spans = self._find_exact_text_spans(chunk_text, [text.strip()])
            if not spans:
                continue
            exact_text = spans[0]["text"]
            key = exact_text.lower()
            if key in seen_relations:
                continue
            seen_relations.add(key)
            inventory["relation_anchors"].append(
                {
                    "text": exact_text,
                    "type_hint": str(type_hint or exact_text).strip(),
                    "anchor_spans": spans,
                }
            )

        seen_attributes = set()
        for anchor in raw_inventory.get("attribute_anchors", []) or []:
            if isinstance(anchor, str):
                text = anchor
                role = "other"
            elif isinstance(anchor, dict):
                text = anchor.get("text")
                role = str(anchor.get("role") or "other").strip().lower()
            else:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            spans = self._find_exact_text_spans(chunk_text, [text.strip()])
            if not spans:
                continue
            exact_text = spans[0]["text"]
            key = (exact_text.lower(), role)
            if key in seen_attributes:
                continue
            seen_attributes.add(key)
            inventory["attribute_anchors"].append(
                {
                    "text": exact_text,
                    "role": role if role in {"condition", "quantitative", "other"} else "other",
                    "anchor_spans": spans,
                }
            )

        return inventory

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

    def _classify_entity_with_ontology(self, entity_text: str) -> str:
        """
        Classify entity using ontology guidance.

        Strategy (in priority order):
        1. Embedding-based cosine similarity against pre-computed ontology class label embeddings
           (threshold ≥ 0.50 required to accept the match).
        2. Exact substring match against ontology class id/label (legacy fallback).
        3. Keyword heuristics (final fallback).
        """
        entity_lower = entity_text.lower()
        schema = self._ontology_schema

        # Strategy 1: embedding similarity (preferred — robust to abbreviations / paraphrases)
        if self._ontology_class_embeddings and self.embedding_function:
            try:
                entity_emb = self.embedding_function.embed_query(entity_text)
                best_class, best_score = None, 0.0
                for cls_id, cls_emb in self._ontology_class_embeddings:
                    # Cosine similarity (both vectors are L2-normalised by the embedding model)
                    score = float(
                        sum(a * b for a, b in zip(entity_emb, cls_emb))
                        / (
                            (sum(a * a for a in entity_emb) ** 0.5 + 1e-9)
                            * (sum(b * b for b in cls_emb) ** 0.5 + 1e-9)
                        )
                    )
                    if score > best_score:
                        best_score, best_class = score, cls_id
                if best_class and best_score >= 0.50:
                    return best_class
            except Exception as e:
                logging.debug(f"Embedding classification failed for '{entity_text}': {e}")

        # Strategy 2: exact substring match against ontology class labels
        for cls in self.ontology_classes:
            if cls['id'].lower() in entity_lower or cls['label'].lower() in entity_lower:
                return cls['id']

        # Strategy 3: keyword heuristics
        fallback = 'Concept'
        if any(word in entity_lower for word in ['disease', 'cancer', 'tumor', 'syndrome', 'disorder', 'carcinoma', 'malignancy']):
            fallback = 'Disease'
        elif any(word in entity_lower for word in ['drug', 'medication', 'treatment', 'therapy', 'surgery', 'chemotherapy', 'radiotherapy']):
            fallback = 'Treatment'
        elif any(word in entity_lower for word in ['patient', 'person', 'individual', 'male', 'female']):
            fallback = 'Patient'
        elif any(word in entity_lower for word in ['doctor', 'physician', 'surgeon', 'specialist', 'oncologist', 'urologist']):
            fallback = 'Physician'
        elif any(word in entity_lower for word in ['hospital', 'clinic', 'center', 'institute', 'department']):
            fallback = 'Hospital'
        elif any(word in entity_lower for word in ['symptom', 'sign', 'manifestation', 'pain', 'fever']):
            fallback = 'Symptom'
        elif any(word in entity_lower for word in ['gene', 'mutation', 'protein', 'biomarker', 'receptor', 'marker']):
            fallback = 'Biomarker'
        elif any(word in entity_lower for word in ['score', 'grade', 'stage', 'classification', 'risk']):
            fallback = 'ClinicalFinding'

        if schema and schema.entity_types:
            matched = self._match_ontology_entity_type(fallback, allow_fuzzy=False)
            if matched:
                return matched
            generic = self._schema_generic_entity_type()
            if generic:
                return generic

        return fallback

    def _classify_relationship_with_ontology(self, source: str, target: str) -> str:
        """
        Classify relationship using ontology guidance
        """
        # ------------------------------------------------------------------
        # Schema-constrained relationship classification (step 5).
        # When OntologySchema is available, find relationship types whose
        # domain/range are compatible with the given entity types, then rank
        # by lexical similarity to the entity names as a tiebreaker.
        # Falls back to hardcoded heuristics only when no schema is loaded.
        # ------------------------------------------------------------------
        schema = self._ontology_schema
        if schema and schema.relationship_types:
            source_type = self._classify_entity_with_ontology(source)
            target_type = self._classify_entity_with_ontology(target)
            candidates = schema.compatible_relationships(source_type, target_type)
            if candidates:
                # Rank by lexical similarity of the relationship label to
                # the concatenated source+target text as a weak domain signal
                combined = (source + " " + target).lower()
                best = max(
                    candidates,
                    key=lambda rt: difflib.SequenceMatcher(
                        None, rt.label.lower(), combined
                    ).ratio(),
                )
                return best.id.replace(' ', '_').replace('-', '_').upper()

        # Heuristic fallback (no schema or no compatible candidates found)
        source_lower = source.lower()
        target_lower = target.lower()
        if any(w in source_lower for w in ['treatment', 'therapy', 'drug']) or \
           any(w in target_lower for w in ['treatment', 'therapy', 'drug']):
            return 'TREATS'
        if any(w in source_lower for w in ['disease', 'cancer']) and \
           any(w in target_lower for w in ['symptom', 'sign']):
            return 'HAS_SYMPTOM'
        if any(w in source_lower for w in ['physician', 'doctor']) and \
           any(w in target_lower for w in ['patient', 'person']):
            return 'DIAGNOSES'
        return 'RELATED_TO'

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

    def _canonicalize_relationship_type(
        self,
        raw_type: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> Optional[str]:
        """Map a raw LLM-generated relationship type to the closest ontology relationship.

        Steps:
        1. Exact label/id match (case/space-insensitive) filtered to schema-compatible
           candidates when source_type and target_type are provided.
        2. Fuzzy match (SequenceMatcher ≥ 0.72) with schema-compatibility boost
           (+0.10) for fully-compatible domain/range matches.
        3. If no schema is active, sanitize the raw type if it passes regex; else
           fall back to ASSOCIATED_WITH.
        4. If a schema is active and no safe match exists, return a schema-defined
           generic relationship type when available; otherwise return None so the
           caller can skip the off-schema edge.
        """
        schema = self._ontology_schema
        if not raw_type:
            return self._schema_generic_relationship_type() if schema else 'ASSOCIATED_WITH'

        normalized_raw = raw_type.lower().replace(' ', '_').replace('-', '_')

        candidates = (
            schema.compatible_relationships(source_type, target_type)
            if schema and (source_type or target_type)
            else (schema.relationship_types if schema else [])
        )
        # Merge with legacy flat list for non-schema path
        ont_rels = self.ontology_relationships

        if schema and candidates:
            # Step 1: exact match within schema-compatible candidates first
            for rt in candidates:
                cand = rt.label.lower().replace(' ', '_')
                if cand == normalized_raw or rt.id.lower() == normalized_raw:
                    logging.debug("Exact schema-compatible rel match: '%s' → '%s'", raw_type, rt.id)
                    return rt.id.replace(' ', '_').replace('-', '_').upper()
            # Step 1b: exact match in full schema (less preferred)
            for rt in schema.relationship_types:
                cand = rt.label.lower().replace(' ', '_')
                if cand == normalized_raw or rt.id.lower() == normalized_raw:
                    return rt.id.replace(' ', '_').replace('-', '_').upper()

            # Step 2: semantic label match using ontology relationship embeddings.
            semantic_match = self._semantic_relationship_match(
                raw_type,
                [rt.id for rt in candidates] or [rt.id for rt in schema.relationship_types],
            )
            if semantic_match:
                matched_id, matched_score = semantic_match
                logging.debug(
                    "Semantic schema rel: '%s' → '%s' (score=%.2f)", raw_type, matched_id, matched_score
                )
                return matched_id.replace(' ', '_').replace('-', '_').upper()

            # Step 3: fuzzy match with schema-compatibility boost
            compat_ids = {rt.id for rt in candidates}
            best_match, best_score = None, 0.0
            for rt in schema.relationship_types:
                cand_label = rt.label.lower().replace(' ', '_')
                score = max(
                    difflib.SequenceMatcher(None, normalized_raw, cand_label).ratio(),
                    difflib.SequenceMatcher(None, normalized_raw, rt.id.lower()).ratio(),
                )
                # Boost schema-compatible candidates
                if rt.id in compat_ids:
                    score += 0.10
                if score > best_score:
                    best_score, best_match = score, rt
            if best_match and best_score >= 0.72:
                logging.debug(
                    "Fuzzy schema rel: '%s' → '%s' (score=%.2f)", raw_type, best_match.id, best_score
                )
                return best_match.id.replace(' ', '_').replace('-', '_').upper()

        elif ont_rels:
            # Legacy flat-list path (no OntologySchema)
            for ont_rel in ont_rels:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                if candidate == normalized_raw or ont_rel['id'].lower() == normalized_raw:
                    return ont_rel['id'].replace(' ', '_').replace('-', '_').upper()
            semantic_match = self._semantic_relationship_match(
                raw_type,
                [rel.get("id", "") for rel in ont_rels if rel.get("id")],
            )
            if semantic_match:
                matched_id, matched_score = semantic_match
                logging.debug(
                    "Semantic rel (legacy): '%s' → '%s' (score=%.2f)", raw_type, matched_id, matched_score
                )
                return matched_id.replace(' ', '_').replace('-', '_').upper()
            best_match, best_score = None, 0.0
            for ont_rel in ont_rels:
                candidate = ont_rel['label'].lower().replace(' ', '_')
                score = difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()
                id_score = difflib.SequenceMatcher(None, normalized_raw, ont_rel['id'].lower()).ratio()
                best_s = max(score, id_score)
                if best_s > best_score:
                    best_score, best_match = best_s, ont_rel
            if best_match and best_score >= 0.72:
                logging.debug(
                    "Fuzzy rel (legacy): '%s' → '%s' (score=%.2f)", raw_type, best_match['id'], best_score
                )
                return best_match['id'].replace(' ', '_').replace('-', '_').upper()

        # Step 3: sanitize raw type
        sanitized = raw_type.strip().replace(' ', '_').replace('-', '_').upper()
        if schema:
            generic_rel = self._schema_generic_relationship_type()
            if generic_rel:
                logging.info(
                    "Coercing off-schema relationship type '%s' to generic ontology type '%s'",
                    raw_type,
                    generic_rel,
                )
                return generic_rel
            logging.warning(
                "Dropping relationship with off-schema type '%s' between %s and %s",
                raw_type,
                source_type or "?",
                target_type or "?",
            )
            return None
        if len(sanitized) > 50 or not re.match(r'^[A-Z][A-Z0-9_]*$', sanitized):
            logging.debug("Rel type '%s' failed sanitization; using ASSOCIATED_WITH", raw_type)
            return 'ASSOCIATED_WITH'
        return sanitized

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

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalize entity text for duplicate detection.

        Applies generic, domain-agnostic normalization:
        - lowercase + collapse whitespace
        - strip leading articles / conjunctions
        - remove punctuation that doesn't affect identity
        - collapse runs of underscores/spaces

        Domain-specific aliases are intentionally NOT hardcoded here.
        The LLM is expected to produce consistent names; UUID5-based
        deduplication in _harmonize_entities handles surface variants.
        """
        normalized = re.sub(r'\s+', ' ', text.lower().strip())

        # Strip leading articles and conjunctions
        normalized = re.sub(r'^(the |an |a |and |or )', '', normalized)

        # Remove punctuation that doesn't carry semantic weight
        normalized = re.sub(r'[,()\[\];:.]', '', normalized)

        # Collapse hyphens/slashes to underscores for stable keys
        normalized = re.sub(r'[\-/]', '_', normalized)

        # Final cleanup
        normalized = re.sub(r'\s+', '_', normalized.strip())
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        return normalized

    _TITLE_AWARE_BUNDLE_DATASETS = {"hotpotqa", "2wikimultihopqa", "musique"}

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

    @staticmethod
    def _entity_type_specificity(entity_type: Any) -> int:
        """Score whether an entity type is more specific than generic fallback labels."""
        generic_types = {"concept", "entity", "unknown", "other", ""}
        normalized = str(entity_type or "").strip().lower()
        return 0 if normalized in generic_types else 1

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

    def _coerce_harmonized_entities_to_schema(
        self,
        harmonized_entities: List[Dict[str, Any]],
        harmonized_relationships: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply ontology type enforcement after harmonization as a final backstop."""
        has_ontology = bool(self._ontology_schema and self._ontology_schema.entity_types) or bool(self.ontology_classes)
        if not has_ontology:
            return harmonized_entities, harmonized_relationships

        filtered_entities: List[Dict[str, Any]] = []
        valid_entity_ids = set()
        dropped_entities = 0
        for entity in harmonized_entities:
            coerced_type = self._coerce_entity_type_with_ontology(
                entity.get("type"),
                entity.get("id"),
            )
            if not coerced_type:
                dropped_entities += 1
                continue
            entity_copy = dict(entity)
            entity_copy["type"] = coerced_type
            properties = dict(entity_copy.get("properties") or {})
            properties["type"] = coerced_type
            entity_copy["properties"] = properties
            filtered_entities.append(entity_copy)
            valid_entity_ids.add(entity_copy.get("uuid"))

        filtered_relationships = [
            rel for rel in harmonized_relationships
            if rel.get("source") in valid_entity_ids and rel.get("target") in valid_entity_ids
        ]
        dropped_relationships = len(harmonized_relationships) - len(filtered_relationships)
        if dropped_entities or dropped_relationships:
            logging.info(
                "Ontology enforcement dropped %d harmonized entities and %d relationships",
                dropped_entities,
                dropped_relationships,
            )
        self._last_schema_enforcement_stats = {
            "dropped_entities": dropped_entities,
            "dropped_relationships": dropped_relationships,
            "kept_entities": len(filtered_entities),
            "kept_relationships": len(filtered_relationships),
        }
        return filtered_entities, filtered_relationships

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

    def _store_graph_enrichment(
        self,
        graph,
        *,
        kg_name: str,
        file_name: str,
        enrichment: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Persist optional summaries, claims, and bridge edges to Neo4j."""
        if not enrichment:
            return {"component_summaries": 0, "claims": 0, "fragmentation_bridges": 0}

        graph_summary = enrichment.get("graph_summary") or {}
        component_summaries = list(enrichment.get("component_summaries") or [])
        claims = list(enrichment.get("claims") or [])
        fragmentation_bridges = list(enrichment.get("fragmentation_bridges") or [])

        chunk_id_by_position = {
            int(chunk.get("position", 0)): hashlib.sha1(
                f"{kg_name}:{file_name}:{chunk['position']}:{chunk['text']}".encode()
            ).hexdigest()
            for chunk in chunks or []
        }

        if graph_summary:
            graph.query(
                """
                MERGE (s:Summary {id: $summary_id})
                SET s.kgName = $kg_name,
                    s.scope = 'graph',
                    s.text = $text,
                    s.componentCount = $component_count,
                    s.createdAt = datetime()
                WITH s
                MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                MERGE (d)-[:HAS_GRAPH_SUMMARY]->(s)
                """,
                {
                    "summary_id": graph_summary["id"],
                    "kg_name": kg_name,
                    "file_name": file_name,
                    "text": graph_summary.get("text", ""),
                    "component_count": int(graph_summary.get("component_count", 0)),
                },
            )

        for summary in component_summaries:
            graph.query(
                """
                MERGE (s:Summary {id: $summary_id})
                SET s.kgName = $kg_name,
                    s.scope = 'component',
                    s.text = $text,
                    s.componentIndex = $component_index,
                    s.entityCount = $entity_count,
                    s.relationshipCount = $relationship_count,
                    s.createdAt = datetime()
                WITH s
                MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                MERGE (d)-[:HAS_COMPONENT_SUMMARY]->(s)
                """,
                {
                    "summary_id": summary["id"],
                    "kg_name": kg_name,
                    "file_name": file_name,
                    "text": summary.get("text", ""),
                    "component_index": int(summary.get("component_index", 0)),
                    "entity_count": int(summary.get("entity_count", 0)),
                    "relationship_count": int(summary.get("relationship_count", 0)),
                },
            )
            if graph_summary:
                graph.query(
                    """
                    MATCH (g:Summary {id: $graph_summary_id})
                    MATCH (s:Summary {id: $summary_id})
                    MERGE (g)-[:ABSTRACTS_COMPONENT]->(s)
                    """,
                    {
                        "graph_summary_id": graph_summary["id"],
                        "summary_id": summary["id"],
                    },
                )
            for entity_id in summary.get("entity_ids") or []:
                graph.query(
                    """
                    MATCH (s:Summary {id: $summary_id})
                    MATCH (e:__Entity__ {id: $entity_id})
                    MERGE (s)-[:SUMMARIZES]->(e)
                    """,
                    {
                        "summary_id": summary["id"],
                        "entity_id": entity_id,
                    },
                )

        for claim in claims:
            graph.query(
                """
                MERGE (c:Claim {id: $claim_id})
                SET c.kgName = $kg_name,
                    c.text = $text,
                    c.polarity = $polarity,
                    c.confidence = $confidence,
                    c.componentIndex = $component_index,
                    c.createdAt = datetime()
                WITH c
                MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                MERGE (d)-[:HAS_CLAIM]->(c)
                """,
                {
                    "claim_id": claim["id"],
                    "kg_name": kg_name,
                    "file_name": file_name,
                    "text": claim.get("text", ""),
                    "polarity": claim.get("polarity", "affirmed"),
                    "confidence": float(claim.get("confidence", 0.0) or 0.0),
                    "component_index": int(claim.get("component_index", 0)),
                },
            )
            for entity_id in claim.get("entity_ids") or []:
                graph.query(
                    """
                    MATCH (c:Claim {id: $claim_id})
                    MATCH (e:__Entity__ {id: $entity_id})
                    MERGE (c)-[:ABOUT]->(e)
                    """,
                    {
                        "claim_id": claim["id"],
                        "entity_id": entity_id,
                    },
                )
            for position in claim.get("provenance_positions") or []:
                chunk_id = chunk_id_by_position.get(int(position))
                if not chunk_id:
                    continue
                graph.query(
                    """
                    MATCH (c:Claim {id: $claim_id})
                    MATCH (ch:Chunk {id: $chunk_id})
                    MERGE (c)-[:SUPPORTED_BY]->(ch)
                    """,
                    {
                        "claim_id": claim["id"],
                        "chunk_id": chunk_id,
                    },
                )

        for bridge in fragmentation_bridges:
            bridge_key = hashlib.sha1(
                f"{kg_name}:{bridge['source_id']}:{bridge['target_id']}:{bridge['reason']}".encode("utf-8")
            ).hexdigest()
            graph.query(
                """
                MATCH (source:__Entity__ {id: $source_id})
                MATCH (target:__Entity__ {id: $target_id})
                MERGE (source)-[r:SOFT_BRIDGE {id: $bridge_id}]->(target)
                SET r.kgName = $kg_name,
                    r.synthetic = true,
                    r.reason = $reason,
                    r.confidence = $confidence,
                    r.sharedAliasKeys = $shared_alias_keys,
                    r.createdAt = datetime()
                """,
                {
                    "bridge_id": bridge_key,
                    "kg_name": kg_name,
                    "source_id": bridge["source_id"],
                    "target_id": bridge["target_id"],
                    "reason": bridge.get("reason", "lexical_guard"),
                    "confidence": float(bridge.get("confidence", 0.0) or 0.0),
                    "shared_alias_keys": list(bridge.get("shared_alias_keys") or []),
                },
            )

        return {
            "component_summaries": len(component_summaries),
            "claims": len(claims),
            "fragmentation_bridges": len(fragmentation_bridges),
        }

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

    def _build_relationship_merge_query(
        self,
        sanitized_rel_type: str,
        *,
        include_condition: bool,
        include_quantitative: bool,
    ) -> str:
        """Build the relationship MERGE query without null-valued qualifiers.

        Neo4j rejects null property values inside MERGE patterns, so optional
        qualifiers must only participate in the identity key when present.
        """
        merge_key_parts = ["negated: $negated"]
        if include_condition:
            merge_key_parts.append("condition: $condition")
        if include_quantitative:
            merge_key_parts.append("quantitative: $quantitative")

        merge_key = ",\n                        ".join(merge_key_parts)
        return f"""
                    MATCH (source:__Entity__ {{id: $source_id}})
                    MATCH (target:__Entity__ {{id: $target_id}})
                    MERGE (source)-[r:{sanitized_rel_type} {{
                        {merge_key}
                    }}]->(target)
                    SET r += $properties,
                        r.provenancePositions =
                            reduce(acc = [], x IN coalesce(r.provenancePositions, []) + $provenance_positions |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.questionIds =
                            reduce(acc = [], x IN coalesce(r.questionIds, []) + $question_ids |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.passageKeys =
                            reduce(acc = [], x IN coalesce(r.passageKeys, []) + $passage_keys |
                                CASE WHEN x IN acc THEN acc ELSE acc + x END)
                    """

    def store_knowledge_graph_with_embeddings(
        self,
        kg: Dict[str, Any],
        file_name: str,
        doc_metadata: dict = None,
        doc_hash: str = None,
        llm=None,
        model_name: str = "openai/gpt-oss-120b:free",
    ) -> bool:
        """
        Store the knowledge graph in Neo4j database with proper embedding support
        """
        try:
            # Try to create Neo4j connection - handle APOC issues gracefully
            try:
                graph = self._create_neo4j_connection()
            except Exception as conn_error:
                if "APOC" in str(conn_error) or "apoc" in str(conn_error):
                    logging.warning(f"APOC not available, skipping advanced KG storage: {conn_error}")
                    return False
                else:
                    raise conn_error

            # Pre-flight: remove any orphaned entities from a previous failed build
            # for this dataset (entities written before the Document node was committed).
            # This prevents unique-constraint violations on __Entity__.id when the LLM
            # assigns a different type label on a subsequent run.
            import uuid
            kg_name_value = kg['metadata'].get('kg_name') or file_name or "default"
            try:
                graph.query(
                    """
                    MATCH (e:__Entity__)
                    WHERE e.id STARTS WITH $prefix
                      AND NOT EXISTS {
                        MATCH (:Chunk)-[:HAS_ENTITY|MENTIONS]->(e)
                      }
                    DETACH DELETE e
                    """,
                    {"prefix": kg_name_value + "_"},
                )
                logging.info(f"[store_kg] Pre-flight cleanup done for '{kg_name_value}'")
            except Exception as _cleanup_err:
                logging.warning(f"[store_kg] Pre-flight cleanup failed (non-fatal): {_cleanup_err}")

            try:
                graph.query(
                    """
                    MATCH (q:Qualifier {kgName: $kg_name})
                    WHERE NOT EXISTS { MATCH ()-[]->(q) }
                       OR NOT EXISTS { MATCH (q)-[]->() }
                    DETACH DELETE q
                    """,
                    {"kg_name": kg_name_value},
                )
            except Exception as _qual_cleanup_err:
                logging.debug("[store_kg] Orphan qualifier cleanup failed (non-fatal): %s", _qual_cleanup_err)

            # Create document node with versioning
            kg_version = str(uuid.uuid4())
            schema_card = self._build_schema_card()
            schema_card_json = json.dumps(schema_card, ensure_ascii=False)
            doc_query = """
            MERGE (d:Document {fileName: $fileName, kgName: $kgName})
            SET d.kgVersion = $kgVersion,
                d.kgName = $kgName,
                d.createdAt = datetime(),
                d.updatedAt = datetime(),
                d.totalChunks = $totalChunks,
                d.totalEntities = $totalEntities,
                d.totalRelationships = $totalRelationships,
                d.ontologyClasses = $ontologyClasses,
                d.ontologyRelationships = $ontologyRelationships,
                d.schemaEnforcementDroppedEntities = $schemaEnforcementDroppedEntities,
                d.schemaEnforcementDroppedRelationships = $schemaEnforcementDroppedRelationships,
                d.harmonizationRelationshipsDroppedUnmapped = $harmonizationRelationshipsDroppedUnmapped,
                d.harmonizationRelationshipsDroppedSchemaMismatch = $harmonizationRelationshipsDroppedSchemaMismatch,
                d.harmonizationRelationshipsDeduped = $harmonizationRelationshipsDeduped,
                d.harmonizationRelationshipContradictionGroups = $harmonizationRelationshipContradictionGroups,
                d.harmonizationRelationshipContradictionEdges = $harmonizationRelationshipContradictionEdges,
                d.contentHash = $contentHash,
                d.schemaCard = $schemaCard,
                d.schemaVersion = $schemaVersion,
                d.schemaHash = $schemaHash,
                d.embeddingModel = $embeddingModel,
                d.provider = $provider,
                d.model = $model,
                d.maxChunks = $maxChunks
            """
            graph.query(doc_query, {
                "kgVersion": kg_version,
                "kgName": kg_name_value,
                "fileName": file_name,
                "totalChunks": kg['metadata']['total_chunks'],
                "totalEntities": kg['metadata']['total_entities'],
                "totalRelationships": kg['metadata']['total_relationships'],
                "ontologyClasses": kg['metadata']['ontology_classes'],
                "ontologyRelationships": kg['metadata']['ontology_relationships'],
                "schemaEnforcementDroppedEntities": kg['metadata'].get('schema_enforcement_dropped_entities', 0),
                "schemaEnforcementDroppedRelationships": kg['metadata'].get('schema_enforcement_dropped_relationships', 0),
                "harmonizationRelationshipsDroppedUnmapped": kg['metadata'].get('harmonization_relationships_dropped_unmapped', 0),
                "harmonizationRelationshipsDroppedSchemaMismatch": kg['metadata'].get('harmonization_relationships_dropped_schema_mismatch', 0),
                "harmonizationRelationshipsDeduped": kg['metadata'].get('harmonization_relationships_deduped', 0),
                "harmonizationRelationshipContradictionGroups": kg['metadata'].get('harmonization_relationship_contradiction_groups', 0),
                "harmonizationRelationshipContradictionEdges": kg['metadata'].get('harmonization_relationship_contradiction_edges', 0),
                "contentHash": doc_hash or "",
                "schemaCard": schema_card_json,
                "schemaVersion": schema_card["schemaVersion"],
                "schemaHash": schema_card["schemaHash"],
                "embeddingModel": self.embedding_model,
                "provider": kg['metadata'].get('provider', 'openai'),
                "model": kg['metadata'].get('model', ''),
                "maxChunks": kg['metadata'].get('max_chunks_setting'),
            })

            # Store document-level metadata from source (e.g. CSV columns like SUBJECT_ID, HADM_ID)
            if doc_metadata:
                safe_meta = self._sanitize_neo4j_properties(doc_metadata)
                if safe_meta:
                    graph.query(
                        "MATCH (d:Document {fileName: $fileName, kgName: $kgName}) SET d += $meta",
                        {"fileName": file_name, "kgName": kg_name_value, "meta": safe_meta},
                    )

            # Create chunk nodes with embeddings.
            # Include kg_name in the hash so identical text in different KGs
            # gets distinct Chunk nodes and retrieval filters are not contaminated.
            for chunk in kg['chunks']:
                chunk_id = hashlib.sha1(f"{kg_name_value}:{file_name}:{chunk['position']}:{chunk['text']}".encode()).hexdigest()
                chunk_query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.kgName = $kg_name,
                    c.position = $position,
                    c.chunkLocalIndex = $chunk_local_index,
                    c.start_pos = $start_pos,
                    c.end_pos = $end_pos,
                    c.source = $source,
                    c.dataset = $dataset,
                    c.questionId = $question_id,
                    c.passageIndex = $passage_index,
                    c.embedding = $embedding
                """
                graph.query(chunk_query, {
                    "chunk_id": chunk_id,
                    "kg_name": kg_name_value,
                    "text": chunk['text'],
                    "position": chunk['position'],
                    "chunk_local_index": chunk.get('chunk_local_index', chunk.get('chunk_id', 0)),
                    "start_pos": chunk['start_pos'],
                    "end_pos": chunk['end_pos'],
                    "source": chunk.get('source'),
                    "dataset": chunk.get('dataset'),
                    "question_id": chunk.get('question_id'),
                    "passage_index": chunk.get('passage_index'),
                    "embedding": chunk.get('embedding')
                })

                # Link chunk to document
                chunk_doc_query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (d:Document {fileName: $fileName, kgName: $kgName})
                MERGE (c)-[:PART_OF]->(d)
                """
                graph.query(chunk_doc_query, {
                    "chunk_id": chunk_id,
                    "fileName": file_name,
                    "kgName": kg_name_value,
                })

                # Create retrieval-only subchunks sized for the embedding model.
                # These keep dense retrieval faithful without shrinking the
                # larger parent chunks used for KG extraction.
                retrieval_subchunks = self._build_retrieval_subchunks(
                    chunk,
                    parent_chunk_id=chunk_id,
                )
                for subchunk in retrieval_subchunks:
                    retrieval_chunk_query = """
                    MERGE (rc:RetrievalChunk {id: $retrieval_chunk_id})
                    SET rc.text = $text,
                        rc.kgName = $kg_name,
                        rc.position = $position,
                        rc.retrievalLocalIndex = $retrieval_local_index,
                        rc.parentChunkId = $parent_chunk_id,
                        rc.chunkLocalIndex = $chunk_local_index,
                        rc.start_pos = $start_pos,
                        rc.end_pos = $end_pos,
                        rc.source = $source,
                        rc.dataset = $dataset,
                        rc.questionId = $question_id,
                        rc.passageIndex = $passage_index,
                        rc.embedding = $embedding
                    """
                    graph.query(retrieval_chunk_query, {
                        "retrieval_chunk_id": subchunk["id"],
                        "kg_name": kg_name_value,
                        "text": subchunk["text"],
                        "position": subchunk["position"],
                        "retrieval_local_index": subchunk["retrieval_local_index"],
                        "parent_chunk_id": subchunk["parent_chunk_id"],
                        "chunk_local_index": subchunk["chunk_local_index"],
                        "start_pos": subchunk["start_pos"],
                        "end_pos": subchunk["end_pos"],
                        "source": subchunk.get("source"),
                        "dataset": subchunk.get("dataset"),
                        "question_id": subchunk.get("question_id"),
                        "passage_index": subchunk.get("passage_index"),
                        "embedding": subchunk.get("embedding"),
                    })
                    graph.query(
                        """
                        MATCH (rc:RetrievalChunk {id: $retrieval_chunk_id})
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (rc)-[:RETRIEVES_FROM]->(c)
                        """,
                        {
                            "retrieval_chunk_id": subchunk["id"],
                            "chunk_id": chunk_id,
                        },
                    )

            # Create entity nodes with embeddings and ontology-based labels.
            # MERGE on node id (kg-prefixed UUID) so each KG's entities are independent.
            # content_hash is stored as a property for reference but is NOT the merge key,
            # which previously caused relationship storage to silently fail: a second KG
            # build would find the existing node by content_hash but leave n.id pointing at
            # the first KG's prefix, so MATCH (source {id: "kg2_uuid"}) never matched.
            for node in kg['nodes']:
                # Generate content-based deduplication hash (stored as property, not merge key)
                original_id = node.get('properties', {}).get('original_id', node['id'])
                normalized_content = self._normalize_entity_text(original_id)
                content_hash = hashlib.md5(f"{node['label']}:{normalized_content}".encode()).hexdigest()
                node['content_hash'] = content_hash

            for node in kg['nodes']:
                properties = node.get('properties', {})
                entity_type = node['label']  # This is the ontology class (Disease, Treatment, etc.)
                # Sanitize entity type for Cypher label compatibility and validate against whitelist
                cypher_safe_entity_type = re.sub(r'[^A-Za-z0-9_]', '_', entity_type).strip('_') or 'Concept'
                # Validate: label must start with a letter, max 64 chars, no injection risk
                if not re.match(r'^[A-Za-z][A-Za-z0-9_]{0,63}$', cypher_safe_entity_type):
                    logging.warning(f"Unsafe entity type '{entity_type}' → falling back to 'Concept'")
                    cypher_safe_entity_type = 'Concept'
                # Blocklist structural/system labels that must not be applied to entity nodes
                _RESERVED_LABELS = {'Document', 'Chunk', 'Mention', 'Entity', '__Entity__',
                                    '__KGDocument__', 'Relationship', 'Node', 'Schema'}
                if cypher_safe_entity_type in _RESERVED_LABELS:
                    logging.warning(f"Entity type '{entity_type}' clashes with structural label — falling back to 'Concept'")
                    cypher_safe_entity_type = 'Concept'

                # Generate embedding for entity if it doesn't have one.
                # Embed the entity name only (not description): the entity_vector
                # index is used for entity→entity ANN matching at query time, where
                # the probe vector is also a short entity-mention string.  Including
                # description here shifts the embedding into description-semantic space,
                # making name-level similarity unreliable (HippoRAG embeds names only).
                entity_embedding = node.get('embedding')
                if entity_embedding is None:
                    entity_text = properties.get('name', node['id'])
                    try:
                        entity_embedding = self.embedding_function.embed_query(entity_text)
                    except Exception as e:
                        logging.warning(f"Failed to generate embedding for entity {node['id']}: {e}")
                        entity_embedding = None

                # MERGE on __Entity__ only (not the specific type label) so that the
                # unique constraint on __Entity__.id is respected even when the LLM assigns
                # a different type label on a re-run.  The specific label is added with SET
                # after the MERGE, which is idempotent on existing nodes.
                node_query = f"""
                MERGE (n:__Entity__ {{id: $id}})
                ON CREATE SET
                    n.name = $name,
                    n.type = $type,
                    n.description = $description,
                    n.embedding = $embedding,
                    n.ontology_class = $entity_type,
                    n.content_hash = $content_hash,
                    n.kgName = $kg_name,
                    n.all_names = $all_names,
                    n.original_ids = $original_ids,
                    n += $extra_properties,
                    n.created_at = datetime()
                ON MATCH SET
                    n.last_accessed = datetime(),
                    n.kgName = $kg_name,
                    n.type = $type,
                    n.ontology_class = $entity_type,
                    n.all_names = coalesce(n.all_names, []) + $all_names,
                    n.original_ids = coalesce(n.original_ids, []) + $original_ids,
                    n += $extra_properties
                SET n:{cypher_safe_entity_type}
                """
                extra_properties = self._sanitize_neo4j_properties(
                    {
                        k: v
                        for k, v in properties.items()
                        if k not in {"name", "description", "type", "all_names", "original_ids", "aliases"}
                    }
                )
                graph.query(node_query, {
                    "id": node['id'],
                    "content_hash": node['content_hash'],
                    "kg_name": kg_name_value,
                    "name": self._coerce_neo4j_property_value(properties.get('name', node['id'])),
                    "type": node['label'],
                    "description": self._coerce_neo4j_property_value(properties.get('description', '')),
                    "embedding": entity_embedding,
                    "entity_type": entity_type,
                    "all_names": self._coerce_neo4j_property_value(
                        list(set(
                            properties.get('all_names', [node['id']])
                            + [a for a in (properties.get('aliases') or []) if isinstance(a, str) and a.strip()]
                        ))
                    ),
                    "original_ids": self._coerce_neo4j_property_value(
                        list(set(properties.get('original_ids', [node['id']])))
                    ),
                    "extra_properties": extra_properties,
                })

            # Build prefixed-UUID → human-readable name lookup for confidence verification.
            # kg['nodes'][i]['id'] is the prefixed UUID (e.g. "kg_abc_<uuid>")
            # kg['nodes'][i]['properties']['name'] is the actual entity name text.
            # rel['from'] / rel['to'] use the same prefixed-UUID format.
            _uuid_to_name = {
                _n['id']: (_n.get('properties', {}).get('name') or _n['id'])
                for _n in kg.get('nodes', [])
                if _n.get('id')
            }

            # Create relationships with improved error handling
            relationships_stored = 0
            relationship_store_failures = 0
            relationships_skipped_low_confidence = 0
            relationships_skipped_schema_mismatch = 0
            relationships_reverified_kept = 0
            relationships_reverified_rejected = 0
            for idx, rel in enumerate(kg['relationships']):
                try:
                    # Filter out fields that are managed explicitly below to avoid duplicates:
                    # 'id' causes duplicate key issues; 'negated'/'condition'/'quantitative'
                    # are promoted to top-level properties_with_confidence below.
                    _exclude = {'id', 'negated', 'condition', 'quantitative'}
                    properties_filtered = {
                        k: v for k, v in rel.get('properties', {}).items()
                        if k not in _exclude
                    }
                    original_properties_filtered = dict(properties_filtered)

                    # Resolve UUIDs to entity names for evidence-grounded confidence check.
                    # Prefer explicit source_name/target_name properties; fall back to name lookup.
                    _src_id = rel.get('from') or rel.get('source', '')
                    _tgt_id = rel.get('to') or rel.get('target', '')
                    _src_node = next((n for n in kg.get('nodes', []) if n.get('id') == _src_id), {})
                    _tgt_node = next((n for n in kg.get('nodes', []) if n.get('id') == _tgt_id), {})
                    source_type = (
                        _src_node.get('label')
                        or (_src_node.get('properties') or {}).get('type')
                        or (_src_node.get('properties') or {}).get('ontology_class')
                    )
                    target_type = (
                        _tgt_node.get('label')
                        or (_tgt_node.get('properties') or {}).get('type')
                        or (_tgt_node.get('properties') or {}).get('ontology_class')
                    )

                    # Canonicalize relationship type against ontology using fuzzy matching
                    sanitized_rel_type = self._canonicalize_relationship_type(
                        rel['type'],
                        source_type=source_type,
                        target_type=target_type,
                    )
                    if not sanitized_rel_type:
                        relationships_skipped_schema_mismatch += 1
                        logging.info(
                            "Skipping relationship with no schema-compatible type: %s -[%s]-> %s",
                            rel.get('from'),
                            rel.get('type'),
                            rel.get('to'),
                        )
                        continue

                    # Use all known surface forms for the entity so canonical-name
                    # mismatches (e.g. "TBK1 kinase" → "TBK1") don't falsely score 0.
                    _src_all_names = _src_node.get('properties', {}).get('all_names') or []
                    _tgt_all_names = _tgt_node.get('properties', {}).get('all_names') or []
                    source_name = (rel.get('properties', {}).get('source_name')
                                   or _uuid_to_name.get(_src_id)
                                   or _src_id)
                    target_name = (rel.get('properties', {}).get('target_name')
                                   or _uuid_to_name.get(_tgt_id)
                                   or _tgt_id)
                    verification_chunks = kg.get('chunks', [])
                    provenance_positions = rel.get('provenance_positions') or []
                    if provenance_positions:
                        scoped_chunks = [
                            chunk for chunk in kg.get('chunks', [])
                            if chunk.get('position') in provenance_positions
                        ]
                        if scoped_chunks:
                            verification_chunks = scoped_chunks

                    restoration = self._verify_relationship_restoration(
                        rel,
                        verification_chunks,
                        source_name=source_name,
                        target_name=target_name,
                        relation_type=sanitized_rel_type,
                        source_aliases=_src_all_names,
                        target_aliases=_tgt_all_names,
                    )
                    properties_filtered["anchor_grounding"] = restoration["anchor_grounding"]
                    properties_filtered["restoration_status"] = restoration["status"]
                    properties_filtered["restoration_verified"] = restoration["verified"]
                    properties_filtered["restoration_grounded_components"] = restoration["grounded_components"]
                    properties_filtered["restoration_grounded_count"] = restoration["grounded_count"]

                    triple_confidence = self._verify_triple_confidence(
                        source_name,
                        target_name,
                        sanitized_rel_type,
                        verification_chunks,
                        source_aliases=_src_all_names,
                        target_aliases=_tgt_all_names,
                    )
                    has_upstream_anchor_grounding = any(
                        original_properties_filtered.get(key)
                        for key in (
                            "anchor_grounding",
                            "source_anchor_spans",
                            "target_anchor_spans",
                            "relation_anchor_spans",
                        )
                    )
                    if restoration["status"] == "full" and has_upstream_anchor_grounding:
                        triple_confidence = max(triple_confidence, 0.95)
                    elif restoration["status"] == "failed":
                        triple_confidence = min(triple_confidence, 0.1)
                    evidence_scope = self._relationship_evidence_scope(
                        verification_chunks,
                        triple_confidence,
                    )

                    llm_reverified = False
                    if (
                        self.enable_low_confidence_triple_reverification
                        and llm is not None
                        and triple_confidence <= self.low_confidence_reverify_threshold
                    ):
                        reverification = self._reverify_low_confidence_triple(
                            source_name=source_name,
                            target_name=target_name,
                            relationship_type=sanitized_rel_type,
                            verification_chunks=verification_chunks,
                            llm=llm,
                            model_name=model_name,
                        )
                        if reverification is False:
                            relationships_reverified_rejected += 1
                            relationships_skipped_low_confidence += 1
                            logging.info(
                                "Skipping low-confidence relationship after LLM reverification: %s -[%s]-> %s",
                                rel.get('from'),
                                sanitized_rel_type,
                                rel.get('to'),
                            )
                            continue
                        if reverification is True:
                            llm_reverified = True
                            relationships_reverified_kept += 1
                            triple_confidence = max(
                                triple_confidence,
                                self.low_confidence_reverify_threshold,
                                self.min_triple_confidence,
                            )

                    # Reject only clear hallucinations: neither entity found anywhere in the document.
                    # Score 0.1 = neither entity present; 0.4+ = at least one entity grounded in text.
                    # Threshold just above 0.1 avoids discarding relationships where one entity
                    # is confirmed (score 0.4) or entity names have minor surface-form mismatches.
                    if triple_confidence < self.min_triple_confidence:
                        relationships_skipped_low_confidence += 1
                        logging.info(
                            "Skipping hallucinated relationship (confidence=%.2f): %s -[%s]-> %s",
                            triple_confidence, rel.get('from'), sanitized_rel_type, rel.get('to'),
                        )
                        continue

                    # Pull negation and qualifiers extracted by LLM
                    negated    = bool(rel.get('negated', False))
                    condition  = rel.get('properties', {}).get('condition') or None
                    quantitative = rel.get('properties', {}).get('quantitative') or None

                    properties_with_confidence = {
                        **properties_filtered,
                        "confidence": triple_confidence,
                        "evidence_scope": evidence_scope,
                        "negated": negated,
                    }
                    if llm_reverified:
                        properties_with_confidence["llm_verified"] = True
                    if condition:
                        properties_with_confidence["condition"] = condition
                    if quantitative:
                        properties_with_confidence["quantitative"] = quantitative
                    properties_with_confidence = self._sanitize_neo4j_properties(
                        properties_with_confidence
                    )

                    # Keep negated in the MERGE key so that opposite claims
                    # (A INHIBITS B negated=false vs true) do not overwrite each other.
                    # Optional qualifiers only participate when present; Neo4j forbids
                    # null property values inside MERGE patterns.
                    rel_query = self._build_relationship_merge_query(
                        sanitized_rel_type,
                        include_condition=condition is not None,
                        include_quantitative=quantitative is not None,
                    )

                    edge_provenance = self._relationship_local_provenance(
                        rel,
                        kg.get("chunks", []),
                    )

                    logging.info(
                        "Creating relationship %d/%d: %s -[%s%s]-> %s (confidence=%.2f)",
                        idx + 1, len(kg['relationships']),
                        rel.get('from'),
                        "NOT " if negated else "",
                        sanitized_rel_type,
                        rel.get('to'), triple_confidence,
                    )

                    graph.query(rel_query, {
                        "source_id": rel.get('from'),
                        "target_id": rel.get('to'),
                        "negated": negated,
                        "condition": condition,
                        "quantitative": quantitative,
                        "properties": properties_with_confidence,
                        "provenance_positions": edge_provenance["provenance_positions"],
                        "question_ids": edge_provenance["question_ids"],
                        "passage_keys": edge_provenance["passage_keys"],
                    })

                    # Create QUALIFIED_BY nodes for significant qualifiers so they
                    # can be traversed independently and appear in path strings.
                    for q_type, q_value in [("condition", condition), ("quantitative", quantitative)]:
                        if not q_value:
                            continue
                        try:
                            qual_id = hashlib.sha1(
                                f"{rel.get('from')}|{sanitized_rel_type}|{rel.get('to')}|{q_type}|{q_value}".encode()
                            ).hexdigest()
                            qual_query = f"""
                            MATCH (source:__Entity__ {{id: $source_id}})
                            MATCH (target:__Entity__ {{id: $target_id}})
                            MERGE (q:Qualifier {{id: $qual_id}})
                            SET q.type = $q_type, q.value = $q_value,
                                q.kgName = $kg_name
                            MERGE (source)-[:{sanitized_rel_type}_QUALIFIED {{negated: $negated}}]->(q)
                            MERGE (q)-[:QUALIFIES]->(target)
                            """
                            graph.query(qual_query, {
                                "source_id": rel.get('from'),
                                "target_id": rel.get('to'),
                                "qual_id":   qual_id,
                                "q_type":    q_type,
                                "q_value":   self._coerce_neo4j_property_value(q_value),
                                "kg_name":   kg.get('metadata', {}).get('kg_name', ''),
                                "negated":   negated,
                            })
                        except Exception as _qe:
                            logging.debug("QUALIFIED_BY node creation failed (non-fatal): %s", _qe)

                    relationships_stored += 1

                except Exception as rel_error:
                    relationship_store_failures += 1
                    logging.error(f"Failed to store relationship {idx+1}: {rel} - Error: {rel_error}")
                    continue

            logging.info(f"Successfully stored {relationships_stored} out of {len(kg['relationships'])} relationships")

            extracted_relationships = len(kg['relationships'])
            attempted_relationships = max(
                0,
                extracted_relationships
                - relationships_skipped_low_confidence
                - relationships_skipped_schema_mismatch,
            )
            relationship_store_ratio = (
                relationships_stored / attempted_relationships
                if attempted_relationships
                else 1.0
            )
            kg.setdefault("metadata", {})
            kg["metadata"]["stored_relationships"] = relationships_stored
            kg["metadata"]["relationship_store_failures"] = relationship_store_failures
            kg["metadata"]["relationships_skipped_low_confidence"] = relationships_skipped_low_confidence
            kg["metadata"]["relationships_skipped_schema_mismatch"] = relationships_skipped_schema_mismatch
            kg["metadata"]["relationships_reverified_kept"] = relationships_reverified_kept
            kg["metadata"]["relationships_reverified_rejected"] = relationships_reverified_rejected
            kg["metadata"]["relationship_store_ratio"] = relationship_store_ratio

            graph.query(
                """
                MATCH (d:Document {fileName: $fileName, kgName: $kgName})
                SET d.totalRelationships = $storedRelationships,
                    d.extractedRelationships = $extractedRelationships,
                    d.relationshipStoreFailures = $relationshipStoreFailures,
                    d.relationshipsSkippedLowConfidence = $relationshipsSkippedLowConfidence,
                    d.relationshipsSkippedSchemaMismatch = $relationshipsSkippedSchemaMismatch,
                    d.relationshipsReverifiedKept = $relationshipsReverifiedKept,
                    d.relationshipsReverifiedRejected = $relationshipsReverifiedRejected
                """,
                {
                    "fileName": file_name,
                    "kgName": kg_name_value,
                    "storedRelationships": relationships_stored,
                    "extractedRelationships": extracted_relationships,
                    "relationshipStoreFailures": relationship_store_failures,
                    "relationshipsSkippedLowConfidence": relationships_skipped_low_confidence,
                    "relationshipsSkippedSchemaMismatch": relationships_skipped_schema_mismatch,
                    "relationshipsReverifiedKept": relationships_reverified_kept,
                    "relationshipsReverifiedRejected": relationships_reverified_rejected,
                },
            )

            if relationship_store_failures > 0:
                # Treat isolated write misses on large graphs as degraded-but-usable.
                # We still fail fast on tiny graphs or when the failure rate suggests
                # a systemic storage problem rather than a one-off edge issue.
                relationship_failure_ratio = (
                    relationship_store_failures / attempted_relationships
                    if attempted_relationships
                    else 1.0
                )
                fail_small_graph = attempted_relationships < 100
                fail_systemic = relationship_failure_ratio > 0.005
                fail_complete_loss = attempted_relationships > 0 and relationships_stored == 0
                should_fail_build = fail_small_graph or fail_systemic or fail_complete_loss

                log_fn = logging.error if should_fail_build else logging.warning
                log_fn(
                    "Relationship storage incomplete for %s: stored=%d attempted=%d skipped_low_confidence=%d failures=%d failure_ratio=%.4f fatal=%s",
                    file_name,
                    relationships_stored,
                    attempted_relationships,
                    relationships_skipped_low_confidence,
                    relationship_store_failures,
                    relationship_failure_ratio,
                    should_fail_build,
                )
                kg["metadata"]["relationship_store_failure_ratio"] = relationship_failure_ratio
                kg["metadata"]["relationship_store_degraded"] = not should_fail_build
                if should_fail_build:
                    return False

            # Link entities to chunks via per-fact provenance Mention nodes
            # Pattern: (Entity)-[:MENTIONED_IN]->(Mention {quote, ...})-[:FROM_CHUNK]->(Chunk)
            def _mention_boundary(name: str) -> re.Pattern:
                """Adaptive boundary pattern — handles names ending in non-word chars like '(' or '-'."""
                prefix = r'(?<!\w)' if not name[:1].isalnum() and name[:1] != '_' else r'\b'
                suffix = r'(?!\w)' if not name[-1:].isalnum() and name[-1:] != '_' else r'\b'
                return re.compile(prefix + re.escape(name) + suffix)

            for chunk in kg['chunks']:
                # Must use the same kg-scoped hash as the chunk CREATE step above.
                chunk_id = hashlib.sha1(f"{kg_name_value}:{file_name}:{chunk['position']}:{chunk['text']}".encode()).hexdigest()
                chunk_text_lower = chunk['text'].lower()

                for node in kg['nodes']:
                    properties = node.get('properties', {})
                    candidate_names = []
                    candidate_names.extend(properties.get('all_names', []) if isinstance(properties.get('all_names', []), list) else [])
                    candidate_names.append(properties.get('name', ''))
                    candidate_names.append(properties.get('original_id', ''))

                    # Keep meaningful normalized names only
                    normalized_names = [n.strip().lower() for n in candidate_names if isinstance(n, str) and len(n.strip()) > 2]
                    matched_name = next(
                        (n for n in normalized_names if _mention_boundary(n).search(chunk_text_lower)),
                        None
                    )
                    if matched_name:
                        # Extract a short quote: the sentence containing the matched name
                        sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
                        quote = next(
                            (s.strip() for s in sentences if matched_name in s.lower()),
                            chunk['text'][:200]
                        )[:500]  # cap at 500 chars
                        # Seed mention_id from node['id'] (the unique key), not content_hash.
                        # content_hash is no longer unique across KGs (Fix 8 changed MERGE to id).
                        mention_id = hashlib.sha256(
                            f"{node['id']}::{chunk_id}".encode()
                        ).hexdigest()
                        mention_query = """
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:__Entity__ {id: $entity_id})
                        MERGE (m:Mention {id: $mention_id})
                        SET m.quote = $quote,
                            m.chunkIndex = $chunk_index,
                            m.chunkLocalIndex = $chunk_local_index,
                            m.chunkStart = $chunk_start,
                            m.chunkEnd = $chunk_end,
                            m.chunkSource = $chunk_source,
                            m.entityName = $entity_name,
                            m.createdAt = datetime()
                        MERGE (e)-[:MENTIONED_IN]->(m)
                        MERGE (m)-[:FROM_CHUNK]->(c)
                        MERGE (c)-[:HAS_ENTITY]->(e)
                        """
                        graph.query(mention_query, {
                            "chunk_id": chunk_id,
                            "entity_id": node['id'],
                            "mention_id": mention_id,
                            "quote": quote,
                            "chunk_index": chunk.get('position', chunk.get('chunk_id', 0)),
                            "chunk_local_index": chunk.get('chunk_local_index', chunk.get('chunk_id', 0)),
                            "chunk_start": chunk.get('start_pos', 0),
                            "chunk_end": chunk.get('end_pos', 0),
                            "chunk_source": chunk.get('source', ''),
                            "entity_name": properties.get('name', ''),
                        })

            enrichment_stats = self._store_graph_enrichment(
                graph,
                kg_name=kg_name_value,
                file_name=file_name,
                enrichment=kg.get("metadata", {}).get("graph_enrichment") or {},
                chunks=kg.get("chunks", []),
            )
            if enrichment_stats:
                kg["metadata"]["component_summary_count"] = enrichment_stats.get("component_summaries", 0)
                kg["metadata"]["claim_count"] = enrichment_stats.get("claims", 0)
                kg["metadata"]["fragmentation_bridge_count"] = enrichment_stats.get("fragmentation_bridges", 0)
                graph.query(
                    """
                    MATCH (d:Document {fileName: $file_name, kgName: $kg_name})
                    SET d.componentSummaryCount = $component_summary_count,
                        d.claimCount = $claim_count,
                        d.fragmentationBridgeCount = $fragmentation_bridge_count
                    """,
                    {
                        "file_name": file_name,
                        "kg_name": kg_name_value,
                        "component_summary_count": int(enrichment_stats.get("component_summaries", 0)),
                        "claim_count": int(enrichment_stats.get("claims", 0)),
                        "fragmentation_bridge_count": int(enrichment_stats.get("fragmentation_bridges", 0)),
                    },
                )

            # Create vector indexes for RAG
            self._create_vector_indexes(graph)

            logging.info(f"Successfully stored ontology-guided knowledge graph for {file_name}")
            return True

        except Exception as e:
            logging.error(f"Error storing knowledge graph: {e}")
            return False

    def _create_vector_indexes(self, graph):
        """
        Create vector indexes and unique constraints for RAG functionality
        """
        try:
            # Create unique constraint for entity IDs to prevent duplicates
            entity_constraint_query = """
            CREATE CONSTRAINT unique_entity_id IF NOT EXISTS
            FOR (e:__Entity__) REQUIRE e.id IS UNIQUE
            """
            graph.query(entity_constraint_query)

            # Create unique constraint for chunk IDs
            chunk_constraint_query = """
            CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
            graph.query(chunk_constraint_query)

            retrieval_chunk_constraint_query = """
            CREATE CONSTRAINT unique_retrieval_chunk_id IF NOT EXISTS
            FOR (rc:RetrievalChunk) REQUIRE rc.id IS UNIQUE
            """
            graph.query(retrieval_chunk_constraint_query)

            # Create composite uniqueness for dataset-scoped documents
            # (same fileName may exist in different kgName datasets)
            doc_constraint_query = """
            CREATE CONSTRAINT unique_document_filename_kgname IF NOT EXISTS
            FOR (d:Document) REQUIRE (d.fileName, d.kgName) IS UNIQUE
            """
            graph.query(doc_constraint_query)

            # Unique constraint for Mention nodes (entity × chunk pair)
            mention_constraint_query = """
            CREATE CONSTRAINT unique_mention_id IF NOT EXISTS
            FOR (m:Mention) REQUIRE m.id IS UNIQUE
            """
            graph.query(mention_constraint_query)

            summary_constraint_query = """
            CREATE CONSTRAINT unique_summary_id IF NOT EXISTS
            FOR (s:Summary) REQUIRE s.id IS UNIQUE
            """
            graph.query(summary_constraint_query)

            claim_constraint_query = """
            CREATE CONSTRAINT unique_claim_id IF NOT EXISTS
            FOR (c:Claim) REQUIRE c.id IS UNIQUE
            """
            graph.query(claim_constraint_query)

            # Create vector index for chunks
            chunk_index_query = f"""
            CREATE VECTOR INDEX vector IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(chunk_index_query)

            retrieval_chunk_index_query = f"""
            CREATE VECTOR INDEX retrieval_vector IF NOT EXISTS
            FOR (rc:RetrievalChunk) ON (rc.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(retrieval_chunk_index_query)

            # Create vector index for entities
            entity_index_query = f"""
            CREATE VECTOR INDEX entity_vector IF NOT EXISTS
            FOR (e:__Entity__) ON (e.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            graph.query(entity_index_query)

            # Create keyword index for full-text search
            keyword_index_query = """
            CREATE FULLTEXT INDEX keyword IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """
            graph.query(keyword_index_query)

            retrieval_keyword_index_query = """
            CREATE FULLTEXT INDEX retrieval_keyword IF NOT EXISTS
            FOR (rc:RetrievalChunk) ON EACH [rc.text]
            """
            graph.query(retrieval_keyword_index_query)

            # Index on entity name for fast text-matching and multi-hop traversal
            # lookups (EnhancedRAGSystem._expand_entities_via_graph seeds from e.id)
            entity_id_index_query = """
            CREATE INDEX entity_id_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.id)
            """
            graph.query(entity_id_index_query)

            entity_name_index_query = """
            CREATE INDEX entity_name_index IF NOT EXISTS
            FOR (e:__Entity__) ON (e.name)
            """
            graph.query(entity_name_index_query)

            # Composite index for chunk→document lookup used in kg_name filtering
            chunk_kg_index_query = """
            CREATE INDEX chunk_document_index IF NOT EXISTS
            FOR ()-[r:PART_OF]-() ON (r)
            """
            try:
                graph.query(chunk_kg_index_query)
            except Exception:
                pass  # Relationship indexes not supported on all Neo4j versions

            logging.info("Created constraints, vector, keyword, and entity lookup indexes for RAG")

        except Exception as e:
            logging.warning(f"Error creating constraints/indexes (may already exist): {e}")

    @staticmethod
    def _kg_eval_normalize_text(value: Any) -> str:
        """Normalize strings for lightweight KG evaluation comparisons."""
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value).strip()).lower()

    @classmethod
    def _kg_eval_entity_lookup(
        cls,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Return id -> canonical entity metadata for evaluation helpers."""
        lookup: Dict[str, Dict[str, str]] = {}
        for node in nodes or []:
            props = node.get("properties") or {}
            entity_id = str(node.get("id") or props.get("id") or "").strip()
            if not entity_id:
                continue
            name = str(
                props.get("name")
                or node.get("name")
                or entity_id
            ).strip()
            entity_type = str(
                node.get("label")
                or node.get("type")
                or props.get("type")
                or props.get("ontology_class")
                or "Unknown"
            ).strip()
            lookup[entity_id] = {
                "id": entity_id,
                "name": name,
                "type": entity_type or "Unknown",
                "name_key": cls._kg_eval_normalize_text(name or entity_id),
                "type_key": cls._kg_eval_normalize_text(entity_type or "Unknown"),
            }
        return lookup

    @classmethod
    def _kg_eval_entity_keys(
        cls,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Return canonical entity key -> display metadata."""
        records: Dict[str, Dict[str, str]] = {}
        for info in cls._kg_eval_entity_lookup(nodes).values():
            key = f"{info['name_key']}||{info['type_key']}"
            records[key] = {
                "id": info["id"],
                "name": info["name"],
                "type": info["type"],
            }
        return records

    @classmethod
    def _kg_eval_relationship_records(
        cls,
        relationships: List[Dict[str, Any]],
        entity_lookup: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Return canonical relationship records for evaluation comparisons."""
        records: List[Dict[str, Any]] = []
        for rel in relationships or []:
            props = rel.get("properties") or {}
            rel_type = str(rel.get("type") or props.get("type") or "").strip()
            if not rel_type:
                continue

            source_ref = rel.get("source") or rel.get("from") or props.get("source")
            target_ref = rel.get("target") or rel.get("to") or props.get("target")
            source_ref = str(source_ref or "").strip()
            target_ref = str(target_ref or "").strip()

            source_name = str(
                props.get("source_name")
                or entity_lookup.get(source_ref, {}).get("name")
                or source_ref
            ).strip()
            target_name = str(
                props.get("target_name")
                or entity_lookup.get(target_ref, {}).get("name")
                or target_ref
            ).strip()

            negated = bool(
                rel.get("negated", props.get("negated", False))
            )
            condition = str(
                rel.get("condition")
                or props.get("condition")
                or ""
            ).strip()
            quantitative = str(
                rel.get("quantitative")
                or props.get("quantitative")
                or ""
            ).strip()

            base_key = "||".join(
                (
                    cls._kg_eval_normalize_text(source_name),
                    rel_type.strip().upper(),
                    cls._kg_eval_normalize_text(target_name),
                )
            )
            qualified_key = "||".join(
                (
                    base_key,
                    "1" if negated else "0",
                    cls._kg_eval_normalize_text(condition),
                    cls._kg_eval_normalize_text(quantitative),
                )
            )
            records.append(
                {
                    "base_key": base_key,
                    "qualified_key": qualified_key,
                    "source_name": source_name,
                    "target_name": target_name,
                    "type": rel_type,
                    "negated": negated,
                    "condition": condition,
                    "quantitative": quantitative,
                }
            )
        return records

    @staticmethod
    def _kg_eval_pr_report(
        predicted_keys: set,
        gold_keys: set,
    ) -> Dict[str, Any]:
        """Compute precision / recall / F1 for canonical key sets."""
        tp = len(predicted_keys & gold_keys)
        fp = len(predicted_keys - gold_keys)
        fn = len(gold_keys - predicted_keys)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall / max(1e-9, precision + recall))
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "gold_count": len(gold_keys),
            "predicted_count": len(predicted_keys),
        }

    def evaluate_knowledge_graph(
        self,
        kg: Dict[str, Any],
        *,
        reference: Optional[Any] = None,
        reference_triples: Optional[List[Dict[str, Any]]] = None,
        print_report: bool = True,
    ) -> Dict[str, Any]:
        """Compute quality metrics for a built KG and optionally compare to a gold set.

        Operates entirely on the in-memory KG dict returned by
        ``generate_knowledge_graph`` / ``generate_knowledge_graph_from_passages``.
        No Neo4j connection required.

        Parameters
        ----------
        kg:
            KG dict as returned by the generate methods.
        reference:
            Optional reference KG dict or list of gold triples. When a KG dict is
            provided, entity precision/recall is computed from its nodes and
            relationship precision/recall is computed from its relationships.
        reference_triples:
            Backward-compatible alias for a list of gold triples. Each entry must
            provide ``source``, ``type``, and ``target`` keys. Relationship
            comparison resolves entity ids back to names when node metadata exists,
            so reference triples can use either names or canonical ids.
        print_report:
            When True, prints a human-readable summary to stdout.

        Returns
        -------
        Dict with sections: ``summary``, ``entity_metrics``, ``relationship_metrics``,
        ``grounding_metrics``, ``pipeline_metrics``, and optionally
        ``entity_precision_recall`` / ``relationship_precision_recall``.
        """
        nodes = kg.get("nodes") or []
        relationships = kg.get("relationships") or []
        chunks = kg.get("chunks") or []
        meta = kg.get("metadata") or {}
        entity_lookup = self._kg_eval_entity_lookup(nodes)

        # ── Entity metrics ─────────────────────────────────────────────────
        type_counts: Dict[str, int] = Counter(
            str(n.get("label") or n.get("type") or "Unknown") for n in nodes
        )
        anchor_grounded = sum(
            1 for n in nodes if (n.get("properties") or {}).get("anchor_spans")
        )
        umls_linked = sum(
            1 for n in nodes if (n.get("properties") or {}).get("umls_cui")
        )
        has_synonyms = sum(
            1 for n in nodes
            if len((n.get("properties") or {}).get("all_names") or []) > 1
        )
        n_entities = len(nodes)

        # degree distribution — entities appear as source/target in relationships
        degree: Dict[str, int] = Counter()
        for rel in relationships:
            src = rel.get("source") or rel.get("from")
            tgt = rel.get("target") or rel.get("to")
            if src:
                degree[str(src)] += 1
            if tgt:
                degree[str(tgt)] += 1
        degrees = list(degree.values()) if degree else [0]
        sorted_degrees = sorted(degrees, reverse=True)
        top10_degree_share = (
            sum(sorted_degrees[:10]) / max(1, sum(sorted_degrees))
        ) if sorted_degrees else 0.0
        isolated_entities = n_entities - len(degree)
        degree_buckets = {"0": 0, "1": 0, "2-4": 0, "5-9": 0, "10+": 0}
        for node in nodes:
            entity_id = str(node.get("id") or "").strip()
            node_degree = int(degree.get(entity_id, 0))
            if node_degree <= 0:
                degree_buckets["0"] += 1
            elif node_degree == 1:
                degree_buckets["1"] += 1
            elif node_degree <= 4:
                degree_buckets["2-4"] += 1
            elif node_degree <= 9:
                degree_buckets["5-9"] += 1
            else:
                degree_buckets["10+"] += 1
        top_hub_entities = []
        for entity_id, node_degree in degree.most_common(10):
            info = entity_lookup.get(entity_id, {})
            top_hub_entities.append(
                {
                    "id": entity_id,
                    "name": info.get("name") or entity_id,
                    "type": info.get("type") or "Unknown",
                    "degree": int(node_degree),
                }
            )

        # ── Relationship metrics ────────────────────────────────────────────
        n_rels = len(relationships)
        rel_type_counts: Dict[str, int] = Counter(
            str(r.get("type") or "UNKNOWN") for r in relationships
        )
        negated_count = sum(1 for r in relationships if r.get("negated"))
        contradiction_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("contradiction_detected")
               or r.get("contradiction_detected")
        )
        conditioned_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("condition")
        )
        quantified_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("quantitative")
        )

        # evidence scope distribution
        scope_counts: Dict[str, int] = Counter()
        confidence_values: List[float] = []
        for r in relationships:
            props = r.get("properties") or {}
            scope = props.get("evidence_scope") or "unknown"
            scope_counts[scope] += 1
            raw_conf = props.get("confidence")
            if raw_conf not in (None, "", "null", "NULL"):
                confidence_values.append(self._confidence_score(raw_conf))

        mean_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else None
        )
        high_conf_rels = sum(1 for c in confidence_values if c >= 0.7)
        low_conf_rels = sum(1 for c in confidence_values if c < 0.4)

        restoration_full = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("restoration_status") == "full"
        )
        restoration_partial = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("restoration_status") == "partial"
        )
        restoration_failed = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("restoration_status") == "failed"
        )

        # ── Grounding metrics ───────────────────────────────────────────────
        n_chunks = len(chunks)
        entities_per_chunk = n_entities / max(1, n_chunks)
        rels_per_chunk = n_rels / max(1, n_chunks)
        anchor_rel_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("anchor_text")
        )
        anchor_grounding_count = sum(
            1 for r in relationships
            if (r.get("properties") or {}).get("anchor_grounding")
        )
        section_names: set = {
            str(c.get("section_name") or "")
            for c in chunks
            if c.get("section_name")
        }

        # ── Pipeline health metrics ─────────────────────────────────────────
        dropped_entities = meta.get("schema_enforcement_dropped_entities", 0)
        dropped_rels_schema = meta.get("schema_enforcement_dropped_relationships", 0)
        dropped_rels_unmapped = meta.get("harmonization_relationships_dropped_unmapped", 0)
        deduped_rels = meta.get("harmonization_relationships_deduped", 0)
        contradiction_groups = meta.get("harmonization_relationship_contradiction_groups", 0)
        stored_relationships = meta.get("stored_relationships")
        relationship_store_failures = meta.get("relationship_store_failures", 0)
        relationships_skipped_low_confidence = meta.get("relationships_skipped_low_confidence", 0)
        relationships_skipped_schema_mismatch = meta.get("relationships_skipped_schema_mismatch", 0)
        relationships_reverified_kept = meta.get("relationships_reverified_kept", 0)
        relationships_reverified_rejected = meta.get("relationships_reverified_rejected", 0)
        relationship_store_ratio = meta.get("relationship_store_ratio")

        raw_entities_approx = n_entities + dropped_entities
        entity_pass_rate = n_entities / max(1, raw_entities_approx)
        raw_rels_approx = n_rels + dropped_rels_schema + dropped_rels_unmapped + deduped_rels
        rel_pass_rate = n_rels / max(1, raw_rels_approx)

        # ── Precision / recall vs. gold ─────────────────────────────────────
        entity_precision_recall: Optional[Dict[str, Any]] = None
        relationship_precision_recall: Optional[Dict[str, Any]] = None
        qualified_relationship_precision_recall: Optional[Dict[str, Any]] = None
        reference_nodes: List[Dict[str, Any]] = []
        reference_relationships: List[Dict[str, Any]] = []
        if isinstance(reference, dict):
            reference_nodes = list(reference.get("nodes") or reference.get("entities") or [])
            reference_relationships = list(reference.get("relationships") or reference.get("triples") or [])
        elif isinstance(reference, list):
            reference_relationships = list(reference)
        if reference_triples:
            reference_relationships = list(reference_relationships) + list(reference_triples)

        if reference_nodes:
            gold_entity_keys = set(self._kg_eval_entity_keys(reference_nodes))
            pred_entity_keys = set(self._kg_eval_entity_keys(nodes))
            entity_precision_recall = self._kg_eval_pr_report(
                pred_entity_keys,
                gold_entity_keys,
            )
            entity_precision_recall["gold_entities"] = entity_precision_recall.pop("gold_count")
            entity_precision_recall["predicted_entities"] = entity_precision_recall.pop("predicted_count")

        if reference_relationships:
            reference_lookup = dict(entity_lookup)
            reference_lookup.update(self._kg_eval_entity_lookup(reference_nodes))
            pred_relationship_records = self._kg_eval_relationship_records(
                relationships,
                entity_lookup,
            )
            gold_relationship_records = self._kg_eval_relationship_records(
                reference_relationships,
                reference_lookup,
            )
            pred_keys = {record["base_key"] for record in pred_relationship_records}
            gold_keys = {record["base_key"] for record in gold_relationship_records}
            relationship_precision_recall = self._kg_eval_pr_report(
                pred_keys,
                gold_keys,
            )
            relationship_precision_recall["gold_triples"] = relationship_precision_recall.pop("gold_count")
            relationship_precision_recall["predicted_triples"] = relationship_precision_recall.pop("predicted_count")

            has_qualified_reference = any(
                record["negated"] or record["condition"] or record["quantitative"]
                for record in (pred_relationship_records + gold_relationship_records)
            )
            if has_qualified_reference:
                pred_qualified_keys = {
                    record["qualified_key"] for record in pred_relationship_records
                }
                gold_qualified_keys = {
                    record["qualified_key"] for record in gold_relationship_records
                }
                qualified_relationship_precision_recall = self._kg_eval_pr_report(
                    pred_qualified_keys,
                    gold_qualified_keys,
                )
                qualified_relationship_precision_recall["gold_triples"] = qualified_relationship_precision_recall.pop("gold_count")
                qualified_relationship_precision_recall["predicted_triples"] = qualified_relationship_precision_recall.pop("predicted_count")

        # ── Assemble report ─────────────────────────────────────────────────
        report: Dict[str, Any] = {
            "summary": {
                "entities": n_entities,
                "relationships": n_rels,
                "chunks": n_chunks,
                "entity_types": len(type_counts),
                "relationship_types": len(rel_type_counts),
                "sections_detected": sorted(section_names),
                "extraction_method": meta.get("extraction_method", "unknown"),
                "kg_name": meta.get("kg_name"),
                "created_at": meta.get("created_at"),
            },
            "entity_metrics": {
                "type_distribution": dict(type_counts.most_common()),
                "anchor_grounded": anchor_grounded,
                "anchor_grounded_pct": round(100 * anchor_grounded / max(1, n_entities), 1),
                "umls_linked": umls_linked,
                "umls_linked_pct": round(100 * umls_linked / max(1, n_entities), 1),
                "with_synonyms": has_synonyms,
                "isolated_no_edges": isolated_entities,
                "mean_degree": round(sum(degrees) / max(1, len(degrees)), 2),
                "max_degree": max(degrees),
                "top10_entity_degree_share_pct": round(100 * top10_degree_share, 1),
                "degree_distribution": degree_buckets,
                "hub_entities_topk": top_hub_entities,
            },
            "relationship_metrics": {
                "type_distribution": dict(rel_type_counts.most_common(20)),
                "negated": negated_count,
                "negated_pct": round(100 * negated_count / max(1, n_rels), 1),
                "contradiction_flagged": contradiction_count,
                "contradiction_groups": contradiction_groups,
                "contradiction_rate_pct": round(100 * contradiction_count / max(1, n_rels), 1),
                "conditioned": conditioned_count,
                "conditioned_pct": round(100 * conditioned_count / max(1, n_rels), 1),
                "quantified": quantified_count,
                "quantified_pct": round(100 * quantified_count / max(1, n_rels), 1),
                "evidence_scope_distribution": dict(scope_counts),
                "mean_confidence": round(mean_confidence, 4) if mean_confidence is not None else None,
                "high_confidence_pct": round(100 * high_conf_rels / max(1, len(confidence_values)), 1),
                "low_confidence_pct": round(100 * low_conf_rels / max(1, len(confidence_values)), 1),
                "restoration_full": restoration_full,
                "restoration_partial": restoration_partial,
                "restoration_failed": restoration_failed,
                "anchor_relation_phrase_pct": round(100 * anchor_rel_count / max(1, n_rels), 1),
                "anchor_grounding_pct": round(100 * anchor_grounding_count / max(1, n_rels), 1),
            },
            "grounding_metrics": {
                "entities_per_chunk": round(entities_per_chunk, 2),
                "relationships_per_chunk": round(rels_per_chunk, 2),
                "chunks_with_section_tags": sum(1 for c in chunks if c.get("section_name")),
            },
            "pipeline_metrics": {
                "schema_enforcement_dropped_entities": dropped_entities,
                "schema_enforcement_dropped_relationships": dropped_rels_schema,
                "harmonization_dropped_unmapped": dropped_rels_unmapped,
                "harmonization_deduped_relationships": deduped_rels,
                "entity_schema_pass_rate_pct": round(100 * entity_pass_rate, 1),
                "relationship_pass_rate_pct": round(100 * rel_pass_rate, 1),
                "stored_relationships": stored_relationships,
                "relationship_store_failures": relationship_store_failures,
                "relationships_skipped_low_confidence": relationships_skipped_low_confidence,
                "relationships_skipped_schema_mismatch": relationships_skipped_schema_mismatch,
                "relationships_reverified_kept": relationships_reverified_kept,
                "relationships_reverified_rejected": relationships_reverified_rejected,
                "relationship_store_ratio": round(float(relationship_store_ratio), 4)
                if isinstance(relationship_store_ratio, (int, float))
                else relationship_store_ratio,
            },
        }
        if entity_precision_recall is not None:
            report["entity_precision_recall"] = entity_precision_recall
        if relationship_precision_recall is not None:
            report["relationship_precision_recall"] = relationship_precision_recall
            # Backward-compatible alias for earlier callers that only expected
            # relationship-level precision / recall.
            report["precision_recall"] = relationship_precision_recall
        if qualified_relationship_precision_recall is not None:
            report["qualified_relationship_precision_recall"] = qualified_relationship_precision_recall

        if print_report:
            self._print_kg_evaluation_report(report)

        return report

    @staticmethod
    def _print_kg_evaluation_report(report: Dict[str, Any]) -> None:
        """Print a human-readable KG evaluation report to stdout."""
        sep = "─" * 60
        s = report["summary"]
        em = report["entity_metrics"]
        rm = report["relationship_metrics"]
        gm = report["grounding_metrics"]
        pm = report["pipeline_metrics"]

        lines = [
            sep,
            "KG EVALUATION REPORT",
            sep,
            f"KG name          : {s.get('kg_name') or '(unnamed)'}",
            f"Extraction method: {s['extraction_method']}",
            f"Created at       : {s.get('created_at') or 'N/A'}",
            "",
            "── SUMMARY ──────────────────────────────────────────────────",
            f"  Entities        : {s['entities']}  ({s['entity_types']} types)",
            f"  Relationships   : {s['relationships']}  ({s['relationship_types']} types)",
            f"  Chunks          : {s['chunks']}",
            f"  Sections        : {', '.join(s['sections_detected']) or 'none detected'}",
            "",
            "── ENTITY QUALITY ───────────────────────────────────────────",
            f"  Anchor-grounded : {em['anchor_grounded']}  ({em['anchor_grounded_pct']}%)",
            f"  UMLS-linked     : {em['umls_linked']}  ({em['umls_linked_pct']}%)",
            f"  With synonyms   : {em['with_synonyms']}",
            f"  Isolated (no edges): {em['isolated_no_edges']}",
            f"  Mean degree     : {em['mean_degree']}  |  Max: {em['max_degree']}",
            f"  Top-10 hub share: {em['top10_entity_degree_share_pct']}% of all edges",
            f"  Degree buckets  : {em['degree_distribution']}",
            f"  Type distribution (top 10):",
        ]
        for etype, cnt in list(em["type_distribution"].items())[:10]:
            lines.append(f"    {etype:<30} {cnt}")
        if em.get("hub_entities_topk"):
            lines.append("  Top hub entities:")
            for hub in em["hub_entities_topk"][:5]:
                lines.append(
                    f"    {hub['name']:<25} {hub['degree']}  ({hub['type']})"
                )

        lines += [
            "",
            "── RELATIONSHIP QUALITY ─────────────────────────────────────",
            f"  Negated         : {rm['negated']}  ({rm['negated_pct']}%)",
            f"  Contradictions  : {rm['contradiction_flagged']} edges in {rm['contradiction_groups']} groups ({rm['contradiction_rate_pct']}%)",
            f"  Conditioned     : {rm['conditioned']}  ({rm['conditioned_pct']}%)",
            f"  Quantified      : {rm['quantified']}  ({rm['quantified_pct']}%)",
            f"  Mean confidence : {rm['mean_confidence']}  |  High-conf: {rm['high_confidence_pct']}%  Low-conf: {rm['low_confidence_pct']}%",
            f"  Anchor relation phrase: {rm['anchor_relation_phrase_pct']}%",
            f"  Anchor grounding      : {rm['anchor_grounding_pct']}%",
            f"  Restoration (full/partial/failed): {rm['restoration_full']} / {rm['restoration_partial']} / {rm['restoration_failed']}",
            f"  Evidence scope:",
        ]
        for scope, cnt in sorted(rm["evidence_scope_distribution"].items(), key=lambda x: -x[1]):
            pct = round(100 * cnt / max(1, s["relationships"]), 1)
            lines.append(f"    {scope:<25} {cnt}  ({pct}%)")

        lines += [
            "",
            "── GROUNDING ────────────────────────────────────────────────",
            f"  Entities/chunk  : {gm['entities_per_chunk']}",
            f"  Relations/chunk : {gm['relationships_per_chunk']}",
            f"  Section-tagged chunks: {gm['chunks_with_section_tags']} / {s['chunks']}",
            "",
            "── PIPELINE HEALTH ──────────────────────────────────────────",
            f"  Entity schema pass rate : {pm['entity_schema_pass_rate_pct']}%",
            f"  Relation pass rate      : {pm['relationship_pass_rate_pct']}%",
            f"  Dropped entities (schema): {pm['schema_enforcement_dropped_entities']}",
            f"  Dropped rels (schema)    : {pm['schema_enforcement_dropped_relationships']}",
            f"  Dropped rels (unmapped)  : {pm['harmonization_dropped_unmapped']}",
            f"  Deduped relationships    : {pm['harmonization_deduped_relationships']}",
            f"  Stored relationships     : {pm['stored_relationships']}",
            f"  Store failures           : {pm['relationship_store_failures']}",
            f"  Skipped low-confidence   : {pm['relationships_skipped_low_confidence']}",
            f"  Skipped schema-mismatch  : {pm['relationships_skipped_schema_mismatch']}",
            f"  Reverified kept/rejected : {pm['relationships_reverified_kept']} / {pm['relationships_reverified_rejected']}",
            f"  Store ratio              : {pm['relationship_store_ratio']}",
        ]

        if "entity_precision_recall" in report:
            pr = report["entity_precision_recall"]
            lines += [
                "",
                "── ENTITY PRECISION / RECALL vs. GOLD ──────────────────────",
                f"  Gold entities   : {pr['gold_entities']}",
                f"  Predicted       : {pr['predicted_entities']}",
                f"  True positives  : {pr['true_positives']}",
                f"  False positives : {pr['false_positives']}",
                f"  False negatives : {pr['false_negatives']}",
                f"  Precision       : {pr['precision']}",
                f"  Recall          : {pr['recall']}",
                f"  F1              : {pr['f1']}",
            ]

        if "precision_recall" in report:
            pr = report["precision_recall"]
            lines += [
                "",
                "── PRECISION / RECALL vs. GOLD ──────────────────────────────",
                f"  Gold triples    : {pr['gold_triples']}",
                f"  Predicted       : {pr['predicted_triples']}",
                f"  True positives  : {pr['true_positives']}",
                f"  False positives : {pr['false_positives']}",
                f"  False negatives : {pr['false_negatives']}",
                f"  Precision       : {pr['precision']}",
                f"  Recall          : {pr['recall']}",
                f"  F1              : {pr['f1']}",
            ]
        if "qualified_relationship_precision_recall" in report:
            pr = report["qualified_relationship_precision_recall"]
            lines += [
                "",
                "── QUALIFIED RELATIONSHIP P/R vs. GOLD ─────────────────────",
                f"  Gold triples    : {pr['gold_triples']}",
                f"  Predicted       : {pr['predicted_triples']}",
                f"  True positives  : {pr['true_positives']}",
                f"  False positives : {pr['false_positives']}",
                f"  False negatives : {pr['false_negatives']}",
                f"  Precision       : {pr['precision']}",
                f"  Recall          : {pr['recall']}",
                f"  F1              : {pr['f1']}",
            ]

        lines.append(sep)
        print("\n".join(lines))

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
