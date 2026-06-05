"""
Regression tests for KG construction contracts.

These are pure unit tests — no live Neo4j or LLM calls.
They lock down the behaviours that must survive refactoring:

  - chunk_id is scoped to (kg_name, file_name, text)
  - _verify_triple_confidence scoring tiers
  - _verify_triple_confidence alias coverage
  - _harmonize_entities deduplication by normalised text
  - relation MERGE key includes negated and only includes qualifiers when present
  - relationship store failures fail small/broken builds but tiny misses are tolerated on large graphs
  - synonym merge type guard reads e.type / e.ontology_class
"""

import hashlib
import json
import re
import sys
import os

import pytest

# Skip the whole module gracefully if heavy KG dependencies aren't installed.
# The project venv includes all of these; this guard is for CI environments
# that only install a subset of requirements.
pytest.importorskip("langchain_neo4j", reason="langchain_neo4j not installed — skipping KG build tests")
pytest.importorskip("langchain", reason="langchain not installed — skipping KG build tests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.kg.builders import ontology_guided_kg_creator as builder_mod
from ontographrag.kg.builders.ontology_guided_kg_creator import OntologyGuidedKGCreator
from ontographrag.schemas.models import EntityType, OntologySchema, RelationshipType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_creator() -> OntologyGuidedKGCreator:
    """Instantiate with minimal config — no Neo4j, no embeddings required."""
    c = OntologyGuidedKGCreator.__new__(OntologyGuidedKGCreator)
    c.chunk_size = 500
    c.chunk_overlap = 50
    c.retrieval_chunk_size = 128
    c.retrieval_chunk_overlap = 32
    c.ontology_classes = []
    c.ontology_relationships = []
    c.schema_card = {}
    c.ontology_path = None
    c.strict_ontology = True
    c.enable_coreference_resolution = False
    c.enable_heuristic_coreference_resolution = True
    c.self_consistency_n = 1
    c.few_shot_example_count = 2
    c.min_triple_confidence = 0.15
    c.relationship_type_similarity_threshold = 0.62
    c.enable_low_confidence_triple_reverification = False
    c.low_confidence_reverify_threshold = 0.4
    c.enable_umls_linking = False
    c.umls_spacy_model = "en_core_sci_sm"
    c.enable_anchor_constrained_extraction = True
    c.enable_anchor_coverage_supplement = True
    c.enable_cross_passage_relation_recovery = True
    c.enable_self_reflection = True
    c.enable_soft_entity_linking = False
    c.soft_entity_similarity_threshold = 0.88
    c.enable_fragmentation_repair = False
    c.fragmentation_bridge_similarity_threshold = 0.92
    c.max_fragmentation_bridges = 8
    c.enable_graph_summaries = False
    c.enable_claim_extraction = False
    c.max_summary_entities = 6
    c.max_summary_relationships = 6
    c.cross_chunk_relation_window = 3
    c.cross_section_relation_window = 2
    c.cross_passage_relation_window = 2
    c.max_relationship_prompt_entities = 40
    c._ontology_schema = None
    c._ontology_class_embeddings = []
    c._ontology_relationship_embeddings = []
    c._umls_linker_state = "disabled"
    c._umls_nlp = None
    c._triple_reverification_cache = {}
    c._last_schema_enforcement_stats = {"dropped_entities": 0, "dropped_relationships": 0, "kept_entities": 0, "kept_relationships": 0}
    c._last_relationship_harmonization_stats = {"kept": 0, "dropped_unmapped": 0, "dropped_schema_mismatch": 0, "deduped": 0}
    c._last_relationship_contradiction_stats = {"contradiction_groups": 0, "contradiction_edges": 0}
    c.embedding_model = "sentence_transformers"
    # Stub embedding so _chunk_text works without a model
    emb = type(
        "Emb",
        (),
        {
            "embed_query": lambda self, t: [0.0] * 384,
            "embed_documents": lambda self, docs: [[0.0] * 384 for _ in docs],
        },
    )()
    c.embedding_function = emb
    c.embedding_dimension = 384
    return c


def _chunk_id(kg_name: str, file_name: str, text: str) -> str:
    return hashlib.sha1(f"{kg_name}:{file_name}:{text}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# 0. entity-name guardrails
# ---------------------------------------------------------------------------

class TestEntityNameValidation:
    def test_allows_named_biomedical_identifier(self):
        assert builder_mod._is_valid_entity_name("IL-6") is True
        assert builder_mod._is_valid_entity_name("p53") is True

    def test_rejects_generic_hub_term(self):
        assert builder_mod._is_valid_entity_name("treatment") is False

    def test_rejects_bare_numeric_fragment(self):
        assert builder_mod._is_valid_entity_name("123") is False
        assert builder_mod._is_valid_entity_name("12.5") is False

    def test_rejects_punctuation_only_fragment(self):
        assert builder_mod._is_valid_entity_name("...") is False
        assert builder_mod._is_valid_entity_name("--") is False


# ---------------------------------------------------------------------------
# 1. chunk_id scoping
# ---------------------------------------------------------------------------

class TestChunkIdScoping:
    """chunk_id must be unique across different (kg_name, file_name, text) tuples."""

    def test_same_text_different_kg_gives_different_id(self):
        t = "Aspirin reduces fever."
        id1 = _chunk_id("kg_a", "doc.txt", t)
        id2 = _chunk_id("kg_b", "doc.txt", t)
        assert id1 != id2

    def test_same_text_different_file_gives_different_id(self):
        t = "Aspirin reduces fever."
        id1 = _chunk_id("kg_a", "doc1.txt", t)
        id2 = _chunk_id("kg_a", "doc2.txt", t)
        assert id1 != id2

    def test_same_inputs_gives_stable_id(self):
        id1 = _chunk_id("kg_a", "doc.txt", "text")
        id2 = _chunk_id("kg_a", "doc.txt", "text")
        assert id1 == id2

    def test_id_is_40_char_hex(self):
        cid = _chunk_id("kg_a", "doc.txt", "hello")
        assert re.fullmatch(r"[0-9a-f]{40}", cid)


# ---------------------------------------------------------------------------
# 1b. Passage provenance survives storage
# ---------------------------------------------------------------------------

class TestPassageProvenanceStorage:
    def _kg(self) -> dict:
        return {
            "nodes": [
                {
                    "id": "musique_entity_alice",
                    "label": "Person",
                    "properties": {
                        "name": "Alice",
                        "type": "Person",
                        "original_id": "Alice",
                    },
                    "embedding": None,
                }
            ],
            "relationships": [],
            "chunks": [
                {
                    "text": "Alice wrote a book.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 19,
                    "source": "musique/q1/p0",
                    "dataset": "musique",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 1,
                "total_relationships": 0,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "musique",
            },
        }

    def test_chunk_storage_includes_passage_provenance(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        ok = creator.store_knowledge_graph_with_embeddings(self._kg(), "musique")

        assert ok is True
        _, params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (c:Chunk {id: $chunk_id})" in query
        )
        assert params["position"] == 7
        assert params["chunk_local_index"] == 0
        assert params["source"] == "musique/q1/p0"
        assert params["dataset"] == "musique"
        assert params["question_id"] == "q1"
        assert params["passage_index"] == 0


    def test_mentions_use_global_chunk_index_and_source(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        ok = creator.store_knowledge_graph_with_embeddings(self._kg(), "musique")

        assert ok is True
        _, params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (m:Mention {id: $mention_id})" in query
        )
        assert params["chunk_index"] == 7
        assert params["chunk_local_index"] == 0
        assert params["chunk_source"] == "musique/q1/p0"


class TestCrossPassageRecoveryScoping:
    def test_cross_passage_groups_respect_question_bundle_boundaries(self):
        creator = _make_creator()
        Passage = type("Passage", (), {})
        passages = []
        for dataset, question_id, idx, text in (
            ("2wikimultihopqa", "q1", 0, "passage q1 p0"),
            ("2wikimultihopqa", "q1", 1, "passage q1 p1"),
            ("2wikimultihopqa", "q2", 0, "passage q2 p0"),
            ("2wikimultihopqa", "q2", 1, "passage q2 p1"),
        ):
            passage = Passage()
            passage.dataset = dataset
            passage.question_id = question_id
            passage.passage_index = idx
            passage.text = text
            passages.append(passage)

        groups = creator._group_passages_for_cross_passage_recovery(
            passages,
            {
                0: [{"id": "a"}],
                1: [{"id": "b"}],
                2: [{"id": "c"}],
                3: [{"id": "d"}],
            },
            {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
            },
        )

        assert len(groups) == 2
        assert groups[0]["source_label"] == "Cross-passage[2wikimultihopqa/q1]"
        assert groups[1]["source_label"] == "Cross-passage[2wikimultihopqa/q2]"
        assert groups[0]["texts"] == ["passage q1 p0", "passage q1 p1"]
        assert groups[1]["texts"] == ["passage q2 p0", "passage q2 p1"]
        assert groups[0]["positions"] == {0: [0], 1: [1]}
        assert groups[1]["positions"] == {0: [2], 1: [3]}


class TestPassageSourceTitlePropagation:
    def test_passage_aware_build_prefers_structured_source_title(self, monkeypatch):
        creator = _make_creator()
        captured = {}

        monkeypatch.setattr(
            creator,
            "_chunk_text_with_section_boundaries",
            lambda text, section_headers=None: [{
                "text": text,
                "chunk_id": 0,
                "position": 0,
                "start_pos": 0,
                "end_pos": len(text),
                "embedding": None,
            }],
        )
        monkeypatch.setattr(creator, "_detect_section_headers", lambda text: [])
        monkeypatch.setattr(creator, "_get_section_for_position", lambda pos, headers: None)
        monkeypatch.setattr(creator, "_prepare_chunk_text_for_extraction", lambda text, **kwargs: text)
        monkeypatch.setattr(
            creator,
            "_extract_entities_and_relationships_with_llm",
            lambda text, llm, model_name, context_header=None, section_header=None: {
                "entities": [{"id": "James Bond", "type": "Person", "properties": {}}],
                "relationships": [],
            },
        )

        def _capture_grounding(chunk_kg, chunk):
            captured["source_title"] = chunk.get("source_title")
            return chunk_kg

        monkeypatch.setattr(creator, "_ground_chunk_extraction", _capture_grounding)
        monkeypatch.setattr(creator, "_extract_relationships_for_segment_windows", lambda *args, **kwargs: [])
        monkeypatch.setattr(creator, "_recover_cross_section_relationships", lambda *args, **kwargs: [])

        Passage = type("Passage", (), {})
        passage = Passage()
        passage.dataset = "hotpotqa"
        passage.question_id = "q1"
        passage.passage_index = 0
        passage.text = "Dr. No. James Bond appears in the film."
        passage.source_title = "Dr. No"

        kg = creator.generate_knowledge_graph_from_passages(
            [passage],
            llm=object(),
            file_name=None,
            model_name="test-model",
            kg_name="hotpotqa",
        )

        assert captured["source_title"] == "Dr. No"
        assert kg["chunks"][0]["source_title"] == "Dr. No"

    def test_storage_sanitizes_nested_metadata_and_relationship_properties(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 1.0

        kg = {
            "nodes": [
                {
                    "id": "musique_entity_believer",
                    "label": "Person",
                    "properties": {
                        "name": "believer",
                        "type": "Person",
                        "original_id": "believer",
                    },
                    "embedding": None,
                },
                {
                    "id": "musique_entity_religion",
                    "label": "Religion",
                    "properties": {
                        "name": "religion",
                        "type": "Religion",
                        "original_id": "religion",
                    },
                    "embedding": None,
                },
            ],
            "relationships": [
                {
                    "id": "musique_rel_believer_devotee_religion_0",
                    "from": "musique_entity_believer",
                    "to": "musique_entity_religion",
                    "source": "musique_entity_believer",
                    "target": "musique_entity_religion",
                    "type": "DEVOTED_TO",
                    "label": "DEVOTED_TO",
                    "negated": False,
                    "properties": {
                        "schema_hint": {
                            "name": "devotees",
                            "description": "Individuals who are dedicated to a particular religion or deity.",
                        }
                    },
                    "provenance_positions": [7],
                }
            ],
            "chunks": [
                {
                    "text": "The believer was devoted to the religion.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 42,
                    "source": "musique/q1/p0",
                    "dataset": "musique",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 2,
                "total_relationships": 1,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "musique",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(
            kg,
            "musique",
            doc_metadata={
                "datasetKgScope": "shared_corpus",
                "selectionDetail": {
                    "subset": "seed42",
                    "count": 100,
                },
            },
        )

        assert ok is True

        _, doc_params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "SET d += $meta" in query
        )
        assert isinstance(doc_params["meta"]["selectionDetail"], str)
        assert '"subset": "seed42"' in doc_params["meta"]["selectionDetail"]

        _, rel_params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (source)-[r:" in query
        )
        assert isinstance(rel_params["properties"]["schema_hint"], str)
        assert '"name": "devotees"' in rel_params["properties"]["schema_hint"]

    def test_relationship_verification_uses_local_provenance_chunks(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        captured = {}

        def fake_verify(source_name, target_name, rel_type, chunks, **kwargs):
            captured["chunks"] = chunks
            return 1.0

        creator._verify_triple_confidence = fake_verify

        kg = {
            "nodes": [
                {
                    "id": "musique_entity_alice",
                    "label": "Person",
                    "properties": {"name": "Alice", "type": "Person", "original_id": "Alice"},
                    "embedding": None,
                },
                {
                    "id": "musique_entity_bob",
                    "label": "Person",
                    "properties": {"name": "Bob", "type": "Person", "original_id": "Bob"},
                    "embedding": None,
                },
            ],
            "relationships": [
                {
                    "id": "musique_rel_alice_knows_bob_0",
                    "from": "musique_entity_alice",
                    "to": "musique_entity_bob",
                    "source": "musique_entity_alice",
                    "target": "musique_entity_bob",
                    "type": "KNOWS",
                    "label": "KNOWS",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [7],
                }
            ],
            "chunks": [
                {
                    "text": "Alice knows Bob from school.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 27,
                    "source": "musique/q1/p0",
                    "dataset": "musique",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                },
                {
                    "text": "Irrelevant extra passage.",
                    "chunk_id": 1,
                    "chunk_local_index": 0,
                    "position": 8,
                    "start_pos": 0,
                    "end_pos": 24,
                    "source": "musique/q2/p0",
                    "dataset": "musique",
                    "question_id": "q2",
                    "passage_index": 0,
                    "embedding": None,
                },
            ],
            "metadata": {
                "total_chunks": 2,
                "total_entities": 2,
                "total_relationships": 1,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "musique",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "musique")

        assert ok is True
        assert [chunk["position"] for chunk in captured["chunks"]] == [7]

    def test_relationship_store_failure_returns_false(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 1.0

        original_query = stub_graph.query

        def flaky_query(query, params=None):
            if "MERGE (source)-[r:" in query:
                raise RuntimeError("forced relationship store failure")
            return original_query(query, params)

        stub_graph.query = flaky_query

        kg = {
            "nodes": [
                {
                    "id": "musique_entity_alice",
                    "label": "Person",
                    "properties": {"name": "Alice", "type": "Person", "original_id": "Alice"},
                    "embedding": None,
                },
                {
                    "id": "musique_entity_bob",
                    "label": "Person",
                    "properties": {"name": "Bob", "type": "Person", "original_id": "Bob"},
                    "embedding": None,
                },
            ],
            "relationships": [
                {
                    "id": "musique_rel_alice_knows_bob_0",
                    "from": "musique_entity_alice",
                    "to": "musique_entity_bob",
                    "source": "musique_entity_alice",
                    "target": "musique_entity_bob",
                    "type": "KNOWS",
                    "label": "KNOWS",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [7],
                }
            ],
            "chunks": [
                {
                    "text": "Alice knows Bob from school.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 27,
                    "source": "musique/q1/p0",
                    "dataset": "musique",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 2,
                "total_relationships": 1,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "musique",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "musique")

        assert ok is False
        assert kg["metadata"]["relationship_store_failures"] == 1
        assert kg["metadata"]["stored_relationships"] == 0

    def test_large_graph_tolerates_single_relationship_store_failure(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 1.0

        original_query = stub_graph.query
        failed_once = {"done": False}

        def flaky_query(query, params=None):
            if "MERGE (source)-[r:" in query and not failed_once["done"]:
                failed_once["done"] = True
                raise RuntimeError("forced relationship store failure")
            return original_query(query, params)

        stub_graph.query = flaky_query

        kg = {
            "nodes": [
                {
                    "id": f"bioasq_entity_{idx}",
                    "label": "Person",
                    "properties": {
                        "name": f"Entity {idx}",
                        "type": "Person",
                        "original_id": f"Entity {idx}",
                    },
                    "embedding": None,
                }
                for idx in range(201)
            ],
            "relationships": [
                {
                    "id": f"bioasq_rel_{idx}",
                    "from": f"bioasq_entity_{idx}",
                    "to": f"bioasq_entity_{idx + 1}",
                    "source": f"bioasq_entity_{idx}",
                    "target": f"bioasq_entity_{idx + 1}",
                    "type": "KNOWS",
                    "label": "KNOWS",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [7],
                }
                for idx in range(200)
            ],
            "chunks": [
                {
                    "text": " ".join(f"Entity {idx} knows Entity {idx + 1}." for idx in range(200)),
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 7,
                    "start_pos": 0,
                    "end_pos": 8192,
                    "source": "bioasq/doc0",
                    "dataset": "bioasq",
                    "question_id": None,
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 201,
                "total_relationships": 200,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "bioasq",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "bioasq")

        assert ok is True
        assert kg["metadata"]["relationship_store_failures"] == 1
        assert kg["metadata"]["stored_relationships"] == 199
        assert kg["metadata"]["relationship_store_degraded"] is True
        assert kg["metadata"]["relationship_store_failure_ratio"] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# 2. wider relation recovery across chunk windows
# ---------------------------------------------------------------------------

class TestCrossWindowRelationRecovery:
    def test_extract_relationships_for_segment_windows_uses_three_chunk_window(self, monkeypatch):
        creator = _make_creator()
        monkeypatch.setattr(builder_mod.time, "sleep", lambda _: None)

        captured_texts = []

        def _fake_extract(text, entities, llm, model_name):
            captured_texts.append(text)
            if "chunk one" in text and "chunk two" in text and "chunk three" in text:
                return [{"source": "alpha", "target": "gamma", "type": "RELATED_TO"}]
            return []

        creator._extract_relationships_only = _fake_extract

        rels = creator._extract_relationships_for_segment_windows(
            ["chunk one mentions alpha", "chunk two bridges beta", "chunk three mentions gamma"],
            {
                0: [{"id": "alpha", "type": "Concept"}],
                1: [{"id": "beta", "type": "Concept"}],
                2: [{"id": "gamma", "type": "Concept"}],
            },
            {0: [0], 1: [1], 2: [2]},
            llm=object(),
            model_name="test-model",
            max_window_size=3,
            scope_label="Cross-chunk",
        )

        assert any(
            "chunk one" in text and "chunk two" in text and "chunk three" in text
            for text in captured_texts
        )
        assert rels[0]["provenance_positions"] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 2b. soft linking / fragmentation repair / summaries / claims
# ---------------------------------------------------------------------------

class TestGraphEnrichmentBuilder:
    @staticmethod
    def _kg() -> dict:
        return {
            "nodes": [
                {
                    "id": "u_tbk1_gene",
                    "label": "Gene",
                    "properties": {"name": "TBK1 kinase", "type": "Gene"},
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "id": "u_tbk1_protein",
                    "label": "Gene",
                    "properties": {"name": "TBK1 protein", "type": "Gene"},
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "id": "u_ifn1",
                    "label": "Pathway",
                    "properties": {"name": "interferon response", "type": "Pathway"},
                    "embedding": [0.0, 1.0, 0.0],
                },
                {
                    "id": "u_ifn2",
                    "label": "Pathway",
                    "properties": {"name": "interferon-response", "type": "Pathway"},
                    "embedding": [0.0, 1.0, 0.0],
                },
            ],
            "relationships": [
                {
                    "source": "u_tbk1_gene",
                    "target": "u_ifn1",
                    "type": "ACTIVATES",
                    "negated": False,
                    "properties": {"confidence": 0.91},
                    "provenance_positions": [0],
                },
                {
                    "source": "u_tbk1_protein",
                    "target": "u_ifn2",
                    "type": "REGULATES",
                    "negated": False,
                    "properties": {"confidence": 0.86},
                    "provenance_positions": [1],
                },
            ],
            "chunks": [
                {"text": "TBK1 kinase activates the interferon response.", "position": 0},
                {"text": "TBK1 protein regulates the interferon-response pathway.", "position": 1},
            ],
            "metadata": {"kg_name": "bio_test"},
        }

    def test_soft_entity_linking_merges_lexical_variants(self):
        creator = _make_creator()
        creator.enable_soft_entity_linking = True

        entities = [
            {
                "uuid": "u1",
                "id": "TBK1 kinase",
                "type": "Gene",
                "properties": {"name": "TBK1 kinase", "description": "Kinase mention"},
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "uuid": "u2",
                "id": "TBK1 protein",
                "type": "Gene",
                "properties": {"name": "TBK1 protein", "description": "Protein mention"},
                "embedding": [1.0, 0.0, 0.0],
            },
        ]
        entity_map = {
            "TBK1 kinase": entities[0],
            "TBK1 protein": entities[1],
        }

        merged_entities, merged_map = creator._apply_soft_entity_linking(entities, entity_map)

        assert len(merged_entities) == 1
        merged = merged_entities[0]
        assert merged["properties"]["soft_linked"] is True
        assert merged["properties"]["soft_link_cluster_size"] == 2
        assert sorted(merged["properties"]["all_names"]) == ["TBK1 kinase", "TBK1 protein"]
        assert merged_map["TBK1 kinase"]["uuid"] == merged["uuid"]
        assert merged_map["TBK1 protein"]["uuid"] == merged["uuid"]

    def test_claim_only_mode_does_not_emit_component_summaries(self):
        creator = _make_creator()
        creator.enable_claim_extraction = True

        enrichment = creator._build_graph_enrichment_records(self._kg())

        assert enrichment["component_summaries"] == []
        assert enrichment["graph_summary"] is None
        assert len(enrichment["claims"]) == 2

    def test_graph_enrichment_builds_summaries_claims_and_bridges(self):
        creator = _make_creator()
        creator.enable_graph_summaries = True
        creator.enable_claim_extraction = True
        creator.enable_fragmentation_repair = True
        creator.fragmentation_bridge_similarity_threshold = 0.8

        enrichment = creator._build_graph_enrichment_records(self._kg())

        assert enrichment["graph_summary"]["component_count"] == 2
        assert len(enrichment["component_summaries"]) == 2
        assert len(enrichment["claims"]) == 2
        assert len(enrichment["fragmentation_bridges"]) == 1
        bridge = enrichment["fragmentation_bridges"][0]
        assert {bridge["source_id"], bridge["target_id"]} == {"u_tbk1_gene", "u_tbk1_protein"}
        assert bridge["reason"] == "alias_overlap"

    def test_graph_enrichment_accepts_verbal_confidence_labels(self):
        creator = _make_creator()
        creator.enable_graph_summaries = True
        creator.enable_claim_extraction = True

        kg = self._kg()
        kg["relationships"][0]["properties"]["confidence"] = "demonstrated"
        kg["relationships"][1]["properties"]["confidence"] = "suggested"

        enrichment = creator._build_graph_enrichment_records(kg)

        assert len(enrichment["component_summaries"]) == 2
        claim_confidences = sorted(
            claim["confidence"] for claim in enrichment["claims"]
        )
        assert claim_confidences == [0.6, 0.95]

    def test_store_graph_enrichment_persists_all_record_types(self, stub_graph):
        creator = _make_creator()
        creator.enable_graph_summaries = True
        creator.enable_claim_extraction = True
        creator.enable_fragmentation_repair = True
        creator.fragmentation_bridge_similarity_threshold = 0.8

        enrichment = creator._build_graph_enrichment_records(self._kg())
        stats = creator._store_graph_enrichment(
            stub_graph,
            kg_name="bio_test",
            file_name="bio_test.json",
            enrichment=enrichment,
            chunks=self._kg()["chunks"],
        )

        assert stats == {
            "component_summaries": 2,
            "claims": 2,
            "fragmentation_bridges": 1,
        }
        queries = [query for query, _ in stub_graph.queries]
        assert any("HAS_GRAPH_SUMMARY" in query for query in queries)
        assert any("HAS_COMPONENT_SUMMARY" in query for query in queries)
        assert any("MERGE (c:Claim" in query for query in queries)
        assert any("SOFT_BRIDGE" in query for query in queries)


# ---------------------------------------------------------------------------
# 3. section-aware chunking / heuristic coref / self-consistency
# ---------------------------------------------------------------------------

class TestSectionAwareChunking:
    def test_section_chunking_preserves_boundaries_and_offsets(self, monkeypatch):
        creator = _make_creator()

        def _fake_chunk_text_fn(*, text, chunk_size, chunk_overlap, embedding_fn):
            return [{
                "text": text,
                "chunk_id": 0,
                "position": 0,
                "start_pos": 0,
                "end_pos": len(text),
                "embedding": None,
            }]

        monkeypatch.setattr(builder_mod, "_chunk_text_fn", _fake_chunk_text_fn)

        text = "History\nalpha finding\n\nMedications\nbeta dose"
        med_start = text.index("Medications")
        chunks = creator._chunk_text_with_section_boundaries(
            text,
            section_headers=[(0, "History"), (med_start, "Medications")],
        )

        assert len(chunks) == 2
        assert chunks[0]["start_pos"] == 0
        assert chunks[0]["end_pos"] == med_start
        assert chunks[1]["start_pos"] == med_start
        assert chunks[1]["text"].startswith("Medications")

    def test_cross_section_recovery_uses_section_segments(self):
        creator = _make_creator()

        text = "Introduction\nalpha finding\n\nResults\nbeta outcome"
        results_start = text.index("Results")
        chunks = [
            {
                "text": text[:results_start],
                "position": 0,
                "start_pos": 0,
                "end_pos": results_start,
                "dataset": "hotpotqa",
                "question_id": "q1",
                "passage_index": 0,
                "source": "hotpotqa/q1/p0",
                "source_title": "Introduction",
                "source_scope_key": "hotpotqa/q1/p0",
            },
            {
                "text": text[results_start:],
                "position": 1,
                "start_pos": results_start,
                "end_pos": len(text),
                "dataset": "hotpotqa",
                "question_id": "q1",
                "passage_index": 0,
                "source": "hotpotqa/q1/p0",
                "source_title": "Introduction",
                "source_scope_key": "hotpotqa/q1/p0",
            },
        ]
        captured = {}

        def _fake_windows(segment_texts, segment_entities, segment_positions, llm, model_name, *, max_window_size, scope_label, relationship_scope_metadata=None):
            captured["segment_texts"] = segment_texts
            captured["segment_positions"] = segment_positions
            captured["max_window_size"] = max_window_size
            captured["scope_label"] = scope_label
            captured["relationship_scope_metadata"] = relationship_scope_metadata
            return [{"source": "alpha", "target": "beta", "type": "RELATED_TO"}]

        creator._extract_relationships_for_segment_windows = _fake_windows

        rels = creator._recover_cross_section_relationships(
            text=text,
            chunks=chunks,
            entities_per_chunk={
                0: [{"id": "alpha", "type": "Concept"}],
                1: [{"id": "beta", "type": "Concept"}],
            },
            section_headers=[(0, "Introduction"), (results_start, "Results")],
            llm=object(),
            model_name="test-model",
            scope_label="Cross-section",
        )

        assert rels == [{"source": "alpha", "target": "beta", "type": "RELATED_TO"}]
        assert captured["segment_texts"][0].startswith("Introduction")
        assert captured["segment_texts"][1].startswith("Results")
        assert captured["segment_positions"] == {0: [0], 1: [1]}
        assert captured["max_window_size"] == 2
        assert captured["scope_label"] == "Cross-section"
        assert captured["relationship_scope_metadata"]["source_scope_key"] == "hotpotqa/q1/p0"

    def test_postprocess_skips_reflection_when_disabled(self):
        creator = _make_creator()
        creator.enable_self_reflection = False

        class FailIfCalledLLM:
            def generate(self, *args, **kwargs):
                raise AssertionError("self-reflection should be skipped when disabled")

        result = creator._postprocess_extraction_result(
            {
                "entities": [
                    {
                        "id": "Aspirin",
                        "type": "Drug",
                        "properties": {"name": "Aspirin", "description": "Drug: Aspirin"},
                    }
                ],
                "relationships": [],
            },
            chunk_text="Aspirin reduces fever.",
            llm=FailIfCalledLLM(),
            model_name="test-model",
        )

        assert len(result["entities"]) == 1
        assert result["entities"][0]["id"] == "Aspirin"


class TestHeuristicCoreference:
    def test_prepare_chunk_text_rewrites_typed_demonstrative(self):
        creator = _make_creator()

        rewritten = creator._prepare_chunk_text_for_extraction(
            "This disease progresses rapidly.",
            previous_entities=[{"id": "amyotrophic lateral sclerosis", "type": "Disease"}],
            previous_texts=["amyotrophic lateral sclerosis is discussed above"],
            llm=None,
        )

        assert rewritten.startswith("amyotrophic lateral sclerosis")


class TestSelfConsistencyVoting:
    def test_majority_vote_keeps_supported_entity(self):
        creator = _make_creator()
        creator.self_consistency_n = 3

        samples = [
            {
                "entities": [{"id": "metformin", "type": "Drug", "properties": {}}],
                "relationships": [],
            },
            {
                "entities": [{"id": "metformin", "type": "Drug", "properties": {}}],
                "relationships": [],
            },
            {
                "entities": [{"id": "aspirin", "type": "Drug", "properties": {}}],
                "relationships": [],
            },
        ]

        def _fake_extract(*args, **kwargs):
            return samples.pop(0)

        creator._extract_entities_and_relationships_with_llm = _fake_extract

        result = OntologyGuidedKGCreator._extract_entities_and_relationships_with_self_consistency(
            creator,
            "Metformin lowers glucose.",
            llm=object(),
            model_name="test-model",
            context_header=None,
            section_header=None,
        )

        assert [entity["id"] for entity in result["entities"]] == ["metformin"]


# ---------------------------------------------------------------------------
# 2. _verify_triple_confidence — scoring tiers
# ---------------------------------------------------------------------------

class TestVerifyTripleConfidence:
    creator = _make_creator()

    def _chunks(self, *texts):
        return [{"text": t} for t in texts]

    def test_same_sentence_returns_1_0(self):
        chunks = self._chunks("Metformin treats type 2 diabetes in patients.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 1.0

    def test_same_chunk_different_sentence_returns_0_7(self):
        # Two sentences in one chunk — not the same sentence
        text = "Metformin is a biguanide drug. Type 2 diabetes affects millions."
        chunks = self._chunks(text)
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.7

    def test_cross_chunk_both_found_returns_0_4(self):
        chunks = self._chunks("Metformin is a biguanide.", "Type 2 diabetes affects millions.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.4

    def test_only_source_found_returns_0_3(self):
        chunks = self._chunks("Metformin is a biguanide drug.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.3

    def test_neither_found_returns_0_1(self):
        chunks = self._chunks("Paracetamol reduces inflammation.")
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", chunks
        )
        assert score == 0.1

    def test_empty_chunks_returns_neutral(self):
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS", []
        )
        assert score == 0.5

    def test_score_in_unit_interval(self):
        chunks = self._chunks("Some irrelevant text about nothing.")
        score = self.creator._verify_triple_confidence("A", "B", "REL", chunks)
        assert 0.0 <= score <= 1.0


class TestEvidenceScopeCategorization:
    def test_cross_section_is_distinguished_from_cross_chunk(self):
        creator = _make_creator()

        cross_section = creator._relationship_evidence_scope(
            [
                {"text": "alpha here", "section_name": "Methods"},
                {"text": "beta there", "section_name": "Results"},
            ],
            0.4,
        )
        cross_chunk = creator._relationship_evidence_scope(
            [
                {"text": "alpha here", "section_name": "Methods"},
                {"text": "beta there", "section_name": "Methods"},
            ],
            0.4,
        )
        partial = creator._relationship_evidence_scope(
            [{"text": "alpha only", "section_name": "Methods"}],
            0.3,
        )

        assert cross_section == "cross_section"
        assert cross_chunk == "cross_chunk"
        assert partial == "partial_grounding"


# ---------------------------------------------------------------------------
# 2b. exact-span anchor grounding + restoration verification
# ---------------------------------------------------------------------------

class TestAnchorGroundingAndRestoration:
    def test_ground_chunk_extraction_adds_exact_spans_and_full_restoration(self):
        creator = _make_creator()
        chunk = {
            "text": "Metformin inhibits AMPK in diabetic patients.",
            "position": 5,
            "start_pos": 100,
            "end_pos": 145,
        }
        chunk_kg = {
            "entities": [
                {"id": "Metformin", "type": "Drug", "properties": {"name": "Metformin"}},
                {"id": "AMPK", "type": "Protein", "properties": {"name": "AMPK"}},
            ],
            "relationships": [
                {
                    "source": "Metformin",
                    "target": "AMPK",
                    "type": "INHIBITS",
                    "properties": {
                        "anchor_text": "inhibits",
                        "description": "inhibits",
                    },
                }
            ],
        }

        grounded = creator._ground_chunk_extraction(chunk_kg, chunk)

        entity_spans = grounded["entities"][0]["properties"]["anchor_spans"]
        assert entity_spans[0]["text"] == "Metformin"
        assert entity_spans[0]["start"] == 100
        assert entity_spans[0]["chunk_position"] == 5

        rel_props = grounded["relationships"][0]["properties"]
        assert rel_props["restoration_status"] == "full"
        assert rel_props["restoration_verified"] is True
        assert rel_props["relation_anchor_spans"][0]["text"].lower() == "inhibits"

    def test_restoration_verifier_marks_partial_when_relation_phrase_is_missing(self):
        creator = _make_creator()
        rel = {
            "type": "INHIBITS",
            "properties": {
                "source_name": "Metformin",
                "target_name": "AMPK",
            },
        }
        chunks = [
            {
                "text": "Metformin and AMPK were both discussed in the study.",
                "position": 0,
                "start_pos": 0,
                "end_pos": 52,
            }
        ]

        restoration = creator._verify_relationship_restoration(
            rel,
            chunks,
            source_name="Metformin",
            target_name="AMPK",
            relation_type="INHIBITS",
        )

        assert restoration["status"] == "partial"
        assert set(restoration["grounded_components"]) == {"source", "target"}

    def test_anchor_constrained_extraction_uses_discovered_anchors(self):
        creator = _make_creator()

        class SequentialLLM:
            def __init__(self, responses):
                self._responses = list(responses)

            def generate(self, prompt, system_message, model_name):
                assert self._responses, "unexpected extra LLM call"
                return self._responses.pop(0)

        llm = SequentialLLM(
            [
                json.dumps(
                    {
                        "entity_anchors": [
                            {"text": "Metformin", "type": "Drug"},
                            {"text": "AMPK", "type": "Protein"},
                        ],
                        "relation_anchors": [
                            {"text": "inhibits", "type_hint": "INHIBITS"}
                        ],
                        "attribute_anchors": [],
                    }
                ),
                json.dumps(
                    {
                        "relationships": [
                            {
                                "source": "Metformin",
                                "target": "AMPK",
                                "type": "INHIBITS",
                                "negated": False,
                                "properties": {
                                    "anchor_text": "inhibits",
                                    "description": "inhibits",
                                    "condition": None,
                                    "quantitative": None,
                                    "confidence": "demonstrated",
                                },
                            }
                        ],
                        "entities": [],
                    }
                ),
                json.dumps({"new_entities": []}),
            ]
        )

        result = creator._extract_entities_and_relationships_with_anchor_constraints(
            "Metformin inhibits AMPK.",
            llm,
            model_name="test-model",
            context_preamble="",
            ontology_section="",
            has_ontology=False,
        )

        assert result is not None
        assert {entity["id"] for entity in result["entities"]} == {"Metformin", "AMPK"}
        rel = result["relationships"][0]
        assert rel["source"] == "Metformin"
        assert rel["target"] == "AMPK"
        assert rel["properties"]["anchor_text"] == "inhibits"


# ---------------------------------------------------------------------------
# 3. _verify_triple_confidence — alias coverage
# ---------------------------------------------------------------------------

class TestVerifyTripleConfidenceAliases:
    creator = _make_creator()

    def test_alias_used_for_same_sentence_match(self):
        chunks = [{"text": "MET lowers blood glucose in diabetic patients."}]
        score = self.creator._verify_triple_confidence(
            "Metformin", "blood glucose", "LOWERS",
            chunks,
            source_aliases=["MET", "metformin hydrochloride"],
        )
        assert score == 1.0

    def test_canonical_name_used_when_alias_absent(self):
        chunks = [{"text": "Metformin lowers blood glucose."}]
        score = self.creator._verify_triple_confidence(
            "Metformin", "blood glucose", "LOWERS",
            chunks,
            source_aliases=["MET"],
        )
        assert score == 1.0

    def test_target_alias_triggers_same_chunk(self):
        # Source in chunk 1, target alias in chunk 2 → cross-chunk (0.4)
        chunks = [
            {"text": "Metformin is prescribed widely."},
            {"text": "T2DM affects insulin sensitivity."},
        ]
        score = self.creator._verify_triple_confidence(
            "Metformin", "type 2 diabetes", "TREATS",
            chunks,
            target_aliases=["T2DM"],
        )
        assert score == 0.4


class TestEntityAppearsInText:
    creator = _make_creator()

    def test_alias_match_works_with_cross_chunk_candidate_filter(self):
        entity = {
            "id": "type_2_diabetes",
            "properties": {
                "name": "type_2_diabetes",
                "all_names": ["T2DM", "type 2 diabetes"],
            },
        }

        assert self.creator._entity_appears_in_text(
            entity,
            "Patients with T2DM often require monitoring.",
        )


# ---------------------------------------------------------------------------
# 4. _harmonize_entities — deduplication contract
# ---------------------------------------------------------------------------

class TestHarmonizeEntities:
    creator = _make_creator()

    def _ent(self, name: str, etype: str = "Disease") -> dict:
        # _harmonize_entities reads entity['type'] (not 'label') — match production shape.
        return {
            "id": name,
            "type": etype,
            "properties": {"name": name, "description": ""},
        }

    def _bundle_ent(
        self,
        name: str,
        *,
        etype: str = "Concept",
        dataset: str = "hotpotqa",
        question_id: str = "q1",
        passage_index: int = 0,
        source_title: str = "",
    ) -> dict:
        entity = self._ent(name, etype)
        entity["properties"].update(
            {
                "dataset": dataset,
                "question_id": question_id,
                "passage_index": passage_index,
                "source_title": source_title,
                "source_scope_key": f"{dataset}/{question_id}/p{passage_index}",
            }
        )
        return entity

    def test_identical_names_deduplicated(self):
        entities = [self._ent("Aspirin"), self._ent("Aspirin")]
        result = self.creator._harmonize_entities(entities)
        ids = [e["id"] for e in result]
        assert ids.count(ids[0]) == 1

    def test_case_insensitive_dedup(self):
        entities = [self._ent("Aspirin"), self._ent("aspirin"), self._ent("ASPIRIN")]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 1

    def test_different_names_kept_separate(self):
        entities = [self._ent("Aspirin"), self._ent("Metformin")]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2

    def test_more_specific_type_wins(self):
        # "Disease" beats generic "Entity"
        entities = [self._ent("Flu", "Entity"), self._ent("Flu", "Disease")]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 1
        assert result[0]["type"] in ("Disease", "disease")

    def test_same_surface_different_specific_types_kept_separate(self):
        # "depression" as a Disease and as a GeologicalFeature must not collapse.
        entities = [
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2
        types = {e["type"] for e in result}
        assert "Disease" in types
        assert "GeologicalFeature" in types

    def test_generic_type_drift_still_merges(self):
        # Same entity tagged Disease in one chunk, generic Concept in another
        # (LLM drift) — must collapse into one Disease node, not split.
        entities = [
            self._ent("Prostate Cancer", "Disease"),
            self._ent("Prostate Cancer", "Concept"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 1
        assert result[0]["type"] in ("Disease", "disease")

    def test_generic_assigned_to_dominant_bucket_on_split(self):
        # Three occurrences: Disease x2, GeologicalFeature x1, Concept x1 (generic).
        # Generics should go to Disease (largest bucket); result: Disease node + GeologicalFeature node.
        entities = [
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
            self._ent("depression", "Concept"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2
        types = {e["type"] for e in result}
        assert "Disease" in types
        assert "GeologicalFeature" in types

    def test_split_entities_have_distinct_uuids(self):
        # End-to-end UUID check: same surface form, different specific types must
        # survive ID generation as two nodes with different UUIDs — not collapse at write.
        entities = [
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
        ]
        result = self.creator._harmonize_entities(entities)
        assert len(result) == 2
        uuids = [e["uuid"] for e in result]
        assert uuids[0] != uuids[1], (
            "Same surface form with different specific types must get distinct UUIDs; "
            "if equal, _generate_entity_id is not including type in the seed"
        )

    def test_merged_entity_stable_uuid(self):
        # LLM type-drift case: Disease + Concept for the same surface form must merge
        # into one node AND produce the same UUID regardless of input order.
        fwd = self.creator._harmonize_entities([
            self._ent("Prostate Cancer", "Disease"),
            self._ent("Prostate Cancer", "Concept"),
        ])
        rev = self.creator._harmonize_entities([
            self._ent("Prostate Cancer", "Concept"),
            self._ent("Prostate Cancer", "Disease"),
        ])
        assert len(fwd) == 1
        assert len(rev) == 1
        assert fwd[0]["uuid"] == rev[0]["uuid"], (
            "Merged entity UUID must be stable regardless of input order"
        )

    def test_relationship_resolves_to_dominant_type(self):
        # Disease x2, GeologicalFeature x1 → Disease is dominant.
        # The relationship target "depression" must resolve to the Disease UUID,
        # not GeologicalFeature, regardless of extraction order.
        entities = [
            self._ent("fluoxetine", "Drug"),
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),       # dominant: 2 occurrences
            self._ent("depression", "GeologicalFeature"),
        ]
        relationships = [
            {"source": "fluoxetine", "target": "depression",
             "type": "TREATS", "negated": False, "properties": {}},
        ]
        result, entity_map = self.creator._harmonize_entities(entities, return_id_map=True)
        harmonized_rels = self.creator._harmonize_relationships(relationships, entity_map)

        assert len(harmonized_rels) == 1, "Relationship must not be dropped after entity split"
        disease_uuid = next(e["uuid"] for e in result if e.get("type") == "Disease")
        geo_uuid = next(e["uuid"] for e in result if e.get("type") == "GeologicalFeature")
        assert harmonized_rels[0]["target"] == disease_uuid, (
            f"Expected target={disease_uuid} (Disease, dominant), got {harmonized_rels[0]['target']}; "
            f"GeologicalFeature uuid={geo_uuid}"
        )

    def test_relationship_resolution_stable_across_order(self):
        # Same as above but with extraction order reversed — resolution must be identical.
        entities_fwd = [
            self._ent("fluoxetine", "Drug"),
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),
            self._ent("depression", "GeologicalFeature"),
        ]
        entities_rev = [
            self._ent("fluoxetine", "Drug"),
            self._ent("depression", "GeologicalFeature"),
            self._ent("depression", "Disease"),
            self._ent("depression", "Disease"),
        ]
        relationships = [
            {"source": "fluoxetine", "target": "depression",
             "type": "TREATS", "negated": False, "properties": {}},
        ]

        _, map_fwd = self.creator._harmonize_entities(entities_fwd, return_id_map=True)
        _, map_rev = self.creator._harmonize_entities(entities_rev, return_id_map=True)
        rels_fwd = self.creator._harmonize_relationships(relationships, map_fwd)
        rels_rev = self.creator._harmonize_relationships(relationships, map_rev)

        assert len(rels_fwd) == 1 and len(rels_rev) == 1
        assert rels_fwd[0]["target"] == rels_rev[0]["target"], (
            "Relationship target UUID must be the same regardless of entity extraction order"
        )

    def test_resolve_relationship_endpoint_handles_short_form(self):
        entity = self._ent("TBK1 kinase", "Protein")
        entity["properties"]["all_names"] = ["TBK1", "TANK-binding kinase 1"]

        resolved = self.creator._resolve_relationship_endpoint("TBK1", [entity])

        assert resolved == "TBK1 kinase"

    def test_relationship_dedup_keeps_qualifier_variants(self):
        entities = [
            self._ent("Aspirin", "Drug"),
            self._ent("Fever", "Disease"),
        ]
        relationships = [
            {
                "source": "Aspirin",
                "target": "Fever",
                "type": "TREATS",
                "negated": False,
                "properties": {"condition": "in adults"},
            },
            {
                "source": "Aspirin",
                "target": "Fever",
                "type": "TREATS",
                "negated": False,
                "properties": {"condition": "in children"},
            },
        ]

        _, entity_map = self.creator._harmonize_entities(entities, return_id_map=True)
        rels = self.creator._harmonize_relationships(relationships, entity_map)

        assert len(rels) == 2, "Condition-specific variants must not collapse before storage"

    def test_bundle_title_entities_split_by_passage_scope(self):
        entities = [
            self._bundle_ent(
                "American Beauty",
                etype="Work",
                passage_index=0,
                source_title="American Beauty (1999 film)",
            ),
            self._bundle_ent(
                "American Beauty",
                etype="Work",
                passage_index=1,
                source_title="American Beauty (album)",
            ),
        ]

        result = self.creator._harmonize_entities(entities)

        assert len(result) == 2
        assert result[0]["uuid"] != result[1]["uuid"]
        assert all(e["properties"].get("title_entity_scoped") is True for e in result)
        assert {
            e["properties"].get("title_scope_key") for e in result
        } == {"hotpotqa/q1/p0", "hotpotqa/q1/p1"}

    def test_relationship_prefers_local_bundle_title_scope(self):
        entities = [
            self._bundle_ent(
                "American Beauty",
                etype="Work",
                passage_index=0,
                source_title="American Beauty (1999 film)",
            ),
            self._bundle_ent(
                "American Beauty",
                etype="Work",
                passage_index=1,
                source_title="American Beauty (album)",
            ),
            self._bundle_ent(
                "Kevin Spacey",
                etype="Person",
                passage_index=0,
                source_title="American Beauty (1999 film)",
            ),
            self._bundle_ent(
                "Billie Dove",
                etype="Person",
                passage_index=1,
                source_title="American Beauty (album)",
            ),
        ]
        relationships = [
            {
                "source": "American Beauty",
                "target": "Kevin Spacey",
                "type": "HAS_ACTOR",
                "negated": False,
                "properties": {"source_scope_key": "hotpotqa/q1/p0"},
            },
            {
                "source": "American Beauty",
                "target": "Billie Dove",
                "type": "HAS_ACTOR",
                "negated": False,
                "properties": {"source_scope_key": "hotpotqa/q1/p1"},
            },
        ]

        result, entity_map = self.creator._harmonize_entities(entities, return_id_map=True)
        rels = self.creator._harmonize_relationships(relationships, entity_map)

        assert len(rels) == 2
        film_uuid = next(
            e["uuid"]
            for e in result
            if e["properties"].get("title_scope_key") == "hotpotqa/q1/p0"
        )
        album_uuid = next(
            e["uuid"]
            for e in result
            if e["properties"].get("title_scope_key") == "hotpotqa/q1/p1"
        )
        rel_by_target = {
            rel["properties"]["target_name"]: rel["source"]
            for rel in rels
        }
        assert rel_by_target["Kevin Spacey"] == film_uuid
        assert rel_by_target["Billie Dove"] == album_uuid


# ---------------------------------------------------------------------------
# 5. relation MERGE key contract (negated always; qualifiers only when present)
# ---------------------------------------------------------------------------

class TestRelationMergeKey:
    """
    The Cypher MERGE for relationships must always include negated and must
    only include optional qualifiers when present. This avoids Neo4j rejecting
    the query with null-valued MERGE properties.
    """

    def setup_method(self):
        self.creator = _make_creator()

    def test_merge_key_includes_negated(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "negated: $negated" in q, "MERGE key must include 'negated'"

    def test_merge_key_omits_absent_condition(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "condition: $condition" not in q, (
            "MERGE key must omit absent 'condition' to avoid null MERGE errors"
        )

    def test_merge_key_includes_present_condition(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=True,
            include_quantitative=False,
        )
        assert "condition: $condition" in q, (
            "MERGE key must include 'condition' when a qualifier is present"
        )

    def test_merge_key_omits_absent_quantitative(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "quantitative: $quantitative" not in q, (
            "MERGE key must omit absent 'quantitative' to avoid null MERGE errors"
        )

    def test_merge_key_includes_present_quantitative(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=True,
        )
        assert "quantitative: $quantitative" in q, (
            "MERGE key must include 'quantitative' when a qualifier is present"
        )

    def test_merge_query_persists_edge_provenance(self):
        q = self.creator._build_relationship_merge_query(
            "TREATS",
            include_condition=False,
            include_quantitative=False,
        )
        assert "r.provenancePositions" in q
        assert "r.questionIds" in q
        assert "r.passageKeys" in q


class TestRelationshipLocalProvenance:
    def test_relationship_local_provenance_resolves_question_and_passage(self):
        creator = _make_creator()
        rel = {"provenance_positions": [7, 8]}
        chunks = [
            {"position": 7, "question_id": "q1", "passage_index": 0},
            {"position": 8, "question_id": "q1", "passage_index": 0},
            {"position": 30, "question_id": "q2", "passage_index": 1},
        ]

        provenance = creator._relationship_local_provenance(rel, chunks)

        assert provenance["provenance_positions"] == [7, 8]
        assert provenance["question_ids"] == ["q1"]
        assert provenance["passage_keys"] == ["q1::p0"]


# ---------------------------------------------------------------------------
# 6. ontology enforcement / node-specificity / synonym guards
# ---------------------------------------------------------------------------

class TestStrictOntologyLoading:
    def test_constructor_raises_when_requested_ontology_fails_to_load(self, monkeypatch, tmp_path):
        ontology_path = tmp_path / "broken.json"
        ontology_path.write_text("{}", encoding="utf-8")

        emb = type("Emb", (), {"embed_query": lambda self, t: [0.0] * 3})()
        monkeypatch.setattr(builder_mod, "load_embedding_model", lambda model: (emb, 3))

        def _boom(self, path):
            raise ValueError("bad ontology")

        monkeypatch.setattr(builder_mod.OntologyGuidedKGCreator, "_load_ontology", _boom)

        with pytest.raises(ValueError):
            builder_mod.OntologyGuidedKGCreator(
                ontology_path=str(ontology_path),
                strict_ontology=True,
            )

    def test_constructor_can_fall_back_when_strict_ontology_disabled(self, monkeypatch, tmp_path):
        ontology_path = tmp_path / "broken.json"
        ontology_path.write_text("{}", encoding="utf-8")

        emb = type("Emb", (), {"embed_query": lambda self, t: [0.0] * 3})()
        monkeypatch.setattr(builder_mod, "load_embedding_model", lambda model: (emb, 3))

        def _boom(self, path):
            raise ValueError("bad ontology")

        monkeypatch.setattr(builder_mod.OntologyGuidedKGCreator, "_load_ontology", _boom)

        creator = builder_mod.OntologyGuidedKGCreator(
            ontology_path=str(ontology_path),
            strict_ontology=False,
        )

        assert creator.ontology_classes == []
        assert creator.ontology_relationships == []


class TestOntologySchemaEnforcement:
    def _schema_creator(self) -> OntologyGuidedKGCreator:
        creator = _make_creator()
        creator._ontology_schema = OntologySchema(
            entity_types=[
                EntityType(id="Drug", label="Drug"),
                EntityType(id="Disease", label="Disease"),
            ],
            relationship_types=[
                RelationshipType(id="TREATS", label="TREATS", domain="Drug", range="Disease"),
            ],
        )
        creator.ontology_classes = [
            {"id": "Drug", "label": "Drug"},
            {"id": "Disease", "label": "Disease"},
        ]
        creator.ontology_relationships = [
            {"id": "TREATS", "label": "TREATS", "domain": "Drug", "range": "Disease"},
        ]
        return creator

    def test_entity_types_are_coerced_back_onto_schema(self):
        creator = self._schema_creator()

        assert creator._coerce_entity_type_with_ontology("Illness", "lung cancer") == "Disease"
        assert creator._coerce_entity_type_with_ontology("Drug", "metformin") == "Drug"
        assert creator._coerce_entity_type_with_ontology("AlienType", "nonsense term") is None

    def test_schema_enforcement_records_drop_counts(self):
        creator = self._schema_creator()

        entities, relationships = creator._coerce_harmonized_entities_to_schema(
            [
                {"id": "metformin", "type": "Drug", "uuid": "u1", "properties": {}},
                {"id": "nonsense term", "type": "AlienType", "uuid": "u2", "properties": {}},
            ],
            [{"source": "u1", "target": "u2", "type": "TREATS"}],
        )

        assert len(entities) == 1
        assert relationships == []
        assert creator._last_schema_enforcement_stats["dropped_entities"] == 1
        assert creator._last_schema_enforcement_stats["dropped_relationships"] == 1

    def test_off_schema_relationship_types_are_dropped_without_generic_fallback(self):
        creator = self._schema_creator()

        assert creator._canonicalize_relationship_type(
            "CAUSES",
            source_type="Drug",
            target_type="Disease",
        ) is None

    def test_storage_skips_schema_mismatch_relationships(self, stub_graph):
        creator = self._schema_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        kg = {
            "nodes": [
                {
                    "id": "kg_drug",
                    "label": "Drug",
                    "properties": {"name": "Metformin", "type": "Drug", "original_id": "Metformin"},
                    "embedding": None,
                },
                {
                    "id": "kg_disease",
                    "label": "Disease",
                    "properties": {"name": "Diabetes", "type": "Disease", "original_id": "Diabetes"},
                    "embedding": None,
                },
            ],
            "relationships": [
                {
                    "id": "kg_rel",
                    "from": "kg_drug",
                    "to": "kg_disease",
                    "source": "kg_drug",
                    "target": "kg_disease",
                    "type": "CAUSES",
                    "label": "CAUSES",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [0],
                }
            ],
            "chunks": [
                {
                    "text": "Metformin treats diabetes.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 0,
                    "start_pos": 0,
                    "end_pos": 26,
                    "source": "schema/q1/p0",
                    "dataset": "schema",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 2,
                "total_relationships": 1,
                "ontology_classes": 2,
                "ontology_relationships": 1,
                "kg_name": "schema",
            },
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "schema")

        assert ok is True
        assert kg["metadata"]["relationships_skipped_schema_mismatch"] == 1
        assert kg["metadata"]["stored_relationships"] == 0

    def test_prompt_builder_surfaces_relevant_late_relationship_type(self):
        creator = self._schema_creator()
        creator._ontology_schema = OntologySchema(
            entity_types=[
                EntityType(id="Drug", label="Drug"),
                EntityType(id="Disease", label="Disease"),
            ],
            relationship_types=[
                *[
                    RelationshipType(id=f"REL_{i}", label=f"REL_{i}")
                    for i in range(25)
                ],
                RelationshipType(
                    id="INHIBITS_SPECIAL_PATHWAY",
                    label="inhibits special pathway",
                    domain="Drug",
                    range="Disease",
                ),
            ],
        )

        section = creator._build_ontology_prompt_section(
            "This paper shows metformin inhibits special pathway signaling in diabetes.",
            max_rel_types=25,
        )

        assert "INHIBITS_SPECIAL_PATHWAY" in section

    def test_semantic_relationship_matching_maps_synonym_label(self):
        creator = self._schema_creator()
        creator.embedding_function = type(
            "Emb",
            (),
            {"embed_query": lambda self, text: [1.0, 0.0] if "suppress" in text.lower() else [0.0, 1.0]},
        )()
        creator._ontology_relationship_embeddings = [
            ("INHIBITS", "inhibits", [1.0, 0.0]),
            ("TREATS", "treats", [0.0, 1.0]),
        ]
        creator._ontology_schema.relationship_types.append(
            RelationshipType(id="INHIBITS", label="inhibits", domain="Drug", range="Disease")
        )

        assert creator._canonicalize_relationship_type(
            "suppresses",
            source_type="Drug",
            target_type="Disease",
        ) == "INHIBITS"


class TestConfidenceThresholdAndReverification:
    def _kg(self) -> dict:
        return {
            "nodes": [
                {
                    "id": "kg_drug",
                    "label": "Drug",
                    "properties": {"name": "Metformin", "type": "Drug", "original_id": "Metformin"},
                    "embedding": None,
                },
                {
                    "id": "kg_disease",
                    "label": "Disease",
                    "properties": {"name": "Diabetes", "type": "Disease", "original_id": "Diabetes"},
                    "embedding": None,
                },
            ],
            "relationships": [
                {
                    "id": "kg_rel",
                    "from": "kg_drug",
                    "to": "kg_disease",
                    "source": "kg_drug",
                    "target": "kg_disease",
                    "type": "TREATS",
                    "label": "TREATS",
                    "negated": False,
                    "properties": {},
                    "provenance_positions": [0],
                }
            ],
            "chunks": [
                {
                    "text": "Metformin treats diabetes.",
                    "chunk_id": 0,
                    "chunk_local_index": 0,
                    "position": 0,
                    "start_pos": 0,
                    "end_pos": 26,
                    "source": "schema/q1/p0",
                    "dataset": "schema",
                    "question_id": "q1",
                    "passage_index": 0,
                    "embedding": None,
                }
            ],
            "metadata": {
                "total_chunks": 1,
                "total_entities": 2,
                "total_relationships": 1,
                "ontology_classes": 0,
                "ontology_relationships": 0,
                "kg_name": "schema",
            },
        }

    def test_storage_uses_configurable_confidence_threshold(self, stub_graph):
        creator = _make_creator()
        creator.min_triple_confidence = 0.5
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 0.4

        kg = self._kg()
        ok = creator.store_knowledge_graph_with_embeddings(kg, "schema")

        assert ok is True
        assert kg["metadata"]["stored_relationships"] == 0
        assert kg["metadata"]["relationships_skipped_low_confidence"] == 1

    def test_low_confidence_reverification_can_keep_relation(self, stub_graph):
        creator = _make_creator()
        creator.min_triple_confidence = 0.5
        creator.enable_low_confidence_triple_reverification = True
        creator.low_confidence_reverify_threshold = 0.4
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 0.4

        class FakeLLM:
            def generate(self, prompt, system_message, model_name):
                return "YES"

        kg = self._kg()
        ok = creator.store_knowledge_graph_with_embeddings(
            kg,
            "schema",
            llm=FakeLLM(),
            model_name="test-model",
        )

        assert ok is True
        assert kg["metadata"]["stored_relationships"] == 1
        assert kg["metadata"]["relationships_reverified_kept"] == 1
        _, rel_params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (source)-[r:" in query
        )
        assert rel_params["properties"]["llm_verified"] is True

    def test_storage_writes_evidence_scope(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 1.0

        kg = self._kg()
        ok = creator.store_knowledge_graph_with_embeddings(kg, "schema")

        assert ok is True
        _, rel_params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (source)-[r:" in query
        )
        assert rel_params["properties"]["evidence_scope"] == "sentence"

    def test_storage_preserves_anchor_metadata_on_nodes_and_relationships(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None
        creator._verify_triple_confidence = lambda *args, **kwargs: 1.0

        kg = self._kg()
        kg["nodes"][0]["properties"]["anchor_spans"] = [
            {"text": "Metformin", "start": 0, "end": 10, "chunk_position": 0}
        ]
        kg["relationships"][0]["properties"] = {
            "anchor_grounding": {
                "source": [{"text": "Metformin", "start": 0, "end": 10, "chunk_position": 0}],
                "target": [{"text": "diabetes", "start": 18, "end": 26, "chunk_position": 0}],
                "relation": [{"text": "treats", "start": 11, "end": 17, "chunk_position": 0}],
            },
            "restoration_status": "full",
        }

        ok = creator.store_knowledge_graph_with_embeddings(kg, "schema")

        assert ok is True
        _, node_params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (n:__Entity__ {id: $id})" in query
        )
        assert "anchor_spans" in node_params["extra_properties"]
        _, rel_params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (source)-[r:" in query
        )
        assert rel_params["properties"]["restoration_status"] == "full"
        assert "anchor_grounding" in rel_params["properties"]


class TestOptionalUMLSLinking:
    def test_entities_with_same_cui_merge_when_enabled(self):
        creator = _make_creator()
        creator.enable_umls_linking = True
        creator._ensure_umls_linker = lambda: object()

        link_map = {
            "myocardial infarction": {"cui": "C0027051", "score": 0.95, "name": "Myocardial Infarction"},
            "heart attack": {"cui": "C0027051", "score": 0.94, "name": "Myocardial Infarction"},
        }
        creator._link_entity_to_umls = lambda name: link_map.get(name)

        representative_a = {
            "id": "myocardial infarction",
            "type": "Disease",
            "uuid": "u1",
            "properties": {"description": "MI", "all_names": ["myocardial infarction"]},
            "embedding": None,
        }
        representative_b = {
            "id": "heart attack",
            "type": "Disease",
            "uuid": "u2",
            "properties": {"description": "heart attack", "all_names": ["heart attack"]},
            "embedding": None,
        }

        entities, entity_map = creator._apply_optional_umls_linking(
            [representative_a, representative_b],
            {
                "myocardial infarction": representative_a,
                "heart attack": representative_b,
            },
        )

        assert len(entities) == 1
        assert entities[0]["properties"]["umls_cui"] == "C0027051"
        assert entity_map["myocardial infarction"] == entity_map["heart attack"]


class TestContradictionDetection:
    def test_marks_opposite_polarity_edges_as_contradictions(self):
        creator = _make_creator()

        rels = creator._mark_relationship_contradictions(
            [
                {
                    "source": "u1",
                    "target": "u2",
                    "type": "INHIBITS",
                    "negated": False,
                    "properties": {},
                },
                {
                    "source": "u1",
                    "target": "u2",
                    "type": "INHIBITS",
                    "negated": True,
                    "properties": {},
                },
            ]
        )

        assert all(rel["properties"]["contradiction_detected"] is True for rel in rels)
        assert creator._last_relationship_contradiction_stats["contradiction_groups"] == 1
        assert creator._last_relationship_contradiction_stats["contradiction_edges"] == 2

    def test_harmonization_merges_anchor_grounding_for_duplicate_relations(self):
        creator = _make_creator()
        source = {"id": "Metformin", "uuid": "u1", "type": "Drug", "properties": {}}
        target = {"id": "AMPK", "uuid": "u2", "type": "Protein", "properties": {}}

        rels = creator._harmonize_relationships(
            [
                {
                    "source": "Metformin",
                    "target": "AMPK",
                    "type": "INHIBITS",
                    "properties": {
                        "anchor_grounding": {
                            "relation": [{"text": "inhibits", "start": 10, "end": 18, "chunk_position": 0}]
                        }
                    },
                    "provenance_positions": [0],
                },
                {
                    "source": "Metformin",
                    "target": "AMPK",
                    "type": "INHIBITS",
                    "properties": {
                        "anchor_grounding": {
                            "relation": [{"text": "suppresses", "start": 40, "end": 50, "chunk_position": 1}]
                        }
                    },
                    "provenance_positions": [1],
                },
            ],
            {"Metformin": source, "AMPK": target},
        )

        assert len(rels) == 1
        assert rels[0]["provenance_positions"] == [0, 1]
        relation_spans = rels[0]["properties"]["anchor_grounding"]["relation"]
        assert len(relation_spans) == 2


class TestNodeSpecificityScoping:
    def test_chunk_storage_writes_kg_name(self, stub_graph):
        creator = _make_creator()
        creator._create_neo4j_connection = lambda: stub_graph
        creator._create_vector_indexes = lambda graph: None

        ok = creator.store_knowledge_graph_with_embeddings(TestPassageProvenanceStorage()._kg(), "musique")

        assert ok is True
        _, params = next(
            (query, params)
            for query, params in stub_graph.queries
            if "MERGE (c:Chunk {id: $chunk_id})" in query
        )
        assert params["kg_name"] == "musique"

    def test_node_specificity_query_scopes_via_document_kg_name(self, stub_graph):
        creator = _make_creator()

        creator._compute_node_specificity_weights(stub_graph, "musique")

        query, params = stub_graph.last_query()
        assert "MATCH (c:Chunk)-[:PART_OF]->(d:Document)" in query
        assert "d.kgName = $kg_name" in query
        assert params["kg_name"] == "musique"


class TestSynonymGuard:
    def test_names_pass_synonym_guard_allows_surface_variants(self):
        creator = _make_creator()

        assert creator._names_pass_synonym_guard("TBK1", "TBK1 kinase")
        assert creator._names_pass_synonym_guard("tumor necrosis factor", "TNF")

    def test_names_pass_synonym_guard_blocks_embedding_only_near_misses(self):
        creator = _make_creator()

        assert not creator._names_pass_synonym_guard("alpha", "beta")
        assert not creator._names_pass_synonym_guard("Paris", "France")


# ---------------------------------------------------------------------------
# 7. synonym merge type guard reads e.type / e.ontology_class
# ---------------------------------------------------------------------------

class TestSynonymMergeTypeProperty:
    """
    The fetch query in merge_synonym_entities must read e.type / e.ontology_class,
    not the never-written e.entity_type.
    """

    def _get_fetch_query(self) -> str:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ontographrag", "kg", "builders", "ontology_guided_kg_creator.py"
        )
        with open(path) as f:
            src = f.read()
        # Find the fetch_q assignment inside merge_synonym_entities
        m = re.search(
            r"fetch_q\s*=\s*f?\"\"\"(.*?)\"\"\"",
            src, re.DOTALL
        )
        return m.group(1) if m else ""

    def test_fetch_reads_type_not_entity_type(self):
        q = self._get_fetch_query()
        assert "e.entity_type" not in q, (
            "fetch query must not use e.entity_type (never written); "
            "use e.type or e.ontology_class"
        )

    def test_fetch_reads_type_or_ontology_class(self):
        q = self._get_fetch_query()
        assert "e.type" in q or "e.ontology_class" in q, (
            "fetch query must read e.type or e.ontology_class"
        )


class TestSynonymMergeExecution:
    def test_synonym_merge_uses_human_names_and_element_ids(self):
        creator = _make_creator()

        class FakeGraph:
            def __init__(self):
                self.queries = []

            def query(self, query, params=None):
                self.queries.append((query, params or {}))
                if "RETURN elementId(e) AS eid" in query:
                    return [
                        {
                            "eid": "entity-el-1",
                            "name": "TBK1",
                            "emb": [1.0, 0.0],
                            "degree": 3,
                            "etype": "Protein",
                        },
                        {
                            "eid": "entity-el-2",
                            "name": "TBK1 kinase",
                            "emb": [0.999, 0.001],
                            "degree": 1,
                            "etype": "Protein",
                        },
                    ]
                if "apoc.refactor.mergeNodes" in query:
                    return [{"merged_id": "merged"}]
                return []

        graph = FakeGraph()
        merged = creator.merge_synonym_entities(graph, similarity_threshold=0.8, kg_name="bioasq")

        assert merged == 1
        fetch_query, fetch_params = graph.queries[0]
        assert "coalesce(e.name, e.original_id, e.id) AS name" in fetch_query
        assert fetch_params["kg_name"] == "bioasq"

        merge_query, merge_params = next(
            (query, params)
            for query, params in graph.queries
            if "apoc.refactor.mergeNodes" in query
        )
        assert "elementId(can) = $can_eid" in merge_query
        assert "elementId(dup) = $dup_eid" in merge_query
        assert merge_params == {"can_eid": "entity-el-1", "dup_eid": "entity-el-2"}


# ---------------------------------------------------------------------------
# evaluate_knowledge_graph
# ---------------------------------------------------------------------------

class TestEvaluateKnowledgeGraph:
    """Tests for OntologyGuidedKGCreator.evaluate_knowledge_graph."""

    def _minimal_kg(self) -> dict:
        return {
            "nodes": [
                {
                    "id": "u1",
                    "label": "Drug",
                    "properties": {
                        "name": "Metformin",
                        "anchor_spans": [{"text": "Metformin", "start": 0, "end": 9}],
                        "umls_cui": "C0025598",
                        "all_names": ["Metformin", "metformin hydrochloride"],
                    },
                },
                {
                    "id": "u2",
                    "label": "Disease",
                    "properties": {"name": "Diabetes"},
                },
                {
                    "id": "u3",
                    "label": "Disease",
                    "properties": {"name": "Hypertension"},
                },
            ],
            "relationships": [
                {
                    "source": "u1",
                    "target": "u2",
                    "type": "TREATS",
                    "negated": False,
                    "properties": {
                        "confidence": 1.0,
                        "evidence_scope": "sentence",
                        "anchor_text": "treats",
                        "anchor_grounding": {"source": [], "relation": [], "target": []},
                        "restoration_status": "full",
                        "condition": "in elderly patients",
                        "quantitative": "50% reduction",
                    },
                },
                {
                    "source": "u1",
                    "target": "u2",
                    "type": "TREATS",
                    "negated": True,
                    "properties": {
                        "confidence": 0.3,
                        "evidence_scope": "partial_grounding",
                        "contradiction_detected": True,
                    },
                },
                {
                    "source": "u1",
                    "target": "u3",
                    "type": "CAUSES",
                    "negated": False,
                    "properties": {
                        "confidence": 0.7,
                        "evidence_scope": "chunk",
                    },
                },
            ],
            "chunks": [
                {"text": "Metformin treats diabetes.", "section_name": "Results", "position": 0},
                {"text": "Metformin does not treat diabetes.", "section_name": "Discussion", "position": 1},
                {"text": "Metformin causes hypertension.", "position": 2},
            ],
            "metadata": {
                "total_entities": 3,
                "total_relationships": 3,
                "extraction_method": "ontology_guided_llm",
                "kg_name": "test_kg",
                "schema_enforcement_dropped_entities": 1,
                "schema_enforcement_dropped_relationships": 0,
                "harmonization_relationships_dropped_unmapped": 0,
                "harmonization_relationships_deduped": 2,
                "harmonization_relationship_contradiction_groups": 1,
                "harmonization_relationship_contradiction_edges": 2,
                "stored_relationships": 2,
                "relationship_store_failures": 1,
                "relationships_skipped_low_confidence": 3,
                "relationships_skipped_schema_mismatch": 1,
                "relationships_reverified_kept": 1,
                "relationships_reverified_rejected": 1,
                "relationship_store_ratio": 0.6667,
                "created_at": "2026-05-09T00:00:00",
            },
        }

    def test_summary_counts_match_kg(self):
        kg = self._minimal_kg()
        report = _make_creator().evaluate_knowledge_graph(kg, print_report=False)
        assert report["summary"]["entities"] == 3
        assert report["summary"]["relationships"] == 3
        assert report["summary"]["chunks"] == 3
        assert report["summary"]["kg_name"] == "test_kg"

    def test_entity_metrics_anchor_and_umls(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        em = report["entity_metrics"]
        assert em["anchor_grounded"] == 1
        assert em["umls_linked"] == 1
        assert em["with_synonyms"] == 1

    def test_entity_metrics_isolated_and_degree(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        em = report["entity_metrics"]
        # u3 has one edge (CAUSES), u1 has 3 edges, u2 has 2 edges → no isolated
        assert em["isolated_no_edges"] == 0
        assert em["max_degree"] >= 3  # u1 appears in all three rels
        assert em["degree_distribution"]["1"] == 1
        assert em["degree_distribution"]["2-4"] == 2
        assert em["hub_entities_topk"][0]["name"] == "Metformin"

    def test_entity_type_distribution(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        td = report["entity_metrics"]["type_distribution"]
        assert td["Disease"] == 2
        assert td["Drug"] == 1

    def test_relationship_metrics_negated_and_contradiction(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        rm = report["relationship_metrics"]
        assert rm["negated"] == 1
        assert rm["contradiction_flagged"] == 1
        assert rm["contradiction_groups"] == 1
        assert rm["contradiction_rate_pct"] == round(100 / 3, 1)

    def test_relationship_metrics_conditioned_and_quantified(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        rm = report["relationship_metrics"]
        assert rm["conditioned"] == 1
        assert rm["quantified"] == 1

    def test_evidence_scope_distribution(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        scopes = report["relationship_metrics"]["evidence_scope_distribution"]
        assert scopes.get("sentence") == 1
        assert scopes.get("chunk") == 1
        assert scopes.get("partial_grounding") == 1

    def test_confidence_stats(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        rm = report["relationship_metrics"]
        assert rm["mean_confidence"] == round((1.0 + 0.3 + 0.7) / 3, 4)
        assert rm["high_confidence_pct"] == round(100 * 2 / 3, 1)
        assert rm["low_confidence_pct"] == round(100 * 1 / 3, 1)

    def test_restoration_counts(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        rm = report["relationship_metrics"]
        assert rm["restoration_full"] == 1
        assert rm["restoration_partial"] == 0
        assert rm["restoration_failed"] == 0

    def test_anchor_relation_phrase_pct(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        # 1 of 3 rels has anchor_text
        assert report["relationship_metrics"]["anchor_relation_phrase_pct"] == round(100 / 3, 1)

    def test_grounding_metrics_section_tags(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        gm = report["grounding_metrics"]
        assert gm["chunks_with_section_tags"] == 2
        assert "Results" in report["summary"]["sections_detected"]
        assert "Discussion" in report["summary"]["sections_detected"]

    def test_pipeline_metrics(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        pm = report["pipeline_metrics"]
        assert pm["schema_enforcement_dropped_entities"] == 1
        assert pm["harmonization_deduped_relationships"] == 2
        # entity pass rate: 3 kept / (3+1 dropped) = 75%
        assert pm["entity_schema_pass_rate_pct"] == 75.0
        assert pm["stored_relationships"] == 2
        assert pm["relationship_store_failures"] == 1
        assert pm["relationships_skipped_low_confidence"] == 3
        assert pm["relationship_store_ratio"] == 0.6667

    def test_precision_recall_perfect(self):
        kg = self._minimal_kg()
        gold = [
            {"source": "u1", "type": "TREATS", "target": "u2"},
            {"source": "u1", "type": "TREATS", "target": "u2"},  # duplicate — deduped in gold set
            {"source": "u1", "type": "CAUSES", "target": "u3"},
        ]
        report = _make_creator().evaluate_knowledge_graph(kg, reference_triples=gold, print_report=False)
        pr = report["precision_recall"]
        # predicted unique keys: TREATS(u1→u2 negated=False), TREATS(u1→u2 negated=True), CAUSES(u1→u3)
        # gold unique keys: TREATS(u1→u2), CAUSES(u1→u3)
        assert pr["gold_triples"] == 2
        assert "precision_recall" in report

    def test_reference_kg_reports_entity_and_relationship_pr(self):
        kg = self._minimal_kg()
        reference = {
            "nodes": [
                {
                    "id": "g1",
                    "label": "Drug",
                    "properties": {"name": "Metformin"},
                },
                {
                    "id": "g2",
                    "label": "Disease",
                    "properties": {"name": "Diabetes"},
                },
                {
                    "id": "g3",
                    "label": "Disease",
                    "properties": {"name": "Hypertension"},
                },
                {
                    "id": "g4",
                    "label": "Disease",
                    "properties": {"name": "Obesity"},
                },
            ],
            "relationships": [
                {"source": "g1", "type": "TREATS", "target": "g2"},
                {"source": "g1", "type": "CAUSES", "target": "g3"},
                {"source": "g1", "type": "PREVENTS", "target": "g4"},
            ],
        }
        report = _make_creator().evaluate_knowledge_graph(
            kg,
            reference=reference,
            print_report=False,
        )
        epr = report["entity_precision_recall"]
        rpr = report["relationship_precision_recall"]
        assert epr["gold_entities"] == 4
        assert epr["predicted_entities"] == 3
        assert epr["precision"] == 1.0
        assert epr["recall"] == 0.75
        assert rpr["gold_triples"] == 3
        assert rpr["predicted_triples"] == 2
        assert rpr["precision"] == 1.0
        assert rpr["recall"] == round(2 / 3, 4)
        assert "qualified_relationship_precision_recall" in report

    def test_precision_recall_no_overlap(self):
        kg = self._minimal_kg()
        gold = [{"source": "X", "type": "INHIBITS", "target": "Y"}]
        report = _make_creator().evaluate_knowledge_graph(kg, reference_triples=gold, print_report=False)
        pr = report["precision_recall"]
        assert pr["true_positives"] == 0
        assert pr["precision"] == 0.0
        assert pr["recall"] == 0.0
        assert pr["f1"] == 0.0

    def test_no_precision_recall_when_no_gold(self):
        report = _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=False)
        assert "precision_recall" not in report

    def test_empty_kg_does_not_crash(self):
        empty_kg = {"nodes": [], "relationships": [], "chunks": [], "metadata": {}}
        report = _make_creator().evaluate_knowledge_graph(empty_kg, print_report=False)
        assert report["summary"]["entities"] == 0
        assert report["summary"]["relationships"] == 0
        assert report["entity_metrics"]["isolated_no_edges"] == 0
        assert report["relationship_metrics"]["mean_confidence"] is None

    def test_print_report_runs_without_error(self, capsys):
        _make_creator().evaluate_knowledge_graph(self._minimal_kg(), print_report=True)
        captured = capsys.readouterr()
        assert "KG EVALUATION REPORT" in captured.out
        assert "SUMMARY" in captured.out
        assert "PIPELINE HEALTH" in captured.out
