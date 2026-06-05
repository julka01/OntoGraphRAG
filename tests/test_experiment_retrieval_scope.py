import pytest
from types import SimpleNamespace


MIRAGEEvaluationPipeline = pytest.importorskip(
    "experiments.experiment",
    reason="experiment module dependencies not installed in this test environment",
).MIRAGEEvaluationPipeline
experiment_mod = pytest.importorskip(
    "experiments.experiment",
    reason="experiment module dependencies not installed in this test environment",
)


def _make_benchmark():
    benchmark = MIRAGEEvaluationPipeline.__new__(MIRAGEEvaluationPipeline)
    benchmark.QUESTION_SCOPED_DATASETS = frozenset({"musique", "hotpotqa", "2wikimultihopqa", "pubmedqa"})
    return benchmark


def test_validate_retrieval_scope_rejects_cross_question_chunks():
    benchmark = _make_benchmark()
    rag_result = {
        "context": {
            "chunks": [
                {"chunk_id": "c1", "kg_name": "musique", "question_id": "q_other", "document": "doc1"},
            ]
        }
    }

    assert benchmark._validate_retrieval_scope(
        rag_result=rag_result,
        dataset_name="musique",
        system_name="KG-RAG",
        question_id="q_target",
    ) is False


def test_validate_retrieval_scope_allows_matching_question_scope():
    benchmark = _make_benchmark()
    rag_result = {
        "context": {
            "chunks": [
                {"chunk_id": "c1", "kg_name": "musique", "question_id": "q_target", "document": "doc1"},
                {"chunk_id": "c2", "kg_name": "musique", "question_id": "q_target", "document": "doc1"},
            ]
        }
    }

    assert benchmark._validate_retrieval_scope(
        rag_result=rag_result,
        dataset_name="musique",
        system_name="KG-RAG",
        question_id="q_target",
    ) is True


def test_collect_sample_responses_preserves_question_scope_and_sampling_pool():
    benchmark = _make_benchmark()
    benchmark.entropy_samples = 2
    benchmark.llm = object()

    calls = []

    class FakeRAGSystem:
        def generate_response(self, **kwargs):
            calls.append(kwargs)
            sample_id = kwargs["retrieval_sample_id"]
            return {
                "response": f"sample-response-{sample_id}",
                "context": {
                    "chunks": [
                        {"text": f"sample chunk {sample_id}"},
                    ]
                },
            }

    responses, retrieved_chunk_texts, overlap = benchmark._collect_sample_responses(
        FakeRAGSystem(),
        question="Who wrote Dune?",
        base_result={
            "response": "base deterministic answer",
            "context": {"chunks": [{"text": "base chunk"}]},
        },
        kg_name="musique",
        question_id="q_target",
    )

    assert responses == ["sample-response-0", "sample-response-1"]
    assert retrieved_chunk_texts == ["base chunk"]
    assert overlap == 0.0
    assert [call["question_id"] for call in calls] == ["q_target", "q_target"]
    assert [call["retrieval_sample_id"] for call in calls] == [0, 1]


def test_retrieval_study_configs_do_not_override_embedding_profile():
    configs = MIRAGEEvaluationPipeline.build_retrieval_study_eval_configs(
        profile="small",
        similarity_thresholds=[0.1],
        max_chunks_values=[10],
        retrieval_temperature_values=[0.0],
        retrieval_shortlist_factor=4,
    )

    assert len(configs) == 5
    for config in configs:
        assert "retrieval_profile" not in config["retrieval_stack"]


def test_final_pair_configs_do_not_override_embedding_profile():
    configs = MIRAGEEvaluationPipeline.build_retrieval_study_eval_configs(
        profile="final_pair",
        similarity_thresholds=[0.1],
        max_chunks_values=[10],
        retrieval_temperature_values=[0.0],
        retrieval_shortlist_factor=4,
    )

    assert len(configs) == 2
    assert [cfg["retrieval_variant"] for cfg in configs] == [
        "dense_floor",
        "kg_entity_first",
    ]
    assert configs[0]["executed_systems"] == ["vanilla_rag"]
    assert configs[1]["executed_systems"] == ["kg_rag"]
    for config in configs:
        assert "retrieval_profile" not in config["retrieval_stack"]


def test_full_kg_builder_profile_enables_stronger_extraction(monkeypatch):
    benchmark = MIRAGEEvaluationPipeline.__new__(MIRAGEEvaluationPipeline)
    benchmark.DATASET_KG_SCOPE_EVALUATION_SUBSET = "evaluation_subset"
    benchmark.dataset_kg_scope = "evaluation_subset"
    benchmark.kg_builder_profile = "full"
    benchmark.neo4j_uri = "bolt://localhost:7687"
    benchmark.neo4j_user = "neo4j"
    benchmark.neo4j_password = "test"
    benchmark.embedding_provider = "sentence_transformers"
    benchmark.kg_llm_provider = object()
    benchmark.llm_model = "gpt-4o-mini"
    benchmark.rebuild_kg = False
    benchmark.DATASET_TRACKS = {"realmedqa": "biomedical_grounding"}
    benchmark.DEFAULT_TRACK = "other"

    benchmark._prepare_dataset_kg_contract = lambda dataset_name, force_resample=False: {
        "passages": [
            SimpleNamespace(
                text="Passage text",
                dataset=dataset_name,
                question_id="q1",
                passage_index=0,
            )
        ],
        "build_meta": {},
        "records_for_kg": [{"id": "q1"}],
        "evaluable_inference_records": [{"id": "q1"}],
    }

    seen = {}

    class FakeBuilder:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def generate_knowledge_graph_from_passages(self, **kwargs):
            seen["generate_kwargs"] = kwargs
            return {
                "metadata": {
                    "stored_in_neo4j": True,
                    "stored_relationships": 0,
                    "total_passages": 1,
                    "total_chunks": 1,
                    "total_entities": 0,
                }
            }

    monkeypatch.setattr(experiment_mod, "UnifiedOntologyGuidedKGCreator", FakeBuilder)

    assert benchmark._build_kg_for_dataset("realmedqa") is True
    assert seen["kwargs"]["enable_anchor_constrained_extraction"] is True
    assert seen["kwargs"]["enable_self_reflection"] is True
    assert seen["kwargs"]["enable_anchor_coverage_supplement"] is True
    assert seen["kwargs"]["enable_cross_passage_relation_recovery"] is True
    assert seen["kwargs"]["self_consistency_n"] == 3
    assert seen["kwargs"]["few_shot_example_count"] == 4
    assert seen["kwargs"]["enable_low_confidence_triple_reverification"] is True
    assert seen["kwargs"]["low_confidence_reverify_threshold"] == 0.55
    assert seen["kwargs"]["min_triple_confidence"] == 0.2
    assert seen["kwargs"]["relationship_type_similarity_threshold"] == 0.7
    assert seen["kwargs"]["enable_umls_linking"] is True
    assert seen["kwargs"]["enable_soft_entity_linking"] is True
    assert seen["kwargs"]["enable_fragmentation_repair"] is True
    assert seen["kwargs"]["enable_graph_summaries"] is True
    assert seen["kwargs"]["enable_claim_extraction"] is True


def test_kg_routing_distribution_counts_entity_first_and_dense_fallback():
    details = [
        {"kg_retrieval_route": "entity_first", "kg_route_reason": "success"},
        {"kg_retrieval_route": "semantic_only", "kg_route_reason": "no_graph_signal"},
        {"kg_retrieval_route": "rfge", "kg_route_reason": "entity_first_failed"},
        {"kg_retrieval_route": "entity_first", "kg_route_reason": "success", "kg_generation_failed": True},
    ]

    dist = MIRAGEEvaluationPipeline._compute_kg_routing_distribution(details)

    assert dist["n"] == 3
    assert dist["routes"]["entity_first"]["count"] == 1
    assert dist["routes"]["semantic_only"]["count"] == 1
    assert dist["routes"]["rfge"]["count"] == 1
    assert dist["pure_entity_first_rate"] == pytest.approx(1 / 3)
    assert dist["dense_fallback_rate"] == pytest.approx(1 / 3)
    assert dist["graph_route_rate"] == pytest.approx(2 / 3)


def test_question_scoped_multihop_kg_build_uses_lightweight_profile(monkeypatch):
    benchmark = _make_benchmark()
    benchmark.dataset_kg_scope = "evaluation_subset"
    benchmark.neo4j_uri = "bolt://localhost:7687"
    benchmark.neo4j_user = "neo4j"
    benchmark.neo4j_password = "test"
    benchmark.embedding_provider = "sentence_transformers"
    benchmark.kg_llm_provider = object()
    benchmark.llm_model = "gpt-4o-mini"
    benchmark.rebuild_kg = False

    benchmark._prepare_dataset_kg_contract = lambda dataset_name, force_resample=False: {
        "passages": [
            SimpleNamespace(
                text="Passage text",
                dataset=dataset_name,
                question_id="q1",
                passage_index=0,
            )
        ],
        "build_meta": {},
        "records_for_kg": [{"id": "q1"}],
        "evaluable_inference_records": [{"id": "q1"}],
    }

    seen = {}

    class FakeBuilder:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def generate_knowledge_graph_from_passages(self, **kwargs):
            seen["generate_kwargs"] = kwargs
            return {
                "metadata": {
                    "stored_in_neo4j": True,
                    "stored_relationships": 0,
                    "total_passages": 1,
                    "total_chunks": 1,
                    "total_entities": 0,
                }
            }

    monkeypatch.setattr(experiment_mod, "UnifiedOntologyGuidedKGCreator", FakeBuilder)

    assert benchmark._build_kg_for_dataset("2wikimultihopqa") is True
    assert seen["kwargs"]["enable_self_reflection"] is False
    assert seen["kwargs"]["enable_anchor_coverage_supplement"] is False
    assert seen["kwargs"]["enable_cross_passage_relation_recovery"] is False
