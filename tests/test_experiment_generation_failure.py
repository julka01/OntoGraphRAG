import pytest
import os


MIRAGEEvaluationPipeline = pytest.importorskip(
    "experiments.experiment",
    reason="experiment module dependencies not installed in this test environment",
).MIRAGEEvaluationPipeline


def _make_pipeline():
    return MIRAGEEvaluationPipeline.__new__(MIRAGEEvaluationPipeline)


def test_exact_insufficient_information_is_failure_when_gold_is_specific():
    pipeline = _make_pipeline()

    assert pipeline._is_generation_failure(
        {"response": "Insufficient Information."},
        "insufficient information.",
        expected_answer="atorvastatin",
    ) is True


def test_exact_insufficient_information_is_not_failure_when_gold_matches():
    pipeline = _make_pipeline()

    assert pipeline._is_generation_failure(
        {"response": "Insufficient Information."},
        "insufficient information.",
        expected_answer="insufficient information.",
    ) is False


def test_small_retrieval_study_builds_expected_variants():
    configs = MIRAGEEvaluationPipeline.build_retrieval_study_eval_configs(
        profile="small",
        similarity_thresholds=[0.1],
        max_chunks_values=[10],
        retrieval_temperature_values=[0.0],
        retrieval_shortlist_factor=4,
    )

    assert [cfg["retrieval_variant"] for cfg in configs] == [
        "dense_floor",
        "modern_vector",
        "kg_entity_first",
        "kg_rfge",
        "kg_hybrid",
    ]
    assert configs[0]["kg_system"]["retrieval_mode"] == "vector_only"
    assert configs[-1]["kg_system"]["retrieval_mode"] == "hybrid_auto"


def test_final_pair_retrieval_study_builds_expected_variants():
    configs = MIRAGEEvaluationPipeline.build_retrieval_study_eval_configs(
        profile="final_pair",
        similarity_thresholds=[0.1],
        max_chunks_values=[10],
        retrieval_temperature_values=[0.0],
        retrieval_shortlist_factor=4,
    )

    assert [cfg["retrieval_variant"] for cfg in configs] == [
        "dense_floor",
        "kg_entity_first",
    ]
    assert configs[0]["kg_system"]["retrieval_mode"] == "vector_only"
    assert configs[1]["kg_system"]["retrieval_mode"] == "entity_first"
    assert configs[0]["executed_systems"] == ["vanilla_rag"]
    assert configs[1]["executed_systems"] == ["kg_rag"]


def test_temporary_env_restores_original_values(monkeypatch):
    monkeypatch.setenv("ONTOGRAPHRAG_QUERY_FUSION", "1")

    with MIRAGEEvaluationPipeline._temporary_env({"ONTOGRAPHRAG_QUERY_FUSION": "0"}):
        assert os.environ["ONTOGRAPHRAG_QUERY_FUSION"] == "0"

    assert os.environ["ONTOGRAPHRAG_QUERY_FUSION"] == "1"
