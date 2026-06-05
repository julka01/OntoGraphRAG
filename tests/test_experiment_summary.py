"""
Regression tests for the experiment summary and leakage validation paths.

Pure unit tests — no live Neo4j, LLM, or dataset files required.
Covers:
  - track_aggregates: correct per-(track, config) accumulation
  - track_aggregates: no first-config bias in multi-config runs
  - judge_independent: checks both provider AND model
  - validate_no_leakage: schema leakage detection
  - validate_no_leakage: answer text leakage detection (including aliases)
  - validate_no_leakage: binary answers exempted
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.dataset_adapters import (
    InferenceRecord,
    GoldRecord,
    validate_no_leakage,
)
# Import from the dependency-light module — no wandb/dotenv/neo4j pulled in.
from experiments.summary_utils import (
    accumulate_track_accuracy,
    compute_accuracy_breakdown,
    compute_hop_accuracy_breakdown,
    select_best_retrieval_configs,
)


# ---------------------------------------------------------------------------
# Helpers — minimal dataset block builder
# ---------------------------------------------------------------------------

def _cfg_res(name: str, vanilla_acc: float, kg_acc: float) -> dict:
    return {
        "config": {"name": name},
        "vanilla_accuracy": vanilla_acc,
        "kg_accuracy": kg_acc,
        "config_results": [],
    }


def _dataset_block(dataset: str, track: str, *cfg_res_list) -> dict:
    return {
        "dataset": dataset,
        "track": track,
        "config_results": list(cfg_res_list),
    }


def _run_track_aggregation(dataset_blocks):
    """Thin wrapper — delegates to the production function."""
    return accumulate_track_accuracy(dataset_blocks)


# ---------------------------------------------------------------------------
# 1. track_aggregates — correct accumulation
# ---------------------------------------------------------------------------

class TestTrackAggregates:

    def test_single_config_single_dataset(self):
        blocks = [
            _dataset_block("pubmedqa", "biomedical",
                           _cfg_res("default", 0.6, 0.7)),
        ]
        agg = _run_track_aggregation(blocks)
        assert "biomedical" in agg
        rows = agg["biomedical"]
        assert len(rows) == 1
        assert rows[0]["vanilla_macro_accuracy"] == pytest.approx(0.6)
        assert rows[0]["kg_macro_accuracy"] == pytest.approx(0.7)

    def test_single_config_two_datasets_macro_average(self):
        # macro = mean of per-dataset accuracies, not pooled
        blocks = [
            _dataset_block("pubmedqa", "biomedical",   _cfg_res("default", 0.6, 0.7)),
            _dataset_block("bioasq",   "biomedical",   _cfg_res("default", 0.8, 0.9)),
        ]
        agg = _run_track_aggregation(blocks)
        row = agg["biomedical"][0]
        assert row["vanilla_macro_accuracy"] == pytest.approx(0.7)   # (0.6+0.8)/2
        assert row["kg_macro_accuracy"]      == pytest.approx(0.8)   # (0.7+0.9)/2
        assert row["num_datasets"] == 2

    def test_multi_config_produces_one_row_per_config(self):
        # Two configs on the same dataset must not collapse into one row
        blocks = [
            _dataset_block("pubmedqa", "biomedical",
                           _cfg_res("threshold_0.1", 0.5, 0.6),
                           _cfg_res("threshold_0.2", 0.55, 0.65)),
        ]
        agg = _run_track_aggregation(blocks)
        rows = agg["biomedical"]
        cfg_names = {r["config_name"] for r in rows}
        assert "threshold_0.1" in cfg_names
        assert "threshold_0.2" in cfg_names
        assert len(rows) == 2

    def test_multi_config_no_first_config_bias(self):
        # Config A appears first; config B second.  With the old first-wins logic,
        # config B's accuracy would be lost.  Each must appear separately.
        blocks = [
            _dataset_block("pubmedqa", "biomedical",
                           _cfg_res("cfg_a", 0.4, 0.5),
                           _cfg_res("cfg_b", 0.8, 0.9)),
        ]
        agg = _run_track_aggregation(blocks)
        rows = {r["config_name"]: r for r in agg["biomedical"]}
        assert rows["cfg_a"]["kg_macro_accuracy"] == pytest.approx(0.5)
        assert rows["cfg_b"]["kg_macro_accuracy"] == pytest.approx(0.9)

    def test_dataset_deduplicated_within_config(self):
        # The same dataset appearing twice under the same config must not
        # be double-counted (e.g. duplicate dataset_blocks in raw results).
        blocks = [
            _dataset_block("pubmedqa", "biomedical", _cfg_res("default", 0.6, 0.7)),
            _dataset_block("pubmedqa", "biomedical", _cfg_res("default", 0.6, 0.7)),
        ]
        agg = _run_track_aggregation(blocks)
        row = agg["biomedical"][0]
        assert row["num_datasets"] == 1

    def test_two_tracks_independent(self):
        blocks = [
            _dataset_block("pubmedqa",   "biomedical", _cfg_res("default", 0.6, 0.7)),
            _dataset_block("hotpotqa",   "multihop",   _cfg_res("default", 0.4, 0.5)),
        ]
        agg = _run_track_aggregation(blocks)
        assert "biomedical" in agg
        assert "multihop" in agg
        assert agg["biomedical"][0]["kg_macro_accuracy"] == pytest.approx(0.7)
        assert agg["multihop"][0]["kg_macro_accuracy"]   == pytest.approx(0.5)


class TestRetrievalConfigSelection:

    def test_selects_best_config_per_dataset_and_system(self):
        blocks = [
            _dataset_block(
                "pubmedqa",
                "biomedical",
                {
                    "config": {"name": "dense_floor"},
                    "vanilla_accuracy": 0.70,
                    "kg_accuracy": 0.68,
                    "vanilla_answer_f1": 0.70,
                    "kg_answer_f1": 0.68,
                    "vanilla_answer_em": 0.70,
                    "kg_answer_em": 0.68,
                    "vanilla_answered_questions": 20,
                    "kg_answered_questions": 20,
                    "vanilla_accuracy_raw": 0.70,
                    "kg_accuracy_raw": 0.68,
                },
                {
                    "config": {"name": "kg_hybrid"},
                    "vanilla_accuracy": 0.72,
                    "kg_accuracy": 0.81,
                    "vanilla_answer_f1": 0.72,
                    "kg_answer_f1": 0.82,
                    "vanilla_answer_em": 0.72,
                    "kg_answer_em": 0.81,
                    "vanilla_answered_questions": 20,
                    "kg_answered_questions": 20,
                    "vanilla_accuracy_raw": 0.72,
                    "kg_accuracy_raw": 0.81,
                },
            ),
        ]

        selection = select_best_retrieval_configs(blocks)

        assert selection["per_dataset"]["pubmedqa"]["vanilla_rag"]["config_name"] == "kg_hybrid"
        assert selection["per_dataset"]["pubmedqa"]["kg_rag"]["config_name"] == "kg_hybrid"

    def test_macro_selection_uses_clean_accuracy_then_f1(self):
        blocks = [
            _dataset_block(
                "pubmedqa",
                "biomedical",
                {
                    "config": {"name": "cfg_a"},
                    "vanilla_accuracy": 0.70,
                    "kg_accuracy": 0.75,
                    "vanilla_answer_f1": 0.70,
                    "kg_answer_f1": 0.75,
                    "vanilla_answer_em": 0.70,
                    "kg_answer_em": 0.75,
                    "vanilla_answered_questions": 20,
                    "kg_answered_questions": 20,
                    "vanilla_accuracy_raw": 0.70,
                    "kg_accuracy_raw": 0.75,
                },
                {
                    "config": {"name": "cfg_b"},
                    "vanilla_accuracy": 0.70,
                    "kg_accuracy": 0.75,
                    "vanilla_answer_f1": 0.74,
                    "kg_answer_f1": 0.80,
                    "vanilla_answer_em": 0.70,
                    "kg_answer_em": 0.75,
                    "vanilla_answered_questions": 20,
                    "kg_answered_questions": 20,
                    "vanilla_accuracy_raw": 0.70,
                    "kg_accuracy_raw": 0.75,
                },
            ),
            _dataset_block(
                "hotpotqa",
                "multihop",
                {
                    "config": {"name": "cfg_a"},
                    "vanilla_accuracy": 0.66,
                    "kg_accuracy": 0.71,
                    "vanilla_answer_f1": 0.66,
                    "kg_answer_f1": 0.71,
                    "vanilla_answer_em": 0.66,
                    "kg_answer_em": 0.71,
                    "vanilla_answered_questions": 20,
                    "kg_answered_questions": 20,
                    "vanilla_accuracy_raw": 0.66,
                    "kg_accuracy_raw": 0.71,
                },
                {
                    "config": {"name": "cfg_b"},
                    "vanilla_accuracy": 0.66,
                    "kg_accuracy": 0.71,
                    "vanilla_answer_f1": 0.71,
                    "kg_answer_f1": 0.78,
                    "vanilla_answer_em": 0.66,
                    "kg_answer_em": 0.71,
                    "vanilla_answered_questions": 20,
                    "kg_answered_questions": 20,
                    "vanilla_accuracy_raw": 0.66,
                    "kg_accuracy_raw": 0.71,
                },
            ),
        ]

        selection = select_best_retrieval_configs(blocks)

        assert selection["overall"]["vanilla_rag"]["best_config"]["config_name"] == "cfg_b"
        assert selection["overall"]["kg_rag"]["best_config"]["config_name"] == "cfg_b"


# ---------------------------------------------------------------------------
# 2. judge_independent logic
# ---------------------------------------------------------------------------

class TestJudgeIndependent:
    """
    judge_independent must be True when either provider OR model differs from the
    generation model/provider — same model string on a different provider is a
    distinct system.
    """

    def _ji(self, judge_model, judge_provider, llm_model, llm_provider):
        return (judge_model != llm_model) or (judge_provider != llm_provider)

    def test_same_model_same_provider_is_not_independent(self):
        assert self._ji("gpt-4o", "openai", "gpt-4o", "openai") is False

    def test_different_model_same_provider_is_independent(self):
        assert self._ji("gpt-4o", "openai", "gpt-4o-mini", "openai") is True

    def test_same_model_different_provider_is_independent(self):
        # Same model string, different provider → independent (could be a fork/mirror)
        assert self._ji("gpt-4o", "openrouter", "gpt-4o", "openai") is True

    def test_different_model_different_provider_is_independent(self):
        assert self._ji("claude-3-opus", "anthropic", "gpt-4o", "openai") is True


# ---------------------------------------------------------------------------
# 3. validate_no_leakage
# ---------------------------------------------------------------------------

def _inf(id_: str, question: str, contexts) -> InferenceRecord:
    return InferenceRecord(id=id_, dataset="test", question=question, contexts=contexts)


def _gold(id_: str, short_answer: str, aliases=None, long_answer=None) -> GoldRecord:
    return GoldRecord(id=id_, short_answer=short_answer, aliases=aliases,
                      long_answer=long_answer)


class TestValidateNoLeakage:

    def test_clean_records_pass(self):
        inf = _inf("1", "Does aspirin treat fever?", ["Aspirin is an NSAID."])
        gold = _gold("1", "yes")
        assert validate_no_leakage([inf], [gold]) is True

    def test_schema_leakage_detected(self):
        # Add an answer-bearing field directly to the InferenceRecord instance
        inf = _inf("1", "Some question?", ["Some context."])
        inf.short_answer = "leaked"   # type: ignore[attr-defined]
        gold = _gold("1", "leaked")
        assert validate_no_leakage([inf], [gold]) is False

    def test_answer_text_leakage_detected(self):
        # Gold answer appears verbatim inside context passage
        inf  = _inf("1", "Who invented the telephone?", ["Alexander Graham Bell invented the telephone in 1876."])
        gold = _gold("1", "Alexander Graham Bell")
        assert validate_no_leakage([inf], [gold]) is False

    def test_alias_leakage_detected(self):
        # Gold answer itself is generic but alias appears verbatim in context
        inf  = _inf("1", "Who invented the telephone?", ["A.G. Bell invented the telephone."])
        gold = _gold("1", "Alexander Graham Bell", aliases=["A.G. Bell"])
        assert validate_no_leakage([inf], [gold]) is False

    def test_binary_answer_exempt(self):
        # "yes" / "no" appear legitimately in any passage — must not flag as leakage
        inf  = _inf("1", "Does aspirin reduce fever?", ["Studies show this is yes, aspirin reduces fever."])
        gold = _gold("1", "yes")
        assert validate_no_leakage([inf], [gold]) is True

    def test_long_answer_leakage_detected(self):
        long = "Aspirin inhibits cyclooxygenase enzymes"
        inf  = _inf("1", "How does aspirin work?", [long])
        gold = _gold("1", "COX inhibitor", long_answer=long)
        assert validate_no_leakage([inf], [gold]) is False

    def test_no_match_passes(self):
        inf  = _inf("1", "What is metformin?", ["Insulin is a hormone."])
        gold = _gold("1", "biguanide antidiabetic")
        assert validate_no_leakage([inf], [gold]) is True


# ---------------------------------------------------------------------------
# 4. clean accuracy excluding generation failures
# ---------------------------------------------------------------------------

class TestComputeAccuracyBreakdown:

    def test_reports_raw_and_clean_accuracy(self):
        rows = [
            {"vanilla_correct": True,  "kg_correct": True,  "vanilla_generation_failed": False, "kg_generation_failed": False},
            {"vanilla_correct": False, "kg_correct": True,  "vanilla_generation_failed": False, "kg_generation_failed": False},
            {"vanilla_correct": False, "kg_correct": False, "vanilla_generation_failed": True,  "kg_generation_failed": False},
            {"vanilla_correct": False, "kg_correct": False, "vanilla_generation_failed": False, "kg_generation_failed": True},
        ]

        out = compute_accuracy_breakdown(rows)

        assert out["total_questions"] == 4
        assert out["num_generation_failures_vanilla"] == 1
        assert out["num_generation_failures_kg"] == 1
        assert out["vanilla_answered_questions"] == 3
        assert out["kg_answered_questions"] == 3
        assert out["shared_answered_questions"] == 2
        assert out["vanilla_accuracy"] == pytest.approx(0.25)
        assert out["kg_accuracy"] == pytest.approx(0.50)
        assert out["vanilla_accuracy_excluding_errors"] == pytest.approx(1 / 3)
        assert out["kg_accuracy_excluding_errors"] == pytest.approx(2 / 3)
        assert out["vanilla_accuracy_shared_clean"] == pytest.approx(0.5)
        assert out["kg_accuracy_shared_clean"] == pytest.approx(1.0)

    def test_empty_rows_return_zeroes(self):
        out = compute_accuracy_breakdown([])
        assert out["total_questions"] == 0
        assert out["vanilla_accuracy"] == 0.0
        assert out["kg_accuracy"] == 0.0
        assert out["vanilla_accuracy_excluding_errors"] == 0.0
        assert out["kg_accuracy_excluding_errors"] == 0.0


class TestComputeHopAccuracyBreakdown:

    def test_groups_rows_by_hop_and_averages_key_metrics(self):
        rows = [
            {
                "hop_count": 2,
                "vanilla_correct": True,
                "kg_correct": False,
                "vanilla_generation_failed": False,
                "kg_generation_failed": False,
                "vanilla_answer_em": 1.0,
                "kg_answer_em": 0.0,
                "vanilla_answer_f1": 1.0,
                "kg_answer_f1": 0.5,
                "vanilla_sd_uq": 0.2,
                "kg_sd_uq": 0.7,
                "vanilla_vn_entropy": 0.1,
                "kg_vn_entropy": 0.4,
                "vanilla_support_entailment_uncertainty": 0.3,
                "kg_support_entailment_uncertainty": 0.8,
            },
            {
                "hop_count": 2,
                "vanilla_correct": False,
                "kg_correct": True,
                "vanilla_generation_failed": False,
                "kg_generation_failed": False,
                "vanilla_answer_em": 0.0,
                "kg_answer_em": 1.0,
                "vanilla_answer_f1": 0.2,
                "kg_answer_f1": 1.0,
                "vanilla_sd_uq": 0.4,
                "kg_sd_uq": 0.6,
                "vanilla_vn_entropy": 0.3,
                "kg_vn_entropy": 0.5,
                "vanilla_support_entailment_uncertainty": 0.6,
                "kg_support_entailment_uncertainty": 0.4,
            },
            {
                "hop_count": 3,
                "vanilla_correct": True,
                "kg_correct": True,
                "vanilla_generation_failed": False,
                "kg_generation_failed": False,
                "vanilla_answer_em": 1.0,
                "kg_answer_em": 1.0,
                "vanilla_answer_f1": 1.0,
                "kg_answer_f1": 1.0,
                "vanilla_sd_uq": 0.9,
                "kg_sd_uq": 0.1,
                "vanilla_vn_entropy": 0.8,
                "kg_vn_entropy": 0.2,
                "vanilla_support_entailment_uncertainty": 0.7,
                "kg_support_entailment_uncertainty": 0.2,
            },
            {
                "hop_count": None,
                "vanilla_correct": False,
                "kg_correct": False,
            },
        ]

        out = compute_hop_accuracy_breakdown(rows)

        assert set(out.keys()) == {"2-hop", "3-hop"}
        assert out["2-hop"]["n"] == 2
        assert out["2-hop"]["vanilla_accuracy"] == pytest.approx(0.5)
        assert out["2-hop"]["kg_accuracy"] == pytest.approx(0.5)
        assert out["2-hop"]["vanilla_answer_em"] == pytest.approx(0.5)
        assert out["2-hop"]["kg_answer_em"] == pytest.approx(0.5)
        assert out["2-hop"]["metrics_by_approach"]["vanilla_rag"]["sd_uq"] == pytest.approx(0.3)
        assert out["2-hop"]["metrics_by_approach"]["kg_rag"]["vn_entropy"] == pytest.approx(0.45)
        assert out["3-hop"]["n"] == 1
        assert out["3-hop"]["kg_answer_f1"] == pytest.approx(1.0)

    def test_accepts_empty_metric_list(self):
        rows = [
            {
                "hop_count": 4,
                "vanilla_correct": True,
                "kg_correct": False,
                "vanilla_generation_failed": False,
                "kg_generation_failed": False,
            }
        ]

        out = compute_hop_accuracy_breakdown(rows, metric_names=[])
        assert out["4-hop"]["metrics_by_approach"]["vanilla_rag"] == {}
        assert out["4-hop"]["metrics_by_approach"]["kg_rag"] == {}
