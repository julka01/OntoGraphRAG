import json

import numpy as np
import pytest

from experiments import hop_stratified_analysis as hop


def test_load_details_raises_on_ambiguous_multi_config(tmp_path, monkeypatch):
    monkeypatch.setattr(hop, "PREFERRED_CONFIG_NAME", None)
    payload = {
        "config_results": [
            {"config": {"name": "cfg_a"}, "details": [{"id": 1}]},
            {"config": {"name": "cfg_b"}, "details": [{"id": 2}]},
        ]
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        hop.load_details(str(path))


def test_load_details_prefers_default_config(tmp_path, monkeypatch):
    monkeypatch.setattr(hop, "PREFERRED_CONFIG_NAME", None)
    payload = {
        "config_results": [
            {"config": {"name": "sweep"}, "details": [{"id": "wrong"}]},
            {"config": {"name": "default"}, "details": [{"id": "right"}]},
        ]
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))

    details = hop.load_details(str(path))

    assert details == [{"id": "right"}]


def test_compute_stats_excludes_generation_failures_and_uses_sps_as_uncertainty():
    details = [
        {
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_subgraph_perturbation_stability": 0.1,
            "kg_graph_path_support": 0.1,
        },
        {
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_subgraph_perturbation_stability": 0.2,
            "kg_graph_path_support": 0.2,
        },
        {
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_subgraph_perturbation_stability": 0.8,
            "kg_graph_path_support": 0.8,
        },
        {
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_subgraph_perturbation_stability": 0.9,
            "kg_graph_path_support": 0.9,
        },
        {
            "kg_correct": False,
            "kg_generation_failed": True,
            "kg_subgraph_perturbation_stability": 0.0,
            "kg_graph_path_support": 0.0,
        },
    ]

    stats = hop.compute_stats(details, "2-hop", "HotpotQA")

    assert stats["n"] == 4
    assert stats["error_rate"] == 0.5
    assert stats["subgraph_perturbation_stability"]["auroc"] == pytest.approx(1.0)
    assert stats["graph_path_support"]["auroc"] == pytest.approx(1.0)
    assert not np.isnan(stats["subgraph_perturbation_stability"]["ppv"])


def test_compute_stats_excludes_gps_null_rows():
    details = [
        {
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.1,
            "kg_graph_path_support_null_reason": "",
        },
        {
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.2,
            "kg_graph_path_support_null_reason": "",
        },
        {
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.8,
            "kg_graph_path_support_null_reason": "",
        },
        {
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.9,
            "kg_graph_path_support_null_reason": "",
        },
        {
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.5,
            "kg_graph_path_support_null_reason": "no_q_entities",
        },
    ]

    stats = hop.compute_stats(details, "2-hop", "HotpotQA")

    assert stats["n"] == 5
    assert stats["graph_path_support"]["auroc"] == pytest.approx(1.0)
    assert stats["graph_path_support"]["ppv"] == pytest.approx(1.0)
