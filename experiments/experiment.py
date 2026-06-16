"""
MIRAGE Dataset Evaluation Pipeline (Experiment Mode)

This script runs sequential evaluation on MIRAGE datasets in experiment mode:
For each dataset:
   1. Build/reuse dataset-scoped KG in Neo4j (kgName filter)
   2. Build KG from that dataset's contexts using proper entity extraction
   3. Run RAG comparison experiments (Vanilla vs KG-RAG)
   4. Optionally compute canonical uncertainty metrics
   5. Save results

Usage:
    python experiments/run_mirage_evaluation.py --num-samples 10
"""

import sys
import os
import json
import hashlib
import logging
import time
import argparse
import re
import subprocess
import shutil
from contextlib import contextmanager
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()
os.environ["EMBEDDING_PROVIDER"] = "sentence_transformers"

from ontographrag.rag.systems.vanilla_rag_system import VanillaRAGSystem
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem
from ontographrag.rag.graph_state import graph_state_diversity, summarize_context_graph_state
from ontographrag.providers.model_providers import (
    get_provider as get_model_provider,
    LangChainRunnableAdapter,
    TemperatureLockedProvider,
)
from ontographrag.kg.builders.enhanced_kg_creator import UnifiedOntologyGuidedKGCreator
from neo4j import GraphDatabase

from experiments.dataset_adapters import (
    normalize_dataset,
    load_raw_dataset,
    build_passage_corpus,
    build_global_corpus_passages,
    get_dataset_corpus_profile,
    infer_hop_count_from_raw,
    validate_no_leakage,
)
from experiments.subset_selection import (
    resolve_question_subset,
    selection_file_path,
)
from experiments.answer_formatting import (
    build_answer_instructions,
    normalize_answer_to_contract,
)
from experiments.kg_reuse import assess_dataset_kg_compatibility
from experiments.official_answer_metrics import (
    compute_answer_em_f1,
    supports_official_answer_metrics,
)

# Import uncertainty metrics (entropy + structural variants)
from experiments.uncertainty_metrics import (
    compute_all_uncertainty_metrics,
    compute_auroc_aurec,
    compute_ece,
    compute_precision_at_k,
    compute_graph_path_support,
    compute_graph_path_support_detailed,
    compute_graph_path_disagreement,
    compute_competing_answer_alternatives,
    compute_evidence_vn_entropy,
    compute_subgraph_informativeness,
    compute_subgraph_perturbation_stability,
    compute_subgraph_perturbation_stability_detailed,
    compute_support_entailment_uncertainty,
    compute_evidence_conflict_uncertainty,
)
from experiments.visualize_results import (
    plot_auroc_aurec_heatmaps,
    plot_metric_bar_charts,
    plot_metric_correlation_matrix,
    plot_reliability_diagrams,
    plot_compute_time_chart,
    plot_auroc_vs_compute_time,
    plot_complementarity_matrix,
    plot_query_type_stratification,
    plot_per_system_auroc_comparison,
    plot_metric_spearman_matrix,
)
from experiments.hop_stratified_analysis import run_hop_stratified_analysis

import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ── Run nomenclature ─────────────────────────────────────────────────────────
def _sanitize_run_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    token = re.sub(r"-{2,}", "-", token).strip("-")
    return token or "unknown"


def _dataset_run_token(datasets: List[str]) -> str:
    cleaned = [_sanitize_run_token(ds) for ds in (datasets or []) if str(ds).strip()]
    if not cleaned:
        return "dataset-unknown"
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) <= 3:
        return "-".join(cleaned)
    return f"multi-{len(cleaned)}ds"


def generate_run_id(
    datasets: List[str],
    num_samples: Optional[int],
    evaluation_mode: Optional[str],
    dataset_kg_scope: Optional[str] = None,
    rebuild_kg: bool = False,
) -> str:
    """Return a sortable, descriptive run ID."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_token = _dataset_run_token(datasets)
    sample_token = "nall" if num_samples is None else f"n{int(num_samples)}"
    mode_token = _sanitize_run_token(evaluation_mode or "default")
    parts = [ts, dataset_token, sample_token, mode_token]
    if dataset_kg_scope:
        parts.append(_sanitize_run_token(dataset_kg_scope))
    if rebuild_kg:
        parts.append("rebuildkg")
    return "-".join(parts)

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def update_runs_index(run_dir: Path, manifest: dict) -> None:
    """Append a one-line summary of this run to the sibling runs index."""
    index_path = run_dir.parent / "index.json"
    entries = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text())
        except Exception:
            entries = []
    entries.append({
        "run_id":     manifest["run_id"],
        "created_at": manifest["created_at"],
        "datasets":   manifest["datasets"],
        "model":      manifest.get("model", "unknown"),
        "git_commit": manifest.get("git_commit", "unknown"),
        "accuracy":   manifest.get("accuracy", {}),
    })
    index_path.write_text(json.dumps(entries, indent=2))


# Re-export from the dependency-light summary_utils module so callers that
# import this symbol from experiment.py continue to work, while tests can
# import from summary_utils directly without pulling in wandb / dotenv / neo4j.
from experiments.summary_utils import (
    accumulate_track_accuracy,
    compute_accuracy_breakdown,
    compute_hop_accuracy_breakdown,
    select_best_retrieval_configs,
)  # noqa: F401


class MIRAGEEvaluationPipeline:
    """Sequential evaluation pipeline for MIRAGE datasets"""

    EVALUATION_MODE_FULL_METRICS = "full_metrics"
    EVALUATION_MODE_ACCURACY_ONLY = "accuracy_only"
    EVALUATION_MODES = {
        EVALUATION_MODE_FULL_METRICS,
        EVALUATION_MODE_ACCURACY_ONLY,
    }
    DATASET_KG_SCOPE_EVALUATION_SUBSET = "evaluation_subset"
    DATASET_KG_SCOPE_FULL_DATASET = "full_dataset"
    DATASET_KG_SCOPES = {
        DATASET_KG_SCOPE_EVALUATION_SUBSET,
        DATASET_KG_SCOPE_FULL_DATASET,
    }

    UNCERTAINTY_METRIC_NAMES = [
        "semantic_entropy",
        "discrete_semantic_entropy",
        "sre_uq",
        "p_true",
        "selfcheckgpt",
        "vn_entropy",
        "sd_uq",
        "graph_path_support",              # structural — does KG path exist from question to answer?
        "graph_path_disagreement",         # structural — entropy over KG neighbor distribution
        "competing_answer_alternatives",   # structural — typed-relation cardinality of competing answers
        "evidence_vn_entropy",             # structural — combined evidence entropy + question alignment (novel)
        "subgraph_informativeness",          # structural — pre-generation answer-space concentration (novel)
        "subgraph_perturbation_stability",   # structural — perturbation fragility of answer-supporting paths (novel)
        "support_entailment_uncertainty",    # grounding — evidence-answer NLI entailment deficit (novel)
        "evidence_conflict_uncertainty",     # grounding — fraction of E-C conflicting chunk pairs (novel)
    ]

    UNCERTAINTY_METRIC_DEFAULTS = {
        "semantic_entropy": 0.0,
        "discrete_semantic_entropy": 0.0,
        "sre_uq": 0.0,
        "p_true": 0.5,
        "selfcheckgpt": 0.5,
        "vn_entropy": 0.0,
        "sd_uq": 0.5,
        "graph_path_support": 0.5,
        "graph_path_disagreement": 0.5,
        "competing_answer_alternatives": 0.0,
        "evidence_vn_entropy": 0.0,
        "subgraph_informativeness": 0.5,
        "subgraph_perturbation_stability": 0.5,
        "support_entailment_uncertainty": 0.5,
        "evidence_conflict_uncertainty": 0.0,
    }
    
    # Temperatures used for multi-temperature entropy sweeps.
    MULTI_TEMPERATURES: List[float] = [0.0, 0.5, 1.0]

    # Per-dataset max_hops for KG traversal and structural metrics.
    # MuSiQue requires up to 4 reasoning hops; all others are 2-hop.
    # pubmedqa / bioasq / realmedqa use max_hops=1: each question is grounded
    # in a single abstract or shared guideline corpus — iterative multi-hop
    # decomposition adds overhead and cross-passage noise without benefit.
    DATASET_MAX_HOPS: Dict[str, int] = {
        "musique":         4,
        "hotpotqa":        2,
        "hotpotqa_fullwiki": 2,
        "2wikimultihopqa": 2,
        "medhop":          3,
        "multihoprag":     4,
        "pubmedqa":        2,
        "realmedqa":       2,
        "bioasq":          2,
    }
    # Retrieval may need deeper traversals than structural uncertainty metrics.
    # On dense shared-corpus graphs such as MultiHopRAG, letting structural
    # metrics inherit the full retrieval hop budget can stall before the first
    # checkpointed question completes.
    DATASET_STRUCTURAL_METRIC_MAX_HOPS: Dict[str, int] = {
        "multihoprag": 2,
    }
    DEFAULT_MAX_HOPS: int = 2

    # Dataset → evaluation track.  Biomedical datasets test factual grounding;
    # Wikipedia multi-hop datasets test graph-structured multi-hop reasoning.
    # These are separate empirical claims and must be reported in separate result tables.
    DATASET_TRACKS: Dict[str, str] = {
        "pubmedqa":         "biomedical_grounding",
        "realmedqa":        "biomedical_grounding",
        "bioasq":           "biomedical_grounding",
        "medhop":           "biomedical_multihop_reasoning",
        "hotpotqa":         "multihop_reasoning",
        "hotpotqa_fullwiki": "multihop_reasoning",
        "2wikimultihopqa":  "multihop_reasoning",
        "musique":          "multihop_reasoning",
        "multihoprag":      "multihop_reasoning",
    }
    DEFAULT_TRACK: str = "other"
    # Datasets whose KG passages are scoped per-question (bundle or abstract contract).
    # For these, chunk retrieval is filtered by questionId to prevent cross-question contamination.
    # Shared-corpus datasets (bioasq, multihoprag, realmedqa, hotpotqa_fullwiki)
    # are NOT in this set.
    QUESTION_SCOPED_DATASETS: frozenset = frozenset({
        "pubmedqa",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
    })
    KG_BUILD_FINGERPRINT_VERSION: int = 2
    KG_BUILDER_NAME: str = "passage_aware_ontology_guided"
    KG_CHUNK_SIZE: int = 1500
    KG_CHUNK_OVERLAP: int = 200
    RETRIEVAL_STUDY_PROFILES: frozenset = frozenset({"small", "final_pair", "strict_entity"})
    KG_BUILDER_PROFILES: frozenset = frozenset({"auto", "full", "lightweight"})
    KG_BUILDER_PROFILE_ALIASES: Dict[str, str] = {"sota": "full"}

    QUESTION_SCOPED_MULTIHOP_DATASETS: frozenset = frozenset({
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
    })

    @classmethod
    def normalize_kg_builder_profile(cls, profile: Optional[str]) -> str:
        """Map legacy profile names onto the canonical public set."""
        normalized = str(profile or "auto").strip().lower()
        return cls.KG_BUILDER_PROFILE_ALIASES.get(normalized, normalized)

    RETRIEVAL_STUDY_SMALL_VARIANTS: Tuple[Dict[str, Any], ...] = (
        {
            "name": "dense_floor",
            "retrieval_stack": {
                "query_fusion": False,
                "late_interaction": False,
                "first_stage_late_interaction": False,
                "reranker": False,
            },
            "kg_system": {
                "retrieval_mode": "vector_only",
                "use_rfge": False,
            },
        },
        {
            "name": "modern_vector",
            "retrieval_stack": {
                "query_fusion": True,
                "late_interaction": True,
                "first_stage_late_interaction": True,
                "reranker": True,
            },
            "kg_system": {
                "retrieval_mode": "vector_only",
                "use_rfge": False,
            },
        },
        {
            "name": "kg_entity_first",
            "retrieval_stack": {
                "query_fusion": True,
                "late_interaction": True,
                "first_stage_late_interaction": True,
                "reranker": True,
            },
            "kg_system": {
                "retrieval_mode": "entity_first",
                "use_rfge": False,
            },
        },
        {
            "name": "kg_rfge",
            "retrieval_stack": {
                "query_fusion": True,
                "late_interaction": True,
                "first_stage_late_interaction": True,
                "reranker": True,
            },
            "kg_system": {
                "retrieval_mode": "rfge",
                "use_rfge": True,
            },
        },
        {
            "name": "kg_hybrid",
            "retrieval_stack": {
                "query_fusion": True,
                "late_interaction": True,
                "first_stage_late_interaction": True,
                "reranker": True,
            },
            "kg_system": {
                "retrieval_mode": "hybrid_auto",
                "use_rfge": True,
            },
        },
    )
    RETRIEVAL_STUDY_FINAL_PAIR_VARIANTS: Tuple[Dict[str, Any], ...] = tuple(
        {
            **variant,
            "executed_systems": (
                ["vanilla_rag"]
                if variant["name"] == "dense_floor"
                else ["kg_rag"]
            ),
        }
        for variant in RETRIEVAL_STUDY_SMALL_VARIANTS
        if variant["name"] in {"dense_floor", "kg_entity_first"}
    )
    RETRIEVAL_STUDY_STRICT_ENTITY_VARIANTS: Tuple[Dict[str, Any], ...] = (
        {
            "name": "kg_strict_entity_first",
            "retrieval_stack": {
                "query_fusion": False,
                "late_interaction": False,
                "first_stage_late_interaction": False,
                "reranker": False,
            },
            "kg_system": {
                "retrieval_mode": "entity_first",
                "use_rfge": False,
                "use_per_entity_ann": False,
                "allow_vector_augmentation": False,
                "allow_vector_fallback": False,
            },
            "kg_generation": {
                "allow_decomposition": False,
                "runtime_guardrail": False,
            },
            "executed_systems": ["kg_rag"],
        },
    )

    def __init__(
        self,
        num_samples: int = None,
        subset_seed: int = 42,
        entropy_samples: int = 5,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        eval_configs: Optional[List[Dict[str, Any]]] = None,
        rebuild_kg: bool = False,
        max_kg_contexts: int = None,
        use_llm_judge: bool = True,
        multi_temperature: bool = False,
        judge_provider: str = None,
        judge_model: str = None,
        evaluation_mode: str = EVALUATION_MODE_FULL_METRICS,
        dataset_kg_scope: str = DATASET_KG_SCOPE_EVALUATION_SUBSET,
        allow_gold_evidence_contexts: bool = False,
        retrieval_study: Optional[str] = None,
        kg_builder_profile: str = "auto",
    ):
        if evaluation_mode not in self.EVALUATION_MODES:
            raise ValueError(
                f"Unsupported evaluation_mode='{evaluation_mode}'. "
                f"Choose one of: {sorted(self.EVALUATION_MODES)}"
            )
        if dataset_kg_scope not in self.DATASET_KG_SCOPES:
            raise ValueError(
                f"Unsupported dataset_kg_scope='{dataset_kg_scope}'. "
                f"Choose one of: {sorted(self.DATASET_KG_SCOPES)}"
            )
        kg_builder_profile = self.normalize_kg_builder_profile(kg_builder_profile)
        if kg_builder_profile not in self.KG_BUILDER_PROFILES:
            raise ValueError(
                f"Unsupported kg_builder_profile='{kg_builder_profile}'. "
                f"Choose one of: {sorted(self.KG_BUILDER_PROFILES)}"
            )
        self.num_samples = num_samples  # None means use all questions
        self.subset_seed = int(subset_seed)
        self.entropy_samples = max(1, min(entropy_samples, 20))  # safety cap
        self.llm_provider_name = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.rebuild_kg = rebuild_kg  # Force rebuild even if KG exists
        self.use_llm_judge = use_llm_judge
        self.evaluation_mode = evaluation_mode
        self.compute_metrics = evaluation_mode == self.EVALUATION_MODE_FULL_METRICS
        self.multi_temperature = bool(multi_temperature and self.compute_metrics)
        self.dataset_kg_scope = dataset_kg_scope
        self.allow_gold_evidence_contexts = bool(allow_gold_evidence_contexts)
        self.retrieval_study = str(retrieval_study or "").strip()
        self.kg_builder_profile = kg_builder_profile
        self._llm_judge_cache: Dict[str, bool] = {}  # cache by (question, expected, response)
        self.max_kg_contexts = max_kg_contexts  # Cap context passages fed into KG build
        # Judge can be a different model/provider from the generation model to avoid
        # circular evaluation where the same model judges its own outputs.
        self.judge_provider_name = judge_provider or llm_provider
        self.judge_model = judge_model or llm_model

        # Retrieval/eval configurations (supports per-config comparisons)
        self.eval_configs = eval_configs or [
            {
                "name": "default",
                "similarity_threshold": 0.1,
                "max_chunks": 10,
            }
        ]

        # Initialize generation LLM
        provider = get_model_provider(self.llm_provider_name, model=self.llm_model)
        self._base_llm_provider = provider
        self.llm = LangChainRunnableAdapter(
            provider, model=self.llm_model, temperature=self.temperature
        )
        # Keep KG extraction deterministic so graph quality is stable across runs.
        self.kg_llm_provider = TemperatureLockedProvider(provider, temperature=0.0)
        # Accuracy measurement uses temperature=0.0 so correctness labels are
        # reproducible across runs regardless of the sampling temperature used
        # for uncertainty estimation.
        self.accuracy_llm = LangChainRunnableAdapter(
            provider, model=self.llm_model, temperature=0.0
        )
        # Judge uses a potentially different provider/model to avoid circular evaluation
        if self.judge_provider_name == self.llm_provider_name and self.judge_model == self.llm_model:
            self.judge_llm_provider = provider
        else:
            self.judge_llm_provider = get_model_provider(self.judge_provider_name,
                                                          model=self.judge_model)
        
        # Get embedding provider for consistent use across all components
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
        
        # RAG systems - use explicit embedding model for consistency
        self.vanilla_rag = VanillaRAGSystem(embedding_model=self.embedding_provider)
        self.kg_rag = EnhancedRAGSystem(embedding_model=self.embedding_provider)
        
        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        # W&B run
        self.wandb_run = None
        self._dataset_cache: Dict[str, Tuple[List[Any], List[Any]]] = {}
        self._selection_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _env_bool(enabled: bool) -> str:
        return "1" if bool(enabled) else "0"

    @staticmethod
    def _retrieval_temperature_suffix(value: float) -> str:
        return f"rt{str(float(value)).replace('.', 'p')}"

    @classmethod
    def build_retrieval_study_eval_configs(
        cls,
        *,
        profile: str,
        similarity_thresholds: List[float],
        max_chunks_values: List[int],
        retrieval_temperature_values: List[float],
        retrieval_shortlist_factor: int,
    ) -> List[Dict[str, Any]]:
        if profile not in cls.RETRIEVAL_STUDY_PROFILES:
            raise ValueError(
                f"Unsupported retrieval study profile={profile!r}. "
                f"Choose one of {sorted(cls.RETRIEVAL_STUDY_PROFILES)}."
            )

        if profile == "small":
            variants = cls.RETRIEVAL_STUDY_SMALL_VARIANTS
        elif profile == "final_pair":
            variants = cls.RETRIEVAL_STUDY_FINAL_PAIR_VARIANTS
        elif profile == "strict_entity":
            variants = cls.RETRIEVAL_STUDY_STRICT_ENTITY_VARIANTS
        else:
            variants = ()

        configs: List[Dict[str, Any]] = []
        for threshold in similarity_thresholds:
            for max_chunks in max_chunks_values:
                for retrieval_temperature in retrieval_temperature_values:
                    for variant in variants:
                        name = (
                            f"{variant['name']}_thr{threshold:g}_k{int(max_chunks)}_"
                            f"{cls._retrieval_temperature_suffix(retrieval_temperature)}"
                        )
                        configs.append({
                            "name": name,
                            "similarity_threshold": float(threshold),
                            "max_chunks": int(max_chunks),
                            "retrieval_temperature": float(retrieval_temperature),
                            "retrieval_shortlist_factor": int(retrieval_shortlist_factor),
                            "retrieval_variant": str(variant["name"]),
                            "retrieval_stack": dict(variant.get("retrieval_stack", {})),
                            "kg_system": dict(variant.get("kg_system", {})),
                            "kg_generation": dict(variant.get("kg_generation", {})),
                            "executed_systems": list(
                                variant.get("executed_systems", ["vanilla_rag", "kg_rag"])
                            ),
                        })
        return configs

    @staticmethod
    @contextmanager
    def _temporary_env(overrides: Dict[str, Any]):
        restore: Dict[str, Optional[str]] = {}
        for key, value in (overrides or {}).items():
            restore[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[str(key)] = str(value)
        try:
            yield
        finally:
            for key, old_value in restore.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    def _retrieval_env_overrides(self, config: Optional[Dict[str, Any]]) -> Dict[str, str]:
        stack = dict((config or {}).get("retrieval_stack") or {})
        overrides: Dict[str, str] = {}
        if "query_fusion" in stack:
            overrides["ONTOGRAPHRAG_QUERY_FUSION"] = self._env_bool(stack["query_fusion"])
        if "late_interaction" in stack:
            overrides["ONTOGRAPHRAG_LATE_INTERACTION"] = self._env_bool(stack["late_interaction"])
        if "first_stage_late_interaction" in stack:
            overrides["ONTOGRAPHRAG_FIRST_STAGE_LATE_INTERACTION"] = self._env_bool(
                stack["first_stage_late_interaction"]
            )
        if "reranker" in stack:
            overrides["ONTOGRAPHRAG_RERANKER"] = self._env_bool(stack["reranker"])
        if stack.get("retrieval_profile"):
            overrides["ONTOGRAPHRAG_RETRIEVAL_PROFILE"] = str(stack["retrieval_profile"])
        return overrides

    @staticmethod
    def _executed_system_map(config: Optional[Dict[str, Any]]) -> Dict[str, bool]:
        raw = (config or {}).get("executed_systems")
        if not raw:
            return {"vanilla_rag": True, "kg_rag": True}

        enabled = {"vanilla_rag": False, "kg_rag": False}
        for system_name in raw:
            if system_name in enabled:
                enabled[system_name] = True
        if not any(enabled.values()):
            return {"vanilla_rag": True, "kg_rag": True}
        return enabled

    @staticmethod
    def _execution_signature(config: Optional[Dict[str, Any]]) -> str:
        enabled = MIRAGEEvaluationPipeline._executed_system_map(config)
        active = [name.replace("_rag", "") for name, is_enabled in enabled.items() if is_enabled]
        token = re.sub(r"[^a-zA-Z0-9._-]+", "_", "_".join(active) or "all_systems")
        return token.strip("_") or "all_systems"

    @staticmethod
    def _compute_kg_routing_distribution(details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise KG retrieval routing so dense fallback is auditable.

        The KG system can answer through pure entity-first traversal, RFGE, or a
        semantic/vector fallback.  Output-side determinism claims should be
        interpreted against this distribution rather than assumed globally.
        """
        routes: Dict[str, int] = {}
        reasons: Dict[str, int] = {}
        total = 0

        for row in details or []:
            if bool(row.get("kg_generation_failed", False)):
                continue
            total += 1
            route = str(row.get("kg_retrieval_route") or "unknown").strip() or "unknown"
            reason = str(row.get("kg_route_reason") or "unknown").strip() or "unknown"
            routes[route] = routes.get(route, 0) + 1
            reasons[reason] = reasons.get(reason, 0) + 1

        def _pct(count: int) -> float:
            return float(count / total) if total else 0.0

        dense_fallback_routes = {"semantic_only", "vector_only", "dense_fallback"}
        graph_routes = {"entity_first", "rfge", "hybrid", "hybrid_vector_primary"}
        dense_fallback_count = sum(routes.get(r, 0) for r in dense_fallback_routes)
        graph_count = sum(routes.get(r, 0) for r in graph_routes)

        return {
            "n": total,
            "routes": {
                route: {"count": count, "pct": _pct(count)}
                for route, count in sorted(routes.items())
            },
            "route_reasons": {
                reason: {"count": count, "pct": _pct(count)}
                for reason, count in sorted(reasons.items())
            },
            "pure_entity_first_rate": _pct(routes.get("entity_first", 0)),
            "graph_route_rate": _pct(graph_count),
            "dense_fallback_rate": _pct(dense_fallback_count),
            "unknown_route_rate": _pct(routes.get("unknown", 0)),
        }

    def _build_rag_systems_for_config(
        self,
        config: Optional[Dict[str, Any]],
    ) -> Tuple[VanillaRAGSystem, EnhancedRAGSystem]:
        kg_system = dict((config or {}).get("kg_system") or {})
        enhanced_kwargs = {
            "embedding_model": self.embedding_provider,
            "retrieval_mode": kg_system.get("retrieval_mode", "hybrid_auto"),
            "use_per_entity_ann": bool(kg_system.get("use_per_entity_ann", True)),
            "use_node_specificity": bool(kg_system.get("use_node_specificity", True)),
            "use_ppr_scoring": bool(kg_system.get("use_ppr_scoring", True)),
            "use_rfge": bool(kg_system.get("use_rfge", True)),
            "use_evidence_block": bool(kg_system.get("use_evidence_block", True)),
            "allow_vector_augmentation": bool(kg_system.get("allow_vector_augmentation", True)),
            "allow_vector_fallback": bool(kg_system.get("allow_vector_fallback", True)),
        }
        if "max_chunks_per_passage" in kg_system:
            enhanced_kwargs["max_chunks_per_passage"] = int(kg_system["max_chunks_per_passage"])
        if "traversal_chunk_min_sim" in kg_system:
            enhanced_kwargs["traversal_chunk_min_sim"] = float(kg_system["traversal_chunk_min_sim"])
        vanilla = VanillaRAGSystem(embedding_model=self.embedding_provider)
        kg = EnhancedRAGSystem(**enhanced_kwargs)
        return vanilla, kg

    @staticmethod
    def _is_insufficient_information_label(text: str) -> bool:
        """Return True when a response is the canonical no-answer label only."""
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        normalized = re.sub(r"[\s\.\!\?]+$", "", normalized)
        return normalized in {
            "insufficient information",
            "context insufficient",
        }

    def _is_generation_failure(
        self,
        result: Dict[str, Any],
        response: str,
        *,
        expected_answer: Optional[str] = None,
    ) -> bool:
        """Heuristic detection for provider/runtime failures in generated outputs."""
        if not isinstance(result, dict) or not result:
            return True
        if result.get("error"):
            return True

        text = str(response or "").strip().lower()
        if not text:
            return True

        failure_markers = [
            "an error occurred while generating the response",
            "too many requests",
            "rate limit",
            "429",
            "timed out",
        ]
        if any(marker in text for marker in failure_markers):
            return True

        # Treat the exact no-answer label as a generation/abstention failure only
        # when the gold answer is *not* also the no-answer label. This preserves
        # valid "Insufficient Information" answers on datasets that explicitly use
        # that label, while still excluding spurious abstentions from clean accuracy.
        if self._is_insufficient_information_label(text):
            return not self._is_insufficient_information_label(expected_answer or "")

        # Semantic refusals — model explicitly abstains due to missing context.
        # These are not provider errors but should be treated as failures for
        # accuracy accounting so they don't silently count as wrong answers.
        refusal_markers = [
            "the context is insufficient",
            "context does not provide",
            "context provided does not",
            "no information provided",
            "not enough information",
            "i cannot answer",
            "i don't have enough",
            "i do not have enough",
            "cannot be determined from",
            "not mentioned in the",
            "not found in the",
        ]
        return any(marker in text for marker in refusal_markers)

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for robust lexical matching."""
        t = str(text or "").lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _contains_expected_text(self, haystack: str, needle: str) -> bool:
        """Robust lexical match with token and typo-tolerant fallbacks."""
        hay = self._normalize_for_matching(haystack)
        ned = self._normalize_for_matching(needle)

        if not hay or not ned:
            return False

        # Very short labels (e.g., yes/no, gene abbreviations): exact token only.
        if len(ned) <= 3:
            return bool(re.search(rf"\b{re.escape(ned)}\b", hay))

        # Exact phrase match first.
        if re.search(rf"\b{re.escape(ned)}\b", hay):
            return True

        hay_tokens = hay.split()
        needle_tokens = ned.split()
        if not hay_tokens or not needle_tokens:
            return False

        # Single-word fuzzy fallback (typo-tolerant, but strict).
        if len(needle_tokens) == 1:
            target = needle_tokens[0]
            return any(SequenceMatcher(None, target, tok).ratio() >= 0.9 for tok in hay_tokens)

        # Multi-word fallback by token coverage.
        matched = sum(1 for tok in needle_tokens if tok in hay_tokens)
        coverage = matched / max(1, len(needle_tokens))

        # Slightly looser threshold for longer phrases to allow minor paraphrase.
        if len(needle_tokens) >= 4:
            return coverage >= 0.75
        return coverage >= 0.85

    def _get_normalized_dataset(self, dataset_name: str):
        """Load dataset through adapters and cache normalized records.

        Returns (inference_records, gold_records, leakage_detected).
        leakage_detected=True means gold answers appear verbatim in contexts;
        the experiment continues but all results for this dataset are flagged
        so the reader knows context-conditioned scores may be inflated.
        """
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]

        inference_records, gold_records = normalize_dataset(dataset_name)
        profile = self._dataset_corpus_profile(dataset_name)
        shared_passages = build_global_corpus_passages(dataset_name)
        shared_corpus_overrides_gold_evidence = (
            profile.get("question_context_role") == "gold_evidence"
            and shared_passages is not None
        )

        if shared_corpus_overrides_gold_evidence:
            leakage_detected = False
            logging.info(
                "Dataset '%s' exposes oracle per-question evidence, but a shared retrieval corpus "
                "is configured. Skipping gold-evidence leakage warnings for the question contexts.",
                dataset_name,
            )
        else:
            leakage_detected = not validate_no_leakage(inference_records, gold_records)
            if leakage_detected:
                if profile.get("question_context_role") == "gold_evidence":
                    logging.error(
                        "Gold-evidence leakage detected for dataset '%s'. "
                        "Per-question contexts contain oracle support text; retrieval benchmarking is invalid "
                        "unless a separate shared corpus is configured.",
                        dataset_name,
                    )
                else:
                    logging.warning(
                        "Answer text appears inside some provided contexts for dataset '%s'. "
                        "This dataset should be interpreted as a closed-corpus / bundled-context benchmark, "
                        "not open-corpus retrieval.",
                        dataset_name,
                    )

        self._dataset_cache[dataset_name] = (inference_records, gold_records, leakage_detected)
        return inference_records, gold_records, leakage_detected

    def _build_eval_records(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Create aligned inference/eval records from normalized adapter outputs."""
        inference_records, gold_records, _ = self._get_normalized_dataset(dataset_name)
        gold_by_id = {g.id: g for g in gold_records}
        has_shared_corpus = build_global_corpus_passages(dataset_name) is not None

        eval_records: List[Dict[str, Any]] = []
        for inf in inference_records:
            gold = gold_by_id.get(inf.id)
            if not gold:
                continue

            contexts = [c.strip() for c in (inf.contexts or []) if str(c).strip()]
            if not contexts and not has_shared_corpus:
                # Skip question-only datasets unless a dataset-level shared corpus is
                # configured for retrieval/KG construction.
                continue

            if not (gold.short_answer or "").strip():
                # Skip questions whose gold answer is empty — cannot evaluate correctness.
                # Affects BioASQ list/summary types which have no short_answer.
                logging.debug(
                    "Skipping record %s (dataset=%s): empty gold short_answer",
                    inf.id, dataset_name,
                )
                continue

            eval_records.append({
                "id": inf.id,
                "question": inf.question,
                "expected_answer": gold.short_answer,
                "aliases": gold.aliases or [],
                "contexts": contexts,
                "options": inf.options or {},
                "task_type": inf.task_type,
            })

        return eval_records

    def _selection_dir(self) -> Path:
        """Directory where persisted dataset question subsets are stored."""
        path = Path("results") / "selections"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _selection_path(self, dataset_name: str) -> Path:
        """Persisted question subset file for one dataset."""
        return selection_file_path(
            self._selection_dir(),
            dataset_name,
            num_samples=self.num_samples,
            subset_seed=self.subset_seed,
        )

    def _subset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Return cached subset metadata for a dataset, if already resolved."""
        return self._selection_cache.get(dataset_name, {})

    def _dataset_corpus_profile(self, dataset_name: str) -> Dict[str, Any]:
        """Return static corpus metadata describing what a dataset's contexts mean."""
        return get_dataset_corpus_profile(dataset_name)

    def _infer_hop_count(
        self,
        dataset_name: str,
        question_id: str,
        raw_question: Dict[str, Any],
    ) -> Optional[int]:
        """
        Infer reasoning hop count from raw dataset metadata.

        Preference order:
          1. Explicit decomposition length from the raw record
          2. Dataset-specific structured evidence count when it directly reflects
             the question's reasoning chain
          3. ID-prefix fallback like ``3hop1__...`` for MuSiQue-style IDs
        """
        return infer_hop_count_from_raw(dataset_name, question_id, raw_question)

    def _hop_bucket_label(self, hop_count: Optional[int]) -> Optional[str]:
        """Return a compact hop bucket label for a positive hop count."""
        if hop_count is None or hop_count <= 0:
            return None
        return f"{hop_count}-hop" if hop_count <= 4 else "5+-hop"

    def _get_dataset_corpus_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Read corpus-construction metadata from the existing dataset KG, if present."""
        driver = self._get_neo4j_driver()
        try:
            with driver.session() as session:
                row = session.run(
                    """
                    MATCH (d:Document {kgName: $kg_name})
                    RETURN
                        d.usesGlobalCorpus AS usesGlobalCorpus,
                        d.corpusSource AS corpusSource,
                        d.questionContextRole AS questionContextRole,
                        d.datasetKgScope AS datasetKgScope,
                        d.selectionKey AS selectionKey,
                        d.subsetId AS subsetId,
                        d.subsetTag AS subsetTag,
                        d.contentHash AS contentHash,
                        d.schemaHash AS schemaHash,
                        d.kgBuildFingerprintVersion AS kgBuildFingerprintVersion,
                        d.kgBuilder AS kgBuilder,
                        d.kgBuilderProfile AS kgBuilderProfile,
                        d.kgBuilderProfileRequested AS kgBuilderProfileRequested,
                        d.kgChunkSize AS kgChunkSize,
                        d.kgChunkOverlap AS kgChunkOverlap,
                        d.kgExtractionProvider AS kgExtractionProvider,
                        d.kgExtractionModel AS kgExtractionModel,
                        d.kgEmbeddingProvider AS kgEmbeddingProvider
                    LIMIT 1
                    """,
                    {"kg_name": dataset_name},
                ).single()
                if not row:
                    return {}
                return {
                    "uses_global_corpus": row.get("usesGlobalCorpus"),
                    "corpus_source": row.get("corpusSource"),
                    "question_context_role": row.get("questionContextRole"),
                    "dataset_kg_scope": row.get("datasetKgScope"),
                    "selection_key": row.get("selectionKey"),
                    "subset_id": row.get("subsetId"),
                    "subset_tag": row.get("subsetTag"),
                    "content_hash": row.get("contentHash"),
                    "schema_hash": row.get("schemaHash"),
                    "kg_build_fingerprint_version": row.get("kgBuildFingerprintVersion"),
                    "kg_builder": row.get("kgBuilder"),
                    "kg_builder_profile": row.get("kgBuilderProfile"),
                    "kg_builder_profile_requested": row.get("kgBuilderProfileRequested"),
                    "kg_chunk_size": row.get("kgChunkSize"),
                    "kg_chunk_overlap": row.get("kgChunkOverlap"),
                    "kg_extraction_provider": row.get("kgExtractionProvider"),
                    "kg_extraction_model": row.get("kgExtractionModel"),
                    "kg_embedding_provider": row.get("kgEmbeddingProvider"),
                }
        except Exception as e:
            logging.warning(
                "[corpus_policy] Failed to read existing corpus metadata for %s: %s",
                dataset_name,
                e,
            )
            return {}
        finally:
            driver.close()

    def _compute_dataset_kg_content_hash(
        self,
        dataset_name: str,
        *,
        passages: List[Any],
        build_meta: Dict[str, Any],
    ) -> str:
        """Create a deterministic fingerprint for the dataset KG build contract."""
        normalized_passages = [
            {
                "dataset": str(getattr(passage, "dataset", dataset_name)),
                "question_id": str(getattr(passage, "question_id", "")),
                "passage_index": int(getattr(passage, "passage_index", 0)),
                "text": str(getattr(passage, "text", "")).strip(),
            }
            for passage in passages
        ]
        payload = {
            "dataset": dataset_name,
            "datasetKgScope": build_meta.get("datasetKgScope", ""),
            "usesGlobalCorpus": bool(build_meta.get("usesGlobalCorpus", False)),
            "corpusSource": build_meta.get("corpusSource", ""),
            "questionContextRole": build_meta.get("questionContextRole", ""),
            "selectionKey": (
                ""
                if build_meta.get("usesGlobalCorpus")
                else build_meta.get("selectionKey", "")
            ),
            "builder": {
                "version": self.KG_BUILD_FINGERPRINT_VERSION,
                "name": self.KG_BUILDER_NAME,
                "profile": build_meta.get("kgBuilderProfile", "full"),
                "chunkSize": self.KG_CHUNK_SIZE,
                "chunkOverlap": self.KG_CHUNK_OVERLAP,
                "extractionProvider": self.llm_provider_name,
                "extractionModel": self.llm_model,
                "embeddingProvider": self.embedding_provider,
            },
            "passages": normalized_passages,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _resolve_kg_builder_profile(self, dataset_name: str) -> str:
        """Choose full-vs-lightweight KG construction for this dataset/run."""
        requested_profile = self.normalize_kg_builder_profile(
            getattr(self, "kg_builder_profile", "auto")
        )
        if requested_profile != "auto":
            return requested_profile

        # Accuracy-only retrieval sweeps need a cheap graph to compare retrieval
        # behavior. Full hallucination/uncertainty runs should keep the full KG
        # construction stack so graph quality is not the confound.
        if (
            getattr(self, "evaluation_mode", self.EVALUATION_MODE_ACCURACY_ONLY) == self.EVALUATION_MODE_ACCURACY_ONLY
            and dataset_name in self.QUESTION_SCOPED_MULTIHOP_DATASETS
        ):
            return "lightweight"
        return "full"

    def _is_biomedical_dataset(self, dataset_name: str) -> bool:
        """Return True for datasets where biomedical entity normalization is useful."""
        track = self.DATASET_TRACKS.get(dataset_name, self.DEFAULT_TRACK)
        return str(track).startswith("biomedical")

    def _kg_builder_kwargs_for_profile(
        self,
        dataset_name: str,
        builder_profile: str,
    ) -> Dict[str, Any]:
        """Return constructor kwargs for the requested KG builder profile."""
        if builder_profile == "lightweight":
            return {
                "enable_anchor_constrained_extraction": False,
                "enable_self_reflection": False,
                "enable_anchor_coverage_supplement": False,
                "enable_cross_passage_relation_recovery": False,
            }

        if builder_profile == "full":
            # Full builder profile:
            # - majority-vote extraction over 3 samples
            # - richer schema few-shot guidance
            # - low-confidence triple reverification
            # - optional UMLS-backed biomedical normalization
            # - soft canonicalisation + graph enrichment / repair
            return {
                "enable_anchor_constrained_extraction": True,
                "enable_self_reflection": True,
                "enable_anchor_coverage_supplement": True,
                "enable_cross_passage_relation_recovery": True,
                "self_consistency_n": 3,
                "few_shot_example_count": 4,
                "min_triple_confidence": 0.2,
                "relationship_type_similarity_threshold": 0.7,
                "enable_low_confidence_triple_reverification": True,
                "low_confidence_reverify_threshold": 0.55,
                "enable_umls_linking": self._is_biomedical_dataset(dataset_name),
                "enable_soft_entity_linking": True,
                "enable_fragmentation_repair": True,
                "enable_graph_summaries": True,
                "enable_claim_extraction": True,
            }

        return {}

    def _prepare_dataset_kg_contract(
        self,
        dataset_name: str,
        *,
        force_resample: bool = False,
    ) -> Dict[str, Any]:
        """
        Resolve the passages and metadata that define the dataset-scoped KG.

        This is shared between KG construction and KG-reuse compatibility checks so
        reruns can safely reuse a compatible graph without forcing rebuilds.
        """
        selected_eval_records = self._selected_eval_records(
            dataset_name,
            force_resample=force_resample,
        )
        inference_records, _, _ = self._get_normalized_dataset(dataset_name)
        selected_question_ids = [str(record["id"]) for record in selected_eval_records]
        if not selected_question_ids:
            raise ValueError(f"No evaluable questions found for {dataset_name}")

        subset_meta = self._subset_metadata(dataset_name)
        corpus_profile = self._dataset_corpus_profile(dataset_name)
        resolved_kg_builder_profile = self._resolve_kg_builder_profile(dataset_name)

        inference_by_id = {str(record.id): record for record in inference_records}
        evaluable_inference_records: List[Any] = []
        missing_eval_ids: List[str] = []
        for question_id in [str(record["id"]) for record in self._build_eval_records(dataset_name)]:
            inference_record = inference_by_id.get(question_id)
            if inference_record is None:
                missing_eval_ids.append(question_id)
                continue
            evaluable_inference_records.append(inference_record)
        if missing_eval_ids:
            preview = ", ".join(missing_eval_ids[:5])
            raise ValueError(
                "Adapter-normalized eval IDs missing from inference records for "
                f"{dataset_name}: {preview}"
            )

        if self.dataset_kg_scope == self.DATASET_KG_SCOPE_FULL_DATASET:
            records_for_kg = evaluable_inference_records
        else:
            missing_selected_ids = [
                question_id for question_id in selected_question_ids
                if question_id not in inference_by_id
            ]
            if missing_selected_ids:
                preview = ", ".join(missing_selected_ids[:5])
                raise ValueError(
                    "Selected question IDs missing from inference records for "
                    f"{dataset_name}: {preview}"
                )
            records_for_kg = [
                inference_by_id[question_id]
                for question_id in selected_question_ids
            ]

        passages = build_global_corpus_passages(dataset_name)
        uses_global_corpus = passages is not None
        if uses_global_corpus:
            corpus_source = "shared_corpus"
        else:
            question_context_role = corpus_profile.get("question_context_role")
            if question_context_role == "gold_evidence" and not self.allow_gold_evidence_contexts:
                raise ValueError(
                    f"Refusing to build KG for {dataset_name} from per-question gold evidence "
                    "without --allow-gold-evidence-contexts."
                )
            passages = build_passage_corpus(
                records_for_kg,
                dedupe_across_questions=(dataset_name not in self.QUESTION_SCOPED_DATASETS),
            )
            corpus_source = (
                "gold_evidence_question_contexts"
                if question_context_role == "gold_evidence"
                else "question_contexts"
            )

        total_passages_before_cap = len(passages)
        if self.max_kg_contexts and len(passages) > self.max_kg_contexts:
            passages = passages[:self.max_kg_contexts]

        build_meta = {
            "datasetKgScope": self.dataset_kg_scope,
            "recordsAvailable": len(evaluable_inference_records),
            "recordsUsed": len(records_for_kg),
            "passagesBeforeCap": total_passages_before_cap,
            "passagesUsed": len(passages),
            "numSamples": self.num_samples if self.num_samples is not None else -1,
            "subsetSeed": self.subset_seed,
            "selectionFile": str(self._selection_path(dataset_name)),
            "selectedQuestionCount": len(selected_question_ids),
            "selectionKey": subset_meta.get("selection_key", ""),
            "subsetId": subset_meta.get("subset_id", ""),
            "subsetTag": subset_meta.get("subset_tag", ""),
            "usesGlobalCorpus": uses_global_corpus,
            "corpusSource": corpus_source,
            "questionContextRole": corpus_profile.get("question_context_role", ""),
            "requiresSharedCorpusForFairRetrieval": bool(
                corpus_profile.get("requires_shared_corpus_for_fair_retrieval", False)
            ),
            "globalCorpusPassages": total_passages_before_cap if uses_global_corpus else 0,
            "kgBuildFingerprintVersion": self.KG_BUILD_FINGERPRINT_VERSION,
            "kgBuilder": self.KG_BUILDER_NAME,
            "kgBuilderProfile": resolved_kg_builder_profile,
            "kgBuilderProfileRequested": self.kg_builder_profile,
            "kgChunkSize": self.KG_CHUNK_SIZE,
            "kgChunkOverlap": self.KG_CHUNK_OVERLAP,
            "kgExtractionProvider": self.llm_provider_name,
            "kgExtractionModel": self.llm_model,
            "kgEmbeddingProvider": self.embedding_provider,
        }
        build_meta["contentHash"] = self._compute_dataset_kg_content_hash(
            dataset_name,
            passages=passages,
            build_meta=build_meta,
        )

        return {
            "selected_eval_records": selected_eval_records,
            "selected_question_ids": selected_question_ids,
            "subset_meta": subset_meta,
            "corpus_profile": corpus_profile,
            "evaluable_inference_records": evaluable_inference_records,
            "records_for_kg": records_for_kg,
            "passages": passages,
            "total_passages_before_cap": total_passages_before_cap,
            "build_meta": build_meta,
        }

    @classmethod
    def _assess_dataset_kg_compatibility(
        cls,
        existing_meta: Dict[str, Any],
        expected_meta: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        return assess_dataset_kg_compatibility(
            existing_meta,
            expected_meta,
            evaluation_subset_scope=cls.DATASET_KG_SCOPE_EVALUATION_SUBSET,
        )

    def _enforce_dataset_corpus_policy(
        self,
        dataset_name: str,
        *,
        existing_chunk_count: int = 0,
    ) -> Tuple[bool, bool]:
        """
        Enforce fail-closed corpus rules before reusing or rebuilding a dataset KG.

        Returns:
            (allowed, force_rebuild_existing_kg)
        """
        profile = self._dataset_corpus_profile(dataset_name)
        question_context_role = profile.get("question_context_role", "retrieval_bundle")
        shared_passages = build_global_corpus_passages(dataset_name)
        has_shared_corpus = shared_passages is not None

        if has_shared_corpus and existing_chunk_count > 0:
            corpus_meta = self._get_dataset_corpus_metadata(dataset_name)
            if corpus_meta.get("uses_global_corpus") is not True:
                logging.warning(
                    "[corpus_policy] Existing KG for %s predates the shared-corpus guardrail "
                    "or was built from per-question evidence. Forcing rebuild.",
                    dataset_name,
                )
                return True, True

        if question_context_role == "gold_evidence" and not has_shared_corpus:
            if self.allow_gold_evidence_contexts:
                logging.warning(
                    "[corpus_policy] Dataset %s only exposes gold-evidence question contexts. "
                    "Proceeding because --allow-gold-evidence-contexts is set. "
                    "Treat this as controlled-evidence QA, not retrieval benchmarking.",
                    dataset_name,
                )
                return True, False

            logging.error(
                "[corpus_policy] Refusing to evaluate %s as a retrieval benchmark: "
                "its per-question contexts are gold evidence and no shared corpus is configured. "
                "Provide a real shared corpus under MIRAGE/rawdata/%s or rerun with "
                "--allow-gold-evidence-contexts for controlled-evidence QA only.",
                dataset_name,
                dataset_name,
            )
            return False, False

        return True, False

    def _resolve_selected_question_ids(
        self,
        dataset_name: str,
        *,
        require_existing: bool = False,
        force_resample: bool = False,
    ) -> List[str]:
        """
        Load or create the deterministic question subset for a dataset.

        The subset is always resolved from adapter-normalized evaluation records,
        so KG construction and downstream evaluation stay aligned to the same
        question IDs.
        """
        eval_records = self._build_eval_records(dataset_name)
        available_question_ids = [str(record["id"]) for record in eval_records]
        selection_path = self._selection_path(dataset_name)
        resolution = resolve_question_subset(
            dataset_name=dataset_name,
            available_question_ids=available_question_ids,
            num_samples=self.num_samples,
            subset_seed=self.subset_seed,
            selection_path=selection_path,
            require_existing=require_existing,
            force_resample=force_resample,
        )
        self._selection_cache[dataset_name] = resolution.payload
        for warning in resolution.warnings:
            logging.warning(warning)
        if resolution.created:
            logging.info(
                "Saved deterministic question subset for %s: %d/%d questions -> %s",
                dataset_name,
                len(resolution.question_ids),
                len(available_question_ids),
                selection_path,
            )
        else:
            logging.info(
                "Reusing deterministic question subset for %s: %d questions from %s",
                dataset_name,
                len(resolution.question_ids),
                selection_path,
            )
        return resolution.question_ids

    def _selected_eval_records(
        self,
        dataset_name: str,
        *,
        require_existing: bool = False,
        force_resample: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return adapter-normalized evaluation records for the persisted subset."""
        eval_records = self._build_eval_records(dataset_name)
        selected_question_ids = self._resolve_selected_question_ids(
            dataset_name,
            require_existing=require_existing,
            force_resample=force_resample,
        )
        by_id = {str(record["id"]): record for record in eval_records}
        missing_ids = [question_id for question_id in selected_question_ids if question_id not in by_id]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"Persisted selection for dataset '{dataset_name}' references IDs not "
                f"present in adapter-normalized eval records: {preview}"
            )
        return [by_id[question_id] for question_id in selected_question_ids]

    def _log_question_table_to_wandb(
        self,
        dataset_name: str,
        config_name: str,
        subset_tag: str,
        details: List[Dict[str, Any]],
    ):
        """Log per-question table (question + responses + metrics) to W&B."""
        if not self.wandb_run:
            return

        base_columns = [
            "dataset",
            "config",
            "question_id",
            "hop_count",
            "hop_bucket",
            "question",
            "expected",
            "vanilla_response",
            "kg_response",
            "vanilla_correct",
            "kg_correct",
        ]
        metric_columns = []
        if self.compute_metrics:
            metric_columns = [
                f"{sys}_{metric}"
                for metric in self.UNCERTAINTY_METRIC_NAMES
                for sys in ("vanilla", "kg")
            ]

        table = wandb.Table(columns=base_columns + metric_columns)

        for d in details:
            row = [
                dataset_name,
                config_name,
                d.get("question_id", ""),
                d.get("hop_count", None),
                d.get("hop_bucket", ""),
                d.get("question", ""),
                d.get("expected", ""),
                d.get("vanilla_response", ""),
                d.get("kg_response", ""),
                int(bool(d.get("vanilla_correct", False))),
                int(bool(d.get("kg_correct", False))),
            ]
            if self.compute_metrics:
                row.extend(
                    float(d.get(f"{sys}_{metric}", 0.0))
                    for metric in self.UNCERTAINTY_METRIC_NAMES
                    for sys in ("vanilla", "kg")
                )
            table.add_data(*row)

        self.wandb_run.log({
            f"tables/{dataset_name}/{subset_tag}/{config_name}/questions_and_responses": table
        })

    def _sanitize_for_filename(self, value: str) -> str:
        """Create a filesystem-safe token for filenames."""
        token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
        return token.strip("_") or "default"

    # Fields kept in the per-question outputs. Checkpoints and final question
    # JSONs both use this lean schema so recovery artifacts match what we
    # actually report.
    _QUESTION_OUTPUT_FIELDS = (
        # Identity
        ["question_id", "hop_count", "hop_bucket", "question", "expected", "task_type"]
        # Ground truth & responses
        + ["vanilla_correct", "kg_correct",
           "vanilla_response", "kg_response",
           "vanilla_generation_failed", "kg_generation_failed",
           "vanilla_system_skipped", "kg_system_skipped"]
        # Official-style answer metrics
        + ["vanilla_answer_em", "kg_answer_em",
           "vanilla_answer_f1", "kg_answer_f1"]
        # Retrieval metadata
        + [
            "grounding_quality", "seed_entity_count",
            "vanilla_search_method", "kg_search_method",
            "kg_retrieval_route", "kg_route_reason",
            "vanilla_retrieval_overlap", "kg_retrieval_overlap",
            "kg_retrieval_mode_config",
        ]
        # Per-sample graph-state diversity across the N uncertainty samples
        + [
            "kg_graph_state_sample_count",
            "kg_seed_entity_jaccard", "kg_path_jaccard",
            "kg_subgraph_jaccard", "kg_chunk_jaccard",
            "kg_seed_entity_entropy", "kg_seed_entity_entropy_norm",
            "kg_path_entropy", "kg_path_entropy_norm",
            "kg_subgraph_entropy", "kg_subgraph_entropy_norm",
            "kg_chunk_entropy", "kg_chunk_entropy_norm",
            "kg_dominant_seed_entity_id", "kg_dominant_seed_entity_fraction",
        ]
        # Canonical UQ metrics × 2 systems
        + [f"{sys}_{m}"
           for m in [
               "semantic_entropy", "discrete_semantic_entropy",
               "p_true", "selfcheckgpt",
               "sre_uq", "vn_entropy", "sd_uq",
               "graph_path_support", "graph_path_disagreement",
               "competing_answer_alternatives",
               "evidence_vn_entropy",
               "subgraph_informativeness",
               "subgraph_perturbation_stability",
               "subgraph_perturbation_stability_null_reason",
               "support_entailment_uncertainty",
               "evidence_conflict_uncertainty",
           ]
           for sys in ("vanilla", "kg")]
    )

    def _canonicalize_detail_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only the reported per-question fields."""
        kept = set(self._QUESTION_OUTPUT_FIELDS)
        return {k: v for k, v in row.items() if k in kept}

    def _write_question_details_file(
        self,
        dataset_name: str,
        config_name: str,
        subset_tag: str,
        details: List[Dict[str, Any]],
    ) -> str:
        """Persist per-question rows to an explicit local JSON file."""
        # Write inside the current run directory when available, fall back to legacy path
        if hasattr(self, "_run_dir"):
            output_dir = self._run_dir / "questions"
        else:
            output_dir = Path("results") / "questions"
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_dataset = self._sanitize_for_filename(dataset_name)
        safe_config = self._sanitize_for_filename(config_name)
        safe_subset = self._sanitize_for_filename(subset_tag)
        output_path = output_dir / f"{safe_dataset}_{safe_subset}_{safe_config}_questions.json"

        # Strip intermediate/auxiliary fields — keep only canonical metrics
        clean_details = [self._canonicalize_detail_row(row) for row in details]

        payload = {
            "dataset": dataset_name,
            "config": config_name,
            "subset_tag": subset_tag,
            "num_questions": len(clean_details),
            "questions": clean_details,
        }

        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)

        return str(output_path)

    def _log_config_summary_to_wandb(
        self,
        dataset_name: str,
        config_results: List[Dict[str, Any]],
    ):
        """Log per-config summary table and grouped bar charts to W&B."""
        if not self.wandb_run or not config_results:
            return

        summary_columns = [
            "dataset",
            "config",
            "vanilla_clean_accuracy",
            "kg_clean_accuracy",
            "clean_accuracy_gain_kg_minus_vanilla",
            "vanilla_answer_em",
            "kg_answer_em",
            "vanilla_answer_f1",
            "kg_answer_f1",
            "kg_pure_entity_first_rate",
            "kg_dense_fallback_rate",
            "kg_unknown_route_rate",
        ]
        if self.compute_metrics:
            summary_columns.extend([
                "vanilla_semantic_entropy",
                "kg_semantic_entropy",
                "vanilla_vn_entropy",
                "kg_vn_entropy",
                "vanilla_graph_path_support",
                "kg_graph_path_support",
            ])

        summary_table = wandb.Table(columns=summary_columns)

        for r in config_results:
            config = r.get("config", {})
            config_name = config.get("name", "default")

            v_acc = float(r.get("vanilla_accuracy", 0.0))
            k_acc = float(r.get("kg_accuracy", 0.0))
            row = [
                dataset_name,
                config_name,
                v_acc,
                k_acc,
                k_acc - v_acc,
                float(r.get("vanilla_answer_em", 0.0)),
                float(r.get("kg_answer_em", 0.0)),
                float(r.get("vanilla_answer_f1", 0.0)),
                float(r.get("kg_answer_f1", 0.0)),
                float(r.get("kg_pure_entity_first_rate", 0.0)),
                float(r.get("kg_dense_fallback_rate", 0.0)),
                float(r.get("kg_unknown_route_rate", 0.0)),
            ]
            if self.compute_metrics:
                row.extend([
                    float(r.get("vanilla_avg_semantic_entropy", 0.0)),
                    float(r.get("kg_avg_semantic_entropy", 0.0)),
                    float(r.get("vanilla_avg_vn_entropy", 0.0)),
                    float(r.get("kg_avg_vn_entropy", 0.0)),
                    float(r.get("vanilla_avg_graph_path_support", 0.5)),
                    float(r.get("kg_avg_graph_path_support", 0.5)),
                ])

            summary_table.add_data(*row)

        self.wandb_run.log({f"tables/{dataset_name}/config_summary": summary_table})

        # Grouped bar charts per measure by configuration
        if HAS_MATPLOTLIB:
            config_names = [r.get("config", {}).get("name", "default") for r in config_results]
            x = list(range(len(config_names)))
            width = 0.36

            metric_specs = [("clean_accuracy", "vanilla_accuracy", "kg_accuracy")]
            if self.compute_metrics:
                metric_specs.extend([
                    ("semantic_entropy", "vanilla_avg_semantic_entropy", "kg_avg_semantic_entropy"),
                    ("vn_entropy", "vanilla_avg_vn_entropy", "kg_avg_vn_entropy"),
                    ("graph_path_support", "vanilla_avg_graph_path_support", "kg_avg_graph_path_support"),
                ])

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for i, (title, v_key, k_key) in enumerate(metric_specs):
                ax = axes[i]
                v_vals = [float(r.get(v_key, 0.0)) for r in config_results]
                k_vals = [float(r.get(k_key, 0.0)) for r in config_results]

                ax.bar([p - width/2 for p in x], v_vals, width, label="Vanilla", color="#1f77b4")
                ax.bar([p + width/2 for p in x], k_vals, width, label="KG-RAG", color="#ff7f0e")
                ax.set_title(title.replace("_", " ").title())
                ax.set_xticks(x)
                ax.set_xticklabels(config_names, rotation=20, ha="right")
                ax.grid(axis="y", alpha=0.3)
                ax.legend()

            for j in range(len(metric_specs), len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            self.wandb_run.log({f"charts/{dataset_name}/metrics_by_config": wandb.Image(fig)})
            plt.close(fig)

    def _log_final_summary_to_wandb(
        self,
        all_results: List[Dict[str, Any]],
        summary_path: str,
    ):
        """Log final cross-dataset summary to W&B as table + artifact."""
        if not self.wandb_run:
            return

        final_columns = [
            "dataset",
            "config",
            "num_questions",
            "vanilla_clean_accuracy",
            "kg_clean_accuracy",
        ]
        if self.compute_metrics:
            final_columns.extend([
                "semantic_entropy_vanilla",
                "semantic_entropy_kg",
                "discrete_semantic_entropy_vanilla",
                "discrete_semantic_entropy_kg",
                "sre_uq_vanilla",
                "sre_uq_kg",
                "p_true_vanilla",
                "p_true_kg",
                "selfcheckgpt_vanilla",
                "selfcheckgpt_kg",
                "vn_entropy_vanilla",
                "vn_entropy_kg",
                "sd_uq_vanilla",
                "sd_uq_kg",
            ])

        final_table = wandb.Table(columns=final_columns)

        total_questions_logged = 0
        for dataset_block in all_results:
            dataset_name = dataset_block.get("dataset", "unknown")
            for cfg_res in dataset_block.get("config_results", []):
                cfg_name = cfg_res.get("config", {}).get("name", "default")
                n_q = int(cfg_res.get("total_questions", 0))
                total_questions_logged += n_q

                row = [
                    dataset_name,
                    cfg_name,
                    n_q,
                    float(cfg_res.get("vanilla_accuracy", 0.0)),
                    float(cfg_res.get("kg_accuracy", 0.0)),
                ]
                if self.compute_metrics:
                    grouped = self._build_grouped_uncertainty_metrics(cfg_res)
                    v = grouped.get("vanilla_rag", {})
                    k = grouped.get("kg_rag", {})
                    row.extend([
                        float(v.get("semantic_entropy", 0.0)),
                        float(k.get("semantic_entropy", 0.0)),
                        float(v.get("discrete_semantic_entropy", 0.0)),
                        float(k.get("discrete_semantic_entropy", 0.0)),
                        float(v.get("sre_uq", 0.0)),
                        float(k.get("sre_uq", 0.0)),
                        float(v.get("p_true", 0.5)),
                        float(k.get("p_true", 0.5)),
                        float(v.get("selfcheckgpt", 0.0)),
                        float(k.get("selfcheckgpt", 0.0)),
                        float(v.get("vn_entropy", 0.0)),
                        float(k.get("vn_entropy", 0.0)),
                        float(v.get("sd_uq", 0.5)),
                        float(k.get("sd_uq", 0.5)),
                    ])
                final_table.add_data(*row)

        self.wandb_run.log({"tables/final_evaluation_summary": final_table})

        if not self.compute_metrics:
            artifact = wandb.Artifact("mirage-summary", type="evaluation-summary")
            artifact.add_file(summary_path)
            self.wandb_run.log_artifact(artifact)
            self.wandb_run.summary["total_questions_logged"] = total_questions_logged
            return

        # ── AUROC / AUREC table ───────────────────────────────────────────
        _uq_metrics = [
            "semantic_entropy", "discrete_semantic_entropy",
            "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq",
        ]
        auroc_cols = (
            ["dataset", "config", "system"]
            + [f"{m}_auroc" for m in _uq_metrics]
            + [f"{m}_aurec" for m in _uq_metrics]
        )
        auroc_table = wandb.Table(columns=auroc_cols)

        for dataset_block in all_results:
            dataset_name = dataset_block.get("dataset", "unknown")
            for cfg_res in dataset_block.get("config_results", []):
                cfg_name = cfg_res.get("config", {}).get("name", "default")
                auroc_aurec = cfg_res.get("auroc_aurec", {})
                for system_label, system_key in (("vanilla_rag", "vanilla_rag"), ("kg_rag", "kg_rag")):
                    prefix = "vanilla" if system_key == "vanilla_rag" else "kg"
                    sys_data = auroc_aurec.get(system_key, {})
                    row = [
                        dataset_name,
                        cfg_name,
                        system_label,
                    ] + [
                        float(sys_data.get(f"{m}_auroc", float("nan"))) for m in _uq_metrics
                    ] + [
                        float(sys_data.get(f"{m}_aurec", float("nan"))) for m in _uq_metrics
                    ]
                    auroc_table.add_data(*row)
                    # Also log as scalars so they show up in run charts
                    for m in _uq_metrics:
                        self.wandb_run.summary[f"auroc/{dataset_name}/{cfg_name}/{prefix}/{m}"] = float(
                            sys_data.get(f"{m}_auroc", float("nan"))
                        )
                        self.wandb_run.summary[f"aurec/{dataset_name}/{cfg_name}/{prefix}/{m}"] = float(
                            sys_data.get(f"{m}_aurec", float("nan"))
                        )

        self.wandb_run.log({"tables/auroc_aurec": auroc_table})

        # ── Bar charts: all metrics Vanilla vs KG-RAG ────────────────────
        plot_metric_bar_charts(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )

        # ── AUROC / AUREC heatmaps ────────────────────────────────────────
        plot_auroc_aurec_heatmaps(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )

        # ── New analysis charts ───────────────────────────────────────────
        plot_metric_correlation_matrix(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )
        plot_reliability_diagrams(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )
        plot_compute_time_chart(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )
        plot_auroc_vs_compute_time(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )

        # ── Complementarity & query-type stratification ───────────────────
        plot_complementarity_matrix(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )
        plot_query_type_stratification(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )

        # ── Per-system AUROC comparison (central paper figure) ────────────
        plot_per_system_auroc_comparison(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )

        # ── Metric Spearman correlation matrix ────────────────────────────
        plot_metric_spearman_matrix(
            all_results=all_results,
            output_dir="results/visualizations",
            wandb_run=self.wandb_run,
        )

        # ── Hop-stratified AUROC / PPV ────────────────────────────────────
        # Build a per-config details_by_dataset map from in-memory all_results.
        # Only include datasets that have hop_count metadata or are known multi-hop.
        _MULTI_HOP_DATASETS = {
            "hotpotqa", "hotpotqa_fullwiki", "2wikimultihopqa", "musique", "multihoprag",
        }
        _hop_details_by_config: dict = {}
        for _dsblock in all_results:
            _dsname = _dsblock.get("dataset", "")
            if _dsname.lower() in _MULTI_HOP_DATASETS:
                for _cfgres in _dsblock.get("config_results", []):
                    _cfgname = _cfgres.get("config", {}).get("name", "default")
                    _dets = _cfgres.get("details", [])
                    if _dets:
                        _hop_details_by_config.setdefault(_cfgname, {})[_dsname] = _dets
        if _hop_details_by_config:
            try:
                _multi_cfg = len(_hop_details_by_config) > 1
                for _cfgname, _hop_details in sorted(_hop_details_by_config.items()):
                    _suffix = ""
                    if _multi_cfg:
                        _suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", _cfgname).strip("_") or "config"
                    _hop_result = run_hop_stratified_analysis(
                        details_by_dataset=_hop_details,
                        output_dir="paper/figures",
                        figure_suffix=_suffix,
                    )
                    for _fig_path in _hop_result.get("saved_figures", []):
                        if not (self.wandb_run and os.path.exists(_fig_path)):
                            continue
                        # W&B image logging expects raster image formats.  The hop
                        # analysis writes both PDFs and PNGs; skip the PDFs here so
                        # post-run logging cannot crash an otherwise completed run.
                        if str(_fig_path).lower().endswith(".pdf"):
                            continue
                        import wandb as _wandb
                        _wandb_key = f"charts/hop_stratified/{_cfgname}/{os.path.basename(_fig_path)}"
                        self.wandb_run.log({_wandb_key: _wandb.Image(_fig_path)})
            except Exception as _e:
                logging.warning("hop_stratified_analysis failed: %s", _e)

        # Add high-level run summary keys for quick W&B overview cards
        self.wandb_run.summary["summary/datasets_evaluated"] = len(all_results)
        self.wandb_run.summary["summary/total_questions_evaluated"] = total_questions_logged

        # Upload final JSON summary as a tracked artifact
        if os.path.exists(summary_path):
            artifact = wandb.Artifact("mirage_evaluation_summary", type="evaluation")
            artifact.add_file(summary_path)
            self.wandb_run.log_artifact(artifact)
            self.wandb_run.summary["summary/final_json_path"] = summary_path

    def _extract_chunk_texts_from_result(self, rag_result: Dict[str, Any]) -> List[str]:
        """Extract retrieved chunk texts robustly from a RAG result."""
        chunk_texts: List[str] = []

        # Prefer used_chunks when available
        used_chunks = rag_result.get("used_chunks", [])
        for c in used_chunks:
            if isinstance(c, dict):
                text = c.get("text", "")
            else:
                text = str(c)
            if text:
                chunk_texts.append(text)

        # Fall back to context.chunks
        if not chunk_texts:
            context = rag_result.get("context", {}) if isinstance(rag_result, dict) else {}
            context_chunks = context.get("chunks", []) if isinstance(context, dict) else []
            for c in context_chunks:
                if isinstance(c, dict):
                    text = c.get("text", "")
                else:
                    text = str(c)
                if text:
                    chunk_texts.append(text)

        # Deduplicate while preserving order
        unique_chunk_texts = []
        seen = set()
        for t in chunk_texts:
            key = t.strip()
            if key and key not in seen:
                seen.add(key)
                unique_chunk_texts.append(key)

        return unique_chunk_texts

    @staticmethod
    def _mean_pairwise_jaccard(chunk_sets: List[set]) -> float:
        """Mean pairwise Jaccard similarity across a list of chunk sets.

        Returns 1.0 if there is only one sample (trivially identical) and 0.0
        if all sets are empty.  Values close to 1.0 indicate deterministic
        retrieval; values close to 0.0 indicate high retrieval stochasticity.
        """
        if len(chunk_sets) < 2:
            return 1.0
        total, count = 0.0, 0
        for i in range(len(chunk_sets)):
            for j in range(i + 1, len(chunk_sets)):
                a, b = chunk_sets[i], chunk_sets[j]
                union = len(a | b)
                if union > 0:
                    total += len(a & b) / union
                # Both sets empty → undefined retrieval; skip pair rather than
                # counting as 1.0 (which would falsely inflate determinism).
                count += 1
        return total / count if count > 0 else 0.0

    def _llm_for_temperature(self, temperature: float):
        """Return a LangChainRunnableAdapter for a specific temperature (shared provider)."""
        return LangChainRunnableAdapter(
            self._base_llm_provider, model=self.llm_model, temperature=temperature
        )

    def _collect_sample_responses(
        self,
        rag_system,
        question: str,
        answer_instructions: str = "",
        base_result: Optional[Dict[str, Any]] = None,
        document_names: Optional[List[str]] = None,
        similarity_threshold: float = 0.1,
        max_chunks: int = 10,
        extra_context_texts: Optional[List[str]] = None,
        kg_name: Optional[str] = None,
        question_id: Optional[str] = None,
        llm=None,
        retrieval_temperature: float = 0.0,
        retrieval_shortlist_factor: int = 4,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        return_graph_state_traces: bool = False,
    ):
        """Collect multiple responses for proper semantic-entropy estimation.

        extra_context_texts is NOT forwarded to sampling calls — retrieval must
        compete on its own. Passing gold contexts to all samples defeats the purpose
        of measuring UQ under realistic retrieval conditions: vanilla RAG naturally
        retrieves varying chunks across samples (different top-k due to nondeterminism),
        while KG-RAG always returns the same graph (deterministic entity lookup).
        That differential is exactly what we want to measure.

        Returns:
            responses: list of sampled response strings
            retrieved_chunk_texts: chunk texts from the first/base sample
            mean_retrieval_jaccard: mean pairwise Jaccard across per-sample chunk sets
                (≈1.0 → deterministic retrieval; ≈0.0 → high stochasticity)
        """
        responses: List[str] = []
        retrieved_chunk_texts: List[str] = []
        per_sample_chunk_sets: List[set] = []
        graph_state_traces: List[Dict[str, Any]] = []

        # Base result is only used as a retrieval-context anchor. The output-side
        # UQ sample set must come from one consistent sampling policy, so we do
        # not mix the deterministic accuracy answer into the stochastic pool.
        if base_result:
            retrieved_chunk_texts = self._extract_chunk_texts_from_result(base_result)

        # Use provided LLM or fall back to default
        sampling_llm = llm if llm is not None else self.llm

        # Generate additional samples if requested
        remaining = max(0, self.entropy_samples)
        base_sample_id = 0
        extra_generate_kwargs = dict(generate_kwargs or {})
        for offset in range(remaining):
            try:
                sampled_result = rag_system.generate_response(
                    question=question,
                    llm=sampling_llm,
                    document_names=document_names,
                    similarity_threshold=similarity_threshold,
                    max_chunks=max_chunks,
                    extra_context_texts=None,  # retrieval must compete, no gold context
                    kg_name=kg_name,
                    answer_instructions=answer_instructions,
                    question_id=question_id,
                    retrieval_temperature=retrieval_temperature,
                    retrieval_shortlist_factor=retrieval_shortlist_factor,
                    retrieval_sample_id=base_sample_id + offset,
                    **extra_generate_kwargs,
                )
                sampled_response = str(sampled_result.get("response", "")).strip()
                if sampled_response:
                    responses.append(sampled_response)

                sample_chunks = self._extract_chunk_texts_from_result(sampled_result)
                per_sample_chunk_sets.append(set(sample_chunks))
                if return_graph_state_traces:
                    sample_context = sampled_result.get("context", {}) or {}
                    graph_state_traces.append(
                        sample_context.get("graph_state")
                        or summarize_context_graph_state(sample_context)
                    )

                # If we still don't have chunk context, take from sampled call
                if not retrieved_chunk_texts:
                    retrieved_chunk_texts = sample_chunks
            except Exception as e:
                logging.warning(f"Failed to collect semantic-entropy sample: {e}")

        # Ensure at least one response exists
        if not responses and base_result:
            responses = [str(base_result.get("response", "")).strip()]

        mean_jaccard = self._mean_pairwise_jaccard(per_sample_chunk_sets)
        if return_graph_state_traces:
            return responses, retrieved_chunk_texts, mean_jaccard, graph_state_traces
        return responses, retrieved_chunk_texts, mean_jaccard
        
    def _get_neo4j_driver(self):
        """Get Neo4j driver"""
        return GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
    def _clear_neo4j(self):
        """Clear all data from Neo4j"""
        logging.info("Clearing Neo4j database...")
        driver = self._get_neo4j_driver()
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logging.info("Neo4j cleared")
        driver.close()

    def _dataset_kg_exists(self, dataset_name: str) -> int:
        """
        Check if a KG already exists for the given dataset in Neo4j.
        Returns the number of chunks found for this dataset, or 0 if not found.
        """
        driver = self._get_neo4j_driver()
        try:
            with driver.session() as session:
                # Check using kgName property on Document nodes
                result = session.run(
                    "MATCH (d:Document {kgName: $kg_name})<-[:PART_OF]-(c:Chunk) "
                    "RETURN count(c) AS n",
                    {"kg_name": dataset_name}
                )
                chunk_count = result.single()["n"]
                logging.info(f"[_dataset_kg_exists] Query for kgName='{dataset_name}': {chunk_count} chunks")
                return chunk_count
        except Exception as e:
            logging.warning(f"[_dataset_kg_exists] Error checking for existing KG: {e}")
            return 0
        finally:
            driver.close()

    def _get_dataset_kg_stats(self, dataset_name: str) -> Dict[str, int]:
        """Get dataset-scoped KG stats to validate isolation and quality."""
        driver = self._get_neo4j_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {kgName: $kg_name})
                    CALL {
                        WITH d
                        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
                        RETURN count(DISTINCT c) AS chunks
                    }
                    CALL {
                        WITH d
                        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)-[:HAS_ENTITY|MENTIONS]->(e:__Entity__)
                        RETURN count(DISTINCT e) AS entities,
                               count(DISTINCT c) AS chunks_with_entities,
                               count(*) AS entity_links
                    }
                    CALL {
                        WITH d
                        OPTIONAL MATCH (d)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY|MENTIONS]->(e:__Entity__)-[r]-(:__Entity__)
                        RETURN count(DISTINCT r) AS relationships
                    }
                    RETURN
                        count(DISTINCT d) AS documents,
                        chunks,
                        entities,
                        relationships,
                        chunks_with_entities,
                        entity_links
                    """,
                    {"kg_name": dataset_name},
                )
                row = result.single()
                return {
                    "documents": int(row["documents"] if row and row["documents"] is not None else 0),
                    "chunks": int(row["chunks"] if row and row["chunks"] is not None else 0),
                    "entities": int(row["entities"] if row and row["entities"] is not None else 0),
                    "relationships": int(row["relationships"] if row and row["relationships"] is not None else 0),
                    "chunks_with_entities": int(row["chunks_with_entities"] if row and row["chunks_with_entities"] is not None else 0),
                    "entity_links": int(row["entity_links"] if row and row["entity_links"] is not None else 0),
                }
        except Exception as e:
            logging.warning(f"[_get_dataset_kg_stats] Error collecting stats for '{dataset_name}': {e}")
            return {
                "documents": 0,
                "chunks": 0,
                "entities": 0,
                "relationships": 0,
                "chunks_with_entities": 0,
                "entity_links": 0,
            }
        finally:
            driver.close()

    @staticmethod
    def _assess_kg_quality(
        stats: Dict[str, int],
        min_chunks: int,
        min_entities: int,
    ) -> Tuple[bool, List[str]]:
        """Apply conservative quality gates so sparse or malformed KGs are rebuilt."""
        reasons: List[str] = []
        documents = int(stats.get("documents", 0) or 0)
        chunks = int(stats.get("chunks", 0) or 0)
        entities = int(stats.get("entities", 0) or 0)
        relationships = int(stats.get("relationships", 0) or 0)
        chunks_with_entities = int(stats.get("chunks_with_entities", 0) or 0)
        entity_links = int(stats.get("entity_links", 0) or 0)

        if documents < 1:
            reasons.append("no_documents")
        if chunks < int(min_chunks):
            reasons.append(f"too_few_chunks<{min_chunks}")
        if entities < int(min_entities):
            reasons.append(f"too_few_entities<{min_entities}")
        if relationships < 1:
            reasons.append("no_relationships")

        chunk_entity_coverage = (chunks_with_entities / chunks) if chunks else 0.0
        avg_entities_per_chunk = (entity_links / chunks) if chunks else 0.0
        relationship_density = (relationships / entities) if entities else 0.0

        if chunks >= 5 and chunk_entity_coverage < 0.35:
            reasons.append(f"low_chunk_entity_coverage<{chunk_entity_coverage:.2f}")
        if chunks >= 10 and avg_entities_per_chunk < 0.5:
            reasons.append(f"low_entities_per_chunk<{avg_entities_per_chunk:.2f}")
        if entities >= 50 and relationship_density < 0.018:
            reasons.append(f"low_relationship_density<{relationship_density:.3f}")

        return (len(reasons) == 0), reasons

    def _verify_kg_quality(
        self,
        dataset_name: str,
        min_chunks: int = 1,
        min_entities: int = 1,
    ) -> bool:
        """Verify minimal quality for a dataset-scoped KG in Neo4j."""
        stats = self._get_dataset_kg_stats(dataset_name)
        logging.info(
            f"[kg_quality] dataset={dataset_name} | "
            f"docs={stats['documents']} chunks={stats['chunks']} "
            f"entities={stats['entities']} relationships={stats['relationships']} "
            f"chunks_with_entities={stats.get('chunks_with_entities', 0)} "
            f"entity_links={stats.get('entity_links', 0)}"
        )

        is_valid, reasons = self._assess_kg_quality(stats, min_chunks=min_chunks, min_entities=min_entities)
        if not is_valid:
            logging.warning(
                f"[kg_quality] dataset={dataset_name} failed minimum thresholds "
                f"(min_chunks={min_chunks}, min_entities={min_entities}) | reasons={reasons}"
            )
        return is_valid

    def _validate_retrieval_scope(
        self,
        rag_result: Dict[str, Any],
        dataset_name: str,
        system_name: str,
        question_id: Optional[str] = None,
    ) -> bool:
        """Check that retrieved chunks stay within the requested dataset and question scope."""
        context = rag_result.get("context", {}) if isinstance(rag_result, dict) else {}
        chunks = context.get("chunks", []) if isinstance(context, dict) else []

        mismatches: List[Dict[str, Any]] = []
        missing_kg_name = 0
        question_scope_mismatches: List[Dict[str, Any]] = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_kg_name = chunk.get("kg_name")
            if chunk_kg_name is None:
                missing_kg_name += 1
                continue
            if chunk_kg_name != dataset_name:
                mismatches.append(
                    {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "document": chunk.get("document", ""),
                        "kg_name": chunk_kg_name,
                    }
                )
            if question_id is not None:
                chunk_question_id = chunk.get("question_id")
                if chunk_question_id not in {None, "", question_id}:
                    question_scope_mismatches.append(
                        {
                            "chunk_id": chunk.get("chunk_id", ""),
                            "document": chunk.get("document", ""),
                            "question_id": chunk_question_id,
                        }
                    )

        if missing_kg_name > 0 and chunks:
            logging.warning(
                f"[{system_name}] {missing_kg_name}/{len(chunks)} retrieved chunks have no kg_name field; "
                "cannot fully verify isolation for those chunks."
            )

        if mismatches:
            logging.error(
                f"[{system_name}] Retrieved chunks from other KG(s) while evaluating dataset='{dataset_name}': "
                f"{mismatches[:5]}"
            )
            return False

        if question_scope_mismatches:
            logging.error(
                f"[{system_name}] Retrieved chunks from other question scopes while evaluating "
                f"dataset='{dataset_name}' question_id='{question_id}': {question_scope_mismatches[:5]}"
            )
            return False

        return True

    def _delete_dataset_kg(self, dataset_name: str):
        """Delete KG for a specific dataset from Neo4j"""
        logging.info(f"Deleting KG for dataset: {dataset_name}")
        driver = self._get_neo4j_driver()
        batch_size = 200

        def _delete_in_batches(session, label: str, query: str) -> int:
            total_deleted = 0
            while True:
                record = session.run(
                    query,
                    {"kg_name": dataset_name, "batch_size": batch_size},
                ).single()
                deleted = int(record["deleted"] or 0) if record else 0
                total_deleted += deleted
                if deleted == 0:
                    break
            logging.info(
                "[_delete_dataset_kg] Deleted %s %s node(s) for dataset '%s'",
                total_deleted,
                label,
                dataset_name,
            )
            return total_deleted

        try:
            with driver.session() as session:
                # Delete retrieval spans first, then chunks/docs, then orphaned
                # entities. Batched deletes keep Neo4j transaction memory bounded.
                _delete_in_batches(
                    session,
                    "retrieval chunk",
                    """
                    CALL {
                        MATCH (rc:RetrievalChunk)-[:RETRIEVES_FROM]->(:Chunk)-[:PART_OF]->(:Document {kgName: $kg_name})
                        WITH rc LIMIT $batch_size
                        DETACH DELETE rc
                        RETURN count(rc) AS deleted
                    }
                    RETURN deleted
                    """,
                )

                _delete_in_batches(
                    session,
                    "qualifier",
                    """
                    CALL {
                        MATCH (q:Qualifier {kgName: $kg_name})
                        WITH q LIMIT $batch_size
                        DETACH DELETE q
                        RETURN count(q) AS deleted
                    }
                    RETURN deleted
                    """,
                )

                _delete_in_batches(
                    session,
                    "chunk",
                    """
                    CALL {
                        MATCH (c:Chunk)-[:PART_OF]->(:Document {kgName: $kg_name})
                        WITH c LIMIT $batch_size
                        DETACH DELETE c
                        RETURN count(c) AS deleted
                    }
                    RETURN deleted
                    """,
                )

                _delete_in_batches(
                    session,
                    "document",
                    """
                    CALL {
                        MATCH (d:Document {kgName: $kg_name})
                        WITH d LIMIT $batch_size
                        DETACH DELETE d
                        RETURN count(d) AS deleted
                    }
                    RETURN deleted
                    """,
                )

                _delete_in_batches(
                    session,
                    "orphan entity",
                    """
                    CALL {
                        MATCH (e:__Entity__ {kgName: $kg_name})
                        WHERE NOT EXISTS {
                            MATCH (:Chunk)-[:HAS_ENTITY|MENTIONS]->(e)
                        }
                        WITH e LIMIT $batch_size
                        DETACH DELETE e
                        RETURN count(e) AS deleted
                    }
                    RETURN deleted
                    """,
                )
                logging.info(f"Deleted KG for dataset: {dataset_name}")
        except Exception as e:
            logging.warning(f"[_delete_dataset_kg] Error deleting KG: {e}")
        finally:
            driver.close()

    def _build_dataset_like_raw_mapping(self, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Build a dict-like dataset mapping from normalized adapters to preserve
        the expected downstream shape in evaluation.
        """
        eval_records = self._build_eval_records(dataset_name)
        raw_records = {
            str(question_id): record
            for question_id, record in load_raw_dataset(dataset_name).items()
            if isinstance(record, dict)
        }
        mapping: Dict[str, Dict[str, Any]] = {}
        for rec in eval_records:
            question_id = str(rec["id"])
            merged = dict(raw_records.get(question_id, {}))
            merged.update({
                "question": rec.get("question", ""),
                "expected_answer": rec.get("expected_answer", ""),
                "aliases": rec.get("aliases", []),
                "contexts": rec.get("contexts", []),
                "options": rec.get("options", {}) or {},
                "task_type": rec.get("task_type", ""),
            })
            mapping[question_id] = merged
        return mapping
        
    def _load_mirage_raw_data(self, dataset_name: str) -> Dict[str, Any]:
        """Load evaluation data through normalized dataset adapters."""
        try:
            adapted = self._build_dataset_like_raw_mapping(dataset_name)
            if not adapted:
                logging.warning(f"No adapted evaluation records found for {dataset_name}")
            return adapted
        except Exception as e:
            logging.error(f"Failed to load adapted dataset '{dataset_name}': {e}")
            return {}
    
    def _extract_contexts_from_question(self, question_data: Dict, dataset_name: str = None) -> List[str]:
        """Extract context passages from a question"""
        # Adapter-normalized shape
        contexts = question_data.get("contexts", [])
        if isinstance(contexts, list) and contexts:
            return [str(c).strip() for c in contexts if str(c).strip()]

        # Try PubMedQA style (CONTEXTS field)
        contexts = question_data.get("CONTEXTS", [])
        if isinstance(contexts, list) and contexts:
            return [c if isinstance(c, str) else c.get("text", "") for c in contexts]
        
        # Try BioASQ style (snippets field)
        snippets = question_data.get("snippets", [])
        if isinstance(snippets, list) and snippets:
            return [s.get("text", "") for s in snippets if s.get("text")]
        
        return []
    
    def _get_answer_from_question(self, question_data: Dict) -> str:
        """Extract ground truth answer from question data"""
        # Adapter-normalized shape
        if "expected_answer" in question_data:
            return str(question_data["expected_answer"]).lower()

        # Try PubMedQA style
        if "final_decision" in question_data:
            return question_data["final_decision"].lower()

        # Try MedQA-style answer_idx with options
        if "answer_idx" in question_data and "options" in question_data:
            try:
                idx = int(question_data["answer_idx"])
                options = question_data.get("options", {})
                if isinstance(options, dict):
                    option_keys = list(options.keys())
                    if 0 <= idx < len(option_keys):
                        return str(options[option_keys[idx]]).lower()
                elif isinstance(options, list) and 0 <= idx < len(options):
                    return str(options[idx]).lower()
            except Exception:
                pass

        # Try MedMCQA-style cop (1-based: 1->A, 2->B, ...)
        if "cop" in question_data:
            try:
                key = chr(ord('A') + int(question_data["cop"]) - 1)
                if key in question_data:
                    return str(question_data[key]).lower()
                op_map = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
                if op_map.get(key) in question_data:
                    return str(question_data[op_map[key]]).lower()
            except Exception:
                pass
        
        # Try BioASQ style
        if "exact_answer" in question_data:
            return str(question_data["exact_answer"]).lower()
        
        # Try generic
        if "answer" in question_data:
            return str(question_data["answer"]).lower()
        if "Answer" in question_data:
            return str(question_data["Answer"]).lower()
        return ""
    
    def _get_question_text(self, question_data: Dict) -> str:
        """Extract question text from question data"""
        if "question" in question_data:
            return question_data["question"]
        # Try BioASQ style
        if "body" in question_data:
            return question_data["body"]
        # Try PubMedQA style
        if "QUESTION" in question_data:
            return question_data["QUESTION"]
        return ""

    def _extract_options_and_task_type(self, question_data: Dict) -> Tuple[Dict[str, str], str]:
        """Extract MCQ options/task type when available from raw dataset record."""
        explicit_task_type = str(question_data.get("task_type", "")).strip().lower()
        options: Dict[str, str] = {}

        raw_options = question_data.get("options")
        if isinstance(raw_options, dict):
            options = {
                str(k).strip().upper(): str(v).strip()
                for k, v in raw_options.items()
                if str(v).strip()
            }
        elif isinstance(raw_options, list):
            options = {
                chr(ord('A') + i): str(v).strip()
                for i, v in enumerate(raw_options)
                if str(v).strip()
            }
        elif any(k in question_data for k in ["A", "B", "C", "D"]):
            options = {
                k: str(question_data.get(k, "")).strip()
                for k in ["A", "B", "C", "D"]
                if str(question_data.get(k, "")).strip()
            }
        elif any(k in question_data for k in ["opa", "opb", "opc", "opd"]):
            med_map = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
            options = {
                k: str(question_data.get(v, "")).strip()
                for k, v in med_map.items()
                if str(question_data.get(v, "")).strip()
            }

        if options:
            return options, (explicit_task_type or "mcq")

        q_type = str(question_data.get("type", "")).strip().lower()
        if q_type in {"yesno", "yes/no", "binary"} or "final_decision" in question_data:
            return {}, "binary"

        return {}, explicit_task_type

    def _normalize_decision_label(self, text: str) -> str:
        """Normalize explicit labels into yes/no/maybe when possible."""
        if not text:
            return ""

        t = str(text).strip().lower()

        # 1) Exact/compact labels (safe, no ambiguity).
        if t in {"yes", "y", "true"}:
            return "yes"
        if t in {"no", "n", "false"}:
            return "no"
        if t in {"maybe", "uncertain", "unknown"}:
            return "maybe"

        # 2) Label at start of answer or in explicit "answer is ..." pattern.
        lead = t[:120]
        if re.search(r"^\s*(yes|true)\b", lead) or re.search(r"\b(answer|conclusion|final answer)\s*(is|:)\s*(yes|true)\b", t):
            return "yes"
        if re.search(r"^\s*(no|false)\b", lead) or re.search(r"\b(answer|conclusion|final answer)\s*(is|:)\s*(no|false)\b", t):
            return "no"
        if re.search(r"^\s*(maybe|uncertain)\b", lead) or re.search(r"\b(answer|conclusion|final answer)\s*(is|:)\s*(maybe|uncertain)\b", t):
            return "maybe"

        return ""

    def _infer_decision_from_response(self, response: str, question: str = "") -> str:
        """
        Infer yes/no/maybe from free-form response text.

        This is intentionally heuristic for PubMedQA-style outputs where the
        model may not explicitly say "yes"/"no" but still provides a clear
        conclusion (e.g., "methods were comparable" for a suitability question).
        """
        if not response:
            return ""

        t = re.sub(r"\s+", " ", str(response).strip().lower())
        q = re.sub(r"\s+", " ", str(question or "").strip().lower())
        lead = t[:220]

        # 1) Explicit labels first.
        if re.search(r"^\s*(yes|true)\b", lead) or re.search(r"\b(answer|conclusion)\s*(is|:)\s*(yes|true)\b", t):
            return "yes"
        if re.search(r"^\s*(no|false)\b", lead) or re.search(r"\b(answer|conclusion)\s*(is|:)\s*(no|false)\b", t):
            return "no"
        # "so yes/no", "therefore yes/no", "thus yes/no" — buried conclusion markers.
        if re.search(r"\b(so|therefore|thus|hence|in summary|in conclusion),?\s*(yes|true)\b", t):
            return "yes"
        if re.search(r"\b(so|therefore|thus|hence|in summary|in conclusion),?\s*(no|false)\b", t):
            return "no"
        if re.search(r"^\s*(maybe|uncertain)\b", lead) or re.search(r"\b(answer|conclusion)\s*(is|:)\s*(maybe|uncertain)\b", t):
            return "maybe"

        # 2) Insufficient/inconclusive context => maybe.
        insufficient_patterns = [
            r"\bnot enough information\b",
            r"\binsufficient (data|evidence|information)\b",
            r"\bcannot (determine|conclude|say)\b",
            r"\binconclusive\b",
            r"\bunclear\b",
            r"\bno direct information\b",
            r"\bcontext does not provide\b",
            r"\bsubject of investigation\b",
        ]
        if any(re.search(p, t) for p in insufficient_patterns):
            return "maybe"

        # 3) Evidence-based polarity cues.
        positive_patterns = [
            r"\b(correlated|associated|connected)\b",
            r"\bcorrelated closely\b",
            r"\bcomparable\b",
            r"\bviable alternative\b",
            r"\bsuitable as an alternative\b",
            r"\b(beneficial|valuable|effective)\b",
            r"\bindicat(es|ing)\b",
            r"\bsupport(s|ed|ive)\b",
            r"\bsignificant (association|correlation|benefit)\b",
        ]
        negative_patterns = [
            r"\bno evidence\b",
            r"\bno (association|correlation|connection|benefit)\b",
            r"\bnot (associated|correlated|connected|beneficial|effective|suitable)\b",
            r"\bfailed to (show|demonstrate)\b",
            r"\bdoes not (support|indicate|show|demonstrate)\b",
            r"\bpoor (patient survival|prognosis|outcome)\b",
            r"\bworse (survival|prognosis|outcome)\b",
            r"\bdoes not contain (any information|information)\b",
        ]

        pos_score = sum(1 for p in positive_patterns if re.search(p, t))
        neg_score = sum(1 for p in negative_patterns if re.search(p, t))

        if pos_score > 0 and neg_score == 0:
            return "yes"
        if neg_score > 0 and pos_score == 0:
            return "no"
        if pos_score > neg_score:
            return "yes"
        if neg_score > pos_score:
            return "no"

        # 4) Question-aware fallback for yes/no prompts.
        if q.startswith(("is ", "are ", "does ", "do ", "can ", "should ", "was ", "were ", "has ", "have ")):
            if re.search(r"\b(comparable|viable alternative|correlated|associated|beneficial|effective)\b", t):
                return "yes"
            if re.search(r"\b(no evidence|not associated|not correlated|not beneficial|not effective)\b", t):
                return "no"

        return ""

    def _llm_judge_correct(
        self,
        question: str,
        expected_answer: str,
        model_response: str,
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """Use the LLM as a judge to evaluate factoid answer correctness.

        Stricter than token-coverage: the judge checks semantic equivalence,
        correctly handles 'I don't know' responses, and handles name variations.
        Aliases (alternative correct answers) are included in the prompt so the
        judge can accept any of them.
        Results are cached so vanilla and KG evaluations of the same response
        don't double-bill.
        """
        alias_key = "|".join(sorted(aliases)) if aliases else ""
        cache_key = f"{question}|||{expected_answer}|||{alias_key}|||{model_response}"
        if cache_key in self._llm_judge_cache:
            return self._llm_judge_cache[cache_key]

        accepted = [expected_answer] + [a for a in (aliases or []) if a and a != expected_answer]
        accepted_str = " / ".join(f'"{a}"' for a in accepted[:6])  # cap to avoid token bloat

        system_prompt = (
            "You are a strict answer evaluator for a factoid question answering task. "
            "Your job is to decide if a model's response is CORRECT.\n\n"
            "Rules:\n"
            "- Reply with exactly one word: 'correct' or 'incorrect'.\n"
            "- The response is CORRECT if it contains an answer semantically equivalent "
            "to ANY of the accepted answers (minor spelling/accent differences are ok).\n"
            "- The response is INCORRECT if the model says it doesn't know, cannot determine, "
            "or provides a factually different answer."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Accepted answers: {accepted_str}\n"
            f"Model response: {model_response}\n\n"
            "Is the model response correct? Reply with one word only: correct or incorrect."
        )

        try:
            raw = self.judge_llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.judge_model,
            )
            verdict = str(raw).strip().lower()
            result = verdict.startswith("correct")
        except Exception as e:
            logging.warning(f"LLM judge failed ({e}), falling back to string match")
            result = self._contains_expected_text(
                model_response.lower(), expected_answer.lower()
            )

        self._llm_judge_cache[cache_key] = result
        return result

    def _is_answer_correct(
        self,
        expected_answer: str,
        model_response: str,
        question: str = "",
        options: Optional[Dict[str, str]] = None,
        task_type: str = "",
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """Flexible correctness check for different question types:
        - Binary (yes/no): checks if response starts with explicit yes/no
        - Factoid: checks if key answer appears in response
        - List: checks if all expected items appear in response
        - Multiple choice: checks if correct option appears
        - Aliases: for multi-hop datasets (2WikiMultiHopQA), also checks alias strings
        """
        expected_answer = str(expected_answer) if expected_answer else ""
        model_response = str(model_response) if model_response else ""
        
        # Normalize for comparison
        expected_lower = expected_answer.lower().strip()
        response_lower = model_response.lower().strip()
        
        # 1. First try binary (yes/no/maybe) matching.
        expected_binary = self._normalize_decision_label(expected_answer)
        predicted_binary = self._normalize_decision_label(model_response)
        if expected_binary and predicted_binary:
            return expected_binary == predicted_binary

        # 1b. If response is free-form for binary tasks, attempt inference.
        if expected_binary and not predicted_binary:
            inferred = self._infer_decision_from_response(model_response, question=question)
            if inferred:
                return inferred == expected_binary

        # 2. MCQ-specific handling (preferred for medqa/medmcqa/mmlu style records).
        if task_type == "mcq" and options:
            normalized_options = {
                str(k).strip().upper(): str(v).strip().lower()
                for k, v in options.items()
                if str(v).strip()
            }
            correct_keys = [
                k for k, v in normalized_options.items()
                if v == expected_lower or self._contains_expected_text(v, expected_lower)
            ]

            # Detect letter-style predictions: "B", "Option C", "Answer: D".
            letter_patterns = [
                r"^\s*([A-D])\b",
                r"\boption\s*([A-D])\b",
                r"\banswer\s*(?:is|:)\s*([A-D])\b",
                r"\(([A-D])\)",
            ]
            predicted_key = ""
            for p in letter_patterns:
                m = re.search(p, response_lower, flags=re.IGNORECASE)
                if m:
                    predicted_key = m.group(1).upper()
                    break

            if predicted_key and correct_keys:
                return predicted_key in correct_keys

            # Fallback: compare expected option text directly in generated response.
            return self._contains_expected_text(response_lower, expected_lower)
        
        # 3. Factoid / list answer matching.
        # Use LLM judge when available (stricter, handles "I don't know", accents, paraphrases).
        # Aliases are passed so the judge can accept any correct surface form.
        if self.use_llm_judge and question and expected_answer and model_response:
            return self._llm_judge_correct(question, expected_answer, model_response, aliases=aliases)

        # Fallback: heuristic token-coverage matching.
        # Handle formats like: "[['xia']]", "[['casirivimab'], ['imdevimab']]", "['answer']"
        expected_items = []
        try:
            import ast
            parsed = ast.literal_eval(expected_answer)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, list):
                        expected_items.extend([str(x).lower().strip() for x in item])
                    else:
                        expected_items.append(str(item).lower().strip())
        except (ValueError, SyntaxError):
            cleaned = expected_answer.strip("[]'\"").lower()
            if cleaned:
                expected_items = [cleaned]

        if not expected_items:
            expected_items = [expected_lower]

        items_found = [self._contains_expected_text(response_lower, item) for item in expected_items]

        if not items_found:
            return False

        if len(expected_items) == 1:
            if items_found[0]:
                return True
            for alias in (aliases or []):
                if alias and self._contains_expected_text(response_lower, alias.lower().strip()):
                    return True
            return False

        return all(items_found)
    
    def _build_kg_for_dataset(self, dataset_name: str) -> bool:
        """
        Build dataset-scoped KG from normalized adapter contexts.

        NOTE: No global Neo4j clearing here. Rebuild behavior is handled in
        run_pipeline via _delete_dataset_kg(dataset_name), so only selected
        datasets are affected.
        """
        logging.info(f"Building KG for dataset: {dataset_name}")

        try:
            kg_contract = self._prepare_dataset_kg_contract(
                dataset_name,
                force_resample=self.rebuild_kg,
            )
        except Exception as e:
            logging.error(f"Failed to prepare KG inputs for '{dataset_name}': {e}")
            return False

        passages = kg_contract["passages"]
        build_meta = kg_contract["build_meta"]
        records_for_kg = kg_contract["records_for_kg"]
        evaluable_inference_records = kg_contract["evaluable_inference_records"]

        logging.info(
            "Building KG from %d context passages for %s (scope=%s, records_used=%d/%d)",
            len(passages),
            dataset_name,
            self.dataset_kg_scope,
            len(records_for_kg),
            len(evaluable_inference_records),
        )

        if not passages:
            logging.warning(f"No contexts found for {dataset_name}")
            return False

        builder_profile = self._resolve_kg_builder_profile(dataset_name)
        builder_kwargs = self._kg_builder_kwargs_for_profile(
            dataset_name,
            builder_profile,
        )
        if builder_profile == "lightweight":
            logging.info(
                "Using lightweight KG builder profile for %s: "
                "anchor_constrained=%s self_reflection=%s "
                "anchor_supplement=%s cross_passage_recovery=%s",
                dataset_name,
                builder_kwargs["enable_anchor_constrained_extraction"],
                builder_kwargs["enable_self_reflection"],
                builder_kwargs["enable_anchor_coverage_supplement"],
                builder_kwargs["enable_cross_passage_relation_recovery"],
            )
        elif builder_profile == "full":
            logging.info(
                "Using full KG builder profile for %s: "
                "anchor_constrained=%s self_reflection=%s "
                "anchor_supplement=%s cross_passage_recovery=%s "
                "self_consistency_n=%s few_shot=%s reverify=%s "
                "min_triple_confidence=%.2f rel_type_similarity=%.2f "
                "umls=%s soft_link=%s fragmentation_repair=%s "
                "summaries=%s claims=%s",
                dataset_name,
                builder_kwargs["enable_anchor_constrained_extraction"],
                builder_kwargs["enable_self_reflection"],
                builder_kwargs["enable_anchor_coverage_supplement"],
                builder_kwargs["enable_cross_passage_relation_recovery"],
                builder_kwargs["self_consistency_n"],
                builder_kwargs["few_shot_example_count"],
                builder_kwargs["enable_low_confidence_triple_reverification"],
                builder_kwargs["min_triple_confidence"],
                builder_kwargs["relationship_type_similarity_threshold"],
                builder_kwargs["enable_umls_linking"],
                builder_kwargs["enable_soft_entity_linking"],
                builder_kwargs["enable_fragmentation_repair"],
                builder_kwargs["enable_graph_summaries"],
                builder_kwargs["enable_claim_extraction"],
            )
        else:
            logging.info(
                "Using full KG builder profile for %s: "
                "anchor_constrained=True self_reflection=True "
                "anchor_supplement=True cross_passage_recovery=True",
                dataset_name,
            )

        try:
            kg_creator = UnifiedOntologyGuidedKGCreator(
                chunk_size=1500,
                chunk_overlap=200,
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
                neo4j_database="neo4j",
                embedding_model=self.embedding_provider,
                **builder_kwargs,
            )

            # Passage-aware extraction: each passage is chunked independently so
            # the LLM never sees entity pairs co-located only because two unrelated
            # passages were concatenated.  Cross-chunk relationship extraction is
            # scoped within each passage.
            kg = kg_creator.generate_knowledge_graph_from_passages(
                passages=passages,
                llm=self.kg_llm_provider,
                file_name=dataset_name,
                model_name=self.llm_model,
                kg_name=dataset_name,
                doc_metadata=build_meta,
                doc_hash=build_meta.get("contentHash"),
            )

            stored = bool(kg.get("metadata", {}).get("stored_in_neo4j", False))
            if not stored:
                logging.error(f"Passage-aware KG pipeline did not store graph for {dataset_name}")
                return False

            stored_relationships = kg.get("metadata", {}).get(
                "stored_relationships",
                kg.get("metadata", {}).get("total_relationships", 0),
            )
            logging.info(
                f"Successfully populated Neo4j for {dataset_name} | "
                f"passages={kg.get('metadata', {}).get('total_passages', 0)}, "
                f"chunks={kg.get('metadata', {}).get('total_chunks', 0)}, "
                f"entities={kg.get('metadata', {}).get('total_entities', 0)}, "
                f"relationships={stored_relationships}"
            )
            return True
            
        except Exception as e:
            logging.error(f"Failed to populate Neo4j for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _default_uncertainty_metrics(self) -> Dict[str, float]:
        """Safe defaults for uncertainty metrics when generation fails."""
        return {
            metric_name: self.UNCERTAINTY_METRIC_DEFAULTS[metric_name]
            for metric_name in (
                "semantic_entropy",
                "discrete_semantic_entropy",
                "sre_uq",
                "p_true",
                "selfcheckgpt",
                "vn_entropy",
                "sd_uq",
            )
        }

    def _structural_metric_max_hops(self, dataset_name: str, retrieval_max_hops: int) -> int:
        """Return a safe hop budget for structural graph metrics."""
        override = self.DATASET_STRUCTURAL_METRIC_MAX_HOPS.get(dataset_name)
        if override is None:
            return retrieval_max_hops
        return max(1, min(int(retrieval_max_hops), int(override)))

    def _build_grouped_uncertainty_metrics(self, config_result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Return strict 8-metric summary grouped by approach for one config result."""
        metric_field_map = {
            "semantic_entropy": ("vanilla_avg_semantic_entropy", "kg_avg_semantic_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["semantic_entropy"]),
            "discrete_semantic_entropy": ("vanilla_avg_discrete_semantic_entropy", "kg_avg_discrete_semantic_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["discrete_semantic_entropy"]),
            "sre_uq": ("vanilla_avg_sre_uq", "kg_avg_sre_uq", self.UNCERTAINTY_METRIC_DEFAULTS["sre_uq"]),
            "p_true": ("vanilla_avg_p_true", "kg_avg_p_true", self.UNCERTAINTY_METRIC_DEFAULTS["p_true"]),
            "selfcheckgpt": ("vanilla_avg_selfcheckgpt", "kg_avg_selfcheckgpt", self.UNCERTAINTY_METRIC_DEFAULTS["selfcheckgpt"]),
            "vn_entropy": ("vanilla_avg_vn_entropy", "kg_avg_vn_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["vn_entropy"]),
            "sd_uq": ("vanilla_avg_sd_uq", "kg_avg_sd_uq", self.UNCERTAINTY_METRIC_DEFAULTS["sd_uq"]),
            "graph_path_support": ("vanilla_avg_graph_path_support", "kg_avg_graph_path_support", self.UNCERTAINTY_METRIC_DEFAULTS["graph_path_support"]),
            "graph_path_disagreement": ("vanilla_avg_graph_path_disagreement", "kg_avg_graph_path_disagreement", self.UNCERTAINTY_METRIC_DEFAULTS["graph_path_disagreement"]),
            "competing_answer_alternatives": ("vanilla_avg_competing_answer_alternatives", "kg_avg_competing_answer_alternatives", self.UNCERTAINTY_METRIC_DEFAULTS["competing_answer_alternatives"]),
            "evidence_vn_entropy": ("vanilla_avg_evidence_vn_entropy", "kg_avg_evidence_vn_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["evidence_vn_entropy"]),
            "subgraph_informativeness": ("vanilla_avg_subgraph_informativeness", "kg_avg_subgraph_informativeness", self.UNCERTAINTY_METRIC_DEFAULTS["subgraph_informativeness"]),
            "subgraph_perturbation_stability": ("vanilla_avg_subgraph_perturbation_stability", "kg_avg_subgraph_perturbation_stability", self.UNCERTAINTY_METRIC_DEFAULTS["subgraph_perturbation_stability"]),
            "support_entailment_uncertainty": ("vanilla_avg_support_entailment_uncertainty", "kg_avg_support_entailment_uncertainty", self.UNCERTAINTY_METRIC_DEFAULTS["support_entailment_uncertainty"]),
            "evidence_conflict_uncertainty": ("vanilla_avg_evidence_conflict_uncertainty", "kg_avg_evidence_conflict_uncertainty", self.UNCERTAINTY_METRIC_DEFAULTS["evidence_conflict_uncertainty"]),
        }

        grouped = {
            "vanilla_rag": {},
            "kg_rag": {},
        }

        for metric_name in self.UNCERTAINTY_METRIC_NAMES:
            vanilla_field, kg_field, default = metric_field_map[metric_name]
            grouped["vanilla_rag"][metric_name] = float(config_result.get(vanilla_field, default))
            grouped["kg_rag"][metric_name] = float(config_result.get(kg_field, default))

        return grouped

    def _apply_clean_accuracy_reporting(self, result_block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make clean accuracy the headline reporting number while preserving raw accuracy.

        Raw accuracy remains available under ``*_accuracy_raw`` for debugging and
        methodological audits, but all user-facing summaries should read the
        overwritten ``*_accuracy`` fields after this helper runs.
        """
        vanilla_raw = float(result_block.get("vanilla_accuracy", 0.0) or 0.0)
        kg_raw = float(result_block.get("kg_accuracy", 0.0) or 0.0)
        vanilla_clean = float(result_block.get("vanilla_accuracy_excluding_errors", vanilla_raw) or vanilla_raw)
        kg_clean = float(result_block.get("kg_accuracy_excluding_errors", kg_raw) or kg_raw)

        result_block["vanilla_accuracy_raw"] = vanilla_raw
        result_block["kg_accuracy_raw"] = kg_raw
        result_block["vanilla_accuracy"] = vanilla_clean
        result_block["kg_accuracy"] = kg_clean
        result_block["reported_accuracy_variant"] = "clean_excluding_generation_failures"
        return result_block

    
    def _run_evaluation_on_dataset(self, dataset_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run RAG comparison: Vanilla RAG vs KG-RAG using Neo4j database"""
        config = config or {"name": "default", "similarity_threshold": 0.1, "max_chunks": 10}
        config_name = config.get("name", "default")
        similarity_threshold = float(config.get("similarity_threshold", 0.1))
        max_chunks = int(config.get("max_chunks", 10))
        retrieval_temperature = float(config.get("retrieval_temperature", 0.0) or 0.0)
        retrieval_shortlist_factor = int(config.get("retrieval_shortlist_factor", 4) or 4)
        kg_generation = dict(config.get("kg_generation") or {})
        retrieval_env_overrides = self._retrieval_env_overrides(config)
        with self._temporary_env(retrieval_env_overrides):
            vanilla_rag, kg_rag = self._build_rag_systems_for_config(config)

        logging.info(
            f"Running RAG comparison on {dataset_name} | config={config_name} "
            f"(threshold={similarity_threshold}, max_chunks={max_chunks}, "
            f"retrieval_temperature={retrieval_temperature:g}, "
            f"retrieval_shortlist_factor={retrieval_shortlist_factor})"
        )
        
        # Load dataset
        dataset = self._load_mirage_raw_data(dataset_name)
        if not dataset:
            logging.error(f"No raw data found for {dataset_name}")
            return {"dataset": dataset_name, "error": "No raw data found"}

        existing_chunk_count = self._dataset_kg_exists(dataset_name)
        existing_kg_meta = (
            self._get_dataset_corpus_metadata(dataset_name)
            if existing_chunk_count > 0
            else {}
        )
        require_existing_selection = (
            self.dataset_kg_scope == self.DATASET_KG_SCOPE_EVALUATION_SUBSET
            and not self.rebuild_kg
            and existing_chunk_count > 0
            and existing_kg_meta.get("uses_global_corpus") is not True
            and bool(existing_kg_meta.get("selection_key"))
        )
        try:
            selected_question_ids = self._resolve_selected_question_ids(
                dataset_name,
                require_existing=require_existing_selection,
            )
        except Exception as e:
            logging.error(
                "Failed to resolve persisted question subset for %s: %s",
                dataset_name,
                e,
            )
            return {"dataset": dataset_name, "error": str(e)}
        subset_meta = self._subset_metadata(dataset_name)
        corpus_profile = self._dataset_corpus_profile(dataset_name)
        has_shared_corpus = build_global_corpus_passages(dataset_name) is not None

        questions = []
        missing_questions: List[str] = []
        for question_id in selected_question_ids:
            q_data = dataset.get(question_id)
            if not isinstance(q_data, dict):
                missing_questions.append(question_id)
                continue
            contexts = self._extract_contexts_from_question(q_data)
            if contexts or has_shared_corpus:
                questions.append((question_id, q_data))
            else:
                missing_questions.append(question_id)
        if missing_questions:
            preview = ", ".join(missing_questions[:5])
            logging.warning(
                "Dataset %s is missing %d selected questions after adapter normalization: %s",
                dataset_name,
                len(missing_questions),
                preview,
            )
        
        # Check for leakage — hits the cache if already validated by _build_dataset_kg / _build_eval_records.
        try:
            _, _, _leakage_detected = self._get_normalized_dataset(dataset_name)
        except Exception as e:
            logging.warning(
                "Leakage validation failed for dataset '%s'; proceeding with leakage_detected=False: %s",
                dataset_name,
                e,
            )
            _leakage_detected = False

        results = {
            "dataset": dataset_name,
            "config": {
                "name": config_name,
                "similarity_threshold": similarity_threshold,
                "max_chunks": max_chunks,
                "retrieval_temperature": retrieval_temperature,
                "retrieval_shortlist_factor": retrieval_shortlist_factor,
                "retrieval_variant": config.get("retrieval_variant", ""),
                "retrieval_stack": dict(config.get("retrieval_stack", {}) or {}),
                "kg_system": dict(config.get("kg_system", {}) or {}),
                "executed_systems": list((config or {}).get("executed_systems") or ["vanilla_rag", "kg_rag"]),
            },
            "evaluation_mode": self.evaluation_mode,
            "subset_seed": self.subset_seed,
            "selection_file": str(self._selection_path(dataset_name)),
            "selection_key": subset_meta.get("selection_key", ""),
            "subset_hash": subset_meta.get("subset_hash", ""),
            "subset_tag": subset_meta.get("subset_tag", ""),
            "subset_id": subset_meta.get("subset_id", ""),
            "selected_question_ids": selected_question_ids,
            "question_context_role": corpus_profile.get("question_context_role", ""),
            "requires_shared_corpus_for_fair_retrieval": bool(
                corpus_profile.get("requires_shared_corpus_for_fair_retrieval", False)
            ),
            "total_questions": len(questions),
            "vanilla_rag_correct": 0,
            "kg_rag_correct": 0,
            "leakage_detected": _leakage_detected,
            "details": []
        }

        # Accumulate per-metric compute times for averaging
        from collections import defaultdict as _defaultdict
        vanilla_compute_times: dict = _defaultdict(list)
        kg_compute_times: dict = _defaultdict(list)
        vanilla_retrieval_overlaps: List[float] = []
        kg_retrieval_overlaps: List[float] = []

        # ── Checkpoint / resume ───────────────────────────────────────────
        # Write a checkpoint file after every question so a crash doesn't
        # lose completed work.  On re-run, already-processed question IDs
        # are skipped and their results are merged back in.
        _ckpt_dir = Path("results") / "checkpoints"
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
        _safe_dataset = re.sub(r"[^a-zA-Z0-9._-]+", "_", dataset_name)
        _safe_config  = re.sub(r"[^a-zA-Z0-9._-]+", "_", config_name)
        _safe_subset = self._sanitize_for_filename(
            subset_meta.get("subset_tag", f"n{self.num_samples}_seed{self.subset_seed}")
        )
        _safe_exec = self._execution_signature(config)
        _ckpt_path = _ckpt_dir / f"{_safe_dataset}_{_safe_subset}_{_safe_config}_{_safe_exec}.jsonl"

        if self.rebuild_kg and _ckpt_path.exists():
            archived_name = (
                f"{_ckpt_path.stem}.pre_rebuild_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                f"{_ckpt_path.suffix}"
            )
            archived_path = _ckpt_path.with_name(archived_name)
            shutil.move(str(_ckpt_path), str(archived_path))
            logging.warning(
                "Archived checkpoint %s -> %s before evaluating rebuilt KG to avoid mixing graph versions.",
                _ckpt_path,
                archived_path,
            )

        # Load existing checkpoint entries
        _seen_ids: set = set()
        if _ckpt_path.exists():
            with _ckpt_path.open() as _f:
                for _line_num, _line in enumerate(_f, start=1):
                    try:
                        _entry = json.loads(_line)
                        _entry = self._canonicalize_detail_row(_entry)
                        _qid = str(_entry.get("question_id", ""))
                        if _qid and _qid not in _seen_ids:
                            results["details"].append(_entry)
                            _seen_ids.add(_qid)
                            if _entry.get("vanilla_correct"):
                                results["vanilla_rag_correct"] += 1
                            if _entry.get("kg_correct"):
                                results["kg_rag_correct"] += 1
                    except Exception as e:
                        logging.warning(
                            "Skipping malformed checkpoint line %d in %s: %s | line=%r",
                            _line_num,
                            _ckpt_path,
                            e,
                            _line[:200],
                        )
            if _seen_ids:
                logging.info(
                    "Checkpoint: resuming from %d previously completed questions (%s)",
                    len(_seen_ids), _ckpt_path,
                )

        _ckpt_file = _ckpt_path.open("a")  # append mode
        executed_systems = self._executed_system_map(config)

        for q_idx, (q_id, q_data) in enumerate(questions):
            question = self._get_question_text(q_data)
            expected_answer = self._get_answer_from_question(q_data)
            hop_count = self._infer_hop_count(dataset_name, str(q_id), q_data if isinstance(q_data, dict) else {})
            hop_bucket = self._hop_bucket_label(hop_count)

            if not question or not expected_answer:
                continue

            # Skip already-processed questions (checkpoint resume)
            if str(q_id) in _seen_ids:
                logging.info(f"[{q_idx+1}/{len(questions)}] Skipping (checkpoint): {question[:50]}...")
                continue

            logging.info(f"[{q_idx+1}/{len(questions)}] Processing: {question[:50]}...")

            # Gold contexts are extracted but only used for KG construction (not inference).
            # Both systems must retrieve on their own — this is the key experimental condition:
            # vanilla RAG retrieves varying chunks per sample; KG-RAG always returns the same graph.
            question_contexts = self._extract_contexts_from_question(q_data, dataset_name=dataset_name)
            options, task_type = self._extract_options_and_task_type(q_data)
            aliases = q_data.get("aliases", [])
            answer_instructions = build_answer_instructions(
                dataset_name,
                task_type,
                options=options,
            )

            # Run only the canonical system pairing for final_pair:
            # dense_floor -> vanilla_rag, kg_entity_first -> kg_rag.
            vanilla_enabled = bool(executed_systems.get("vanilla_rag", True))
            kg_enabled = bool(executed_systems.get("kg_rag", True))

            # Run Vanilla RAG
            vanilla_result = {}
            _question_id = (
                str(q_id)
                if dataset_name in self.QUESTION_SCOPED_DATASETS
                else None
            )
            if vanilla_enabled:
                try:
                    with self._temporary_env(retrieval_env_overrides):
                        vanilla_result = vanilla_rag.generate_response(
                            question=question,
                            llm=self.accuracy_llm,
                            similarity_threshold=similarity_threshold,
                            max_chunks=max_chunks,
                            kg_name=dataset_name,
                            extra_context_texts=None,  # no gold context — retrieval competes
                            answer_instructions=answer_instructions,
                            question_id=_question_id,
                            retrieval_temperature=retrieval_temperature,
                            retrieval_shortlist_factor=retrieval_shortlist_factor,
                            retrieval_sample_id=0,
                        )
                    vanilla_response_raw = vanilla_result.get("response", "")
                    vanilla_response = normalize_answer_to_contract(
                        dataset_name,
                        task_type,
                        vanilla_response_raw,
                        question=question,
                    )
                    if vanilla_response != vanilla_response_raw:
                        vanilla_result["response_raw"] = vanilla_response_raw
                        vanilla_result["response"] = vanilla_response
                    vanilla_response = vanilla_response.lower()
                except Exception as e:
                    logging.error(f"Vanilla RAG error: {e}")
                    vanilla_response = ""

                vanilla_generation_failed = self._is_generation_failure(
                    vanilla_result,
                    vanilla_response,
                    expected_answer=expected_answer,
                )

                self._validate_retrieval_scope(
                    rag_result=vanilla_result,
                    dataset_name=dataset_name,
                    system_name="Vanilla RAG",
                    question_id=_question_id,
                )
            else:
                vanilla_response = ""
                vanilla_generation_failed = True

            # Run KG-RAG
            _dataset_max_hops = self.DATASET_MAX_HOPS.get(dataset_name, self.DEFAULT_MAX_HOPS)
            kg_result = {}
            if kg_enabled:
                try:
                    # Binary and single-document questions don't decompose into
                    # sub-questions — skip iterative decomposition to avoid 3-4
                    # extra LLM calls, while keeping full graph traversal depth.
                    _allow_decomposition = task_type not in {"binary"}
                    if "allow_decomposition" in kg_generation:
                        _allow_decomposition = bool(kg_generation["allow_decomposition"])
                    _kg_runtime_guardrail = kg_generation.get("runtime_guardrail")
                    with self._temporary_env(retrieval_env_overrides):
                        kg_result = kg_rag.generate_response(
                            question=question,
                            llm=self.accuracy_llm,
                            similarity_threshold=similarity_threshold,
                            max_chunks=max_chunks,
                            kg_name=dataset_name,
                            extra_context_texts=None,  # no gold context — retrieval competes
                            max_hops=_dataset_max_hops,
                            answer_instructions=answer_instructions,
                            question_id=_question_id,
                            allow_decomposition=_allow_decomposition,
                            runtime_guardrail=_kg_runtime_guardrail,
                            retrieval_temperature=retrieval_temperature,
                            retrieval_shortlist_factor=retrieval_shortlist_factor,
                            retrieval_sample_id=0,
                        )
                    kg_response_raw = kg_result.get("response", "")
                    kg_response = normalize_answer_to_contract(
                        dataset_name,
                        task_type,
                        kg_response_raw,
                        question=question,
                    )
                    if kg_response != kg_response_raw:
                        kg_result["response_raw"] = kg_response_raw
                        kg_result["response"] = kg_response
                    kg_response = kg_response.lower()
                except Exception as e:
                    logging.error(f"KG-RAG error: {e}")
                    kg_response = ""

                kg_generation_failed = self._is_generation_failure(
                    kg_result,
                    kg_response,
                    expected_answer=expected_answer,
                )

                self._validate_retrieval_scope(
                    rag_result=kg_result,
                    dataset_name=dataset_name,
                    system_name="KG-RAG",
                    question_id=_question_id,
                )
            else:
                kg_response = ""
                kg_generation_failed = True
            
            # Check correctness
            vanilla_correct = False
            kg_correct = False
            if not vanilla_generation_failed:
                vanilla_correct = self._is_answer_correct(
                    expected_answer,
                    vanilla_response,
                    question=question,
                    options=options,
                    task_type=task_type,
                    aliases=aliases,
                )
            if not kg_generation_failed:
                kg_correct = self._is_answer_correct(
                    expected_answer,
                    kg_response,
                    question=question,
                    options=options,
                    task_type=task_type,
                    aliases=aliases,
                )

            vanilla_answer_em = 0.0
            vanilla_answer_f1 = 0.0
            kg_answer_em = 0.0
            kg_answer_f1 = 0.0
            if supports_official_answer_metrics(dataset_name):
                if not vanilla_generation_failed:
                    vanilla_answer_em, vanilla_answer_f1 = compute_answer_em_f1(
                        vanilla_response,
                        expected_answer,
                        aliases=aliases,
                    )
                if not kg_generation_failed:
                    kg_answer_em, kg_answer_f1 = compute_answer_em_f1(
                        kg_response,
                        expected_answer,
                        aliases=aliases,
                    )
            
            # Track correctness
            if vanilla_correct:
                results["vanilla_rag_correct"] += 1
            if kg_correct:
                results["kg_rag_correct"] += 1

            # NOTE: per-question scalar logging intentionally removed to reduce noisy W&B line charts.
            # Per-question visibility is provided via W&B tables.
            detail_entry = {
                "question_id": q_id,
                "hop_count": hop_count,
                "hop_bucket": hop_bucket,
                "question": question[:500],
                "expected": expected_answer,
                "task_type": task_type,
                "vanilla_correct": vanilla_correct,
                "kg_correct": kg_correct,
                "vanilla_response": vanilla_response[:500],
                "kg_response": kg_response[:500],
                "vanilla_generation_failed": vanilla_generation_failed,
                "kg_generation_failed": kg_generation_failed,
                "vanilla_system_skipped": not vanilla_enabled,
                "kg_system_skipped": not kg_enabled,
                "vanilla_answer_em": vanilla_answer_em,
                "kg_answer_em": kg_answer_em,
                "vanilla_answer_f1": vanilla_answer_f1,
                "kg_answer_f1": kg_answer_f1,
                "vanilla_search_method": str(vanilla_result.get("context", {}).get("search_method", "") or ""),
                "kg_search_method": str(kg_result.get("context", {}).get("search_method", "") or ""),
                "kg_retrieval_route": str(kg_result.get("context", {}).get("retrieval_route", "") or ""),
                "kg_route_reason": str(kg_result.get("context", {}).get("route_reason", "") or ""),
                "kg_retrieval_mode_config": str(
                    kg_result.get("context", {})
                    .get("diagnostics", {})
                    .get("retrieval_mode_config", "")
                    or ""
                ),
                "vanilla_late_interaction_stage_applied": bool(
                    vanilla_result.get("late_interaction_stage", {}).get("applied", False)
                ),
                "kg_late_interaction_stage_applied": bool(
                    kg_result.get("late_interaction_stage", {}).get("applied", False)
                ),
                "vanilla_late_interaction_applied": bool(
                    vanilla_result.get("late_interaction", {}).get("applied", False)
                ),
                "kg_late_interaction_applied": bool(
                    kg_result.get("late_interaction", {}).get("applied", False)
                ),
                "vanilla_reranker_applied": bool(
                    vanilla_result.get("reranker", {}).get("applied", False)
                ),
                "kg_reranker_applied": bool(
                    kg_result.get("reranker", {}).get("applied", False)
                ),
                # KG retrieval meta-signals (needed for hop_depth / grounding_threshold ablations)
                "grounding_quality": float(
                    kg_result.get("context", {}).get("grounding_quality", 0.0) or 0.0
                ),
                "seed_entity_count": kg_result.get("context", {}).get("seed_entity_count", 0),
                "retrieval_temperature": retrieval_temperature,
            }

            if self.compute_metrics:
                # Collect multiple responses for the canonical uncertainty metrics,
                # using the same dataset-scoped KG filter as the main responses.
                if vanilla_generation_failed:
                    vanilla_samples = []
                    vanilla_chunk_texts = self._extract_chunk_texts_from_result(vanilla_result)
                    vanilla_retrieval_overlap = 0.0
                    vanilla_uncertainty = self._default_uncertainty_metrics()
                else:
                    with self._temporary_env(retrieval_env_overrides):
                        vanilla_samples, vanilla_chunk_texts, vanilla_retrieval_overlap = self._collect_sample_responses(
                            rag_system=vanilla_rag,
                            question=question,
                            answer_instructions=answer_instructions,
                            base_result=vanilla_result,
                            similarity_threshold=similarity_threshold,
                            max_chunks=max_chunks,
                            kg_name=dataset_name,
                            question_id=_question_id,
                            retrieval_temperature=retrieval_temperature,
                            retrieval_shortlist_factor=retrieval_shortlist_factor,
                        )
                    vanilla_context_str = "\n\n".join(vanilla_chunk_texts[:3]) if vanilla_chunk_texts else ""
                    vanilla_uncertainty = compute_all_uncertainty_metrics(
                        responses=vanilla_samples,
                        prompt=question,
                        context=vanilla_context_str,
                    )
                    for _m, _t in vanilla_uncertainty.get("compute_times", {}).items():
                        vanilla_compute_times[_m].append(_t)

                if kg_generation_failed:
                    kg_samples = []
                    kg_chunk_texts = self._extract_chunk_texts_from_result(kg_result)
                    kg_retrieval_overlap = 0.0
                    kg_graph_state_traces = []
                    kg_graph_state_diversity = graph_state_diversity([])
                    kg_uncertainty = self._default_uncertainty_metrics()
                else:
                    with self._temporary_env(retrieval_env_overrides):
                        (
                            kg_samples,
                            kg_chunk_texts,
                            kg_retrieval_overlap,
                            kg_graph_state_traces,
                        ) = self._collect_sample_responses(
                            rag_system=kg_rag,
                            question=question,
                            answer_instructions=answer_instructions,
                            base_result=kg_result,
                            similarity_threshold=similarity_threshold,
                            max_chunks=max_chunks,
                            kg_name=dataset_name,
                            question_id=_question_id,
                            retrieval_temperature=retrieval_temperature,
                            retrieval_shortlist_factor=retrieval_shortlist_factor,
                            return_graph_state_traces=True,
                            generate_kwargs={
                                "allow_decomposition": bool(
                                    kg_generation.get("allow_decomposition", task_type not in {"binary"})
                                ),
                                "runtime_guardrail": kg_generation.get("runtime_guardrail"),
                            },
                        )
                    kg_graph_state_diversity = graph_state_diversity(kg_graph_state_traces)
                    kg_context_str = "\n\n".join(kg_chunk_texts[:3]) if kg_chunk_texts else ""
                    kg_uncertainty = compute_all_uncertainty_metrics(
                        responses=kg_samples,
                        prompt=question,
                        context=kg_context_str,
                    )
                    for _m, _t in kg_uncertainty.get("compute_times", {}).items():
                        kg_compute_times[_m].append(_t)

                vanilla_retrieval_overlaps.append(float(vanilla_retrieval_overlap or 0.0))
                kg_retrieval_overlaps.append(float(kg_retrieval_overlap or 0.0))
                detail_entry.update({
                    "kg_graph_state_sample_count": kg_graph_state_diversity.get("sample_count", 0),
                    "kg_seed_entity_entropy": kg_graph_state_diversity.get("seed_entity_entropy", 0.0),
                    "kg_seed_entity_entropy_norm": kg_graph_state_diversity.get("seed_entity_entropy_norm", 0.0),
                    "kg_path_entropy": kg_graph_state_diversity.get("path_entropy", 0.0),
                    "kg_path_entropy_norm": kg_graph_state_diversity.get("path_entropy_norm", 0.0),
                    "kg_subgraph_entropy": kg_graph_state_diversity.get("subgraph_entropy", 0.0),
                    "kg_subgraph_entropy_norm": kg_graph_state_diversity.get("subgraph_entropy_norm", 0.0),
                    "kg_chunk_entropy": kg_graph_state_diversity.get("chunk_entropy", 0.0),
                    "kg_chunk_entropy_norm": kg_graph_state_diversity.get("chunk_entropy_norm", 0.0),
                    "kg_seed_entity_jaccard": kg_graph_state_diversity.get("seed_entity_jaccard", 0.0),
                    "kg_path_jaccard": kg_graph_state_diversity.get("path_jaccard", 0.0),
                    "kg_subgraph_jaccard": kg_graph_state_diversity.get("subgraph_jaccard", 0.0),
                    "kg_chunk_jaccard": kg_graph_state_diversity.get("chunk_jaccard", 0.0),
                    "kg_dominant_seed_entity_id": kg_graph_state_diversity.get("dominant_seed_entity_id", ""),
                    "kg_dominant_seed_entity_fraction": kg_graph_state_diversity.get("dominant_seed_entity_fraction", 0.0),
                    # Keep bounded raw traces for targeted debugging/intervention selection.
                    "kg_graph_state_traces": kg_graph_state_traces[: self.entropy_samples],
                })

                # Structural metrics: graph-based uncertainty (no extra LLM sampling).
                _structural_max_hops = self._structural_metric_max_hops(
                    dataset_name,
                    _dataset_max_hops,
                )
                _graph_kwargs = dict(
                    neo4j_uri=self.neo4j_uri,
                    neo4j_user=self.neo4j_user,
                    neo4j_password=self.neo4j_password,
                    kg_name=dataset_name,
                    question_id=_question_id,
                    max_hops=_structural_max_hops,
                )
                _t0 = time.perf_counter()
                _vanilla_gps = (
                    {"score": 0.5, "null_reason": "generation_failed"} if vanilla_generation_failed else
                    compute_graph_path_support_detailed(question=question, answer=vanilla_response, **_graph_kwargs)
                )
                _kg_gps = (
                    {"score": 0.5, "null_reason": "generation_failed"} if kg_generation_failed else
                    compute_graph_path_support_detailed(question=question, answer=kg_response, **_graph_kwargs)
                )
                vanilla_graph_path_uq = _vanilla_gps["score"]
                kg_graph_path_uq = _kg_gps["score"]
                vanilla_gps_null_reason = _vanilla_gps.get("null_reason") or ""
                kg_gps_null_reason = _kg_gps.get("null_reason") or ""
                _elapsed = time.perf_counter() - _t0
                vanilla_compute_times["graph_path_support"].append(_elapsed)
                kg_compute_times["graph_path_support"].append(_elapsed)

                graph_path_disagreement = compute_graph_path_disagreement(
                    question=question, **_graph_kwargs
                )
                _competing_kwargs = {k: v for k, v in _graph_kwargs.items() if k != "max_hops"}
                vanilla_competing_uq = (
                    0.0 if vanilla_generation_failed else
                    compute_competing_answer_alternatives(
                        question=question, answer=vanilla_response, **_competing_kwargs
                    )
                )
                kg_competing_uq = (
                    0.0 if kg_generation_failed else
                    compute_competing_answer_alternatives(
                        question=question, answer=kg_response, **_competing_kwargs
                    )
                )
                evidence_vn_uq = compute_evidence_vn_entropy(question=question, **_graph_kwargs)
                subgraph_info_uq = compute_subgraph_informativeness(question=question, **_graph_kwargs)
                _t0 = time.perf_counter()
                _vanilla_sps = (
                    {"score": 0.5, "null_reason": "generation_failed"} if vanilla_generation_failed else
                    compute_subgraph_perturbation_stability_detailed(
                        question=question,
                        answer=vanilla_response,
                        **_graph_kwargs,
                    )
                )
                _kg_sps = (
                    {"score": 0.5, "null_reason": "generation_failed"} if kg_generation_failed else
                    compute_subgraph_perturbation_stability_detailed(
                        question=question,
                        answer=kg_response,
                        **_graph_kwargs,
                    )
                )
                vanilla_css_uq = _vanilla_sps["score"]
                kg_css_uq = _kg_sps["score"]
                vanilla_sps_null_reason = _vanilla_sps.get("null_reason") or ""
                kg_sps_null_reason = _kg_sps.get("null_reason") or ""
                _elapsed = time.perf_counter() - _t0
                vanilla_compute_times["subgraph_perturbation_stability"].append(_elapsed)
                kg_compute_times["subgraph_perturbation_stability"].append(_elapsed)

                _t0 = time.perf_counter()
                # SEU/ECU measure evidence-answer entailment — they do not depend on
                # sampling so they work even under context collapse. Crucially, they
                # are also meaningful when generation failed: the retrieved chunks
                # still exist and we can measure whether they entail the expected
                # answer, using it as the hypothesis when the model abstained.
                _vanilla_seu_answer = vanilla_response if not vanilla_generation_failed else expected_answer
                _kg_seu_answer      = kg_response      if not kg_generation_failed      else expected_answer
                _vanilla_seu_chunks = vanilla_chunk_texts or self._extract_chunk_texts_from_result(vanilla_result)
                _kg_seu_chunks      = kg_chunk_texts      or self._extract_chunk_texts_from_result(kg_result)
                vanilla_seu = (
                    self.UNCERTAINTY_METRIC_DEFAULTS["support_entailment_uncertainty"] if not _vanilla_seu_chunks else
                    compute_support_entailment_uncertainty(
                        chunks=_vanilla_seu_chunks,
                        answer=_vanilla_seu_answer,
                    )
                )
                kg_seu = (
                    self.UNCERTAINTY_METRIC_DEFAULTS["support_entailment_uncertainty"] if not _kg_seu_chunks else
                    compute_support_entailment_uncertainty(
                        chunks=_kg_seu_chunks,
                        answer=_kg_seu_answer,
                    )
                )
                _elapsed = time.perf_counter() - _t0
                vanilla_compute_times["support_entailment_uncertainty"].append(_elapsed)
                kg_compute_times["support_entailment_uncertainty"].append(_elapsed)

                _t0 = time.perf_counter()
                vanilla_ecu = (
                    self.UNCERTAINTY_METRIC_DEFAULTS["evidence_conflict_uncertainty"] if not _vanilla_seu_chunks else
                    compute_evidence_conflict_uncertainty(
                        chunks=_vanilla_seu_chunks,
                        answer=_vanilla_seu_answer,
                    )
                )
                kg_ecu = (
                    self.UNCERTAINTY_METRIC_DEFAULTS["evidence_conflict_uncertainty"] if not _kg_seu_chunks else
                    compute_evidence_conflict_uncertainty(
                        chunks=_kg_seu_chunks,
                        answer=_kg_seu_answer,
                    )
                )
                _elapsed = time.perf_counter() - _t0
                vanilla_compute_times["evidence_conflict_uncertainty"].append(_elapsed)
                kg_compute_times["evidence_conflict_uncertainty"].append(_elapsed)

                detail_entry.update({
                    "vanilla_retrieval_overlap": float(vanilla_retrieval_overlap or 0.0),
                    "kg_retrieval_overlap": float(kg_retrieval_overlap or 0.0),
                    "vanilla_semantic_entropy": vanilla_uncertainty.get("semantic_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["semantic_entropy"]),
                    "kg_semantic_entropy": kg_uncertainty.get("semantic_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["semantic_entropy"]),
                    "vanilla_discrete_semantic_entropy": vanilla_uncertainty.get("discrete_semantic_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["discrete_semantic_entropy"]),
                    "kg_discrete_semantic_entropy": kg_uncertainty.get("discrete_semantic_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["discrete_semantic_entropy"]),
                    "vanilla_p_true": vanilla_uncertainty.get("p_true", self.UNCERTAINTY_METRIC_DEFAULTS["p_true"]),
                    "kg_p_true": kg_uncertainty.get("p_true", self.UNCERTAINTY_METRIC_DEFAULTS["p_true"]),
                    "vanilla_selfcheckgpt": vanilla_uncertainty.get("selfcheckgpt", self.UNCERTAINTY_METRIC_DEFAULTS["selfcheckgpt"]),
                    "kg_selfcheckgpt": kg_uncertainty.get("selfcheckgpt", self.UNCERTAINTY_METRIC_DEFAULTS["selfcheckgpt"]),
                    "vanilla_sre_uq": vanilla_uncertainty.get("sre_uq", self.UNCERTAINTY_METRIC_DEFAULTS["sre_uq"]),
                    "kg_sre_uq": kg_uncertainty.get("sre_uq", self.UNCERTAINTY_METRIC_DEFAULTS["sre_uq"]),
                    "vanilla_vn_entropy": vanilla_uncertainty.get("vn_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["vn_entropy"]),
                    "kg_vn_entropy": kg_uncertainty.get("vn_entropy", self.UNCERTAINTY_METRIC_DEFAULTS["vn_entropy"]),
                    "vanilla_sd_uq": vanilla_uncertainty.get("sd_uq", self.UNCERTAINTY_METRIC_DEFAULTS["sd_uq"]),
                    "kg_sd_uq": kg_uncertainty.get("sd_uq", self.UNCERTAINTY_METRIC_DEFAULTS["sd_uq"]),
                    "vanilla_graph_path_support": vanilla_graph_path_uq,
                    "kg_graph_path_support": kg_graph_path_uq,
                    "vanilla_graph_path_support_null_reason": vanilla_gps_null_reason,
                    "kg_graph_path_support_null_reason": kg_gps_null_reason,
                    "vanilla_graph_path_disagreement": graph_path_disagreement,
                    "kg_graph_path_disagreement": graph_path_disagreement,
                    "vanilla_competing_answer_alternatives": vanilla_competing_uq,
                    "kg_competing_answer_alternatives": kg_competing_uq,
                    "vanilla_evidence_vn_entropy": evidence_vn_uq,
                    "kg_evidence_vn_entropy": evidence_vn_uq,
                    "vanilla_subgraph_informativeness": subgraph_info_uq,
                    "kg_subgraph_informativeness": subgraph_info_uq,
                    "vanilla_subgraph_perturbation_stability": vanilla_css_uq,
                    "kg_subgraph_perturbation_stability": kg_css_uq,
                    "vanilla_subgraph_perturbation_stability_null_reason": vanilla_sps_null_reason,
                    "kg_subgraph_perturbation_stability_null_reason": kg_sps_null_reason,
                    "vanilla_support_entailment_uncertainty": vanilla_seu,
                    "kg_support_entailment_uncertainty": kg_seu,
                    "vanilla_evidence_conflict_uncertainty": vanilla_ecu,
                    "kg_evidence_conflict_uncertainty": kg_ecu,
                })

            # Multi-temperature entropy sweep: collect N samples at T=0, 0.5, 1.0
            # and store per-temperature uncertainty metrics with a _t{suffix} key.
            # This lets downstream analysis show whether geometric metrics (VN-Entropy,
            # SD-UQ) retain discriminative signal across temperatures while NLI-based
            # metrics collapse at all temperatures under KG-RAG context determinism.
            if self.compute_metrics and self.multi_temperature and not vanilla_generation_failed and not kg_generation_failed:
                _mt_metrics = ["sre_uq", "vn_entropy", "sd_uq", "selfcheckgpt", "discrete_semantic_entropy"]
                vanilla_context_str_mt = "\n\n".join(vanilla_chunk_texts[:3]) if vanilla_chunk_texts else ""
                kg_context_str_mt = "\n\n".join(kg_chunk_texts[:3]) if kg_chunk_texts else ""
                for _t_val in self.MULTI_TEMPERATURES:
                    _t_str = ("t" + str(_t_val).replace(".", ""))  # 0.0→t00, 0.5→t05, 1.0→t10
                    if abs(_t_val - self.temperature) < 1e-9:
                        # Already computed — copy existing values to avoid extra API calls
                        for _mk in _mt_metrics:
                            detail_entry["vanilla_%s_%s" % (_mk, _t_str)] = detail_entry.get("vanilla_%s" % _mk, 0.0)
                            detail_entry["kg_%s_%s" % (_mk, _t_str)] = detail_entry.get("kg_%s" % _mk, 0.0)
                    else:
                        try:
                            _t_llm = self._llm_for_temperature(_t_val)
                            with self._temporary_env(retrieval_env_overrides):
                                _v_samps, _, _ = self._collect_sample_responses(
                                    rag_system=vanilla_rag,
                                    question=question,
                                    answer_instructions=answer_instructions,
                                    similarity_threshold=similarity_threshold,
                                    max_chunks=max_chunks,
                                    kg_name=dataset_name,
                                    question_id=_question_id,
                                    llm=_t_llm,
                                )
                                _k_samps, _, _ = self._collect_sample_responses(
                                    rag_system=kg_rag,
                                    question=question,
                                    answer_instructions=answer_instructions,
                                    similarity_threshold=similarity_threshold,
                                    max_chunks=max_chunks,
                                    kg_name=dataset_name,
                                    question_id=_question_id,
                                    llm=_t_llm,
                                )
                            _v_uq = compute_all_uncertainty_metrics(
                                responses=_v_samps, prompt=question, context=vanilla_context_str_mt
                            )
                            _k_uq = compute_all_uncertainty_metrics(
                                responses=_k_samps, prompt=question, context=kg_context_str_mt
                            )
                            for _mk in _mt_metrics:
                                detail_entry["vanilla_%s_%s" % (_mk, _t_str)] = _v_uq.get(_mk, 0.0)
                                detail_entry["kg_%s_%s" % (_mk, _t_str)] = _k_uq.get(_mk, 0.0)
                        except Exception as _e:
                            logging.warning("Multi-temperature sweep at T=%s failed: %s", _t_val, _e)

            results["details"].append(detail_entry)

            # Write checkpoint immediately so a crash doesn't lose this question
            try:
                _ckpt_file.write(
                    json.dumps(self._canonicalize_detail_row(detail_entry), default=str) + "\n"
                )
                _ckpt_file.flush()
            except Exception as _ckpt_err:
                logging.warning("Checkpoint write failed (non-fatal): %s", _ckpt_err)

            # Real-time W&B logging — one step per question so the dashboard
            # updates as the experiment runs rather than only at the end.
            if self.wandb_run:
                try:
                    _running_stats = compute_accuracy_breakdown(results["details"])
                    _n_done = len(results["details"])
                    _running_vanilla_acc = _running_stats["vanilla_accuracy_excluding_errors"]
                    _running_kg_acc = _running_stats["kg_accuracy_excluding_errors"]
                    self.wandb_run.log({
                        f"live/{dataset_name}/vanilla_accuracy":  _running_vanilla_acc,
                        f"live/{dataset_name}/kg_accuracy":       _running_kg_acc,
                        f"live/{dataset_name}/vanilla_correct":   int(vanilla_correct),
                        f"live/{dataset_name}/kg_correct":        int(kg_correct),
                        f"live/{dataset_name}/questions_done":    _n_done,
                        f"live/{dataset_name}/grounding_quality": detail_entry.get("grounding_quality", 0.0),
                    }, step=_n_done)
                except Exception as _wb_err:
                    logging.debug("W&B live log failed (non-fatal): %s", _wb_err)

            logging.info(f"  Vanilla: {vanilla_response[:40]}... (correct: {vanilla_correct})")
            logging.info(f"  KG-RAG:  {kg_response[:40]}... (correct: {kg_correct})")
        
        # Close checkpoint file; delete it now that the run completed cleanly
        try:
            _ckpt_file.close()
            _ckpt_path.unlink(missing_ok=True)
            logging.info("Checkpoint deleted after clean completion: %s", _ckpt_path)
        except Exception:
            pass

        # Calculate metrics using actually evaluated rows.
        total = len(results["details"])
        results.update(compute_accuracy_breakdown(results["details"]))
        self._apply_clean_accuracy_reporting(results)
        routing_distribution = self._compute_kg_routing_distribution(results["details"])
        results["kg_routing_distribution"] = routing_distribution
        results["kg_pure_entity_first_rate"] = routing_distribution.get("pure_entity_first_rate", 0.0)
        results["kg_dense_fallback_rate"] = routing_distribution.get("dense_fallback_rate", 0.0)
        results["kg_unknown_route_rate"] = routing_distribution.get("unknown_route_rate", 0.0)
        if total > 0:
            results["vanilla_answer_em"] = sum(
                float(d.get("vanilla_answer_em", 0.0)) for d in results["details"]
            ) / total
            results["kg_answer_em"] = sum(
                float(d.get("kg_answer_em", 0.0)) for d in results["details"]
            ) / total
            results["vanilla_answer_f1"] = sum(
                float(d.get("vanilla_answer_f1", 0.0)) for d in results["details"]
            ) / total
            results["kg_answer_f1"] = sum(
                float(d.get("kg_answer_f1", 0.0)) for d in results["details"]
            ) / total
        else:
            results["vanilla_answer_em"] = 0.0
            results["kg_answer_em"] = 0.0
            results["vanilla_answer_f1"] = 0.0
            results["kg_answer_f1"] = 0.0
        if total > 0 and self.compute_metrics:
            metric_defaults = dict(self.UNCERTAINTY_METRIC_DEFAULTS)
            for metric_name, default in metric_defaults.items():
                vanilla_vals = [d.get(f"vanilla_{metric_name}", default) for d in results["details"]]
                kg_vals = [d.get(f"kg_{metric_name}", default) for d in results["details"]]
                results[f"vanilla_avg_{metric_name}"] = (
                    sum(vanilla_vals) / len(vanilla_vals) if vanilla_vals else default
                )
                results[f"kg_avg_{metric_name}"] = (
                    sum(kg_vals) / len(kg_vals) if kg_vals else default
                )

            results["vanilla_avg_retrieval_overlap"] = (
                sum(vanilla_retrieval_overlaps) / len(vanilla_retrieval_overlaps)
                if vanilla_retrieval_overlaps else 0.0
            )
            results["kg_avg_retrieval_overlap"] = (
                sum(kg_retrieval_overlaps) / len(kg_retrieval_overlaps)
                if kg_retrieval_overlaps else 0.0
            )

            # ── AUROC / AUREC ─────────────────────────────────────────────
            auroc_aurec = compute_auroc_aurec(results["details"], metric_names=self.UNCERTAINTY_METRIC_NAMES)
            results["auroc_aurec"] = auroc_aurec
            for system_key, prefix in (("vanilla_rag", "vanilla"), ("kg_rag", "kg")):
                for metric_key, val in auroc_aurec.get(system_key, {}).items():
                    results[f"{prefix}_avg_{metric_key}"] = val

            # ── ECE ───────────────────────────────────────────────────────
            ece_results = compute_ece(results["details"], metric_names=self.UNCERTAINTY_METRIC_NAMES)
            results["ece"] = ece_results
            for system_key, prefix in (("vanilla_rag", "vanilla"), ("kg_rag", "kg")):
                for metric_key, val in ece_results.get(system_key, {}).items():
                    results[f"{prefix}_avg_{metric_key}"] = val

            # ── Precision@k ───────────────────────────────────────────────
            ppv_results = compute_precision_at_k(results["details"], metric_names=self.UNCERTAINTY_METRIC_NAMES)
            results["precision_at_k"] = ppv_results
            for system_key, prefix in (("vanilla_rag", "vanilla"), ("kg_rag", "kg")):
                for metric_key, val in ppv_results.get(system_key, {}).items():
                    results[f"{prefix}_avg_{metric_key}"] = val

            # ── Complementarity analysis (Zhang et al. 2025 methodology) ──
            # Classify each question into one of four quadrants:
            #   both_correct, vanilla_only, kg_only, neither_correct
            _details = results["details"]
            _both   = sum(1 for d in _details if d.get("vanilla_correct") and d.get("kg_correct"))
            _v_only = sum(1 for d in _details if d.get("vanilla_correct") and not d.get("kg_correct"))
            _k_only = sum(1 for d in _details if not d.get("vanilla_correct") and d.get("kg_correct"))
            _neither = sum(1 for d in _details if not d.get("vanilla_correct") and not d.get("kg_correct"))
            _n = max(1, len(_details))
            results["complementarity"] = {
                "both_correct":    _both,
                "vanilla_only":    _v_only,
                "kg_only":         _k_only,
                "neither_correct": _neither,
                "both_correct_pct":    round(_both   / _n * 100, 1),
                "vanilla_only_pct":    round(_v_only / _n * 100, 1),
                "kg_only_pct":         round(_k_only / _n * 100, 1),
                "neither_correct_pct": round(_neither / _n * 100, 1),
            }
            logging.info(
                "Complementarity: both=%d (%.0f%%), vanilla-only=%d (%.0f%%), "
                "kg-only=%d (%.0f%%), neither=%d (%.0f%%)",
                _both,    _both    / _n * 100,
                _v_only,  _v_only  / _n * 100,
                _k_only,  _k_only  / _n * 100,
                _neither, _neither / _n * 100,
            )

            # ── Query-type stratified accuracy & AUROC ────────────────────
            # Group detail entries by task_type and compute accuracy +
            # per-metric AUROC for each group.  Stored in results so
            # visualize_results.py can plot per-type breakdowns.
            from collections import defaultdict as _defaultdict
            _by_type: Dict[str, List[Dict]] = _defaultdict(list)
            for _d in _details:
                _tt = str(_d.get("task_type") or "unknown").strip() or "unknown"
                _by_type[_tt].append(_d)

            _type_stats: Dict[str, Any] = {}
            for _tt, _rows in _by_type.items():
                _acc_t = compute_accuracy_breakdown(_rows)
                _type_stats[_tt] = {
                    "n": len(_rows),
                    "vanilla_accuracy": round(_acc_t["vanilla_accuracy_excluding_errors"], 4),
                    "kg_accuracy": round(_acc_t["kg_accuracy_excluding_errors"], 4),
                    "vanilla_accuracy_raw": round(_acc_t["vanilla_accuracy"], 4),
                    "kg_accuracy_raw": round(_acc_t["kg_accuracy"], 4),
                }
                if self.compute_metrics:
                    _type_stats[_tt]["auroc_aurec"] = compute_auroc_aurec(
                        _rows, metric_names=self.UNCERTAINTY_METRIC_NAMES
                    )
            results["accuracy_by_task_type"] = _type_stats

        # ── Average per-metric compute times ──────────────────────────────
        results["accuracy_by_hop_count"] = compute_hop_accuracy_breakdown(
            results["details"],
            metric_names=(
                [
                    "sd_uq",
                    "vn_entropy",
                    "support_entailment_uncertainty",
                ]
                if self.compute_metrics
                else []
            ),
        )
        results["vanilla_avg_compute_times"] = {
            m: (sum(times) / len(times)) for m, times in vanilla_compute_times.items() if times
        }
        results["kg_avg_compute_times"] = {
            m: (sum(times) / len(times)) for m, times in kg_compute_times.items() if times
        }

        logging.info(
            "%s Results: Vanilla=%0.2f%% clean (%d answered, raw %0.2f%%), "
            "KG-RAG=%0.2f%% clean (%d answered, raw %0.2f%%)",
            dataset_name,
            100 * results.get("vanilla_accuracy", 0.0),
            int(results.get("vanilla_answered_questions", 0)),
            100 * results.get("vanilla_accuracy_raw", 0.0),
            100 * results.get("kg_accuracy", 0.0),
            int(results.get("kg_answered_questions", 0)),
            100 * results.get("kg_accuracy_raw", 0.0),
        )

        # Log per-question details table for this dataset/config
        self._log_question_table_to_wandb(
            dataset_name=dataset_name,
            config_name=config_name,
            subset_tag=subset_meta.get("subset_tag", "default"),
            details=results.get("details", []),
        )

        question_table_key = (
            f"tables/{dataset_name}/{subset_meta.get('subset_tag', 'default')}/"
            f"{config_name}/questions_and_responses"
        )
        question_details_path = self._write_question_details_file(
            dataset_name=dataset_name,
            config_name=config_name,
            subset_tag=subset_meta.get("subset_tag", "default"),
            details=results.get("details", []),
        )

        results["question_details_file"] = question_details_path
        results["question_table_key"] = question_table_key
        results["num_questions_logged"] = len(results.get("details", []))

        logging.info(
            f"Saved per-question Q/A + metrics to local file: {question_details_path}"
        )
        logging.info(
            f"Per-question table logged to W&B key: {question_table_key}"
        )

        # Small scalar to make table visibility explicit in W&B metric view.
        if self.wandb_run:
            self.wandb_run.log({
                f"tables/{dataset_name}/{config_name}/num_questions_logged": len(results.get("details", []))
            })
            self.wandb_run.summary[
                f"artifacts/{dataset_name}/{config_name}/questions_table_key"
            ] = question_table_key
            self.wandb_run.summary[
                f"artifacts/{dataset_name}/{config_name}/local_details_file"
            ] = question_details_path
        
        return results
    
    def run_pipeline(self, datasets: List[str] = None, output_dir: str = "results"):
        """Run the full evaluation pipeline"""
        if datasets is None:
            datasets = ["pubmedqa", "bioasq", "medhop", "hotpotqa", "2wikimultihopqa", "musique"]
            if "multihoprag" not in datasets:
                datasets.append("multihoprag")

        # ── Create run directory ─────────────────────────────────────────
        run_id = generate_run_id(
            datasets=datasets,
            num_samples=self.num_samples,
            evaluation_mode=self.evaluation_mode,
            dataset_kg_scope=self.dataset_kg_scope,
            rebuild_kg=getattr(self, "rebuild_kg", False),
        )
        output_root = Path(output_dir)
        run_dir = output_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "questions").mkdir(exist_ok=True)
        output_dir = str(run_dir)   # all outputs go here
        logging.info("Run ID: %s  →  %s", run_id, run_dir)

        # Write initial manifest (completed at end)
        manifest = {
            "run_id":       run_id,
            "created_at":   datetime.now().isoformat(),
            "datasets":     datasets,
            "num_samples":  self.num_samples,
            "subset_seed":  self.subset_seed,
            "evaluation_mode": self.evaluation_mode,
            "retrieval_study": self.retrieval_study or None,
            "kg_builder_profile": self.kg_builder_profile,
            "dataset_kg_scope": self.dataset_kg_scope,
            "allow_gold_evidence_contexts": self.allow_gold_evidence_contexts,
            "model":        os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
            "git_commit":   _git_commit(),
            "rebuild_kg":   getattr(self, "rebuild_kg", False),
            "selection_files": {},
            "subset_ids": {},
            "dataset_corpus_profiles": {},
            "accuracy":     {},   # filled in at end
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Initialize W&B
        wandb_entity = os.getenv("WANDB_ENTITY", "julka01")
        wandb_mode = os.getenv("WANDB_MODE", "online").strip().lower() or "online"
        run = wandb.init(
            entity=wandb_entity,
            project="mirage-kg-evaluation",
            name=run_id,
            config=manifest,
            mode=wandb_mode
        )
        self.wandb_run = run
        self._run_id  = run_id
        self._run_dir = run_dir
        
        all_results = []
        
        for dataset_name in datasets:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing dataset: {dataset_name}")
            logging.info(f"{'='*50}")
            manifest["dataset_corpus_profiles"][dataset_name] = self._dataset_corpus_profile(dataset_name)
            selection_path = self._selection_path(dataset_name)
            if selection_path.exists():
                manifest["selection_files"][dataset_name] = str(selection_path)
                manifest["subset_ids"][dataset_name] = self._subset_metadata(dataset_name).get("subset_id")
            
            # Step 1: Build KG for this dataset
            # Check if KG already exists for this specific dataset
            existing_chunk_count = self._dataset_kg_exists(dataset_name)
            allowed_by_policy, force_rebuild_existing_kg = self._enforce_dataset_corpus_policy(
                dataset_name,
                existing_chunk_count=existing_chunk_count,
            )
            if not allowed_by_policy:
                logging.error(
                    "Skipping %s because its current corpus setup is not safe for retrieval benchmarking.",
                    dataset_name,
                )
                continue
            should_rebuild_kg = self.rebuild_kg or force_rebuild_existing_kg
            if force_rebuild_existing_kg:
                if existing_chunk_count > 0:
                    logging.info(
                        "Deleting existing KG for %s because it does not satisfy the current corpus policy.",
                        dataset_name,
                    )
                    self._delete_dataset_kg(dataset_name)
                    existing_chunk_count = 0
            if not should_rebuild_kg and existing_chunk_count > 0:
                existing_kg_meta = self._get_dataset_corpus_metadata(dataset_name)
                try:
                    expected_contract = self._prepare_dataset_kg_contract(
                        dataset_name,
                        force_resample=False,
                    )
                except Exception as e:
                    logging.warning(
                        "Failed to compute KG compatibility contract for %s; "
                        "reusing existing KG as a legacy fallback: %s",
                        dataset_name,
                        e,
                    )
                else:
                    compatible, incompatibility_reasons = self._assess_dataset_kg_compatibility(
                        existing_kg_meta,
                        expected_contract["build_meta"],
                    )
                    if not compatible:
                        logging.warning(
                            "Existing KG for %s is incompatible with the current experiment "
                            "settings; rebuilding. %s",
                            dataset_name,
                            "; ".join(incompatibility_reasons),
                        )
                        self._delete_dataset_kg(dataset_name)
                        existing_chunk_count = 0
                        should_rebuild_kg = True
            
            if should_rebuild_kg:
                # --rebuild-kg: Delete existing KG and rebuild from scratch
                if existing_chunk_count > 0:
                    logging.info(
                        f"Rebuilding KG for {dataset_name} "
                        f"({'--rebuild-kg set' if self.rebuild_kg else 'corpus policy changed'}). "
                        f"Deleting existing {existing_chunk_count} chunks first."
                    )
                    self._delete_dataset_kg(dataset_name)
                kg_built = self._build_kg_for_dataset(dataset_name)
                if not kg_built:
                    logging.error(f"Skipping {dataset_name} due to KG build failure")
                    continue
                if not self._verify_kg_quality(dataset_name):
                    logging.error(
                        f"Skipping {dataset_name}: rebuilt KG failed post-build quality verification."
                    )
                    continue
            else:
                # Default behavior: Reuse existing KG if available, otherwise build new
                if existing_chunk_count > 0:
                    logging.info(
                        f"Reusing existing KG for {dataset_name} ({existing_chunk_count} chunks). "
                        f"Use --rebuild-kg to force rebuild."
                    )
                    if not self._verify_kg_quality(dataset_name):
                        logging.warning(
                            f"Existing KG for {dataset_name} failed quality checks; rebuilding now."
                        )
                        self._delete_dataset_kg(dataset_name)
                        kg_built = self._build_kg_for_dataset(dataset_name)
                        if not kg_built or not self._verify_kg_quality(dataset_name):
                            logging.error(
                                f"Skipping {dataset_name}: failed to rebuild a valid dataset-scoped KG."
                            )
                            continue
                else:
                    logging.info(f"No existing KG for {dataset_name}, building now...")
                    kg_built = self._build_kg_for_dataset(dataset_name)
                    if not kg_built:
                        logging.error(f"Skipping {dataset_name} due to KG build failure")
                        continue
                    if not self._verify_kg_quality(dataset_name):
                        logging.error(
                            f"Skipping {dataset_name}: newly built KG failed post-build quality verification."
                        )
                        continue

            selection_path = self._selection_path(dataset_name)
            if selection_path.exists():
                manifest["selection_files"][dataset_name] = str(selection_path)
                manifest["subset_ids"][dataset_name] = self._subset_metadata(dataset_name).get("subset_id")
            
            # Step 2: Run evaluation for each configuration
            dataset_config_results = []
            for config in self.eval_configs:
                dataset_results = self._run_evaluation_on_dataset(dataset_name, config=config)
                if dataset_results.get("error"):
                    logging.error(
                        "Skipping %s [%s] due to evaluation error: %s",
                        dataset_name,
                        config.get("name", "default"),
                        dataset_results.get("error"),
                    )
                    continue
                dataset_config_results.append(dataset_results)

                config_name = dataset_results.get("config", {}).get("name", "default")
                log_payload = {
                    # Accuracy
                    f"accuracy/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_accuracy", 0),
                    f"accuracy/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_accuracy", 0),
                    f"accuracy_raw/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_accuracy_raw", 0),
                    f"accuracy_raw/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_accuracy_raw", 0),
                    f"answer_em/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_answer_em", 0),
                    f"answer_em/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_answer_em", 0),
                    f"answer_f1/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_answer_f1", 0),
                    f"answer_f1/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_answer_f1", 0),
                }
                if self.compute_metrics:
                    log_payload.update({
                        # Entropy Family (2)
                        f"semantic_entropy/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_avg_semantic_entropy", 0),
                        f"semantic_entropy/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_avg_semantic_entropy", 0),
                        f"discrete_semantic_entropy/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_avg_discrete_semantic_entropy", 0),
                        f"discrete_semantic_entropy/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_avg_discrete_semantic_entropy", 0),
                        # Calibration Family (2)
                        f"p_true/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_avg_p_true", 0.5),
                        f"p_true/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_avg_p_true", 0.5),
                        # Similarity Family (1)
                        f"selfcheckgpt/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_avg_selfcheckgpt", 0.0),
                        f"selfcheckgpt/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_avg_selfcheckgpt", 0.0),
                        # Perturbation Family (1)
                        f"sre_uq/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_avg_sre_uq", 0),
                        f"sre_uq/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_avg_sre_uq", 0),
                        # Geometric Family (2)
                        f"vn_entropy/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_avg_vn_entropy", 0.0),
                        f"vn_entropy/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_avg_vn_entropy", 0.0),
                        f"sd_uq/{dataset_name}/{config_name}/vanilla": dataset_results.get("vanilla_avg_sd_uq", 0.0),
                        f"sd_uq/{dataset_name}/{config_name}/kg_rag": dataset_results.get("kg_avg_sd_uq", 0.0),
                        # AUROC / AUREC per metric
                        **{
                            f"auroc/{m}/{dataset_name}/{config_name}/vanilla": dataset_results.get(f"vanilla_avg_{m}_auroc", float("nan"))
                            for m in ["semantic_entropy", "discrete_semantic_entropy",
                                      "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq"]
                        },
                        **{
                            f"auroc/{m}/{dataset_name}/{config_name}/kg_rag": dataset_results.get(f"kg_avg_{m}_auroc", float("nan"))
                            for m in ["semantic_entropy", "discrete_semantic_entropy",
                                      "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq"]
                        },
                        **{
                            f"aurec/{m}/{dataset_name}/{config_name}/vanilla": dataset_results.get(f"vanilla_avg_{m}_aurec", float("nan"))
                            for m in ["semantic_entropy", "discrete_semantic_entropy",
                                      "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq"]
                        },
                        **{
                            f"aurec/{m}/{dataset_name}/{config_name}/kg_rag": dataset_results.get(f"kg_avg_{m}_aurec", float("nan"))
                            for m in ["semantic_entropy", "discrete_semantic_entropy",
                                      "sre_uq", "p_true", "selfcheckgpt", "vn_entropy", "sd_uq"]
                        },
                    })
                self.wandb_run.log(log_payload)

            # Log config-level table + grouped bar chart for this dataset
            self._log_config_summary_to_wandb(dataset_name, dataset_config_results)

            # Strip intermediate fields from detail rows before saving
            kept = set(self._QUESTION_OUTPUT_FIELDS)
            clean_config_results = []
            for cfg in dataset_config_results:
                clean_cfg = {k: v for k, v in cfg.items() if k != "details"}
                clean_cfg["details"] = [
                    {k: v for k, v in row.items() if k in kept}
                    for row in cfg.get("details", [])
                ]
                clean_config_results.append(clean_cfg)

            dataset_block = {
                "dataset": dataset_name,
                "config_results": clean_config_results,
            }
            all_results.append(dataset_block)

            # Save intermediate results
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"mirage_{dataset_name}_results.json")
            with open(output_path, 'w') as f:
                json.dump(dataset_block, f, indent=2)
        
        # Build summary grouped by evaluation track (biomedical_grounding vs multihop_reasoning).
        # These are separate empirical claims and must not be aggregated across tracks.
        summary_results = []
        for dataset_block in all_results:
            dataset_name = dataset_block.get("dataset", "unknown")
            track = self.DATASET_TRACKS.get(dataset_name, self.DEFAULT_TRACK)
            dataset_summary = {
                "dataset": dataset_name,
                "track": track,
                "config_results": [],
            }
            for cfg_res in dataset_block.get("config_results", []):
                cfg_entry = {
                    "dataset": dataset_name,
                    "track": track,
                    "config": cfg_res.get("config", {}),
                    "subset_seed": cfg_res.get("subset_seed"),
                    "selection_file": cfg_res.get("selection_file", ""),
                    "selection_key": cfg_res.get("selection_key", ""),
                    "subset_hash": cfg_res.get("subset_hash", ""),
                    "subset_tag": cfg_res.get("subset_tag", ""),
                    "subset_id": cfg_res.get("subset_id", ""),
                    "num_questions_logged": int(
                        cfg_res.get("num_questions_logged", len(cfg_res.get("details", [])))
                    ),
                    "reported_accuracy_variant": cfg_res.get("reported_accuracy_variant", "clean_excluding_generation_failures"),
                    "vanilla_accuracy": cfg_res.get("vanilla_accuracy", 0.0),
                    "kg_accuracy": cfg_res.get("kg_accuracy", 0.0),
                    "vanilla_accuracy_raw": cfg_res.get("vanilla_accuracy_raw", 0.0),
                    "kg_accuracy_raw": cfg_res.get("kg_accuracy_raw", 0.0),
                    "vanilla_answered_questions": cfg_res.get("vanilla_answered_questions", 0),
                    "kg_answered_questions": cfg_res.get("kg_answered_questions", 0),
                    "question_details_file": cfg_res.get("question_details_file", ""),
                    "question_table_key": cfg_res.get("question_table_key", ""),
                    "vanilla_answer_em": cfg_res.get("vanilla_answer_em", 0.0),
                    "kg_answer_em": cfg_res.get("kg_answer_em", 0.0),
                    "vanilla_answer_f1": cfg_res.get("vanilla_answer_f1", 0.0),
                    "kg_answer_f1": cfg_res.get("kg_answer_f1", 0.0),
                    "structural_metrics": {},
                    "metrics_by_approach": {},
                    "auroc_aurec": {},
                    "complementarity": cfg_res.get("complementarity", {}),
                    "accuracy_by_task_type": cfg_res.get("accuracy_by_task_type", {}),
                    "accuracy_by_hop_count": cfg_res.get("accuracy_by_hop_count", {}),
                    "retrieval_determinism_confound": None,
                    "kg_routing_distribution": cfg_res.get("kg_routing_distribution", {}),
                    "kg_pure_entity_first_rate": cfg_res.get("kg_pure_entity_first_rate"),
                    "kg_dense_fallback_rate": cfg_res.get("kg_dense_fallback_rate"),
                    "kg_unknown_route_rate": cfg_res.get("kg_unknown_route_rate"),
                    "kg_avg_retrieval_overlap": cfg_res.get("kg_avg_retrieval_overlap"),
                    "vanilla_avg_retrieval_overlap": cfg_res.get("vanilla_avg_retrieval_overlap"),
                }
                if self.compute_metrics:
                    cfg_entry["structural_metrics"] = {
                        "vanilla_avg_graph_path_support": cfg_res.get("vanilla_avg_graph_path_support"),
                        "kg_avg_graph_path_support": cfg_res.get("kg_avg_graph_path_support"),
                        "vanilla_avg_graph_path_disagreement": cfg_res.get("vanilla_avg_graph_path_disagreement"),
                        "kg_avg_graph_path_disagreement": cfg_res.get("kg_avg_graph_path_disagreement"),
                        "vanilla_avg_competing_answer_alternatives": cfg_res.get("vanilla_avg_competing_answer_alternatives"),
                        "kg_avg_competing_answer_alternatives": cfg_res.get("kg_avg_competing_answer_alternatives"),
                        "vanilla_avg_evidence_vn_entropy": cfg_res.get("vanilla_avg_evidence_vn_entropy"),
                        "kg_avg_evidence_vn_entropy": cfg_res.get("kg_avg_evidence_vn_entropy"),
                        "vanilla_avg_subgraph_informativeness": cfg_res.get("vanilla_avg_subgraph_informativeness"),
                        "kg_avg_subgraph_informativeness": cfg_res.get("kg_avg_subgraph_informativeness"),
                        "vanilla_avg_subgraph_perturbation_stability": cfg_res.get("vanilla_avg_subgraph_perturbation_stability"),
                        "kg_avg_subgraph_perturbation_stability": cfg_res.get("kg_avg_subgraph_perturbation_stability"),
                        "vanilla_avg_support_entailment_uncertainty": cfg_res.get("vanilla_avg_support_entailment_uncertainty"),
                        "kg_avg_support_entailment_uncertainty": cfg_res.get("kg_avg_support_entailment_uncertainty"),
                        "vanilla_avg_evidence_conflict_uncertainty": cfg_res.get("vanilla_avg_evidence_conflict_uncertainty"),
                        "kg_avg_evidence_conflict_uncertainty": cfg_res.get("kg_avg_evidence_conflict_uncertainty"),
                    }
                    cfg_entry["metrics_by_approach"] = self._build_grouped_uncertainty_metrics(cfg_res)
                    cfg_entry["auroc_aurec"] = cfg_res.get("auroc_aurec", {})
                    cfg_entry["ece"] = cfg_res.get("ece", {})
                    cfg_entry["precision_at_k"] = cfg_res.get("precision_at_k", {})
                    cfg_entry["retrieval_determinism_confound"] = (
                        float(cfg_res.get("kg_avg_retrieval_overlap", 0) or 0)
                        - float(cfg_res.get("vanilla_avg_retrieval_overlap", 0) or 0)
                    ) >= 0.3
                dataset_summary["config_results"].append(cfg_entry)
            summary_results.append(dataset_summary)

        # Delegate to the module-level helper so the logic is tested directly
        # rather than mirrored in a parallel test copy.  all_results has no
        # track field; resolve it from DATASET_TRACKS here.
        track_aggregates = accumulate_track_accuracy(
            [{"dataset": ds["dataset"],
              "track": self.DATASET_TRACKS.get(ds["dataset"], self.DEFAULT_TRACK),
              "config_results": ds.get("config_results", [])}
             for ds in all_results]
        )

        summary = {
            "datasets": datasets,
            "num_samples_per_dataset": self.num_samples,
            "subset_seed": self.subset_seed,
            "evaluation_mode": self.evaluation_mode,
            "retrieval_study": self.retrieval_study or None,
            "kg_builder_profile": self.kg_builder_profile,
            "judge_model": self.judge_model,
            "generation_model": self.llm_model,
            # True when either the provider or the model differs — same model string on
            # a different provider is still a separate system and counts as independent.
            "judge_independent": (
                self.judge_model != self.llm_model
                or self.judge_provider_name != self.llm_provider_name
            ),
            "metric_names": self.UNCERTAINTY_METRIC_NAMES if self.compute_metrics else [],
            "results": summary_results,
            "track_aggregates": track_aggregates,
            "retrieval_selection": select_best_retrieval_configs(all_results),
        }
        
        # Save final summary
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "mirage_evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Log final summary to W&B BEFORE finishing run
        self._log_final_summary_to_wandb(all_results=all_results, summary_path=summary_path)

        # ── Finalise manifest & runs index ───────────────────────────────
        if hasattr(self, "_run_dir"):
            accuracy_summary = {}
            for db in all_results:
                dn = db.get("dataset", "unknown")
                for cfg in db.get("config_results", []):
                    key = f"{dn}/{cfg.get('config', {}).get('name', 'default')}"
                    accuracy_summary[key] = {
                        "vanilla": round(cfg.get("vanilla_accuracy", 0), 4),
                        "kg":      round(cfg.get("kg_accuracy", 0), 4),
                        "vanilla_raw": round(cfg.get("vanilla_accuracy_raw", 0), 4),
                        "kg_raw": round(cfg.get("kg_accuracy_raw", 0), 4),
                        "vanilla_answer_em": round(cfg.get("vanilla_answer_em", 0), 4),
                        "kg_answer_em": round(cfg.get("kg_answer_em", 0), 4),
                        "vanilla_answer_f1": round(cfg.get("vanilla_answer_f1", 0), 4),
                        "kg_answer_f1": round(cfg.get("kg_answer_f1", 0), 4),
                        "n":       cfg.get("num_questions_logged", 0),
                    }
            manifest["accuracy"]     = accuracy_summary
            manifest["completed_at"] = datetime.now().isoformat()
            manifest["wandb_run_id"] = self.wandb_run.id if self.wandb_run else None
            (self._run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            update_runs_index(self._run_dir, manifest)
            logging.info("Run manifest written: %s/manifest.json", self._run_dir)
            logging.info("Runs index updated:   %s", self._run_dir.parent / "index.json")

        if self.wandb_run:
            self.wandb_run.finish()

        logging.info("\n" + "="*50)
        logging.info("EVALUATION COMPLETE  (run: %s)", getattr(self, "_run_id", "?"))
        logging.info("="*50)
        for dataset_block in all_results:
            dataset_name = dataset_block.get("dataset", "unknown")
            for cfg_res in dataset_block.get("config_results", []):
                cfg_name = cfg_res.get("config", {}).get("name", "default")
                logging.info(
                    f"{dataset_name} [{cfg_name}]: "
                    f"Vanilla={cfg_res.get('vanilla_accuracy', 0):.2%} "
                    f"(raw {cfg_res.get('vanilla_accuracy_raw', 0):.2%}), "
                    f"KG={cfg_res.get('kg_accuracy', 0):.2%} "
                    f"(raw {cfg_res.get('kg_accuracy_raw', 0):.2%}), "
                    f"EM/F1 Vanilla={cfg_res.get('vanilla_answer_em', 0):.2%}/{cfg_res.get('vanilla_answer_f1', 0):.2%} "
                    f"KG={cfg_res.get('kg_answer_em', 0):.2%}/{cfg_res.get('kg_answer_f1', 0):.2%}"
                )
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run MIRAGE dataset evaluation pipeline (Experiment Mode)")
    parser.add_argument("--num-samples", type=int, default=None, 
                        help="Number of samples per dataset (default: all)")
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help=(
            "Seed used to deterministically sample the dataset question subset. "
            "When a dataset KG is rebuilt, the sampled IDs are persisted and later "
            "non-rebuild runs will reuse those same IDs."
        ),
    )
    parser.add_argument("--entropy-samples", type=int, default=5,
                        help="Number of generations per question for semantic entropy (default: 5, max: 20)")
    parser.add_argument("--similarity-thresholds", nargs="+", type=float, default=[0.1],
                        help="One or more similarity thresholds to evaluate as configurations")
    parser.add_argument("--max-chunks-values", nargs="+", type=int, default=[10],
                        help="One or more max_chunks values to evaluate as configurations")
    parser.add_argument("--llm-provider", type=str, default="openai",
                        help="LLM provider to use for KG extraction and response generation")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini",
                        help="LLM model name for the selected provider")
    parser.add_argument("--datasets", nargs="+", 
                        default=["pubmedqa", "bioasq"],
                        help=(
                            "Datasets to evaluate "
                            "(e.g., pubmedqa realmedqa bioasq medhop multihoprag hotpotqa hotpotqa_fullwiki 2wikimultihopqa musique)"
                        ))
    parser.add_argument("--rebuild-kg", action="store_true", default=False,
                        help="Force rebuild the KG even if it already exists for the dataset")
    parser.add_argument("--max-kg-contexts", type=int, default=None,
                        help="Cap the number of context passages used to build the KG (useful for quick tests)")
    parser.add_argument(
        "--dataset-kg-scope",
        type=str,
        choices=sorted(MIRAGEEvaluationPipeline.DATASET_KG_SCOPES),
        default=MIRAGEEvaluationPipeline.DATASET_KG_SCOPE_EVALUATION_SUBSET,
        help=(
            "evaluation_subset = build the dataset KG from the same subset being evaluated; "
            "full_dataset = build from the full normalized dataset before evaluating the requested subset"
        ),
    )
    parser.add_argument(
        "--allow-gold-evidence-contexts",
        action="store_true",
        default=False,
        help=(
            "Allow datasets whose per-question contexts are oracle gold evidence to be indexed directly. "
            "Use only for controlled-evidence QA analyses, not retrieval benchmarking."
        ),
    )
    parser.add_argument("--no-llm-judge", action="store_true", default=False,
                        help="Disable LLM-as-judge evaluation and use heuristic string matching instead")
    parser.add_argument("--judge-provider", type=str, default=None,
                        help="LLM provider for the answer judge (default: same as --llm-provider). "
                             "Set to a different provider/model to avoid circular self-evaluation.")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="LLM model for the answer judge (default: same as --llm-model). "
                             "E.g., --llm-model gpt-4o-mini --judge-model gpt-4o for independent judging.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="LLM sampling temperature (default: 1.0; use 0.0 for greedy, 0.5 for SE-optimal)")
    parser.add_argument(
        "--retrieval-temperature-values",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "One or more final-stage retrieval sampling temperatures to evaluate. "
            "0.0 preserves deterministic top-k; larger values sample from a larger shortlist."
        ),
    )
    parser.add_argument(
        "--retrieval-shortlist-factor",
        type=int,
        default=4,
        help=(
            "Shortlist multiplier for stochastic final-stage retrieval selection. "
            "For example, k=10 with factor=4 samples the final 10 from the top 40 candidates."
        ),
    )
    parser.add_argument(
        "--retrieval-study",
        type=str,
        choices=sorted(MIRAGEEvaluationPipeline.RETRIEVAL_STUDY_PROFILES),
        default=None,
        help=(
            "Run a built-in small retrieval ablation instead of the plain threshold/k sweep. "
            "This expands each threshold/k setting into a small family of retrieval variants "
            "and writes an automatic best-config recommendation into the final summary."
        ),
    )
    parser.add_argument(
        "--kg-builder-profile",
        type=MIRAGEEvaluationPipeline.normalize_kg_builder_profile,
        choices=sorted(MIRAGEEvaluationPipeline.KG_BUILDER_PROFILES),
        default="auto",
        help=(
            "full = use the strongest current KG construction stages, including "
            "multi-sample extraction, richer schema guidance, low-confidence "
            "reverification, soft entity linking, fragmentation repair, and "
            "biomedical UMLS linking where applicable; "
            "lightweight = disable expensive anchor/self-reflection/cross-passage extras "
            "for quick retrieval sweeps; auto = lightweight only for accuracy-only "
            "question-scoped multihop runs."
        ),
    )
    parser.add_argument("--multi-temperature", action="store_true", default=False,
                        help="Run each query at T=0, 0.5, 1.0 and store metrics with _t00/_t05/_t10 suffixes")
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        choices=sorted(MIRAGEEvaluationPipeline.EVALUATION_MODES),
        default=MIRAGEEvaluationPipeline.EVALUATION_MODE_FULL_METRICS,
        help=(
            "full_metrics = compute the canonical 13 uncertainty metrics; "
            "accuracy_only = only generate answers and score correctness for cheap test runs"
        ),
    )
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to write result JSON files (default: results/)")

    args = parser.parse_args()
    
    if args.retrieval_study:
        eval_configs = MIRAGEEvaluationPipeline.build_retrieval_study_eval_configs(
            profile=args.retrieval_study,
            similarity_thresholds=args.similarity_thresholds,
            max_chunks_values=args.max_chunks_values,
            retrieval_temperature_values=args.retrieval_temperature_values,
            retrieval_shortlist_factor=args.retrieval_shortlist_factor,
        )
    else:
        eval_configs = []
        for threshold in args.similarity_thresholds:
            for max_chunks in args.max_chunks_values:
                for retrieval_temperature in args.retrieval_temperature_values:
                    rt_suffix = f"_rt{str(float(retrieval_temperature)).replace('.', 'p')}"
                    eval_configs.append({
                        "name": f"thr{threshold:g}_k{max_chunks}{rt_suffix}",
                        "similarity_threshold": float(threshold),
                        "max_chunks": int(max_chunks),
                        "retrieval_temperature": float(retrieval_temperature),
                        "retrieval_shortlist_factor": int(args.retrieval_shortlist_factor),
                    })

    pipeline = MIRAGEEvaluationPipeline(
        num_samples=args.num_samples,
        subset_seed=args.subset_seed,
        entropy_samples=args.entropy_samples,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
        eval_configs=eval_configs,
        rebuild_kg=args.rebuild_kg,
        max_kg_contexts=args.max_kg_contexts,
        use_llm_judge=not args.no_llm_judge,
        multi_temperature=args.multi_temperature,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        evaluation_mode=args.evaluation_mode,
        dataset_kg_scope=args.dataset_kg_scope,
        allow_gold_evidence_contexts=args.allow_gold_evidence_contexts,
        retrieval_study=args.retrieval_study,
        kg_builder_profile=args.kg_builder_profile,
    )
    pipeline.run_pipeline(datasets=args.datasets, output_dir=args.output_dir)
    
    print(f"\nResults saved to results/mirage_evaluation_summary.json")


if __name__ == "__main__":
    main()
