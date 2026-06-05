#!/usr/bin/env python3
"""
Visualize RAG experiment results with grouped bar charts.
Supports both single-dataset and multi-dataset result formats.
Results are aggregated by configuration and logged to Weights & Biases.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from math import ceil
import numpy as np

# Try to import matplotlib and wandb
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Will skip local plotting.")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Will skip wandb logging.")


UNCERTAINTY_METRIC_SPECS = [
    ("semantic_entropy", "Semantic Entropy", "SE", True),
    ("discrete_semantic_entropy", "Discrete Sem. Entropy", "DSE", True),
    ("sre_uq", "SRE-UQ", "SRE-UQ", True),
    ("p_true", "P(True)", "P(True)", False),
    ("selfcheckgpt", "SelfCheckGPT", "SelfCheckGPT", True),
    ("vn_entropy", "VN-Entropy", "VN-Ent", True),
    ("sd_uq", "SD-UQ", "SD-UQ", True),
    ("graph_path_support", "Graph Path Support", "GPS", True),
    ("graph_path_disagreement", "Graph Path Disagreement", "GPD", True),
    ("competing_answer_alternatives", "Competing Answer Alternatives", "CAA", True),
    ("evidence_vn_entropy", "Evidence VN-Entropy", "EVN-Ent", True),
    ("subgraph_informativeness", "Subgraph Informativeness", "SGI", True),
    ("subgraph_perturbation_stability", "Subgraph Perturbation Stability", "SPS-UQ", True),
    ("support_entailment_uncertainty", "Support Entailment Uncertainty", "SEU", True),
    ("evidence_conflict_uncertainty", "Evidence Conflict Uncertainty", "ECU", True),
]
DEFAULT_UNCERTAINTY_METRICS = [name for name, _, _, _ in UNCERTAINTY_METRIC_SPECS]
METRIC_DISPLAY_NAMES = {name: display for name, display, _, _ in UNCERTAINTY_METRIC_SPECS}
METRIC_SHORT_LABELS = {name: short for name, _, short, _ in UNCERTAINTY_METRIC_SPECS}
METRIC_LOWER_IS_BETTER = {name: lower_is_better for name, _, _, lower_is_better in UNCERTAINTY_METRIC_SPECS}


def _has_new_summary_schema(results: dict) -> bool:
    """Detect strict summary schema written by experiments/experiment.py."""
    try:
        dataset_blocks = results.get('results', [])
        if not dataset_blocks:
            return False
        first_block = dataset_blocks[0]
        first_cfg = first_block.get('config_results', [])[0]
        return isinstance(first_cfg.get('metrics_by_approach', None), dict)
    except Exception:
        return False


def _get_metric_names(results: dict) -> list:
    metric_names = results.get('metric_names', [])
    if metric_names and isinstance(metric_names, list):
        return metric_names
    return DEFAULT_UNCERTAINTY_METRICS


def load_results(results_path: str) -> dict:
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def aggregate_by_config(results: dict) -> dict:
    """Aggregate metrics by configuration (similarity_threshold, max_chunks)."""
    config_metrics = defaultdict(lambda: {
        'vanilla_rag': defaultdict(list),
        'kg_rag': defaultdict(list)
    })
    metric_names = _get_metric_names(results)

    if _has_new_summary_schema(results):
        # New strict summary schema
        for dataset_block in results.get('results', []):
            for cfg in dataset_block.get('config_results', []):
                config = cfg.get('config', {})
                config_name = config.get('name') or f"thresh={config.get('similarity_threshold', 'unknown')}_chunks={config.get('max_chunks', 'unknown')}"
                metric_groups = cfg.get('metrics_by_approach', {})

                for system_key in ('vanilla_rag', 'kg_rag'):
                    system_metrics = metric_groups.get(system_key, {})
                    for metric_name in metric_names:
                        config_metrics[config_name][system_key][metric_name].append(
                            float(system_metrics.get(metric_name, 0.0))
                        )

        aggregated = {}
        for config_key, systems in config_metrics.items():
            aggregated[config_key] = {}
            for system, metrics in systems.items():
                aggregated[config_key][system] = {
                    metric: np.mean(values) if values else 0
                    for metric, values in metrics.items()
                }
        return aggregated
    
    # Check if this is a multi-dataset result
    if 'per_dataset_results' in results:
        # Multi-dataset format
        for dataset_data in results.get('per_dataset_results', {}).values():
            for result in dataset_data.get('results', []):
                config = result.get('config', {})
                threshold = config.get('similarity_threshold', 'unknown')
                max_chunks = config.get('max_chunks', 'unknown')
                
                config_key = f"thresh={threshold}_chunks={max_chunks}"
                
                # Vanilla RAG metrics
                if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['vanilla_rag_metrics']
                    for metric_name, value in metrics.items():
                        config_metrics[config_key]['vanilla_rag'][metric_name].append(value)
                
                # KG-RAG metrics
                if 'kg_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['kg_rag_metrics']
                    for metric_name, value in metrics.items():
                        config_metrics[config_key]['kg_rag'][metric_name].append(value)
    else:
        # Single dataset format
        for result in results.get('results', []):
            config = result.get('config', {})
            threshold = config.get('similarity_threshold', 'unknown')
            max_chunks = config.get('max_chunks', 'unknown')
            
            config_key = f"thresh={threshold}_chunks={max_chunks}"
            
            # Vanilla RAG metrics
            if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['vanilla_rag_metrics']
                for metric_name, value in metrics.items():
                    config_metrics[config_key]['vanilla_rag'][metric_name].append(value)
            
            # KG-RAG metrics
            if 'kg_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['kg_rag_metrics']
                for metric_name, value in metrics.items():
                    config_metrics[config_key]['kg_rag'][metric_name].append(value)
    
    # Compute averages
    aggregated = {}
    for config_key, systems in config_metrics.items():
        aggregated[config_key] = {}
        for system, metrics in systems.items():
            aggregated[config_key][system] = {
                metric: np.mean(values) if values else 0 
                for metric, values in metrics.items()
            }
    
    return aggregated


def aggregate_overall(results: dict) -> dict:
    """Aggregate overall metrics across all configurations."""
    overall = {
        'vanilla_rag': defaultdict(list),
        'kg_rag': defaultdict(list)
    }
    metric_names = _get_metric_names(results)

    if _has_new_summary_schema(results):
        for dataset_block in results.get('results', []):
            for cfg in dataset_block.get('config_results', []):
                metric_groups = cfg.get('metrics_by_approach', {})
                for system_key in ('vanilla_rag', 'kg_rag'):
                    system_metrics = metric_groups.get(system_key, {})
                    for metric_name in metric_names:
                        overall[system_key][metric_name].append(
                            float(system_metrics.get(metric_name, 0.0))
                        )

        return {
            'vanilla_rag': {
                metric: np.mean(values) if values else 0
                for metric, values in overall['vanilla_rag'].items()
            },
            'kg_rag': {
                metric: np.mean(values) if values else 0
                for metric, values in overall['kg_rag'].items()
            }
        }
    
    # Check if this is a multi-dataset result
    if 'per_dataset_results' in results:
        # Multi-dataset format - use combined_analysis if available
        combined = results.get('combined_analysis', {})
        if combined:
            overall_vanilla = combined.get('overall_vanilla_rag', {})
            overall_kg = combined.get('overall_kg_rag', {})
            return {
                'vanilla_rag': overall_vanilla,
                'kg_rag': overall_kg
            }
        
        # Otherwise aggregate manually
        for dataset_data in results.get('per_dataset_results', {}).values():
            for result in dataset_data.get('results', []):
                # Vanilla RAG metrics
                if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['vanilla_rag_metrics']
                    for metric_name, value in metrics.items():
                        overall['vanilla_rag'][metric_name].append(value)
                
                # KG-RAG metrics
                if 'kg_rag_metrics' in result.get('evaluation', {}):
                    metrics = result['evaluation']['kg_rag_metrics']
                    for metric_name, value in metrics.items():
                        overall['kg_rag'][metric_name].append(value)
    else:
        # Single dataset format
        for result in results.get('results', []):
            # Vanilla RAG metrics
            if 'vanilla_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['vanilla_rag_metrics']
                for metric_name, value in metrics.items():
                    overall['vanilla_rag'][metric_name].append(value)
            
            # KG-RAG metrics
            if 'kg_rag_metrics' in result.get('evaluation', {}):
                metrics = result['evaluation']['kg_rag_metrics']
                for metric_name, value in metrics.items():
                    overall['kg_rag'][metric_name].append(value)
    
    # Compute averages
    return {
        'vanilla_rag': {
            metric: np.mean(values) if values else 0 
            for metric, values in overall['vanilla_rag'].items()
        },
        'kg_rag': {
            metric: np.mean(values) if values else 0 
            for metric, values in overall['kg_rag'].items()
        }
    }


def print_per_dataset_summary(results: dict):
    """Print summary per dataset for multi-dataset results."""
    if 'per_dataset_results' not in results:
        return
    
    print("\n=== Per Dataset Summary ===")
    per_dataset = results.get('combined_analysis', {}).get('per_dataset', {})
    
    for dataset_name, analysis in per_dataset.items():
        print(f"\n{dataset_name}:")
        vanilla_stats = analysis.get('vanilla_rag_stats', {})
        kg_stats = analysis.get('kg_rag_stats', {})
        
        print(f"  Vanilla RAG:")
        for metric, value in vanilla_stats.items():
            print(f"    {metric}: {value:.2f}")
        
        print(f"  KG-RAG:")
        for metric, value in kg_stats.items():
            print(f"    {metric}: {value:.2f}")
        
        comparisons = analysis.get('comparisons', {})
        print(f"  Comparisons: vanilla_better={comparisons.get('vanilla_better', 0)}, "
              f"kg_better={comparisons.get('kg_better', 0)}, "
              f"tie={comparisons.get('tie', 0)}")


def create_comparison_chart(aggregated: dict, metric_names: list, output_path: str = None):
    """Create grouped bar chart comparing Vanilla RAG vs KG-RAG by config."""
    if not HAS_MATPLOTLIB:
        print("Skipping chart creation (matplotlib not available)")
        return
    
    configs = sorted(aggregated.keys())
    metrics = metric_names
    
    n_configs = len(configs)
    n_metrics = len(metrics)

    if n_metrics == 0:
        print("No metrics found for charting.")
        return

    n_cols = min(4, n_metrics)
    n_rows = ceil(n_metrics / n_cols)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    
    x = np.arange(n_configs)
    width = 0.35
    
    colors = {'vanilla_rag': '#1f77b4', 'kg_rag': '#ff7f0e'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        vanilla_values = [aggregated[c]['vanilla_rag'].get(metric, 0) for c in configs]
        kg_values = [aggregated[c]['kg_rag'].get(metric, 0) for c in configs]
        
        ax.bar(x - width/2, vanilla_values, width, label='Vanilla RAG', color=colors['vanilla_rag'])
        ax.bar(x + width/2, kg_values, width, label='KG-RAG', color=colors['kg_rag'])
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart to {output_path}")
    
    return fig


def create_overall_comparison_chart(overall: dict, metric_names: list, output_path: str = None):
    """Create overall comparison bar chart."""
    if not HAS_MATPLOTLIB:
        print("Skipping chart creation (matplotlib not available)")
        return
    
    metrics = metric_names
    
    vanilla_values = [overall['vanilla_rag'].get(m, 0) for m in metrics]
    kg_values = [overall['kg_rag'].get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, vanilla_values, width, label='Vanilla RAG', color='#1f77b4')
    bars2 = ax.bar(x + width/2, kg_values, width, label='KG-RAG', color='#ff7f0e')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Overall Uncertainty Metrics: Vanilla RAG vs KG-RAG')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart to {output_path}")
    
    return fig


def plot_metric_bar_charts(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Bar charts: Vanilla RAG vs KG-RAG on accuracy + all live uncertainty metrics.

    Layout: dynamic grid. Each panel shows one metric with one grouped
    bar pair per (dataset, config). Saved locally as PNG and logged to W&B.
    Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping bar charts (matplotlib not available)")
        return []

    panels = [("Accuracy", "vanilla_accuracy", "kg_accuracy", False)]
    for metric_name, display_name, _, lower_is_better in UNCERTAINTY_METRIC_SPECS:
        panels.append((
            display_name,
            f"vanilla_avg_{metric_name}",
            f"kg_avg_{metric_name}",
            lower_is_better,
        ))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Collect (label, vanilla_val, kg_val) per panel
    bar_data = {title: {"labels": [], "vanilla": [], "kg": []} for title, *_ in panels}

    for dataset_block in all_results:
        dataset_name = dataset_block.get("dataset", "unknown")
        for cfg_res in dataset_block.get("config_results", []):
            cfg_name = cfg_res.get("config", {}).get("name", "default")
            label = f"{dataset_name} / {cfg_name}"
            for title, v_key, k_key, _ in panels:
                vanilla_val = cfg_res.get(v_key)
                kg_val = cfg_res.get(k_key)
                bar_data[title]["labels"].append(label)
                bar_data[title]["vanilla"].append(
                    float(vanilla_val) if vanilla_val is not None else np.nan
                )
                bar_data[title]["kg"].append(
                    float(kg_val) if kg_val is not None else np.nan
                )

    n_cols = 4
    n_rows = ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(10, n_rows * 3.2)))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle("Vanilla RAG vs KG-RAG — All Uncertainty Metrics",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.93, hspace=0.55, wspace=0.35)

    width = 0.35
    colors = {"vanilla": "#4C72B0", "kg": "#DD8452"}

    for idx, (title, v_key, k_key, lower_is_better) in enumerate(panels):
        ax = axes[idx]
        data = bar_data[title]
        labels = data["labels"]
        vanilla_vals = data["vanilla"]
        kg_vals = data["kg"]
        n = len(labels)
        x = np.arange(n)

        b_v = ax.bar(x - width / 2, vanilla_vals, width,
                     label="Vanilla RAG", color=colors["vanilla"], alpha=0.85)
        b_k = ax.bar(x + width / 2, kg_vals, width,
                     label="KG-RAG", color=colors["kg"], alpha=0.85)

        # Annotate bars with values
        for bar in list(b_v) + list(b_k):
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

        direction = "↓ better" if lower_is_better else "↑ better"
        ax.set_title(f"{title}\n({direction})", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    out_path = str(Path(output_dir) / "metric_bar_charts.png")
    fig.savefig(out_path, dpi=100)
    print(f"Saved metric bar charts → {out_path}")

    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({"charts/metric_bar_charts": _wandb.Image(fig)})
        except Exception as e:
            print(f"W&B bar chart logging failed: {e}")

    plt.close(fig)
    return [out_path]


def plot_auroc_aurec_heatmaps(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Plot AUROC and AUREC as side-by-side 2D heatmaps (metrics × systems).

    Layout
    ------
    Rows    : all live uncertainty metrics
    Columns : one column per (dataset, config, system) combination,
              grouped as vanilla | kg pairs
    Left    : AUROC heatmap  (higher = better, centre = 0.5 random baseline)
    Right   : AUREC heatmap  (lower = better)

    Saves PNG(s) locally and, when wandb_run is provided, logs as W&B images.
    Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping heatmap (matplotlib not available)")
        return []

    metric_order = DEFAULT_UNCERTAINTY_METRICS

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # ── Collect columns ───────────────────────────────────────────────────
    # Each column = one (dataset, config, system) triplet.
    # We pair vanilla/kg side by side within each dataset×config group.
    col_labels = []   # x-axis tick labels
    col_groups = []   # group label for separator lines (dataset/config)
    auroc_cols = []   # list of {metric: value} per column
    aurec_cols = []

    for dataset_block in all_results:
        dataset_name = dataset_block.get("dataset", "unknown")
        for cfg_res in dataset_block.get("config_results", []):
            cfg_name = cfg_res.get("config", {}).get("name", "default")
            auroc_aurec = cfg_res.get("auroc_aurec", {})
            group = f"{dataset_name}\n{cfg_name}"
            for system_key, sys_label in (("vanilla_rag", "Vanilla"), ("kg_rag", "KG-RAG")):
                sys_data = auroc_aurec.get(system_key, {})
                col_labels.append(sys_label)
                col_groups.append(group)
                auroc_cols.append({m: _display_eval_metric_value(sys_data, m, "auroc") for m in metric_order})
                aurec_cols.append({m: _display_eval_metric_value(sys_data, m, "aurec") for m in metric_order})

    if not col_labels:
        return []

    n_metrics = len(metric_order)
    n_cols    = len(col_labels)

    def _build_matrix(col_dicts):
        mat = np.full((n_metrics, n_cols), np.nan)
        for c, col in enumerate(col_dicts):
            for r, m in enumerate(metric_order):
                mat[r, c] = col.get(m, np.nan)
        return mat

    auroc_mat = _build_matrix(auroc_cols)
    aurec_mat = _build_matrix(aurec_cols)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(max(8, n_cols * 1.6 + 3), n_metrics * 0.75 + 2.5))
    fig.suptitle("Uncertainty Metric Quality: AUROC & AUREC\n(Vanilla RAG vs KG-RAG)",
                 fontsize=13, fontweight="bold", y=1.01)

    row_labels = [METRIC_SHORT_LABELS[m] for m in metric_order]
    # Build combined x-axis labels showing group above system label
    xtick_labels = []
    for i, (lbl, grp) in enumerate(zip(col_labels, col_groups)):
        # Only show group label on the first column of each group pair
        if i == 0 or col_groups[i] != col_groups[i - 1]:
            xtick_labels.append(f"{grp}\n{lbl}")
        else:
            xtick_labels.append(lbl)

    def _draw_heatmap(ax, mat, title, cmap, vmin, vmax, fmt_fn, cbar_label):
        masked = np.ma.array(mat, mask=np.isnan(mat))
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#cccccc")

        im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax,
                       interpolation="nearest")

        # Annotate cells
        for r in range(n_metrics):
            for c in range(n_cols):
                val = mat[r, c]
                if not np.isnan(val):
                    text_color = "white" if abs(val - (vmin + vmax) / 2) > (vmax - vmin) * 0.3 else "black"
                    ax.text(c, r, fmt_fn(val), ha="center", va="center",
                            fontsize=8, color=text_color, fontweight="bold")
                else:
                    ax.text(c, r, "N/A", ha="center", va="center",
                            fontsize=7, color="#888888")

        # Draw separator lines between dataset×config groups
        prev_group = None
        for c, grp in enumerate(col_groups):
            if prev_group is not None and grp != prev_group:
                ax.axvline(c - 0.5, color="white", linewidth=2)
            prev_group = grp

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(xtick_labels, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    _draw_heatmap(
        axes[0], auroc_mat,
        title="AUROC  (↑ better, 0.5 = random)",
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        fmt_fn=lambda v: f"{v:.2f}",
        cbar_label="AUROC",
    )
    _draw_heatmap(
        axes[1], aurec_mat,
        title="AUREC  (↓ better, rejection-error)",
        cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
        fmt_fn=lambda v: f"{v:.2f}",
        cbar_label="AUREC",
    )

    plt.tight_layout()

    out_path = str(Path(output_dir) / "auroc_aurec_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    saved_paths.append(out_path)
    print(f"Saved AUROC/AUREC heatmap → {out_path}")

    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({"charts/auroc_aurec_heatmap": _wandb.Image(fig)})
        except Exception as e:
            print(f"W&B heatmap logging failed: {e}")

    plt.close(fig)
    return saved_paths


def plot_metric_correlation_matrix(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Spearman correlation heatmap between all live uncertainty metrics.

    Collects per-question uncertainty scores from both Vanilla and KG systems
    across all datasets/configs, then computes and plots the Spearman correlation
    matrix.  Saved locally as PNG and logged to W&B.
    Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    metric_order = DEFAULT_UNCERTAINTY_METRICS

    # Collect per-question scores (combine both systems for more data points)
    scores: dict = {m: [] for m in metric_order}
    for dataset_block in all_results:
        for cfg_res in dataset_block.get("config_results", []):
            for detail in cfg_res.get("details", []):
                for m in metric_order:
                    for prefix in ("vanilla", "kg"):
                        val = detail.get(f"{prefix}_{m}")
                        if val is not None:
                            scores[m].append(float(val))

    # Need at least 2 data points per metric
    n = min(len(v) for v in scores.values())
    if n < 2:
        return []

    # Trim all series to the same length (zip-shortest)
    arr = np.array([scores[m][:n] for m in metric_order])

    # Compute Spearman correlation matrix
    try:
        from scipy.stats import spearmanr
        corr_mat, _ = spearmanr(arr.T)
        if corr_mat.ndim == 0:
            corr_mat = np.array([[1.0]])
    except Exception:
        # Fallback: Pearson via numpy
        corr_mat = np.corrcoef(arr)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_m = len(metric_order)
    labels = [METRIC_SHORT_LABELS[m] for m in metric_order]

    fig, ax = plt.subplots(figsize=(max(9, n_m * 0.6), max(8, n_m * 0.55)))
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Annotate cells
    for r in range(n_m):
        for c in range(n_m):
            val = corr_mat[r, c]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Uncertainty Metric Correlations (Spearman \u03c1)", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman \u03c1", fontsize=9)

    plt.tight_layout()

    out_path = str(Path(output_dir) / "metric_correlation_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved metric correlation matrix -> {out_path}")

    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({"charts/metric_correlation_matrix": _wandb.Image(fig)})
        except Exception as e:
            print(f"W&B correlation matrix logging failed: {e}")

    plt.close(fig)
    return [out_path]


def plot_reliability_diagrams(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Reliability (calibration) diagrams for all live uncertainty metrics.

    For each metric, questions are binned into 5 equal-frequency bins by
    uncertainty score.  Within each bin the mean uncertainty and fraction of
    incorrect answers (error rate) are computed and plotted together with a
    perfect-calibration diagonal.  Layout: 3x3 subplot grid.
    Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    metric_order = DEFAULT_UNCERTAINTY_METRICS

    # Collect per-question (uncertainty, is_incorrect) pairs — combine both systems
    data: dict = {m: {"scores": [], "errors": []} for m in metric_order}
    for dataset_block in all_results:
        for cfg_res in dataset_block.get("config_results", []):
            for detail in cfg_res.get("details", []):
                for prefix, correct_key in (("vanilla", "vanilla_correct"), ("kg", "kg_correct")):
                    is_incorrect = 0 if detail.get(correct_key, False) else 1
                    for m in metric_order:
                        val = detail.get(f"{prefix}_{m}")
                        if val is not None:
                            data[m]["scores"].append(float(val))
                            data[m]["errors"].append(is_incorrect)

    # Need at least 5 data points to bin
    has_data = any(len(v["scores"]) >= 5 for v in data.values())
    if not has_data:
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_bins = 5
    n_cols = 4
    n_rows = ceil(len(metric_order) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, max(10, n_rows * 3.0)))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle("Reliability Diagrams — Uncertainty Calibration", fontsize=13, fontweight="bold")

    for idx, m in enumerate(metric_order):
        ax = axes[idx]
        scores = np.array(data[m]["scores"])
        errors = np.array(data[m]["errors"])

        if len(scores) < 5:
            ax.set_title(METRIC_SHORT_LABELS[m], fontsize=9)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_visible(True)
            continue

        # Equal-frequency binning
        sorted_idx = np.argsort(scores)
        bin_size = len(scores) // n_bins
        bin_means = []
        bin_error_rates = []
        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else len(scores)
            bin_idx = sorted_idx[start:end]
            bin_means.append(float(np.mean(scores[bin_idx])))
            bin_error_rates.append(float(np.mean(errors[bin_idx])))

        ax.plot(bin_means, bin_error_rates, marker="o", linewidth=1.5,
                color="#4C72B0", label="Observed")
        # Perfect calibration diagonal
        all_vals = [0.0] + bin_means + [1.0]
        diag_min, diag_max = min(all_vals), max(all_vals)
        ax.plot([diag_min, diag_max], [diag_min, diag_max],
                linestyle="--", color="gray", linewidth=1, label="Perfect cal.")

        ax.set_xlabel("Mean uncertainty", fontsize=8)
        ax.set_ylabel("Error rate", fontsize=8)
        ax.set_title(METRIC_SHORT_LABELS[m], fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes[len(metric_order):]:
        ax.set_visible(False)

    plt.tight_layout()

    out_path = str(Path(output_dir) / "reliability_diagrams.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved reliability diagrams -> {out_path}")

    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({"charts/reliability_diagrams": _wandb.Image(fig)})
        except Exception as e:
            print(f"W&B reliability diagram logging failed: {e}")

    plt.close(fig)
    return [out_path]


def plot_compute_time_chart(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Horizontal bar chart of per-metric computation time (log scale).

    Averages vanilla_avg_compute_times and kg_avg_compute_times across all
    datasets/configs and plots side-by-side horizontal bars sorted by time
    (slowest first) with log-scale x-axis.
    Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    metric_order = DEFAULT_UNCERTAINTY_METRICS

    # Collect times across datasets/configs
    from collections import defaultdict as _dd
    vanilla_times: dict = _dd(list)
    kg_times: dict = _dd(list)

    for dataset_block in all_results:
        for cfg_res in dataset_block.get("config_results", []):
            for m, t in cfg_res.get("vanilla_avg_compute_times", {}).items():
                vanilla_times[m].append(float(t))
            for m, t in cfg_res.get("kg_avg_compute_times", {}).items():
                kg_times[m].append(float(t))

    # Need at least one metric
    all_metrics = set(vanilla_times.keys()) | set(kg_times.keys())
    if not all_metrics:
        return []

    # Average across configs/datasets
    vanilla_avg = {m: (sum(vanilla_times[m]) / len(vanilla_times[m])) if vanilla_times[m] else 0.0
                   for m in metric_order if m in all_metrics}
    kg_avg = {m: (sum(kg_times[m]) / len(kg_times[m])) if kg_times[m] else 0.0
              for m in metric_order if m in all_metrics}

    # Sort by average time (slowest first)
    metrics_present = [m for m in metric_order if m in all_metrics]
    metrics_sorted = sorted(metrics_present,
                            key=lambda m: (vanilla_avg.get(m, 0) + kg_avg.get(m, 0)) / 2,
                            reverse=True)

    labels = [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics_sorted]
    v_vals = [vanilla_avg.get(m, 1e-9) for m in metrics_sorted]
    k_vals = [kg_avg.get(m, 1e-9) for m in metrics_sorted]

    # Clamp to a tiny minimum to avoid log(0)
    v_vals = [max(v, 1e-9) for v in v_vals]
    k_vals = [max(v, 1e-9) for v in k_vals]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.65 + 1.5)))

    bar_height = 0.35
    y = np.arange(n)

    bars_v = ax.barh(y + bar_height / 2, v_vals, bar_height,
                     label="Vanilla RAG", color="#4C72B0", alpha=0.85)
    bars_k = ax.barh(y - bar_height / 2, k_vals, bar_height,
                     label="KG-RAG", color="#DD8452", alpha=0.85)

    # Annotate with value in ms
    for bar, val in zip(list(bars_v) + list(bars_k), v_vals + k_vals):
        w = bar.get_width()
        ax.text(w * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val * 1000:.1f} ms", va="center", fontsize=7)

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Computation time (seconds) — log scale", fontsize=9)
    ax.set_title("Per-Metric Computation Time (seconds per question)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    out_path = str(Path(output_dir) / "compute_time_chart.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved compute time chart -> {out_path}")

    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({"charts/compute_time_chart": _wandb.Image(fig)})
        except Exception as e:
            print(f"W&B compute time chart logging failed: {e}")

    plt.close(fig)
    return [out_path]


def plot_auroc_vs_compute_time(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Scatter plot of AUROC vs log10(compute time) with Pareto frontier.

    For each metric x system, collects the average compute time and average AUROC,
    plots them as a scatter with metric labels, draws the Pareto frontier (highest
    AUROC for each compute budget), colours Vanilla and KG-RAG differently, and
    adds an AUROC=0.5 reference line.
    Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    metric_order = DEFAULT_UNCERTAINTY_METRICS

    from collections import defaultdict as _dd

    # Collect (compute_time, auroc) per metric x system across datasets/configs
    vanilla_times: dict = _dd(list)
    kg_times: dict = _dd(list)
    vanilla_auroc: dict = _dd(list)
    kg_auroc: dict = _dd(list)

    for dataset_block in all_results:
        for cfg_res in dataset_block.get("config_results", []):
            auroc_aurec = cfg_res.get("auroc_aurec", {})
            for m in metric_order:
                v_t = cfg_res.get("vanilla_avg_compute_times", {}).get(m)
                k_t = cfg_res.get("kg_avg_compute_times", {}).get(m)
                v_auroc = _display_eval_metric_value(auroc_aurec.get("vanilla_rag", {}), m, "auroc")
                k_auroc = _display_eval_metric_value(auroc_aurec.get("kg_rag", {}), m, "auroc")
                if v_t is not None and v_auroc is not None and not np.isnan(float(v_auroc)):
                    vanilla_times[m].append(float(v_t))
                    vanilla_auroc[m].append(float(v_auroc))
                if k_t is not None and k_auroc is not None and not np.isnan(float(k_auroc)):
                    kg_times[m].append(float(k_t))
                    kg_auroc[m].append(float(k_auroc))

    # Build list of (log10_time, auroc, label, system) points
    points = []
    for m in metric_order:
        label = METRIC_SHORT_LABELS.get(m, m)
        if vanilla_times[m]:
            avg_t = sum(vanilla_times[m]) / len(vanilla_times[m])
            avg_a = sum(vanilla_auroc[m]) / len(vanilla_auroc[m])
            points.append((np.log10(max(avg_t, 1e-9)), avg_a, label, "vanilla"))
        if kg_times[m]:
            avg_t = sum(kg_times[m]) / len(kg_times[m])
            avg_a = sum(kg_auroc[m]) / len(kg_auroc[m])
            points.append((np.log10(max(avg_t, 1e-9)), avg_a, label, "kg"))

    if not points:
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"vanilla": "#4C72B0", "kg": "#DD8452"}
    markers = {"vanilla": "o", "kg": "s"}
    plotted_systems: set = set()

    for log_t, auroc, label, system in points:
        lbl = ("Vanilla RAG" if system == "vanilla" else "KG-RAG") if system not in plotted_systems else None
        plotted_systems.add(system)
        ax.scatter(log_t, auroc, color=colors[system], marker=markers[system],
                   s=80, alpha=0.85, label=lbl, zorder=3)
        ax.annotate(label, (log_t, auroc), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)

    # Pareto frontier: points where no other point has both lower log_t AND higher auroc
    all_log_t = np.array([p[0] for p in points])
    all_auroc = np.array([p[1] for p in points])

    pareto_mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            # j dominates i if j has lower or equal time AND strictly higher AUROC
            if all_log_t[j] <= all_log_t[i] and all_auroc[j] > all_auroc[i]:
                pareto_mask[i] = False
                break

    pareto_pts = [(all_log_t[i], all_auroc[i]) for i in range(len(points)) if pareto_mask[i]]
    if pareto_pts:
        pareto_pts_sorted = sorted(pareto_pts, key=lambda p: p[0])
        px, py = zip(*pareto_pts_sorted)
        ax.step(px, py, where="post", color="green", linewidth=1.8,
                linestyle="--", label="Pareto frontier", zorder=2)

    # AUROC = 0.5 reference
    ax.axhline(0.5, color="red", linewidth=1, linestyle=":", label="AUROC=0.5 (random)")

    ax.set_xlabel("log\u2081\u2080(Compute time per question / s)", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=10)
    ax.set_title("AUROC vs Computation Time (Pareto Frontier)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    out_path = str(Path(output_dir) / "auroc_vs_compute_time.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved AUROC vs compute time chart -> {out_path}")

    if wandb_run is not None:
        try:
            import wandb as _wandb
            wandb_run.log({"charts/auroc_vs_compute_time": _wandb.Image(fig)})
        except Exception as e:
            print(f"W&B AUROC-vs-compute-time logging failed: {e}")

    plt.close(fig)
    return [out_path]


def plot_complementarity_matrix(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
):
    """
    Plot a 2×2 complementarity matrix per dataset (Zhang et al. 2025 methodology).

    Quadrants:
      - Both correct  (top-right, green)
      - Vanilla-only  (top-left,  blue)
      - KG-only       (bot-right, orange)
      - Neither       (bot-left,  red)

    Returns list of saved file paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for dataset_block in all_results:
        dataset_name = dataset_block.get("dataset", "unknown")
        for cfg_res in dataset_block.get("config_results", []):
            comp = cfg_res.get("complementarity")
            if not comp:
                continue

            labels = ["Both\ncorrect", "Vanilla\nonly", "KG\nonly", "Neither"]
            values = [
                comp.get("both_correct_pct", 0),
                comp.get("vanilla_only_pct", 0),
                comp.get("kg_only_pct", 0),
                comp.get("neither_correct_pct", 0),
            ]
            colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
            for bar, pct in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                )
            cfg_name = cfg_res.get("config", {}).get("name", "default")
            ax.set_title(f"Complementarity — {dataset_name} ({cfg_name})", fontsize=13)
            ax.set_ylabel("% of questions", fontsize=11)
            ax.set_ylim(0, max(values) * 1.25 + 5)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            fname = out_dir / f"complementarity_{dataset_name}_{cfg_name}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            saved.append(str(fname))
            if wandb_run:
                import wandb as _wandb
                wandb_run.log({f"charts/{dataset_name}/{cfg_name}/complementarity": _wandb.Image(fig)})
            plt.close(fig)

    return saved


def plot_query_type_stratification(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
):
    """
    Plot accuracy and best-metric AUROC broken down by task_type per dataset.

    For each dataset×config that has accuracy_by_task_type data, produces:
      1. Grouped bar chart: Vanilla vs KG accuracy per task type.
      2. Grouped bar chart: Vanilla vs KG AUROC (semantic_entropy) per task type.

    Returns list of saved file paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    _AUROC_METRIC = "semantic_entropy"  # representative metric for stratification plot

    for dataset_block in all_results:
        dataset_name = dataset_block.get("dataset", "unknown")
        for cfg_res in dataset_block.get("config_results", []):
            by_type = cfg_res.get("accuracy_by_task_type")
            if not by_type or len(by_type) < 2:
                # Only one task type — nothing interesting to stratify
                continue

            cfg_name = cfg_res.get("config", {}).get("name", "default")
            # Sort types by descending n so most common come first
            sorted_types = sorted(by_type.keys(), key=lambda t: -by_type[t].get("n", 0))

            x = list(range(len(sorted_types)))
            width = 0.38
            v_acc = [by_type[t].get("vanilla_accuracy", 0) * 100 for t in sorted_types]
            k_acc = [by_type[t].get("kg_accuracy", 0) * 100     for t in sorted_types]
            n_labels = [f"{t}\n(n={by_type[t].get('n',0)})" for t in sorted_types]

            # Accuracy panel
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            ax.bar([p - width / 2 for p in x], v_acc, width, label="Vanilla RAG", color="#1f77b4", alpha=0.85)
            ax.bar([p + width / 2 for p in x], k_acc, width, label="KG-RAG",      color="#ff7f0e", alpha=0.85)
            ax.set_title(f"Accuracy by task type — {dataset_name}", fontsize=12)
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(n_labels, fontsize=9)
            ax.set_ylim(0, 110)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()

            # AUROC panel (semantic_entropy as representative)
            v_auroc, k_auroc = [], []
            for t in sorted_types:
                auroc_data = by_type[t].get("auroc_aurec", {})
                v_auroc.append(_display_eval_metric_value(auroc_data.get("vanilla_rag", {}), _AUROC_METRIC, "auroc") or 0)
                k_auroc.append(_display_eval_metric_value(auroc_data.get("kg_rag", {}), _AUROC_METRIC, "auroc") or 0)

            ax2 = axes[1]
            ax2.bar([p - width / 2 for p in x], v_auroc, width, label="Vanilla RAG", color="#1f77b4", alpha=0.85)
            ax2.bar([p + width / 2 for p in x], k_auroc, width, label="KG-RAG",      color="#ff7f0e", alpha=0.85)
            ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="random")
            ax2.set_title(f"SE AUROC by task type — {dataset_name}", fontsize=12)
            ax2.set_ylabel("AUROC")
            ax2.set_xticks(x)
            ax2.set_xticklabels(n_labels, fontsize=9)
            ax2.set_ylim(0, 1.05)
            ax2.grid(axis="y", alpha=0.3)
            ax2.legend()

            plt.suptitle(f"{dataset_name} / {cfg_name}", fontsize=13, y=1.02)
            plt.tight_layout()

            fname = out_dir / f"query_type_stratification_{dataset_name}_{cfg_name}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            saved.append(str(fname))
            if wandb_run:
                import wandb as _wandb
                wandb_run.log({f"charts/{dataset_name}/{cfg_name}/query_type_stratification": _wandb.Image(fig)})
            plt.close(fig)

    return saved


def log_to_wandb(results: dict, aggregated: dict, overall: dict, metric_names: list, project_name: str = "kg-rag-evaluation"):
    """Log results to Weights & Biases."""
    if not HAS_WANDB:
        print("Skipping wandb logging (wandb not installed)")
        return
    
    # Initialize wandb
    run = wandb.init(project=project_name, job_type="evaluation")
    
    # Log overall metrics as a table
    metrics_table = wandb.Table(columns=["Metric", "Vanilla RAG", "KG-RAG", "Difference"])
    for metric in metric_names:
        v = overall['vanilla_rag'].get(metric, 0)
        k = overall['kg_rag'].get(metric, 0)
        diff = v - k
        metrics_table.add_data(metric.replace('_', ' ').title(), f"{v:.2f}", f"{k:.2f}", f"{diff:+.2f}")
    
    wandb.log({"overall_metrics": metrics_table})
    
    # Log aggregated metrics by config
    for config, systems in aggregated.items():
        payload = {}
        for metric in metric_names:
            payload[f"config/{config}/vanilla_rag_{metric}"] = systems['vanilla_rag'].get(metric, 0)
            payload[f"config/{config}/kg_rag_{metric}"] = systems['kg_rag'].get(metric, 0)
        wandb.log(payload)
    
    # Log per-dataset metrics
    if 'per_dataset_results' in results:
        for dataset_name, dataset_data in results['per_dataset_results'].items():
            analysis = dataset_data.get('analysis', {})
            for metric, value in analysis.get('vanilla_rag_stats', {}).items():
                wandb.log({f"dataset/{dataset_name}/vanilla_rag_{metric}": value})
            for metric, value in analysis.get('kg_rag_stats', {}).items():
                wandb.log({f"dataset/{dataset_name}/kg_rag_{metric}": value})
    
    # Log comparison chart
    if HAS_MATPLOTLIB:
        fig = create_overall_comparison_chart(overall, metric_names)
        wandb.log({"overall_comparison": wandb.Image(fig)})
        plt.close(fig)
        
        fig2 = create_comparison_chart(aggregated, metric_names)
        wandb.log({"comparison_by_config": wandb.Image(fig2)})
        plt.close(fig2)
    
    # Log experiment metadata
    wandb.log({
        "experiment_id": results.get('experiment_id', 'unknown'),
        "total_experiments": results.get('combined_analysis', {}).get('total_experiments', 0) or results.get('analysis', {}).get('total_experiments', 0),
    })
    
    run.finish()
    print(f"Logged to wandb project: {project_name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RAG experiment results")
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='results/visualizations', help='Output directory for charts')
    parser.add_argument('--project', type=str, default='kg-rag-evaluation', help='Wandb project name')
    parser.add_argument('--no-wandb', action='store_true', help='Skip wandb logging')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    # Check if multi-dataset
    is_multi_dataset = 'per_dataset_results' in results
    
    # Aggregate results
    print("Aggregating results by configuration...")
    metric_names = _get_metric_names(results)
    aggregated = aggregate_by_config(results)
    overall = aggregate_overall(results)
    
    # Print summary
    print("\n=== Overall Results ===")
    print(f"Vanilla RAG: {overall['vanilla_rag']}")
    print(f"KG-RAG: {overall['kg_rag']}")
    
    # Print per-dataset summary if available
    if is_multi_dataset:
        print_per_dataset_summary(results)
    
    print("\n=== By Configuration ===")
    for config, systems in sorted(aggregated.items()):
        print(f"\n{config}:")
        vanilla_preview = ", ".join(
            [f"{m}={systems['vanilla_rag'].get(m, 0):.4f}" for m in metric_names[:3]]
        )
        kg_preview = ", ".join(
            [f"{m}={systems['kg_rag'].get(m, 0):.4f}" for m in metric_names[:3]]
        )
        print(f"  Vanilla RAG: {vanilla_preview}")
        print(f"  KG-RAG:      {kg_preview}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate charts
    if HAS_MATPLOTLIB:
        print("\nGenerating charts...")
        overall_chart_path = output_dir / "overall_comparison.png"
        config_chart_path = output_dir / "comparison_by_config.png"
        
        create_overall_comparison_chart(overall, metric_names, str(overall_chart_path))
        create_comparison_chart(aggregated, metric_names, str(config_chart_path))
    
    # Log to wandb
    if not args.no_wandb and HAS_WANDB:
        print("\nLogging to Weights & Biases...")
        log_to_wandb(results, aggregated, overall, metric_names, args.project)
    elif args.no_wandb:
        print("\nSkipping wandb logging (--no-wandb flag)")
    else:
        print("\nSkipping wandb logging (wandb not installed)")
    
    print("\nDone!")


# ── Per-system AUROC comparison chart ─────────────────────────────────────────

_ALL_METRIC_LABELS = {
    "semantic_entropy":               "SE",
    "discrete_semantic_entropy":      "DSE",
    "sre_uq":                         "SRE-UQ",
    "p_true":                         "P(True)",
    "selfcheckgpt":                   "SelfCk",
    "vn_entropy":                     "VN-Ent",
    "sd_uq":                          "SD-UQ",
    "graph_path_support":             "GPS",
    "graph_path_disagreement":        "GPD",
    "competing_answer_alternatives":  "CAA",
    "evidence_vn_entropy":            "EVN-Ent",
    "subgraph_informativeness":       "SI",
    "subgraph_perturbation_stability": "SPS",
    "support_entailment_uncertainty": "SEU",
    "evidence_conflict_uncertainty":  "ECU",
}

_METRIC_FAMILY = {
    "semantic_entropy":               "output",
    "discrete_semantic_entropy":      "output",
    "sre_uq":                         "output",
    "p_true":                         "output",
    "selfcheckgpt":                   "output",
    "vn_entropy":                     "output",
    "sd_uq":                          "output",
    "graph_path_support":             "structural",
    "graph_path_disagreement":        "structural",
    "competing_answer_alternatives":  "structural",
    "evidence_vn_entropy":            "structural",
    "subgraph_informativeness":       "structural",
    "subgraph_perturbation_stability": "structural",
    "support_entailment_uncertainty": "grounding",
    "evidence_conflict_uncertainty":  "grounding",
}

_FAMILY_COLOR = {"output": "#3A7DC9", "structural": "#E07B39", "grounding": "#4BAE8A"}


def _display_eval_metric_value(
    system_metrics: dict,
    metric_name: str,
    suffix: str,
):
    """Return the display value for an evaluation metric, preferring conditional structural variants."""
    if metric_name in {"graph_path_support", "subgraph_perturbation_stability"}:
        non_null_key = f"{metric_name}_{suffix}_non_null"
        if non_null_key in system_metrics:
            return system_metrics.get(non_null_key, float("nan"))
    return system_metrics.get(f"{metric_name}_{suffix}", float("nan"))


def _metric_score_for_correlation(
    detail: dict,
    *,
    prefix: str,
    metric_name: str,
):
    """Return a score suitable for cross-metric correlation analysis."""
    if bool(detail.get(f"{prefix}_generation_failed", False)):
        return float("nan")
    if metric_name == "graph_path_support":
        if str(detail.get(f"{prefix}_graph_path_support_null_reason", "")):
            return float("nan")
    if metric_name == "subgraph_perturbation_stability":
        null_key = f"{prefix}_subgraph_perturbation_stability_null_reason"
        if str(detail.get(null_key, "")):
            return float("nan")
        if null_key not in detail:
            legacy_value = detail.get(f"{prefix}_{metric_name}")
            if legacy_value is not None:
                try:
                    if abs(float(legacy_value) - 0.5) <= 1e-12:
                        return float("nan")
                except (TypeError, ValueError):
                    return float("nan")

    value = detail.get(f"{prefix}_{metric_name}")
    if value is None:
        return float("nan")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if metric_name == "p_true":
        return -score
    return score


def plot_per_system_auroc_comparison(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Grouped bar chart: AUROC per metric, Vanilla vs KG-RAG side-by-side.

    This is the central paper figure showing that structural metrics (GPS, CAA, SI)
    have higher AUROC for KG-RAG outputs than output-side metrics do, while output
    metrics dominate for Vanilla RAG. Bars are coloured by metric family.

    One chart per dataset+config combination. Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    import matplotlib.ticker as mticker

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    metric_order = [m for m in _ALL_METRIC_LABELS if m in _METRIC_FAMILY]

    for dataset_block in all_results:
        dataset_name = dataset_block.get("dataset", "unknown")
        for cfg_res in dataset_block.get("config_results", []):
            auroc_data = cfg_res.get("auroc_aurec", {})
            if not auroc_data:
                continue

            cfg_name = cfg_res.get("config", {}).get("name", "default")

            v_data = auroc_data.get("vanilla_rag", {})
            k_data = auroc_data.get("kg_rag", {})

            # Only keep metrics present in at least one system
            present = [
                m for m in metric_order
                if not (
                    np.isnan(float(_display_eval_metric_value(v_data, m, "auroc")))
                    and np.isnan(float(_display_eval_metric_value(k_data, m, "auroc")))
                )
            ]
            if not present:
                continue

            n = len(present)
            x = np.arange(n)
            bar_w = 0.35

            fig, ax = plt.subplots(figsize=(max(10, n * 0.85 + 2), 5))

            for xi, metric in enumerate(present):
                v_val = float(_display_eval_metric_value(v_data, metric, "auroc"))
                k_val = float(_display_eval_metric_value(k_data, metric, "auroc"))
                family = _METRIC_FAMILY.get(metric, "output")
                fc = _FAMILY_COLOR[family]

                if not np.isnan(v_val):
                    ax.bar(xi - bar_w / 2, v_val, bar_w,
                           color=fc, alpha=0.55, edgecolor="white", linewidth=0.8)
                if not np.isnan(k_val):
                    ax.bar(xi + bar_w / 2, k_val, bar_w,
                           color=fc, alpha=1.0, edgecolor="white", linewidth=0.8)

            ax.axhline(0.5, color="#999999", linewidth=1.0, linestyle="--", zorder=0, label="chance")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_ALL_METRIC_LABELS.get(m, m) for m in present],
                rotation=35, ha="right", fontsize=9,
            )
            ax.set_ylabel("AUROC", fontsize=10)
            ax.set_ylim(0.2, 1.02)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.set_title(
                f"Per-metric AUROC — {dataset_name} ({cfg_name})\n"
                f"Light bar = Vanilla RAG   Dark bar = KG-RAG   Colour = family",
                fontsize=10, fontweight="bold",
            )

            # Family legend patches
            import matplotlib.patches as mpatches
            legend_handles = [
                mpatches.Patch(color=_FAMILY_COLOR["output"],     label="Output"),
                mpatches.Patch(color=_FAMILY_COLOR["structural"], label="Structural"),
                mpatches.Patch(color=_FAMILY_COLOR["grounding"],  label="Grounding"),
                mpatches.Patch(color="#aaaaaa", alpha=0.5,        label="Vanilla RAG (light)"),
                mpatches.Patch(color="#aaaaaa", alpha=1.0,        label="KG-RAG (dark)"),
            ]
            ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=2)

            plt.tight_layout()
            fname = out_dir / f"per_system_auroc_{dataset_name}_{cfg_name}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            saved.append(str(fname))
            if wandb_run:
                import wandb as _wandb
                wandb_run.log({f"charts/{dataset_name}/per_system_auroc": _wandb.Image(fig)})
            plt.close(fig)

    return saved


def plot_metric_spearman_matrix(
    all_results: list,
    output_dir: str = "results/visualizations",
    wandb_run=None,
) -> list:
    """Pairwise Spearman ρ heatmap across all uncertainty metrics.

    Justifies claiming 15 "complementary" metrics by showing which pairs are
    correlated (redundant) vs. independent (complementary). One chart pooling
    all questions across all datasets and both RAG systems.

    Renders three panels:
      - Vanilla RAG scores
      - KG-RAG scores
      - Difference (KG ρ − Vanilla ρ), showing where KG decorrelates metrics

    Returns list of saved PNG paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    try:
        from scipy.stats import spearmanr
    except ImportError:
        return []

    metric_order = list(_ALL_METRIC_LABELS.keys())
    labels = [_ALL_METRIC_LABELS[m] for m in metric_order]

    # Collect pooled, row-aligned score vectors so pairwise correlations are
    # computed on the same question rows after per-metric validity filtering.
    vanilla_rows = []
    kg_rows = []
    for dataset_block in all_results:
        for cfg_res in dataset_block.get("config_results", []):
            for detail in cfg_res.get("details", []):
                vanilla_rows.append({
                    m: _metric_score_for_correlation(detail, prefix="vanilla", metric_name=m)
                    for m in metric_order
                })
                kg_rows.append({
                    m: _metric_score_for_correlation(detail, prefix="kg", metric_name=m)
                    for m in metric_order
                })

    n_metrics = len(metric_order)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def _spearman_matrix(rows: list) -> np.ndarray:
        mat = np.full((n_metrics, n_metrics), float("nan"))
        if not rows:
            return mat
        per_metric = {
            metric_name: np.array([row[metric_name] for row in rows], dtype=float)
            for metric_name in metric_order
        }
        for i, mi in enumerate(metric_order):
            for j, mj in enumerate(metric_order):
                xi = per_metric[mi]
                xj = per_metric[mj]
                mask = (~np.isnan(xi)) & (~np.isnan(xj))
                if int(mask.sum()) < 10:
                    continue
                try:
                    rho, _ = spearmanr(xi[mask], xj[mask])
                    mat[i, j] = float(rho)
                except Exception:
                    pass
        return mat

    v_mat = _spearman_matrix(vanilla_rows)
    k_mat = _spearman_matrix(kg_rows)
    diff_mat = k_mat - v_mat

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    titles = ["Vanilla RAG — Spearman ρ", "KG-RAG — Spearman ρ", "Δ ρ  (KG − Vanilla)"]
    matrices = [v_mat, k_mat, diff_mat]
    cmaps = ["RdYlGn", "RdYlGn", "RdBu"]
    vmins = [-1.0, -1.0, -0.5]
    vmaxs = [1.0, 1.0, 0.5]

    family_sep_positions = []
    prev_family = None
    for idx, m in enumerate(metric_order):
        fam = _METRIC_FAMILY.get(m, "output")
        if prev_family is not None and fam != prev_family:
            family_sep_positions.append(idx)
        prev_family = fam

    for ax, mat, title, cmap, vmin, vmax in zip(axes, matrices, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(n_metrics):
            for j in range(n_metrics):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if abs(v) < 0.6 else "white")

        # Family separator lines
        for sep in family_sep_positions:
            ax.axhline(sep - 0.5, color="white", linewidth=2.0)
            ax.axvline(sep - 0.5, color="white", linewidth=2.0)

        ax.set_xticks(range(n_metrics))
        ax.set_yticks(range(n_metrics))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    fig.suptitle(
        "Metric Spearman correlation — pooled across all datasets & questions\n"
        "Family separators: Output | Structural | Grounding",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    fname = out_dir / "metric_spearman_matrix.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    saved.append(str(fname))
    if wandb_run:
        import wandb as _wandb
        wandb_run.log({"charts/metric_spearman_matrix": _wandb.Image(fig)})
    plt.close(fig)
    return saved


if __name__ == "__main__":
    main()
