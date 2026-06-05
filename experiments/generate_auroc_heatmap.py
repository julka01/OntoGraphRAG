"""Generate paper-quality AUROC heatmap (Figure 5).

Saves to paper/figures/auroc_heatmap.pdf (and .png for preview).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import seaborn as sns

# ── Data sources ──────────────────────────────────────────────────────────────
RUNS = {
    "BioASQ":        "results/runs/20260402-172730-bioasq-n100-full-metrics-evaluation-subset/mirage_bioasq_results.json",
    "RealMedQA":     "results/runs/20260402-155138-realmedqa-n100-full-metrics-evaluation-subset/mirage_realmedqa_results.json",
    "PubMedQA":      "results/runs/20260401-074100-pubmedqa-n100-full-metrics-evaluation-subset-rebuildkg/mirage_pubmedqa_results.json",
    "2Wiki":         "results/runs/20260402-185601-2wikimultihopqa-n100-full-metrics-evaluation-subset/mirage_2wikimultihopqa_results.json",
    "HotpotQA":      "results/runs/20260403-103006-hotpotqa-n100-full-metrics-evaluation-subset/mirage_hotpotqa_results.json",
    "MultiHopRAG":   "results/runs/20260402-201208-multihoprag-n100-full-metrics-evaluation-subset/mirage_multihoprag_results.json",
}

# ── Measure layout ────────────────────────────────────────────────────────────
# (internal_key, display_label, family)
MEASURES = [
    # Output estimators
    ("semantic_entropy",              "SE",             "output"),
    ("discrete_semantic_entropy",     "DSE",            "output"),
    ("sre_uq",                        "SRE-UQ",         "output"),
    ("p_true",                        "P(True)",        "output"),
    ("selfcheckgpt",                  "SelfCheckGPT",   "output"),
    ("vn_entropy",                    "VN-Ent.",        "output"),
    ("sd_uq",                         "SD-UQ",          "output"),
    # Structural measures
    ("graph_path_support",            "GPS",            "structural"),
    ("graph_path_disagreement",       "GPD",            "structural"),
    ("competing_answer_alternatives", "CAA",            "structural"),
    ("evidence_vn_entropy",           "EVN-Ent.",       "structural"),
    ("subgraph_informativeness",      "SGI",            "structural"),
    ("subgraph_perturbation_stability", "SPS-UQ",       "structural"),
    # Grounding measures
    ("support_entailment_uncertainty","SEU",            "grounding"),
    ("evidence_conflict_uncertainty", "ECU",            "grounding"),
]

FAMILY_COLORS = {
    "output":     "#3A7DC9",   # blue
    "structural": "#E07B39",   # orange
    "grounding":  "#4BAE8A",   # green
}

DATASET_ORDER = ["BioASQ", "RealMedQA", "PubMedQA", "2Wiki", "HotpotQA", "MultiHopRAG"]
PREFERRED_CONFIG_NAME = os.environ.get("MIRAGE_CONFIG_NAME")


def _select_config_result(path, results_doc):
    config_results = results_doc.get("config_results", [])
    if not config_results:
        raise ValueError(f"{path} contains no config_results")
    if len(config_results) == 1:
        return config_results[0]

    if PREFERRED_CONFIG_NAME:
        for cfg_res in config_results:
            if cfg_res.get("config", {}).get("name") == PREFERRED_CONFIG_NAME:
                return cfg_res

    for cfg_res in config_results:
        if cfg_res.get("config", {}).get("name") == "default":
            return cfg_res

    config_names = [cfg.get("config", {}).get("name", "<unnamed>") for cfg in config_results]
    raise ValueError(
        f"{path} contains multiple configs {config_names}; "
        "set MIRAGE_CONFIG_NAME to choose one explicitly"
    )


def load_auroc(path, system):
    """Return {measure_key: auroc_value} for a system ('vanilla_rag' or 'kg_rag')."""
    with open(path) as f:
        d = json.load(f)
    cfg = _select_config_result(path, d)
    auroc_block = cfg.get("auroc_aurec", {}).get(system, {})
    out = {
        k.replace("_auroc", ""): v
        for k, v in auroc_block.items()
        if k.endswith("_auroc")
    }
    for structural_key in ("graph_path_support", "subgraph_perturbation_stability"):
        non_null_key = f"{structural_key}_auroc_non_null"
        if non_null_key in auroc_block:
            out[structural_key] = auroc_block[non_null_key]
    return out


def build_matrix(system):
    """Build (n_measures × n_datasets) matrix for a given system."""
    mat = np.full((len(MEASURES), len(DATASET_ORDER)), np.nan)
    for j, ds in enumerate(DATASET_ORDER):
        path = RUNS[ds]
        try:
            auroc = load_auroc(path, system)
        except Exception:
            continue
        for i, (key, _, family) in enumerate(MEASURES):
            mat[i, j] = auroc.get(key, np.nan)
    return mat


# ── Build matrices ────────────────────────────────────────────────────────────
mat_v = build_matrix("vanilla_rag")
mat_k = build_matrix("kg_rag")

n_measures = len(MEASURES)
n_datasets = len(DATASET_ORDER)
row_labels = [m[1] for m in MEASURES]
families   = [m[2] for m in MEASURES]

# Tighter range: 0.25–0.95 makes mid-range differences visible
VMIN, VCENTER, VMAX = 0.25, 0.50, 0.95
norm = TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)
cmap = sns.diverging_palette(10, 130, s=80, l=50, as_cmap=True)  # red→green

# ── Plot ──────────────────────────────────────────────────────────────────────
sns.set_style("white")
fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5.6),
    gridspec_kw={"wspace": 0.05},
)

def draw_panel(ax, mat, title, show_yticklabels=True):
    # Mask for seaborn (NaN → blank)
    mask = np.isnan(mat)
    mat_clipped = np.clip(mat, VMIN, VMAX)

    im = sns.heatmap(
        mat_clipped,
        ax=ax,
        mask=mask,
        cmap=cmap,
        norm=norm,
        linewidths=0.4,
        linecolor="#dddddd",
        square=False,
        cbar=False,
        annot=False,
        xticklabels=DATASET_ORDER,
        yticklabels=row_labels if show_yticklabels else False,
    )

    # Manual annotations (bold, readable)
    for r in range(n_measures):
        for c in range(n_datasets):
            val = mat[r, c]
            if np.isnan(val):
                ax.text(c + 0.5, r + 0.5, "—", ha="center", va="center",
                        fontsize=8, color="#aaaaaa")
            else:
                brightness = abs(val - VCENTER) / (VMAX - VMIN)
                txt_color = "white" if brightness > 0.30 else "#1a1a1a"
                ax.text(c + 0.5, r + 0.5, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color, fontweight="bold")

    # Family separator lines
    for r in range(1, n_measures):
        if families[r] != families[r - 1]:
            ax.axhline(r, color="white", linewidth=3.0, zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=30, labelsize=9, length=0)
    ax.tick_params(axis="y", rotation=0,  labelsize=9, length=0)
    ax.xaxis.set_tick_params(pad=2)

    return im

draw_panel(axes[0], mat_v, "Vanilla RAG", show_yticklabels=True)
draw_panel(axes[1], mat_k, "KG-RAG",      show_yticklabels=False)

# ── Family row labels in left margin ─────────────────────────────────────────
family_spans = []
current, start = families[0], 0
for i, f in enumerate(families[1:], 1):
    if f != current:
        family_spans.append((current, start, i - 1))
        current, start = f, i
family_spans.append((current, start, n_measures - 1))

family_display = {"output": "Output", "structural": "Structural", "grounding": "Grounding"}
for family, r0, r1 in family_spans:
    mid = (r0 + r1) / 2 + 0.5
    color = FAMILY_COLORS[family]
    axes[0].annotate(
        family_display[family],
        xy=(0, mid), xycoords=("axes fraction", "data"),
        xytext=(-0.22, mid), textcoords=("axes fraction", "data"),
        ha="center", va="center",
        fontsize=8.5, color=color, fontweight="bold", rotation=90,
        annotation_clip=False,
    )
    for sign, row in [(1, r0), (-1, r1)]:
        axes[0].annotate(
            "", xy=(-0.12, row + 0.5 - sign * 0.45),
            xycoords=("axes fraction", "data"),
            xytext=(-0.12, row + 0.5 + sign * 0.45),
            textcoords=("axes fraction", "data"),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.6),
            annotation_clip=False,
        )

# ── Shared colorbar ───────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.18, 0.016, 0.65])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("AUROC", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
cb.set_ticks([0.25, 0.40, 0.50, 0.65, 0.80, 0.95])
cb.ax.axhline((0.5 - VMIN) / (VMAX - VMIN), color="#333333",
              linewidth=1.0, linestyle="--")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(out_dir / "auroc_heatmap.pdf", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "auroc_heatmap.png", dpi=180, bbox_inches="tight")
print("Saved paper/figures/auroc_heatmap.pdf")
print("Saved paper/figures/auroc_heatmap.png")
plt.close(fig)
