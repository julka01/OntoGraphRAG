"""
Generate publication-quality paper figures for ACM SIGCONF two-column layout.

Figures:
  1. answer_auroc_heatmap.pdf     — answer-side AUROC, vanilla vs KG-RAG
  2. structural_auroc_heatmap.pdf — KG-only GPS AUROC with usable denominators
  3. context_collapse_ablation.pdf — RealMedQA strict-retrieval mechanism plot
  4. calibration_paradox.pdf      — Accuracy vs AUROC scatter (legacy)
  5. coverage_accuracy.pdf        — Selective-prediction curves by diagnostic family
  6. gps_abstention_map.pdf       — GPS usable/unavailable denominators
  7. adaptive_kg_auroc_heatmap.pdf — adaptive KG diagnostic AUROC only
"""

import glob
import json, os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from experiments.gps_v3_depth_matched import (  # noqa: E402
    GAMMA as GPS_V3_GAMMA,
    TAU as GPS_V3_TAU,
    expected_hop as gps_v3_expected_hop,
    load_artifact as load_gps_v3_artifact,
    row_map as gps_v3_row_map,
    score_from_store_depth_matched,
)

# ── Publication style ─────────────────────────────────────────────────────────
# Fonts and sizes tuned for two-column SIGCONF (column width ≈ 3.33 in).
# Full-width figures (figure*) can be up to ~7.0 in wide.

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Linux Libertine O", "Palatino", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "cm",          # Computer Modern math — matches LaTeX body
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        200,
    "savefig.dpi":       300,
    "axes.linewidth":    0.6,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.4,
    "lines.linewidth":   1.4,
    "patch.linewidth":   0.5,
})

# ── Colour palette (colour-blind-friendly, matches paper text colours) ────────
C_VANILLA  = "#2166AC"    # blue
C_KGRAG    = "#D6604D"    # red-orange
C_HIGHG    = "#1A9641"    # green
C_LOWG     = "#D7191C"    # red
C_ALL      = "#4393C3"    # light blue
C_RAND     = "#888888"    # grey dashed baseline

DATASET_PALETTE = {
    "Pubmedqa":        "#D6604D",
    "Realmedqa":       "#1A9641",
    "Hotpotqa":        "#AAAAAA",
    "HotpotqaFullWiki":"#7F7F7F",
    "2Wikimultihopqa": "#4393C3",
    "Musique":         "#8856A7",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def safe_auroc(scores, labels, *, higher_is_more_certain=False):
    pairs = [(s, l) for s, l in zip(scores, labels)
             if s is not None and l is not None]
    if len(pairs) < 4 or len(set(l for _, l in pairs)) < 2:
        return None
    s, l = zip(*pairs)
    uncertainty = [-x for x in s] if higher_is_more_certain else list(s)
    # Match experiments/uncertainty_metrics.py: AUROC predicts correctness
    # from negative uncertainty, after failed generations are excluded.
    return roc_auc_score(l, [-x for x in uncertainty])


def valid_metric_rows(rows, metric_key):
    """Drop structural fallback rows before AUROC, matching experiment code."""
    if metric_key.endswith("graph_path_support"):
        prefix = metric_key.split("_graph_path_support")[0]
        null_key = f"{prefix}_graph_path_support_null_reason"
        if any(null_key in r for r in rows):
            return [r for r in rows if not str(r.get(null_key, ""))]
        # Legacy paper-facing runs did not always persist an explicit GPS null
        # reason. In those artifacts, the documented abstention fallback is
        # exactly 0.5, so exclude that spike before AUROC.
        return [
            r for r in rows
            if abs(float(r.get(metric_key, 0.5)) - 0.5) > 1e-12
        ]
    return rows


def load_results():
    """Load the latest paper-facing result bundle.

    The paper compares dense vanilla RAG against entity-first KG-RAG.  The
    experiment files contain both retrieval configs for both systems, so we
    merge rows by question id: vanilla_* fields come from the dense config and
    kg_* fields come from the entity-first config.
    """
    root = REPO_ROOT
    dense_cfg = "dense_floor_thr0.1_k10_rt0p0"
    kg_cfg = "kg_entity_first_thr0.1_k10_rt0p0"
    dataset_slugs = {
        "Pubmedqa": "pubmedqa",
        "Realmedqa": "realmedqa",
        "Hotpotqa": "hotpotqa",
        "HotpotqaFullWiki": "hotpotqa_fullwiki",
        "2Wikimultihopqa": "2wikimultihopqa",
        "Musique": "musique",
    }
    search_roots = [
        os.path.join(root, "results", "latest_kg_design_final_metrics", "runs"),
        os.path.join(root, "results", "runs"),
    ]

    def select_config(doc, name):
        for cfg in doc.get("config_results", []):
            if cfg.get("config", {}).get("name") == name:
                return cfg
        raise ValueError(f"missing config {name}")

    def run_id(path):
        return os.path.basename(os.path.dirname(path))

    def has_required_configs(doc):
        try:
            dense = select_config(doc, dense_cfg)
            kg = select_config(doc, kg_cfg)
        except ValueError:
            return False
        return bool(dense.get("details")) and bool(kg.get("details"))

    latest = {}
    for name, slug in dataset_slugs.items():
        candidates = []
        for search_root in search_roots:
            pattern = os.path.join(search_root, "*", f"mirage_{slug}_results.json")
            candidates.extend(glob.glob(pattern))
        for path in sorted(set(candidates), key=run_id, reverse=True):
            with open(path) as f:
                doc = json.load(f)
            if has_required_configs(doc):
                latest[name] = path
                break
        if name not in latest:
            raise FileNotFoundError(
                f"Could not find a valid paper-facing result file for {name} "
                f"with configs {dense_cfg!r} and {kg_cfg!r}"
            )

    def row_id(row):
        return str(row.get("question_id") or row.get("question") or "")

    out = {}
    for name, path in latest.items():
        with open(path) as f:
            doc = json.load(f)
        dense = select_config(doc, dense_cfg)
        kg = select_config(doc, kg_cfg)
        kg_by_id = {row_id(row): row for row in kg.get("details", [])}
        rows = []
        for vanilla_row in dense.get("details", []):
            merged = dict(vanilla_row)
            kg_row = kg_by_id.get(row_id(vanilla_row))
            if kg_row:
                for key, value in kg_row.items():
                    if key.startswith("kg_"):
                        merged[key] = value
            rows.append(merged)
        out[name] = rows
    manifest_path = os.path.join(os.path.dirname(__file__), "latest_results_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                dataset: {
                    "run_id": run_id(path),
                    "result_path": os.path.relpath(path, root),
                }
                for dataset, path in sorted(latest.items())
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved {manifest_path}")
    apply_gps_v3_overlay(out)
    return out


# ── GPS overlay ───────────────────────────────────────────────────────────────
# GPS reuses the saved GPS replay stores (entity links + path lengths) but
# applies depth-matched distance decay.  No retrieval, generation, entity
# linking, or KG construction is rerun.

GPS_V3_SLUGS = {
    "Pubmedqa": "pubmedqa",
    "Realmedqa": "realmedqa",
    "Hotpotqa": "hotpotqa",
    "HotpotqaFullWiki": "hotpotqa_fullwiki",
    "2Wikimultihopqa": "2wikimultihopqa",
    "Musique": "musique",
}


def apply_gps_v3_overlay(results):
    for name, rows in results.items():
        slug = GPS_V3_SLUGS.get(name)
        if not slug:
            continue
        try:
            artifact = load_gps_v3_artifact(slug)
        except FileNotFoundError:
            print(f"WARNING: no GPS replay artifact for {name}; keeping logged values")
            continue
        hop_rows = gps_v3_row_map(artifact["source_result"])
        stores = {str(q["question_id"]): q.get("store", {}) for q in artifact["questions"]}
        for row in rows:
            qid = str(row.get("question_id"))
            store = stores.get(str(row.get("question_id")), {})
            for side in ("kg", "vanilla"):
                result = score_from_store_depth_matched(
                    store.get(side),
                    gps_v3_expected_hop(slug, hop_rows.get(qid)),
                )
                score = result["score"]
                abstained = bool(result["null_reason"])
                if not abstained and abs(score - 0.5) < 1e-12:
                    # nudge legitimate 0.5 scores off the abstention sentinel
                    score = 0.5 + 1e-6
                row[f"{side}_graph_path_support"] = score
    print(f"Applied GPS overlay (tau={GPS_V3_TAU}, gamma={GPS_V3_GAMMA})")
    return results


# ── Heatmap inputs ────────────────────────────────────────────────────────────

METRIC_PAIRS = [
    # (vanilla_key, kg_key, label, kg_only)
    ("vanilla_discrete_semantic_entropy",   "kg_discrete_semantic_entropy",   "DSE",         False),
    ("vanilla_p_true",                      "kg_p_true",                      r"$\mathrm{P(True)}$", False),
    ("vanilla_selfcheckgpt",                "kg_selfcheckgpt",                "SelfChkGPT",  False),
    ("vanilla_sre_uq",                      "kg_sre_uq",                      "SRE-UQ",      False),
    ("vanilla_vn_entropy",                  "kg_vn_entropy",                  "VN-Ent.",     False),
    ("vanilla_sd_uq",                       "kg_sd_uq",                       "SD-UQ",       False),
    ("vanilla_graph_path_support",          "kg_graph_path_support",          "GPS",         True),
    ("vanilla_support_entailment_uncertainty","kg_support_entailment_uncertainty","SEU",      False),
]
FAMILY_SPANS  = [(0,6), (6,7), (7,8)]
FAMILY_LABELS = ["Output", "Struct.", "Ground."]
FAMILY_HEADER_COLORS = ["#E8F1FB", "#F3EEE8", "#EAF6EA"]
CERTAINTY_METRICS = {"p_true"}

DATASETS_ORDER = [
    "Pubmedqa",
    "Realmedqa",
    "Hotpotqa",
    "HotpotqaFullWiki",
    "2Wikimultihopqa",
    "Musique",
]
DATASET_LABELS = {
    "2Wikimultihopqa": "2WikiMHQA",
    "Realmedqa":       "RealMedQA",
    "Musique":         "MuSiQue",
    "Pubmedqa":        "PubMedQA",
    "Hotpotqa":        "HotpotQA",
    "HotpotqaFullWiki":"HotpotQA-FW",
}
MIN_STRUCTURAL_ROWS = 50

# colours for heatmap cells: RdYlGn with custom anchors
from matplotlib.colors import LinearSegmentedColormap
_HMAP_COLORS = [
    (0.00, "#B2182B"),   # 0.00 — deep red (inverted ranking)
    (0.35, "#D6604D"),   # 0.35 — orange-red
    (0.50, "#F7F7F7"),   # 0.50 — neutral white (chance baseline)
    (0.62, "#E2EDF5"),   # 0.62 — barely tinted: near-chance cells stay neutral
    (0.72, "#AECFE6"),   # 0.72 — light blue
    (0.82, "#6FAAD2"),   # 0.82 — mid blue: spreads the 0.7-0.85 working range
    (1.00, "#2166AC"),   # 1.00 — deep blue (colourblind-safe vs red)
]
HMAP_CMAP = LinearSegmentedColormap.from_list(
    "auroc_cmap",
    [(v, c) for v, c in _HMAP_COLORS]
)
HMAP_CMAP.set_bad("#F4F4F4")


def make_heatmap(results):
    datasets = [d for d in DATASETS_ORDER if d in results]
    n_d, n_m = len(datasets), len(METRIC_PAIRS)

    vanilla_mat = np.full((n_d, n_m), np.nan)
    kg_mat      = np.full((n_d, n_m), np.nan)
    vanilla_labels = [["" for _ in range(n_m)] for _ in range(n_d)]
    kg_labels = [["" for _ in range(n_m)] for _ in range(n_d)]

    for di, ds in enumerate(datasets):
        rows = results[ds]
        for mi, (vm, km, _, kg_only) in enumerate(METRIC_PAIRS):
            v_rows = [r for r in rows if not r.get("vanilla_generation_failed")]
            k_rows = [r for r in rows if not r.get("kg_generation_failed")]
            v_rows = valid_metric_rows(v_rows, vm)
            k_rows = valid_metric_rows(k_rows, km)
            v_labels = [int(r.get("vanilla_correct", 0)) for r in v_rows]
            k_labels = [int(r.get("kg_correct",      0)) for r in k_rows]
            v_metric = vm.replace("vanilla_", "")
            k_metric = km.replace("kg_", "")
            if not kg_only:
                v = safe_auroc(
                    [r.get(vm) for r in v_rows],
                    v_labels,
                    higher_is_more_certain=v_metric in CERTAINTY_METRICS,
                )
                if v is not None:
                    vanilla_mat[di, mi] = v
                else:
                    vanilla_labels[di][mi] = "n/a"
            else:
                # Graph-only metrics stay undefined in vanilla RAG.
                vanilla_labels[di][mi] = "—"
            if kg_only and len(k_rows) < MIN_STRUCTURAL_ROWS:
                kg_labels[di][mi] = "n/a"
            else:
                k = safe_auroc(
                    [r.get(km) for r in k_rows],
                    k_labels,
                    higher_is_more_certain=k_metric in CERTAINTY_METRICS,
                )
                if k is not None:
                    kg_mat[di, mi] = k
                else:
                    kg_labels[di][mi] = "n/a"

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.82), sharey=True,
                              gridspec_kw={"wspace": 0.04})
    kw = dict(vmin=0.0, vmax=1.0, cmap=HMAP_CMAP, aspect="auto")
    xlabels = [m for _, _, m, _ in METRIC_PAIRS]
    ylabels = [DATASET_LABELS[d] for d in datasets]

    for ax_idx, (ax, mat, title) in enumerate(zip(
            axes,
            [vanilla_mat, kg_mat],
            ["Vanilla RAG", "KG-RAG"])):
        im = ax.imshow(mat, **kw)
        ax.grid(False)
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(xlabels, rotation=36, ha="right", fontsize=7.2)
        ax.set_yticks(range(n_d))
        if ax_idx == 0:
            ax.set_yticklabels(ylabels, fontsize=8.0)
            ax.tick_params(axis="y", labelleft=True)
        else:
            # Shared y-axes can drop labels on *both* panels if we set an empty
            # ticklabel list here, so hide only the right-panel labels.
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=24)
        ax.set_xticks(np.arange(-0.5, n_m, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_d, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for di in range(n_d):
            for mi in range(n_m):
                v = mat[di, mi]
                if np.isnan(v):
                    label = vanilla_labels[di][mi] if ax_idx == 0 else kg_labels[di][mi]
                    ax.text(mi, di, label or "n/a", ha="center", va="center",
                            fontsize=6.2, color="#888888",
                            style="italic" if label == "n/a" else "normal")
                else:
                    col = "white" if (v < 0.38 or v > 0.72) else "#222222"
                    ax.text(mi, di, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=col, fontweight="bold")

        # family separators
        for lo, _ in FAMILY_SPANS[1:]:
            ax.axvline(lo - 0.5, color="white", linewidth=2.5, zorder=4)

        # family labels above each panel
        from matplotlib.transforms import blended_transform_factory
        xform = blended_transform_factory(ax.transData, ax.transAxes)
        for (lo, hi), flabel, fcolor in zip(FAMILY_SPANS, FAMILY_LABELS, FAMILY_HEADER_COLORS):
            mid = (lo + hi - 1) / 2
            ax.add_patch(
                Rectangle(
                    (lo - 0.5, 1.005),
                    hi - lo,
                    0.072,
                    transform=xform,
                    facecolor=fcolor,
                    edgecolor="none",
                    clip_on=False,
                    zorder=2,
                )
            )
            ax.text(mid, 1.01, flabel, ha="center", va="bottom",
                    fontsize=6.7, color="#444444", fontweight="bold",
                    transform=xform)

    fig.subplots_adjust(left=0.12, right=0.93, top=0.84, bottom=0.18)
    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015, shrink=0.88)
    cbar.set_label("AUROC", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")

    out_pdf = os.path.join(os.path.dirname(__file__), "auroc_heatmap.pdf")
    out_png = os.path.join(os.path.dirname(__file__), "auroc_heatmap.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


def make_answer_heatmap(results):
    """Output + grounding AUROC for both systems, with no graph-only blanks."""
    metric_pairs = [pair for pair in METRIC_PAIRS if not pair[3]]
    datasets = [d for d in DATASETS_ORDER if d in results]
    n_d, n_m = len(datasets), len(metric_pairs)
    mats = [np.full((n_d, n_m), np.nan), np.full((n_d, n_m), np.nan)]

    for di, ds in enumerate(datasets):
        rows = results[ds]
        for mi, (vm, km, _, _) in enumerate(metric_pairs):
            for ai, (prefix, key) in enumerate((("vanilla", vm), ("kg", km))):
                app_rows = [r for r in rows if not r.get(f"{prefix}_generation_failed")]
                labels = [int(r.get(f"{prefix}_correct", 0)) for r in app_rows]
                metric = key.replace(f"{prefix}_", "")
                val = safe_auroc(
                    [r.get(key) for r in app_rows],
                    labels,
                    higher_is_more_certain=metric in CERTAINTY_METRICS,
                )
                if val is not None:
                    mats[ai][di, mi] = val

    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 3.35), sharey=True, gridspec_kw={"wspace": 0.04}
    )
    xlabels = [m for _, _, m, _ in metric_pairs]
    ylabels = [DATASET_LABELS[d] for d in datasets]
    im = None
    for ax_idx, (ax, mat, title) in enumerate(
            zip(axes, mats, ["Vanilla Dense", "Entity-First KG"])):
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=HMAP_CMAP, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(xlabels, rotation=34, ha="right", fontsize=7.3)
        ax.set_yticks(range(n_d))
        if ax_idx == 0:
            ax.set_yticklabels(ylabels, fontsize=8.0)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
        ax.set_xticks(np.arange(-0.5, n_m, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_d, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for di in range(n_d):
            for mi in range(n_m):
                val = mat[di, mi]
                if np.isnan(val):
                    ax.text(mi, di, "n/a", ha="center", va="center",
                            fontsize=6.2, color="#888888", style="italic")
                else:
                    col = "white" if (val < 0.38 or val > 0.72) else "#222222"
                    ax.text(mi, di, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=col, fontweight="bold")

    fig.subplots_adjust(left=0.12, right=0.93, top=0.91, bottom=0.19)
    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015, shrink=0.88)
    cbar.set_label("AUROC", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"answer_auroc_heatmap.{ext}")
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def make_structural_heatmap(results):
    """KG-only GPS AUROC with usable denominators printed in each cell."""
    datasets = [d for d in DATASETS_ORDER if d in results]
    metrics = [
        ("kg_graph_path_support", "GPS"),
    ]
    mat = np.full((len(datasets), len(metrics)), np.nan)
    labels = [["" for _ in metrics] for _ in datasets]

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    replay_path = os.path.join(
        root,
        "results/latest_kg_design_final_metrics/runs/"
        "20260527-163249-realmedqa-n230-full-metrics-evaluation-subset-rebuildkg/"
        "realmedqa_structural_metric_replay_20260528.json",
    )
    replay = None  # superseded by the GPS overlay applied in load_results

    for di, ds in enumerate(datasets):
        rows = [r for r in results[ds] if not r.get("kg_generation_failed")]
        answered = len(rows)
        for mi, (mkey, _) in enumerate(metrics):
            if ds == "Realmedqa" and replay:
                usable = int(replay.get("gps_defined", 0))
                val = float(replay.get("gps_auroc_non_null", np.nan))
                mat[di, mi] = val
                labels[di][mi] = f"{val:.2f}\n{usable}/{answered}"
                continue

            metric_rows = valid_metric_rows(rows, mkey)
            usable = len(metric_rows)
            if usable >= MIN_STRUCTURAL_ROWS:
                val = safe_auroc(
                    [r.get(mkey) for r in metric_rows],
                    [int(r.get("kg_correct", 0)) for r in metric_rows],
                )
                if val is not None:
                    mat[di, mi] = val
                    labels[di][mi] = f"{val:.2f}\n{usable}/{answered}"
                    continue
            labels[di][mi] = f"low n\n{usable}/{answered}"

    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=HMAP_CMAP, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], fontsize=8)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets], fontsize=8)
    ax.set_title("KG Structural Diagnostic", fontsize=9.5, fontweight="bold", pad=7)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for di in range(len(datasets)):
        for mi in range(len(metrics)):
            val = mat[di, mi]
            if np.isnan(val):
                ax.text(mi, di, labels[di][mi], ha="center", va="center",
                        fontsize=6.7, color="#777777", style="italic")
            else:
                col = "white" if (val < 0.38 or val > 0.72) else "#222222"
                ax.text(mi, di, labels[di][mi], ha="center", va="center",
                        fontsize=6.8, color=col, fontweight="bold", linespacing=0.95)

    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04, shrink=0.92)
    cbar.set_label("AUROC", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")
    ax.text(
        0.5, -0.16, "Cell text: AUROC and usable/answered rows.",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color="#555555"
    )
    fig.subplots_adjust(left=0.31, right=0.87, top=0.91, bottom=0.17)

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"structural_auroc_heatmap.{ext}")
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── Main-paper replacements: compact diagnostic figures ─────────────────────

OUTPUT_KEYS = [
    ("discrete_semantic_entropy", "DSE"),
    ("p_true", r"$\mathrm{P(True)}$"),
    ("selfcheckgpt", "SelfCheckGPT"),
    ("sre_uq", "SRE-UQ"),
    ("vn_entropy", "VN-Ent."),
    ("sd_uq", "SD-UQ"),
]


def _gps_summary_for_dataset(ds, rows):
    """Return (auroc, usable, answered), using the GPS replay overlay."""
    answered = len([r for r in rows if not r.get("kg_generation_failed")])
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    replay_path = os.path.join(
        root,
        "results/latest_kg_design_final_metrics/runs/"
        "20260527-163249-realmedqa-n230-full-metrics-evaluation-subset-rebuildkg/"
        "realmedqa_structural_metric_replay_20260528.json",
    )
    # Dataset-specific overrides are unnecessary: GPS is overlaid upstream.

    kg_rows = [r for r in rows if not r.get("kg_generation_failed")]
    metric_rows = valid_metric_rows(kg_rows, "kg_graph_path_support")
    usable = len(metric_rows)
    val = safe_auroc(
        [r.get("kg_graph_path_support") for r in metric_rows],
        [int(r.get("kg_correct", 0)) for r in metric_rows],
    )
    return (val, usable, answered)


def _best_output_auroc(rows, prefix):
    """Best output-side AUROC and its metric label for one system."""
    app_rows = [r for r in rows if not r.get(f"{prefix}_generation_failed")]
    labels = [int(r.get(f"{prefix}_correct", 0)) for r in app_rows]
    best = (None, None)
    for key, label in OUTPUT_KEYS:
        metric_key = f"{prefix}_{key}"
        val = safe_auroc(
            [r.get(metric_key) for r in app_rows],
            labels,
            higher_is_more_certain=key in CERTAINTY_METRICS,
        )
        if val is not None and (best[0] is None or val > best[0]):
            best = (val, label)
    return best


def _seu_auroc(rows, prefix):
    app_rows = [r for r in rows if not r.get(f"{prefix}_generation_failed")]
    return safe_auroc(
        [r.get(f"{prefix}_support_entailment_uncertainty") for r in app_rows],
        [int(r.get(f"{prefix}_correct", 0)) for r in app_rows],
    )


def make_family_auroc_summary(results):
    """Compact family-level AUROC figure replacing the large metric heatmap."""
    datasets = [d for d in DATASETS_ORDER if d in results]
    y = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(7.05, 3.45))

    rows_out = []
    for di, ds in enumerate(datasets):
        rows = results[ds]
        v_best, v_label = _best_output_auroc(rows, "vanilla")
        k_best, k_label = _best_output_auroc(rows, "kg")
        kg_seu = _seu_auroc(rows, "kg")
        gps, gps_usable, gps_answered = _gps_summary_for_dataset(ds, rows)
        rows_out.append((v_best, v_label, k_best, k_label, kg_seu, gps, gps_usable, gps_answered))

        if v_best is not None:
            ax.scatter(v_best, di - 0.20, s=46, marker="o", color=C_VANILLA,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(v_best + 0.012, di - 0.20, v_label, fontsize=5.8,
                    va="center", color=C_VANILLA)
        if k_best is not None:
            ax.scatter(k_best, di - 0.06, s=48, marker="o", color=C_KGRAG,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(k_best + 0.012, di - 0.06, k_label, fontsize=5.8,
                    va="center", color=C_KGRAG)
        if kg_seu is not None:
            ax.scatter(kg_seu, di + 0.08, s=54, marker="D", color="#6A51A3",
                       edgecolor="white", linewidth=0.6, zorder=4)
        if gps is not None and not np.isnan(gps):
            coverage = gps_usable / gps_answered if gps_answered else 0.0
            alpha = 0.95 if gps_usable >= MIN_STRUCTURAL_ROWS else 0.35
            size = 34 + 62 * coverage
            ax.scatter(gps, di + 0.22, s=size, marker="s", color="#238B45",
                       alpha=alpha, edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(gps + 0.012, di + 0.22, f"{gps_usable}/{gps_answered}",
                    fontsize=5.6, va="center", color="#238B45", alpha=alpha)
        else:
            ax.text(0.505, di + 0.22, f"GPS n/a ({gps_usable}/{gps_answered})",
                    fontsize=5.6, va="center", color="#777777", style="italic")

    ax.axvline(0.5, color=C_RAND, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(0.20, 0.92)
    ax.set_ylim(len(datasets) - 0.55, -0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets], fontsize=8)
    ax.set_xlabel("AUROC for detecting incorrect answers", fontsize=8)
    ax.set_title("Family-level uncertainty diagnostics", fontsize=10, fontweight="bold", pad=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_VANILLA,
               markeredgecolor="white", markersize=6.0, label="Best vanilla output"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_KGRAG,
               markeredgecolor="white", markersize=6.0, label="Best KG output"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#6A51A3",
               markeredgecolor="white", markersize=6.0, label="KG SEU"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#238B45",
               markeredgecolor="white", markersize=6.0, label="KG GPS (label = usable/answered)"),
        Line2D([0], [0], color=C_RAND, linestyle="--", linewidth=0.9, label="Random"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.31),
              ncol=5, fontsize=6.2, framealpha=0.95, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.88, bottom=0.27)

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"family_auroc_summary.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def make_gps_coverage_diagnostic(results):
    """Show GPS as a conditional diagnostic: coverage and AUROC together."""
    fig, ax = plt.subplots(figsize=(3.45, 3.15))

    for ds in [d for d in DATASETS_ORDER if d in results]:
        gps, usable, answered = _gps_summary_for_dataset(ds, results[ds])
        coverage = usable / answered if answered else 0.0
        label = DATASET_LABELS.get(ds, ds)
        color = DATASET_PALETTE.get(ds, "#777777")
        if gps is None or np.isnan(gps):
            gps_y = 0.5
            marker = "x"
            alpha = 0.55
        else:
            gps_y = gps
            marker = "o"
            alpha = 0.95 if usable >= MIN_STRUCTURAL_ROWS else 0.45
        ax.scatter(coverage, gps_y, s=68, color=color, marker=marker,
                   edgecolor="white" if marker != "x" else color,
                   linewidth=0.7, alpha=alpha, zorder=4)
        dx = 0.018 if coverage < 0.72 else -0.018
        ha = "left" if coverage < 0.72 else "right"
        ax.text(coverage + dx, gps_y, f"{label}\n{usable}/{answered}",
                fontsize=6.0, ha=ha, va="center", color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.82))

    ax.axhline(0.5, color=C_RAND, linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color=C_RAND, linestyle=":", linewidth=0.8)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0.30, 0.72)
    ax.set_xlabel("GPS usable coverage", fontsize=8)
    ax.set_ylabel("GPS AUROC", fontsize=8)
    ax.set_title("GPS is conditional, not always-on", fontsize=9.2, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.17)

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"gps_coverage_diagnostic.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def make_context_collapse_ablation():
    """RealMedQA strict graph-only stress test from per-question metric logs."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    adaptive_path = os.path.join(
        root,
        "results/latest_kg_design_final_metrics/runs/"
        "20260527-163249-realmedqa-n230-full-metrics-evaluation-subset-rebuildkg/"
        "mirage_realmedqa_results.json",
    )
    strict_path = os.path.join(
        root,
        "results/latest_kg_design_final_metrics/runs/"
        "20260528-174538-realmedqa-n230-full-metrics-evaluation-subset/"
        "mirage_realmedqa_results.json",
    )

    with open(adaptive_path) as f:
        adaptive_doc = json.load(f)
    with open(strict_path) as f:
        strict_doc = json.load(f)

    cfgs = {c["config"]["name"]: c for c in adaptive_doc["config_results"]}
    dense_cfg = cfgs["dense_floor_thr0.1_k10_rt0p0"]
    adaptive_cfg = cfgs["kg_entity_first_thr0.1_k10_rt0p0"]
    strict_cfg = strict_doc["config_results"][0]

    policies = [
        ("Dense", dense_cfg["details"], "vanilla", C_VANILLA),
        ("Adaptive KG", adaptive_cfg["details"], "kg", C_KGRAG),
        ("Strict KG", strict_cfg["details"], "kg", "#9B2226"),
    ]
    groups = []
    for label, rows, prefix, color in policies:
        app = [r for r in rows if not r.get(f"{prefix}_generation_failed")]
        groups.append({
            "label": label,
            "color": color,
            "correct_sd": [float(r[f"{prefix}_sd_uq"]) for r in app
                           if r.get(f"{prefix}_correct") and r.get(f"{prefix}_sd_uq") is not None],
            "wrong_sd": [float(r[f"{prefix}_sd_uq"]) for r in app
                         if not r.get(f"{prefix}_correct") and r.get(f"{prefix}_sd_uq") is not None],
            "correct_seu": [float(r[f"{prefix}_support_entailment_uncertainty"]) for r in app
                            if r.get(f"{prefix}_correct") and r.get(f"{prefix}_support_entailment_uncertainty") is not None],
            "wrong_seu": [float(r[f"{prefix}_support_entailment_uncertainty"]) for r in app
                          if not r.get(f"{prefix}_correct") and r.get(f"{prefix}_support_entailment_uncertainty") is not None],
            "n_correct": sum(1 for r in app if r.get(f"{prefix}_correct")),
            "n_wrong": sum(1 for r in app if not r.get(f"{prefix}_correct")),
        })

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.9),
                             gridspec_kw={"wspace": 0.28})
    rng = np.random.default_rng(7)
    positions = np.arange(len(groups))
    offsets = [-0.16, 0.16]
    width = 0.24
    correct_color = "#3B6EA8"
    wrong_color = "#C45A4A"

    def draw_split_boxes(ax, key_correct, key_wrong, transform=lambda x: x,
                         ylabel="", ylim=None, title=""):
        data, pos, colors = [], [], []
        for i, group in enumerate(groups):
            correct_vals = [transform(v) for v in group[key_correct]]
            wrong_vals = [transform(v) for v in group[key_wrong]]
            data.extend([correct_vals, wrong_vals])
            pos.extend([positions[i] + offsets[0], positions[i] + offsets[1]])
            colors.extend([correct_color, wrong_color])
        vp = ax.violinplot(
            data,
            positions=pos,
            widths=width * 1.5,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor("none")
        vp["cmedians"].set_color("#222222")
        vp["cmedians"].set_linewidth(1.0)
        for i, group in enumerate(groups):
            for vals, off, color in [
                ([transform(v) for v in group[key_correct]], offsets[0], correct_color),
                ([transform(v) for v in group[key_wrong]], offsets[1], wrong_color),
            ]:
                if not vals:
                    continue
                sample = vals if len(vals) <= 90 else rng.choice(vals, 90, replace=False)
                jitter = rng.normal(0, 0.022, len(sample))
                ax.scatter(
                    np.full(len(sample), positions[i] + off) + jitter,
                    sample,
                    s=7,
                    color=color,
                    alpha=0.33,
                    linewidths=0,
                    zorder=2,
                )
            ax.text(
                positions[i],
                ax.get_ylim()[0] if ylim is None else ylim[0],
                f"{group['n_correct']}/{group['n_wrong']}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#555555",
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([g["label"] for g in groups], fontsize=7.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.8, fontweight="bold")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    draw_split_boxes(
        axes[0],
        "correct_sd",
        "wrong_sd",
        transform=lambda x: np.log10(max(float(x), 1e-12)),
        ylabel=r"$\log_{10}(\mathrm{SD\!-\!UQ}+10^{-12})$",
        ylim=(-12.9, -0.25),
        title="Answer-state dispersion (SD-UQ)",
    )
    axes[0].annotate(
        "wrong answers\ncollapse to zero",
        xy=(positions[2] + offsets[1], -12.0),
        xytext=(1.45, -9.1),
        arrowprops=dict(arrowstyle="->", lw=0.75, color=wrong_color),
        fontsize=6.2,
        color=wrong_color,
        ha="center",
    )
    # per-policy floor share among wrong answers + AUROC, printed in-panel
    for i, group in enumerate(groups):
        floor_share = (np.mean([v <= 1e-9 for v in group["wrong_sd"]])
                       if group["wrong_sd"] else float("nan"))
        sd_scores = group["correct_sd"] + group["wrong_sd"]
        sd_labels = [1] * len(group["correct_sd"]) + [0] * len(group["wrong_sd"])
        auc = safe_auroc(sd_scores, sd_labels)
        axes[0].text(positions[i], -0.40,
                     f"AUROC {auc:.2f}\n{floor_share:.0%} of wrong at floor",
                     ha="center", va="top", fontsize=5.4, color="#333333")

    draw_split_boxes(
        axes[1],
        "correct_seu",
        "wrong_seu",
        ylabel="SEU uncertainty",
        ylim=(-0.03, 1.16),
        title="Evidence-state support (SEU)",
    )
    axes[1].annotate(
        "SEU stays defined\n(partly neutral default;\nsee caption)",
        xy=(positions[2] + offsets[1], 0.50),
        xytext=(1.28, 0.86),
        arrowprops=dict(arrowstyle="->", lw=0.75, color="#6A51A3"),
        fontsize=6.0,
        color="#6A51A3",
        ha="center",
    )
    for i, group in enumerate(groups):
        seu_scores = group["correct_seu"] + group["wrong_seu"]
        seu_labels = [1] * len(group["correct_seu"]) + [0] * len(group["wrong_seu"])
        auc = safe_auroc(seu_scores, seu_labels)
        axes[1].text(positions[i], 1.13, f"AUROC {auc:.2f}",
                     ha="center", va="top", fontsize=5.4, color="#333333")

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=correct_color,
               markeredgecolor="white", markersize=6, label="Correct"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=wrong_color,
               markeredgecolor="white", markersize=6, label="Wrong"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, fontsize=7, framealpha=0.95, edgecolor="#cccccc")
    fig.text(
        0.5,
        0.01,
        "Numbers beneath each policy give correct/wrong answered counts.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.19)

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"context_collapse_ablation.{ext}")
        fig.savefig(out, dpi=240 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── Legacy stratified AUROC helper  (not used in the current main paper) ─────

STRAT_METRICS = [
    ("kg_sre_uq",                          "SRE-UQ"),
    ("kg_vn_entropy",                       "VN-Ent."),
    ("kg_sd_uq",                            "SD-UQ"),
]
STRAT_DATASETS = ["2Wikimultihopqa", "Musique"]


def make_stratified(results):
    nd = len(STRAT_DATASETS)
    fig, axes = plt.subplots(1, nd, figsize=(7.0, 3.15), sharey=True,
                             gridspec_kw={"wspace": 0.14})
    if nd == 1:
        axes = [axes]

    stratum_styles = [
        ("All",    C_ALL,   "//",  1.0),
        ("High-$g$", C_HIGHG, "\\\\", 1.0),
        ("Low-$g$",  C_LOWG,  "xx",   1.0),
    ]

    for ax, ds in zip(axes, STRAT_DATASETS):
        rows   = [r for r in results.get(ds, []) if not r.get("kg_generation_failed")]
        if any("kg_graph_path_support_null_reason" in r for r in rows):
            high_g = [r for r in rows if not str(r.get("kg_graph_path_support_null_reason", ""))]
            low_g  = [r for r in rows if str(r.get("kg_graph_path_support_null_reason", ""))]
        else:
            high_g = [r for r in rows if r.get("kg_graph_path_support") != 0.5]
            low_g  = [r for r in rows if r.get("kg_graph_path_support") == 0.5]
        groups = [rows, high_g, low_g]

        n_m, width = len(STRAT_METRICS), 0.20
        y = np.arange(n_m)
        group_vals = []

        for gi, ((glabel, color, hatch, alpha), grows) in enumerate(
                zip(stratum_styles, groups)):
            vals = []
            for mkey, _ in STRAT_METRICS:
                metric_rows = valid_metric_rows(grows, mkey)
                a = safe_auroc(
                    [r.get(mkey) for r in metric_rows],
                    [int(r.get("kg_correct", 0)) for r in metric_rows],
                )
                vals.append(a)
            group_vals.append(vals)
            offset = (gi - 1) * width
            bar_widths = [v if v is not None else 0.0 for v in vals]
            bars = ax.barh(y + offset, bar_widths, height=width * 0.88,
                          color=color, hatch=hatch, alpha=alpha,
                          edgecolor="white", linewidth=0.6,
                          label=glabel, zorder=3)
            for bar, v in zip(bars, vals):
                if v is not None and v > 0.02:
                    ax.text(v + 0.018,
                            bar.get_y() + bar.get_height() / 2,
                            f"{v:.2f}",
                            ha="left", va="center", fontsize=6.2,
                            fontweight="bold")

        ax.axvline(0.5, color=C_RAND, linestyle="--", linewidth=0.9,
                   zorder=2, label="Random (0.5)")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.45, n_m - 0.45)
        ax.set_yticks(y)
        ax.set_yticklabels([m for _, m in STRAT_METRICS], fontsize=7.6)
        ax.set_xlabel("AUROC", fontsize=8)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10, fontweight="bold", pad=20)
        n_hg = sum(1 for r in high_g if r.get("kg_correct"))
        n_lg = sum(1 for r in low_g  if r.get("kg_correct"))
        ax.invert_yaxis()

        for mi in range(n_m):
            if all(vals[mi] is None for vals in group_vals):
                ax.text(0.05, y[mi], "n/a", ha="left", va="center",
                        fontsize=6.0, color="#777777", style="italic")

        ax.text(
            0.5, 1.03,
            f"Grounded n={len(high_g)} (corr={n_hg})   |   Abstained n={len(low_g)} (corr={n_lg})",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.3,
            color="#555555",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    axes[0].set_ylabel("Metric", fontsize=8)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    handles = [
        mpatches.Patch(facecolor=C_ALL,   hatch="//",  label="All queries",   edgecolor="white"),
        mpatches.Patch(facecolor=C_HIGHG, hatch="\\\\", label=r"Grounded GPS", edgecolor="white"),
        mpatches.Patch(facecolor=C_LOWG,  hatch="xx",  label=r"GPS abstained", edgecolor="white"),
        Line2D([0],[0], color=C_RAND, linestyle="--", linewidth=0.9, label="Random (0.5)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               fontsize=6.5, ncol=4, framealpha=0.9, edgecolor="#cccccc")
    fig.subplots_adjust(top=0.78, left=0.12, right=0.995, bottom=0.18)

    out = os.path.join(os.path.dirname(__file__), "stratified_auroc.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Figure 3: Calibration paradox  (figure — single column, 3.3 in) ──────────

BEST_GENERATIVE = "sre_uq"


def make_calibration_paradox(results):
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    label_offsets = {
        "Pubmedqa": (8, 6),
        "Realmedqa": (8, -10),
        "Hotpotqa": (-38, -4),
        "HotpotqaFullWiki": (-46, 10),
        "2Wikimultihopqa": (-24, 12),
        "Musique": (-26, 10),
    }

    deltas = []
    for ds, rows in results.items():
        color = DATASET_PALETTE.get(ds, "#888888")
        v_rows = [r for r in rows if not r.get("vanilla_generation_failed")]
        k_rows = [r for r in rows if not r.get("kg_generation_failed")]
        v_labels = [int(r.get("vanilla_correct", 0)) for r in v_rows]
        k_labels = [int(r.get("kg_correct",      0)) for r in k_rows]
        v_acc    = float(np.mean(v_labels)) if v_labels else np.nan
        k_acc    = float(np.mean(k_labels)) if k_labels else np.nan
        v_auroc  = safe_auroc([r.get(f"vanilla_{BEST_GENERATIVE}") for r in v_rows], v_labels)
        k_auroc  = safe_auroc([r.get(f"kg_{BEST_GENERATIVE}")      for r in k_rows], k_labels)
        if v_auroc is None or k_auroc is None:
            continue

        delta_acc = k_acc - v_acc
        delta_auroc = k_auroc - v_auroc
        deltas.append((delta_acc, delta_auroc))
        label = DATASET_LABELS.get(ds, ds)
        dx, dy = label_offsets.get(ds, (8, 6))
        ax.scatter(delta_acc, delta_auroc, marker="o", color=color, s=62,
                   zorder=4, linewidths=0.9, edgecolors="white")
        txt = ax.annotate(
            label,
            xy=(delta_acc, delta_auroc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            color=color,
            va="center",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.14",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])

    ax.axhline(0.0, color=C_RAND, linestyle="--", linewidth=0.9, zorder=1)
    ax.axvline(0.0, color=C_RAND, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlabel(r"$\Delta$ Accuracy (KG - Vanilla)", fontsize=8)
    ax.set_ylabel(r"$\Delta$ AUROC (KG - Vanilla)", fontsize=8)
    if deltas:
        xs, ys = zip(*deltas)
        x_lo, x_hi = min(min(xs), 0.0), max(max(xs), 0.0)
        y_lo, y_hi = min(min(ys), 0.0), max(max(ys), 0.0)
        x_pad = max(0.012, 0.18 * max(x_hi - x_lo, 1e-6))
        y_pad = max(0.025, 0.16 * max(y_hi - y_lo, 1e-6))
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:+.0f}%"))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))
    ax.set_title("Change from vanilla to KG-RAG",
                 fontsize=8.1, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(os.path.dirname(__file__), "calibration_paradox.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── Figure 4: Selective-prediction curves  (figure* — full width, 7 in) ──────

CAC_DATASETS = ["Realmedqa", "2Wikimultihopqa", "Hotpotqa", "Musique"]
CAC_FAMILY_METRICS = [
    ("kg_sd_uq", "SD-UQ (answer-state)", C_KGRAG, "-"),
    ("kg_support_entailment_uncertainty", "SEU (evidence-state)", "#6A51A3", "-"),
    ("kg_graph_path_support", "GPS (retrieval-state)", "#238B45", "-"),
    ("__combined_audit__", "Combined audit", "#E69F00", "--"),
]


def _coverage_accuracy(scores, labels):
    """Sort by increasing uncertainty (→ decreasing confidence), accumulate."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    coverages, accs = [], []
    correct = 0
    for i, (_, lbl) in enumerate(pairs):
        correct += lbl
        coverages.append((i + 1) / len(pairs))
        accs.append(correct / (i + 1))
    return np.array(coverages), np.array(accs)


def _percentile_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    if len(values) == 1:
        return ranks
    for rank, idx in enumerate(order):
        ranks[idx] = rank / (len(values) - 1)
    return ranks


def _gps_risk(row):
    value = row.get("kg_graph_path_support")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if abs(value - 0.5) < 1e-12:
        return 1.0
    return value


def _combined_audit_pairs(rows):
    usable = [r for r in rows if not r.get("kg_generation_failed")]
    if len(usable) < 12:
        return []
    sd = [
        float(r.get("kg_sd_uq", 0.0))
        if r.get("kg_sd_uq") is not None else 0.0
        for r in usable
    ]
    seu = [
        float(r.get("kg_support_entailment_uncertainty", 0.5))
        if r.get("kg_support_entailment_uncertainty") is not None else 0.5
        for r in usable
    ]
    gps = [_gps_risk(r) for r in usable]
    sd_r, seu_r, gps_r = _percentile_ranks(sd), _percentile_ranks(seu), _percentile_ranks(gps)
    scores = [(a + b + c) / 3.0 for a, b, c in zip(sd_r, seu_r, gps_r)]
    labels = [int(r.get("kg_correct", 0)) for r in usable]
    return list(zip(scores, labels))


def make_gps_abstention_map(results):
    """Stacked denominator map for the conditional GPS structural diagnostic."""
    datasets = [d for d in DATASETS_ORDER if d in results]
    labels = [DATASET_LABELS.get(ds, ds) for ds in datasets]
    usable, binary_unavailable, graph_unavailable = [], [], []

    for ds in datasets:
        _, gps_usable, answered = _gps_summary_for_dataset(ds, results[ds])
        unavailable = max(0, answered - gps_usable)
        usable.append(gps_usable)
        if ds == "Pubmedqa":
            binary_unavailable.append(unavailable)
            graph_unavailable.append(0)
        else:
            binary_unavailable.append(0)
            graph_unavailable.append(unavailable)

    y = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.barh(y, usable, color="#238B45", label="GPS usable")
    ax.barh(
        y,
        graph_unavailable,
        left=usable,
        color="#D6604D",
        label="Entity/path denominator unavailable",
    )
    left = np.array(usable) + np.array(graph_unavailable)
    ax.barh(
        y,
        binary_unavailable,
        left=left,
        color="#8073AC",
        label="Binary answer format",
    )

    for idx, ds in enumerate(datasets):
        total = usable[idx] + graph_unavailable[idx] + binary_unavailable[idx]
        ax.text(
            max(total, 1) + 3,
            idx,
            f"{usable[idx]}/{total}",
            va="center",
            ha="left",
            fontsize=6.4,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Answered KG rows")
    ax.set_title("GPS denominator map", fontsize=9.2, fontweight="bold")
    ax.set_xlim(0, max(np.array(usable) + np.array(graph_unavailable) + np.array(binary_unavailable)) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=1,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=6.1,
    )
    fig.subplots_adjust(left=0.31, right=0.96, top=0.88, bottom=0.31)

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"gps_abstention_map.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def make_adaptive_kg_auroc_heatmap(results):
    """Adaptive KG-only AUROC heatmap, grouped by diagnostic family."""
    metrics = [
        ("kg_discrete_semantic_entropy", "DSE"),
        ("kg_sre_uq", "SRE-UQ"),
        ("kg_vn_entropy", "VN-Ent."),
        ("kg_sd_uq", "SD-UQ"),
        ("kg_support_entailment_uncertainty", "SEU"),
        ("__gps_conditional__", "GPS"),
        ("__combined_audit__", "Combined"),
    ]
    datasets = [d for d in DATASETS_ORDER if d in results]
    matrix = np.full((len(datasets), len(metrics)), np.nan)
    annotations = [["" for _ in metrics] for _ in datasets]

    for di, ds in enumerate(datasets):
        rows = [r for r in results[ds] if not r.get("kg_generation_failed")]
        labels = [int(r.get("kg_correct", 0)) for r in rows]
        for mi, (key, _) in enumerate(metrics):
            if key == "__gps_conditional__":
                value, gps_usable, answered = _gps_summary_for_dataset(ds, results[ds])
                if value is None:
                    annotations[di][mi] = "n/a"
                    continue
                annotations[di][mi] = f"{value:.2f}"
            elif key == "__combined_audit__":
                pairs = _combined_audit_pairs(rows)
                if pairs:
                    scores, pair_labels = zip(*pairs)
                    value = safe_auroc(list(scores), list(pair_labels))
                else:
                    value = None
                annotations[di][mi] = f"{value if value is not None else 0.5:.2f}"
            else:
                value = safe_auroc([r.get(key) for r in rows], labels)
                annotations[di][mi] = f"{value if value is not None else 0.5:.2f}"
            matrix[di, mi] = 0.5 if value is None else float(value)

    fig, ax = plt.subplots(figsize=(7.05, 3.15))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=HMAP_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([lab for _, lab in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets])

    for di in range(len(datasets)):
        for mi, (key, _) in enumerate(metrics):
            v = matrix[di, mi]
            if np.isnan(v):
                ax.text(mi, di, annotations[di][mi], ha="center", va="center",
                        fontsize=6.4, color="#888888")
                continue
            text_color = "white" if v < 0.40 or v > 0.76 else "#222222"
            ax.text(mi, di, annotations[di][mi], ha="center", va="center",
                    fontsize=6.6, color=text_color)

    # family separators and headers; the Combined column gets a heavier rule
    for x in (3.5, 4.5):
        ax.axvline(x, color="white", linewidth=3.0)
        ax.axvline(x, color="#555555", linewidth=0.8)
    ax.axvline(5.5, color="white", linewidth=5.0)
    ax.axvline(5.5, color="#222222", linewidth=1.6)
    headers = [(1.5, "Answer-state"), (4.0, "Evidence"),
               (5.0, "Retrieval"), (6.0, "Composite")]
    for x, lab in headers:
        ax.text(x, -0.78, lab, ha="center", va="bottom", fontsize=7.4,
                fontweight="bold", color="#333333")

    ax.set_title("Adaptive KG diagnostic AUROC", fontsize=10,
                 fontweight="bold", pad=22)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.grid(which="major", visible=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("AUROC")
    fig.subplots_adjust(left=0.14, right=0.94, top=0.84, bottom=0.24)

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"adaptive_kg_auroc_heatmap.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def _strict_realmedqa_rows():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    path = os.path.join(
        root,
        "results/latest_kg_design_final_metrics/runs/"
        "20260528-174538-realmedqa-n230-full-metrics-evaluation-subset/"
        "mirage_realmedqa_results.json")
    with open(path) as f:
        doc = json.load(f)
    return doc["config_results"][0]["details"]


def make_coverage_accuracy(results):
    datasets = [d for d in CAC_DATASETS if d in results]
    panels = [(ds, [r for r in results[ds]
                    if not r.get("kg_generation_failed")], False)
              for ds in datasets]
    panels.append(("RealMedQA strict",
                   [r for r in _strict_realmedqa_rows()
                    if not r.get("kg_generation_failed")], True))
    nd = len(panels)

    fig, axes = plt.subplots(1, nd, figsize=(7.05, 2.5), sharey=False,
                              gridspec_kw={"wspace": 0.34})
    if nd == 1:
        axes = [axes]

    for ai, (ax, (ds, rows, is_strict)) in enumerate(zip(axes, panels)):
        baseline_labels = [int(r.get("kg_correct", 0)) for r in rows]
        if baseline_labels:
            ax.axhline(np.mean(baseline_labels), color="#444444", lw=0.7,
                       linestyle=":", alpha=0.65)

        for metric_key, label, color, linestyle in CAC_FAMILY_METRICS:
            if is_strict and metric_key in ("kg_graph_path_support",
                                            "__combined_audit__"):
                continue   # GPS replay values are not row-level in strict logs
            if metric_key == "__combined_audit__":
                pairs = _combined_audit_pairs(rows)
            else:
                metric_rows = valid_metric_rows(rows, metric_key)
                pairs = [
                    (r.get(metric_key), int(r.get("kg_correct", 0)))
                    for r in metric_rows
                    if r.get(metric_key) is not None
                ]
            if len(pairs) < 12 or len({lbl for _, lbl in pairs}) < 2:
                continue
            scores, labels = zip(*pairs)
            cov, acc = _coverage_accuracy(list(scores), list(labels))
            lw = 1.75 if metric_key == "__combined_audit__" else 1.45
            ax.plot(cov, acc, color=color, linestyle=linestyle, lw=lw,
                    label=label)

        ax.axvspan(0.0, 0.2, color="#999999", alpha=0.12, lw=0)
        ax.set_xlim(0.0, 1.04); ax.set_ylim(0.0, 1.04)
        ax.set_xlabel(r"Accepted coverage $\pi$", fontsize=7.5)
        if ai == 0:
            ax.set_ylabel("Accuracy after abstention", fontsize=7)
        else:
            ax.set_ylabel("")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=8.5, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    handles = [
        Line2D([0], [0], color=C_KGRAG, linewidth=1.45, label="SD-UQ (answer-state)"),
        Line2D([0], [0], color="#6A51A3", linewidth=1.45, label="SEU (evidence-state)"),
        Line2D([0], [0], color="#238B45", linewidth=1.45, label="GPS (retrieval-state, where defined)"),
        Line2D([0], [0], color="#E69F00", linestyle="--", linewidth=1.75, label="Combined audit"),
        Line2D([0], [0], color="#444444", linestyle=":", linewidth=0.8, label="No abstention"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=5, fontsize=6.2, framealpha=0.92, edgecolor="#cccccc")
    fig.subplots_adjust(top=0.78, left=0.08, right=0.995, bottom=0.20)

    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__), f"coverage_accuracy.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def make_routing_distribution(results):
    """Optional route-distribution chart, emitted only for reruns with route logs."""
    datasets, entity_first, dense_fallback, other = [], [], [], []
    for ds in DATASETS_ORDER:
        rows = [r for r in results.get(ds, []) if not r.get("kg_generation_failed")]
        if not rows or not any(str(r.get("kg_retrieval_route", "")).strip() for r in rows):
            continue
        n = len(rows)
        routes = [str(r.get("kg_retrieval_route") or "unknown") for r in rows]
        ef = sum(1 for route in routes if route == "entity_first") / n
        dense = sum(1 for route in routes if route in {"semantic_only", "vector_only", "dense_fallback"}) / n
        oth = max(0.0, 1.0 - ef - dense)
        datasets.append(ds)
        entity_first.append(ef)
        dense_fallback.append(dense)
        other.append(oth)

    if not datasets:
        print("Skipped routing_distribution.pdf (no kg_retrieval_route fields in loaded artifacts)")
        return

    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(x, entity_first, color=C_KGRAG, label="Entity-first")
    ax.bar(x, dense_fallback, bottom=entity_first, color=C_VANILLA, label="Dense fallback")
    bottom = np.array(entity_first) + np.array(dense_fallback)
    ax.bar(x, other, bottom=bottom, color="#999999", label="Other graph route")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of KG queries")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=20, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("KG Routing Distribution", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9, edgecolor="#cccccc")

    out = os.path.join(os.path.dirname(__file__), "routing_distribution.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)




def make_family_disagreement_scatter(results):
    """Pooled adaptive-KG SD-UQ vs SEU scatter with quadrant statistics.

    Hollow markers mark all-neutral SEU=0.5 rows (the NLI said nothing);
    quadrant annotations give n and error rate among non-neutral rows.
    """
    xs, ys, correct = [], [], []
    for ds in DATASETS_ORDER:
        if ds not in results:
            continue
        for r in results[ds]:
            if r.get("kg_generation_failed"):
                continue
            sd = r.get("kg_sd_uq")
            seu = r.get("kg_support_entailment_uncertainty")
            if sd is None or seu is None:
                continue
            xs.append(np.log10(max(float(sd), 1e-12)))
            ys.append(float(seu))
            correct.append(bool(r.get("kg_correct")))
    xs = np.array(xs); ys = np.array(ys); correct = np.array(correct)
    neutral = np.isclose(ys, 0.5)
    x_med = float(np.median(xs))

    fig, ax = plt.subplots(figsize=(7.05, 3.4))
    c_ok, c_bad = "#3B6EA8", "#C45A4A"
    rng = np.random.default_rng(11)
    jitter = rng.normal(0, 0.012, len(ys))
    nn = ~neutral
    hb = ax.hexbin(xs[nn], ys[nn] + jitter[nn], gridsize=(38, 22),
                   cmap="Greys", mincnt=1, linewidths=0, zorder=1,
                   vmin=0, vmax=10, alpha=0.8)
    wrong_nn = (~correct) & nn
    ax.scatter(xs[wrong_nn], ys[wrong_nn] + jitter[wrong_nn], s=12,
               color=c_bad, alpha=0.65, linewidths=0,
               label="Wrong (non-neutral SEU)", zorder=3)
    ax.axvline(x_med, color="#555555", linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")

    def qstats(qx, qy):
        if qx == "lo":
            mx = xs <= x_med
        else:
            mx = xs > x_med
        if qy == "hi":
            my = ys > 0.5
        else:
            my = ys < 0.5
        m = mx & my
        n = int(m.sum())
        err = float((~correct[m]).mean()) if n else float("nan")
        return n, err

    annots = [
        ("lo", "hi", 0.115, 0.97, "calm + contradicted"),
        ("lo", "lo", 0.115, 0.06, "calm + supported"),
    ]
    for qx, qy, ax_x, ax_y, name in annots:
        n, err = qstats(qx, qy)
        ax.text(ax_x, ax_y, f"{name}\nn={n}, {err:.0%} wrong",
                transform=ax.transAxes, fontsize=6.8, color="#222222",
                ha="left" if ax_x < 0.5 else "right",
                va="top" if ax_y > 0.5 else "bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb",
                          lw=0.5, alpha=0.85))

    ax.set_xlabel(r"$\log_{10}(\mathrm{SD\!-\!UQ}+10^{-12})$")
    ax.set_ylabel("SEU uncertainty")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Family disagreement, pooled adaptive KG", fontsize=9.5,
                 fontweight="bold")
    handles = [
        plt.matplotlib.patches.Patch(color="#bbbbbb",
                                     label="all answers (density)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_bad,
               markersize=5, label="wrong answers"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=2,
               fontsize=6.6, framealpha=0.92, edgecolor="#cccccc")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.16)
    for ext in ("pdf", "png"):
        out = os.path.join(os.path.dirname(__file__),
                           f"family_disagreement_scatter.{ext}")
        fig.savefig(out, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = load_results()
    print("Loaded datasets:", list(results.keys()))
    make_answer_heatmap(results)
    make_structural_heatmap(results)
    make_context_collapse_ablation()
    make_calibration_paradox(results)
    make_coverage_accuracy(results)
    make_gps_abstention_map(results)
    make_adaptive_kg_auroc_heatmap(results)
    make_family_disagreement_scatter(results)
    make_routing_distribution(results)
    print("All figures saved.")
