"""Wave-1 log-only analyses (no generation reruns).

Emits results/analyses/wave1_analysis.json with:
  V1.2  silent-error threshold sensitivity: silent rate among wrong answers
        under progressively relaxed calmness definitions, per policy (pooled).
  V1.4  learned certificate: leave-one-dataset-out logistic regression on
        (sd_rank, seu, gps, gps_abstain) vs the hard conjunctive certificate,
        coverage compared at matched train precision.
  T7    Wilson 95% CIs for the silent-failure rates (per dataset x policy).
  T16   SEU-gated selective risk at 80% coverage on the adaptive runs, with
        randomised tie-breaking over the SEU=0.5 plateau (mean of 200 draws).
  T6b   observed error shares for the 2x2 taxonomy cells (proxy
        operationalisation, per policy, pooled).
  T8    wrong-answer counts per hop slice for the 2Wiki hopwise table.
  V1.3  per-metric average compute times aggregated across the six headline
        runs (KG side), mapped to families.
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.trust_analysis import (rows_for, gps_scores, REPO, MANIFEST,  # noqa: E402
                                        KEY, STRICT_RUNS, SD_FLOOR)
from experiments.gps_v2_paper_numbers import percentile_ranks  # noqa: E402

RNG = np.random.default_rng(42)


# ── helpers ──────────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def all_rows(policy):
    """(slug, row) pairs for every dataset under a policy."""
    out = []
    for slug in KEY:
        if policy == "strict" and slug not in STRICT_RUNS:
            continue
        for r in rows_for(slug, policy):
            out.append((slug, r))
    return out


# ── V1.2: threshold sensitivity ──────────────────────────────────────────────

def threshold_sensitivity():
    """Silent rate among wrong answers under nested relaxations of 'calm'.

    The SD-UQ floor is a jittered point mass (~1e-12), so percentile cutoffs
    are ill-posed; the ladder instead relaxes in terms of sample agreement,
    which is the quantity the answer-state family actually observes.
    With N=5 samples DSE is quantised: DSE=0 means all five samples agree;
    DSE<=0.51 admits at most one dissenting sample (4-of-5 agreement,
    DSE=0.5004)."""
    defs = {}
    policies = ("vanilla", "kg", "strict")
    rows_by_policy = {p: all_rows(p) for p in policies}
    conditions = [
        ("paper definition: DSE=0 and SD-UQ at floor",
         lambda r: r["dse"] is not None and abs(r["dse"]) < 1e-9
         and r["sd"] is not None and r["sd"] <= SD_FLOOR),
        ("unanimous: DSE=0, any SD-UQ",
         lambda r: r["dse"] is not None and abs(r["dse"]) < 1e-9),
        ("near-unanimous: <=1 dissenting sample (DSE<=0.51)",
         lambda r: r["dse"] is not None and r["dse"] <= 0.51),
    ]
    for name, cond in conditions:
        defs[name] = {}
        for p in policies:
            wrong = [r for _, r in rows_by_policy[p] if not r["correct"]]
            calm = [1 for r in wrong if cond(r)]
            defs[name][p] = {"rate": (sum(calm) / len(wrong)) if wrong else None,
                             "n_wrong": len(wrong)}
    return defs


# ── T7: Wilson CIs for the paper-definition silent rates ─────────────────────

def silent_cis():
    out = {}
    for policy in ("vanilla", "kg", "strict"):
        for slug in KEY:
            if policy == "strict" and slug not in STRICT_RUNS:
                continue
            rows = rows_for(slug, policy)
            wrong = [r for r in rows if not r["correct"]]
            calm = [r for r in wrong
                    if r["dse"] is not None and abs(r["dse"]) < 1e-9
                    and r["sd"] is not None and r["sd"] <= SD_FLOOR]
            out.setdefault(slug, {})[policy] = {
                "k": len(calm), "n": len(wrong),
                "rate": len(calm) / len(wrong) if wrong else None,
                "ci95": wilson_ci(len(calm), len(wrong))}
    # pooled
    out["pooled"] = {}
    for policy in ("vanilla", "kg", "strict"):
        k = sum(out[s][policy]["k"] for s in KEY if policy in out.get(s, {}))
        n = sum(out[s][policy]["n"] for s in KEY if policy in out.get(s, {}))
        out["pooled"][policy] = {"k": k, "n": n, "rate": k / n if n else None,
                                 "ci95": wilson_ci(k, n)}
    return out


# ── T16: SEU-gated selective risk at 80% coverage (adaptive) ────────────────

def seu_at_coverage(draws=200, cov=0.8):
    out = {}
    for slug in KEY:
        rows = [r for r in rows_for(slug, "kg") if r["seu"] is not None]
        if len(rows) < 20:
            continue
        seu = np.array([float(r["seu"]) for r in rows])
        corr = np.array([r["correct"] for r in rows], float)
        n = len(rows)
        k = max(1, int(round(cov * n)))
        errs = []
        for _ in range(draws):
            jitter = RNG.random(n) * 1e-9
            order = np.argsort(seu + jitter)
            errs.append(1.0 - corr[order][:k].mean())
        tie_mass = float((seu == np.median(seu)).mean())
        out[slug] = {"n": n, "err_at_80": float(np.mean(errs)),
                     "err_sd_over_draws": float(np.std(errs)),
                     "frac_at_modal_seu": float((seu == 0.5).mean())}
    return out


# ── T6b: taxonomy cell prevalence (proxy operationalisation) ─────────────────

def taxonomy_prevalence():
    """Among answered rows per policy (pooled):
       calm  = DSE=0 & SD at floor (high answer agreement)
       For wrong answers: calm-wrong = lock-in cell candidates;
         flagged = SEU>0.5 or GPS>0.5 or GPS abstained (discordant signal).
       For correct answers: certificate-pass share (trustworthy cell)."""
    out = {}
    gps_cache = {}
    for policy in ("vanilla", "kg", "strict"):
        tot = wrong = calm_wrong = calm_wrong_flagged = 0
        noisy_wrong = 0
        correct_cert = correct_n = 0
        for slug in KEY:
            if policy == "strict" and slug not in STRICT_RUNS:
                continue
            rows = rows_for(slug, policy)
            if policy != "vanilla":
                key = (slug, policy)
                if key not in gps_cache:
                    try:
                        gps_cache[key] = gps_scores(slug, strict=(policy == "strict"))
                    except Exception:
                        gps_cache[key] = {}
                gps = gps_cache[key]
            else:
                gps = {}
            sd_vals = [r["sd"] for r in rows if r["sd"] is not None]
            sd_med = float(np.median(sd_vals)) if sd_vals else None
            for r in rows:
                tot += 1
                calm = (r["dse"] is not None and abs(r["dse"]) < 1e-9
                        and r["sd"] is not None and r["sd"] <= SD_FLOOR)
                g = gps.get(r["qid"])
                flagged = ((r["seu"] is not None and r["seu"] > 0.5)
                           or (policy != "vanilla" and (g is None or g > 0.5)))
                if not r["correct"]:
                    wrong += 1
                    if calm:
                        calm_wrong += 1
                        if flagged:
                            calm_wrong_flagged += 1
                    else:
                        noisy_wrong += 1
                else:
                    correct_n += 1
                    cert = (sd_med is not None and r["sd"] is not None
                            and r["sd"] <= sd_med
                            and r["seu"] is not None and r["seu"] <= 0.5
                            and (policy == "vanilla" or (g is not None and g <= 0.5)))
                    if cert:
                        correct_cert += 1
        out[policy] = {
            "answered": tot, "wrong": wrong,
            "calm_wrong": calm_wrong,
            "calm_wrong_share_of_errors": calm_wrong / wrong if wrong else None,
            "calm_wrong_flagged_by_other_families":
                calm_wrong_flagged / calm_wrong if calm_wrong else None,
            "noisy_wrong_share_of_errors": noisy_wrong / wrong if wrong else None,
            "correct_passing_certificate_share":
                correct_cert / correct_n if correct_n else None,
        }
    return out


# ── V1.4: learned certificate (leave-one-dataset-out) ───────────────────────

def hard_certificate_mask(slug, rows, gps):
    sd_vals = [r["sd"] for r in rows if r["sd"] is not None]
    sd_med = float(np.median(sd_vals))
    mask = []
    for r in rows:
        g = gps.get(r["qid"])
        mask.append(r["sd"] is not None and r["sd"] <= sd_med
                    and r["seu"] is not None and r["seu"] <= 0.5
                    and g is not None and g <= 0.5)
    return np.array(mask)


def learned_certificate():
    data = {}
    for slug in KEY:
        rows = [r for r in rows_for(slug, "kg")
                if r["sd"] is not None and r["seu"] is not None]
        gps = gps_scores(slug)
        sd_rank = percentile_ranks([float(r["sd"]) for r in rows])
        feats, labels = [], []
        for r, sr in zip(rows, sd_rank):
            g = gps.get(r["qid"])
            feats.append([sr, float(r["seu"]),
                          0.5 if g is None else float(g),
                          1.0 if g is None else 0.0])
            labels.append(int(r["correct"]))
        data[slug] = {"X": np.array(feats), "y": np.array(labels),
                      "hard": hard_certificate_mask(slug, rows, gps)}

    per_ds, pooled = {}, {"hard": [0, 0], "learned": [0, 0], "n": 0}
    for held in KEY:
        train_X = np.vstack([data[s]["X"] for s in KEY if s != held])
        train_y = np.concatenate([data[s]["y"] for s in KEY if s != held])
        clf = LogisticRegression(max_iter=1000).fit(train_X, train_y)
        # threshold chosen on TRAIN to match the hard certificate's train precision
        train_hard = np.concatenate([data[s]["hard"] for s in KEY if s != held])
        hard_prec_train = train_y[train_hard].mean()
        p_train = clf.predict_proba(train_X)[:, 1]
        ths = np.unique(p_train)[::-1]
        thr = None
        for t in ths:
            sel = p_train >= t
            if sel.sum() >= 10 and train_y[sel].mean() >= hard_prec_train:
                thr = t
        if thr is None:
            thr = float(np.max(p_train))
        X, y, hard = data[held]["X"], data[held]["y"], data[held]["hard"]
        p = clf.predict_proba(X)[:, 1]
        sel = p >= thr
        per_ds[held] = {
            "n": int(len(y)),
            "base_acc": float(y.mean()),
            "hard_cov": float(hard.mean()),
            "hard_prec": float(y[hard].mean()) if hard.sum() else None,
            "learned_cov": float(sel.mean()),
            "learned_prec": float(y[sel].mean()) if sel.sum() else None,
            "auroc_p_correct": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        }
        pooled["hard"][0] += int(y[hard].sum()); pooled["hard"][1] += int(hard.sum())
        pooled["learned"][0] += int(y[sel].sum()); pooled["learned"][1] += int(sel.sum())
        pooled["n"] += len(y)
    summary = {
        "per_dataset": per_ds,
        "pooled": {
            "hard": {"coverage": pooled["hard"][1] / pooled["n"],
                     "precision": pooled["hard"][0] / pooled["hard"][1]},
            "learned": {"coverage": pooled["learned"][1] / pooled["n"],
                        "precision": pooled["learned"][0] / pooled["learned"][1]},
            "n": pooled["n"],
        },
    }
    return summary


# ── T8: hopwise wrong counts ─────────────────────────────────────────────────

def hopwise_wrong_counts():
    path = os.path.join(REPO, "results", "analyses", "current_hopwise_2wiki.json")
    art = json.load(open(path))
    out = []
    for row in art:
        if not isinstance(row, dict):
            continue
        n = row.get("n"); acc = row.get("accuracy", row.get("acc"))
        rec = {k: row.get(k) for k in ("policy", "hop", "n", "accuracy", "acc")}
        if n is not None and acc is not None:
            rec["n_wrong"] = int(round(n * (1 - acc)))
        out.append(rec)
    return out


# ── V1.3: compute-time aggregation ───────────────────────────────────────────

def compute_times():
    agg = {}
    for slug in KEY:
        doc = json.load(open(os.path.join(REPO, MANIFEST[KEY[slug]]["result_path"])))
        cfg = next(c for c in doc["config_results"]
                   if c["config"]["name"].startswith("kg_entity_first"))
        for k, v in (cfg.get("kg_avg_compute_times") or {}).items():
            agg.setdefault(k, []).append(float(v))
    return {k: {"mean_s": float(np.mean(v)), "min_s": float(np.min(v)),
                "max_s": float(np.max(v)), "n_runs": len(v)}
            for k, v in agg.items()}


def main():
    out = {
        "threshold_sensitivity": threshold_sensitivity(),
        "silent_wilson_cis": silent_cis(),
        "seu_at_80_adaptive": seu_at_coverage(),
        "taxonomy_prevalence": taxonomy_prevalence(),
        "learned_certificate": learned_certificate(),
        "hopwise_wrong_counts": hopwise_wrong_counts(),
        "compute_times_kg_side": compute_times(),
    }
    dest = os.path.join(REPO, "results", "analyses", "wave1_analysis.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
