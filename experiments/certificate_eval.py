"""Certificate evaluation (review-6 Issue 3): baselines + split-half.

(a) Matched-coverage baselines: SD-UQ-only gate (per-dataset percentile)
    accepting the same pooled coverage as the conjunctive certificate.
(b) Split-half validation: the only data-dependent threshold (the
    per-dataset SD-UQ median) is computed on a random half and the
    certificate is evaluated on the other half (B repeats).
(c) Per-dataset certificate coverage/precision.

Writes results/analyses/certificate_eval.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.trust_analysis import rows_for, gps_scores, REPO, KEY  # noqa: E402

RNG = np.random.default_rng(42)
B = 200


def load_all():
    data = {}
    for slug in KEY:
        rows = [r for r in rows_for(slug, "kg")
                if r["sd"] is not None and r["seu"] is not None]
        gps = gps_scores(slug)
        data[slug] = [(float(r["sd"]), float(r["seu"]), gps.get(r["qid"]),
                       bool(r["correct"])) for r in rows]
    return data


def certificate_mask(rows, sd_thresh):
    return np.array([sd <= sd_thresh and seu <= 0.5
                     and g is not None and g <= 0.5
                     for sd, seu, g, _ in rows])


def main():
    data = load_all()
    out = {}

    # (c) per-dataset certificate, thresholds in-sample (as in the paper)
    per_ds = {}
    sel_total = corr_total = n_total = 0
    for slug, rows in data.items():
        sd_med = float(np.median([sd for sd, _, _, _ in rows]))
        m = certificate_mask(rows, sd_med)
        y = np.array([c for _, _, _, c in rows])
        per_ds[slug] = {"n": len(rows), "coverage": float(m.mean()),
                        "selected": int(m.sum()),
                        "precision": float(y[m].mean()) if m.sum() else None}
        sel_total += int(m.sum()); corr_total += int(y[m].sum())
        n_total += len(rows)
    pooled_cov = sel_total / n_total
    out["per_dataset_certificate"] = per_ds
    out["pooled_certificate"] = {"coverage": pooled_cov,
                                 "precision": corr_total / sel_total,
                                 "selected": sel_total, "n": n_total}

    # (a) matched-coverage baselines at the same pooled coverage
    def gate_baseline(score_fn, name):
        sel = corr = 0
        for slug, rows in data.items():
            scores = np.array([score_fn(r) for r in rows])
            y = np.array([c for _, _, _, c in rows])
            k = int(round(pooled_cov * len(rows)))
            if k == 0:
                continue
            idx = np.argsort(scores)[:k]
            sel += k; corr += int(y[idx].sum())
        return {"coverage": sel / n_total, "precision": corr / sel,
                "selected": sel}

    out["baseline_sd_only"] = gate_baseline(lambda r: r[0], "sd")
    # percentile-rank combined (the audit score) at matched coverage
    def combined_rank(rows):
        sd = np.array([r[0] for r in rows]); seu = np.array([r[1] for r in rows])
        gpsr = np.array([1.0 if r[2] is None else r[2] for r in rows])
        def pr(v):
            order = v.argsort().argsort()
            return order / max(1, len(v) - 1)
        return (pr(sd) + pr(seu) + pr(gpsr)) / 3
    sel = corr = 0
    for slug, rows in data.items():
        scores = combined_rank(rows)
        y = np.array([c for _, _, _, c in rows])
        k = int(round(pooled_cov * len(rows)))
        if k == 0:
            continue
        idx = np.argsort(scores)[:k]
        sel += k; corr += int(y[idx].sum())
    out["baseline_combined_rank"] = {"coverage": sel / n_total,
                                     "precision": corr / sel, "selected": sel}

    # (b) split-half threshold validation
    precs, covs = [], []
    for _ in range(B):
        sel = corr = n_eval = 0
        for slug, rows in data.items():
            idx = RNG.permutation(len(rows))
            half = len(rows) // 2
            dev, test = idx[:half], idx[half:]
            sd_med = float(np.median([rows[i][0] for i in dev]))
            test_rows = [rows[i] for i in test]
            m = certificate_mask(test_rows, sd_med)
            y = np.array([c for _, _, _, c in test_rows])
            sel += int(m.sum()); corr += int(y[m].sum()); n_eval += len(test_rows)
        if sel:
            precs.append(corr / sel); covs.append(sel / n_eval)
    out["split_half"] = {
        "precision_mean": float(np.mean(precs)),
        "precision_p5": float(np.percentile(precs, 5)),
        "precision_p95": float(np.percentile(precs, 95)),
        "coverage_mean": float(np.mean(covs)),
        "B": B,
    }

    dest = os.path.join(REPO, "results", "analyses", "certificate_eval.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
