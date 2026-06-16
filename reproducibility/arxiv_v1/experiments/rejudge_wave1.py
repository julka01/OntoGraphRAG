"""Wave-1 independent re-judge (V1.1 / review W5).

Extends the RealMedQA re-judge (rejudge_independent.py) to the remaining
free-text datasets: HotpotQA, HotpotQA FullWiki, 2WikiMultiHopQA (adaptive +
strict), MuSiQue.  PubMedQA is excluded: its labels come from yes/no/maybe
normalisation, not the LLM judge.

Same judge model, prompt, and protocol as the original A3 check.  No
generation or retrieval is rerun; only judging.  Writes
results/analyses/rejudge_wave1.json.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.rejudge_independent import judge_one, get_client, JUDGE_MODEL  # noqa: E402
from experiments.trust_analysis import rows_for, REPO, MANIFEST, KEY, STRICT_RUNS  # noqa: E402
from experiments.gps_v3_depth_matched import scores_for as gps_v3_scores_for  # noqa: E402

TARGETS = [
    ("hotpotqa", "kg"),
    ("hotpotqa_fullwiki", "kg"),
    ("2wikimultihopqa", "kg"),
    ("2wikimultihopqa", "strict"),
    ("musique", "kg"),
]


def load_answers(slug, policy):
    if policy == "strict":
        doc = json.load(open(os.path.join(REPO, STRICT_RUNS[slug])))
        details = doc["config_results"][0]["details"]
    else:
        doc = json.load(open(os.path.join(REPO, MANIFEST[KEY[slug]]["result_path"])))
        details = next(c for c in doc["config_results"]
                       if c["config"]["name"].startswith("kg_entity_first"))["details"]
    return [(str(r["question_id"]), r["question"], r["expected"],
             r["kg_response"], bool(r["kg_correct"]))
            for r in details if not r.get("kg_generation_failed")]


def gps_scores(slug, strict):
    return gps_v3_scores_for(slug, strict=strict, side="kg")


def metric_aurocs(slug, policy, labels_by_qid):
    rows = rows_for(slug, policy)
    try:
        gps = gps_scores(slug, strict=(policy == "strict"))
    except Exception:
        gps = {}
    out = {}
    for name, getter in [("sd_uq", lambda r: r["sd"]), ("seu", lambda r: r["seu"]),
                         ("gps", lambda r: gps.get(r["qid"]))]:
        s, w = [], []
        for r in rows:
            lbl = labels_by_qid.get(r["qid"])
            v = getter(r)
            if lbl is None or v is None:
                continue
            s.append(float(v)); w.append(int(not lbl))
        out[name] = (float(roc_auc_score(w, s)) if len(set(w)) > 1 and len(s) > 10
                     else None, len(s))
    return out


def main():
    client = get_client()
    results = {"judge_model": JUDGE_MODEL}
    for slug, policy in TARGETS:
        tag = f"{slug}_{policy}"
        rows = load_answers(slug, policy)
        print(f"[{tag}] judging {len(rows)} answers with {JUDGE_MODEL} ...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            verdicts = list(ex.map(
                lambda t: judge_one(client, t[1], t[2], t[3]), rows))
        ok = [(r, v) for r, v in zip(rows, verdicts) if v is not None]
        orig = np.array([r[4] for r, _ in ok], int)
        new = np.array([v for _, v in ok], int)
        agree = float((orig == new).mean())
        kappa = float(cohen_kappa_score(orig, new))
        labels = {r[0]: bool(v) for r, v in ok}
        aurocs = metric_aurocs(slug, policy, labels)
        results[tag] = {
            "n_judged": len(ok), "n_failed": len(rows) - len(ok),
            "agreement": agree, "kappa": kappa,
            "accuracy_original": float(orig.mean()),
            "accuracy_independent": float(new.mean()),
            "auroc_under_independent_labels": {
                k: {"auroc": v[0], "n": v[1]} for k, v in aurocs.items()},
        }
        print(f"[{tag}] agreement={agree:.3f} kappa={kappa:.3f} "
              f"acc orig={orig.mean():.3f} indep={new.mean():.3f}", flush=True)
        for k, v in aurocs.items():
            print(f"[{tag}]   {k}: AUROC="
                  f"{v[0] if v[0] is None else round(v[0], 3)} (n={v[1]})", flush=True)
    dest = os.path.join(REPO, "results", "analyses", "rejudge_wave1.json")
    json.dump(results, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)


if __name__ == "__main__":
    main()
