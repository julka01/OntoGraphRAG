"""V3.4 / review M4: verbalised-confidence P(True) baseline from saved answers.

The paper's P(True) proxy is cluster agreement, which is monotone in DSE
under black-box sampling.  The original P(True) asks the model for a
probability.  This script queries the same generator family (GPT-4o-mini via
OpenRouter) once per saved question/answer pair:

    "Probability that the proposed answer is correct: <float>"

and evaluates AUROC of (1 - p) against the paper's correctness labels, per
dataset, for the adaptive-KG and dense policies.  No retrieval or answer
generation is rerun.  Writes results/analyses/verbalized_confidence.json.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.trust_analysis import REPO, MANIFEST, KEY  # noqa: E402
from experiments.rejudge_independent import get_client  # noqa: E402

MODEL = "openai/gpt-4o-mini"
SYSTEM = ("You estimate the probability that a proposed answer to a question "
          "is correct. Respond with ONLY a number between 0 and 1.")
USER_TMPL = ("Question: {question}\n"
             "Proposed answer: {response}\n\n"
             "Probability that the proposed answer is correct (0 to 1):")


def ask_conf(client, question, response, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=8,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER_TMPL.format(
                              question=question, response=response)}])
            txt = (r.choices[0].message.content or "").strip()
            m = re.search(r"\d*\.?\d+", txt)
            if m:
                p = float(m.group(0))
                if p > 1:
                    p = p / 100.0 if p <= 100 else None
                if p is not None and 0.0 <= p <= 1.0:
                    return p
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def load_details(slug, policy):
    doc = json.load(open(os.path.join(REPO, MANIFEST[KEY[slug]]["result_path"])))
    cfg_prefix = "kg_entity_first" if policy == "kg" else "dense_floor"
    details = next(c for c in doc["config_results"]
                   if c["config"]["name"].startswith(cfg_prefix))["details"]
    prefix = "kg" if policy == "kg" else "vanilla"
    return [(str(r["question_id"]), r["question"], r[f"{prefix}_response"],
             bool(r[f"{prefix}_correct"]))
            for r in details if not r.get(f"{prefix}_generation_failed")]


def main():
    client = get_client()
    out = {"model": MODEL}
    for policy in ("kg", "vanilla"):
        out[policy] = {}
        pooled_u, pooled_w = [], []
        for slug in KEY:
            rows = load_details(slug, policy)
            print(f"[{slug}/{policy}] querying {len(rows)} confidences ...",
                  flush=True)
            with ThreadPoolExecutor(max_workers=8) as ex:
                ps = list(ex.map(lambda t: ask_conf(client, t[1], t[2]), rows))
            ok = [(r, p) for r, p in zip(rows, ps) if p is not None]
            u = [1.0 - p for _, p in ok]
            w = [int(not r[3]) for r, _ in ok]
            auc = (float(roc_auc_score(w, u))
                   if len(set(w)) > 1 and len(w) > 10 else None)
            out[policy][slug] = {
                "n": len(ok), "n_failed": len(rows) - len(ok),
                "auroc": auc,
                "mean_conf_correct": float(np.mean(
                    [p for (r, p) in ok if r[3]])) if any(r[3] for r, _ in ok) else None,
                "mean_conf_wrong": float(np.mean(
                    [p for (r, p) in ok if not r[3]])) if any(not r[3] for r, _ in ok) else None,
            }
            pooled_u += u; pooled_w += w
            print(f"[{slug}/{policy}] AUROC={auc if auc is None else round(auc,3)} "
                  f"(n={len(ok)})", flush=True)
        out[policy]["pooled"] = {
            "n": len(pooled_w),
            "auroc": (float(roc_auc_score(pooled_w, pooled_u))
                      if len(set(pooled_w)) > 1 else None)}
    dest = os.path.join(REPO, "results", "analyses", "verbalized_confidence.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)


if __name__ == "__main__":
    main()
