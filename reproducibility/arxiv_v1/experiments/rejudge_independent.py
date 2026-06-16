"""Independent re-judge of the headline RealMedQA runs (A3).

Re-labels the saved answers with a judge from a different model family
(Llama-3.3-70B via OpenRouter) using the exact judge prompt from the paper
appendix, then reports agreement with the original GPT-4o-mini labels and the
headline AUROCs recomputed under the independent labels.
No generation or retrieval is rerun; only judging.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gps_v2_replay import _load_env  # noqa: E402
from experiments.gps_v3_depth_matched import scores_for as gps_v3_scores_for  # noqa: E402
from experiments.trust_analysis import rows_for, REPO, MANIFEST, KEY, STRICT_RUNS  # noqa: E402

JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
SYSTEM_PROMPT = """You are a strict answer evaluator for a factoid question answering
task. Your job is to decide if a model's response is CORRECT.

Rules:
- Reply with exactly one word: 'correct' or 'incorrect'.
- The response is CORRECT only if it contains an answer semantically
  equivalent to the expected answer (minor spelling/accent differences
  are ok).
- The response is INCORRECT if the model says it doesn't know, cannot
  determine, or provides a factually different answer."""
USER_TMPL = """Question: {question}
Expected answer: {expected}
Model response: {response}

Is the model response correct? Reply with one word only:
correct or incorrect."""


def get_client():
    _load_env()
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1",
                  api_key=os.environ["OPENROUTER_API_KEY"])


def judge_one(client, q, expected, response, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0.0, max_tokens=4,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": USER_TMPL.format(
                              question=q, expected=expected, response=response)}])
            word = (r.choices[0].message.content or "").strip().lower()
            if "incorrect" in word:
                return False
            if "correct" in word:
                return True
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return None


def load_run(policy):
    """Return rows with question/expected/response/original label for RealMedQA."""
    if policy == "strict":
        doc = json.load(open(os.path.join(REPO, STRICT_RUNS["realmedqa"])))
        details = doc["config_results"][0]["details"]
        prefix = "kg"
        out = [(str(r["question_id"]), r["question"], r["expected"],
                r[f"{prefix}_response"], bool(r[f"{prefix}_correct"]))
               for r in details if not r.get(f"{prefix}_generation_failed")]
        return out
    doc = json.load(open(os.path.join(REPO, MANIFEST[KEY["realmedqa"]]["result_path"])))
    cfg = next(c for c in doc["config_results"]
               if c["config"]["name"].startswith("kg_entity_first"))
    return [(str(r["question_id"]), r["question"], r["expected"],
             r["kg_response"], bool(r["kg_correct"]))
            for r in cfg["details"] if not r.get("kg_generation_failed")]


def metric_aurocs(policy, labels_by_qid):
    """Recompute SD-UQ / SEU / GPS AUROC under given labels (KG side)."""
    rows = rows_for("realmedqa", policy if policy != "adaptive" else "kg")
    gps = gps_v3_scores_for("realmedqa", strict=(policy == "strict"), side="kg")
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
        out[name] = (float(roc_auc_score(w, s)) if len(set(w)) > 1 else None,
                     len(s))
    return out


def main():
    client = get_client()
    results = {"judge_model": JUDGE_MODEL}
    for policy in ("adaptive", "strict"):
        rows = load_run("strict" if policy == "strict" else "kg")
        print(f"[{policy}] judging {len(rows)} answers with {JUDGE_MODEL} ...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            verdicts = list(ex.map(
                lambda t: judge_one(client, t[1], t[2], t[3]), rows))
        ok = [(r, v) for r, v in zip(rows, verdicts) if v is not None]
        orig = np.array([r[4] for r, _ in ok], int)
        new = np.array([v for _, v in ok], int)
        agree = float((orig == new).mean())
        kappa = float(cohen_kappa_score(orig, new))
        labels = {r[0]: bool(v) for r, v in ok}
        aurocs = metric_aurocs(policy, labels)
        results[policy] = {
            "n_judged": len(ok), "n_failed": len(rows) - len(ok),
            "agreement": agree, "kappa": kappa,
            "accuracy_original": float(orig.mean()),
            "accuracy_independent": float(new.mean()),
            "auroc_under_independent_labels": {
                k: {"auroc": v[0], "n": v[1]} for k, v in aurocs.items()},
        }
        print(f"[{policy}] agreement={agree:.3f} kappa={kappa:.3f} "
              f"acc orig={orig.mean():.3f} indep={new.mean():.3f}")
        for k, v in aurocs.items():
            print(f"[{policy}]   {k}: AUROC={v[0] if v[0] is None else round(v[0],3)} (n={v[1]})")
    dest = os.path.join(REPO, "results", "analyses", "rejudge_independent.json")
    json.dump(results, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)


if __name__ == "__main__":
    main()
