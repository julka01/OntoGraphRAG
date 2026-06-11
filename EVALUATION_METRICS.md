# Evaluation Metrics

This document describes the **current** experiment-time evaluation stack used by [experiment.py](experiments/experiment.py).

It replaces the older description of 6 LLM-scored quality metrics. That older setup is no longer the main evaluation path.

## Current Evaluation Layers

The runner evaluates each system output at four levels:

1. **Task-aware correctness**
2. **Answer EM/F1** for supported datasets
3. **Generation-failure-aware accuracy breakdown**
4. **Uncertainty ranking metrics** with `AUROC` / `AUREC`

The two systems compared are:
- **Vanilla RAG**
- **KG-RAG**

## 1. Task-Aware Correctness

Per-question correctness is computed by [_is_answer_correct()](experiments/experiment.py#L1756).

The logic depends on task type.

### Binary questions

For binary tasks such as `yes/no/maybe`, the runner:
- normalizes explicit decision labels
- optionally infers the label from free-form text

So correctness is:

```text
correct = 1[predicted_label == gold_label]
```

where `predicted_label` and `gold_label` are normalized into `{yes, no, maybe}` when possible.

### Multiple-choice questions

For `mcq` tasks, the runner prefers:
- option-letter matching like `A`, `Option B`, `Answer: C`
- then falls back to matching the correct option text

So correctness is:

```text
correct = 1[predicted_option in correct_options]
```

or, if no option letter is found:

```text
correct = 1[response semantically matches correct option text]
```

### Free-text / factoid questions

For free-text questions, the runner uses:
- an **LLM judge** by default, via [_llm_judge_correct()](experiments/experiment.py#L1698)
- a lexical fallback if the judge fails

Aliases are passed into the judge when available.

So the effective decision is:

```text
correct = judge(question, accepted_answers, response)
```

with fallback to heuristic matching if the judge errors.

## 2. Official-Style Answer EM/F1

For selected datasets, the runner also computes official-style answer normalization with [official_answer_metrics.py](experiments/official_answer_metrics.py).

Currently supported:
- `hotpotqa`
- `2wikimultihopqa`
- `musique`
- `multihoprag`

The normalization is HotpotQA-style:
- lowercase
- strip punctuation
- remove articles
- normalize whitespace

### Exact Match

```text
EM(pred, gold) = 1[normalize(pred) == normalize(gold)]
```

If aliases exist, the maximum EM over the accepted answer set is used.

### F1

For tokenized normalized answers:

```text
precision = overlap(pred_tokens, gold_tokens) / |pred_tokens|
recall    = overlap(pred_tokens, gold_tokens) / |gold_tokens|
F1        = 2 * precision * recall / (precision + recall)
```

Again, the maximum F1 over the gold answer plus aliases is used.

Special labels such as `yes`, `no`, `maybe`, and `insufficient information` are treated as exact categorical labels.

## 3. Accuracy Breakdown

The runner computes accuracy using [compute_accuracy_breakdown()](experiments/summary_utils.py#L13).

Let:
- `N` = total questions
- `F_v` = vanilla generation failures
- `F_k` = KG generation failures
- `C_v` = vanilla correct answers
- `C_k` = KG correct answers

### Raw accuracy

```text
vanilla_accuracy = C_v / N
kg_accuracy      = C_k / N
```

### Clean accuracy excluding each system's own generation failures

```text
vanilla_accuracy_excluding_errors = vanilla_correct_on_answered / (N - F_v)
kg_accuracy_excluding_errors      = kg_correct_on_answered / (N - F_k)
```

### Shared clean accuracy

This is computed on the subset where **neither** system failed generation:

```text
shared_clean_accuracy = correct_on_shared_answered / shared_answered_count
```

These are the main robustness numbers used to separate:
- answer quality
- provider/runtime failures

In current experiment reporting, `clean accuracy` is the headline number.
Raw accuracy is retained only as a secondary audit/debug field.

## 4. Uncertainty Metrics

In `full_metrics` mode, the runner computes the current **15-metric** suite.

### Output-side metrics

- `semantic_entropy`
- `discrete_semantic_entropy`
- `sre_uq`
- `p_true`
- `selfcheckgpt`
- `vn_entropy`
- `sd_uq`

These are computed from sampled responses and chunk-conditioned context in [uncertainty_metrics.py](experiments/uncertainty_metrics.py).

### Structural metrics

- `graph_path_support`
- `graph_path_disagreement`
- `competing_answer_alternatives`
- `evidence_vn_entropy`
- `subgraph_informativeness`
- `subgraph_perturbation_stability`

These are graph-native metrics that depend on the dataset-scoped KG.

### Grounding metrics

- `support_entailment_uncertainty`
- `evidence_conflict_uncertainty`

These are evidence-answer metrics computed over the retrieved chunk texts.

## 5. Selective-Prediction Metrics

For each uncertainty metric, the runner computes `AUROC` and `AUREC` when `full_metrics` is enabled.

### AUROC

`AUROC` measures how well an uncertainty score separates correct from incorrect answers.

Conceptually:

```text
AUROC = area under ROC curve for predicting correctness from uncertainty
```

Higher is better.

### AUREC

`AUREC` is the **Area Under the Rejection-Error Curve**.

Interpretation:
- sort answers by uncertainty
- abstain on the most uncertain answers first
- measure how quickly error falls

Lower is better.

## 6. What Is No Longer Current

The following description is **outdated** and should not be used to describe the current experiment runner:
- “6 LLM-scored quality metrics”
- generic `Correctness / Completeness / Relevance / Coherence / Factuality / Hallucination Level` scoring on a `0–10` scale

That older framing does not match the current code path.

## 7. Practical Summary

If you need the shortest accurate description of the current evaluation:

- **Per-question correctness** is task-aware and optionally judged by an independent LLM
- **Answer EM/F1** is computed for supported QA datasets
- **Clean accuracy** is the default reported accuracy
- **Raw accuracy** is retained only for debugging and audit trails
- **15 uncertainty metrics** are computed in `full_metrics` mode
- **AUROC/AUREC** evaluate how well each uncertainty metric predicts error / supports abstention

## Code References

- [experiment.py](experiments/experiment.py)
- [_is_answer_correct()](experiments/experiment.py#L1756)
- [_llm_judge_correct()](experiments/experiment.py#L1698)
- [official_answer_metrics.py](experiments/official_answer_metrics.py)
- [summary_utils.py](experiments/summary_utils.py#L13)
- [uncertainty_metrics.py](experiments/uncertainty_metrics.py)
