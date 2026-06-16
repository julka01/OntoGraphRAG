#!/bin/bash
# Sequential graph-state diversity runs on live KGs (no rebuild), via OpenRouter.
# Waits for the in-flight realmedqa run, then runs musique, then hotpotqa_fullwiki.
cd "/Users/sahibjulka/Documents/Documents backup/tool-2025-kg-rag" || exit 1
PY=.venv/bin/python
export WANDB_MODE=disabled
CH=results/diversity_chain.log
echo "chain started $(date)" > "$CH"

# 1. wait for the realmedqa diversity run to finish
while pgrep -f "datasets realmedqa" >/dev/null; do sleep 30; done
echo "realmedqa finished; starting musique $(date)" >> "$CH"

# 2. musique (live KG: full profile, n=100, seed 42)
$PY -m experiments.experiment --datasets musique --num-samples 100 --subset-seed 42 \
  --entropy-samples 5 --evaluation-mode full_metrics --retrieval-study final_pair \
  --kg-builder-profile full --dataset-kg-scope evaluation_subset \
  --llm-provider openrouter --llm-model openai/gpt-4o-mini \
  --output-dir results/diversity_run > results/diversity_run_musique.log 2>&1
echo "musique finished; starting hotpotqa_fullwiki $(date)" >> "$CH"

# 3. hotpotqa_fullwiki (live KG: lightweight profile, n=250, seed 42)
$PY -m experiments.experiment --datasets hotpotqa_fullwiki --num-samples 250 --subset-seed 42 \
  --entropy-samples 5 --evaluation-mode full_metrics --retrieval-study final_pair \
  --kg-builder-profile lightweight --dataset-kg-scope evaluation_subset \
  --llm-provider openrouter --llm-model openai/gpt-4o-mini \
  --output-dir results/diversity_run > results/diversity_run_fullwiki.log 2>&1
echo "CHAIN COMPLETE $(date)" >> "$CH"
