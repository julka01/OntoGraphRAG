#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Intentionally omit --rebuild-kg and keep evaluation_subset scope so a
# compatible shared-corpus RealMedQA KG can be reused instead of rebuilt.
python experiments/experiment.py \
  --datasets realmedqa \
  --num-samples 230 \
  --subset-seed 42 \
  --entropy-samples 5 \
  --evaluation-mode full_metrics \
  --retrieval-study final_pair \
  --kg-builder-profile full \
  --similarity-thresholds 0.1 \
  --max-chunks-values 10 \
  --dataset-kg-scope evaluation_subset \
  --output-dir results/latest_kg_design_final_metrics
