#!/usr/bin/env bash
# Run only on the existing HUST GPU server with the original source caches.
# B0 recovery warmup 10 epochs + joint training 50 epochs; A0 training 50.
set -euo pipefail
: "$SOURCE_STRONG_ROOT"
: "$TARGET_STRONG_ROOT"
: "$TARGET_EXTREME_ROOT"

python -m pytest -q tests/test_hust_prototypes_a0_b0.py
python scripts/hust_prototypes_experiment.py smoke
python scripts/hust_prototypes_experiment.py train \
  --methods b0 a0 --source-strong-root "$SOURCE_STRONG_ROOT"
python scripts/hust_prototypes_experiment.py evaluate \
  --methods b0 a0 --strong-root "$TARGET_STRONG_ROOT" \
  --extreme-root "$TARGET_EXTREME_ROOT" --conditions 5 0 -5 -10
python scripts/hust_prototypes_experiment.py summarize \
  --methods b0 a0 --conditions 5 0 -5 -10
echo "A0/B0 SOURCE-ONLY RESEARCH PROTOTYPES FINISHED"
