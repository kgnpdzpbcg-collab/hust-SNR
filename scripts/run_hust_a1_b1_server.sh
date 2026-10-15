#!/usr/bin/env bash
# Source-only A1/B1: test and smoke first; train and evaluate isolated files.
# Expects ROOT env vars to already point to the original *read-only* caches.
set -euo pipefail
: "$SOURCE_SPLIT_ROOT"
: "$TARGET_STRONG_ROOT"
: "$TARGET_EXTREME_ROOT"

python -m pytest -q tests/test_source_cycle_split.py tests/test_hust_prototypes_a1_b1.py
python scripts/hust_a1_b1_experiment.py smoke
python scripts/hust_a1_b1_experiment.py train \
  --methods a1 b1 --source-split-root "$SOURCE_SPLIT_ROOT"
python scripts/hust_a1_b1_experiment.py evaluate \
  --methods a1 b1 \
  --strong-root "$TARGET_STRONG_ROOT" \
  --extreme-root "$TARGET_EXTREME_ROOT" \
  --conditions 5 0 -5 -10
python scripts/hust_a1_b1_experiment.py summarize \
  --methods a1 b1 --conditions 5 0 -5 -10
echo "A1/B1 source-only experiment outputs written to isolated output root"
