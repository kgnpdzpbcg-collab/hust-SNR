#!/usr/bin/env bash
# 3 standard baselines only: SVM-RBF, RF, 1D-ResNet. Never Final-Holdout.
set -euo pipefail
: "${SOURCE_STRONG_ROOT:?Set source strong-data parent, containing generated/source/SNR_0 and SNR_5}"
: "${TARGET_SNR_ROOT:?Set original measured target SNR_10..SNR_40 parent}"
: "${TARGET_STRONG_ROOT:?Set generated target 0/5 dB strong-data parent}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLASSICAL_OUT="${CLASSICAL_OUT:-outputs/classical_baselines_v1}"
"$PYTHON_BIN" scripts/classical_baseline_models.py
"$PYTHON_BIN" scripts/classical_baseline_experiment.py smoke
"$PYTHON_BIN" -m pytest -q tests/test_classical_baselines.py
"$PYTHON_BIN" scripts/classical_baseline_experiment.py train \
  --methods svm_rbf random_forest resnet1d --seeds 42 43 44 \
  --source-strong-root "$SOURCE_STRONG_ROOT" --out "$CLASSICAL_OUT"
"$PYTHON_BIN" scripts/classical_baseline_experiment.py evaluate \
  --methods svm_rbf random_forest resnet1d --seeds 42 43 44 \
  --snr-root "$TARGET_SNR_ROOT" --strong-root "$TARGET_STRONG_ROOT" \
  --out "$CLASSICAL_OUT"
"$PYTHON_BIN" scripts/classical_baseline_experiment.py summarize \
  --methods svm_rbf random_forest resnet1d --seeds 42 43 44 \
  --out "$CLASSICAL_OUT"
printf 'DONE: source-only classical baselines; report at %s/reports\n' "$CLASSICAL_OUT"
