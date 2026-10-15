#!/usr/bin/env bash
# Independent plain CNN1D comparison to the already completed ResNet1D baseline.
# Must execute in the SAME root/project/uv environment as V5/classical baseline.
# No previous experiment checkpoint is read, modified, or overwritten.
set -euo pipefail
: "${SOURCE_STRONG_ROOT:?set original strong-data parent (containing generated/source)}"
: "${TARGET_SNR_ROOT:?set original target SNR 10..40 parent}"
: "${TARGET_STRONG_ROOT:?set strong-data parent (containing generated/target)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${PLAIN_CNN_OUT:-outputs/plain_cnn1d_control_v1}"
"$PYTHON_BIN" -m pytest -q tests/test_plain_cnn1d_control.py
"$PYTHON_BIN" scripts/plain_cnn1d_experiment.py smoke
"$PYTHON_BIN" scripts/plain_cnn1d_experiment.py train \
  --seeds 42 43 44 --source-strong-root "$SOURCE_STRONG_ROOT" --out "$OUT"
"$PYTHON_BIN" scripts/plain_cnn1d_experiment.py evaluate \
  --seeds 42 43 44 --snr-root "$TARGET_SNR_ROOT" \
  --strong-root "$TARGET_STRONG_ROOT" --out "$OUT"
"$PYTHON_BIN" scripts/plain_cnn1d_experiment.py summarize \
  --seeds 42 43 44 --out "$OUT"
echo "PlainCNN1D completed at $OUT/reports. Historical ResNet1D and V5 results unchanged."
