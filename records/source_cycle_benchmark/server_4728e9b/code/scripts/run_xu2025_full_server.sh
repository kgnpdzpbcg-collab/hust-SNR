#!/usr/bin/env bash
# Xu et al. (TPEL 2025) FULL CAE+DML baseline ONLY, for comparison with
# previously completed A' / V5 experiments. No full-Xu LP3/LP4/NA-HSF arms.
# Execute at repository root in the SAME validated Python/CUDA environment.
# Uses matched V5 noise/data/selection protocol, not the paper's original
# unspecified training implementation. Does not touch Final-Holdout.
set -euo pipefail
: "${SOURCE_STRONG_ROOT:?set parent folder containing source SNR_0 and SNR_5 MAT files}"
: "${TARGET_SNR_ROOT:?set parent folder containing target SNR_10...SNR_40 files}"
: "${TARGET_STRONG_ROOT:?set parent folder containing generated target SNR_0 and SNR_5 files}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${XU_FULL_OUT:-outputs/xu2025_full}"

"$PYTHON_BIN" scripts/xu2025_full_model.py
"$PYTHON_BIN" scripts/xu2025_full_experiment.py smoke
"$PYTHON_BIN" -m pytest -q \
  tests/test_xu2025_full_model.py \
  tests/test_xu2025_full_experiment.py

# ONLY ONE model type: xu_full (complete encoder+decoder+classifier+DML).
# Three seeds are replicates of that SAME baseline, not three architectures.
"$PYTHON_BIN" scripts/xu2025_full_experiment.py train \
  --track matched --modes xu_full --seeds 42 43 44 \
  --source-strong-root "$SOURCE_STRONG_ROOT" --out "$OUT"

"$PYTHON_BIN" scripts/xu2025_full_experiment.py evaluate \
  --track matched --modes xu_full --seeds 42 43 44 \
  --snr-root "$TARGET_SNR_ROOT" --strong-root "$TARGET_STRONG_ROOT" --out "$OUT"

"$PYTHON_BIN" scripts/xu2025_full_experiment.py summarize \
  --track matched --modes xu_full --seeds 42 43 44 --out "$OUT"

echo "Xu-Full ONLY complete. See $OUT/matched/reports/. Existing V5 experiments remain unchanged."
