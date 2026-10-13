#!/usr/bin/env bash
# Paper-grounded Xu2025 CAE+DML: server execution with explicit provenance.
# Must run in repo root with the SAME v2/v5 validated Python environment.
# WARNING: no training/evaluation uses Target-Final-Holdout or source test.
set -euo pipefail
: "${SOURCE_STRONG_ROOT:?set absolute path containing SNR_0 and SNR_5 source MAT subdirectories}"
: "${TARGET_SNR_ROOT:?set absolute path containing original SNR_10...SNR_40 target MAT directories}"
: "${TARGET_STRONG_ROOT:?set absolute path containing generated target SNR_0 and SNR_5 MAT directories}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${XU_FULL_OUT:-outputs/xu2025_full}"

"$PYTHON_BIN" scripts/xu2025_full_model.py
"$PYTHON_BIN" scripts/xu2025_full_experiment.py smoke
"$PYTHON_BIN" -m pytest -q \
  tests/test_xu2025_full_model.py \
  tests/test_xu2025_full_experiment.py

# Keep the two study tracks separate; identical source-record IDs but distinctly
# named training protocols. Each run refuses checkpoint/evaluation overwrites.
"$PYTHON_BIN" scripts/xu2025_full_experiment.py train \
  --track paper_like --modes xu_full --out "$OUT"

"$PYTHON_BIN" scripts/xu2025_full_experiment.py train \
  --track matched \
  --modes xu_full xu_full_lp3 xu_full_lp4 xu_full_nahsf xu_cae_classify \
  --source-strong-root "$SOURCE_STRONG_ROOT" --out "$OUT"

for track in matched paper_like; do
  if [[ "$track" == matched ]]; then
    modes=(xu_full xu_full_lp3 xu_full_lp4 xu_full_nahsf xu_cae_classify)
  else
    modes=(xu_full)
  fi
  "$PYTHON_BIN" scripts/xu2025_full_experiment.py evaluate \
    --track "$track" --modes "${modes[@]}" \
    --snr-root "$TARGET_SNR_ROOT" --strong-root "$TARGET_STRONG_ROOT" --out "$OUT"
  "$PYTHON_BIN" scripts/xu2025_full_experiment.py summarize \
    --track "$track" --modes "${modes[@]}" --out "$OUT"
done

echo "Complete. Review results in $OUT/{matched,paper_like}/reports/ before publication."
