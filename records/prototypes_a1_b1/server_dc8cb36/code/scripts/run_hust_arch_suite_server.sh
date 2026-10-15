#!/usr/bin/env bash
# Run ALL nine architecture methods × three seeds with one frozen protocol.
# Invoke from the existing hust-SNR repository root, after git switch.
set -euo pipefail
: "${SOURCE_STRONG_ROOT:?Set parent directory containing generated/source/SNR_0 and SNR_5}"
: "${TARGET_SNR_ROOT:?Set parent directory with original Target-Dev SNR MAT folders}"
: "${TARGET_STRONG_ROOT:?Set parent directory containing generated/target/SNR_0 and SNR_5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${HUST_ARCH_SUITE_OUT:-outputs/hust_arch_suite_v1}"
"$PYTHON_BIN" -m pytest -q tests/test_hust_arch_suite.py
"$PYTHON_BIN" scripts/hust_arch_suite_experiment.py smoke
"$PYTHON_BIN" scripts/hust_arch_suite_experiment.py train --source-strong-root "$SOURCE_STRONG_ROOT" --out "$OUT"
"$PYTHON_BIN" scripts/hust_arch_suite_experiment.py evaluate --snr-root "$TARGET_SNR_ROOT" --strong-root "$TARGET_STRONG_ROOT" --out "$OUT"
"$PYTHON_BIN" scripts/hust_arch_suite_experiment.py summarize --out "$OUT"
echo "COMPLETED all nine architecture variants and 3 seeds. Reports: $OUT/reports"
