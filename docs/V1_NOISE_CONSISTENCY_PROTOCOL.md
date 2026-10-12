# HUST-SNR V1 — paired noise consistency experiment

## Aim / status

Evaluate whether adding **a paired cosine feature-consistency term** to a conventional Gaussian-augmentation CNN improves six-class sim-to-real open-circuit fault diagnosis under SNR perturbations. **Method implementation is complete; the official V1 three-seed, seven-SNR GPU run is not yet recorded.** Do not confuse the quick local smoke run with a publication experiment.

Original HUST Gate0 reference results (target Dev only, three seeds): baseline **96.77% Clean / 78.50% 10 dB Macro-F1**; Gaussian augmentation **95.08% Clean / 83.73% 10 dB Macro-F1**. These are **previous Gate0**, not V1 results.

## Fixed comparisons

| Name | Training data | Objective |
|---|---|---|
| M0 `baseline` | labeled simulated Clean | CE(clean) |
| M1 `augmentation` | paired simulated Clean + Gaussian-noisy | 0.5 CE(clean) + 0.5 CE(noisy) |
| M2 `consistency` | **same paired views as M1** | M1 + **0.1 · mean[1 − cosine(h_clean,h_noisy)]** |

Same 25,620-param 1D-CNN as `scripts/train.py`. The feature `h` is **50-dimensional, after LeakyReLU and before the last FC layer**. Source sim full-record Gaussian noise per phase, SNR chosen uniformly from 10/15/20/25/30/35/40 dB per source record per epoch. Apply noise **before** unchanged phase-current normalization and 128-point resampling. In a given seed+epoch, M1 and M2 receive identically generated noisy examples and matched minibatch permutations. They initialize identically under a given seed.

Training: 50 epochs, Adam(lr=0.001, weight_decay=0), batch 32, seeds 42/43/44; source 34 training records (738 windows), 15 validation records (302 windows), 10 source test records (174 windows) from unchanged `configs/splits_gate0_v1.json`. No target labels or target batch normalization during training.

## Checkpoint selection: shared source-only noise-aware rule

Construct **fixed noise realizations on source Validation raw records**, independently at 10/20/30 dB, once per run. Measure Clean Val F1 and average three noisy Val F1 values each epoch. Eligible checkpoints must be **within 1 percentage point of the highest source Clean Val F1 across the 50 epochs**. Among eligible epochs select the one with highest mean source noisy Val Macro-F1; ties use higher Clean Val F1, then earlier epoch.

**All three methods use this same selection rule**. No target Dev/Final-Holdout information selects epoch, lambda, SNR range or checkpoint. Lambda fixed to 0.1 prior to V1 evaluation (no SNR-based hyperparameter search).

## Evaluation

Reuse unchanged `outputs/gate0_v1/target_dev_clean_windows.json`: exactly 88 unique raw record families. For each record, reuse **Clean-derived cycle boundaries**; evaluate Clean and 10/15/20/25/30/35/40 dB (the author's paired noisy files) with identical windows. Recompute normalize/resample from raw phase currents for each condition. Calculate Macro-F1, Accuracy, Balanced Accuracy, per-class F1/Recall, 6×6 confusion matrix, and **normal false-alarm rate** = fraction of true healthy windows predicted as any fault. Save per-window predictions/truth for independent recomputation. Distinguish *normal→fault false alarms* from *fault→normal misses*.

SNR noisy files are *derived from the same 440 original target records*: repeated SNR variants are NOT independent acquisitions. Never mix windows/SNR variants of an original record across Dev and Final-Holdout. **Final-Holdout 347 record families are not evaluated by this V1 entrypoint**.

## Running on the original GPU server

Run within the `hust-SNR` repository, using the *existing* prepared gate0 artifacts in `outputs/gate0_v1` and `data/source`/`data/target_dev/clean`. The `scripts/prepare.py` module is reused unchanged.

```bash
python scripts/v1_noise_consistency.py smoke
pytest -q tests/test_v1_consistency.py
python scripts/v1_noise_consistency.py train --methods baseline augmentation consistency --seeds 42 43 44
python scripts/v1_noise_consistency.py evaluate --methods baseline augmentation consistency --seeds 42 43 44 --snr-root "/PATH/TO/Five-phase-IM-AA-experimental-domain-different-SNR-data-set"
python scripts/v1_noise_consistency.py summarize --methods baseline augmentation consistency --seeds 42 43 44
```

The last `--snr-root` is a directory containing `SNR_10/`, `SNR_15/`, ..., `SNR_40/`, each with the 88 Dev record `.mat` files (or all 440). Alternatively put these files in `data/target_dev/snr10` ... `snr40` and omit the flag. The script **fails visibly** rather than substituting synthetic target noise when an official SNR file is missing.

Model checkpoints, history, prediction JSON, `SUMMARY.json` and `PER_SEED_RESULTS.csv` go into `outputs/v1_noise_consistency/` (gitignored). Training refuses to overwrite an existing run; use a distinct `--output` for reruns. V0 files under `records/gate0_v1/`, `configs/` and `scripts/train.py` are unchanged.

## Required acceptance checks

1. Compare M1 vs M2 on 10 dB, 15 dB, ... 40 dB using **paired seeds**; check whether gain persists, not just mean 10 dB.
2. Check normal false-alarm rates, Clean Macro-F1 and hardest individual fault categories; robust noise learning should not substantially trade away healthy detection.
3. Run independent 347-record Final-Holdout **once after freezing** the V1 method/analysis; that evaluation needs a separate, explicitly authorized script and cannot be used for V1 design choices.
4. No claim of novelty or superiority until M2 convincingly exceeds M1 across multiple SNRs and seeds.

## Local preflight record

- 2026-10-12: synthetic tensor `smoke` completed, plus `pytest` six tests. The same V1 training entrypoint was exercised using **real unpacked HUST simulation MAT** with deliberately reduced 3-train-condition/1-val-condition subset, seed 42, 2 epochs, all three methods; this was strictly a CPU pipeline smoke check. It is **not** the frozen 34-record/50-epoch V1 experiment and not reported as comparative accuracy.
- This runtime does not have the author's 1.26-GiB 7-SNR archive or access to the user's GPU server. No V1 official noisy-target accuracy has been fabricated.