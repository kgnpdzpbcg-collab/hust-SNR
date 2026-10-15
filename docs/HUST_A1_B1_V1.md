# A1/B1 – independently trained, noise-aware prototype V1

**Status: MODEL CODE COMMITTED, SERVER GPU TRAINING NOT RUN.** Never present predicted performance or validation assumptions as measured results.

## Data protocol

Repo branch: research/a1-b1-noise-structure-v1, forked from research/harmonic-recovery-wavelet-a0-b0. Historical A0/B0 results and weights remain unchanged.

* Five-phase IM input: five synchronized currents of one complete electrical cycle, [B,5,4096], original file-level frozen segmentation.
* Training source: 34 distinct simulation recordings / 738 cycles; source val: 15 independent simulation recordings / 302 cycles; six fault classes. No target-domain training.
* Source train noise: same original per-full-record per-phase RMS calibrated additive Gaussian RNG, 10/15/20/25/30/35/40 dB; clean/noisy paired cycles. Source val: clean and 0/5/10/20/30 dB from frozen deterministic noise. Three seeds 42,43,44.
* Source-only checkpoint selection: first clean Macro-F1 within one percentage point of max, then largest five-band source noisy Macro-F1 average, then larger clean F1. **If all F1 tie, use smaller mean source noisy cross entropy**, then earliest epoch. This directly addresses observed early source-validation saturation, but is not a guarantee of better sim-to-real transfer. The selection change must be disclosed in comparisons with historical A0/B0/LP4.
* Measured Dev: same 88 families, 2474 cycles in each of 5/0/-5/-10 dB; these levels derive from scaling ONE author's 10 dB residual, not independent measurements. Eval only after source-selected weights saved. Leave 347-record Final-Holdout untouched.
* Adam lr1e-3, batch32, 50 classification epochs, B1 additional 10 restoration pretrain epochs. All outputs use a new experiment root. No old checkpoint overwrites.

## A1 – Noise-sensitive wavelet reliability CNN

Reference: MWA-CNN (Wang et al., MSSP 2023); adapts DWT/FAM rather than reimplementing author architecture. Script scripts/hust_a1_reliability_wavelet.py.

**Forward [B,5,4096] -> [B,6]**:

1. Compute each phase's time-harmonic k1 magnitude from orthonormal rFFT; use the **median across five phases** to obtain a single robust, observed fundamental amplitude. Normalize all phases by this shared scale to preserve weak-phase ratios; pool 4096->512 by averaging consecutive groups of 8. The scale does not assume known SNR or a fault label.
2. At each of four levels use orthonormal Haar DWT on every channel, producing low/detail halves. The low half is preserved **by the attention gate**. Compute RMS of the low and detail coefficients, not signed temporal mean. Predict detail gain = sigmoid( log(low_rms+epsilon)-log(high_rms+epsilon) + correction_MLP(log1p low RMS, log1p high RMS, log1p high/low ratio) ). The correction's last layer is initialized to zero so high/detail energy ratio controls initial shrinkage. Gain always between zero and one, unlike archived A0's 1+sigmoid always-enhancing FAM. Apply dropout only to detail branch.
3. Concatenate low and attenuated detail, then Conv1D(k3)+GroupNorm+GELU. Four widths 24,48,64,96 and time lengths 256,128,64,32. GAP and linear six-class head. No LP4, ResNet, auxiliary loss or noise-consistency.
4. Source supervised loss = 0.5 CE(clean) + 0.5 CE(noisy).

Caveats: high-detail RMS also increases when there is genuine high-frequency fault content; low-band is preserved only at gate, not all future convs; DWT octave scales are not integer electrical harmonic orders; the new fundamental scale may still be affected by -10dB noise. Actual improvement is unverified.

## B1 – Phase-coupled, noise-calibrated k0..4 harmonic recovery

Reference ideas: SSLWPT supervised structure restoration, DRSN shrinkage, multiphase current relative phase. The actual VSD spatial transform of the six-phase reference paper is **not** implemented. Script scripts/hust_b1_phase_harmonic.py.

**Restoration [B,5,4096] -> [B,5,5] complex harmonics plus [B,5,4096] real reconstructed current**:

1. Per-phase FFT, retain k=0..4 (prior B0 frozen-weight investigation showed k5..8 harmed very-low-SNR classification). Use observed five-phase median k1 complex magnitude as one common scale. No per-phase normalization or arbitrary complex cross-phase mixing.
2. Compute five-phase observed high-band k20..80 median power as *noise proxy*. Multiply proxy by learned positive per-order exp(log_noise_order_scale[k]), estimating low-order noise variance. This may generalize badly from simulated Gaussian noise to measured colored residuals and is not exact SNR.
3. Build input-only, global-cycle-phase-invariant descriptors from each observed coefficient's power, median-relative power, same-order cross-phase complex Gram coherence, normalized noise proxy, log(signal/noise power) and order index. Conv1D hidden width 48 predicts bounded logit correction.
4. Actual restoration is **multiplicative and nonamplifying**: estimated spectrum = observed spectrum * sigmoid(log power - log noise variance + 2*tanh(correction)). There is no additive complex correction and no manufactured phasor. Circular time translation rotates coefficients but leaves gains unchanged.
5. Inverse FFT after zeroing k>4. ResNet input has two five-phase views: original MD shape (includes the documented discontinuous open-phase threshold) and separate common-normalized five-phase waveform (keeps weak-phase relative amplitude). Concatenate to [B,10,128]; replace only ResNet stem input 5->10, otherwise preserve standard ResNet blocks.
6. Source clean/noisy pair restoration loss has: (i) normalized complex SmoothL1; (ii) SmoothL1 of log1p of relative phase harmonic energy (weak open phases protected, without assuming healthy symmetry); (iii) log-noise-power calibration against *known source injected noise*, averaged over batch and phase. Restoration objective = complex + 0.4*phase + 0.1*noise_cal. Ten restoration-only pretrain epochs, then 50 joint epochs with paired CE + 1.0*restoration loss.

Caveats: multiplicative shrink cannot correct noisy phase; the source noise profile may not extrapolate to the measured residual at -10dB; MD threshold remains discontinuous; k0..4 may discard genuine information. Clean SOURCE is a supervised training target only; CLEAN measured signal or label is never used at inference.

## Commands (existing server environment and data caches)

1. python -m pytest -q tests/test_hust_prototypes_a1_b1.py
2. python scripts/hust_a1_b1_experiment.py smoke
3. python scripts/hust_a1_b1_experiment.py train --methods a1 b1 --source-strong-root /path/to/strong_root
4. python scripts/hust_a1_b1_experiment.py evaluate --methods a1 b1 --strong-root /path/to/strong_root/generated/target --extreme-root /path/to/extreme_dev_parent --conditions 5 0 -5 -10
5. python scripts/hust_a1_b1_experiment.py summarize --methods a1 b1

The extreme root must contain both SNR_-5 and SNR_-10 with 88 frozen matching Dev MAT families (already generated in previous runs). Do not run an old dataset generator again if those paths exist.

Use scripts/run_hust_a1_b1_server.sh after configuring all three root environment variables. The stage runner is isolated: no Final-Holdout or Source-Test evaluation logic is provided.

## Quality gate and interpretation

Model files and a source-only runner have been reviewed and core forward/backward on CPU locally checked. Full repo tests, server CUDA deterministic checks and all five-phase actual data training/evaluation remain pending. A1/B1 are deliberately separate trained models. Compare them with historical A0/B0 and ResNet+LP4, disclose source CE tie selection and B1 extra pretrain budget, keep per-seed Macro-F1 and class confusion records, and do not finalize a conference-paper method on unverified gains. Six-phase PMSM integration is a separate task.
