# HUST A0 / B0 Independent Algorithm Prototypes (V0)

Status: **implementation committed; GPU dataset training and Target-Dev scoring NOT RUN**.
Branch: research/harmonic-recovery-wavelet-a0-b0
Base: research/architecture-sweep-v1 (commit 12fc3667).

## 1. Problem and non-negotiable controls

Target problem: severe-noise sim-to-real open-switch diagnosis in multiphase
converter currents. Train/val on HUST five-phase IM-AA **simulation ONLY**;
Target-Dev contains 88 ORIGINAL measured recording families, 2474 full-cycle
windows, six labels: Healthy, OUS(A), OLS(A), OP(A), OUS(A)+OUS(B),
OLS(A)+OP(C). Holdout of 347 distinct measured families is **not accessed**.

* Frozen source train: 34 records / 738 cycles; source val: 15 / 302.
* All experiments use seeds **42, 43, 44** only.
* Source augmentation: per-record, per-phase RMS-calibrated Gaussian
  10,15,20,25,30,35,40 dB, new realization per epoch (existing RNG).
* Source checkpoint selector: among epochs within 1 percentage point
  of the run's peak clean Macro-F1, choose the maximum five-band
  source-val mean Macro-F1 at 0,5,10,20,30 dB.
* Main measured Target-Dev reporting: **5,0,-5,-10 dB only**.
  5/0/-5/-10 are ALL synthetic stress levels derived by scaling the
  SAME author-provided 10dB measured noise residual. They are NOT independent
  field measurements. The SNR rating is whole-record and per-phase, not
  exactly the SNR of every segmented cycle.
* Target-Dev is used only for evaluation; it has been examined in previous
  project design iterations, so it is NOT a fully untouched method-design set.
  Final-Holdout MUST remain unexamined until the method is frozen.

The main competitor is previously trained ResNet1D+LP4:
5dB 98.10%, 0dB 94.44%, -5dB 80.43% (three-seed Target-Dev Macro-F1).
Its -10dB performance has not been computed.

The two proposed candidates do NOT inherit LP4; they are alternative
algorithms whose usefulness is entirely contingent on the experiments.

## 2. B0: supervised multiphase complex-harmonic recovery

Reference inspirations:
* Kang et al. (2026), SSLWPT, Applied Soft Computing 191, 114713:
  restoration model with a structural signal prior before diagnosis.
* Zhao et al. (2020), DRSN, IEEE TII 16(7): adaptive shrinkage.
* Xu et al. (2026), dual Y-PMSM, IEEE TPEL: phase-coupled harmonic
  representations, without copying PMSM-specific transform matrices.

Implementation: scripts/hust_b0_harmonic_recovery.py.

Input [B,5,4096] angle-synchronous source cycles; each phase undergoes
orthonormal real FFT. Keep complex harmonic orders **k=0..8**; this K=8
is a deliberate V0 hypothesis, not a proven optimum. Compute a SHARED
five-phase fundamental-amplitude reference scale, avoiding per-phase
normalization of an open phase. Compute an input-only, per-phase noise
descriptor using the median k=20..80 power normalized by this scale.
This is a noise proxy, NOT an oracle or certified SNR estimator.

Input to a 3-layer width-48 Conv1D spectral encoder along harmonic order
is 10 real/imag channels + 5 noise channels + 1 order-position channel:
[B,16,9]. Predict gain [B,5,9] and a bounded complex residual [B,10,9]:

    estimated_complex = common_scale * (
         sigmoid(gain_head(h)) * normalized_observed_complex
         + 0.35 * complex(tanh(residual_head(h)))
    )

Gain initialized sigmoid(4)=0.982; residual initialized zero.
Only DC imaginary coefficients are explicitly zeroed. Orders >8 are
zeroed before 4096-point irFFT. No known noise level, label, clean target,
or other domain data enters forward().

Decoder is FFT inverse (not a trained decoder). The waveform is passed
through the ORIGINAL, one-pass MDNormalization (same weak-phase branch,
4096->128 deterministic interpolation) and the existing 241702-param
five-phase ResNet stem/layers/GAP/classifier, bypassing its own forward()
to prevent an accidental second MD transform.

Restoration source supervision: pair CLEAN simulation cycle and its
generated NOISY cycle; target is the CLEAN k=0..8 complex spectrum.
Compute smooth-L1 on real/imag of the complex error divided by the
sample's CLEAN shared-fundamental scale. This avoids gross domination
by operating-point amplitude, but it is not a guarantee of fault-specific
information preservation.

Stage I: 10 epochs of recovery-only training on source noise.
Stage II: 50 epochs of joint source-only training:

    L = 0.5*CE(clean view) + 0.5*CE(noisy view)
        + 0.2*recovery_loss(reconstructed NOISY spectrum, CLEAN source)

The 10 extra recovery epochs must be reported as extra training budget,
not presented as cost-identical with ResNet+LP4. Both weights were fixed
before testing; do not tune on Target-Dev results.

Main risks: phase-shift sensitivity; high-band residual power not an
unbiased estimator of low-order noise; 10..40dB source restoration
training may fail to extrapolate to -10dB; K8 may reintroduce nuisance
harmonics; new Fourier gain cannot recover information absent from
the noisy measurement. The current model does NOT implement full VSD or
equivariant cross-phase covariance/Gram modules; those are possible
future mechanisms, not claimed implemented in B0.

## 3. A0: multi-phase multi-layer wavelet-attention classifier

Reference: Wang et al. (2023), MWA-CNN, Mechanical Systems and Signal
Processing 195, 110314. This is an ADAPTATION, not a reproduction of the
authors' 4096x1 / six-DWA-layer network.

Implementation: scripts/hust_a0_wavelet_attention.py.

Source input [B,5,4096] is scaled by ONE shared five-phase RMS per cycle:
cross-phase ratios, sign and DC offsets are retained; no per-phase
extremum centering. Block-average every 8 interpolation samples to
[B,5,512] before wavelet analysis (not the original MD procedure).
The first Haar DWT is exact orthonormal analysis on adjacent pairs:

    L = (x_even+x_odd)/sqrt(2), H = (x_even-x_odd)/sqrt(2).

Concatenate low/high along channels [B,2C,L/2]; MWA-inspired FAM
uses signed temporal GAP -> Linear -> ReLU -> Linear -> Sigmoid and
outputs F*(1+attention). Apply dropout 0.1. Each level then performs
Conv1D k3 + GroupNorm + ReLU, jointly mixing five-phase information.
Four stages produce the sizes:

    [B,5,512]
       DWA -> [B,10,256]  / Conv -> [B,24,256]
       DWA -> [B,48,128] / Conv -> [B,48,128]
       DWA -> [B,96,64]  / Conv -> [B,64,64]
       DWA -> [B,128,32] / Conv -> [B,96,32]
       GAP -> 96 -> six-way logits.

A0 is an independent classification network; no FFT, LP4, ResNet or
signal reconstruction. Training is the original paired source
clean/noisy CE, **50 epochs**; no noise-consistency penalty.

Main risks: 512-point averaging can erase useful fault features;
signed temporal GAP can miss sign-changing high-frequency energy;
channel residual enhancement is not an actual noise energy estimator;
Haar octave bands do not isolate individual integer electrical
harmonics; high-noise source/target mismatch remains.

## 4. Server commands (from repository root)

Existing root/cache/uv environment is required; original data files and
trained historical models are NOT shipped with this branch.

1. Verify import/gradients/test (CPU or server):

    python -m pytest -q tests/test_hust_prototypes_a0_b0.py
    python scripts/hust_prototypes_experiment.py smoke

2. Make the -5/-10dB **Dev-only** stress MAT records:

    python scripts/generate_extreme_dev_snr.py \
      --clean-root /path/to/original/experimental_clean_mat \
      --snr10-root /path/to/original/SNR_10_mat \
      --out /path/to/new/extreme_dev_root

Output has SNR_-5/ and SNR_-10/ subfolders (88 files in each)
with the original field name I_noisy. Their time/omega columns and
whole-record phase SNR are checked before/after writing. The path
MUST NOT exist before generation; all original MATs are read-only.

3. Train B0 FIRST (all seeds), then A0 (all seeds):

    python scripts/hust_prototypes_experiment.py train \
      --methods b0 --source-strong-root /path/to/old_strong_data_root

    python scripts/hust_prototypes_experiment.py train \
      --methods a0 --source-strong-root /path/to/old_strong_data_root

This expects the same source generated 0/5 validation MATs at
SOURCE_STRONG_ROOT/generated/source/SNR_0, SNR_5 as the nine-arm
architecture experiment. Training produces checkpoints in
outputs/hust_prototypes_a0_b0_v0/models/{b0,a0}/seed{42,43,44}.
Existing outputs are NOT overwritten.

4. Target-Dev inference from FROZEN source-selected weights:

    python scripts/hust_prototypes_experiment.py evaluate \
      --methods b0 a0 \
      --strong-root /path/to/old_strong_data_root/generated/target \
      --extreme-root /path/to/new/extreme_dev_root \
      --conditions 5 0 -5 -10

    python scripts/hust_prototypes_experiment.py summarize --methods b0 a0

The runner uses the existing record loader and asserts the exact
88-family / 2474-window label order. It will not load other target
records. Results are written as per-seed JSON plus CSV:
PER_SEED_RESULTS, PER_CLASS_PER_SEED and SUMMARY (mean/sample SD).

A single all-in-one shell wrapper is also provided:
scripts/run_hust_prototypes_server.sh; the separate commands above
are safer for inspecting B0 before training A0.

## 5. Limits / criteria for proceeding

A0 and B0 have NOT been evaluated on Target-Dev or Final-Holdout here.
No guarantee exists that either outperforms the strong ResNet+LP4 baseline.
Results under -10dB must be actually measured, not extrapolated from -5dB.

Before finalizing the conference paper, compare all methods on consistent
5,0,-5,-10dB Target-Dev conditions, disclose the B0 extra pretraining
budget, check per-class error patterns, and then freeze a final architecture
for a one-time independent Final-Holdout evaluation. Later six-phase
PMSM support requires separate preprocessing/channel-count/label-adapter
work; five-phase V0 is not falsely advertised as six-phase runnable.
