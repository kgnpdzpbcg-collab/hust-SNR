# Hang (IEEE TIE 2023) vs. current HUST five-phase adaptation: code and method audit

**Status:** source-paper/code/static audit plus direct inspection of the user's 63 source-domain MAT records. No retraining and no target-domain performance claims. Historical V2/V3/V4/V5 branches and weights are unchanged.

## Primary source

Jun Hang, Xiaoman Shu, Shichuan Ding, Yourui Huang, *Robust Open-Circuit Fault Diagnosis for PMSM Drives Using Wavelet Convolutional Neural Network With Small Samples of Normalized Current Vector Trajectory Graph*, IEEE Transactions on Industrial Electronics 70(8), 7653–7662 (2023), DOI `10.1109/TIE.2022.3231304`.

Checked original PDF: Section II-B equations (1)–(6), Figs. 2–4; Section II-C Eqs. (7)–(13), Figs. 5–7; Section III, text describing `I1,C2,W3,C4,W5,C6,C7,C8,F9,O7`; Sections III/IV discussing 15/20 dB AWGN and mixed noisy/clean trajectory images.

## Audited source code (V5 branch)

- `scripts/jung_hang_models_v5.py::hang_trajectory`: fundamental five-phase VSD followed by normalization **at each time instant**, rasterization to a 40×40 binary image.
- `scripts/jung_hang_models_v3.py::HangWCNN`: five Conv2D stages with channels `4,8,16,32,64`, kernel `3×3`, BN and ReLU, two Haar LL-only pooling stages after the first two convolutions, classifier output 6.
- `scripts/jung_hang_models_v3.py::HaarDWTLLPool`: unnormalized 2×2 all-ones filter, stride 2. This implements published Eq. (11) **as printed**.
- `scripts/jung_hang_cycle_v5.py`: paired clean/noisy source training; AWGN nominal 15 or 20 dB; 50 epochs, Adam 1e-3, batch 32, source-checkpoint selection based partly on extreme 0/5/10 dB source validation. These details are **project assumptions**; original paper does not publish a complete training schedule or checkpoint selection rule.

## Critical ambiguity in the published paper: Eq. (6) vs. Figs. 3–4

Eq. (6) explicitly writes the normalized current vector as `Is/|Is|`. Applying this literal formula **to every fault-state time point** maps every nonzero vector onto a circle of constant radius. All radial magnitude variation is mathematically erased. Yet original Figs. 3–4 draw single-switch half-circles and multi-switch trajectories with inward *radial line segments* and different radial geometries. Such figures cannot be created from a time-series of strictly unit-norm vectors alone (apart from exact zero points), so the paper's equations/illustrations **do not uniquely define** normalization of a faulted vector.

The V5 code implements the literal time-point denominator. It therefore matches one reading of Eq. (6) but **is not faithful to the published fault-trajectory geometry**. Do not assert either that the author's code used our per-point implementation or that the author's code used our chosen per-window reference scaler. This is an unresolved original-method detail.

The V6 audit adds `hang_trajectory_reference_scaled()`: it uses one positive reference radius (maximum magnitude of the current **window's** fundamental VSD trajectory). This preserves fault-related radial segments and normalizes common current scale. It is explicitly a *figure-consistent engineering assumption*, not an author-verified normalization rule. Both `pointwise_eq6` (unchanged V5 behavior) and `reference_radius_figures` are selectable without altering previous V5 results.

## Actual source-domain experiment: irreversible information collision

Using **all 63 source simulation MAT files**, record field `I_We`, fixed 10 kHz clock, five current phases plus electrical speed, the same phase-angle-integral fault steady-cycle boundaries (`0.5s + 2 electrical cycles` to `0.98s`), 128 interpolated points per electrical cycle and 40×40 binary trajectory rasterization:

- Healthy current cross-plane verification: the third-harmonic VSD plane / fundamental plane RMS ratio has a **median 1.025e-4** over the 63 pre-fault healthy reference segments. This supports the chosen 0/72/144/216/288-degree phase ordering as internally consistent on these recordings; it does not alone prove physical wiring metadata.
- On the **34-record source Train operating-condition groups**, there are **631 fault-state single-cycle windows** spanning filename fault group identifiers 1–5.
- With V5 `pointwise_eq6`: **27 distinct raster images** occur under more than one fault-group label; **70 / 631 = 11.09%** of fault windows are assigned an exact image that also belongs to another class.
- With the shared-radius representation applied to the same current windows: **6 distinct ambiguous images** and **23 / 631 = 3.65%** of windows.
- On the **15-record source Val operating-condition groups**, there are 260 fault-state windows: pointwise **10 / 260 = 3.85%**, shared-radius **4 / 260 = 1.54%**.
- The rates are descriptive **input-collision measures**, *not* fitted classifier error rates or target-domain Macro-F1 values. Duplicate imaging can arise both from actual equivalent trajectories and from 40×40 raster quantization; a full model cannot disambiguate samples given **identical** images. They were computed on source data only, without target labels.

Notable exact collision example: a fault-state image from `S-400-0-1-O` has the same pixel array as images from `S-400-0-2-O` and `S-400-0-3-O` under pointwise normalization. These three filenames have different fault-group identifiers. The source-train/val summary above excludes the 400-rpm record group and does not reuse it in training.

## Component-level verdict

| Item | Paper | Current V5 | Verdict |
|---|---|---|---|
| Three-phase Clarke plane | `i_alpha, i_beta` of PMSM | Five-phase fundamental VSD `alpha1, beta1` | Necessary topology adaptation; healthy energy check strongly supports selected phase order |
| Normalized trajectory | Eq. (6) pointwise-looking text; Figs. 3–4 include radial paths | literal pointwise division | **Critical, unresolved paper ambiguity** with observed source input-label collisions |
| Plot sampling | No exact points per trajectory/window specified | 128 points across one electrical cycle | Implementation assumption; must be treated as such |
| Trajectory image | No complete rasterization/line thickness algorithm specified | binary 40×40 one-pixel raster with interpolation | Implementation assumption; may remove subtle class differences |
| WCNN conv count | WCNN-5 | five convolution layers | Consistent |
| WCNN channels | 4,8,16,32,64 | same | Consistent |
| Conv kernels | all 3×3 | same | Consistent |
| Wavelet pooling | two DWT-LL poolings in Fig. 6, Eq. (11) | two unnormalized Haar low-low pools after first two convs | Consistent with printed formula |
| Last mapped feature | C7 has 10×10×32; C8 has 64 filters | 40→20→10 and C7 gives 10×10×32 | Internally consistent; 40×40 input inferred, not separately stated |
| Batch norm | used, Eq. (8) | BN after conv | Consistent at component level |
| Loss | printed Eq. (13) binary CE-like, 22 exclusive fault classes | torch multiclass CE on six exclusive labels | Defensible adaptation; Eq. (13) is not sufficient to uniquely specify authors' exact Keras loss |
| AWGN | 15 and 20 dB described in simulations; clean/noisy images mixed | new noise each epoch, random 15/20 dB records, 0.5 CE for clean/noisy | Method-consistent idea but **not verified original training protocol** |
| Model selection | not fully specified | custom source Clean+0/5/10/20/30 dB F1/CE rule | Project protocol; not author reproduction |
| Fault taxonomy | three-phase two-level PMSM, 22 states | five-phase IM, 6 states | Non-equivalent physical fault task; paper-level numerical scores cannot be expected |

## Correct implementation next steps

1. **Do not edit or overwrite V5 checkpoints or history**. Use branch `research/hang-paper-fidelity-audit-v6` and `configs/hang_only_v6_figures.json` in a new output directory.
2. Run `python scripts/jung_hang_cycle_v5.py preflight --config configs/hang_only_v6_figures.json ...`. The new `PREFLIGHT_HANG_SOURCE_COLLISIONS.json` compares both input representations using only source Train; preflight also retains existing preflight checks.
3. Inspect multiple source fault classes: raw five-phase current, pre/post VSD `alpha beta` trajectory, normalized 40×40 image, image pixel-collision report. Confirm which representation reproduces Figure 3/4's **radial** geometry rather than merely maximizing accuracy.
4. Training (only after input audit): method=`hang`, seeds=42/44; source Train/Val only. Evaluate clean and noise across target-Dev without target-based model selection. Exact published SNR experiments 15/20 dB are not equivalent to project's more severe -10dB stress test.
5. If paper author's exact normalization/plot-generation code is unavailable, describe final baseline as **Hang et al. (WCNN, five-phase adaptation)** and state which trajectory normalization was used. Do not claim exact reproduction.

**Final status:** Original paper/CNN/DWT matching checked; a significant **representation-level problem** was empirically demonstrated. Modified source-train auditable pathway and explicit alternative normalization are committed, but no GPU model has been retrained. Future results may or may not improve.
