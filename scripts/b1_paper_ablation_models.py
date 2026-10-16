"""Five *independent* B1 paper ablations, sharing the frozen B1 implementation.

A1: fixed LP4 + exactly B1's two-view classifier (no restorer training).
A2: replace the per-sample noise proxy by a source-train-only fixed scalar.
A3: mask only relative-power/Gram inputs to the adjustment Conv1d.
A4: use only MD view with the unchanged B1 restorer and recovery losses.
A5: use full B1 forward, but remove the phase-strength recovery loss.

No change is made to hust_b1_phase_harmonic.py. The original B1 stays the
reference control. All variants are trained from scratch, never by pruning
a previously selected B1 checkpoint.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from hust_b1_phase_harmonic import (
    B1PhaseAwareRecoveryNet,
    MAX_ORDER,
    PhaseAwareHarmonicRestorer,
    b1_recovery_objective,
    shared_harmonic_scale,
)
from v2_paifilter import CNN_POINTS, PHASES, RAW_POINTS, deterministic_linear_resample

# IDs are part of the frozen config and the saved checkpoint provenance.
A1 = "a1_fixed_lp4_dual"
A2 = "a2_fixed_noise_proxy"
A3 = "a3_no_interphase_descriptors"
A4 = "a4_md_only"
A5 = "a5_no_strength_loss"
METHODS = (A1, A2, A3, A4, A5)


class FixedLP4Restorer(nn.Module):
    """Exact k=0..4 passthrough and k>4 zero, no trainable parameters.

    The details use the same four-tuple contract as B1's restorer, to make
    the *dual-view classification* path identical. Variance is only a
    placeholder; this model has no recovery supervision.
    """

    def forward(self, raw: torch.Tensor, return_details: bool = False):
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError(f"Expected [batch,{PHASES},{RAW_POINTS}]")
        full = torch.fft.rfft(raw, dim=-1, norm="ortho")
        harmonics = full[:, :, :MAX_ORDER + 1]
        zeros = torch.zeros_like(full)
        restored = torch.fft.irfft(
            torch.cat((harmonics, zeros[:, :, MAX_ORDER + 1:]), dim=-1),
            n=RAW_POINTS, dim=-1, norm="ortho")
        if return_details:
            return (restored, harmonics, torch.zeros_like(harmonics.real),
                    torch.ones_like(harmonics.real))
        return restored


class A1FixedLP4DualView(B1PhaseAwareRecoveryNet):
    def __init__(self):
        super().__init__()
        self.restorer = FixedLP4Restorer()


class FixedSourceNoiseProxyRestorer(PhaseAwareHarmonicRestorer):
    """Noise proxy = one SOURCE-TRAIN calibrated constant for every sample.

    Replaces BOTH uses of noise_proxy in parent forward (noise variance and
    adjustment features); retains the learnable order calibration a_k.

    fixed_proxy is a persistent checkpoint buffer, never a learnable param.
    It is intentionally invalid until initialized using source training data.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("fixed_proxy", torch.tensor(float("nan")))

    @torch.no_grad()
    def calibrate_source(self, noisy_source: torch.Tensor, batch_size: int = 128):
        if noisy_source.ndim != 3 or tuple(noisy_source.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError("Source noise calibration requires [N,5,4096]")
        if noisy_source.size(0) == 0 or batch_size < 1:
            raise ValueError("Source noise calibration needs nonempty data")
        observations = []
        for block in noisy_source.split(batch_size, dim=0):
            spectrum = torch.fft.rfft(block, dim=-1, norm="ortho")
            scale = shared_harmonic_scale(spectrum)
            # Explicit call to the BASE implementation, not self.noise_proxy.
            observations.append(
                PhaseAwareHarmonicRestorer.noise_proxy(spectrum, scale).reshape(-1))
        proxy = torch.cat(observations).median()
        if not bool(torch.isfinite(proxy)) or not bool(proxy > 0):
            raise FloatingPointError("Invalid source-only noise proxy")
        self.fixed_proxy.copy_(proxy.to(self.fixed_proxy))

    def noise_proxy(self, full_spectrum: torch.Tensor, shared_scale: torch.Tensor):
        if not bool(torch.isfinite(self.fixed_proxy)) or not bool(self.fixed_proxy > 0):
            raise RuntimeError("A2 fixed noise proxy not calibrated from Source-Train")
        return self.fixed_proxy.to(
            dtype=shared_scale.dtype).expand(full_spectrum.size(0), PHASES)


class A2FixedNoiseProxy(B1PhaseAwareRecoveryNet):
    def __init__(self):
        super().__init__()
        self.restorer = FixedSourceNoiseProxyRestorer()


class MaskInterphaseDescriptors(nn.Module):
    """Explicitly disable only two five-channel input feature groups.

    Original adjustment input layout (five phases):
       0:5  log(signal power)          unchanged
       5:10 log(relative phase power) MASK
      10:15 normalized real Gram      MASK
      15:20 log(noise proxy)          unchanged
      20:25 tanh(log observed SNR/8)  unchanged
      25    harmonic order           unchanged

    No change to trainable Conv1d parameter count or tensor shape.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 5 * PHASES + 1:
            raise ValueError("Unexpected B1 adjustment descriptor layout")
        return torch.cat((features[:, :PHASES, :],
                          torch.zeros_like(features[:, PHASES:3 * PHASES, :]),
                          features[:, 3 * PHASES:, :]), dim=1)


class A3NoInterphaseDescriptors(B1PhaseAwareRecoveryNet):
    def __init__(self):
        super().__init__()
        # No copied B1 forward: mask applied exactly where adjustment reads F.
        original_adjustment = self.restorer.adjustment
        self.restorer.adjustment = nn.Sequential(
            MaskInterphaseDescriptors(), *list(original_adjustment.children()))


class A4MDOnly(B1PhaseAwareRecoveryNet):
    def __init__(self):
        super().__init__()
        # Keep all residual blocks, but accept only five phase-wise MD channels.
        self.backbone.stem[0] = nn.Conv1d(
            PHASES, 32, 7, padding=3, bias=False)

    def forward(self, raw: torch.Tensor, return_recovery: bool = False):
        restored, harmonics, variance, gain = self.restorer(
            raw, return_details=True)
        md = self.normalize(restored)
        hidden = self.backbone.pool(
            self.backbone.layers(self.backbone.stem(md))).flatten(1)
        logits = self.backbone.classifier(hidden)
        if return_recovery:
            return logits, harmonics, variance, gain
        return logits


class A5NoStrengthLoss(B1PhaseAwareRecoveryNet):
    """Full B1 forward unchanged; only the training objective differs."""


def build_model(method: str) -> nn.Module:
    classes = {
        A1: A1FixedLP4DualView,
        A2: A2FixedNoiseProxy,
        A3: A3NoInterphaseDescriptors,
        A4: A4MDOnly,
        A5: A5NoStrengthLoss,
    }
    if method not in classes:
        raise ValueError(f"Unknown B1 ablation: {method}")
    return classes[method]()


def calibrate_a2_from_source(model: nn.Module, raw: dict, metadata: list,
                             seed: int, noise_levels: list | tuple):
    """Deterministic calibration on SOURCE TRAIN noisy inputs only.

    Uses the SAME training-data augmentation routine and first-epoch seed as
    the original run, without any target measurements or their labels.
    """
    if not isinstance(model, A2FixedNoiseProxy):
        return
    import numpy as np
    import v2_experiment as data

    source_noisy = data.fresh_noisy_source(
        raw, metadata, seed, 1, noise_levels=noise_levels)
    if not isinstance(source_noisy, np.ndarray):
        raise TypeError("Expected numpy source noisy cycles")
    model.restorer.calibrate_source(torch.from_numpy(source_noisy))


def recovery_loss(method: str, noisy: torch.Tensor, clean: torch.Tensor,
                  recovered: torch.Tensor, noise_variance: torch.Tensor):
    return b1_recovery_objective(
        noisy, clean, recovered, noise_variance,
        w_phase=0.0 if method == A5 else 0.4)


def paired_loss(method: str, model: nn.Module,
                clean: torch.Tensor, noisy: torch.Tensor,
                labels: torch.Tensor, lambda_restore: float = 1.0):
    if method == A1:
        # Identical clean/noisy cross-entropy formulation used by all
        # four normal CNN/LP3/LP4 controls in the original benchmark.
        logits = model(torch.cat((clean, noisy), dim=0))
        n = labels.shape[0]
        ce = 0.5 * (
            F.cross_entropy(logits[:n], labels) +
            F.cross_entropy(logits[n:], labels))
        return ce, {"ce": ce.detach()}

    n = labels.shape[0]
    if n == 0 or clean.shape != noisy.shape or labels.shape != (n,):
        raise ValueError("Bad paired source training batch")
    logits, recovered, variance, _ = model(
        torch.cat((clean, noisy), dim=0), return_recovery=True)
    ce = 0.5 * (
        F.cross_entropy(logits[:n], labels) +
        F.cross_entropy(logits[n:], labels))
    rec, parts = recovery_loss(
        method, noisy, clean, recovered[n:], variance[n:])
    total = ce + lambda_restore * rec
    return total, {"ce": ce.detach(), "restore": rec.detach(), **parts}


def pretrain_restorer(method: str, model: nn.Module, raw: dict, metadata: list,
                     clean, seed: int, config: dict, device: torch.device):
    """Exact B1 ten-epoch recovery pretraining schedule for A2..A5.

    A1 has no trainable restorer and must not run recovery pretraining.
    """
    if method == A1:
        return []
    import numpy as np
    import v2_experiment as data

    opt = torch.optim.Adam(
        model.restorer.parameters(), lr=config["learning_rate"], weight_decay=0)
    history = []
    for epoch in range(1, config["b1_pretrain_epochs"] + 1):
        noisy = data.fresh_noisy_source(
            raw, metadata, seed, epoch, noise_levels=config["train_noise_db"])
        perm = np.random.default_rng(seed + 11001 + epoch).permutation(len(clean))
        model.restorer.train()
        losses = []
        for pos in range(0, len(perm), config["batch_size"]):
            idx = perm[pos:pos + config["batch_size"]]
            nx = torch.from_numpy(noisy[idx]).to(device)
            cx = torch.from_numpy(clean[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            _, spec, variance, _ = model.restorer(nx, return_details=True)
            loss, _ = recovery_loss(method, nx, cx, spec, variance)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite recovery loss: {method}/{seed}/{epoch}")
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all())
                       for p in model.restorer.parameters()):
                raise FloatingPointError("Nonfinite recovery gradient")
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        if epoch == 1 or epoch == config["b1_pretrain_epochs"]:
            print(f"{method} seed{seed} recovery pretrain {epoch}: {history[-1]:.6f}",
                  flush=True)
    return history
