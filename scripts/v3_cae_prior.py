"""V3 HUST-paper CAE-Classification with a FilterNet-inspired low-pass prior.

Xu et al., IEEE TPEL 2025, Fig.4/Table III: existing OriginalCNN is already
the paper's encoder and FC classification head. Decoder here is an independently
implemented shape-reconciled approximation; see docs/V3_CAE_PRIOR_PROTOCOL.md.
"""
import torch
from torch import nn
from torch.nn import functional as F
from v2_paifilter import (RAW_POINTS, TRAINABLE_BANDS, EXACT_LOW_HARMONICS,
    HarmonicShaping, MDNormalization, OriginalCNN, deterministic_linear_resample)

METHODS = ("cae_identity", "cae_lowpass3", "cae_prior", "cae_prior_no_rec")


class PriorInitializedHarmonicFilter(nn.Module):
    """Five-phase-shared real gains; nearly hard three-harmonic low-pass at init."""
    def __init__(self, raw_points=RAW_POINTS, logit_magnitude=4.6, update_scale=8.0):
        super().__init__()
        if raw_points != RAW_POINTS or logit_magnitude <= 0 or update_scale <= 0:
            raise ValueError("Invalid frozen V3 spectral configuration")
        self.raw_points = raw_points
        self.update_scale = float(update_scale)
        self.residual = nn.Parameter(torch.zeros(TRAINABLE_BANDS))
        prior = torch.full((raw_points//2,), -float(logit_magnitude))
        prior[:3] = float(logit_magnitude)
        self.register_buffer("prior_logit", prior)

    def gains(self, reference):
        low = self.residual[:EXACT_LOW_HARMONICS]
        high = self.residual[EXACT_LOW_HARMONICS:].view(1,1,-1)
        expanded = deterministic_linear_resample(
            high, self.raw_points//2-EXACT_LOW_HARMONICS).reshape(-1)
        change = torch.cat((low, expanded))
        learned = torch.sigmoid(self.prior_logit + self.update_scale*change)
        return torch.cat((reference.new_ones(1), learned), dim=0)

    def forward(self, raw):
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (5,self.raw_points):
            raise ValueError("Expected five-phase raw [B,5,4096]")
        X = torch.fft.rfft(raw,dim=-1,norm="ortho")
        return torch.fft.irfft(X*self.gains(raw).to(X.dtype)[None,None,:],
                               n=self.raw_points,dim=-1,norm="ortho")


class PaperShapeReconciledDecoder(nn.Module):
    """Table III kernel sizes 3/3/15/15 and lengths 26/56/128/128.

    The printed table is internally inconsistent with ordinary Conv1d shape
    arithmetic. ConvTranspose1d plus explicit symmetric one-point padding
    is this project's stated implementation choice, not the authors' code.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(32,32,3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.ConvTranspose1d(32,16,3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.ConvTranspose1d(16,16,15)
        self.bn3 = nn.BatchNorm1d(16)
        self.out = nn.Conv1d(16,5,15,padding=7)

    def forward(self, h):
        if h.ndim!=3 or tuple(h.shape[1:])!=(32,11):
            raise ValueError("Decoder requires [B,32,11]")
        h=F.interpolate(h,scale_factor=2,mode="nearest")
        h=F.leaky_relu(self.bn1(F.pad(self.conv1(h),(1,1))),.01) # 26
        h=F.interpolate(h,scale_factor=2,mode="nearest")
        h=F.leaky_relu(self.bn2(F.pad(self.conv2(h),(1,1))),.01) # 56
        h=F.interpolate(h,scale_factor=2,mode="nearest")
        h=F.leaky_relu(self.bn3(F.pad(self.conv3(h),(1,1))),.01) # 128
        return self.out(h) # signed reconstruction [B,5,128]


class CAEFrequencyClassifier(nn.Module):
    def __init__(self,mode="cae_prior"):
        super().__init__()
        if mode not in METHODS:
            raise ValueError(mode)
        self.mode=mode
        if mode=="cae_identity":
            self.shaping=HarmonicShaping("identity")
        elif mode=="cae_lowpass3":
            self.shaping=HarmonicShaping("lowpass3")
        else:
            self.shaping=PriorInitializedHarmonicFilter()
        self.normalization=MDNormalization()
        self.backbone=OriginalCNN() # unchanged 25,620-param encoder+classifier
        self.decoder=(None if mode=="cae_prior_no_rec"
                      else PaperShapeReconciledDecoder())

    def forward(self,raw,with_reconstruction=False):
        signal=self.normalization(self.shaping(raw))
        features=self.backbone.features(signal)
        logits=self.backbone.classifier(features)
        if not with_reconstruction:
            return logits
        rec=self.decoder(features) if self.decoder is not None else None
        return logits,rec,signal


def paired_cae_objective(model,clean,noisy,label,lambda_rec=1.0):
    if lambda_rec<0 or clean.shape!=noisy.shape:
        raise ValueError("Invalid lambda or paired tensor shapes")
    logits,rec,signal=model(torch.cat((clean,noisy),dim=0),
                             with_reconstruction=True)
    n=len(label)
    ce=.5*(F.cross_entropy(logits[:n],label)+
           F.cross_entropy(logits[n:],label))
    # Autoencode the actual filtered inputs, NOT a Clean target from noisy.
    # Detach the target, preventing moving-target reconstruction gradients.
    mse=(.5*(F.mse_loss(rec[:n],signal[:n].detach())+
             F.mse_loss(rec[n:],signal[n:].detach()))
         if rec is not None else ce.new_zeros(()))
    return ce+lambda_rec*mse,{"ce":float(ce.detach()),
                              "reconstruction_mse":float(mse.detach())}


def count_parameters(model):
    return {"encoder_classifier":sum(p.numel() for p in model.backbone.parameters()),
            "decoder":sum(p.numel() for p in model.decoder.parameters()) if model.decoder else 0,
            "spectral":sum(p.numel() for p in model.shaping.parameters()),
            "total":sum(p.numel() for p in model.parameters())}


def smoke():
    torch.manual_seed(42)
    clean=torch.randn(3,5,RAW_POINTS)
    noisy=clean+.1*torch.randn_like(clean)
    for mode in METHODS:
        model=CAEFrequencyClassifier(mode)
        logits,rec,x=model(clean,with_reconstruction=True)
        assert logits.shape==(3,6) and x.shape==(3,5,128)
        assert (rec is None) if mode=="cae_prior_no_rec" else rec.shape==(3,5,128)
        loss,info=paired_cae_objective(model,clean,noisy,torch.tensor([0,1,5]))
        loss.backward()
        if "prior" in mode:
            g=model.shaping.residual.grad
            assert g is not None and g.abs().sum()>0 and torch.isfinite(g).all()
        print(mode,count_parameters(model),info)
    print("V3 architecture smoke PASSED")


if __name__=="__main__":
    smoke()
