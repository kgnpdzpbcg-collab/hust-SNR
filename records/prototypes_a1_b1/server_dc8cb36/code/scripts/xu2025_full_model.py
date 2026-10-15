"""Xu et al. (IEEE TPEL 2025) full CAE + DML: paper-grounded reimplementation.

Source: DOI 10.1109/TPEL.2024.3461159, Fig. 3/4, Table III, Eq. (5)-(20).
NOT an author-code reproduction: published Table III has channel/FC inconsistencies
and does not specify the decoder's padding, optimizer weight-decay value, or loss
weights. Choices are documented in docs/XU2025_FULL_REPRODUCTION.md.

The core XuCAE takes [B,5,128] normalized currents. XuFullRaw adapts the frozen
HUST raw [B,5,4096] cycle protocol for *matched* robust experiments only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS, POINTS, RAW_POINTS, CLASSES, EMBEDDING = 5, 128, 4096, 6, 352
VARIANTS = ("xu_full", "xu_full_lp3", "xu_full_lp4", "xu_full_nahsf", "xu_cae_classify")


def conv_block(in_ch: int, out_ch: int, kernel: int, padding: int = 0,
               activation: bool = True) -> nn.Sequential:
    parts: list[nn.Module] = [nn.Conv1d(in_ch, out_ch, kernel, padding=padding),
                              nn.BatchNorm1d(out_ch)]
    if activation:
        parts.append(nn.LeakyReLU(negative_slope=0.01))
    return nn.Sequential(*parts)


class XuCAE(nn.Module):
    """Complete three-block 1D convolutional autoencoder and two-layer classifier.

    Table III exact feature shapes: 128 -> 113 -> 56 -> 53 -> 26 -> 23 -> 11,
    flatten 32*11=352; decoder 11->22->26->52->56->112->128->128.
    Decoder zero paddings (3,3,15,7) make the table's reported shapes possible;
    these paddings are NOT specified by the paper and are explicit assumptions.
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(5,16,16), nn.MaxPool1d(2),
            conv_block(16,32,4), nn.MaxPool1d(2),
            conv_block(32,32,4), nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv_block(32,32,3,padding=3),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv_block(32,16,3,padding=3),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv_block(16,16,15,padding=15),
            conv_block(16,5,15,padding=7,activation=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING,50),nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(50,CLASSES),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNELS,POINTS):
            raise ValueError(f"XuCAE expects [B,5,128], received {tuple(x.shape)}")
        encoded = self.encoder(x)
        if encoded.shape[1:] != (32,11):
            raise RuntimeError("Encoder must reproduce the Table-III 32x11 latent")
        embedding = encoded.flatten(1)
        reconstruction = self.decoder(encoded)
        if reconstruction.shape != x.shape:
            raise RuntimeError("Decoder must reproduce Table-III 5x128 output")
        return {"logits":self.classifier(embedding), "reconstruction":reconstruction,
                "features":embedding}


def xu_metric_losses(features: torch.Tensor, labels: torch.Tensor,
                     n_classes: int = CLASSES) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Eq (14)-(18): average class-wise L1 intraclass minus L1 interclass.

    The article defines C class centroids from a training set but does not say
    how to compute them in mini-batches. We use PRESENT batch classes only;
    if <2 classes occur, interclass is 0. This is explicitly an approximation.
    Autograd flows through samples and centroids, with no detached prototypes.
    """
    if features.ndim != 2 or features.size(1) != EMBEDDING:
        raise ValueError("DML requires [B,352] flattened encoder features")
    if labels.ndim != 1 or labels.numel() != features.shape[0]:
        raise ValueError("DML requires one label per feature")
    if labels.numel()==0 or int(labels.min())<0 or int(labels.max())>=n_classes:
        raise ValueError("Invalid class labels")
    centers = []
    intra_terms = []
    for c in range(n_classes):
        mask = labels == c
        if bool(mask.any()):
            members = features[mask]
            center = members.mean(0)
            centers.append(center)
            intra_terms.append((members - center).abs().mean())
    intra = torch.stack(intra_terms).mean()
    if len(centers)>1:
        center_stack = torch.stack(centers)
        inter = (center_stack - center_stack.mean(dim=0,keepdim=True)).abs().mean()
    else:
        inter = features.new_zeros(())
    return intra, inter, intra - inter


def xu_joint_loss(output: dict[str,torch.Tensor], normalized_input: torch.Tensor,
                  labels: torch.Tensor, *, reconstruction: str = "paper_sum",
                  w_recon: float = 1., w_dml: float = 1., use_dml: bool = True
                  ) -> dict[str, torch.Tensor]:
    """Xu Eq (11),(13),(18),(20), with explicit paper/reduction ambiguity.

    'paper_sum' reproduces literal 1/N sum_i ||x_i-xhat_i||_2^2.
    'element_mean' reproduces common PyTorch MSE interpretation (a sensitivity
    choice, NOT an identical reading of the displayed equation).
    """
    if reconstruction not in ("paper_sum", "element_mean"):
        raise ValueError("Unknown reconstruction reduction")
    diffs = (normalized_input-output["reconstruction"]).square()
    recon_loss = (diffs.flatten(1).sum(dim=1).mean() if reconstruction=="paper_sum"
                  else diffs.mean())
    classification = F.cross_entropy(output["logits"],labels)
    if use_dml:
        intra, inter, dml = xu_metric_losses(output["features"],labels)
    else:
        intra = recon_loss.new_zeros(())
        inter = recon_loss.new_zeros(())
        dml = recon_loss.new_zeros(())
    total = classification+w_recon*recon_loss+w_dml*dml
    return {"total":total, "ce":classification, "recon":recon_loss,
            "dml":dml,"intra":intra,"inter":inter}


def fixed_harmonic_filter(raw:torch.Tensor, order:int)->torch.Tensor:
    """Optional NEW harmonic filtering branch, not part of Xu original model."""
    if order not in (3,4):
        raise ValueError("Only pre-registered LP3/LP4")
    spectrum = torch.fft.rfft(raw,n=RAW_POINTS,dim=-1,norm="ortho")
    mask = raw.new_zeros((RAW_POINTS//2+1,))
    mask[:order+1] = 1
    return torch.fft.irfft(spectrum*mask.view(1,1,-1),n=RAW_POINTS,dim=-1,norm="ortho")


class XuFullRaw(nn.Module):
    """Full paper CAE+DML with raw cycle adapter, optionally preceded by C.

    Core network XuCAE is INDEPENDENT of the simplified A' OriginalCNN class.
    Raw-path MD matches frozen V4 preprocessing for paired method comparisons.
    This is not represented as an exact reproduction of author's sampling code.
    """
    def __init__(self, variant:str="xu_full") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant=variant
        # Instantiate paper CAE first, so source seed parity across all arms.
        self.network=XuCAE()
        from v2_paifilter import MDNormalization
        self.normalize=MDNormalization()
        if variant=="xu_full_nahsf":
            from v5_nahsf import NoiseAwareHarmonicFusion
            self.shaping=NoiseAwareHarmonicFusion()
        else:
            self.shaping=None

    def forward(self,raw:torch.Tensor,return_gate:bool=False)->dict[str,torch.Tensor]:
        if raw.ndim!=3 or tuple(raw.shape[1:])!=(CHANNELS,RAW_POINTS):
            raise ValueError("Raw adapter requires [B,5,4096]")
        alpha=None
        if self.variant=="xu_full_lp3":raw=fixed_harmonic_filter(raw,3)
        elif self.variant=="xu_full_lp4":raw=fixed_harmonic_filter(raw,4)
        elif self.variant=="xu_full_nahsf":
            if return_gate:
                raw,alpha=self.shaping(raw,return_gate=True)
            else:raw=self.shaping(raw)
        normalized=self.normalize(raw)
        out=self.network(normalized)
        out["normalized_input"]=normalized
        if return_gate:
            out["alpha"]=alpha
        return out


def parameter_counts(model:nn.Module)->dict[str,int]:
    if isinstance(model,XuFullRaw):
        base=sum(p.numel() for p in model.network.parameters())
        spectral=sum(p.numel() for p in model.parameters())-base
        return {"paper_cae_dml":base,"spectral":spectral,"total":base+spectral}
    return {"paper_cae_dml":sum(p.numel() for p in model.parameters()),
            "spectral":0,"total":sum(p.numel() for p in model.parameters())}


def smoke()->None:
    torch.manual_seed(42)
    x=torch.randn(8,5,128)
    model=XuCAE()
    o=model(x)
    labels=torch.arange(8)%6
    terms=xu_joint_loss(o,x,labels)
    assert o["features"].shape==(8,352) and o["reconstruction"].shape==x.shape
    assert all(bool(torch.isfinite(v).all()) for v in terms.values())
    terms["total"].backward()
    assert model.decoder[1][0].weight.grad is not None
    assert model.encoder[0][0].weight.grad is not None
    print("Xu-2025 full CAE/DML smoke PASS", parameter_counts(model),
          {k:round(float(v.detach()),4) for k,v in terms.items()})

if __name__=="__main__":smoke()
