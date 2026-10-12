"""V3 paper-derived CAE and low-pass-prior regression tests."""
import pytest
import torch
from v3_cae_prior import (METHODS, CAEFrequencyClassifier,
    PriorInitializedHarmonicFilter, PaperShapeReconciledDecoder,
    paired_cae_objective, count_parameters)
from v2_paifilter import RAW_POINTS


def test_all_model_shapes_and_counts():
    for mode in METHODS:
        m=CAEFrequencyClassifier(mode)
        logits,rec,x=m(torch.randn(2,5,RAW_POINTS),with_reconstruction=True)
        assert logits.shape==(2,6) and x.shape==(2,5,128)
        assert (rec is None) if mode=="cae_prior_no_rec" else rec.shape==(2,5,128)
        n=count_parameters(m)
        assert n["encoder_classifier"]==25620
        assert n["decoder"]==(0 if mode=="cae_prior_no_rec" else 9845)
        assert n["spectral"]==(128 if "prior" in mode else 0)


def test_decoder_backward():
    dec=PaperShapeReconciledDecoder()
    h=torch.randn(2,32,11,requires_grad=True)
    r=dec(h)
    assert r.shape==(2,5,128)
    r.square().mean().backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0


def test_prior_matches_lowpass_nearly():
    f=PriorInitializedHarmonicFilter()
    g=f.gains(torch.zeros(1,5,RAW_POINTS))
    assert g.shape==(2049,) and g[0]==1
    assert (g[1:4]>.98).all()
    assert (g[4:]<.011).all()


def test_fft_gates_no_phase_rotation():
    f=PriorInitializedHarmonicFilter()
    t=torch.arange(RAW_POINTS,dtype=torch.float32)
    x=(torch.sin(2*torch.pi*t/RAW_POINTS)+
       torch.cos(2*torch.pi*64*t/RAW_POINTS))[None,None,:].repeat(1,5,1)
    y=f(x)
    X=torch.fft.rfft(x,dim=-1)
    Y=torch.fft.rfft(y,dim=-1)
    assert .98<(Y[0,0,1].abs()/X[0,0,1].abs()).item()<1.
    assert (Y[0,0,64].abs()/X[0,0,64].abs()).item()<.011


def test_end_to_end_prior_and_decoder_gradients():
    m=CAEFrequencyClassifier("cae_prior")
    x=torch.randn(2,5,RAW_POINTS)
    loss,d=paired_cae_objective(m,x,x+.1*torch.randn_like(x),torch.tensor([1,3]))
    assert loss.item()>0 and d["reconstruction_mse"]>0
    loss.backward()
    assert m.decoder.conv1.weight.grad is not None
    assert torch.isfinite(m.shaping.residual.grad).all()
    assert m.shaping.residual.grad.abs().sum()>0


def test_no_rec_ablation():
    m=CAEFrequencyClassifier("cae_prior_no_rec")
    x=torch.randn(2,5,RAW_POINTS)
    loss,d=paired_cae_objective(m,x,x,torch.tensor([0,2]))
    assert loss.item()>0 and d["reconstruction_mse"]==0
    assert m.decoder is None


def test_prior_can_adjust_both_directions():
    f=PriorInitializedHarmonicFilter()
    before=f.gains(torch.zeros(1,5,RAW_POINTS)).detach().clone()
    opt=torch.optim.Adam(f.parameters(),lr=.1)
    g=f.gains(torch.zeros(1,5,RAW_POINTS))
    loss=-(g[6]-g[1])
    opt.zero_grad();loss.backward();opt.step()
    after=f.gains(torch.zeros(1,5,RAW_POINTS))
    assert after[6]>before[6] and after[1]<before[1]


def test_invalid_input_guards():
    with pytest.raises(ValueError): PriorInitializedHarmonicFilter(update_scale=0)
    with pytest.raises(ValueError): CAEFrequencyClassifier("unknown")
    with pytest.raises(ValueError): PriorInitializedHarmonicFilter()(torch.randn(1,5,128))
    m=CAEFrequencyClassifier("cae_identity")
    x=torch.randn(1,5,RAW_POINTS)
    with pytest.raises(ValueError): paired_cae_objective(m,x,x,torch.tensor([0]),-1)
