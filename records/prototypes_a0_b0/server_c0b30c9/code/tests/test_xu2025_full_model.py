"""Paper-source invariants and loss differentiation tests for Xu full replication."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pytest
import torch
from xu2025_full_model import XuCAE,xu_metric_losses,xu_joint_loss,parameter_counts


def test_table_iii_encoder_and_decoder_shapes():
    m=XuCAE();z=torch.rand(16,5,128)
    assert m.encoder(z).shape==(16,32,11)
    result=m(z)
    assert result['reconstruction'].shape==(16,5,128)
    assert result['features'].shape==(16,352)
    assert result['logits'].shape==(16,6)
    assert parameter_counts(m)['paper_cae_dml']>25620


def test_gradients_reach_decoder_encoder_classifier():
    torch.manual_seed(6)
    m=XuCAE()
    x=torch.randn(12,5,128)
    y=torch.tensor([0,1,2,3,4,5]*2)
    output=m(x)
    loss=xu_joint_loss(output,x,y)
    loss['total'].backward()
    for layer in (m.encoder[0][0],m.decoder[1][0],m.classifier[0],m.classifier[-1]):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


def test_intra_and_inter_exact_tiny_reference():
    feats=torch.zeros(4,352)
    feats[0,:]=0;feats[1,:]=2
    feats[2,:]=4;feats[3,:]=6
    labels=torch.tensor([0,0,1,1])
    a,b,c=xu_metric_losses(feats,labels)
    assert a.item()==pytest.approx(1.)
    assert b.item()==pytest.approx(2.)
    assert c.item()==pytest.approx(-1.)


def test_paper_literal_reconstruction_is_640_times_element_mean():
    torch.manual_seed(42)
    m=XuCAE()
    x=torch.randn(8,5,128)
    o=m(x)
    y=torch.tensor([0,0,1,1,2,2,3,3])
    literal=xu_joint_loss(o,x,y,reconstruction='paper_sum')
    mean=xu_joint_loss(o,x,y,reconstruction='element_mean')
    assert literal['recon'].item()==pytest.approx(640*mean['recon'].item(),rel=1e-5)
    assert literal['ce'].item()==pytest.approx(mean['ce'].item())


def test_optional_dml_ablation_has_no_metric_contribution():
    m=XuCAE();x=torch.randn(8,5,128);y=torch.arange(8)%6
    a=xu_joint_loss(m(x),x,y,use_dml=False)
    assert a['dml'].item()==0
    assert a['total'].item()==pytest.approx(a['ce'].item()+a['recon'].item(),rel=1e-5)


def test_missing_class_in_batch_is_handled_and_not_invented():
    feats=torch.randn(3,352,requires_grad=True)
    labels=torch.zeros(3,dtype=torch.long)
    intra,inter,dml=xu_metric_losses(feats,labels)
    assert inter.item()==0
    dml.backward()
    assert feats.grad is not None


def test_invalid_shapes_fail():
    m=XuCAE()
    with pytest.raises(ValueError):m(torch.randn(2,5,4096))
    with pytest.raises(ValueError):xu_metric_losses(torch.randn(2,351),torch.tensor([0,1]))
