"""V3 driver reuses frozen V2, rather than redefining split or noise RNG."""
import v3_experiment as v3
import v2_experiment as v2


def test_driver_patches_exact_v2_interfaces():
    assert v2.METHODS == v3.METHODS
    assert v2.HarmonicCNN.__name__ == "CAEFrequencyClassifier"
    assert v2.DEFAULT_OUT.as_posix()=="outputs/v3_cae_prior"


def test_training_loss_contains_reconstruction():
    import torch
    from v3_cae_prior import CAEFrequencyClassifier
    clean=torch.randn(2,5,4096)
    model=CAEFrequencyClassifier("cae_identity")
    loss=v2.paired_cross_entropy(model,clean,clean+.1*torch.randn_like(clean),
                                 torch.tensor([0,1]))
    assert torch.isfinite(loss) and loss.item()>0


def test_no_target_in_v3_objective():
    from v3_cae_prior import paired_cae_objective
    assert "target" not in paired_cae_objective.__code__.co_varnames
