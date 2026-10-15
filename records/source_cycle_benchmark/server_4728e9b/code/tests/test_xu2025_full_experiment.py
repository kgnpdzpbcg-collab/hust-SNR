"""Protocol-level static sanity checks without access to private server data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from xu2025_full_model import XuCAE,VARIANTS


def test_standalone_full_model_does_not_reuse_simplified_aprime():
    model=XuCAE()
    assert hasattr(model,'decoder')
    assert hasattr(model,'classifier')
    assert not hasattr(model,'backbone')
    assert len(model.encoder)==6
    assert 'xu_full_nahsf' in VARIANTS


def test_decoder_has_four_convolutions_as_paper_table():
    model=XuCAE()
    layers=[layer for block in model.decoder for layer in
            (list(block.children()) if isinstance(block,torch.nn.Sequential) else [block])]
    assert len([layer for layer in layers if isinstance(layer,torch.nn.Conv1d)])==4


def test_decoder_padding_explicit_reproducibility_assumptions():
    m=XuCAE()
    assert [m.decoder[i][0].padding[0] for i in [1,3,5,6]]==[3,3,15,7]
