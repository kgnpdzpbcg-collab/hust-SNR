"""Tests for frozen source protocol and V4 baseline comparison guards."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import v5_experiment as v5
from v5_nahsf import NAHSFCNN


def test_frozen_protocol_and_mode():
    assert v5.METHOD == 'nahsf'
    assert v5.SEEDS == (42,43,44)
    assert v5.ORIGINAL_CONDITIONS == (None,10,15,20,25,30,35,40)
    assert v5.SELECT_RULE.startswith('source ')
    assert v5.VERSION == 'HUST-V5-NA-HSF-v0'


def test_train_argument_and_overwrite_guard(tmp_path):
    with pytest.raises(ValueError):
        v5.train(42,tmp_path,epochs=0)
    with pytest.raises(ValueError):
        v5.train(42,tmp_path,batch=0)
    folder = tmp_path / 'models' / 'nahsf' / 'seed42'
    folder.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        v5.train(42,tmp_path)


def test_raw_cycle_training_smoke():
    v5.smoke()


def test_predict_gate_shapes_and_distribution():
    model = NAHSFCNN()
    raw = np.random.default_rng(0).normal(size=(5,5,4096)).astype('float32')
    pred,alpha,z = v5.predict_with_gate(model,raw,torch.device('cpu'),batch_size=2)
    assert pred.shape == (5,) and alpha.shape == (5,) and z.shape == (5,3)
    stat = v5.gate_summary(alpha,z,np.arange(5,dtype=np.int64))
    assert stat['mean'] == pytest.approx(.5)
    assert stat['p10'] == pytest.approx(.5)
    assert len(stat['by_true_class_mean']) == 6


def _make_eval(root, method, seed, cond, preds, truth, is_v5):
    p=root/method/f'seed{seed}'/f'target_dev_{cond}.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    from v1_noise_consistency import metrics
    record={'method':method,'seed':seed,'condition':cond,'record_families':88,
            'windows':2474,'truth':truth,'predicted':preds,
            'metrics':metrics(truth,preds),'final_holdout_evaluated':False}
    if is_v5: record['gate'] = {'mean':.5,'p10':.3,'p90':.7}
    p.write_text(json.dumps(record))
    return p


def test_summary_uses_exact_historical_control_and_pairs_by_seed(tmp_path):
    out=tmp_path/'output'
    old=tmp_path/'old'
    stress=tmp_path/'stress'
    truth=np.array([0,1,2,3,4,5]*(2474//6)+[0,1])
    assert len(truth)==2474
    for seed in (42,43):
        new=truth.copy()
        new[0:3]=5
        lp3=truth.copy()
        lp3[0:10]=5
        lp4=truth.copy()
        lp4[0:5]=5
        for cond,root in [('clean',old),('snr0',stress)]:
            _make_eval(out/'evaluations','nahsf',seed,cond,new.tolist(),truth.tolist(),True)
            _make_eval(root,'lowpass3',seed,cond,lp3.tolist(),truth.tolist(),False)
            _make_eval(root,'lowpass4',seed,cond,lp4.tolist(),truth.tolist(),False)
    summary=v5.summarize(out,[42,43],[None,0],old,stress)
    assert len(summary['aggregates'])==4
    assert all(a['positive_seeds']==2 for a in summary['aggregates'])
    assert (out/'PAIRED_V4_COMPARISON.csv').exists()


def test_summary_rejects_different_window_truth(tmp_path):
    out=tmp_path/'output'
    old=tmp_path/'old'
    true=[0]*2474
    _make_eval(out/'evaluations','nahsf',42,'clean',true,true,True)
    _make_eval(old,'lowpass3',42,'clean',true,[1]*2474,False)
    with pytest.raises(ValueError,match='window order'):
        v5.summarize(out,[42],[None],old,tmp_path/'stress')


def test_evaluate_requires_original_or_strong_input(tmp_path):
    # Verify preflight guard by creating source-only syntactically valid checkpoint.
    ckpt_dir=tmp_path/'models'/'nahsf'/'seed42'
    ckpt_dir.mkdir(parents=True)
    torch.save({'state':NAHSFCNN().state_dict(),'code_family':v5.VERSION,
        'method':'nahsf','seed':42,'source_only':True,'raw_points':4096,
        'train_protocol':'V4_original_noise_10_to_40',
        'val_noise_levels_db':[10,20,30],'selected':{'epoch':1}},ckpt_dir/'best.pt')
    with pytest.raises(ValueError,match='strong-snr-root'):
        v5.evaluate(42,tmp_path,None,None,[0])
    with pytest.raises(ValueError,match='snr-root'):
        v5.evaluate(42,tmp_path,None,None,[10])
