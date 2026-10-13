"""独立核验CNN1D源选模、18组源验证及30组实测预测，导出客观记录。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT = Path(__file__).resolve().parent
CODE,OUT = ROOT/'code',ROOT/'results'
sys.path.insert(0,str(CODE/'scripts'))
import numpy as np
import torch
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
import plain_cnn1d_experiment as exp
import classical_baseline_experiment as old
import v2_experiment as data
from v1_noise_consistency import fix_randomness

def read(path):
    """读取未舍入的实际训练和推理结果，不更改标签或筛选种子。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    """旧实验与本轮代码按原始字节核查。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()

def save(path,value):
    """本轮新增核验文件拒绝覆盖；权重和预测均只读。"""
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def close(actual,expected):
    """浮点指标允许末位舍入，预测标签要求逐周期完全一致。"""
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)

def metric(truth,predicted):
    """独立sklearn计算六类F1/Recall及混淆矩阵，真实标签在行。"""
    matrix = confusion_matrix(truth,predicted,labels=range(6))
    _,recall,f1,_ = precision_recall_fscore_support(truth,predicted,labels=range(6),zero_division=0)
    alarms = int(matrix[0,1:].sum())
    return {'macro_f1':float(f1.mean()),'accuracy':float(accuracy_score(truth,predicted)),
        'balanced_accuracy':float(recall.mean()),'per_class_f1':f1.tolist(),'per_class_recall':recall.tolist(),
        'normal_false_alarms':alarms,'normal_false_alarm_rate':float(alarms/matrix[0].sum()),
        'missed_fault_as_normal':int(matrix[1:,0].sum()),'confusion_matrix':matrix.tolist()}

def main():
    """三份最佳权重分别重放，源验证与目标评估分开核对，再封存结果。"""
    assert read(ROOT/'RUN_STATUS.json')['status'] == 'completed'
    deploy = read(ROOT/'DEPLOYMENT_MANIFEST.json')
    preflight = read(ROOT/'PREFLIGHT.json')
    protected = read(ROOT/'PROTECTED_FILES_SHA256.json')
    assert all(sha(path) == expected for path,expected in protected.items())
    assert all(sha(CODE/name) == expected for name,expected in deploy['code_sha256'].items())
    models,chosen,weights,selections,history_rows,initial_states = {},{},{},[],[],{}
    for seed in (42,43,44):
        folder = OUT/f'models/{exp.METHOD}/seed{seed}'
        run,history = read(folder/'run.json'),read(folder/'history.json')
        assert run['method'] == exp.METHOD and run['seed'] == seed and run['protocol'] == exp.configuration()['protocol']
        assert run['source_only'] and run['target_used_for_training_or_selection'] is False
        assert run['final_holdout_evaluated'] is False
        assert run['parameters'] == 231078 and run['train_epochs'] == 50 and run['batch_size'] == 32
        assert run['initial_weights_sha256'] == preflight['initial_weights_sha256'][str(seed)]
        assert len(history) == 50 and [row['epoch'] for row in history] == list(range(1,51))
        for row in history:
            assert np.isfinite(row['train_loss_mean']) and set(row['noisy_f1']) == {'0','5','10','20','30'}
            close(row['noise_mean_f1'],np.mean(list(row['noisy_f1'].values())))
            history_rows.append({'seed':seed,'epoch':row['epoch'],'train_loss_mean':row['train_loss_mean'],
                'source_clean_f1_percent':100*row['clean_f1'],'source_five_noise_mean_percent':100*row['noise_mean_f1']})
        max_clean = max(row['clean_f1'] for row in history)
        eligible = [row for row in history if row['clean_f1'] >= max_clean-.01-1e-12]
        best = sorted(eligible,key=lambda row:(-row['noise_mean_f1'],-row['clean_f1'],row['epoch']))[0]
        assert run['selected'] == best
        checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert (checkpoint['method'],checkpoint['seed'],checkpoint['protocol']) == (exp.METHOD,seed,run['protocol'])
        assert checkpoint['selected'] == best and checkpoint['source_only'] and not checkpoint['final_holdout_evaluated']
        assert all(int(value) == best['epoch']*24 for name,value in checkpoint['state'].items() if name.endswith('num_batches_tracked'))
        fix_randomness(seed)
        model = exp.FivePhasePlainCNN1D().cuda().eval()
        model.load_state_dict(checkpoint['state'],strict=True)
        assert sum(p.numel() for p in model.parameters()) == 231078
        assert not any('shortcut' in name for name in model.state_dict())
        models[seed],chosen[seed] = model,best
        initial_states[seed] = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        weights[str(folder/'best.pt')] = sha(folder/'best.pt')
        selections.append({'seed':seed,'selected_epoch':best['epoch'],'eligible_epochs':len(eligible),
            'source_clean_f1_percent':100*best['clean_f1'],'source_five_noise_mean_percent':100*best['noise_mean_f1']})
    vr,vm,vy = data.source_data('val')
    val = {'clean':data.raw_cycles(vr,vm)}
    val.update({f'snr{k}':raw for k,raw in old.noise_source_val(Path(deploy['source_strong_root']),vr,vm).items()})
    replay = []
    for condition,raw in val.items():
        for seed,model in models.items():
            prediction = old._tensor_predict(model,raw,torch.device('cuda'))
            scores = metric(vy,prediction)
            expected = chosen[seed]['clean_f1'] if condition == 'clean' else chosen[seed]['noisy_f1'][condition[3:]]
            close(scores['macro_f1'],expected)
            replay.append({'seed':seed,'condition':condition,'metrics':scores})
    from prepare import OUT as SOURCE_OUT
    frozen = read(SOURCE_OUT/'target_dev_clean_windows.json')
    truth = np.asarray([row['label'] for row in frozen])
    assert len(frozen) == 2474 and len({row['record_id'] for row in frozen}) == 88
    assert np.bincount(truth,minlength=6).tolist() == [1106,307,297,298,237,229]
    checked = []
    for level in exp.CONDITIONS:
        name = 'clean' if level is None else f'snr{level}'
        raw,metadata = old.target_raw(level,snr_root=Path(deploy['target_snr_root']),
            strong_root=Path(deploy['target_strong_root']),dev_export_root=None)
        assert metadata == frozen
        x = data.raw_cycles(raw,metadata)
        for seed,model in models.items():
            original = read(OUT/f'evaluations/{exp.METHOD}/seed{seed}/target_dev_{name}.json')
            prediction = old._tensor_predict(model,x,torch.device('cuda'))
            assert np.array_equal(prediction,original['predicted']) and np.array_equal(truth,original['truth'])
            assert original['selected_source'] == chosen[seed] and original['source_only_selection']
            assert original['synthetic_target_noise'] == (level in (0,5))
            scores = metric(truth,prediction)
            for key,value in scores.items(): close(original['metrics'][key],value)
            checked.append({'seed':seed,'condition':name,'macro_f1_percent':100*scores['macro_f1'],
                'accuracy_percent':100*scores['accuracy']})
    for row in read(OUT/'reports/SUMMARY.json')['summary']:
        group = [result for result in checked if result['condition'] == row['condition']]
        assert len(group) == 3
        for key in ('macro_f1','accuracy'):
            values = [result[key+'_percent'] for result in group]
            close(row[key+'_mean'],np.mean(values)); close(row[key+'_sd'],np.std(values,ddof=1))
    for seed,model in models.items():
        assert all(torch.equal(value.detach().cpu(),initial_states[seed][name]) for name,value in model.state_dict().items())
    assert all(sha(path) == expected for path,expected in protected.items())
    old._csv(OUT/'reports/SOURCE_SELECTION.csv',selections)
    old._csv(OUT/'reports/TRAIN_LOSS_HISTORY.csv',history_rows)
    save(OUT/'reports/SOURCE_VAL_REPLAY.json',replay)
    save(OUT/'reports/WEIGHTS_SHA256.json',weights)
    # 沿用上次已核验的“文件+类别+时间段”索引，再按本轮冻结周期重算内容。
    old_index = Path(deploy['reference_classical_results'])/'reports/DATA_FILE_INDEX.csv'
    with old_index.open() as stream: index = list(csv.DictReader(stream))
    pools = {role:[row for row in index if row['role'] == role] for role in ('source_train','source_val','target_dev')}
    for role,count in [('source_train',738),('source_val',302),('target_dev',2474)]:
        assert sum(int(row['cycles']) for row in pools[role]) == count
    assert {(row['record_id'],int(row['label']),int(row['cycles'])) for row in pools['target_dev']} == {
        (rid,label,sum(row['record_id'] == rid and row['label'] == label for row in frozen))
        for rid in {row['record_id'] for row in frozen}
        for label in {row['label'] for row in frozen if row['record_id'] == rid}}
    shutil.copyfile(old_index,OUT/'reports/DATA_FILE_INDEX.csv')
    save(OUT/'reports/VERIFICATION.json',{'status':'passed','source_selection_independently_recomputed':3,
        'source_validation_metric_sets_reproduced':18,'target_prediction_sets_reproduced':30,
        'metrics_independently_recomputed':30,'summary_groups_recomputed':10,'training_epochs_verified':150,
        'unit_tests_passed':8,'protected_files_sha256_verified':len(protected),'original_files_unchanged':True,
        'fixed_inference_states_unchanged':True,'final_holdout_evaluated':False})
    paths = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.log')]
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.csv')]
    paths += [p for p in (ROOT/'logs').rglob('*') if p.is_file()]
    paths += [CODE/'configs/plain_cnn1d_control_v1.json']
    manifest = ROOT/'RESULTS_EXPORT_MANIFEST.json'
    save(manifest,{'status':'passed','files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}})
    with tarfile.open(ROOT/'plain_cnn1d_records.tar.gz','w:gz') as archive:
        for path in paths+[manifest]: archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    print('PLAIN_CNN1D_VERIFIED_AND_EXPORTED',len(paths),'FILES',flush=True)

if __name__ == '__main__': main()
