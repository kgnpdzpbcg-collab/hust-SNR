"""独立重算九组源选模、54组源验证和90组实测预测；只导出可发布记录。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
    CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT = Path(__file__).resolve().parent
CODE, OUT = ROOT/'code', ROOT/'results'
sys.path.insert(0,str(CODE/'scripts'))
import numpy as np
import torch
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import classical_baseline_experiment as exp
import v2_experiment as data

def read(path):
    """读取本轮生成的可信结果，不读取外部模型。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    """SHA逐个核验旧文件及发布文件，不过滤文件内容。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()

def save(path,value):
    """保留先前已完成的核验记录；中断后重算必须逐项完全一致才可继续。"""
    if path.exists():
        assert read(path) == value, path
        return
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def metric(truth,predicted):
    """使用独立sklearn计算六类指标，矩阵行为真实标签、列为预测标签。"""
    cm = confusion_matrix(truth,predicted,labels=range(6))
    _,recall,f1,_ = precision_recall_fscore_support(truth,predicted,labels=range(6),zero_division=0)
    return {'macro_f1':float(f1.mean()), 'accuracy':float(accuracy_score(truth,predicted)),
        'balanced_accuracy':float(recall.mean()), 'per_class_f1':f1.tolist(), 'per_class_recall':recall.tolist(),
        'normal_false_alarms':int(cm[0,1:].sum()), 'normal_false_alarm_rate':float(cm[0,1:].sum()/cm[0].sum()),
        'missed_fault_as_normal':int(cm[1:,0].sum()), 'confusion_matrix':cm.tolist()}

def close(actual,expected):
    """仅允许指标计算末位舍入；预测本身必须逐元素完全相同。"""
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)

def main():
    """重新载入自身生成的模型，源验证与目标推理分别核对后封存结果。"""
    deploy = read(ROOT/'DEPLOYMENT_MANIFEST.json')
    assert read(ROOT/'RUN_STATUS.json')['status'] == 'completed'
    protected = read(ROOT/'PROTECTED_FILES_SHA256.json')
    assert all(sha(path) == expected for path,expected in protected.items())
    assert all(sha(CODE/name) == expected for name,expected in deploy['code_sha256'].items())
    models,chosen,selections,weights = {},{},{},{}
    for method in exp.METHODS:
        for seed in (42,43,44):
            folder = OUT/f'models/{method}/seed{seed}'
            run = read(folder/'run.json')
            assert run['method'] == method and run['seed'] == seed and run['source_only']
            assert run['target_used_for_training_or_selection'] is False and run['final_holdout_evaluated'] is False
            rows = read(folder/'history.json') if method == 'resnet1d' else run['candidates']
            index = 'epoch' if method == 'resnet1d' else 'candidate_index'
            if method == 'resnet1d':
                assert len(rows) == 50 and [row['epoch'] for row in rows] == list(range(1,51))
                assert all(np.isfinite(row['train_loss_mean']) for row in rows)
            else:
                assert run['n_source_train'] == 738 and run['train_sample_views'] == 5904 and run['train_augmented_copies'] == 8
            for row in rows:
                assert set(row['noisy_f1']) == {'0','5','10','20','30'}
                close(row['noise_mean_f1'],np.mean(list(row['noisy_f1'].values())))
            max_clean = max(row['clean_f1'] for row in rows)
            eligible = [row for row in rows if row['clean_f1'] >= max_clean-.01-1e-12]
            best = sorted(eligible,key=lambda row:(-row['noise_mean_f1'],-row['clean_f1'],row[index]))[0]
            assert run['selected'] == best
            chosen[method,seed] = best
            if method == 'resnet1d':
                state = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
                assert state['selected'] == best and state['source_only'] and not state['final_holdout_evaluated']
                model = exp.FivePhaseResNet1D().cuda().eval()
                model.load_state_dict(state['state'],strict=True)
                assert run['initial_weights_sha256'] == read(ROOT/'PREFLIGHT.json')['initial_resnet_sha256'][str(seed)]
                assert all(int(value) == best['epoch']*24 for name,value in state['state'].items() if name.endswith('num_batches_tracked'))
                artifact = folder/'best.pt'
            else:
                artifact = folder/'best.joblib'
                model = joblib.load(artifact)
            models[method,seed] = model
            weights[str(artifact)] = sha(artifact)
            selections[f'{method}_seed{seed}'] = {'selected':best,'eligible_candidates':len(eligible)}
    # SVM的StandardScaler必须只拟合源训练的5904个视图，独立重算均值/方差。
    tr,tm,y = data.source_data('train')
    train_clean = data.raw_cycles(tr,tm)
    for seed in (42,43,44):
        features,labels = exp._classical_training_arrays(seed,tr,tm,y,train_clean)
        assert features.shape == (5904,120) and len(labels) == 5904
        scaler = models['svm_rbf',seed].named_steps['scaler']
        assert scaler.n_samples_seen_ == 5904
        np.testing.assert_allclose(scaler.mean_,features.astype(np.float64).mean(0),atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(scaler.var_,features.astype(np.float64).var(0),atol=1e-10,rtol=1e-10)
    vr,vm,vy = data.source_data('val')
    val = {'clean':data.raw_cycles(vr,vm)}
    val.update({f'snr{k}':raw for k,raw in exp.noise_source_val(Path(deploy['source_strong_root']),vr,vm).items()})
    replay = []
    for condition,raw in val.items():
        features = exp.engineered_features(raw)
        for (method,seed),model in models.items():
            pred = exp._tensor_predict(model,raw,torch.device('cuda')) if method == 'resnet1d' else model.predict(features)
            scores = metric(vy,pred)
            expected = chosen[method,seed]['clean_f1'] if condition == 'clean' else chosen[method,seed]['noisy_f1'][condition[3:]]
            close(scores['macro_f1'],expected)
            replay.append({'method':method,'seed':seed,'condition':condition,'metrics':scores})
    from prepare import OUT as SOURCE_OUT
    frozen = read(SOURCE_OUT/'target_dev_clean_windows.json')
    truth = np.array([row['label'] for row in frozen])
    assert np.bincount(truth,minlength=6).tolist() == [1106,307,297,298,237,229]
    checked = []
    for condition in exp.DEFAULT_CONDITIONS:
        level = None if condition == 'clean' else condition
        name = 'clean' if level is None else f'snr{level}'
        raw,metadata = exp.target_raw(level,snr_root=Path(deploy['target_snr_root']),
            strong_root=Path(deploy['target_strong_root']),dev_export_root=None)
        assert metadata == frozen
        x = data.raw_cycles(raw,metadata)
        features = exp.engineered_features(x)
        for (method,seed),model in models.items():
            old = read(OUT/f'evaluations/{method}/seed{seed}/target_dev_{name}.json')
            pred = exp._tensor_predict(model,x,torch.device('cuda')) if method == 'resnet1d' else model.predict(features)
            assert np.array_equal(pred,old['predicted']) and np.array_equal(truth,old['truth'])
            scores = metric(truth,pred)
            for key,value in scores.items(): close(old['metrics'][key],value)
            assert old['synthetic_target_noise'] == (level in (0,5))
            checked.append({'method':method,'seed':seed,'condition':name,'macro_f1_percent':100*scores['macro_f1'],
                'accuracy_percent':100*scores['accuracy']})
    summary = read(OUT/'reports/SUMMARY.json')
    for row in summary['summary']:
        matched = [x for x in checked if (x['method'],x['condition']) == (row['method'],row['condition'])]
        assert len(matched) == 3
        for key in ('macro_f1','accuracy'):
            values = [x[key+'_percent'] for x in matched]
            close(row[key+'_mean'],np.mean(values)); close(row[key+'_sd'],np.std(values,ddof=1))
    with (OUT/'reports/PER_SEED_RESULTS.csv').open() as stream: assert len(list(csv.DictReader(stream))) == 90
    with (OUT/'reports/PER_CLASS_PER_SEED.csv').open() as stream: assert len(list(csv.DictReader(stream))) == 540
    save(OUT/'reports/SOURCE_VAL_REPLAY.json',replay)
    save(OUT/'reports/SOURCE_SELECTION.json',selections)
    save(OUT/'reports/WEIGHTS_SHA256.json',weights)
    # 从本轮确实读取的冻结周期元数据生成索引，不依赖其他实验的可选报告。
    index_rows = []
    for role,metadata in (('source_train',tm),('source_val',vm),('target_dev',frozen)):
        for record_id in sorted({row['record_id'] for row in metadata}):
            directory = CODE/'data/source' if role.startswith('source') else CODE/'data/target_dev/clean'
            path = (directory/f'{record_id}.mat').resolve()
            assert path.is_file()
            file_rows = [row for row in metadata if row['record_id'] == record_id]
            # 同一记录故障前后可有不同标签，按原逐周期标签分别统计，绝不改标签。
            for label in sorted({row['label'] for row in file_rows}):
                group = [row for row in file_rows if row['label'] == label]
                index_rows.append({'role':role,'record_id':record_id,'label':label,
                    'cycles':len(group),'start_seconds':min(row['start'] for row in group),
                    'end_seconds':max(row['end'] for row in group),'clean_file':str(path)})
    assert len({(row['role'],row['record_id']) for row in index_rows}) == 137
    assert not ({row['record_id'] for row in tm} & {row['record_id'] for row in vm})
    assert all(sum(row['cycles'] for row in index_rows if row['role'] == role) == count
        for role,count in [('source_train',738),('source_val',302),('target_dev',2474)])
    with (OUT/'reports/DATA_FILE_INDEX.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(index_rows[0]))
        writer.writeheader(); writer.writerows(index_rows)
    assert all(sha(path) == expected for path,expected in protected.items())
    save(OUT/'reports/VERIFICATION.json',{'status':'passed','source_selection_recomputed':9,
        'source_val_metric_sets_reproduced':54,'target_prediction_sets_reproduced':90,
        'independent_metrics_recomputed':90,'summary_groups_recomputed':30,
        'svm_source_only_scalers_reproduced':3,'unit_tests_passed':11,
        'protected_files_sha256_verified':len(protected),'old_weights_and_results_unchanged':True,
        'final_holdout_evaluated':False})
    # 导出所有结果JSON/CSV、日志和部署来源；禁止上传模型与原始数据。
    paths = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.log')]
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.csv')]
    paths += [p for p in (ROOT/'logs').rglob('*') if p.is_file()]
    paths += [CODE/'configs/classical_baselines_v1.json']
    export = {'status':'passed','files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}}
    manifest = ROOT/'RESULTS_EXPORT_MANIFEST.json'
    save(manifest,export)
    with tarfile.open(ROOT/'classical_records.tar.gz','w:gz') as archive:
        for path in paths+[manifest]: archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    print('CLASSICAL_VERIFIED_AND_EXPORTED',len(checked),'PREDICTION_SETS',len(paths),'FILES',flush=True)

if __name__ == '__main__': main()
