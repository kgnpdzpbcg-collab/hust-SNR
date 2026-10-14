"""独立核验九模型1350轮记录、27组源选模、162组源验证及270组目标预测。"""
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
import hust_arch_suite_experiment as exp

def read(path):
    """读取未舍入的正式数据，不排除低分种子或改写旧结果。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    """以流式SHA核对模型、代码、输入与发布文件的原字节。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()

def close(actual,expected):
    """独立指标只允许末位舍入；预测向量必须逐周期完全相同。"""
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)

def metric(truth,predicted):
    """由sklearn独立计算六类指标和混淆矩阵，行真实列预测。"""
    matrix = confusion_matrix(truth,predicted,labels=range(6))
    _,recall,f1,_ = precision_recall_fscore_support(truth,predicted,labels=range(6),zero_division=0)
    alarms = int(matrix[0,1:].sum())
    return {'macro_f1':float(f1.mean()),'accuracy':float(accuracy_score(truth,predicted)),
        'balanced_accuracy':float(recall.mean()),'per_class_f1':f1.tolist(),'per_class_recall':recall.tolist(),
        'normal_false_alarms':alarms,'normal_false_alarm_rate':float(alarms/matrix[0].sum()),
        'missed_fault_as_normal':int(matrix[1:,0].sum()),'confusion_matrix':matrix.tolist()}

def main():
    """全部27组完成后统一核验，模型重放及汇总通过才导出结果。"""
    assert read(ROOT/'RUN_STATUS.json')['status'] == 'completed'
    deploy,preflight = read(ROOT/'DEPLOYMENT_MANIFEST.json'),read(ROOT/'PREFLIGHT.json')
    protected = read(ROOT/'PROTECTED_FILES_SHA256.json')
    assert all(sha(path) == expected for path,expected in protected.items())
    assert all(sha(CODE/name) == expected for name,expected in deploy['code_sha256'].items())
    models,chosen,weights,before,selections,loss_rows = {},{},{},{},[],[]
    for method in exp.METHODS:
        for seed in exp.SEEDS:
            folder = OUT/f'models/{method}/seed{seed}'
            run,history = read(folder/'run.json'),read(folder/'history.json')
            assert (run['method'],run['seed'],run['protocol']) == (method,seed,exp.PROTOCOL)
            assert run['training'] == exp.configuration() and run['model_init_from_scratch'] and run['source_only']
            assert not run['target_used_for_training_or_selection'] and not run['final_holdout_evaluated']
            assert [run[key] for key in ('source_train_records','source_train_cycles','source_val_records','source_val_cycles')] == [34,738,15,302]
            assert run['parameters'] == preflight['parameter_counts'][method]
            assert run['initial_parameters_sha256'] == preflight['initial_parameters_sha256'][str(seed)][method]
            assert len(history) == 50 and [row['epoch'] for row in history] == list(range(1,51))
            for row in history:
                assert np.isfinite(row['train_loss_mean']) and row['train_loss_mean'] >= 0
                assert set(row['noisy_f1']) == {'0','5','10','20','30'}
                assert all(0 <= value <= 1 for value in [row['clean_f1'],*row['noisy_f1'].values()])
                close(row['noise_mean_f1'],np.mean(list(row['noisy_f1'].values())))
                loss_rows.append({'method':method,'seed':seed,'epoch':row['epoch'],
                    'train_loss_mean':row['train_loss_mean'],'source_clean_f1_percent':100*row['clean_f1'],
                    'source_five_noise_mean_percent':100*row['noise_mean_f1']})
            max_clean = max(row['clean_f1'] for row in history)
            eligible = [row for row in history if row['clean_f1'] >= max_clean-.01-1e-12]
            best = sorted(eligible,key=lambda row:(-row['noise_mean_f1'],-row['clean_f1'],row['epoch']))[0]
            assert run['selected'] == best
            checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            assert (checkpoint['method'],checkpoint['seed'],checkpoint['protocol']) == (method,seed,exp.PROTOCOL)
            assert checkpoint['selected'] == best and checkpoint['source_only'] and not checkpoint['final_holdout_evaluated']
            assert all(int(value) == best['epoch']*24 for name,value in checkpoint['state'].items()
                if name.endswith('num_batches_tracked'))
            # 正式evaluate关闭TF32并启用确定性算法；核验也必须调用同一原函数。
            # 默认CUDA卷积精度可能改变边界样本argmax，不能用默认环境替代原推理。
            exp.fix_randomness(seed)
            model = exp.HUSTArchitectureCandidate(method).cuda().eval()
            model.load_state_dict(checkpoint['state'],strict=True)
            assert exp.parameter_count(model) == run['parameters']
            models[method,seed],chosen[method,seed] = model,best
            before[method,seed] = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            weights[str(folder/'best.pt')] = sha(folder/'best.pt')
            selections.append({'method':method,'seed':seed,'selected_epoch':best['epoch'],
                'eligible_epochs':len(eligible),'source_clean_f1_percent':100*best['clean_f1'],
                'source_five_noise_mean_percent':100*best['noise_mean_f1']})
    (_,tm,_,_),(_,vm,vy,val_clean,val_noise) = exp.source_views(Path(deploy['source_strong_root']))
    val = {'clean':val_clean,**{f'snr{k}':raw for k,raw in val_noise.items()}}
    source_replay = []
    for condition,raw in val.items():
        for (method,seed),model in models.items():
            predicted = exp.infer(model,raw,torch.device('cuda'))
            score = metric(vy,predicted)
            expected = chosen[method,seed]['clean_f1'] if condition == 'clean' else chosen[method,seed]['noisy_f1'][condition[3:]]
            close(score['macro_f1'],expected)
            source_replay.append({'method':method,'seed':seed,'condition':condition,'metrics':score})
    print('SOURCE_VALIDATION_REPLAYED',len(source_replay),'SETS',flush=True)
    from prepare import OUT as PREPARE_OUT
    frozen = read(PREPARE_OUT/'target_dev_clean_windows.json')
    truth = np.asarray([row['label'] for row in frozen])
    assert len(frozen) == 2474 and len({row['record_id'] for row in frozen}) == 88
    assert np.bincount(truth,minlength=6).tolist() == [1106,307,297,298,237,229]
    checked = []
    for level in exp.CONDITIONS:
        name = 'clean' if level is None else f'snr{level}'
        raw,metadata = exp.target_raw(level,snr_root=Path(deploy['target_snr_root']),
            strong_root=Path(deploy['target_strong_root']),dev_export_root=None)
        assert metadata == frozen
        x = exp.data.raw_cycles(raw,metadata)
        assert x.shape == (2474,5,4096)
        for (method,seed),model in models.items():
            original = read(OUT/f'evaluations/{method}/seed{seed}/target_dev_{name}.json')
            predicted = exp.infer(model,x,torch.device('cuda'))
            assert np.array_equal(truth,original['truth']) and np.array_equal(predicted,original['predicted'])
            assert original['selected_source'] == chosen[method,seed] and original['source_only_selection']
            assert original['synthetic_target_noise'] == (level in (0,5)) and not original['final_holdout_evaluated']
            score = metric(truth,predicted)
            for key,value in score.items(): close(original['metrics'][key],value)
            checked.append({'method':method,'seed':seed,'condition':name,
                'macro_f1_percent':100*score['macro_f1'],'accuracy_percent':100*score['accuracy']})
        print('TARGET_PREDICTIONS_REPLAYED',name,'27 models',flush=True)
    summary = read(OUT/'reports/SUMMARY.json')
    assert len(summary['summary']) == 90 and summary['seeds'] == [42,43,44] and summary['methods'] == list(exp.METHODS)
    for row in summary['summary']:
        group = [result for result in checked if (result['method'],result['condition']) == (row['method'],row['condition'])]
        assert len(group) == 3
        for key in ('macro_f1','accuracy'):
            values = [result[key+'_percent'] for result in group]
            close(row[key+'_mean'],np.mean(values)); close(row[key+'_sd'],np.std(values,ddof=1))
    with (OUT/'reports/PER_SEED_RESULTS.csv').open() as stream: assert len(list(csv.DictReader(stream))) == 270
    with (OUT/'reports/PER_CLASS_PER_SEED.csv').open() as stream: assert len(list(csv.DictReader(stream))) == 1620
    for key,model in models.items():
        assert all(torch.equal(value.detach().cpu(),before[key][name]) for name,value in model.state_dict().items())
    assert all(sha(path) == expected for path,expected in protected.items())
    exp.write_csv(OUT/'reports/SOURCE_SELECTION.csv',selections)
    exp.write_csv(OUT/'reports/TRAIN_LOSS_HISTORY.csv',loss_rows)
    exp.write_json(OUT/'reports/SOURCE_VAL_REPLAY.json',source_replay)
    exp.write_json(OUT/'reports/WEIGHTS_SHA256.json',weights)
    # 用本轮冻结元数据重建索引：同文件的健康/故障时间段分别统计，避免单文件单标签误述。
    index = []
    for role,metadata in [('source_train',tm),('source_val',vm),('target_dev',frozen)]:
        for record_id in sorted({row['record_id'] for row in metadata}):
            directory = CODE/'data/source' if role.startswith('source') else CODE/'data/target_dev/clean'
            path = (directory/f'{record_id}.mat').resolve()
            assert path.is_file()
            samples = [row for row in metadata if row['record_id'] == record_id]
            for label in sorted({row['label'] for row in samples}):
                group = [row for row in samples if row['label'] == label]
                index.append({'role':role,'record_id':record_id,'label':label,'cycles':len(group),
                    'start_seconds':min(row['start'] for row in group),
                    'end_seconds':max(row['end'] for row in group),'clean_file':str(path)})
    assert len({(row['role'],row['record_id']) for row in index}) == 137
    assert not ({row['record_id'] for row in tm} & {row['record_id'] for row in vm})
    exp.write_csv(OUT/'reports/DATA_FILE_INDEX.csv',index)
    exp.write_json(OUT/'reports/VERIFICATION.json',{'status':'passed','models_verified':27,
        'training_epochs_verified':1350,'source_selections_independently_recomputed':27,
        'source_validation_sets_reproduced':162,'target_prediction_sets_reproduced':270,
        'independent_target_metric_sets_recomputed':270,'summary_groups_recomputed':90,
        'target_per_class_rows_verified':1620,'unit_tests_passed':25,
        'protected_files_sha256_verified':len(protected),'old_files_unchanged':True,
        'inference_model_states_unchanged':True,'original_deterministic_tf32_settings_reused':True,
        'verification_settings_correction_record':'../../REPLAY_SETTINGS_DIAGNOSIS.json',
        'final_holdout_evaluated':False})
    # 仅导出指标、日志和配置；权重/原始电流/高维数组均留在服务器。
    paths = [path for path in ROOT.iterdir() if path.is_file() and path.suffix in ('.py','.json','.log')]
    paths += [path for path in OUT.rglob('*') if path.is_file() and path.suffix in ('.json','.csv')]
    paths += [path for path in (ROOT/'logs').rglob('*') if path.is_file()]
    paths += [CODE/'configs/hust_arch_suite_v1.json']
    manifest = ROOT/'RESULTS_EXPORT_MANIFEST.json'
    exp.write_json(manifest,{'status':'passed','files':{path.relative_to(ROOT).as_posix():
        {'bytes':path.stat().st_size,'sha256':sha(path)} for path in paths}})
    with tarfile.open(ROOT/'architecture_suite_records.tar.gz','w:gz') as archive:
        for path in paths+[manifest]: archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    print('NINE_ARM_VERIFIED_AND_EXPORTED',len(paths),'FILES',flush=True)

if __name__ == '__main__': main()
