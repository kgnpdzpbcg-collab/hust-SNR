"""选定V5展示种子的七模型编码特征：补0/5 dB并加入原ResNet。

仅展示种子按已有V5 Dev分数挑选，模型仍使用原源验证最佳权重。
训练正式汇总保留全部三种子；本可视化不能代表种子平均或最终测试集。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
from six_model_tsne import read, write, sha, save_csv, project_one

CONFIG = Path(__file__).resolve().parents[1]/'configs/seven_model_tsne_v1.json'

def select_seed(four_results):
    """只决定展示种子：已有V5四条件等权平均最高，并列取较小种子。"""
    candidates = []
    for seed in read(CONFIG)['seed_candidates']:
        scores = {condition: read(four_results/f'evaluations/nahsf/seed{seed}/target_dev_{condition}.json')
            ['metrics']['macro_f1'] for condition in read(CONFIG)['conditions']}
        candidates.append({'seed': seed, 'scores': scores, 'mean_macro_f1': float(np.mean(list(scores.values())))})
    return max(candidates, key=lambda row: (row['mean_macro_f1'], -row['seed'])), candidates

def extract(base, classical_results, output):
    """读取固定Dev和原最佳权重，钩取编码器表示，并验证原分类头预测。"""
    import torch
    import four_models_expanded_val as common
    import fixed_half_experiment as fixed
    import xu2025_full_experiment as xu
    import classical_baseline_experiment as classical
    import v2_experiment as data
    from prepare import OUT
    from v1_noise_consistency import fix_randomness
    protocol = read(CONFIG)
    assert protocol['tsne'] == read(Path(__file__).resolve().parents[1]/'configs/six_model_tsne_v1.json')['tsne']
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    four = base/'experiments/four_models_expanded_val/56513c7/results'
    previous = base/'experiments/six_model_tsne/f398d30/results'
    roots = {method: four for method in common.METHODS}
    roots.update(fixed_half=base/'experiments/fixed_half_fusion/3c38037/results',
        xu_full=base/'experiments/xu2025_full/bb2f219/results/matched', resnet1d=classical_results)
    assert read(previous/'VERIFICATION.json')['status'] == 'passed'
    assert read(classical_results/'reports/VERIFICATION.json')['status'] == 'passed'
    protected = read(classical_results.parent/'PROTECTED_FILES_SHA256.json')
    protected.update({str(p): sha(p) for p in classical_results.rglob('*') if p.is_file()})
    assert all(sha(path) == expected for path, expected in protected.items())
    write(output/'PROTECTED_FILES_SHA256.json', protected)
    chosen, candidates = select_seed(four)
    seed = chosen['seed']
    write(output/'DISPLAY_SEED_SELECTION.json', {'criterion': protocol['seed_selection'],
        'candidates': candidates, 'selected_seed': seed, 'posthoc_target_dev_selection': True,
        'used_for_training_or_checkpoint_selection': False})
    metadata = read(OUT/'target_dev_clean_windows.json')
    truth = np.array([row['label'] for row in metadata], dtype=np.int64)
    ids = np.array([row['record_id'] for row in metadata])
    assert len(truth) == 2474 and len(set(ids)) == 88
    old_entries = read(previous/'FEATURE_MANIFEST.json')['files']
    entries = []
    for condition in protocol['conditions']:
        reused_methods = set(protocol['models'][:-1]) if condition in ('clean','snr10') else set()
        # 旧六模型Clean/10 dB的特征、坐标均已核验；原样引用，禁止重拟合挑图。
        for method in protocol['models']:
            if method not in reused_methods: continue
            old = next(row for row in old_entries if (row['model'],row['seed'],row['condition'],row['layer'])
                == (method,seed,condition,'encoder352'))
            assert sha(old['file']) == old['feature_file_sha256']
            entry = copy.deepcopy(old)
            entry.update(layer='encoder', reused_projection=True,
                previous_metric=str(previous/f'metrics/{method}_seed{seed}_{condition}_encoder352.json'))
            entries.append(entry)
        level = None if condition == 'clean' else int(condition[3:])
        source = base/'datasets/strong_snr_v1_20261012/generated/target' if level in (0,5) else four.parent/'target_snr'
        raw, actual = data.target_dev_from_mat(level, source)
        assert actual == metadata
        x = data.raw_cycles(raw, metadata)
        for method in protocol['models']:
            if method in reused_methods: continue
            fix_randomness(seed)
            folder = roots[method]/f'models/{method}/seed{seed}'
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            assert checkpoint['seed'] == seed and checkpoint['source_only'] and not checkpoint['final_holdout_evaluated']
            if method == 'resnet1d':
                assert checkpoint['method'] == method and checkpoint['protocol'] == classical.config()['protocol']
                model = classical.FivePhaseResNet1D().cuda().eval()
                head = first = model.classifier
            elif method == 'xu_full':
                assert checkpoint['track'] == 'matched' and checkpoint['mode'] == method
                model = xu.make_model(method,'matched').cuda().eval()
                head = model.network.classifier
                first = head[0]
            else:
                if method == 'fixed_half':
                    common.check_checkpoint(checkpoint,method,seed,protocol=fixed.config(),version=fixed.VERSION)
                    model = fixed.make_model(method).cuda().eval()
                else:
                    common.check_checkpoint(checkpoint,method,seed)
                    model = common.make_model(method).cuda().eval()
                first = model.backbone.classifier[0]
                head = model.backbone.classifier[1:]
            model.load_state_dict(checkpoint['state'], strict=True)
            before = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
            parts = []
            def capture(_module, inputs):
                """只观察分类前编码特征，不改变网络输出或缓冲区。"""
                parts.append(inputs[0].detach().flatten(1).cpu().numpy().copy())
            hook = first.register_forward_pre_hook(capture)
            predicted = []
            with torch.no_grad():
                for start in range(0,len(x),128):
                    result = model(torch.from_numpy(x[start:start+128]).cuda())
                    logits = result['logits'] if method == 'xu_full' else result
                    predicted.append(logits.argmax(1).cpu().numpy())
            hook.remove()
            predicted = np.concatenate(predicted)
            features = np.concatenate(parts)
            assert features.shape == (2474,protocol['feature_dimensions'][method]) and np.isfinite(features).all()
            original_path = roots[method]/f'evaluations/{method}/seed{seed}/target_dev_{condition}.json'
            original = read(original_path)
            assert np.array_equal(truth,original['truth']) and np.array_equal(predicted,original['predicted'])
            replay = []
            with torch.no_grad():
                for start in range(0,len(features),128):
                    replay.append(head(torch.from_numpy(features[start:start+128]).cuda()).argmax(1).cpu().numpy())
            assert np.array_equal(predicted,np.concatenate(replay))
            assert all(torch.equal(value.detach().cpu(),before[key]) for key,value in model.state_dict().items())
            path = output/'features'/f'{method}_seed{seed}_{condition}_encoder.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,features=features,truth=truth,predicted=predicted,record_ids=ids)
            entries.append({'model':method,'seed':seed,'condition':condition,'layer':'encoder',
                'feature_dim':features.shape[1],'file':str(path),'feature_file_sha256':sha(path),
                'feature_array_sha256':hashlib.sha256(features.tobytes()).hexdigest(),
                'weights':str(folder/'best.pt'),'weights_sha256':sha(folder/'best.pt'),
                'original_prediction_file':str(original_path),'prediction_file_sha256':sha(original_path),
                'macro_f1':original['metrics']['macro_f1'],'selected_epoch':checkpoint['selected']['epoch'],
                'captured_head_predictions_exact':True,'reused_projection':False})
            del model
            print('FEATURE_EXTRACTED',method,seed,condition,flush=True)
    assert len(entries) == 28 and sum(row['reused_projection'] for row in entries) == 12
    assert all(sha(path) == expected for path,expected in protected.items())
    write(output/'FEATURE_MANIFEST.json', {'protocol':protocol,'selected_seed':seed,'files':entries,
        'class_counts':np.bincount(truth,minlength=6).tolist(),'old_results_and_weights_unchanged':True})

def project(output):
    """复用12组旧投影，只计算16组新投影；同一固定参数，不按效果重算。"""
    if (output/'coordinates').exists(): raise FileExistsError('拒绝覆盖投影')
    manifest = read(output/'FEATURE_MANIFEST.json')
    protocol = read(CONFIG)
    assert manifest['protocol'] == protocol
    results = []
    started = time.monotonic()
    for entry in manifest['files']:
        if not entry['reused_projection']: continue
        old = read(entry['previous_metric'])
        source = Path(old['coordinate_file'])
        assert sha(source) == old['coordinate_file_sha256']
        target = output/'coordinates'/f"{entry['model']}_seed{entry['seed']}_{entry['condition']}_encoder.csv"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,target)
        record = {**old,'layer':'encoder','coordinate_file':str(target),'reused_projection':True,
            'previous_metric_file':entry['previous_metric']}
        write(output/'metrics'/(target.stem+'.json'),record)
        results.append(record)
    tasks = [(entry,str(output),protocol) for entry in manifest['files'] if not entry['reused_projection']]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(project_one,task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            results.append(record)
            print('PROJECTION_COMPLETED',len(results),'/28',record['model'],record['condition'],flush=True)
    results.sort(key=lambda row:(protocol['conditions'].index(row['condition']),protocol['models'].index(row['model'])))
    fields = ('model','seed','condition','feature_dim','silhouette_macro','cross_record_5nn_purity_macro',
        'tsne_trustworthiness_k15','classifier_macro_f1_percent')
    save_csv(output/'FEATURE_SEPARABILITY.csv',[{key:row[key] for key in fields} for row in results])
    write(output/'SUMMARY.json',{'protocol':protocol,'selected_seed':manifest['selected_seed'],'results':results,
        'reused_projections':12,'new_projections':16,'wall_seconds':time.monotonic()-started,
        'illustration_seed_selected_posthoc':True,'final_holdout_evaluated':False})
    protected = read(output/'PROTECTED_FILES_SHA256.json')
    assert all(sha(path) == expected for path,expected in protected.items())
    print('SEVEN_MODEL_TSNE_COMPLETED',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('extract','project'))
    parser.add_argument('--base',type=Path)
    parser.add_argument('--classical-results',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.stage == 'extract':
        if args.base is None or args.classical_results is None: parser.error('需要既有项目及基线结果目录')
        extract(args.base,args.classical_results,args.output)
    else: project(args.output)
