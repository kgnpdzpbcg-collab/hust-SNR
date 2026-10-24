"""论文V3.13的347记录Final-Holdout固定权重评价，独立于历史Dev入口。

流程：冻结论文范围及24份best -> 在旧Dev回放验证读取与模型版本 ->
按原规则制作347记录的同一套周期 -> 三组独立进程执行固定权重推理 ->
独立重算指标、核对预测重放及受保护文件。原算法、划分、权重和历史
记录均不改写；A2直接加载已保存的源统计缓冲区，不用目标域重新标定。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import zipfile

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')


def read(path):
    """读取明确的冻结元数据，不依据测试结果寻找其他模型版本。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """只向本轮独立目录保存UTF-8记录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def sha(path):
    """流式计算字节哈希，避免缓存大数组时产生额外内存副本。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    """保存完整逐种子指标，列顺序由首条记录确定。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def context(config_path):
    """服务器路径由既有项目定位，输出固定在独立实验目录。"""
    project = next(Path('/home/wang/projects').glob('*/hust'))
    config = read(config_path)
    output = project / 'experiments/final_holdout_v313/20261024'
    roots = {k: project / 'experiments' / v for k, v in config['families'].items()}
    return project, config, output, roots


def assert_protected(output):
    """每个阶段核对冻结代码和权重，防止推理过程中发生隐式更新。"""
    plan = read(output / 'FROZEN_PLAN.json')
    assert sha(Path(__file__)) == plan['script_sha256'], '本轮评价脚本在冻结后改变'
    for path, digest in plan['protected_files'].items():
        assert sha(path) == digest, path
    return plan


def frozen_models(config, family):
    return [m for m in config['models'] if m['family'] == family]


def family_api(family, roots):
    """各归档在独立子进程中导入，避免同名历史模块相互覆盖。"""
    sys.path.insert(0, str(roots[family] / 'code/scripts'))
    module = {'reference': 'source_cycle_benchmark', 'ablation': 'b1_paper_ablations',
              'jung': 'compare_jung_hang_v4'}[family]
    api = importlib.import_module(module)
    jung_config = read(roots['jung'] / 'code/configs/jung_hang_cycle_v4_v1.json') if family == 'jung' else None
    return api, jung_config


def score(y, pred):
    """显式固定六类顺序，与既有Macro-F1定义一致。"""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=list(range(6)), zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1.mean()),
            'confusion_matrix': confusion_matrix(y, pred, labels=list(range(6))).tolist(),
            'precision_by_class': precision.tolist(), 'recall_by_class': recall.tolist(),
            'f1_by_class': f1.tolist(), 'support_by_class': support.tolist()}


def predict(model, inputs, batch):
    """固定eval推理；徐模型只取logits；输入批次复制后上GPU，不改共享缓存。"""
    import numpy as np
    import torch
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch):
            tensor = torch.from_numpy(np.array(inputs[start:start + batch], copy=True)).cuda()
            value = model(tensor)
            if isinstance(value, dict):
                value = value['logits']
            assert value.shape == (len(tensor), 6) and torch.isfinite(value).all()
            outputs.append(value.cpu().numpy())
    return np.concatenate(outputs)


def load_models(config, roots, family):
    """严格加载每个种子的既有best，不重新选择轮次或拟合任何目标域参数。"""
    import torch
    api, jung_config = family_api(family, roots)
    from v1_noise_consistency import fix_randomness
    loaded = []
    for item in frozen_models(config, family):
        for seed in config['seeds']:
            folder = roots[family] / f"results/models/{item['method']}/seed{seed}"
            checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            run = read(folder / 'run.json')
            assert checkpoint['method'] == run['method'] == item['method']
            assert checkpoint['seed'] == run['seed'] == seed
            assert checkpoint['selected'] == run['selected']
            assert not checkpoint['final_holdout_evaluated']
            assert not run['target_used_for_training_or_selection']
            assert run['config']['source_split_manifest_sha256'] == config['source_split_sha256']
            fix_randomness(seed)
            model = (api.get_model('jung', jung_config) if family == 'jung' else api.build_model(item['method'])).cuda().eval()
            model.load_state_dict(checkpoint['state'], strict=True)
            loaded.append((item, seed, model, checkpoint['selected'], folder / 'best.pt'))
    return loaded


def freeze(config_path):
    """在首次读取Holdout波形前冻结范围、原代码、既有权重及Dev回放输入。"""
    import numpy as np
    import torch
    project, config, output, roots = context(config_path)
    assert not output.exists(), '拒绝覆盖已存在的正式Holdout实验目录'
    assert config['seeds'] == [42, 44] and config['conditions_db'] == [5, 0, -5, -10]
    assert len(config['models']) == 12 and not config['model_training'] and not config['checkpoint_selection']
    metadata_root = project / 'datasets/strong_snr_v1_20261012/metadata'
    split_path = metadata_root / 'splits_gate0_v1.json'
    assert sha(split_path) == config['split_sha256']
    split = read(split_path)
    holdout, dev = sorted(split['target_holdout_records']), set(split['target_dev_records'])
    assert len(holdout) == len(set(holdout)) == 347 and len(dev) == 88 and not set(holdout) & dev
    assert set(holdout) | dev == set(split['target_canonical_records'])
    for group in split['target_duplicate_groups']:
        assert not (set(group) & set(holdout) and set(group) & dev)
    protected = {str(split_path): sha(split_path), str(config_path): sha(config_path)}
    weights = []
    for family, root in roots.items():
        deployment = read(root / 'DEPLOYMENT_MANIFEST.json')
        for relative, expected in deployment['code_sha256'].items():
            path = root / 'code' / relative
            assert sha(path) == expected, path
            protected[str(path)] = expected
        for item in frozen_models(config, family):
            for seed in config['seeds']:
                folder = root / f"results/models/{item['method']}/seed{seed}"
                for name in ('best.pt', 'last.pt', 'run.json', 'history.json'):
                    path = folder / name
                    assert path.is_file(), path
                    protected[str(path)] = sha(path)
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
                assert checkpoint['method'] == item['method'] and checkpoint['seed'] == seed
                weights.append({**item, 'seed': seed, 'checkpoint_path': str(folder / 'best.pt'),
                                'sha256': protected[str(folder / 'best.pt')], 'selected_epoch': checkpoint['selected']['epoch']})
    assert len(weights) == 24
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    (output / 'cache').mkdir()
    write(output / 'FROZEN_PLAN.json', {'config': config, 'weights': weights, 'holdout_records': holdout,
          'protected_files': protected, 'frozen_before_holdout_waveform_read': True,
          'holdout_waveforms_read_at_freeze': False, 'script_sha256': sha(Path(__file__))})
    # 仅使用原88条Dev制作回放输入。通过回放后才创建Holdout周期。
    sys.path.insert(0, str(roots['reference'] / 'code/scripts'))
    import source_cycle_benchmark as reference
    strong = project / 'datasets/strong_snr_v1_20261012/generated/target'
    raw, metadata = reference.data.target_dev_from_mat(5, strong)
    assert len(raw) == 88 and len(metadata) == 2474
    x = reference.data.raw_cycles(raw, metadata)
    np.save(output / 'cache/dev_cycles_snr5.npy', x)
    sys.path.insert(0, str(roots['jung'] / 'code/scripts'))
    import compare_jung_hang_v4 as jung
    jung_config = read(roots['jung'] / 'code/configs/jung_hang_cycle_v4_v1.json')
    assert jung_config['jung_cycles'] == 1
    np.save(output / 'cache/dev_jung_snr5.npy', jung.build_inputs('jung', metadata, raw, jung_config))
    write(output / 'DEV_REPLAY_INPUT.json', {'truth': [r['label'] for r in metadata],
          'windows': len(metadata), 'records': 88, 'holdout_read': False})
    print('FROZEN: 12 models, 24 best weights, 4 SNRs; no Holdout waveform read', flush=True)


def dev_replay(config_path, family):
    """按各模型真实输入规则重放Dev的5 dB预测，先验证版本及读取没有偏移。"""
    import numpy as np
    import torch
    _, config, output, roots = context(config_path)
    assert_protected(output)
    y = np.asarray(read(output / 'DEV_REPLAY_INPUT.json')['truth'])
    name = 'jung' if family == 'jung' else 'cycles'
    x = np.load(output / f'cache/dev_{name}_snr5.npy', mmap_mode='r')
    results = []
    for item, seed, model, selected, checkpoint in load_models(config, roots, family):
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pred = predict(model, x, config['batch_size']).argmax(1)
        prior_path = roots[family] / f"results/evaluations/{item['method']}/seed{seed}/target_dev_snr5.json"
        prior = read(prior_path)
        np.testing.assert_array_equal(y, prior['truth'])
        np.testing.assert_array_equal(pred, prior['predicted'])
        assert all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items())
        results.append({'method': item['method'], 'seed': seed, 'windows': len(y), 'prediction_exact': True,
                        'macro_f1': score(y, pred)['macro_f1'], 'weights_sha256': sha(checkpoint), 'epoch': selected['epoch']})
    write(output / f'DEV_REPLAY_{family}.json', results)
    assert_protected(output)
    print(f'DEV REPLAY PASSED: {family}, {len(results)} frozen models', flush=True)


def prepare_holdout(config_path):
    """读取授权的347条保留记录，核对既有噪声后制作所有模型共用的时间窗口。"""
    import numpy as np
    from numpy.lib.format import open_memmap
    from scipy.io import loadmat
    from collections import Counter, defaultdict
    project, config, output, roots = context(config_path)
    plan = assert_protected(output)
    replay = [row for family in roots for row in read(output / f'DEV_REPLAY_{family}.json')]
    assert len(replay) == 24 and all(row['prediction_exact'] for row in replay)
    assert not (output / 'WINDOWS_FINAL_HOLDOUT.json').exists()
    sys.path.insert(0, str(roots['reference'] / 'code/scripts'))
    import source_cycle_benchmark as reference
    from prepare import windows
    sys.path.insert(0, str(roots['jung'] / 'code/scripts'))
    import compare_jung_hang_v4 as jung
    jung_config = read(roots['jung'] / 'code/configs/jung_hang_cycle_v4_v1.json')
    base = project / 'datasets/strong_snr_v1_20261012'
    inputs = read(base / 'metadata/INPUT_MANIFEST.json')['target_files']
    labels = read(base / 'metadata/LABELS.json')['experimental']
    source_hashes, clean_records, author_records, metadata = {}, {}, {}, []
    for rid in plan['holdout_records']:
        pair = []
        for directory, field in [('target_clean', 'name'), ('target_snr10', 'I_noisy')]:
            path = base / 'inputs' / directory / f'{rid}.mat'
            digest = sha(path)
            assert digest == inputs[f'{directory}/{rid}.mat']['sha256'], path
            source_hashes[str(path)] = digest
            value = loadmat(path, variable_names=[field])[field]
            assert value.shape == (10031, 7) and value.dtype == np.float64 and np.isfinite(value).all()
            pair.append(value)
        clean, author = pair
        np.testing.assert_array_equal(clean[:, [0, 6]], author[:, [0, 6]])
        clean_records[rid], author_records[rid] = clean, author
        t, start, count = np.unique(clean[:, 0], return_index=True, return_counts=True)
        merged = np.add.reduceat(clean[:, 1:], start, axis=0) / count[:, None]
        assert np.all(np.diff(t) > 0) and np.all(merged[:, -1] > 0)
        for window in windows(t, merged[:, -1]):
            metadata.append({'record_id': rid, 'label': 0 if window['state'] == 0 else labels[rid]['fault_label'],
                             'start': window['start'], 'end': window['end'], 'state': window['state']})
    assert set(row['record_id'] for row in metadata) == set(plan['holdout_records'])
    assert set(row['label'] for row in metadata) == set(range(6))
    write(output / 'WINDOWS_FINAL_HOLDOUT.json', metadata)
    by_record = defaultdict(list)
    for index, row in enumerate(metadata):
        by_record[row['record_id']].append(index)
    generation_roots = {
        5: base / 'generated', 0: base / 'generated',
        -5: project / 'datasets/strong_snr_minus5_v1_20261014/generated',
        -10: project / 'datasets/strong_snr_minus10_v1_20261015/generated',
    }
    caches, max_snr_error, max_residual_error = {}, 0., 0.
    for level in config['conditions_db']:
        generation = generation_roots[level]
        manifest_path = generation / 'GENERATION_MANIFEST.json'
        source_hashes[str(manifest_path)] = sha(manifest_path)
        entries = {entry['file']: entry for entry in read(manifest_path)['files']}
        cycle_path, jung_path = output / f'cache/cycles_snr{level}.npy', output / f'cache/jung_snr{level}.npy'
        assert not cycle_path.exists() and not jung_path.exists()
        cycle_cache = open_memmap(cycle_path, mode='w+', dtype=np.float32, shape=(len(metadata), 5, 4096))
        jung_cache = open_memmap(jung_path, mode='w+', dtype=np.float32, shape=(len(metadata), 5, 128))
        for rid in plan['holdout_records']:
            relative = f'target/SNR_{level}/{rid}.mat'
            path = generation / relative
            digest = sha(path)
            assert digest == entries[relative]['file_sha256'], path
            source_hashes[str(path)] = digest
            noisy = loadmat(path, variable_names=['I_noisy'])['I_noisy']
            clean, author = clean_records[rid], author_records[rid]
            assert noisy.shape == clean.shape and noisy.dtype == np.float64 and np.isfinite(noisy).all()
            np.testing.assert_array_equal(noisy[:, [0, 6]], clean[:, [0, 6]])
            residual = noisy[:, 1:6] - clean[:, 1:6]
            measured = 10 * np.log10(np.mean(clean[:, 1:6] ** 2, axis=0) / np.mean(residual ** 2, axis=0))
            error = float(np.abs(measured - level).max())
            max_snr_error = max(max_snr_error, error)
            assert error < 1e-6
            expected_residual = (author[:, 1:6] - clean[:, 1:6]) * 10 ** ((10 - level) / 20)
            np.testing.assert_allclose(residual, expected_residual, rtol=1e-12, atol=1e-12)
            max_residual_error = max(max_residual_error, float(np.abs(residual - expected_residual).max()))
            t, start, count = np.unique(noisy[:, 0], return_index=True, return_counts=True)
            merged = np.add.reduceat(noisy[:, 1:], start, axis=0) / count[:, None]
            raw = {rid: (t, merged[:, :5], merged[:, -1])}
            positions = by_record[rid]
            rows = [metadata[i] for i in positions]
            # 六个基准及五个消融共用完全相同的5×4096输入；Jung保留原V4的5×128输入。
            cycle_cache[positions] = reference.data.raw_cycles(raw, rows)
            jung_cache[positions] = jung.build_inputs('jung', rows, raw, jung_config)
        cycle_cache.flush()
        jung_cache.flush()
        del cycle_cache, jung_cache
        caches[str(level)] = {'cycles': str(cycle_path), 'cycles_sha256': sha(cycle_path),
                              'jung': str(jung_path), 'jung_sha256': sha(jung_path)}
        print(f'PREPARED Holdout SNR {level}: {len(metadata)} cycles from 347 records', flush=True)
    write(output / 'PREPARED_INPUTS.json', {'status': 'passed', 'records': 347, 'cycles': len(metadata),
          'class_counts': [Counter(row['label'] for row in metadata)[i] for i in range(6)],
          'windows_sha256': sha(output / 'WINDOWS_FINAL_HOLDOUT.json'), 'caches': caches,
          'input_files_sha256': source_hashes, 'max_whole_record_per_phase_snr_error_db': max_snr_error,
          'max_author_residual_scaling_absolute_error': max_residual_error,
          'time_speed_columns_unchanged': True, 'same_windows_for_all_96_evaluations': True,
          'new_noise_generated': False, 'source_or_dev_records_in_holdout': False})
    assert_protected(output)


def evaluate_family(config_path, family):
    """同一组冻结模型逐档评价；第二次推理仅验证重放，不改变模型或评估方案。"""
    import numpy as np
    import torch
    _, config, output, roots = context(config_path)
    assert_protected(output)
    prepared = read(output / 'PREPARED_INPUTS.json')
    assert prepared['status'] == 'passed'
    metadata = read(output / 'WINDOWS_FINAL_HOLDOUT.json')
    y = np.asarray([row['label'] for row in metadata], dtype=np.int64)
    loaded = load_models(config, roots, family)
    rows = []
    for level in config['conditions_db']:
        data_kind = 'jung' if family == 'jung' else 'cycles'
        cache = prepared['caches'][str(level)]
        assert sha(cache[data_kind]) == cache[data_kind + '_sha256']
        x = np.load(cache[data_kind], mmap_mode='r')
        assert len(x) == len(y)
        for item, seed, model, selected, checkpoint in loaded:
            destination = output / f"results/evaluations/{item['method']}/seed{seed}/final_holdout_snr{level}.json"
            assert not destination.exists(), destination
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            start = time.monotonic()
            logits = predict(model, x, config['batch_size'])
            pred = logits.argmax(1)
            replay_logits = predict(model, x, config['batch_size'])
            np.testing.assert_array_equal(pred, replay_logits.argmax(1))
            assert all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items())
            result = {'protocol': config['protocol'], **item, 'seed': seed, 'snr_db': level,
                      'record_families': 347, 'windows': len(y), 'windows_sha256': prepared['windows_sha256'],
                      'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                      'source_selected_epoch': selected['epoch'], 'truth': y.tolist(), 'predicted': pred.tolist(),
                      'metrics': score(y, pred), 'prediction_replay_exact': True,
                      'logits_replay_exact': bool(np.array_equal(logits, replay_logits)),
                      'model_parameters_and_bn_unchanged': True, 'retrained': False,
                      'target_used_for_training_or_selection': False, 'final_holdout_evaluated': True,
                      'elapsed_seconds_including_replay': time.monotonic() - start}
            write(destination, result)
            rows.append({**item, 'seed': seed, 'snr_db': level, 'macro_f1': result['metrics']['macro_f1'],
                         'accuracy': result['metrics']['accuracy'], 'prediction_replay_exact': True})
            write(output / f'PROGRESS_{family}.json', {'completed': len(rows), 'total': len(loaded) * 4})
            print(f"EVALUATED {item['method']} seed{seed} SNR{level}: F1={100*result['metrics']['macro_f1']:.4f}%", flush=True)
        del x
    assert_protected(output)
    write(output / f'COMPLETED_{family}.json', rows)


def verify_and_report(config_path):
    """不调用sklearn计分函数，从96组预测独立构建混淆矩阵和指标。"""
    import numpy as np
    _, config, output, roots = context(config_path)
    plan = assert_protected(output)
    prepared = read(output / 'PREPARED_INPUTS.json')
    metadata = read(output / 'WINDOWS_FINAL_HOLDOUT.json')
    truth = np.asarray([r['label'] for r in metadata], dtype=np.int64)
    records = set(row['record_id'] for row in metadata)
    assert len(records) == 347 and records == set(plan['holdout_records'])
    rows, classes, summary = [], [], []
    for item in config['models']:
        for seed in config['seeds']:
            for level in config['conditions_db']:
                path = output / f"results/evaluations/{item['method']}/seed{seed}/final_holdout_snr{level}.json"
                result = read(path)
                assert result['checkpoint_sha256'] == sha(result['checkpoint'])
                assert result['windows_sha256'] == prepared['windows_sha256']
                assert result['prediction_replay_exact'] and result['model_parameters_and_bn_unchanged']
                assert result['final_holdout_evaluated'] and not result['target_used_for_training_or_selection']
                np.testing.assert_array_equal(result['truth'], truth)
                pred = np.asarray(result['predicted'], dtype=np.int64)
                assert pred.shape == truth.shape and np.all((pred >= 0) & (pred < 6))
                cm = np.bincount(truth * 6 + pred, minlength=36).reshape(6, 6)
                tp, support, predicted_count = np.diag(cm), cm.sum(1), cm.sum(0)
                f1 = np.divide(2 * tp, support + predicted_count, out=np.zeros(6), where=support + predicted_count > 0)
                recall = np.divide(tp, support, out=np.zeros(6), where=support > 0)
                precision = np.divide(tp, predicted_count, out=np.zeros(6), where=predicted_count > 0)
                np.testing.assert_array_equal(cm, result['metrics']['confusion_matrix'])
                np.testing.assert_allclose(f1, result['metrics']['f1_by_class'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(recall, result['metrics']['recall_by_class'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(precision, result['metrics']['precision_by_class'], atol=1e-12, rtol=0)
                assert abs(float(f1.mean()) - result['metrics']['macro_f1']) < 1e-12
                assert abs(float(tp.sum() / len(truth)) - result['metrics']['accuracy']) < 1e-12
                rows.append({**item, 'seed': seed, 'snr_db': level, 'records': 347, 'cycles': len(truth),
                             'macro_f1_percent': float(f1.mean() * 100), 'accuracy_percent': float(tp.sum() / len(truth) * 100),
                             'selected_epoch': result['source_selected_epoch'], 'weights_sha256': result['checkpoint_sha256']})
                for label in range(6):
                    classes.append({'method': item['method'], 'seed': seed, 'snr_db': level, 'label': label,
                                    'precision': float(precision[label]), 'recall': float(recall[label]),
                                    'f1': float(f1[label]), 'support': int(support[label])})
        for level in config['conditions_db']:
            chosen = [r for r in rows if r['method'] == item['method'] and r['snr_db'] == level]
            assert len(chosen) == 2
            f1 = [r['macro_f1_percent'] for r in chosen]
            acc = [r['accuracy_percent'] for r in chosen]
            summary.append({**item, 'snr_db': level, 'seeds': '42,44', 'macro_f1_mean_percent': float(np.mean(f1)),
                            'macro_f1_sample_sd_percent': float(np.std(f1, ddof=1)),
                            'accuracy_mean_percent': float(np.mean(acc)), 'accuracy_sample_sd_percent': float(np.std(acc, ddof=1))})
    assert len(rows) == 96 and len(classes) == 576 and len(summary) == 48
    # 原始噪声文件、缓存和原代码、历史权重在整个推理后都必须保持原字节。
    for path, digest in prepared['input_files_sha256'].items():
        assert sha(path) == digest, path
    for cache in prepared['caches'].values():
        for kind in ('cycles', 'jung'):
            assert sha(cache[kind]) == cache[kind + '_sha256']
    assert_protected(output)
    reports = output / 'results/reports'
    save_csv(reports / 'PER_SEED_RESULTS.csv', rows)
    save_csv(reports / 'PER_CLASS_RESULTS.csv', classes)
    save_csv(reports / 'SUMMARY.csv', summary)
    save_csv(reports / 'WEIGHTS_FROZEN.csv', plan['weights'])
    verification = {'status': 'passed', 'evaluations': 96, 'fixed_best_weights': 24,
                    'holdout_records': 347, 'cycles_per_condition': len(truth),
                    'class_counts': prepared['class_counts'], 'prediction_replays_exact': 96,
                    'independent_numpy_metrics_verified': 96,
                    'dev_replay_models_before_holdout': 24, 'weight_and_archived_code_sha256_unchanged': True,
                    'input_file_sha256_unchanged': True, 'model_parameters_and_bn_unchanged': True,
                    'all_models_same_record_ids_windows_labels': True,
                    'retrained': False, 'checkpoint_reselected': False, 'a2_target_recalibration': False,
                    'final_holdout_evaluated': True, 'target_used_for_training_or_selection': False,
                    'max_whole_record_per_phase_snr_error_db': prepared['max_whole_record_per_phase_snr_error_db']}
    write(output / 'VERIFICATION.json', verification)
    write(reports / 'SUMMARY.json', {'config': config, 'verification': verification, 'per_seed': rows, 'summary': summary})
    text = [
        '# 论文V3.13：347记录Final-Holdout固定权重评价', '', '## Material Passport', '',
        '- 日期：2026-10-24；academic-research-suite / experiment-agent；Mode: run。',
        '- 12个模型配置、种子42/44、5/0/−5/−10 dB，共96组评价。',
        '- 只用既有best.pt推理，没有训练、重新选权重或使用目标域重标定A2。',
        '- 独立指标复算、逐样本预测重放、模型状态及文件指纹核验通过。', '',
        '## 论文与模型来源', '',
        f"- 论文：`{config['paper_path']}`；SHA256 `{config['paper_sha256']}`。",
        '- AHR-Net对应B1，HR-Net对应B0。Jung采用论文数值对应的单周期V4，不使用双周期V6。',
        '- 六个基准/完整模型：`source_cycle_benchmark/4728e9b`。',
        '- 五项消融：`b1_paper_ablations/d9947f0_20261016`。',
        '- Jung：`jung_hang_cycle_v4/a0c78d8_20261019`。',
        '- 模型和种子在读取Holdout波形之前冻结，权重路径、最佳轮次及SHA见WEIGHTS_FROZEN.csv。',
        '- 42/44沿用用户查看历史六种子结果后指定的展示种子；本次两种子全部报告，不按Holdout成绩筛选。', '',
        '## 数据及评价规则', '',
        f"- 固定347个保留编号，每档{len(truth)}个完整电周期，六类数量为{prepared['class_counts']}。",
        '- 与88个Dev及其完全重复记录隔离；编号来自原冻结划分，没有重新抽样。',
        '- 原规则：健康0.1–0.45秒；故障从0.5秒后跳过两周期到0.98秒，五相同步。',
        '- 六个基准和五个消融使用5×4096输入；Jung沿用V4预处理与5×128输入，同一套周期边界及标签。',
        '- 复用此前为全部440编号生成的四档噪声文件，没有新增噪声。整记录、分相SNR与作者10 dB残差缩放独立核验。',
        '- 各噪声版本来自同一批记录，不是不同的独立采集。',
        '- 指标按周期计算，六类Macro-F1与Accuracy；各类Precision/Recall/F1及混淆矩阵完整保存。',
        '- 各种子来自既有训练。本次347记录不用于训练、权重选择或模型配置修改。', '',
    ]
    for seed in config['seeds']:
        for group, heading in [('main', '主对比'), ('ablation', '消融')]:
            text.extend([f'## 种子{seed}：{heading}Macro-F1（%）', '', '| 模型 | 5 dB | 0 dB | −5 dB | −10 dB |', '|---|---:|---:|---:|---:|'])
            for item in config['models']:
                if item['group'] != group:
                    continue
                values = [next(r['macro_f1_percent'] for r in rows if r['method'] == item['method'] and r['seed'] == seed and r['snr_db'] == level) for level in config['conditions_db']]
                text.append('| ' + item['display'] + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
            text.append('')
    text.extend(['## 验证与文件', '',
                 '- 正式评价前24组5 dB Dev预测与各自历史预测逐项相同。',
                 '- 96组Holdout预测第二次推理重放完全一致；NumPy独立混淆矩阵及指标复算通过。',
                 '- 不改写原best/last权重、模型参数、BN缓冲区或历史代码；受保护文件SHA256一致。',
                 '- PER_SEED_RESULTS.csv：96行逐种子指标；PER_CLASS_RESULTS.csv：576行类别指标。',
                 '- SUMMARY.csv：两种子的均值及样本SD；SD仅描述两次训练结果，不代表独立采集误差或显著性结论。',
                 '- results/evaluations：96组完整真值、预测及混淆矩阵。',
                 '- 原始电流、模型权重和输入缓存只保留服务器，不包含在结果下载包。',
                 '- 本记录只记本次固定方案的评价结果，不改变论文正文或旧Dev结果。', ''])
    (output / 'SERVER_RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'VERIFIED 96 evaluations; {len(truth)} cycles/condition; 24 fixed best weights unchanged', flush=True)


def export_results(output):
    """只打包可审查记录、代码和预测，不下载原始电流、缓存或权重。"""
    assert read(output / 'VERIFICATION.json')['status'] == 'passed'
    bundle = output / 'final_holdout_v313_results_20261024.zip'
    paths = [p for p in output.rglob('*') if p.is_file() and
             p.parts[len(output.parts)] in {'results', 'logs', 'code'}]
    paths += [p for p in output.iterdir() if p.is_file() and p.suffix in {'.json', '.md'}]
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, arcname=str(path.relative_to(output)))
    write(output / 'EXPORT_MANIFEST.json', {'archive': str(bundle), 'archive_sha256': sha(bundle),
          'archive_bytes': bundle.stat().st_size, 'files': len(set(paths)), 'raw_waveforms_included': False,
          'weights_included': False, 'input_caches_included': False})
    print('EXPORTED ' + str(bundle), flush=True)


def parallel_stage(config_path, output, stage):
    """三个历史代码组分别使用独立进程，最多三组并发，不自动重试失败任务。"""
    jobs = []
    for family in ('reference', 'ablation', 'jung'):
        log = (output / f'logs/{stage}_{family}.log').open('w', encoding='utf-8')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), stage, '--config', str(config_path), '--family', family]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=output)
        jobs.append((family, process, log))
    write(output / f'PROCESS_{stage}.json', {family: {'pid': process.pid} for family, process, _ in jobs})
    while any(process.poll() is None for _, process, _ in jobs):
        time.sleep(.5)
    failed = []
    for family, process, log in jobs:
        log.close()
        if process.returncode:
            failed.append({'family': family, 'exit_code': process.returncode,
                           'log_tail': (output / f'logs/{stage}_{family}.log').read_text(encoding='utf-8')[-4500:]})
    if failed:
        raise RuntimeError(json.dumps(failed, ensure_ascii=False))


def run(config_path):
    """单次运行完整授权流程，明确记录进程、阶段、异常和完成状态。"""
    _, config, output, _ = context(config_path)
    started = time.monotonic()
    stage = 'freeze'
    try:
        freeze(config_path)
        (output / 'code').mkdir()
        (output / 'code/evaluate_final_holdout_v313.py').write_bytes(Path(__file__).read_bytes())
        (output / 'code/config.json').write_bytes(Path(config_path).read_bytes())
        for stage in ('dev', 'prepare', 'evaluate', 'verify'):
            write(output / 'STATUS.json', {'status': 'running', 'stage': stage, 'pid': os.getpid(),
                  'elapsed_seconds': time.monotonic() - started})
            print('STAGE ' + stage, flush=True)
            if stage in ('dev', 'evaluate'):
                parallel_stage(config_path, output, stage)
            elif stage == 'prepare':
                prepare_holdout(config_path)
            else:
                verify_and_report(config_path)
        write(output / 'STATUS.json', {'status': 'completed', 'stage': 'verified', 'pid': os.getpid(),
              'elapsed_seconds': time.monotonic() - started, 'evaluations': 96, 'fixed_weights': 24,
              'retrained': False, 'protocol': config['protocol']})
        export_results(output)
    except Exception as exc:
        if output.exists():
            write(output / 'STATUS.json', {'status': 'failed', 'stage': stage, 'pid': os.getpid(),
                  'elapsed_seconds': time.monotonic() - started, 'error': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'freeze', 'dev', 'prepare', 'evaluate', 'verify', 'export'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--family', choices=['reference', 'ablation', 'jung'])
    args = parser.parse_args()
    if args.stage == 'run':
        run(args.config)
    elif args.stage == 'freeze':
        freeze(args.config)
    elif args.stage == 'dev':
        assert args.family
        dev_replay(args.config, args.family)
    elif args.stage == 'prepare':
        prepare_holdout(args.config)
    elif args.stage == 'evaluate':
        assert args.family
        evaluate_family(args.config, args.family)
    elif args.stage == 'verify':
        verify_and_report(args.config)
    else:
        export_results(context(args.config)[2])

