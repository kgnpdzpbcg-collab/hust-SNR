"""重放最佳权重，独立复算指标，并将五个消融与既有Full B1逐种子配对。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
RUN = Path('/home/wang/projects/会议论文/hust/experiments/b1_paper_ablations/d9947f0_20261016')
CODE = RUN / 'code'
OUT = RUN / 'results'
sys.path.insert(0, str(CODE / 'scripts'))
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from b1_paper_ablations import configuration, build_model, predict, checked_source, target_cycles
from hust_a1_b1_experiment import select_source_checkpoint
from v1_noise_consistency import fix_randomness
from hust_b1_phase_harmonic import B1PhaseAwareRecoveryNet


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf8')


def write_csv(path, rows):
    """以显式表头保存每一实际种子结果，平均值只作为另一个汇总表。"""
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_metrics(obj, expected_truth):
    """不依赖训练入口的metrics实现，以sklearn独立重算六类指标。"""
    truth = np.asarray(obj['truth'])
    pred = np.asarray(obj['predicted'])
    assert np.array_equal(truth, expected_truth)
    score = obj['metrics']
    np.testing.assert_allclose(score['macro_f1'], f1_score(truth, pred, average='macro'), rtol=0, atol=1e-15)
    np.testing.assert_allclose(score['accuracy'], accuracy_score(truth, pred), rtol=0, atol=1e-15)
    np.testing.assert_allclose(score['per_class_f1'], f1_score(truth, pred, average=None), rtol=0, atol=1e-15)
    np.testing.assert_allclose(score['per_class_recall'], recall_score(truth, pred, average=None), rtol=0, atol=1e-15)
    cm = confusion_matrix(truth, pred, labels=np.arange(6))
    assert np.array_equal(cm, score['confusion_matrix'])
    assert score['missed_fault_as_normal'] == int(cm[1:, 0].sum())
    assert score['normal_false_alarms'] == int(cm[0, 1:].sum())


def main():
    """训练结束后只读回放；不更新参数，不重选epoch，不读取最终保留集。"""
    start = time.time()
    assert read(RUN / 'PIPELINE_STATUS.json')['state'] == 'completed'
    torch.set_num_threads(2)
    config = configuration()
    manifest = read(RUN / 'DEPLOYMENT_MANIFEST.json')
    source = Path(manifest['source_root'])
    baseline = Path(manifest['baseline_root'])
    (tr, val), provenance = checked_source(source, config)
    _, _, vy, val_clean, val_noise = val
    models, selected_rows, histories, pretraining, source_replays = {}, [], [], [], []
    new_weight_sha = {str(p): sha(p) for p in (OUT / 'models').rglob('*.pt')}
    assert len(new_weight_sha) == 20
    device = torch.device('cuda')
    for method in config['methods']:
        for seed in config['seeds']:
            folder = OUT / 'models' / method / f'seed{seed}'
            run, history = read(folder / 'run.json'), read(folder / 'history.json')
            saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            assert len(history) == 50 and all(r['optimizer_steps'] == 31 for r in history)
            assert len(run['pretrain_history']) == (0 if method == config['methods'][0] else 10)
            assert run['config'] == config and saved['config'] == config
            assert not run['target_used_for_training_or_selection'] and not run['final_holdout_evaluated']
            assert run['source_split'] == provenance == saved['source_split']
            chosen = select_source_checkpoint(history)
            assert chosen == saved['selected'] == run['selected']
            old_run = read(baseline / 'results/models/b1' / f'seed{seed}/run.json')
            assert run['first_training_noise_sha256'] == old_run['first_training_noise_sha256']
            fix_randomness(seed)
            model = build_model(method).to(device).eval()
            model.load_state_dict(saved['state'], strict=True)
            models[method, seed] = model
            full_saturated = [r['epoch'] for r in history if r['clean_f1'] == 1 and all(v == 1 for v in r['noisy_f1'].values())]
            selected_rows.append({'method': method, 'seed': seed, 'selected_epoch': chosen['epoch'],
                'source_clean_f1_percent': 100*chosen['clean_f1'], 'source_noise_mean_f1_percent': 100*chosen['noise_mean_f1'],
                'source_noise_mean_ce': chosen['noise_mean_ce'], 'all_six_conditions_perfect_epochs': len(full_saturated),
                'first_all_six_perfect_epoch': full_saturated[0] if full_saturated else '',
                'parameters': run['parameters'], 'a2_fixed_proxy': float(model.restorer.fixed_proxy) if 'fixed_proxy' in model.restorer._buffers else ''})
            for name, values in [('clean', val_clean), *[(f'snr{s}', values) for s, values in val_noise.items()]]:
                pred, ce = predict(model, values, device, vy)
                f1 = f1_score(vy, pred, average='macro')
                expected_f1 = chosen['clean_f1'] if name == 'clean' else chosen['noisy_f1'][name[3:]]
                expected_ce = chosen['clean_ce'] if name == 'clean' else chosen['noisy_ce'][name[3:]]
                np.testing.assert_allclose(f1, expected_f1, rtol=0, atol=1e-15)
                np.testing.assert_allclose(ce, expected_ce, rtol=0, atol=1e-9)
                source_replays.append({'method': method, 'seed': seed, 'condition': name, 'macro_f1': f1, 'ce': ce})
            for row in history:
                assert np.isfinite(row['train_loss_mean']) and np.isfinite(row['train_loss_sample_weighted'])
                histories.append({'method': method, 'seed': seed, 'epoch': row['epoch'],
                    'train_loss_mean': row['train_loss_mean'], 'train_loss_sample_weighted': row['train_loss_sample_weighted'],
                    'ce': row['train_components']['ce'], 'restore': row['train_components'].get('restore', 0),
                    'clean_f1': row['clean_f1'], 'noise_mean_f1': row['noise_mean_f1'], 'noise_mean_ce': row['noise_mean_ce']})
            pretraining.extend({'method': method, 'seed': seed, 'epoch': i+1, 'loss': value}
                               for i, value in enumerate(run['pretrain_history']))
    rows, perclass = [], []
    target_replays = 0
    # Full B1也回读既有权重重放；不重新训练，不借本次结果重新选轮次。
    for relative, expected in read(RUN / 'BASELINE_FILES_SHA256.json').items():
        assert sha(baseline / relative) == expected
    for seed in config['seeds']:
        saved = torch.load(baseline / f'results/models/b1/seed{seed}/best.pt', map_location='cpu', weights_only=True)
        model = B1PhaseAwareRecoveryNet().to(device).eval()
        model.load_state_dict(saved['state'], strict=True)
        models['b1', seed] = model
    states_before = {key: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for key, model in models.items()}
    for snr in config['target_conditions']:
        x, metadata = target_cycles(snr, None, Path(manifest['strong_root']), Path(manifest['extreme_root']))
        truth = np.asarray([r['label'] for r in metadata], dtype=np.int64)
        for method in [*config['methods'], 'b1']:
            for seed in config['seeds']:
                root = baseline / 'results' if method == 'b1' else OUT
                obj = read(root / 'evaluations' / method / f'seed{seed}/target_dev_snr{snr}.json')
                assert obj['source_split'] == provenance and not obj['final_holdout_evaluated']
                check_metrics(obj, truth)
                pred, _ = predict(models[method, seed], x, device)
                assert np.array_equal(pred, obj['predicted'])
                target_replays += 1
                score = obj['metrics']
                rows.append({'method': method, 'seed': seed, 'condition': f'snr{snr}',
                    'macro_f1_percent': 100*score['macro_f1'], 'accuracy_percent': 100*score['accuracy'],
                    'fault_as_healthy': score['missed_fault_as_normal'], 'healthy_as_fault': score['normal_false_alarms']})
                perclass.extend({'method': method, 'seed': seed, 'condition': f'snr{snr}', 'label': label,
                    'f1_percent': 100*f1, 'recall_percent': 100*recall} for label, (f1, recall) in
                    enumerate(zip(score['per_class_f1'], score['per_class_recall'])))
    for key, model in models.items():
        for name, value in model.state_dict().items():
            assert torch.equal(value.detach().cpu(), states_before[key][name]), (key, name)
    summary, paired = [], []
    for method in [*config['methods'], 'b1']:
        for snr in config['target_conditions']:
            subset = [r for r in rows if r['method'] == method and r['condition'] == f'snr{snr}']
            summary.append({'method': method, 'condition': f'snr{snr}', 'seeds': 2,
                'macro_f1_mean': float(np.mean([r['macro_f1_percent'] for r in subset])),
                'macro_f1_sd': float(np.std([r['macro_f1_percent'] for r in subset], ddof=1))})
            if method != 'b1':
                for row in subset:
                    full = next(r for r in rows if r['method'] == 'b1' and r['seed'] == row['seed'] and r['condition'] == row['condition'])
                    paired.append({'method': method, 'seed': row['seed'], 'condition': row['condition'],
                        'full_b1_macro_f1': full['macro_f1_percent'], 'ablation_macro_f1': row['macro_f1_percent'],
                        'full_minus_ablation_pp': full['macro_f1_percent'] - row['macro_f1_percent']})
    for row in read(OUT / 'reports/SUMMARY.json')['summary']:
        checked = next(r for r in summary if (r['method'], r['condition']) == (row['method'], row['condition']))
        np.testing.assert_allclose([checked['macro_f1_mean'], checked['macro_f1_sd']],
                                  [row['macro_f1_mean'], row['macro_f1_sd']], rtol=0, atol=1e-12)
    for manifest_name in ('PROTECTED_INPUTS_BEFORE.json', 'PROTECTED_WEIGHTS_BEFORE.json'):
        for path, expected in read(RUN / manifest_name).items():
            assert sha(path) == expected, path
    for name, expected in manifest['code_sha256'].items():
        assert sha(CODE / name) == expected
    for path, expected in new_weight_sha.items():
        assert sha(path) == expected
    reports = OUT / 'reports'
    for filename, values in [('WITH_FULL_B1_PER_SEED.csv', rows), ('WITH_FULL_B1_SUMMARY.csv', summary),
        ('WITH_FULL_B1_PER_CLASS.csv', perclass), ('PAIRED_FULL_MINUS_ABLATION.csv', paired),
        ('SOURCE_SELECTED_CHECKPOINTS.csv', selected_rows), ('SOURCE_VALIDATION_REPLAY.csv', source_replays),
        ('TRAINING_HISTORY.csv', histories), ('PRETRAIN_HISTORY.csv', pretraining)]:
        write_csv(reports / filename, values)
    write(reports / 'WEIGHTS_SHA256.json', new_weight_sha)
    result = {'status': 'passed', 'verification_scope': 'saved-checkpoint source/target replay and independent metric recomputation; not a second complete training',
        'new_trainings': 10, 'source_checkpoint_replays': len(source_replays), 'exact_target_prediction_replays': target_replays,
        'exact_new_target_replays': 40, 'exact_historical_b1_target_replays': 8,
        'independent_target_metrics_checked_including_full_b1': len(rows), 'training_history_rows': len(histories),
        'pretrain_history_rows': len(pretraining), 'summary_rows_including_full_b1': len(summary), 'paired_rows': len(paired),
        'protected_input_files_unchanged': len(read(RUN / 'PROTECTED_INPUTS_BEFORE.json')),
        'historical_weights_unchanged': len(read(RUN / 'PROTECTED_WEIGHTS_BEFORE.json')),
        'new_weights_unchanged_during_replay': len(new_weight_sha), 'model_parameters_and_bn_unchanged_by_inference': True,
        'final_holdout_evaluated': False, 'seconds': time.time()-start}
    write(RUN / 'VERIFICATION.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
