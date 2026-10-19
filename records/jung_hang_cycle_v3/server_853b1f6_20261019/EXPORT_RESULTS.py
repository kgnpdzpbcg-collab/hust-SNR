"""导出V3独立实验记录，保留原始文件字节，另存V2/V3逐种子比较。"""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

RUN = Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v3/853b1f6_20261019')
OUT = RUN / 'results'


def read(path):
    """读取服务器生成的UTF-8 JSON记录。"""
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    """计算原始字节SHA，供历史保护和导出核对使用。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    """写入派生比较表；原指标文件保持原字节。"""
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf8', newline='\n')


def main():
    """核验成功后打包；不包含原始MAT、模型权重或缓存。"""
    pipeline = read(RUN / 'PIPELINE_STATUS.json')
    assert pipeline['state'] == 'completed'
    assert read(OUT / 'VERIFICATION.json')['status'] == 'passed'
    manifest = read(RUN / 'DEPLOYMENT_MANIFEST.json')
    for name, expected in manifest['code_sha256'].items():
        assert sha(RUN / 'code' / name) == expected, name
    protected = read(RUN / 'PROTECTED_FILES_BEFORE.json')
    for name, expected in protected.items():
        assert sha(Path(name)) == expected, name
    weights = {str(p): sha(p) for p in (OUT / 'models').rglob('*.pt')}
    assert len(weights) == 8
    write_json(OUT / 'reports/WEIGHTS_SHA256.json', weights)

    # 合表的十二个旧模型，以及V2两模型结果，均保存原CSV和可追溯来源。
    previous = Path(manifest['previous_output']) / 'results/reports/PER_SEED_RESULTS.csv'
    v2 = Path(manifest['v2_root']) / 'results/reports/PER_SEED_RESULTS.csv'
    references = [
        (previous, 'REFERENCE_12_MODELS_PER_SEED.csv', '4a6f4b9590fd6e88e46bb3dc579c7f966d74b2fe', 168),
        (v2, 'REFERENCE_V2_PER_SEED_RESULTS.csv', '7671926fc65066af50b198b4207e2bb4d2f23697', 28),
    ]
    provenance = []
    for source, target, commit, count in references:
        assert len(rows(source)) == count
        (OUT / 'reports' / target).write_bytes(source.read_bytes())
        provenance.append({'path': str(source), 'archive_name': target, 'sha256': sha(source), 'record_commit': commit, 'rows': count})
    write_json(OUT / 'reports/REFERENCE_PROVENANCE.json', provenance)

    current = rows(OUT / 'reports/PER_SEED_RESULTS.csv')
    old = {(r['method'], int(r['seed']), int(r['snr_db'])): r for r in rows(v2)}
    comparison = []
    for row in current:
        key = row['method'], int(row['seed']), int(row['snr_db'])
        prior = old[key]
        comparison.append({'method': key[0], 'seed': key[1], 'snr_db': key[2],
                           'v2_macro_f1_percent': prior['macro_f1_percent'],
                           'v3_macro_f1_percent': row['macro_f1_percent'],
                           'v3_minus_v2_percentage_points': float(row['macro_f1_percent']) - float(prior['macro_f1_percent'])})
    assert len(comparison) == 28
    write_csv(OUT / 'reports/V2_V3_PER_SEED_COMPARISON.csv', comparison)
    write_json(RUN / 'EXPORT_VERIFICATION.json', {'status': 'passed', 'protected_files_unchanged': len(protected),
               'weights_hashed': len(weights), 'code_snapshot_hashes_verified': len(manifest['code_sha256']),
               'v2_v3_paired_rows': len(comparison)})

    config = manifest['config']
    aliases = {'jung': 'Jung Modified CNN', 'hang': 'Hang WCNN'}
    selected = rows(OUT / 'reports/SELECTED_CHECKPOINTS.csv')
    lines = ['# Jung Modified CNN与Hang WCNN V3：服务器实验记录', '', '## 版本与数据', '',
             '- 执行日期：2026-10-19；代码分支：`' + manifest['model_branch'] + '`；代码提交：`' + manifest['model_commit'] + '`。',
             '- 两模型为论文机制指导下的五相任务适配，不是作者公开代码的逐参数复现。',
             '- Source-Cycle V2：63条仿真记录，962训练周期、221验证周期、78隔离周期，全部沿用既有清单。',
             '- 固定88个实测Dev编号，每档2474周期；测试5、4、0、−4、−5、−8、−10 dB，复用已有文件。',
             '- 目标Dev不用于训练或选模；未评估347条Final-Holdout。', '', '## 冻结训练及选模', '',
             '- 两模型均运行42、44种子，各50轮；Adam，lr=0.001，weight_decay=0，batch32，每轮31次更新。',
             '- Jung：每相中点去偏置、五相共享尺度，输入5×128；Clean与门控带噪样本CE各占0.5。',
             '- Jung训练噪声：10/15/20/25/30/35/40 dB逐记录逐相AWGN，持续低电流区限制加噪，再标定整记录每相SNR。',
             '- Hang：基波VSD平面、逐时刻模长归一化、单通道40×40图；五个卷积阶段，前两阶段后固定Haar LL池化。',
             '- Hang仅使用Clean源样本的CE训练；未使用噪声训练视图。两模型采样顺序及更新次数相同。',
             '- 验证：Clean与固定0/5/10/20/30 dB；V3验证也使用静默区门控噪声，随机数命名空间为HUST-V3。',
             '- 选模：Clean F1距峰值≤1个百分点；依次按五噪声平均F1、Clean F1、最小五噪声平均CE、最早轮次排序。',
             '- V2/V3存在输入、训练视图和源验证噪声变化；版本间差值不构成单因素消融。',
             '- 42、44种子沿用用户此前在查看六种子结果后的指定；本轮未增加种子。', '', '## 入口修复', '',
             '- 原提交noise_records引用了setup局部导入的paper，会触发NameError；补入一行显式导入。',
             '- 未修改模型、归一化、噪声公式或冻结超参数；修复差异见[ENTRY_FIX.patch](ENTRY_FIX.patch)。',
             '- 新增训练/验证噪声入口回归测试；原模型8项、V3模型8项、入口回归2项共18项通过。', '',
             '## 参数与源域选模', '', '| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五噪声源F1（%） |',
             '|---|---:|---:|---:|---:|---:|']
    for row in selected:
        run = read(OUT / f"models/{row['method']}/seed{row['seed']}/run.json")
        chosen = run['selected']
        lines.append(f"| {aliases[row['method']]} | {row['seed']} | {row['parameters']} | {row['selected_epoch']} | {100*chosen['clean_f1']:.2f} | {100*chosen['noise_mean_f1']:.2f} |")
    for seed in config['seeds']:
        lines += ['', f'## 种子{seed}：Macro-F1（%）', '', '| 版本/模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in config['models']:
            subset = [r for r in comparison if r['method'] == method and r['seed'] == seed]
            for version in ('v2', 'v3'):
                values = [next(float(r[version + '_macro_f1_percent']) for r in subset if r['snr_db'] == snr) for snr in config['target_conditions_db']]
                lines.append('| ' + version.upper() + ' ' + aliases[method] + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    lines += ['', '## 核验与环境', '',
              '- 两模型CUDA梯度有限；28组目标预测回放完全一致，24组源验证F1及CE回放一致。',
              '- sklearn独立复算Macro-F1、Accuracy、各类F1/Recall及混淆矩阵；推理未改变权重和BN。',
              '- 200轮训练、6200次更新；100个种子/轮次的模型间采样顺序一致。',
              f'- {len(protected)}个历史输入、权重和结果文件SHA未改变；8份新best/last权重留在服务器，发布SHA清单。',
              f'- 实际源记录加噪后的逐相整记录SNR最大绝对误差：{manifest["max_training_full_record_phase_snr_error_db"]:.3g} dB。',
              f'- GPU：{manifest["gpu"]}；PyTorch `{manifest["torch"]}`；复用解释器`{manifest["interpreter"]}`。',
              f'- 四进程训练墙钟：{pipeline["training_seconds"]:.3f}秒；测试：{pipeline["evaluate"]["seconds"]:.3f}秒；核验：{pipeline["verify"]["seconds"]:.3f}秒。',
              f'- 完整训练/测试/回放/汇总流程：{pipeline["elapsed_seconds"]:.3f}秒，不含部署、预检、导出和文档整理。', '',
              '## 文件', '',
              '- [28行V3逐种子指标](results/reports/PER_SEED_RESULTS.csv)；[168行逐类指标](results/reports/PER_CLASS_PER_SEED.csv)。',
              '- [V2/V3逐种子比较](results/reports/V2_V3_PER_SEED_COMPARISON.csv)；[引用文件指纹](results/reports/REFERENCE_PROVENANCE.json)。',
              '- [十二个旧模型与V3两模型合表](results/reports/ALL_14_MODELS_PER_SEED.csv)；[均值及样本标准差](results/reports/ALL_14_MODELS_SUMMARY.csv)。',
              '- [完整训练历史及选模](results/models/)；[训练历史CSV](results/reports/TRAINING_HISTORY.csv)。',
              '- [28份逐周期真值/预测及指标](results/evaluations/)；[源验证回放](results/reports/SOURCE_VALIDATION_REPLAY.csv)；[核验状态](results/VERIFICATION.json)。',
              '- [代码/配置/测试快照](code/)；[训练测试日志](logs/)；[部署清单](DEPLOYMENT_MANIFEST.json)；[导出核验](EXPORT_VERIFICATION.json)。',
              '- 原始MAT、模型权重、环境与高维NPZ不包含在发布内容中。', '']
    (RUN / 'SERVER_RESULTS.md').write_text('\n'.join(lines), encoding='utf8', newline='\n')
    files = [p for name in ('code', 'logs', 'results') for p in (RUN / name).rglob('*')
             if p.is_file() and p.suffix.lower() not in ('.pt', '.pth', '.npz', '.mat')
             and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    files += [RUN / name for name in ('DEPLOYMENT_MANIFEST.json', 'PROTECTED_FILES_BEFORE.json', 'PIPELINE_STATUS.json',
                                     'RUN_PIPELINE.py', 'EXPORT_RESULTS.py', 'ENTRY_FIX.patch', 'EXPORT_VERIFICATION.json', 'SERVER_RESULTS.md')]
    write_json(RUN / 'EXPORT_MANIFEST.json', {p.relative_to(RUN).as_posix(): sha(p) for p in files})
    files.append(RUN / 'EXPORT_MANIFEST.json')
    archive = Path('/tmp/hust_jung_hang_cycle_v3_853b1f6_results.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(files):
            target.write(path, path.relative_to(RUN).as_posix())
    print(json.dumps({'files': len(files), 'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size}))


if __name__ == '__main__':
    main()
