"""核验并导出V4默认双模型结果，保留V2/V3原记录和逐种子对照。"""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

RUN = Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v4/a0c78d8_20261019')
OUT = RUN / 'results'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf8', newline='\n')


def write_csv(path, records):
    """仅写新增派生对照，既有CSV原字节不改。"""
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    """完成回放和历史保护检查后导出，不包含权重、原始数据及缓存。"""
    pipeline = read(RUN/'PIPELINE_STATUS.json')
    assert pipeline['state'] == 'completed'
    assert read(OUT/'VERIFICATION.json')['status'] == 'passed'
    manifest = read(RUN/'DEPLOYMENT_MANIFEST.json')
    for name, expected in manifest['code_sha256'].items():
        assert sha(RUN/'code'/name) == expected, name
    protected = read(RUN/'PROTECTED_FILES_BEFORE.json')
    for name, expected in protected.items():
        assert sha(Path(name)) == expected, name
    weights = {str(p):sha(p) for p in (OUT/'models').rglob('*.pt')}
    assert len(weights) == 8
    write_json(OUT/'reports/WEIGHTS_SHA256.json', weights)
    config = manifest['config']
    # 默认双模型配对训练应使用完全相同的原始噪声及采样顺序，逐轮核对。
    for seed in config['seeds']:
        a = read(OUT/f'models/jung/seed{seed}/history.json')
        b = read(OUT/f'models/hang/seed{seed}/history.json')
        assert len(a) == len(b) == 50
        for x, y in zip(a,b):
            for key in ('epoch', 'optimizer_steps', 'sampling_order_sha256', 'raw_training_noise_sha256'):
                assert x[key] == y[key], (seed, x['epoch'], key)
            assert x['raw_training_noise_sha256'] is not None and x['optimizer_steps'] == 31
    refs = [
        (Path(manifest['previous_output'])/'results/reports/PER_SEED_RESULTS.csv', 'REFERENCE_12_MODELS_PER_SEED.csv',
         '4a6f4b9590fd6e88e46bb3dc579c7f966d74b2fe',168),
        (Path(manifest['v2_root'])/'results/reports/PER_SEED_RESULTS.csv', 'REFERENCE_V2_PER_SEED_RESULTS.csv',
         '7671926fc65066af50b198b4207e2bb4d2f23697',28),
        (Path(manifest['previous_v3_root'])/'results/reports/PER_SEED_RESULTS.csv', 'REFERENCE_V3_PER_SEED_RESULTS.csv',
         'cbcc697f21d1169c0fb91a243dee769954d32d05',28),
    ]
    provenance = []
    versions = {}
    for source, target, commit, count in refs:
        parsed = rows(source)
        assert len(parsed) == count
        (OUT/'reports'/target).write_bytes(source.read_bytes())
        provenance.append({'path':str(source), 'archive_name':target, 'sha256':sha(source), 'record_commit':commit, 'rows':count})
        if count == 28:
            versions['v2' if 'V2_' in target else 'v3'] = {(r['method'],int(r['seed']),int(r['snr_db'])):r for r in parsed}
    write_json(OUT/'reports/REFERENCE_PROVENANCE.json', provenance)
    current = rows(OUT/'reports/PER_SEED_RESULTS.csv')
    comparison = []
    for row in current:
        key = row['method'],int(row['seed']),int(row['snr_db'])
        f1 = float(row['macro_f1_percent'])
        v2,v3 = (float(versions[v][key]['macro_f1_percent']) for v in ('v2','v3'))
        comparison.append({'method':key[0], 'seed':key[1], 'snr_db':key[2],
                           'v2_macro_f1_percent':v2, 'v3_macro_f1_percent':v3, 'v4_macro_f1_percent':f1,
                           'v4_minus_v3_percentage_points':f1-v3, 'v4_minus_v2_percentage_points':f1-v2})
    assert len(comparison) == 28
    write_csv(OUT/'reports/V2_V3_V4_PER_SEED_COMPARISON.csv', comparison)
    write_json(RUN/'EXPORT_VERIFICATION.json', {'status':'passed', 'protected_files_unchanged':len(protected),
               'weights_hashed':8, 'code_snapshot_hashes_verified':len(manifest['code_sha256']),
               'same_training_noise_and_sampling_all_100_seed_epochs':True, 'version_comparison_rows':28,
               'default_configuration_only':True, 'preset_configurations_trained':0})

    alias = {'jung':'Jung Modified CNN','hang':'Hang WCNN'}
    lines = ['# Jung Modified CNN与Hang WCNN V4：默认双模型服务器实验', '', '## Material Passport', '',
             '- 模式：run；日期：2026-10-19；状态：训练、源选模、七档Dev测试和回放完成。',
             '- 分支：`'+manifest['model_branch']+'`；代码提交：`'+manifest['model_commit']+'`。',
             '- 仓库模型代码与默认配置逐字节保留；本轮未修改算法、网络及超参数。',
             '- 两模型是论文机制指导下的五相任务适配，不是作者代码逐参数复现。', '', '## 数据与训练', '',
             '- 沿用Source-Cycle V2：63条仿真记录、962训练周期、221验证周期、78隔离周期。',
             '- 沿用42/44种子，各50轮；Adam lr=0.001、weight_decay=0、batch32，每轮31次更新。',
             '- 默认Jung输入为单周期五相128点，共享幅值归一化；未运行双周期配置。',
             '- 默认Hang为两个40×40 VSD轨迹通道，不去相均值，共享窗口尺度，保留径向幅值信息。',
             '- 两模型均使用0.5 CE(Clean)+0.5 CE(Noisy)配对训练；整记录逐相AWGN为10/15/20/25/30/35/40 dB，每轮重抽。',
             '- 验证Clean及0/5/10/20/30 dB；使用V4普通AWGN及HUST-V4:awgn随机数命名空间。',
             '- 原始训练噪声和采样顺序在两模型间的100个种子/轮次全部相同。',
             '- 选模沿用源域Clean F1距峰值≤1个百分点，随后依次按五噪声平均F1、Clean F1、最小五噪声平均CE、最早轮次排序。',
             '- 固定实测88个Dev编号，每档2474周期；5/4/0/−4/−5/−8/−10 dB均复用已有数据。',
             '- 目标Dev不用于参数拟合或权重选择，347条Final-Holdout未评估。',
             '- 9份受控实验预设只生成配置，未运行；本轮仅默认双模型。',
             '- 42/44是用户查看六种子结果后指定的展示种子，本轮未增加种子。',
             '- V2/V3/V4存在输入和训练/验证噪声变化；版本对照不是单因素消融。', '', '## 源域输入预检', '',
             '- 编译检查及26项CPU测试通过，两模型真实源样本CUDA梯度有限。',
             '- Hang训练/干净验证均无全零图；未发现跨标签完全相同的训练图像哈希。',
             '- [六类轨迹检查图](SOURCE_HANG_TRAJECTORIES.png)：六列依次为类别0–5；四行依次为样本平面1/2、各类均值平面1/2。',
             '- 每类示例固定取清单中的首个训练周期，编号/边界及图像统计见[输入检查记录](SOURCE_HANG_INPUT_AUDIT.json)。',
             '- 辅助绘图最初因环境缺少Matplotlib停止；改用NumPy及标准库生成PNG，未增加训练依赖。模型训练尚未开始时已完成修正。', '',
             '## 参数量与源验证选模', '', '| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五噪声源F1（%） |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows(OUT/'reports/SELECTED_CHECKPOINTS.csv'):
        obj=read(OUT/f"models/{row['method']}/seed{row['seed']}/run.json")['selected']
        lines.append(f"| {alias[row['method']]} | {row['seed']} | {row['parameters']} | {row['selected_epoch']} | {100*obj['clean_f1']:.2f} | {100*obj['noise_mean_f1']:.2f} |")
    for seed in config['seeds']:
        lines += ['',f'## 种子{seed}：Macro-F1（%）','','| 版本/模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in config['models']:
            subset=[r for r in comparison if r['method']==method and r['seed']==seed]
            for version in ('v2','v3','v4'):
                values=[next(r[version+'_macro_f1_percent'] for r in subset if r['snr_db']==snr) for snr in config['target_conditions_db']]
                lines.append('| '+version.upper()+' '+alias[method]+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['', '## 核验、环境及时间', '',
              '- 28组目标预测重新加载best后完全一致，24组源验证F1/CE和选模记录回放一致。',
              '- sklearn独立复算Macro-F1、Accuracy、六类F1/Recall和混淆矩阵；推理不改变模型权重和BN。',
              f'- 200轮、6200次更新全部完成；{len(protected)}个历史文件指纹不变。8份新best/last留在服务器，仅发布SHA。',
              f'- 实际源记录0/5/10/20/40 dB加噪定标检查最大误差：{manifest["max_training_full_record_phase_snr_error_db"]:.3g} dB。',
              f'- GPU：{manifest["gpu"]}；PyTorch `{manifest["torch"]}`；复用解释器`{manifest["interpreter"]}`。',
              f'- 四进程训练墙钟{pipeline["training_seconds"]:.3f}秒；测试{pipeline["evaluate"]["seconds"]:.3f}秒；核验{pipeline["verify"]["seconds"]:.3f}秒。',
              f'- 训练/测试/核验/汇总总墙钟{pipeline["elapsed_seconds"]:.3f}秒，不含预检、传输和文档整理。', '', '## 文件', '',
              '- [V4逐种子28行指标](results/reports/PER_SEED_RESULTS.csv)；[逐类168行指标](results/reports/PER_CLASS_PER_SEED.csv)。',
              '- [V2/V3/V4逐种子对照](results/reports/V2_V3_V4_PER_SEED_COMPARISON.csv)；[引用来源指纹](results/reports/REFERENCE_PROVENANCE.json)。',
              '- [十二旧模型与V4两模型196行合表](results/reports/ALL_14_MODELS_PER_SEED.csv)；[均值和样本标准差](results/reports/ALL_14_MODELS_SUMMARY.csv)。',
              '- [200轮训练历史](results/reports/TRAINING_HISTORY.csv)；[完整训练/选模记录](results/models/)。',
              '- [28份逐周期预测](results/evaluations/)；[24组源验证回放](results/reports/SOURCE_VALIDATION_REPLAY.csv)；[核验状态](results/VERIFICATION.json)。',
              '- [原代码与配置快照](code/)；[日志](logs/)；[部署清单](DEPLOYMENT_MANIFEST.json)；[导出核验](EXPORT_VERIFICATION.json)。',
              '- 本归档不包含原始MAT、模型权重、环境及高维NPZ。', '']
    (RUN/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    files=[p for folder in ('code','logs','results','preset_configs') for p in (RUN/folder).rglob('*')
           if p.is_file() and p.suffix.lower() not in ('.pt','.pth','.mat','.npz')
           and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    files += [RUN/name for name in ('DEPLOYMENT_MANIFEST.json','PROTECTED_FILES_BEFORE.json','PIPELINE_STATUS.json',
              'RUN_PIPELINE.py','PREFLIGHT.py','EXPORT_RESULTS.py','EXPORT_VERIFICATION.json','SERVER_RESULTS.md',
              'SOURCE_HANG_TRAJECTORIES.png','SOURCE_HANG_INPUT_AUDIT.json')]
    write_json(RUN/'EXPORT_MANIFEST.json',{p.relative_to(RUN).as_posix():sha(p) for p in files})
    files.append(RUN/'EXPORT_MANIFEST.json')
    archive=Path('/tmp/hust_jung_hang_cycle_v4_a0c78d8_results.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as target:
        for p in sorted(files):
            target.write(p,p.relative_to(RUN).as_posix())
    print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}))


if __name__=='__main__':
    main()
