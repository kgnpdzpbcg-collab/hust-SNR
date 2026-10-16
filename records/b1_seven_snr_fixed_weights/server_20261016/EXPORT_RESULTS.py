"""打包八模型七档固定权重结果及核验记录，不包含MAT、权重或环境。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

RUN=Path('/home/wang/projects/会议论文/hust/experiments/b1_seven_snr_fixed_weights/20261016')
ROOT=RUN.parents[2]
OUT=RUN/'output'
MODELS=['b1','a1_fixed_lp4_dual','a2_fixed_noise_proxy','a3_no_interphase_descriptors','a4_md_only','a5_no_strength_loss','xu_full','b0']
NAMES={'b1':'完整B1','a1_fixed_lp4_dual':'A1 固定LP4＋双视图','a2_fixed_noise_proxy':'A2 固定噪声估计',
    'a3_no_interphase_descriptors':'A3 屏蔽两组跨相描述','a4_md_only':'A4 仅MD视图',
    'a5_no_strength_loss':'A5 去掉相间强度损失','xu_full':'Xu-Full','b0':'B0'}
SNRS=[5,4,0,-4,-5,-8,-10]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8',newline='\n')


def main():
    assert read(OUT/'VERIFICATION.json')['status']=='passed'
    assert all(row['exit_code']==0 for row in read(RUN/'STAGE_STATUS.json')['jobs'])
    for name,expected in read(RUN/'DEPLOYMENT_MANIFEST.json')['code_sha256'].items():
        assert sha(RUN/name)==expected
    before=read(OUT/'PROTECTED_FILES_SHA256.json')
    assert all(sha(Path(p))==expected for p,expected in before.items())
    sources=read(OUT/'RESULT_PROVENANCE.json')
    reused=0
    for row in sources:
        path=OUT/f"results/evaluations/{row['method']}/seed{row['seed']}/target_dev_snr{row['snr_db']}.json"
        assert sha(path)==row['source_sha256']
        if row['kind']=='reused':
            assert sha(Path(row['source_path']))==sha(path)
            reused+=1
    assert reused==70 and len(sources)==112
    # 复制原训练的只读元数据；模型参数仍只保留在原服务器目录。
    for method in MODELS:
        base=ROOT/('experiments/b1_paper_ablations/d9947f0_20261016' if method.startswith('a') else 'experiments/source_cycle_benchmark/4728e9b')
        for seed in (42,44):
            source=base/f'results/models/{method}/seed{seed}/run.json'
            target=OUT/f'checkpoint_metadata/{method}/seed{seed}/run.json'
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    generation=ROOT/'experiments/additional_snr_seed42_44/20261015_snr4_m4_m8/data/GENERATION_MANIFEST.json'
    value=read(generation)
    write(OUT/'DATA_REFERENCE.json',{'generation_manifest_path':str(generation),'generation_manifest_sha256':sha(generation),
        'files':value['files'],'input_sha256':value['input_sha256'],'regenerated':False,
        'formula':'Clean + (author10 - Clean) * 10**((10 - target_db)/20)',
        'snr_scope':'whole original record, per phase; cycle SNR may differ','independent_noise_realizations':False})
    rows=list(csv.DictReader((OUT/'results/reports/PER_SEED_RESULTS.csv').open()))
    summary=list(csv.DictReader((OUT/'results/reports/SUMMARY.csv').open()))
    timing=read(RUN/'STAGE_STATUS.json')
    preflight=read(OUT/'PREFLIGHT.json')
    lines=['# 八模型七档SNR固定权重服务器实验记录','','## Material Passport','',
        '- 执行日期：2026-10-16（Asia/Shanghai）；模式：固定权重推理、预测回放、汇总。',
        '- 核验范围：42组新增预测及70组历史预测均回放，112组指标独立复算；没有重新训练。',
        '- 模型：完整B1、五个B1消融、Xu-Full、B0；指定种子42、44。',
        '- 原完整B1/Xu/B0训练提交：`4728e9b82c4c181564f0247aa4b1028a35a8e8d2`。',
        '- 原五个消融训练提交：`d9947f0a037c2938480b15bb1423c22e8d79a273`。','',
        '## 输入与执行协议','',
        '- 复用Source-Cycle V2的962 Train、221 Val，原每相4096点；本轮不训练、不选新epoch。',
        '- 使用原16份源验证best，严格加载完整state_dict，包括BN缓冲区和A2固定噪声代理。',
        '- 训练条件仍为原10–40 dB配对增强、Clean及0/5/10/20/30 dB源验证；没有修改原配置。',
        '- 固定88个实测Dev编号，七档5/4/0/−4/−5/−8/−10 dB，每档2474个周期；六类数量[1106,307,297,298,237,229]。',
        '- 已有4/−4/−8 dB的264份MAT直接复用，并重新核查作者残差缩放、SHA256、时间及转速列。',
        f'- 264份输入的整记录逐相SNR最大误差：{preflight["max_whole_record_per_phase_snr_error_db"]:.16g} dB。',
        '- SNR定义按原整记录逐相计算；不同档位共享同一作者10 dB残差，不是独立噪声实现。',
        '- A1–A5、Xu-Full、B0共7方法×2种子×3档，新增42组；其余70份历史预测按原字节引用。',
        '- A1为本轮固定LP4＋双视图消融，不是历史小波A1，也不同于单视图resnet_lp4。',
        '- 42/44为用户在查看此前六种子结果后指定的子集；此前全部六种子记录保留。',
        '- 本轮不读取Final-Holdout波形，不进行t-SNE或重新训练。','']
    for seed in (42,44):
        lines += [f'## 种子{seed}：Macro-F1（%）','','| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in MODELS:
            values=[next(float(r['macro_f1_percent']) for r in rows if r['method']==method and int(r['seed'])==seed and int(r['snr_db'])==snr) for snr in SNRS]
            lines.append('| '+NAMES[method]+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
        lines.append('')
    lines += ['## 两个指定种子的均值±样本标准差（%）','','| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in MODELS:
        values=[next(r for r in summary if r['method']==method and int(r['snr_db'])==snr) for snr in SNRS]
        lines.append('| '+NAMES[method]+' | '+' | '.join(f"{float(v['macro_f1_mean']):.2f} ± {float(v['macro_f1_sd']):.2f}" for v in values)+' |')
    lines += ['','## 运行与核验','',f'- GPU：{preflight["gpu"]}；PyTorch `{preflight["torch"]}`；复用解释器`{preflight["interpreter"]}`。',
        '- 16份best真实源样本前向检查通过；96组原最佳源验证F1/CE重放一致。',
        '- 112组固定Dev逐周期预测回放一致；112组六类F1、Recall、Accuracy、混淆矩阵、健康误报和故障漏报使用sklearn独立复算。',
        '- 70份被引用的历史预测与原文件逐字节一致，42份新增预测保留完整真值和预测。',
        '- 原84份统一基准权重及20份消融权重的SHA核验通过；推理未改变16个模型的参数或BN。',
        f'- {len(before)}个受保护文件在执行前后指纹一致。',
        '- 以下为各阶段子进程墙钟时间，不含部署、传输及文档整理。','',
        '| 阶段 | 秒 | 退出码 |','|---|---:|---:|']
    for row in timing['jobs']:
        lines.append(f"| {row['stage']} | {row['seconds']:.3f} | {row['exit_code']} |")
    lines += ['',f"- 四阶段合计：{sum(row['seconds'] for row in timing['jobs']):.3f}秒。",'',
        '## 结果文件','',
        '- [112行逐种子结果](output/results/reports/PER_SEED_RESULTS.csv)；[56行均值/样本SD](output/results/reports/SUMMARY.csv)。',
        '- [672行六类指标](output/results/reports/PER_CLASS_PER_SEED.csv)；[98行同种子完整B1配对差值](output/results/reports/PAIRED_FULL_B1_MINUS_MODELS.csv)。',
        '- [16份最佳权重身份与轮次](output/results/reports/SELECTED_CHECKPOINTS.csv)；[96组源验证回放](output/results/reports/SOURCE_VALIDATION_REPLAY.csv)。',
        '- [全部112份逐周期预测](output/results/evaluations/)；[每组结果来源](output/RESULT_PROVENANCE.json)。',
        '- [核验记录](output/VERIFICATION.json)；[数据来源及264份MAT指纹](output/DATA_REFERENCE.json)。',
        '- [原训练元数据](output/checkpoint_metadata/)；[本次入口与配置](code/)；[运行日志](logs/)。',
        '- 原MAT、模型权重和环境留在服务器。','']
    (RUN/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    write(RUN/'PUBLICATION_VERIFICATION.json',{'status':'passed','reused_prediction_files_byte_identical':70,
        'total_prediction_files':112,'protected_files_unchanged':len(before),'source_entry_sha256_unchanged':True})
    files=[p for sub in ('code','logs','output') for p in (RUN/sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [RUN/name for name in ('SERVER_RESULTS.md','STAGE_STATUS.json','DEPLOYMENT_MANIFEST.json','RUN_STAGE.py','EXPORT_RESULTS.py','PUBLICATION_VERIFICATION.json')]
    assert all(p.suffix.lower() not in ('.pt','.mat','.npz') for p in files)
    write(RUN/'EXPORT_MANIFEST.json',{p.relative_to(RUN).as_posix():sha(p) for p in files})
    files.append(RUN/'EXPORT_MANIFEST.json')
    archive=Path('/tmp/hust_b1_seven_snr_fixed_weights_20261016.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):z.write(path,path.relative_to(RUN).as_posix())
    print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}))


if __name__=='__main__':
    main()
