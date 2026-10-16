"""导出可核查的代码、日志、逐周期预测和汇总；不导出MAT、权重或环境。"""
from pathlib import Path
import csv
import hashlib
import io
import json
import shutil
import zipfile

RUN = Path('/home/wang/projects/会议论文/hust/experiments/b1_paper_ablations/d9947f0_20261016')
OUT = RUN / 'results'
CODE = RUN / 'code'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    """将新实验与旧Full B1引用一并列出；均值、SD与所有单种子均保留。"""
    assert read(RUN/'VERIFICATION.json')['status'] == 'passed'
    manifest = read(RUN/'DEPLOYMENT_MANIFEST.json')
    config = read(CODE/'configs/b1_paper_ablations_v1.json')
    pipeline = read(RUN/'PIPELINE_STATUS.json')
    summary = list(csv.DictReader((OUT/'reports/WITH_FULL_B1_SUMMARY.csv').open()))
    per_seed = list(csv.DictReader((OUT/'reports/WITH_FULL_B1_PER_SEED.csv').open()))
    aliases = {'b1':'Full B1（既有最佳权重）', 'a1_fixed_lp4_dual':'A1 固定LP4＋双视图',
        'a2_fixed_noise_proxy':'A2 固定源训练噪声代理', 'a3_no_interphase_descriptors':'A3 屏蔽两组跨相描述',
        'a4_md_only':'A4 仅MD视图', 'a5_no_strength_loss':'A5 去掉强度损失'}
    lines = ['# B1 五项消融服务器实验记录', '', '## Material Passport', '',
        '- 模式：run；数据访问：源仿真与固定实测Dev逐周期数据。',
        '- 核验范围：真实训练完成；最佳权重源验证回放、固定Dev预测回放及独立指标复算通过。未进行第二次完整训练复现。',
        '- 代码分支：`research/b1-paper-ablations-v1`。', f'- 代码提交：`{manifest["commit"]}`。',
        '- 执行日期：2026-10-16（Asia/Shanghai）。', '', '## 数据与训练配置', '',
        f'- 服务器：`wang@172.28.128.160`；GPU：{manifest["gpu"]}。',
        f'- 复用解释器：`{manifest["interpreter"]}`；PyTorch `{manifest["torch"]}`。',
        '- Source-Cycle V2：63个原始仿真文件，962 Train、221 Val、78 Gap；每周期五相×4096点。',
        f'- 源划分SHA256：`{config["source_split_manifest_sha256"]}`。',
        '- Train：Clean/Noisy配对；Noisy为每轮重新生成的10/15/20/25/30/35/40 dB噪声。',
        '- Val：Clean＋固定0/5/10/20/30 dB；只用源域验证选best。',
        '- Adam lr=0.001，weight_decay=0，batch=32，每轮31次更新，联合训练50轮；尾部2个基础周期的批次处理与既有基准一致。',
        '- A2–A5额外10轮源恢复器预训练；A1无可训练恢复器，预训练0轮。',
        '- 种子42、44；5个消融×2种子，共10组新训练；最多6进程，每进程2 CPU线程。',
        '- Target-Dev：固定88个文件编号，每档2474周期；测试5/0/−5/−10 dB。',
        '- 目标噪声沿用作者10 dB残差缩放；未生成新噪声，未读取Final-Holdout波形。',
        '- 原Full B1不重训，引用Source-Cycle V2基准4728e9b的seed42/44最佳权重，并在相同数据上重新回放核对。',
        '- 本轮A1–A5是消融编号，与历史小波A1模型编号不同。A1双视图分类器与历史单视图resnet_lp4也不同。',
        '- 42/44为用户在查看既有六种子结果后指定的展示种子；保留此前全部六种子结果。',
        '- 源验证选模：Clean F1距峰值≤1个百分点；最大五噪声平均F1，其后Clean F1、最小五噪声平均CE、最早epoch。', '',
        '## Macro-F1：两个指定种子的均值±样本标准差（%）', '',
        '| 模型 | 5 dB | 0 dB | −5 dB | −10 dB |', '|---|---:|---:|---:|---:|']
    for method in ['b1',*config['methods']]:
        cells=[]
        for snr in config['target_conditions']:
            row=next(r for r in summary if r['method']==method and r['condition']==f'snr{snr}')
            cells.append(f'{float(row["macro_f1_mean"]):.2f} ± {float(row["macro_f1_sd"]):.2f}')
        lines.append('| '+aliases[method]+' | '+' | '.join(cells)+' |')
    lines += ['', '## 每个实际种子的Macro-F1（%）', '', '| 模型 | 种子 | 5 dB | 0 dB | −5 dB | −10 dB |', '|---|---:|---:|---:|---:|---:|']
    for method in ['b1',*config['methods']]:
        for seed in config['seeds']:
            cells=[]
            for snr in config['target_conditions']:
                row=next(r for r in per_seed if r['method']==method and int(r['seed'])==seed and r['condition']==f'snr{snr}')
                cells.append(f'{float(row["macro_f1_percent"]):.2f}')
            lines.append(f'| {aliases[method]} | {seed} | '+' | '.join(cells)+' |')
    lines += ['', '## 时间与核验', '', f'- 训练编排墙钟时间：{pipeline["training_seconds"]:.3f}秒。',
        f'- 固定权重测试：{pipeline["evaluate"]["seconds"]:.3f}秒；汇总：{pipeline["summarize"]["seconds"]:.3f}秒。',
        f'- 完整训练/测试/汇总流程：{pipeline["elapsed_seconds"]:.3f}秒。',
        '- 227项代码测试通过；5项真实源样本CUDA梯度检查通过。首次全仓测试缺少旧V4测试输入，原样补入已归档输入后复测通过，未改算法或测试代码。',
        '- 60次最佳源验证指标回放一致；40组消融＋8组既有Full B1逐周期预测回放完全一致。',
        '- 48组指标使用sklearn独立复算；500行联合训练历史、80行预训练历史。',
        '- 512个受保护输入、84份历史权重、20份本轮权重的相应指纹核验通过；推理不改变模型参数或BN状态。',
        '- 完整记录见`VERIFICATION.json`；源验证饱和轮次与被选epoch见`results/reports/SOURCE_SELECTED_CHECKPOINTS.csv`。',
        '- 本报告不执行显著性检验；两个展示种子的SD只描述这两次运行。', '', '## 文件', '',
        '- `results/reports/WITH_FULL_B1_PER_SEED.csv`：48行，六模型、两种子、四档条件。',
        '- `results/reports/WITH_FULL_B1_SUMMARY.csv`：24行，均值及样本SD。',
        '- `results/reports/PAIRED_FULL_MINUS_ABLATION.csv`：40行，同种子Full B1减消融的百分点差值。',
        '- `results/reports/WITH_FULL_B1_PER_CLASS.csv`：288行，各类F1与Recall。',
        '- `results/evaluations/`：40份消融逐周期真值/预测与混淆矩阵。',
        '- `full_b1_reference/`：本轮使用的8份既有评价与4份训练记录，保留原字节。',
        '- `code/`：本次运行的源代码、配置、测试、文档快照。',
        '- `logs/`：各次训练、测试及预检日志；权重与MAT只保留服务器。', '']
    (RUN/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    files={}
    for folder in ('scripts','configs','tests','docs'):
        for path in (CODE/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                files['code/'+path.relative_to(CODE).as_posix()]=path
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix in ('.csv','.json'):
            files[path.relative_to(RUN).as_posix()]=path
    for path in (RUN/'logs').glob('*'):
        if path.is_file(): files[path.relative_to(RUN).as_posix()]=path
    for name in ('SERVER_RESULTS.md','VERIFICATION.json','DEPLOYMENT_MANIFEST.json','PIPELINE_STATUS.json',
        'PROTECTED_INPUTS_BEFORE.json','PROTECTED_WEIGHTS_BEFORE.json','BASELINE_FILES_SHA256.json',
        'TARGET_DATA_BINDING.json','RUN_PIPELINE.py','VERIFY_RESULTS.py','EXPORT_RESULTS.py'):
        files[name]=RUN/name
    base=Path(manifest['baseline_root'])
    for relative in read(RUN/'BASELINE_FILES_SHA256.json'):
        files['full_b1_reference/'+relative]=base/relative
    exported={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    (RUN/'EXPORT_MANIFEST.json').write_text(json.dumps(exported,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')
    files['EXPORT_MANIFEST.json']=RUN/'EXPORT_MANIFEST.json'
    archive=Path('/tmp/hust_b1_ablations_d9947f0_results.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zip:
        for name,path in sorted(files.items()): zip.write(path,name)
    print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))


if __name__=='__main__':
    main()
