"""导出两新模型的统一协议记录和十四模型合表，原始输入、权重留服务器。"""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

RUN=Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v2/2b0ab5a_20261019')
OUT=RUN/'results'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert read(RUN/'PIPELINE_STATUS.json')['state']=='completed'
    assert read(OUT/'VERIFICATION.json')['status']=='passed'
    manifest=read(RUN/'DEPLOYMENT_MANIFEST.json')
    for name,h in manifest['code_sha256'].items():assert sha(RUN/'code'/name)==h
    protected=read(RUN/'PROTECTED_FILES_BEFORE.json')
    for name,h in protected.items():assert sha(Path(name))==h
    weights={str(p):sha(p) for p in (OUT/'models').rglob('*.pt')}
    assert len(weights)==8
    (OUT/'reports/WEIGHTS_SHA256.json').write_text(json.dumps(weights,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n')
    # 保留十四模型合表引用的原十二模型CSV原字节及来源指纹。
    previous=Path(manifest['previous_output'])/'results/reports/PER_SEED_RESULTS.csv'
    previous_content=previous.read_bytes()
    (OUT/'reports/REFERENCE_12_MODELS_PER_SEED.csv').write_bytes(previous_content)
    reference={'path':str(previous),'sha256':sha(previous),'record_commit':'4a6f4b9590fd6e88e46bb3dc579c7f966d74b2fe','rows':168}
    (OUT/'reports/REFERENCE_PROVENANCE.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    rows=list(csv.DictReader((OUT/'reports/PER_SEED_RESULTS.csv').open()))
    selected=list(csv.DictReader((OUT/'reports/SELECTED_CHECKPOINTS.csv').open()))
    pipeline=read(RUN/'PIPELINE_STATUS.json');c=manifest['config']
    lines=['# Jung Modified CNN与Hang WCNN：统一周期划分服务器实验','','## Material Passport','',
        '- 执行日期：2026-10-19（Asia/Shanghai）；模式：源训练、固定权重目标Dev测试、预测回放。',
        '- 模型代码分支：`research/noise-consistency-v1`；模型提交：`'+c['model_commit']+'`。',
        '- 网络与输入适配器为仓库近似复现/任务适配，不是作者发布的逐参数原实现。',
        '- 模型结构及`compare_jung_hang.build_inputs`未修改；新增独立统一协议入口。','',
        '## 数据与协议','',
        '- Source-Cycle V2：63条原始仿真记录，962 Train、221 Val、78 Gap，沿用原周期边界。',
        '- 种子42、44；50轮、Adam lr=0.001、weight_decay=0、batch32，每轮31次更新；尾部2个基础样本批次处理保持一致。',
        '- 两模型都使用Clean/Noisy配对CE，各权重0.5；整记录逐相10/15/20/25/30/35/40 dB增强，每轮重抽。',
        '- 同种子、轮次的原始噪声SHA与采样顺序在两模型间完全相同；第1轮噪声与4728e9b基准逐字节一致。',
        '- 验证为Clean及固定0/5/10/20/30 dB；原始验证噪声逐周期与既有基准相同。',
        '- 选模：Clean F1距峰值≤1个百分点；五噪声平均F1、Clean F1、最小五噪声平均CE、最早epoch依次排序。',
        '- Jung原输入为五相MD归一化后128点；Hang原输入为五相128点VSD轨迹、两个64×64通道。',
        '- 与新分支默认训练器不同：采用当前周期划分、42/44种子、五档验证及CE平局规则，并将Hang改为配对噪声训练。论文分支原训练器仍保留。',
        '- 该比较统一原始数据、噪声、更新次数和选模；两个方法的输入表示不同。',
        '- 固定88个实测Dev编号、每档2474周期，测试5/4/0/−4/−5/−8/−10 dB；不重新生成目标噪声。',
        '- 目标Dev仅在源训练和源选模结束后评估；不读取Final-Holdout波形。',
        '- 展示种子42/44来自用户此前在查看六种子结果后的指定，不替代六种子稳定性研究。','',
        '## 参数量与最佳轮次','','| 模型 | 种子 | 参数量 | 源验证选中的epoch |','|---|---:|---:|---:|']
    aliases={'jung':'Jung Modified CNN','hang':'Hang WCNN'}
    for r in selected:lines.append(f"| {aliases[r['method']]} | {r['seed']} | {r['parameters']} | {r['selected_epoch']} |")
    for seed in (42,44):
        lines += ['',f'## 种子{seed}：Macro-F1（%）','','| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in ('jung','hang'):
            values=[next(float(r['macro_f1_percent']) for r in rows if r['method']==method and int(r['seed'])==seed and int(r['snr_db'])==snr) for snr in c['target_conditions_db']]
            lines.append('| '+aliases[method]+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['','## 检查、环境与时间','',
        '- 原模型8项CPU测试通过；两模型真实源样本CUDA梯度检查通过。',
        '- Jung参数量7110；Hang参数量123234。模型默认VSD相轴为0/72/144/216/288度。',
        '- 健康源样本基波相对相位中位数为'+str(manifest['phase_diagnostics']['healthy_source_relative_phase_median_deg'])+'度；相轴未调节。',
        '- 28组Dev预测重新加载best后完全一致；24组最佳源验证F1/CE回放一致；原空间六类指标以sklearn独立复算。',
        '- 同种子两模型全部100个种子/epoch的噪声及采样SHA一致；200轮训练、6200次参数更新，未提前停止。',
        '- '+str(len(protected))+'个旧输入、历史权重及预测文件的指纹未变；推理不改变模型参数或BN。',
        f'- GPU：{manifest["gpu"]}；PyTorch `{manifest["torch"]}`；解释器`{manifest["interpreter"]}`。',
        f'- 四进程并发训练墙钟时间：{pipeline["training_seconds"]:.3f}秒；测试：{pipeline["evaluate"]["seconds"]:.3f}秒；核验：{pipeline["verify"]["seconds"]:.3f}秒。',
        f'- 训练/测试/核验/汇总流程：{pipeline["elapsed_seconds"]:.3f}秒，不含部署、代码检查、传输和文档整理。','',
        '## 文件','',
        '- [28行新模型逐种子指标](results/reports/PER_SEED_RESULTS.csv)；[168行新模型逐类指标](results/reports/PER_CLASS_PER_SEED.csv)。',
        '- [196行十四模型合表](results/reports/ALL_14_MODELS_PER_SEED.csv)；[98行十四模型均值/样本SD](results/reports/ALL_14_MODELS_SUMMARY.csv)。',
        '- [200轮训练历史](results/reports/TRAINING_HISTORY.csv)；[完整训练/选模记录](results/models/)；[24组源验证回放](results/reports/SOURCE_VALIDATION_REPLAY.csv)。',
        '- [28份逐周期真值/预测](results/evaluations/)；[核验状态](results/VERIFICATION.json)；[旧十二模型引用来源](results/reports/REFERENCE_PROVENANCE.json)。',
        '- [运行代码、原模型、配置及测试快照](code/)；[训练和测试日志](logs/)。',
        '- 原始MAT、模型best/last及环境留在服务器，发布权重SHA清单。','']
    (RUN/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    files=[p for folder in ('code','logs','results') for p in (RUN/folder).rglob('*') if p.is_file() and p.suffix!='.pt' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    files += [RUN/name for name in ('DEPLOYMENT_MANIFEST.json','PROTECTED_FILES_BEFORE.json','PIPELINE_STATUS.json','RUN_PIPELINE.py','EXPORT_RESULTS.py','SERVER_RESULTS.md')]
    assert all(p.suffix.lower() not in ('.pt','.pth','.npz','.mat') for p in files)
    manifest_export={p.relative_to(RUN).as_posix():sha(p) for p in files}
    (RUN/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest_export,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8',newline='\n')
    files.append(RUN/'EXPORT_MANIFEST.json')
    archive=Path('/tmp/hust_jung_hang_cycle_v2_2b0ab5a_results_v2.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(RUN).as_posix())
    print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}))


if __name__=='__main__':main()
