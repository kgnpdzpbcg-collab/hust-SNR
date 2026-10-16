"""导出十二模型七档完整结果，保留所有旧预测原字节，不导出电流或权重。"""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

RUN=Path('/home/wang/projects/会议论文/hust/experiments/baseline_four_snr_completion/20261016')
OUT=RUN/'output'
NAMES={'b1':'完整B1','a1_fixed_lp4_dual':'A1 固定LP4＋双视图','a2_fixed_noise_proxy':'A2 固定噪声估计',
    'a3_no_interphase_descriptors':'A3 屏蔽两组跨相描述','a4_md_only':'A4 仅MD视图','a5_no_strength_loss':'A5 去掉强度损失',
    'xu_full':'Xu-Full','b0':'B0','resnet1d':'ResNet1D','cnn1d_plain':'Plain CNN1D','resnet_lp3':'ResNet＋LP3','resnet_lp4':'ResNet＋LP4'}
SNRS=(5,4,0,-4,-5,-8,-10)


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(OUT/'VERIFICATION.json')['status']=='passed'
    assert all(r['exit_code']==0 for r in read(RUN/'STAGE_STATUS.json')['jobs'])
    for rel,h in read(RUN/'DEPLOYMENT_MANIFEST.json')['code_sha256'].items():assert sha(RUN/rel)==h
    rows=list(csv.DictReader((OUT/'results/reports/PER_SEED_RESULTS.csv').open()))
    means=list(csv.DictReader((OUT/'results/reports/SUMMARY.csv').open()))
    for item in read(OUT/'RESULT_PROVENANCE.json'):
        p=OUT/f"results/evaluations/{item['method']}/seed{item['seed']}/target_dev_snr{item['snr_db']}.json"
        assert sha(p)==item['source_sha256']==sha(Path(item['source_path']))
    lines=['# 十二模型七档SNR完整固定权重记录','','## Material Passport','',
        '- 日期：2026-10-16（Asia/Shanghai）；执行内容：LP3补测、旧记录原字节引用、预测重放、汇总。',
        '- 新增6组LP3评价，复用162份旧预测，合计168组；没有重新训练或改选最佳epoch。','',
        '## 数据、权重与补齐范围','',
        '- Source-Cycle V2：原63个仿真文件的962 Train、221 Val、78 Gap；每相4096点，沿用原训练及源验证协议。',
        '- 固定42、44两个指定种子，原24份源验证best。训练提交4728e9b（七个基准模型）与d9947f0（五个消融）。',
        '- 固定88个实测Dev编号，每档2474周期；七档5/4/0/−4/−5/−8/−10 dB。',
        '- ResNet1D、Plain CNN1D、ResNet＋LP4的4/−4/−8 dB共18份旧预测直接引用；ResNet＋LP3这三档新增6份预测。',
        '- 新增噪声仍读取既有264份派生MAT，不重新生成；噪声为作者10 dB残差按整记录逐相缩放。',
        '- 旧八模型112份预测全部保留，四基线的原四档32份预测引用原统一基准；162份引用均逐字节一致。',
        '- 本轮A1为固定LP4＋双视图消融，与单视图ResNet＋LP4独立。',
        '- 42/44是查看此前六种子结果后由用户指定的展示子集；Final-Holdout未评估。','']
    for seed in (42,44):
        lines += [f'## 种子{seed}：Macro-F1（%）','','| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method,label in NAMES.items():
            values=[next(float(r['macro_f1_percent']) for r in rows if r['method']==method and int(r['seed'])==seed and int(r['snr_db'])==snr) for snr in SNRS]
            lines.append('| '+label+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
        lines.append('')
    lines += ['## 核验与文件','',
        '- 168组完整逐周期预测重新加载原权重后回放一致；144组最佳源验证F1/CE重放一致。',
        '- 168组六类F1、Recall、Accuracy、混淆矩阵、健康误报和故障漏报独立复算。',
        '- 推理未改变参数或BN缓冲区，旧权重、原输入及历史预测指纹未变。',
        '- [168行逐种子指标](output/results/reports/PER_SEED_RESULTS.csv)；[84行均值/样本SD](output/results/reports/SUMMARY.csv)。',
        '- [1008行逐类F1/Recall](output/results/reports/PER_CLASS_PER_SEED.csv)；[154行同种子Full B1配对差值](output/results/reports/PAIRED_FULL_B1_MINUS_MODELS.csv)。',
        '- [全部168份预测](output/results/evaluations/)；[原始来源及SHA](output/RESULT_PROVENANCE.json)。',
        '- [144组源验证回放](output/results/reports/SOURCE_VALIDATION_REPLAY.csv)；[核验记录](output/VERIFICATION.json)。',
        '- [运行日志](logs/)；[实际代码及配置](code/)；原MAT和模型权重保留服务器。','']
    (RUN/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    files=[p for sub in ('output','code','logs') for p in (RUN/sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [RUN/name for name in ('SERVER_RESULTS.md','DEPLOYMENT_MANIFEST.json','STAGE_STATUS.json','RUN_STAGE.py','EXPORT_RESULTS.py')]
    assert all(p.suffix.lower() not in ('.mat','.pt','.pth','.npz') for p in files)
    manifest={p.relative_to(RUN).as_posix():sha(p) for p in files}
    (RUN/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n')
    files.append(RUN/'EXPORT_MANIFEST.json')
    archive=Path('/tmp/hust_baseline_four_snr_completion_20261016.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(RUN).as_posix())
    print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}))


if __name__=='__main__':main()
