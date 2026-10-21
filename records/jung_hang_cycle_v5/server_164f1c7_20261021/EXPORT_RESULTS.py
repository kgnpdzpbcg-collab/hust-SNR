"""导出V5独立记录；逐项核对共同锚点，历史拼接表单独标明队列差异。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile

RUN=Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v5/164f1c7_20261021')
OUT=RUN/'results'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8',newline='\n')


def rows(path):
    with path.open(encoding='utf8',newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    """只在训练及回放成功后导出；原波形、权重及缓存留服务器。"""
    pipeline=read(RUN/'PIPELINE_STATUS.json');assert pipeline['state']=='completed'
    assert read(OUT/'VERIFICATION.json')['status']=='passed'
    m=read(RUN/'DEPLOYMENT_MANIFEST.json');c=m['config']
    for name,h in m['code_sha256'].items():assert sha(RUN/'code'/name)==h,name
    protected=read(RUN/'PROTECTED_FILES_BEFORE.json')
    for name,h in protected.items():assert sha(Path(name))==h,name
    weights={str(p):sha(p) for p in (OUT/'models').rglob('*.pt')};assert len(weights)==8
    write(OUT/'reports/WEIGHTS_SHA256.json',weights)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path.insert(0,str(RUN/'code/scripts'))
    import numpy as np
    import jung_hang_cycle_v5 as runner
    args=argparse.Namespace(config=RUN/'code/configs/jung_hang_v5_v1.json',reference_root=Path(m['reference_root']),
        source_root=Path(m['source_root']),additional_root=Path(m['additional_root']),previous_output=Path(m['previous_output']),output=OUT)
    _,ref,_,_,_=runner.setup(args)
    # 从原始冻结元数据独立重建二周期锚点，不再次读取实测波形。
    from prepare import OUT as metadata_root,read as read_metadata
    from jung_hang_models_v5 import forward_two_cycle_indices
    metadata=read_metadata(metadata_root/'target_dev_clean_windows.json')
    expected=np.array([row[0] for row in forward_two_cycle_indices(metadata)])
    truth=np.array([metadata[i]['label'] for i in expected])
    assert len(metadata)==2474 and len(expected)==2298
    for path in (OUT/'evaluations').rglob('*.json'):
        obj=read(path)
        np.testing.assert_array_equal(obj['original_cycle_anchors'],expected)
        np.testing.assert_array_equal(obj['truth'],truth)
        assert obj['windows']==len(expected) and len(obj['predicted'])==len(expected)
    total_steps=0;training_epochs=0
    for method in c['models']:
        for seed in c['seeds']:
            history=read(OUT/f'models/{method}/seed{seed}/history.json')
            assert len(history)==c['epochs_by_method'][method]
            assert all(r['raw_training_noise_sha256'] is not None for r in history)
            training_epochs+=len(history);total_steps+=sum(r['optimizer_steps'] for r in history)
    assert training_epochs==160 and total_steps==4780
    # 保存上游汇总器引用文件，明确历史2474周期与V5共同2298锚点不是同一评价队列。
    previous=Path(m['previous_output'])/'results/reports/PER_SEED_RESULTS.csv'
    assert len(rows(previous))==168
    (OUT/'reports/REFERENCE_12_MODELS_PER_SEED.csv').write_bytes(previous.read_bytes())
    write(OUT/'reports/REFERENCE_PROVENANCE.json',{'path':str(previous),'sha256':sha(previous),'rows':168,
        'record_commit':'4a6f4b9590fd6e88e46bb3dc579c7f966d74b2fe'})
    cohort={'v5_windows_per_condition':len(expected),'historical_windows_per_condition':2474,
        'v5_class_counts':np.bincount(truth,minlength=6).tolist(),'common_v5_anchors_verified_for_all_prediction_sets':True,
        'jung_support':'anchor current cycle + next contiguous same-record same-label cycle',
        'hang_support':'anchor current cycle only; restricted to Jung-compatible anchor IDs',
        'upstream_ALL_14_tables_status':'unchanged raw concatenation of historical twelve-model results and V5; different cohorts; not a matched-cohort comparison',
        'historical_models_retrained_or_reevaluated':False}
    write(OUT/'reports/COHORT_NOTICE.json',cohort)
    verified={'status':'passed','protected_files_unchanged':len(protected),'code_files_verified':len(m['code_sha256']),
              'new_weights_hashed':8,'prediction_sets_with_common_anchors_verified':28,'original_target_cycles':2474,
              'v5_target_windows':len(expected),'training_epochs':training_epochs,'optimizer_steps':total_steps,
              'upstream_code_or_config_changed':False,'historical_cohort_notice_written':True}
    write(RUN/'EXPORT_VERIFICATION.json',verified)
    per_seed=rows(OUT/'reports/PER_SEED_RESULTS.csv');selected=rows(OUT/'reports/SELECTED_CHECKPOINTS.csv')
    alias={'jung':'Jung Modified CNN V5','hang':'Hang WCNN V5'}
    audit=m['source_paper_audit'];input_audit=read(RUN/'SOURCE_HANG_INPUT_AUDIT.json')
    lines=['# Jung/Hang V5：服务器复现记录','','## Material Passport','',
        '- 模式：run；执行日期：2026-10-21；状态：源训练、源选模、七档实测Dev测试及回放完成。',
        '- 分支：`'+m['model_branch']+'`；代码提交：`'+m['model_commit']+'`。',
        '- 原模型代码、优化器和冻结配置逐字节保留；未根据目标结果改参数或重跑。',
        '- 这是原论文机制导向的五相任务适配，不是作者官方代码的逐参数复现。','','## 数据、输入与训练','',
        '- Source-Cycle V2清单仍为63条记录、962 Train、221 Val、78 Gap；源划分指纹不变。',
        '- Jung按分组内正向连续两周期生成5×256输入，实际Train=884、Val=143；每个分组末端缺下一周期的锚点丢弃。',
        '- Hang单周期电流经基波VSD和逐时刻矢量模长归一化，栅格为1×40×40；Train=962、Val=221。',
        '- Hang默认quiet_ratio=0，模长≤1e-12的点跳过；未添加启发式低电流比例阈值。',
        '- 种子42、44；Jung使用AdaMod，30轮、batch32；Hang使用Adam，50轮、batch32。两者lr=0.001、weight_decay=0。',
        '- Jung训练使用THD+N=3/5/7/10%，健康参考段估计原生THD，故障相持续低电流区限制新增噪声。',
        '- Hang训练使用15/20 dB逐记录逐相AWGN；两模型均为0.5 CE(Clean)+0.5 CE(Noisy)。',
        '- 两模型均用Clean和0/5/10/20/30 dB源验证；验证噪声使用各自HUST-V5方法命名空间，非同一噪声实现。',
        '- 源选模：Clean F1距峰值≤1个百分点，随后依次按五档平均F1、Clean F1、最小五档平均CE、最早轮次排序。',
        '- Jung每轮28次、Hang每轮31次更新；四组共160轮、4780次更新，没有提前停止。',
        '- 本轮未训练或评价PMSM；未读取347条Final-Holdout波形。',
        '- 42/44种子沿用用户在查看历史六种子结果后的指定；本轮没有增加种子。','','## 源预检记录','',
        '- 37项CPU测试通过，两模型真实源样本CUDA梯度有限；源输入没有全零样本。',
        '- 63条源健康记录相位拟合最大偏差为0.01107度，支持配置中的通道顺序。此为波形证据，不是硬件接线独立认证；原审计physical标志保持false。',
        f'- 原生THD估计中位数{100*audit["native_thd_median"]:.4f}%，最大{100*audit["native_thd_max"]:.4f}%；来自0.10–0.45秒健康参考段FFT。',
        '- 目标THD+N为3/5/7/10%时，315个记录×通道的非零注噪比例依次为63.49%/90.48%/93.65%/100%；估计原生THD超过目标时按代码不加噪。',
        '- 这些比例依赖当前FFT估计方法，不是独立仪器标定的THD测量。',
        f'- Hang源训练962个样本中有{input_audit["all_unique_input_hashes"]}种不同图像；跨标签完全相同图像共有{input_audit["cross_label_identical_input_hashes"]}组、涉及{input_audit["samples_in_cross_label_identical_input_groups"]}个样本。',
        '- [源输入检查图](SOURCE_HANG_INPUTS.png)：列依次为0–5标签，上行为每类首个训练样本，下行为类平均图。',
        '- [原分支源预检](results/PREFLIGHT_PAPER_AUDIT.json)；[相序拟合](SOURCE_PHASE_ORDER_AUDIT.json)；[预检审阅](PREFLIGHT_REVIEW.json)；[同图跨标签统计](SOURCE_HANG_INPUT_AUDIT.json)。',
        '','## 参数与最佳权重','','| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五档源F1（%） |','|---|---:|---:|---:|---:|---:|']
    for r in selected:
        best=read(OUT/f"models/{r['method']}/seed{r['seed']}/run.json")['selected']
        lines.append(f"| {alias[r['method']]} | {r['seed']} | {r['parameters']} | {r['selected_epoch']} | {100*best['clean_f1']:.2f} | {100*best['noise_mean_f1']:.2f} |")
    lines+=['','## 共同目标评价队列','',
        '- 原固定88个Dev编号、2474个周期中，共2298个锚点具有相邻下一周期。两模型、两种子和七档全部使用同一2298个锚点。',
        '- 真值类别数依次为'+str(cohort['v5_class_counts'])+'。Jung看连续两周期，Hang看锚点单周期。',
        '- 七档为5/4/0/−4/−5/−8/−10 dB，复用已有固定噪声文件；实测不用于参数训练或选模。',
        '- 历史十二模型使用2474周期。上游汇总入口附带的ALL_14文件保持原样，是不同评价队列的原始拼接，本记录没有据此作跨版本/跨模型排名。见[队列说明](results/reports/COHORT_NOTICE.json)。']
    for seed in c['seeds']:
        lines+=['',f'## 种子{seed}：共同2298锚点Macro-F1（%）','','| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in c['models']:
            values=[next(float(r['macro_f1_percent']) for r in per_seed if r['method']==method and int(r['seed'])==seed and int(r['snr_db'])==snr) for snr in c['target_conditions_db']]
            lines.append('| '+alias[method]+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','## 预测、核验与环境','',
        '- Hang两个种子的七档预测均为：全部2298样本预测标签4。Accuracy=9.5735%，Macro-F1=2.9124%。逐周期预测完整保留。',
        '- 28组best目标预测回放、24组best源验证F1/CE和源选模回放均通过；推理未改变权重或BN。',
        '- sklearn独立复算宏F1、准确率、各类F1/Recall及混淆矩阵；另外从冻结元数据重新生成共同锚点并逐文件核对。',
        f'- {len(protected)}个历史数据、代码、权重和结果文件SHA保持不变；8份新权重留服务器，只发布SHA清单。',
        f'- GPU：{m["gpu"]}；PyTorch `{m["torch"]}`；复用解释器`{m["interpreter"]}`，未新建环境。',
        f'- 四进程训练墙钟{pipeline["training_seconds"]:.3f}秒；测试{pipeline["evaluate"]["seconds"]:.3f}秒；核验{pipeline["verify"]["seconds"]:.3f}秒。',
        f'- 训练、测试、核验、汇总总墙钟{pipeline["elapsed_seconds"]:.3f}秒，不含预检、传输及整理。','','## 文件','',
        '- [28行逐种子指标](results/reports/PER_SEED_RESULTS.csv)；[168行逐类指标](results/reports/PER_CLASS_PER_SEED.csv)。',
        '- [160轮训练历史](results/reports/TRAINING_HISTORY.csv)；[完整训练及选模](results/models/)；[24组源验证回放](results/reports/SOURCE_VALIDATION_REPLAY.csv)。',
        '- [28份逐周期真值、预测及原周期锚点](results/evaluations/)；[核验状态](results/VERIFICATION.json)。',
        '- [代码、配置与测试快照](code/)；[日志](logs/)；[部署指纹](DEPLOYMENT_MANIFEST.json)；[导出核验](EXPORT_VERIFICATION.json)。',
        '- 原始MAT、模型权重、环境及高维特征不包含在发布内容中。','']
    (RUN/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    files=[p for folder in ('code','logs','results') for p in (RUN/folder).rglob('*')
           if p.is_file() and p.suffix.lower() not in ('.pt','.pth','.mat','.npz') and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    files += [RUN/name for name in ('DEPLOYMENT_MANIFEST.json','PROTECTED_FILES_BEFORE.json','PIPELINE_STATUS.json','RUN_PIPELINE.py',
        'PREFLIGHT.py','PREFLIGHT_REVIEW.json','AUDIT_INPUTS.py','EXPORT_RESULTS.py','EXPORT_VERIFICATION.json','SERVER_RESULTS.md',
        'SOURCE_PHASE_ORDER_AUDIT.json','SOURCE_HANG_INPUTS.png','SOURCE_HANG_INPUT_AUDIT.json')]
    write(RUN/'EXPORT_MANIFEST.json',{p.relative_to(RUN).as_posix():sha(p) for p in files});files.append(RUN/'EXPORT_MANIFEST.json')
    archive=Path('/tmp/hust_jung_hang_cycle_v5_164f1c7_results.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as target:
        for p in sorted(files):target.write(p,p.relative_to(RUN).as_posix())
    print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}))


if __name__=='__main__':
    main()
