"""归档本地V6复现，保留原源码、SNR修复补丁和参考段访问协议证据。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import zipfile

RUN=Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v6/local_4b59ac8c_20261021')
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
    """核验权重、数据指纹及原始队列后导出；不包含原始波形或模型权重。"""
    pipeline=read(RUN/'PIPELINE_STATUS.json');assert pipeline['state']=='completed'
    assert read(OUT/'VERIFICATION.json')['status']=='passed'
    m=read(RUN/'DEPLOYMENT_MANIFEST.json');c=m['config']
    for name,h in m['code_sha256'].items():assert sha(RUN/'code'/name)==h,name
    original=read(RUN/'LOCAL_SOURCE_MANIFEST.json')
    for name,info in original['local_original_files'].items():assert sha(RUN/'local_original'/name)==info['sha256'],name
    protected=read(RUN/'PROTECTED_FILES_BEFORE.json')
    for name,h in protected.items():assert sha(Path(name))==h,name
    weights={str(p):sha(p) for p in (OUT/'models').rglob('*.pt')};assert len(weights)==8
    write(OUT/'reports/WEIGHTS_SHA256.json',weights)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    # 独立元数据核对使用原参考读取器，保证没有因本地脚本目录切换而换数据。
    sys.path.insert(0,str(Path(m['reference_root'])/'code/scripts'))
    import source_cycle_benchmark,source_cycle_split
    from prepare import OUT as metadata_root,read as read_metadata
    import numpy as np
    metadata=read_metadata(metadata_root/'target_dev_clean_windows.json')
    truth=np.array([r['label'] for r in metadata]);assert len(truth)==2474
    for path in (OUT/'evaluations').rglob('*.json'):
        obj=read(path);np.testing.assert_array_equal(obj['truth'],truth)
        assert obj['windows']==2474 and len(obj['predicted'])==2474 and obj['record_families']==88
    total_epochs=0;total_steps=0
    for method in c['models']:
        for seed in c['seeds']:
            history=read(OUT/f'models/{method}/seed{seed}/history.json')
            assert len(history)==c['epochs_by_method'][method]
            assert all(r['training_noise_sha256'] and r['optimizer_steps']==31 for r in history)
            total_epochs+=len(history);total_steps+=sum(r['optimizer_steps'] for r in history)
    assert (total_epochs,total_steps)==(160,4960)
    previous=Path(m['previous_output'])/'results/reports/PER_SEED_RESULTS.csv'
    assert len(rows(previous))==168
    (OUT/'reports/REFERENCE_12_MODELS_PER_SEED.csv').write_bytes(previous.read_bytes())
    write(OUT/'reports/REFERENCE_PROVENANCE.json',{'path':str(previous),'sha256':sha(previous),'rows':168,
        'record_commit':'4a6f4b9590fd6e88e46bb3dc579c7f966d74b2fe'})
    reference=read(RUN/'REFERENCE_PROTOCOL_AUDIT.json')
    notice={'target_records':88,'target_windows':2474,'target_class_counts':np.bincount(truth,minlength=6).tolist(),
        'source_train_cycles':962,'source_val_cycles':221,'reference_window_s':c['reference_window_s'],
        'additional_observation_beyond_current_cycle':True,'source_val_healthy_windows_in_reference':reference['source_val_healthy_windows_overlapping_reference'],
        'target_reference_from_same_noisy_condition':True,'clean_measured_counterpart_used':False,
        'strictly_train_only_fitted_preprocessing':False,
        'ALL_14_tables_status':'original concatenation; cohort IDs match 2474-window historical results, but V6 uses additional per-record healthy reference and overlapping source-Val reference data; not a matched-information protocol comparison',
        'historical_models_retrained':False}
    write(OUT/'reports/PROTOCOL_NOTICE.json',notice)
    write(RUN/'EXPORT_VERIFICATION.json',{'status':'passed','code_files_verified':len(m['code_sha256']),
        'original_local_files_verified':len(original['local_original_files']),'protected_files_unchanged':len(protected),
        'weights_hashed':8,'all_28_target_truth_vectors_verified':True,'target_windows':2474,
        'epochs_verified':total_epochs,'optimizer_steps_verified':total_steps,'validation_snr_fix_disclosed':True,'reference_access_disclosed':True})
    selected=rows(OUT/'reports/SELECTED_CHECKPOINTS.csv');metrics=rows(OUT/'reports/PER_SEED_RESULTS.csv')
    audit=read(OUT/'PREFLIGHT_PAPER_AUDIT.json');aliases={'jung':'Jung V6','hang':'Hang V6'}
    lines=['# 本地Jung/Hang V6：服务器实验记录','','## Material Passport','',
        '- 执行日期：2026-10-21；模式：run；四组训练、28组七档Dev评价和回放已完成。',
        '- 输入版本：用户本地V6，原始七文件快照SHA256：`'+m['local_source_snapshot_sha256']+'`。',
        '- 数据/公共模块底座提交：`'+m['base_commit']+'`；该提交本身不是V6模型提交。',
        '- [原始本地源码快照](local_original/)和[源码指纹](LOCAL_SOURCE_MANIFEST.json)保留。',
        '- 两模型为论文机制指导下的五相任务适配，不是作者官方源码逐参数复现。','','## 验证入口修复','',
        '- 原本地noise_records把Jung固定验证的dB数值传给了THD+N百分比增强函数，原0 dB验证实际未加噪。',
        '- 唯一代码修复：THD+N仅在Jung训练（snr=None）执行；两个模型的固定验证都调用SNR-AWGN。',
        '- 模型结构、训练噪声、优化器、配置及原十项模型测试保持本地原样；新增八项入口回归测试。',
        '- [原问题复现](ORIGINAL_VALIDATION_BUG.json)；[最小补丁](VALIDATION_SNR_FIX.patch)；[入口回归测试](code/tests/test_jung_hang_cycle_v6_noise.py)。','','## 记录级参考段及划分边界','',
        '- V6使用同记录0.10–0.45秒健康参考段估计逐相直流偏置与Hang标量尺度。Jung也使用该段估计的直流偏置。',
        '- 训练和验证输入各自从整条源记录构造参考，参考区间覆盖了38个源验证健康窗口；训练预处理并非只使用Train时段。',
        '- 实测测试从对应SNR的同一含噪记录估计参考，不读取其干净对应波形；参考可能包含当前健康测试窗口之后的采样。',
        '- 这些是本地V6的现有设计，本轮未更改。当前结果记录为带记录级健康参考校准的版本实验，不标为仅当前窗口的严格独立预处理协议。',
        '- [参考段审计](REFERENCE_PROTOCOL_AUDIT.json)；[协议说明](results/reports/PROTOCOL_NOTICE.json)。','','## 数据与训练配置','',
        '- 原Source-Cycle V2清单不变：63记录、962 Train、221 Val、78 Gap；划分SHA256保持冻结值。',
        '- 两模型均使用单电周期的全部962/221窗口；Jung输入5×200，Hang双VSD平面2×40×40。',
        '- Jung：健康参考段整数周期FFT，原生THD仅统计2/3/4阶；训练THD+N=3%，仅标签2/3/5按BLOCKED_PHASES限幅。限幅后不再把噪声重新标定回目标RMS。',
        '- Hang：基波和三次VSD双平面，共享标量尺度max(记录健康模长,当前窗两平面峰值)，训练15/20 dB AWGN。',
        '- 种子42/44；Jung为AdaMod、30轮；Hang为Adam、50轮；lr=0.001、weight_decay=0、batch32。',
        '- 两模型每批配对Clean/Noisy，CE各占0.5；每轮31次更新，共160轮、4960次更新。',
        '- 修复后固定源验证为Clean与0/5/10/20/30 dB AWGN；两方法使用不同HUST-V6方法命名空间的随机噪声。',
        '- 源选模依次使用Clean距峰值≤1个百分点、最大五噪声平均F1、Clean F1、最小五噪声平均CE及最早轮次。',
        '- 实测使用原88个Dev编号、每档2474周期，5/4/0/−4/−5/−8/−10 dB文件不重新生成。',
        '- 实测数据不用于权重拟合或checkpoint选择；上述参考预处理访问已单独披露。未读取347条Final-Holdout波形。',
        '- 42/44种子延续用户在查看历史六种子结果后的指定；本轮未增加种子，也未运行PMSM。','','## 源域预检','',
        '- 正式pytest共26项通过：V3公共网络8项、本地V6模型10项、验证/训练噪声入口回归8项。两模型真实源样本GPU梯度有限。',
        f'- 原生THD中位数{audit["jung_thdn"]["native_thd_percent"]["median"]:.5f}%，最大{audit["jung_thdn"]["native_thd_percent"]["max"]:.5f}%；3%档非零注噪覆盖率100%。',
        f'- Hang训练图不同图像{audit["hang"]["distinct"]}种，跨标签完全相同组{audit["hang"]["duplicate_groups"]}组；验证图不同{audit["hang"]["val_image_audit"]["distinct"]}种，跨标签组{audit["hang"]["val_image_audit"]["duplicate_groups"]}组。',
        '- 健康参考段没有回退到整记录的源文件。相序证据复用同一源清单和同一VSD角度的已验证源波形拟合，最大偏差0.01107度，不表示硬件接线独立认证。',
        '- [源预检](results/PREFLIGHT_PAPER_AUDIT.json)；[真实源记录验证SNR检查](DEPLOYMENT_MANIFEST.json)。','','## 参数及源选模','',
        '| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五噪声源F1（%） |','|---|---:|---:|---:|---:|---:|']
    for r in selected:
        obj=read(OUT/f"models/{r['method']}/seed{r['seed']}/run.json")['selected']
        lines.append(f"| {aliases[r['method']]} | {r['seed']} | {r['parameters']} | {r['selected_epoch']} | {100*obj['clean_f1']:.2f} | {100*obj['noise_mean_f1']:.2f} |")
    for seed in c['seeds']:
        lines+=['',f'## 种子{seed}：Macro-F1（%）','','| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in c['models']:
            values=[next(float(r['macro_f1_percent']) for r in metrics if r['method']==method and int(r['seed'])==seed and int(r['snr_db'])==snr) for snr in c['target_conditions_db']]
            lines.append('| '+aliases[method]+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','## 核验、环境及文件','',
        '- 28组best实测预测、24组best源验证F1/CE与源选模回放均通过；推理不改变模型参数或BN。',
        '- sklearn独立复算Macro-F1、Accuracy、六类F1/Recall和混淆矩阵；冻结元数据真值向量逐文件核对。',
        f'- {len(protected)}个历史输入、代码、权重与结果指纹不变；新best/last共8份留在服务器，归档只含SHA。',
        f'- GPU：{m["gpu"]}；PyTorch `{m["torch"]}`；复用解释器`{m["interpreter"]}`，未新建环境。',
        f'- 并发训练墙钟{pipeline["training_seconds"]:.3f}秒；测试{pipeline["evaluate"]["seconds"]:.3f}秒；核验{pipeline["verify"]["seconds"]:.3f}秒。',
        f'- 训练/评价/核验/汇总总墙钟{pipeline["elapsed_seconds"]:.3f}秒，不含部署、预检和整理。',
        '- [28行逐种子指标](results/reports/PER_SEED_RESULTS.csv)；[168行逐类指标](results/reports/PER_CLASS_PER_SEED.csv)；[160轮历史](results/reports/TRAINING_HISTORY.csv)。',
        '- [完整训练与选模记录](results/models/)；[28份预测](results/evaluations/)；[24组源验证回放](results/reports/SOURCE_VALIDATION_REPLAY.csv)；[核验状态](results/VERIFICATION.json)。',
        '- [实际运行代码](code/)；[训练测试日志](logs/)；[部署清单](DEPLOYMENT_MANIFEST.json)；[导出核验](EXPORT_VERIFICATION.json)。',
        '- [服务器数据路径启动包装器](ENTRY_BOOTSTRAP.py)只预载入既有冻结读取器，不修改V6算法；实际命令见[流程记录](PIPELINE_STATUS.json)。',
        '- 上游汇总器附带十四模型表保持原样，但V6参考段信息访问与旧模型不同；没有据此做直接排名。',
        '- 原始MAT、权重、环境、高维NPZ未包含在发布内容中。','']
    (RUN/'SERVER_RESULTS.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
    files=[p for folder in ('code','logs','results','local_original') for p in (RUN/folder).rglob('*')
           if p.is_file() and p.suffix.lower() not in ('.pt','.pth','.mat','.npz') and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    files += [RUN/name for name in ('LOCAL_SOURCE_MANIFEST.json','DEPLOYMENT_SOURCE_MANIFEST.json','ORIGINAL_VALIDATION_BUG.json','VALIDATION_SNR_FIX.patch',
        'DEPLOYMENT_MANIFEST.json','PROTECTED_FILES_BEFORE.json','REFERENCE_PATHS.json','REFERENCE_PROTOCOL_AUDIT.json','SOURCE_PHASE_ORDER_AUDIT.json',
        'PIPELINE_STATUS.json','RUN_PIPELINE.py','ENTRY_BOOTSTRAP.py','PREFLIGHT.py','EXPORT_RESULTS.py','EXPORT_VERIFICATION.json','SERVER_RESULTS.md')]
    write(RUN/'EXPORT_MANIFEST.json',{p.relative_to(RUN).as_posix():sha(p) for p in files});files.append(RUN/'EXPORT_MANIFEST.json')
    archive=Path('/tmp/hust_jung_hang_cycle_v6_local_4b59ac8c_results.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as target:
        for p in sorted(files):target.write(p,p.relative_to(RUN).as_posix())
    print(json.dumps({'files':len(files),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}))


if __name__=='__main__':
    main()
