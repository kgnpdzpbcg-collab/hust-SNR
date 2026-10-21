"""补充只读统计：交叉输入敏感模式及128点重采样后的极值保留。

不调用模型、不改变预测。分类计数按周期×种子判决报告，不能解释为
独立采集数量或因果归因比例；极值统计按唯一周期×相报告。
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import numpy as np


def quartiles(values):
    """报告经验分布四分位数，不将周期相关数据转换为置信区间。"""
    return {"n":int(values.size),"q25":float(np.quantile(values,.25)),
            "median":float(np.median(values)),"q75":float(np.quantile(values,.75))}


def main(args):
    """由保存的预测和原输入生成补充记录，禁止覆盖旧结果。"""
    run=args.run_root.resolve();diagnosis=args.diagnostics.resolve()
    target=diagnosis/'ADDITIONAL_CHECKS.json';assert not target.exists()
    with (diagnosis/'CYCLE_ERRORS.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==2474*3
    counts=Counter();by_class={}
    for row in rows:
        if row['new_error']!='True':continue
        y=int(row['label'])
        cn=int(row['clean_with_noise_parameters_prediction'])!=y
        nc=int(row['noise_with_clean_parameters_prediction'])!=y
        key=f'CN_wrong={cn},NC_wrong={nc}'
        counts[key]+=1;by_class.setdefault(str(y),Counter())[key]+=1
    with np.load(run/'dev_clean.npz') as f:clean=f['x'].copy();labels=f['y'].copy()
    with np.load(run/'dev_snr10.npz') as f:noisy=f['x'].copy();assert np.array_equal(labels,f['y'])
    with (diagnosis/'PHASE_PREPROCESSING.csv').open(encoding='utf-8') as f:phases=list(csv.DictReader(f))
    active=np.array([r['clean_low']=='False' for r in phases]).reshape(2474,5)
    cp=np.max(np.abs(clean),axis=2);npk=np.max(np.abs(noisy),axis=2)
    peak={'active_phase_windows':int(active.sum()),'clean_input_abs_peak':quartiles(cp[active]),
          'noise_input_abs_peak':quartiles(npk[active]),'noise_peak_below_0_9':int(np.sum(npk[active]<.9)),
          'per_class':[{'label':y,'clean_peak':quartiles(cp[active & (labels[:,None]==y)]),
                        'noise_peak':quartiles(npk[active & (labels[:,None]==y)])} for y in range(6)]}
    inputs=[diagnosis/'CYCLE_ERRORS.csv',diagnosis/'PHASE_PREPROCESSING.csv',run/'dev_clean.npz',run/'dev_snr10.npz']
    result={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            'new_error_decisions':sum(counts.values()),'sensitivity_patterns':dict(counts),
            'per_class_sensitivity_patterns':{k:dict(v) for k,v in by_class.items()},
            'resampling_peaks':peak,
            'scope':'Fixed-Dev sensitivity patterns, not causal shares. NC uses oracle Clean parameters and exceeds original input range.'}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'new_error_decisions':sum(counts.values()),'sensitivity_patterns':dict(counts),
                     'resampling_peaks':{k:v for k,v in peak.items() if k!='per_class'}},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--diagnostics',type=Path,required=True)
    main(parser.parse_args())
