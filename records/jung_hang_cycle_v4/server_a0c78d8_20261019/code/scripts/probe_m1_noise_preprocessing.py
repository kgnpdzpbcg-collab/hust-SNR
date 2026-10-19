"""核对M1实际训练增强与实测10dB的低电流相处理。

只重放各最佳权重所经历的增强输入，不更新模型；重放首轮输入必须
匹配原V1哈希。实测噪声仅检查健康/故障段幅度，不推定作者生成算法。
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import sys
import numpy as np


def stats(values):
    """有限观测的经验中位数和四分位数；不报告独立样本置信区间。"""
    a=np.asarray([v for v in values if v is not None],dtype=float)
    if not len(a):return {'n':0}
    return {'n':int(a.size),'q25':float(np.quantile(a,.25)),
            'median':float(np.median(a)),'q75':float(np.quantile(a,.75))}


def noise_levels(raw, seed, epoch):
    """逐字对应V1的SNR抽样和每记录随机数流，返回原分辨率含噪电流。"""
    rng=np.random.default_rng(seed+15000+epoch)
    levels={rid:int(rng.choice([10,15,20,25,30,35,40])) for rid in sorted(raw)}
    noisy={}
    for i,rid in enumerate(sorted(raw)):
        current=raw[rid][1]
        rng=np.random.default_rng(seed+15000+epoch+997*(i+1))
        noise=rng.standard_normal(current.shape)
        power=np.mean(current**2,axis=0)
        noise*=np.sqrt(power/np.mean(noise**2,axis=0))/(10**(levels[rid]/20))
        noisy[rid]=current+noise
    return levels,noisy


def segment_noise(time, clean, noisy, phase, start, end):
    """用健康0.1–0.45s及实际故障周期边界比较物理电流噪声RMS。"""
    noise=noisy[:,phase]-clean[:,phase]
    healthy=(time>=.1)&(time<=.45)
    fault=(time>=start)&(time<=end)
    a=float(np.sqrt(np.mean(noise[healthy]**2)));b=float(np.sqrt(np.mean(noise[fault]**2)))
    return {'healthy_noise_rms':a,'fault_noise_rms':b,'fault_over_healthy_noise_rms':b/a if a>0 else None}


def main(args):
    """生成源增强分支切换计数及目标噪声段幅度；不读取Holdout。"""
    run=args.run_root.resolve();base=run.parents[2];out=args.diagnostics.resolve()
    target=out/'SOURCE_TARGET_NOISE_CHECK.json';assert not target.exists()
    sys.path.insert(0,str(run/'code/scripts'))
    import v1_noise_consistency as v1
    from prepare import preprocess
    from diagnose_m1_snr10 import profile
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    meta=read(base/'outputs/gate0_v1/source_train_windows.json')
    raw=v1.load_source_raw(r['record_id'] for r in meta)
    low_rows=[r for r in meta if any(r['low_phase_flags'])]
    counts=Counter();norm_values=defaultdict(list);source_segments=[];prefix={};hashes={}
    for seed in [42,43,44]:
        best=read(run/f'results/models/augmentation/seed{seed}/run.json')['chosen']['epoch']
        prefix[str(seed)]=best
        for epoch in range(1,best+1):
            levels,noisy=noise_levels(raw,seed,epoch)
            if epoch==1:
                x=np.stack([preprocess(raw[r['record_id']][0],noisy[r['record_id']],r['start'],r['end'])[0] for r in meta])
                digest=hashlib.sha256(x.tobytes()).hexdigest()
                assert digest==read(run/'results/PREFLIGHT.json')['first_epoch_paired_noise_sha256'][str(seed)]
                hashes[str(seed)]=digest
            for row in low_rows:
                rid=row['record_id'];time=raw[rid][0]
                p=profile(time,noisy[rid],row)
                x=preprocess(time,noisy[rid],row['start'],row['end'])[0]
                for phase,low in enumerate(row['low_phase_flags']):
                    if not low:continue
                    flipped=not p['low'][phase]
                    key=(levels[rid],row['label'],'ABCDE'[phase])
                    counts[(key,'views')]+=1;counts[(key,'flips')]+=int(flipped)
                    norm_values[key].append(float(np.sqrt(np.mean(x[phase]**2))))
            if 10 in levels.values():
                for rid in raw:
                    fault=[r for r in meta if r['record_id']==rid and r['label'] in [3,5]]
                    if not fault or levels[rid]!=10:continue
                    phase=0 if fault[0]['label']==3 else 2
                    source_segments.append({'seed':seed,'epoch':epoch,'record_id':rid,
                        **segment_noise(raw[rid][0],raw[rid][1],noisy[rid],phase,min(r['start'] for r in fault),max(r['end'] for r in fault))})
        print('REPLAYED seed',seed,'prefix_epochs',best,flush=True)
    source=[]
    for key in sorted(norm_values):
        total=counts[(key,'views')];flips=counts[(key,'flips')]
        source.append({'snr':key[0],'label':key[1],'phase':key[2],'views':total,'low_to_active_views':flips,
                      'low_to_active_percent':100*flips/total,'normalized_phase_rms':stats(norm_values[key])})
    target_meta=read(base/'outputs/gate0_v1/target_dev_clean_windows.json')
    target_segments=[]
    for rid in sorted({r['record_id'] for r in target_meta if r['label'] in [3,5]}):
        rows=[r for r in target_meta if r['record_id']==rid and r['label'] in [3,5]]
        phase=0 if rows[0]['label']==3 else 2
        t,c,_=v1.target_condition_record(rid,None,run/'target_snr')
        nt,n,_=v1.target_condition_record(rid,10,run/'target_snr');assert np.array_equal(t,nt)
        target_segments.append({'record_id':rid,'phase':'ABCDE'[phase],
            **segment_noise(t,c,n,phase,min(r['start'] for r in rows),max(r['end'] for r in rows))})
    report={'status':'passed','training_prefix_epochs':prefix,'source_low_phase_windows':sum(sum(r['low_phase_flags']) for r in low_rows),
            'first_epoch_inputs_match_original_hashes':hashes,'source_training_views':source,
            'source_10db_segments':source_segments,'target_10db_segments':target_segments,
            'source_10db_fault_over_healthy_noise_rms':stats([r['fault_over_healthy_noise_rms'] for r in source_segments]),
            'target_10db_fault_over_healthy_noise_rms':stats([r['fault_over_healthy_noise_rms'] for r in target_segments]),
            'model_updated':False,'holdout_read':False,'publisher_noise_generation_algorithm_verified':False,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['source_training_views','source_10db_fault_over_healthy_noise_rms','target_10db_fault_over_healthy_noise_rms']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--diagnostics',type=Path,required=True)
    main(parser.parse_args())
