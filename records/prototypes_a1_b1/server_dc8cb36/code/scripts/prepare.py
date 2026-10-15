"""HUST完整周期预处理：原始数据只读，输出独立的先导实验数据。

源域先按工况划分；相同健康记录只保留一个确定性来源。目标Clean/10dB
共用Clean时间边界，Holdout波形未上传，不能被此入口用于选模或评估。
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gate0_v1'

def read(path):
    """读取已冻结JSON，不从文件名猜测标签。"""
    return json.loads(Path(path).read_text())

def dump(path,obj):
    """保存可阅读中文元数据，路径必须由本项目指定。"""
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def sha(path):
    """逐块校验输入、分组和权重的原字节。"""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def matrix(path,field):
    """只接受审计确认的MAT字段和有限数值矩阵。"""
    value=loadmat(path,variable_names=[field])[field]
    assert value.dtype==np.float64 and np.isfinite(value).all()
    return value

def record(name,domain,condition='clean'):
    """源无时间列，以已交叉核对的10kHz重建；实测同时间的测量取平均。

    返回时间、一条记录的5相电流、角速度。时钟处理不使用故障标签，
    同一目标记录的Clean/noisy用完全相同的合并规则。
    """
    if domain=='source':
        value=matrix(ROOT/'data/source'/f'{name}.mat','I_We')
        assert value.shape==(10000,6)
        return np.arange(10000)/10000,value[:,:5].copy(),value[:,-1].copy()
    value=matrix(ROOT/'data/target_dev'/condition/f'{name}.mat','name' if condition=='clean' else 'I_noisy')
    assert value.shape==(10031,7)
    if condition!='clean':
        clean=matrix(ROOT/'data/target_dev/clean'/f'{name}.mat','name')
        assert np.array_equal(value[:,0],clean[:,0]) and np.array_equal(value[:,-1],clean[:,-1])
    time,start,count=np.unique(value[:,0],return_index=True,return_counts=True)
    merged=np.add.reduceat(value[:,1:],start,axis=0)/count[:,None]
    assert np.all(np.diff(time)>0)
    return time,merged[:,:5],merged[:,-1]

def windows(time,omega):
    """累计电角度划完整周期；稳态区间预先固定，不依赖预测结果。

    电角速度变动时用梯形积分跟踪2π相位，稳定速度时等价T=2π/omega。
    健康0.1–0.45s；故障从0.5s后两周期开始，到0.98s结束。
    """
    assert np.all(omega>0)
    angle=np.r_[0,np.cumsum((omega[1:]+omega[:-1])*.5*np.diff(time))]
    bounds=[]
    for state,first,last in [(0,float(np.interp(.1,time,angle)),float(np.interp(.45,time,angle))),
                             (1,float(np.interp(.5,time,angle)+4*np.pi),float(np.interp(.98,time,angle)))]:
        count=int(np.floor((last-first)/(2*np.pi)))
        for index in range(count):
            a,b=np.interp([first+index*2*np.pi,first+(index+1)*2*np.pi],angle,time)
            bounds.append({'state':state,'start':float(a),'end':float(b)})
    return bounds

def preprocess(time,currents,start,end):
    """按原论文MD式(6)/(7)处理五相，再线性重采样完整周期为128点。

    极值中心；meanabs>=0.2除最大绝对值，低电流相除100，保护断相特征。
    先取完整周期的原分辨率测量，再归一化，不在归一化后添加噪声。
    """
    knots=np.r_[start,time[(time>start)&(time<end)],end]
    raw=np.stack([np.interp(knots,time,currents[:,i]) for i in range(5)])
    centered=raw-(raw.max(1)+raw.min(1))[:,None]/2
    low=np.mean(np.abs(centered),axis=1)<.2
    divisor=np.where(low,100.,np.max(np.abs(centered),axis=1))
    assert np.all(divisor>0)
    normalized=centered/divisor[:,None]
    grid=np.linspace(start,end,128)
    x=np.stack([np.interp(grid,knots,channel) for channel in normalized]).astype(np.float32)
    assert x.shape==(5,128) and np.isfinite(x).all()
    return x,low.tolist()

def prepare():
    """验证输入清单后构造四组数据；目标仅开发集，源Test单独保存。"""
    manifest=read(ROOT/'configs/TRANSFER_MANIFEST_gate0_v1.json')
    for name,info in manifest['files'].items():
        path=ROOT/name;assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],name
    split=read(ROOT/'configs/splits_gate0_v1.json');labels=read(ROOT/'analysis/data_audit_v1/LABELS.json')
    assert set(split['target_dev_records']).isdisjoint(split['target_holdout_records'])
    assert not (ROOT/'data/target_holdout').exists()
    healthy_allowed=set()
    for group in split['source_healthy_duplicate_groups']:healthy_allowed.add(min(group))
    for name in labels['simulation']:
        if not any(name in group for group in split['source_healthy_duplicate_groups']):healthy_allowed.add(name)
    sets={}
    for part,names in split['source_records'].items():
        x,y,meta=[],[],[]
        for name in names:
            time,current,omega=record(name,'source')
            for w in windows(time,omega):
                if w['state']==0 and name not in healthy_allowed:continue
                label=0 if w['state']==0 else labels['simulation'][name]['fault_label']
                value,low=preprocess(time,current,w['start'],w['end']);x.append(value);y.append(label)
                meta.append({'record_id':name,'label':label,'start':w['start'],'end':w['end'],'low_phase_flags':low})
        sets['source_'+part]=(np.stack(x),np.array(y,dtype=np.int64),meta)
    clean_x,noise_x,y,meta=[],[],[],[]
    for name in split['target_dev_records']:
        time,current,omega=record(name,'target','clean');nt,noise,nw=record(name,'target','snr10')
        assert np.array_equal(time,nt) and np.array_equal(omega,nw)
        for w in windows(time,omega):
            label=0 if w['state']==0 else labels['experimental'][name]['fault_label']
            a,la=preprocess(time,current,w['start'],w['end']);b,lb=preprocess(nt,noise,w['start'],w['end'])
            clean_x.append(a);noise_x.append(b);y.append(label)
            meta.append({'record_id':name,'label':label,'start':w['start'],'end':w['end'],'clean_low_phase_flags':la,'snr10_low_phase_flags':lb})
    sets['target_dev_clean']=(np.stack(clean_x),np.array(y,dtype=np.int64),meta)
    sets['target_dev_snr10']=(np.stack(noise_x),np.array(y,dtype=np.int64),meta)
    assert not (OUT/'DATA_PREPARATION.json').exists()
    OUT.mkdir(parents=True,exist_ok=True);summary={}
    for key,(x,y,meta) in sets.items():
        assert set(y.tolist())==set(range(6))
        np.savez_compressed(OUT/(key+'.npz'),x=x,y=y)
        dump(OUT/(key+'_windows.json'),meta)
        summary[key]={'samples':len(y),'class_counts':np.bincount(y,minlength=6).tolist(),'shape':list(x.shape),'tensor_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),'npz_sha256':sha(OUT/(key+'.npz'))}
    dump(OUT/'DATA_PREPARATION.json',{'status':'passed','source_records':{k:len(v) for k,v in split['source_records'].items()},'target_dev_records':88,'target_holdout_records':347,'holdout_waveforms_present':False,'sets':summary,'input_files_verified':len(manifest['files']),'protocol_sha256':sha(ROOT/'configs/protocol_gate0_v1.json'),'split_sha256':sha(ROOT/'configs/splits_gate0_v1.json')})
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':prepare()
