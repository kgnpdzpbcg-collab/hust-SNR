"""HUST简化先导实验：固定普通CNN，仅训练Baseline及可选噪声增强。

训练输入限于源Train，最佳权重由源Clean-Val决定。三个种子并发，
训练入口不读取任何目标域特征/标签，不引入旧CCSAE代码或目标BN适配。
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,math,multiprocessing as mp,os,random,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from torch import nn
from prepare import ROOT,OUT,read,dump,sha,record,preprocess
SEEDS=(42,43,44)

class CNN(nn.Module):
    """五相×128点→六类；论文式编码器的小CNN，不称为CAE/DML复现。"""
    def __init__(self):
        super().__init__()
        blocks=[]
        for inp,out,kernel in [(5,16,16),(16,32,4),(32,32,4)]:
            blocks.extend([nn.Conv1d(inp,out,kernel),nn.BatchNorm1d(out),nn.LeakyReLU(.01),nn.MaxPool1d(2)])
        self.features=nn.Sequential(*blocks)
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(352,50),nn.LeakyReLU(.01),nn.Linear(50,6))
    def forward(self,x):return self.classifier(self.features(x))

def configure(seed):
    """固定随机性和FP32路径，复用服务器环境；禁止目标域BN统计更新。"""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def state_digest(state):
    """记录参数和BN缓冲原字节，用于冻结、初始化及重载验证。"""
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        array=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()

def cpu_state(model):
    """克隆权重，防止后续更新悄悄修改最佳状态。"""
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def metrics(truth,pred):
    """六类混淆矩阵手算F1；后续用独立实现复核。"""
    cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(truth,pred),1)
    denominator=cm.sum(1)+cm.sum(0)
    f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(6),where=denominator>0)
    recall=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(6),where=cm.sum(1)>0)
    return {'macro_f1':float(f1.mean()),'accuracy':float(np.trace(cm)/cm.sum()),'balanced_accuracy':float(recall.mean()),'confusion_matrix':cm.tolist(),'per_class_f1':f1.tolist()}

@torch.no_grad()
def predict(model,x):
    """使用eval和固定batch128；输入CPU/GPU张量均可，不改变BN状态。"""
    model.eval();scores=[]
    for begin in range(0,len(x),128):scores.append(model(x[begin:begin+128].cuda()).cpu().numpy())
    return np.concatenate(scores)

def load_set(name):
    """只加载准备好的源数据；目标数据在独立评估入口中加载。"""
    with np.load(OUT/(name+'.npz')) as f:return torch.from_numpy(f['x'].copy()).cuda(),torch.from_numpy(f['y'].copy()).cuda()

def augmentation(epoch,seed,meta,raw):
    """按整条源记录功率产生加噪视图，然后复用相同窗口/预处理。

    每条每轮从七个已定SNR抽一档；Gaussian方向逐相按整记录RMS校准。
    即使断相后的电流近零，噪声尺度也由健康+故障整记录计算，不会消失。
    """
    rng=np.random.default_rng(seed+15000+epoch);noisy={}
    for name,(time,current,omega) in sorted(raw.items()):
        snr=int(rng.choice([10,15,20,25,30,35,40]));power=np.mean(current**2,axis=0)
        noise=rng.standard_normal(current.shape)
        noise*=np.sqrt(power/np.mean(noise**2,axis=0))/10**(snr/20)
        noisy[name]=current+noise
    values=[]
    for row in meta:
        time,_,_=raw[row['record_id']]
        value,_=preprocess(time,noisy[row['record_id']],row['start'],row['end']);values.append(value)
    return torch.from_numpy(np.stack(values)).cuda()

def job(condition,seed):
    """完整50轮分类更新；两个条件共享新模型初值和逐轮样本顺序。"""
    configure(seed);folder=OUT/'models'/condition/f'seed{seed}';assert not folder.exists();folder.mkdir(parents=True)
    x,y=load_set('source_train');vx,vy=load_set('source_val');meta=read(OUT/'source_train_windows.json')
    model=CNN().cuda();initial=cpu_state(model);torch.save({'state':initial},folder/'initial.pt')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
    criterion=nn.CrossEntropyLoss();history=[];best=-1.;best_epoch=None;order_digest=hashlib.sha256();updates=0
    raw={name:record(name,'source') for name in read(ROOT/'configs/splits_gate0_v1.json')['source_records']['train']} if condition=='augmentation' else {}
    started=time.perf_counter()
    for epoch in range(1,51):
        model.train();permutation=np.random.default_rng(seed+901+epoch).permutation(len(y));order_digest.update(permutation.astype('<i8').tobytes())
        augmented=augmentation(epoch,seed,meta,raw) if condition=='augmentation' else None
        total,count=0.,0
        for begin in range(0,len(y),32):
            ids=torch.from_numpy(permutation[begin:begin+32]).cuda();optimizer.zero_grad(set_to_none=True)
            if augmented is None:loss=criterion(model(x[ids]),y[ids])
            else:
                scores=model(torch.cat([x[ids],augmented[ids]],dim=0));n=len(ids)
                loss=.5*criterion(scores[:n],y[ids])+.5*criterion(scores[n:],y[ids])
            assert torch.isfinite(loss);loss.backward();optimizer.step();updates+=1
            total+=float(loss.detach())*len(ids);count+=len(ids)
        scores=predict(model,vx);m=metrics(vy.cpu().numpy(),scores.argmax(1))
        history.append({'epoch':epoch,'train_loss':total/count,'updates':updates,**m})
        if m['macro_f1']>best:
            best=m['macro_f1'];best_epoch=epoch
            torch.save({'state':cpu_state(model),'epoch':epoch,'val_macro_f1':best},folder/'best.pt')
        dump(folder/'history.json',history)
        if epoch%10==0:print(f'{condition} seed{seed}: epoch{epoch}/50 loss={total/count:.4f} valF1={m["macro_f1"]:.4f}',flush=True)
    last=cpu_state(model);torch.save({'state':last},folder/'last.pt')
    result={'condition':condition,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,
            'best_val_macro_f1':best,'updates':updates,'epochs':50,'source_train_size':len(y),'source_val_size':len(vy),
            'wall_seconds':time.perf_counter()-started,'initial_state_sha256':state_digest(initial),'last_state_sha256':state_digest(last),
            'best_checkpoint_sha256':sha(folder/'best.pt'),'sample_order_sha256':order_digest.hexdigest(),'test_evaluated':False,
            'no_target_data_read_by_training':True,'forward_views_per_sample':1 if condition=='baseline' else 2}
    dump(folder/'TRAINING.json',result);return result

def run(condition):
    """三个进程复用同一GPU，各4 CPU线程；错误直接报告，不自动换参数重训。"""
    assert read(OUT/'PRETRAIN_VERIFICATION.json')['status']=='passed'
    if condition=='augmentation':assert read(OUT/'BASELINE_GATE.json')['run_augmentation']
    path=OUT/(condition.upper()+'_TRAINING.json');assert not path.exists()
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        tasks=[pool.submit(job,condition,seed) for seed in SEEDS]
        for future in as_completed(tasks):results.append(future.result())
    dump(path,{'status':'completed','models':3,'condition':condition,'wall_seconds':time.perf_counter()-started,'results':sorted(results,key=lambda r:r['seed']),'target_test_evaluated':False})
    print(condition,'三个种子完成',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('condition',choices=['baseline','augmentation']);a=p.parse_args();run(a.condition)
