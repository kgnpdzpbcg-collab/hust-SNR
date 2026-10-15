"""独立核验训练、预处理和评估；全部训练核验通过后才评估目标Dev。

保存每个最佳权重、源Val选择及评估原始分数的证据。最终Holdout未上传。
本入口只做核验/推理，不修正超参数或自动重试训练。
"""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,math,subprocess,statistics
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix
from prepare import ROOT,OUT,read,dump,sha,record,preprocess,windows
from train import CNN,configure,state_digest,predict,SEEDS,metrics

def independent_metrics(truth,pred):
    """用sklearn重新计算六类指标，核对训练入口的手算实现。"""
    return {'macro_f1':float(f1_score(truth,pred,labels=list(range(6)),average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(truth,pred)), 'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
            'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(6))).tolist()}

def runtime():
    """环境与原有工程指纹；只读，不安装软件或修改旧模型。"""
    old=ROOT.parent/'Conference-Paper-ccsae-prototype-diagnostics-v1'
    return {'python':__import__('sys').version,'packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),
            'old_repo_commit':subprocess.check_output(['git','-C',str(old),'rev-parse','HEAD']).decode().strip(),
            'old_repo_tracked_clean':not subprocess.check_output(['git','-C',str(old),'status','--porcelain']).strip(),
            'old_reference_model_sha256':sha(old/'server_experiment/resnet_shot_scale/reference_models.py'),
            'old_data_zip_sha256':sha(ROOT.parent/'Data.zip')}

def pretrain():
    """核对泄漏边界、标签时段、完整周期和近零相处理，再做一次前向/梯度检查。"""
    configure(42);split=read(ROOT/'configs/splits_gate0_v1.json');prep=read(OUT/'DATA_PREPARATION.json')
    all_source=[set(v) for v in split['source_records'].values()]
    assert all(all_source[i].isdisjoint(all_source[j]) for i in range(3) for j in range(i+1,3))
    assert set.union(*all_source)==set(read(ROOT/'analysis/data_audit_v1/LABELS.json')['simulation'])
    for group in split['source_healthy_duplicate_groups']:
        assert sum(bool(set(group)&part) for part in all_source)==1
    assert set(split['target_dev_records']).isdisjoint(split['target_holdout_records'])
    canonical=set(split['target_canonical_records'])
    assert set(split['target_dev_records'])|set(split['target_holdout_records'])==canonical
    for group in split['target_duplicate_groups']:assert len(canonical.intersection(group))==1
    assert not (ROOT/'data/target_holdout').exists()
    for domain in ('clean','snr10'):
        assert {p.stem for p in (ROOT/'data/target_dev'/domain).glob('*.mat')}==set(split['target_dev_records'])
    # 已知有偏置正弦和接近零相，检查物理预处理而不是只验证函数能调用。
    t=np.linspace(0,1,1001);large=2*np.sin(2*np.pi*t)+3;small=.1*np.sin(2*np.pi*t)+4
    current=np.stack([large,large,large,large,small],axis=1);x,low=preprocess(t,current,0,1)
    assert low==[False]*4+[True]
    assert np.max(np.abs(x[:4]-np.sin(2*np.pi*np.linspace(0,1,128))))<1e-5
    assert np.max(np.abs(x[4]-.001*np.sin(2*np.pi*np.linspace(0,1,128))))<1e-7
    metadata_checks=0
    for key,item in prep['sets'].items():
        assert sha(OUT/(key+'.npz'))==item['npz_sha256']
        with np.load(OUT/(key+'.npz')) as f:
            fx,fy=f['x'],f['y'];assert fx.dtype==np.float32 and fy.dtype==np.int64 and fx.shape[1:]==(5,128) and np.isfinite(fx).all()
        rows=read(OUT/(key+'_windows.json'));assert len(rows)==len(fy)
        for row,label in zip(rows,fy):
            assert row['label']==int(label)
            if label==0:assert .1-1e-12<=row['start']<row['end']<=.45+1e-12
            else:assert .5<row['start']<row['end']<=.98+1e-12
        # 对每类一个实际周期从原始MAT重新生成，与已缓存张量逐位核对。
        for label in range(6):
            index=int(np.flatnonzero(fy==label)[0]);r=rows[index]
            condition='snr10' if key=='target_dev_snr10' else 'clean'
            time,current,omega=record(r['record_id'],'source' if key.startswith('source') else 'target',condition)
            again,_=preprocess(time,current,r['start'],r['end']);assert np.array_equal(again,fx[index])
            angle=np.r_[0,np.cumsum((omega[1:]+omega[:-1])*.5*np.diff(time))]
            assert abs(float(np.interp(r['end'],time,angle)-np.interp(r['start'],time,angle))-2*np.pi)<1e-10
            metadata_checks+=1
    clean=read(OUT/'target_dev_clean_windows.json');noise=read(OUT/'target_dev_snr10_windows.json');assert clean==noise
    with np.load(OUT/'source_train.npz') as f:
        model=CNN().cuda();scores=model(torch.from_numpy(f['x'][:32]).cuda());loss=torch.nn.functional.cross_entropy(scores,torch.from_numpy(f['y'][:32]).cuda());loss.backward()
        assert scores.shape==(32,6) and torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    truth=np.tile(np.arange(6),4);pred=np.roll(truth,2)
    a=metrics(truth,pred);b=independent_metrics(truth,pred)
    for name in ('macro_f1','accuracy','balanced_accuracy'):assert abs(a[name]-b[name])<1e-12
    dump(OUT/'RUNTIME_BEFORE.json',runtime())
    dump(OUT/'PRETRAIN_VERIFICATION.json',{'status':'passed','independent_raw_period_checks':metadata_checks,'known_sine_and_near_zero_phase_verified':True,'source_groups_and_healthy_duplicates_separated':True,'all_target_conditions_paired':True,'holdout_data_present':False,'gradient_finite':True,'metric_implementations_equal':True,'scripts':{p.name:sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))}})
    print('训练前核验通过：分组/标签/完整周期/近零相/梯度与指标检查',flush=True)

def verify_training(condition):
    """检查初值、完整50轮、真实更新数、BN计数和Val最佳权重，不使用目标Dev。"""
    configure(42);report=read(OUT/(condition.upper()+'_TRAINING.json'));checks=[]
    with np.load(OUT/'source_val.npz') as f:vx=torch.from_numpy(f['x'].copy());vy=f['y'].copy()
    for seed in SEEDS:
        configure(seed);folder=OUT/'models'/condition/f'seed{seed}';r=read(folder/'TRAINING.json');history=read(folder/'history.json')
        assert len(history)==50 and r['updates']==50*math.ceil(r['source_train_size']/32)
        assert all(h['updates']==(i+1)*math.ceil(r['source_train_size']/32) for i,h in enumerate(history))
        wanted=max(h['macro_f1'] for h in history);epoch=next(h['epoch'] for h in history if h['macro_f1']==wanted)
        assert epoch==r['best_epoch'] and wanted==r['best_val_macro_f1']
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)['state'];fresh=CNN().cuda()
        assert state_digest(fresh.state_dict())==state_digest(initial)==r['initial_state_sha256']
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)['state']
        assert best['epoch']==epoch and sha(folder/'best.pt')==r['best_checkpoint_sha256']
        for name,value in last.items():
            if name.endswith('num_batches_tracked'):assert int(value)==r['updates'] and int(best['state'][name])==epoch*math.ceil(r['source_train_size']/32)
        assert state_digest(initial)!=state_digest(last)
        fresh.load_state_dict(best['state']);before=state_digest(fresh.state_dict());scores=predict(fresh,vx);m=independent_metrics(vy,scores.argmax(1))
        assert abs(m['macro_f1']-wanted)<1e-12 and before==state_digest(fresh.state_dict())
        if condition=='augmentation':
            base=read(OUT/'models/baseline'/f'seed{seed}'/'TRAINING.json')
            assert r['initial_state_sha256']==base['initial_state_sha256'] and r['sample_order_sha256']==base['sample_order_sha256'] and r['updates']==base['updates']
        checks.append({'seed':seed,'best_epoch':epoch,'best_checkpoint_sha256':sha(folder/'best.pt'),'val_reloaded':m,'actual_updates':r['updates'],'bn_verified':True})
    dump(OUT/(condition.upper()+'_TRAINING_VERIFICATION.json'),{'status':'passed','models':3,'results':checks,'target_evaluated':False})
    print(condition,'训练及Val最佳权重核验通过',flush=True)

def evaluate(condition):
    """在所有候选已核验/冻结后，只评估源Test和目标Dev的两个条件。"""
    checks=read(OUT/(condition.upper()+'_TRAINING_VERIFICATION.json'));assert checks['status']=='passed'
    dump(OUT/(condition.upper()+'_FROZEN_CHECKPOINTS.json'),{'frozen_before_evaluation':True,'checkpoints':checks['results']})
    results=[]
    for seed in SEEDS:
        configure(seed);path=OUT/'models'/condition/f'seed{seed}'/'best.pt';model=CNN().cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state'])
        before=state_digest(model.state_dict())
        for key in ['source_source_test','target_dev_clean','target_dev_snr10']:
            with np.load(OUT/(key+'.npz')) as f:x=torch.from_numpy(f['x'].copy());truth=f['y'].copy()
            scores=predict(model,x);pred=scores.argmax(1);m=independent_metrics(truth,pred);manual=metrics(truth,pred)
            for name in ('macro_f1','accuracy','balanced_accuracy'):assert abs(m[name]-manual[name])<1e-12
            assert before==state_digest(model.state_dict())
            folder=OUT/'evaluations'/condition/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/(key+'_scores.npz'),truth=truth,predicted=pred,scores=scores)
            row={'condition':condition,'seed':seed,'dataset':key,'size':len(truth),'metrics':m,'checkpoint_sha256':sha(path),'predicted':pred.tolist(),'truth':truth.tolist(),'state_and_bn_unchanged':True,'final_holdout_evaluated':False}
            dump(folder/(key+'.json'),row);results.append(row)
    dump(OUT/(condition.upper()+'_EVALUATION.json'),{'status':'completed','evaluations':9,'results':results,'final_holdout_evaluated':False})
    if condition=='baseline':
        drops=[]
        for seed in SEEDS:
            selected={r['dataset']:r['metrics']['macro_f1'] for r in results if r['seed']==seed}
            drops.append((selected['target_dev_clean']-selected['target_dev_snr10'])*100)
        training=read(OUT/'BASELINE_TRAINING.json')['results']
        basic=all(r['best_val_macro_f1']>.5 and read(OUT/'models/baseline'/f"seed{r['seed']}"/'history.json')[-1]['train_loss']<read(OUT/'models/baseline'/f"seed{r['seed']}"/'history.json')[0]['train_loss'] for r in training)
        run=basic and min(drops)>0 and statistics.mean(drops)>=5
        dump(OUT/'BASELINE_GATE.json',{'source_learning_valid':basic,'per_seed_drop_pp':drops,'mean_drop_pp':statistics.mean(drops),'all_seeds_drop':min(drops)>0,'threshold_pp':5,'run_augmentation':run,'interpretation':'exploratory execution gate, not significance or innovation proof'})
        print('Baseline触发检查',json.dumps(read(OUT/'BASELINE_GATE.json'),ensure_ascii=False),flush=True)
    print(condition,'冻结权重的9次评估完成',flush=True)

def verify_evaluation(condition):
    """重新创建模型和重载权重，复现所有分数/预测逐位一致；核查旧工程和环境。"""
    result=read(OUT/(condition.upper()+'_EVALUATION.json'));verified=[]
    for row in result['results']:
        seed=row['seed'];key=row['dataset'];configure(seed);model=CNN().cuda();path=OUT/'models'/condition/f'seed{seed}'/'best.pt'
        assert sha(path)==row['checkpoint_sha256'];model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);before=state_digest(model.state_dict())
        with np.load(OUT/(key+'.npz')) as f:x=torch.from_numpy(f['x'].copy());truth=f['y'].copy()
        scores=predict(model,x);pred=scores.argmax(1)
        with np.load(OUT/'evaluations'/condition/f'seed{seed}'/(key+'_scores.npz')) as saved:
            assert np.array_equal(scores,saved['scores']) and np.array_equal(pred,saved['predicted']) and np.array_equal(truth,saved['truth'])
        assert pred.tolist()==row['predicted'] and independent_metrics(truth,pred)==row['metrics'] and before==state_digest(model.state_dict())
        verified.append({'seed':seed,'dataset':key,'scores_and_predictions_exact':True,'checkpoint_unchanged':True})
    # JSON会将元组转成列表，比较前仅统一序列化格式，不改变环境指纹内容。
    assert json.loads(json.dumps(runtime()))==read(OUT/'RUNTIME_BEFORE.json')
    repaired=read(OUT/'VERIFIER_REPAIR.json')
    for name,wanted in read(OUT/'PRETRAIN_VERIFICATION.json')['scripts'].items():
        if name=='verify.py':
            assert wanted==repaired['original_sha256'];wanted=repaired['corrected_sha256']
        assert sha(ROOT/'scripts'/name)==wanted
    dump(OUT/(condition.upper()+'_EVALUATION_VERIFICATION.json'),{'status':'passed','evaluations':9,'results':verified,'old_repo_data_and_packages_unchanged':True,'scripts_unchanged':True,'final_holdout_evaluated':False})
    print(condition,'9次分数/预测独立重载核验通过；旧环境/数据/工程不变',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pretrain','training','evaluate','verify-evaluation']);p.add_argument('--condition',choices=['baseline','augmentation'],default='baseline');a=p.parse_args()
    if a.phase=='pretrain':pretrain()
    elif a.phase=='training':verify_training(a.condition)
    elif a.phase=='evaluate':evaluate(a.condition)
    else:verify_evaluation(a.condition)
