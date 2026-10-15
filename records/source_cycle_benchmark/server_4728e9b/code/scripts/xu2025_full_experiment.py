"""Xu-2025 full CAE + decoder + classification + DML, source-only experiments.

Two explicitly distinct tracks:
  paper_like: XuCAE on 5x128 MD-preprocessed cycles, CLEAN-only source training;
              16-batch, 50 epochs, optional stated-as-assumption LR decay.
  matched:    XuFullRaw on 5x4096 cycles, original frozen V4/V5 matched 10-40 dB
              paired noise training, 0/5/10/20/30 source validation;
              DEFAULT experiment: xu_full ONLY, as the unmodified full-paper baseline.
              Additional full-backbone modes remain explicitly opt-in; they are
              not used in the current A-prime/V5 paper comparison.

No target-domain training/checkpoint selection and NO FINAL HOLDOUT ACCESS.
No pretrained A' checkpoint is used. All arms independently train FULL Xu encoder.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from xu2025_full_model import XuCAE, XuFullRaw, VARIANTS, xu_joint_loss, parameter_counts
from v1_noise_consistency import fix_randomness, metrics, score_checkpoints
import v2_experiment as data

DEFAULT_SEEDS=(42,43,44)
DEFAULT_CONDITIONS=(None,10,15,20,25,30,35,40,5,0)
SOURCE_VAL_SNRS=(0,5,10,20,30)
DEFAULT_OUT=Path("outputs/xu2025_full")


def result_label(snr:int|None)->str:
    return "clean" if snr is None else f"snr{snr}"


def prepare_source_npz(part:str):
    """For paper_like ONLY: already MD-normalized [N,5,128] archive."""
    from prepare import OUT
    filename=OUT / f"source_{part}.npz"
    with np.load(filename,allow_pickle=False) as f:
        x=f["x"].astype(np.float32)
        y=f["y"].astype(np.int64)
    expected=738 if part=="train" else 302
    if x.shape!=(expected,5,128) or y.shape!=(expected,):
        raise ValueError(f"Frozen normalized source mismatch {filename}")
    return x,y


def make_model(mode:str,track:str)->torch.nn.Module:
    if mode not in VARIANTS:
        raise ValueError(f"Unsupported model: {mode}")
    if track=="paper_like":
        if mode!="xu_full":
            raise ValueError("paper_like track only supports original full XuCAE")
        return XuCAE()
    if track=="matched":return XuFullRaw(mode)
    raise ValueError("Unknown track")


def _loss(model,raw,labels,track,reduction,rec_weight,dml_weight):
    out=(model(raw) if track=="matched" else model(raw))
    target=(out["normalized_input"] if track=="matched" else raw)
    return xu_joint_loss(out,target,labels,reconstruction=reduction,
                         w_recon=rec_weight,w_dml=dml_weight,
                         use_dml=getattr(model,"variant", "xu_full")!="xu_cae_classify")


def paired_source_loss(model,clean,noisy,y,*,reduction="paper_sum",
                       rec_weight=1.,dml_weight=1.):
    """Matched robust protocol: per-view CE/MSE, joint present-class DML.

    Entire mixed batch is passed in one forward (BatchNorm sees both views).
    This noisy-paired objective extends the paper; it is NOT the paper training.
    """
    o=model(torch.cat((clean,noisy),dim=0))
    yc=torch.cat((y,y))
    terms=xu_joint_loss(o,o["normalized_input"],yc,reconstruction=reduction,
                        w_recon=rec_weight,w_dml=dml_weight,
                        use_dml=model.variant!="xu_cae_classify")
    return terms


@torch.no_grad()
def predict(model,arrays:np.ndarray,device,batch:int=128,track="matched"):
    model.eval()
    ys=[]
    for pos in range(0,len(arrays),batch):
        x=torch.as_tensor(arrays[pos:pos+batch],device=device)
        out=model(x)
        ys.append(out["logits"].argmax(1).cpu().numpy())
    return np.concatenate(ys)


@torch.no_grad()
def val_loss(model,arrays,y,device,track,reduction,rec_weight,dml_weight):
    model.eval()
    components=[]
    for pos in range(0,len(y),128):
        xb=torch.from_numpy(arrays[pos:pos+128]).to(device)
        yb=torch.from_numpy(y[pos:pos+128]).to(device)
        terms=_loss(model,xb,yb,track,reduction,rec_weight,dml_weight)
        components.append((float(terms["total"].detach()),len(yb)))
    return sum(v*n for v,n in components)/len(y)


def select_paper_like(history:list[dict]):
    """Paper says lowest loss and highest validation accuracy; no tie algorithm.

    Our declared deterministic approximation: highest Clean Macro-F1, then
    minimum joint validation loss, then earliest epoch.
    """
    return max(history,key=lambda r:(r["clean_f1"],-r["clean_val_total_loss"],-r["epoch"]))


def train(mode:str,seed:int,out:Path,*,track:str="matched",epochs:int=50,
          batch:int|None=None,lr:float=.001,weight_decay:float|None=None,
          source_strong_root:Path|None=None,reconstruction:str="paper_sum",
          rec_weight:float=1.,dml_weight:float=1.,lr_step:int=20,
          lr_gamma:float=.5):
    if epochs<1 or lr<=0 or rec_weight<0 or dml_weight<0:
        raise ValueError("Invalid train budget or weights")
    if batch is None:batch=16 if track=="paper_like" else 32
    if weight_decay is None:weight_decay=1e-4 if track=="paper_like" else 0.
    if batch<=0 or weight_decay<0:raise ValueError("Invalid batch/decay")
    if track=="matched" and source_strong_root is None:
        raise ValueError("Five-SNR matched validation requires --source-strong-root (source/SNR_0/5)")
    if track=="paper_like" and mode!="xu_full":raise ValueError("Paper-like requires xu_full")
    dst=out/track/"models"/mode/f"seed{seed}"
    if dst.exists():raise FileExistsError(f"Refusing to overwrite {dst}")
    fix_randomness(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if track=="paper_like":
        clean_x,y=prepare_source_npz("train")
        valid_x,vy=prepare_source_npz("val")
        train_raw=meta=None
        valid_views={}
    else:
        train_raw,meta,y=data.source_data("train")
        val_raw,vm,vy=data.source_data("val")
        clean_x=data.raw_cycles(train_raw,meta)
        valid_x=data.raw_cycles(val_raw,vm)
        valid_views=data.fixed_noisy_validation(val_raw,vm,noise_levels=SOURCE_VAL_SNRS,
                                                materialized_root=source_strong_root)
    model=make_model(mode,track).to(device)
    init_hash=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                     for p in model.parameters())).hexdigest()
    optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler=(torch.optim.lr_scheduler.StepLR(optim,step_size=lr_step,gamma=lr_gamma)
               if track=="paper_like" and lr_step>0 else None)
    history=[]
    states={}
    for epoch in range(1,epochs+1):
        noisy_x=(data.fresh_noisy_source(train_raw,meta,seed,epoch) if track=="matched" else None)
        perm=np.random.default_rng(seed+901+epoch).permutation(len(y))
        model.train()
        sums={"total":0.,"ce":0.,"recon":0.,"dml":0.,"intra":0.,"inter":0.}
        sample_count=0
        for offset in range(0,len(perm),batch):
            ids=perm[offset:offset+batch]
            xb=torch.from_numpy(clean_x[ids]).to(device)
            yb=torch.from_numpy(y[ids]).to(device)
            optim.zero_grad(set_to_none=True)
            if track=="paper_like":
                terms=_loss(model,xb,yb,track,reconstruction,rec_weight,dml_weight)
            else:
                nb=torch.from_numpy(noisy_x[ids]).to(device)
                terms=paired_source_loss(model,xb,nb,yb,reduction=reconstruction,
                                         rec_weight=rec_weight,dml_weight=dml_weight)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError("Nonfinite Xu CAE/DML objective")
            terms["total"].backward()
            optim.step()
            for name in sums:sums[name]+=float(terms[name].detach())*len(ids)
            sample_count+=len(ids)
        if scheduler:scheduler.step()
        clean_f1=metrics(vy,predict(model,valid_x,device,track=track))["macro_f1"]
        noisy_f1={str(snr):metrics(vy,predict(model,v,device,track=track))["macro_f1"]
                  for snr,v in valid_views.items()}
        row={"epoch":epoch,"train_loss":{k:v/sample_count for k,v in sums.items()},
             "clean_f1":clean_f1,"noisy_f1":noisy_f1,
             "noise_mean_f1":float(np.mean(list(noisy_f1.values()))) if noisy_f1 else None,
             "lr":float(optim.param_groups[0]["lr"])}
        if track=="paper_like":
            row["clean_val_total_loss"]=val_loss(model,valid_x,vy,device,track,
                                       reconstruction,rec_weight,dml_weight)
        history.append(row)
        states[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch in (1,epochs) or epoch%10==0:
            print(f"{track} {mode} seed={seed} epoch={epoch}/{epochs} "
                  f"clean={clean_f1:.4f} noisy={row['noise_mean_f1']} ",flush=True)
    chosen=(select_paper_like(history) if track=="paper_like" else score_checkpoints(history))
    dst.mkdir(parents=True,exist_ok=False)
    saved={"state":states[chosen["epoch"]],"mode":mode,"seed":seed,"track":track,
           "selected":chosen,"source_only":True,"final_holdout_evaluated":False,
           "paper_core_model":"XuCAE_2025_Fig4_TableIII_with_documented_decoder_padding",
           "loss_config":{"reconstruction":reconstruction,"w_recon":rec_weight,"w_dml":dml_weight}}
    torch.save(saved,dst/"best.pt")
    (dst/"history.json").write_text(json.dumps(history,indent=2))
    (dst/"run.json").write_text(json.dumps({
        "mode":mode,"seed":seed,"track":track,"initial_model_sha256":init_hash,
        "selected":chosen,"epochs":epochs,"batch":batch,"lr":lr,
        "weight_decay":weight_decay,"step_lr":lr_step if scheduler else None,
        "gamma":lr_gamma if scheduler else None,
        "source_val_snrs":SOURCE_VAL_SNRS if track=="matched" else [],
        "training_noisy_pair":track=="matched","source_only":True,
        "target_used_for_training_or_selection":False,"final_holdout_evaluated":False,
        "loss_config":saved["loss_config"],"parameter_counts":parameter_counts(model),
        "source_split":"frozen Gate0 34 train / 15 val, 738/302 cycles",
    },indent=2))
    print("SAVED",dst/"best.pt",flush=True)


def target_dev(snr,*,snr_root=None,strong_root=None,export_root=None):
    if export_root is not None:
        return data.target_dev_from_export(snr,export_root)
    root=(strong_root if snr in (0,5) else snr_root)
    return data.target_dev_from_mat(snr,root)


def normalized_target_cycles(raw,meta):
    """Paper-like track: original project 128-point MD preprocessing."""
    from prepare import preprocess
    vals=[]
    for row in meta:
        t,c,_=raw[row["record_id"]]
        x,_=preprocess(t,c,row["start"],row["end"])
        vals.append(x)
    result=np.stack(vals).astype(np.float32)
    if result.shape!=(2474,5,128):raise ValueError("Target-Dev cycle integrity")
    return result


@torch.no_grad()
def evaluate(mode,seed,out,*,track="matched",conditions=DEFAULT_CONDITIONS,
             snr_root=None,strong_root=None,export_root=None):
    from prepare import OUT,read
    folder=out/track/"models"/mode/f"seed{seed}"
    saved=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    if not saved["source_only"] or saved["final_holdout_evaluated"] or saved["mode"]!=mode or saved["track"]!=track:
        raise ValueError("Wrong Xu checkpoint provenance")
    fix_randomness(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=make_model(mode,track).to(device)
    model.load_state_dict(saved["state"],strict=True)
    truths=read(OUT/"target_dev_clean_windows.json")
    if len(truths)!=2474 or len({r["record_id"] for r in truths})!=88:
        raise ValueError("Must use frozen 88/2474 Target-Dev")
    reference_labels=[r["label"] for r in truths]
    out_folder=out/track/"evaluations"/mode/f"seed{seed}"
    if out_folder.exists() and any((out_folder/f"target_dev_{result_label(s)}.json").exists()
                                   for s in conditions):
        raise FileExistsError("Refuse overwrite existing evaluation JSON")
    out_folder.mkdir(parents=True,exist_ok=True)
    for snr in conditions:
        raw,meta=target_dev(snr,snr_root=snr_root,strong_root=strong_root,export_root=export_root)
        if [r["label"] for r in meta]!=reference_labels:
            raise ValueError("Target condition window ordering mismatch")
        input_x=(normalized_target_cycles(raw,meta) if track=="paper_like" else data.raw_cycles(raw,meta))
        pred=predict(model,input_x,device,track=track)
        score=metrics(reference_labels,pred)
        label=result_label(snr)
        record={"track":track,"mode":mode,"seed":seed,"condition":label,
                "windows":2474,"record_families":88,"metrics":score,
                "truth":reference_labels,"predicted":pred.tolist(),
                "selected_source_epoch":saved["selected"]["epoch"],
                "target_used_for_training_or_selection":False,"final_holdout_evaluated":False,
                "synthetic_target_noise":snr in (0,5)}
        (out_folder/f"target_dev_{label}.json").write_text(json.dumps(record,indent=2))
        print(track,mode,seed,label,round(score["macro_f1"],5),flush=True)


def summarize(out,*,track="matched",modes=None,seeds=DEFAULT_SEEDS,
              conditions=DEFAULT_CONDITIONS):
    modes=list(modes) if modes is not None else ["xu_full"]
    rows=[]
    all_scores={}
    for mode in modes:
        for seed in seeds:
            run=json.loads((out/track/"models"/mode/f"seed{seed}"/"run.json").read_text())
            if run["track"]!=track or run["target_used_for_training_or_selection"]:
                raise ValueError("Run provenance mismatch")
            for snr in conditions:
                label=result_label(snr)
                path=out/track/"evaluations"/mode/f"seed{seed}"/f"target_dev_{label}.json"
                rec=json.loads(path.read_text())
                if (rec["mode"],rec["seed"],rec["track"],rec["condition"])!=(mode,seed,track,label):
                    raise ValueError("Evaluation metadata mismatch")
                if rec["record_families"]!=88 or rec["windows"]!=2474 or rec["final_holdout_evaluated"]:
                    raise ValueError("Holdout/Dev integrity failure")
                v=100*rec["metrics"]["macro_f1"]
                rows.append({"mode":mode,"seed":seed,"condition":label,"macro_f1_percent":v,
                             "accuracy_percent":100*rec["metrics"]["accuracy"]})
                all_scores[mode,seed,label]=v
    agg=[]
    for mode in modes:
        for snr in conditions:
            label=result_label(snr)
            s=np.array([all_scores[mode,seed,label] for seed in seeds])
            agg.append({"mode":mode,"condition":label,"macro_f1_mean":float(s.mean()),
                        "macro_f1_sample_sd":float(s.std(ddof=1)) if len(s)>1 else 0.,
                        "values":s.tolist(),"n_seeds":len(seeds)})
    output=out/track/"reports";output.mkdir(parents=True,exist_ok=True)
    with (output/"PER_SEED_RESULTS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/"SUMMARY.json").write_text(json.dumps({"track":track,"aggregates":agg,
        "final_holdout_evaluated":False,"target_training":False},indent=2))
    return agg


def smoke():
    from xu2025_full_model import smoke as core_smoke
    core_smoke()
    fix_randomness(42)
    x=torch.randn(7,5,4096)
    y=torch.tensor([0,1,2,3,4,5,0])
    for mode in VARIANTS:
        m=make_model(mode,"matched")
        term=paired_source_loss(m,x,x+.02*torch.randn_like(x),y)
        if not torch.isfinite(term["total"]):raise RuntimeError("Nonfinite")
        term["total"].backward()
        assert m.network.decoder[1][0].weight.grad is not None
        assert m.network.encoder[0][0].weight.grad is not None
        if mode=="xu_full_nahsf":
            assert m.shaping.gate[-1].weight.grad is not None
        print("PASS",mode,parameter_counts(m),flush=True)


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("smoke","train","evaluate","summarize"))
    parser.add_argument("--track",choices=("matched","paper_like"),default="matched")
    parser.add_argument("--modes",nargs="+",choices=VARIANTS)
    parser.add_argument("--seeds",nargs="+",type=int,default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch",type=int)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--weight-decay",type=float)
    parser.add_argument("--reconstruction",choices=("paper_sum","element_mean"),default="paper_sum")
    parser.add_argument("--rec-weight",type=float,default=1.)
    parser.add_argument("--dml-weight",type=float,default=1.)
    parser.add_argument("--lr-step",type=int,default=20)
    parser.add_argument("--lr-gamma",type=float,default=.5)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--source-strong-root",type=Path)
    parser.add_argument("--snr-root",type=Path)
    parser.add_argument("--strong-root",type=Path)
    parser.add_argument("--dev-export-root",type=Path)
    parser.add_argument("--conditions",nargs="+",default=["clean","10","15","20","25","30","35","40","5","0"])
    args=parser.parse_args()
    if args.stage=="smoke":smoke();return
    modes=args.modes or ["xu_full"]
    conditions=[None if s=="clean" else int(s) for s in args.conditions]
    if any(c is not None and c not in (0,5,10,15,20,25,30,35,40) for c in conditions):
        parser.error("Unsupported SNR condition")
    if args.stage=="train":
        for mode in modes:
            for seed in args.seeds:
                train(mode,seed,args.out,track=args.track,epochs=args.epochs,batch=args.batch,
                      lr=args.lr,weight_decay=args.weight_decay,
                      source_strong_root=args.source_strong_root,
                      reconstruction=args.reconstruction,rec_weight=args.rec_weight,
                      dml_weight=args.dml_weight,lr_step=args.lr_step,lr_gamma=args.lr_gamma)
    elif args.stage=="evaluate":
        if args.dev_export_root is None and args.snr_root is None and any(c is not None for c in conditions):
            parser.error("No experimental noise data root provided")
        if args.dev_export_root is None and args.strong_root is None and any(c in (0,5) for c in conditions):
            parser.error("0/5 dB requires synthetic strong-root or exported Dev files")
        for mode in modes:
            for seed in args.seeds:
                evaluate(mode,seed,args.out,track=args.track,conditions=conditions,
                         snr_root=args.snr_root,strong_root=args.strong_root,
                         export_root=args.dev_export_root)
    else:
        summary=summarize(args.out,track=args.track,modes=modes,seeds=args.seeds,conditions=conditions)
        print(json.dumps(summary,indent=2))

if __name__=="__main__":cli()
