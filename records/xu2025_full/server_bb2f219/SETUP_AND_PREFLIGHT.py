"""部署仓库Xu-Full原代码；仅检查matched三种子协议和真实GPU损失/梯度。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT/"code"
FOUR = BASE/"experiments/four_models_expanded_val/56513c7"
EXTRA = BASE/"experiments/five_models_extra_seeds/3f07228"
STRONG = BASE/"datasets/strong_snr_v1_20261012"


def sha(path):
    """流式保护原数据/结果字节，不改写输入或旧模型。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    """检查记录仅写入本轮独立目录。"""
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def main():
    """固定配置、原数据SHA和GPU检查通过后才运行正式三种子训练。"""
    assert not CODE.exists() and not (ROOT/"results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT/"code.tar") as archive:
        archive.extractall(CODE,filter="data")
    (CODE/"data/target_dev").mkdir(parents=True)
    (CODE/"outputs").mkdir()
    for link,target in ((CODE/"data/source",FOUR/"code/data/source"),
        (CODE/"data/target_dev/clean",FOUR/"code/data/target_dev/clean"),
        (CODE/"outputs/gate0_v1",FOUR/"code/outputs/gate0_v1"),
        (ROOT/"target_snr",FOUR/"target_snr")):
        link.symlink_to(target,target_is_directory=True)
    prior = json.loads((EXTRA/"DEPLOYMENT_MANIFEST.json").read_text())
    for name in ("scripts/v2_experiment.py","scripts/v2_paifilter.py","scripts/v1_noise_consistency.py",
                 "scripts/prepare.py","configs/splits_gate0_v1.json"):
        assert sha(CODE/name) == prior["code_sha256"][name], name
    protected = json.loads((EXTRA/"ORIGINAL_FILES_SHA256.json").read_text())
    protected.update({str(p):sha(p) for p in (EXTRA/"results").rglob("*") if p.is_file()})
    assert all(sha(p)==expected for p,expected in protected.items())
    write(ROOT/"ORIGINAL_FILES_SHA256.json",protected)
    protocol=json.loads((CODE/"configs/xu2025_full_v1.json").read_text())
    matched=protocol["matched_track"]
    expected={"modes":["xu_full"],"seeds":[42,43,44],"epochs":50,"batch":32,"initial_lr":.001,
              "weight_decay":0,"source_train_noise_db":[10,15,20,25,30,35,40],"source_val_noise_db":[0,5,10,20,30]}
    assert all(matched[k]==v for k,v in expected.items())
    write(ROOT/"DEPLOYMENT_MANIFEST.json",{
        "code_commit":"bb2f219cd4e9a34fde9f8fe7f55a19244bcced9d","branch":"research/xu2025-full-cae-dml",
        "code_sha256":{p.relative_to(CODE).as_posix():sha(p) for p in CODE.rglob("*") if p.is_file() and p.suffix in (".py",".json",".md",".sh")},
        "archive_sha256":sha(ROOT/"code.tar"),"protocol":protocol,"track":"matched","mode":"xu_full",
        "reconstruction":"paper_sum","rec_weight":1.,"dml_weight":1.,"environment_reused":True,
        "python_executable":sys.executable,"parallel_workers":3,"source_strong_root":str(STRONG/"generated/source"),
        "target_strong_root":str(STRONG/"generated/target"),"reference_four_results":str(FOUR/"results")})
    os.environ.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command=[sys.executable,"-m","pytest","-q","tests/test_xu2025_full_model.py","tests/test_xu2025_full_experiment.py"]
    done=subprocess.run(command,cwd=CODE,env=os.environ.copy(),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(done.stdout,flush=True)
    assert done.returncode==0
    sys.path.insert(0,str(CODE/"scripts"))
    import numpy as np
    import torch
    import v2_experiment as data
    import xu2025_full_experiment as experiment
    from xu2025_full_model import parameter_counts
    from v1_noise_consistency import fix_randomness
    tr,tm,y=data.source_data("train")
    vr,vm,_=data.source_data("val")
    assert (len(tr),len(vr),len(tm),len(vm))==(34,15,738,302)
    fixed=data.fixed_noisy_validation(vr,vm,noise_levels=(0,5),materialized_root=STRONG/"generated/source")
    assert all(np.array_equal(fixed[k],v) for k,v in data.fixed_noisy_validation(vr,vm,noise_levels=(0,5)).items())
    clean=data.raw_cycles(tr,tm)
    previous=json.loads((FOUR/"PREFLIGHT.json").read_text())
    initial,noise_sha,gpu={},{},{}
    for seed in (42,43,44):
        fix_randomness(seed)
        model=experiment.make_model("xu_full","matched").cuda()
        counts=parameter_counts(model)
        assert counts=={"paper_cae_dml":35475,"spectral":0,"total":35475}
        initial[str(seed)]=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        noisy=data.fresh_noisy_source(tr,tm,seed,1)
        noise_sha[str(seed)]=hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noise_sha[str(seed)]==previous["first_epoch_noise_sha256"][str(seed)]
        # 正式训练的首批次顺序，真实Clean/Noisy拼接，不用随机假波形代替。
        ids=np.random.default_rng(seed+902).permutation(len(y))[:32]
        terms=experiment.paired_source_loss(model,torch.from_numpy(clean[ids]).cuda(),
                    torch.from_numpy(noisy[ids]).cuda(),torch.from_numpy(y[ids]).cuda(),
                    reduction="paper_sum",rec_weight=1.,dml_weight=1.)
        assert all(torch.isfinite(v) for v in terms.values())
        terms["total"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        item={k:float(v.detach().cpu()) for k,v in terms.items()}
        item["recon_to_ce_ratio"]=item["recon"]/item["ce"]
        item["all_parameter_gradients_finite"]=True
        item["parameter_counts"]=counts
        gpu[str(seed)]=item
        print("REAL_GPU_LOSS_COMPONENTS",seed,json.dumps(item),flush=True)
        torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.).step()
        del model
    write(ROOT/"PREFLIGHT.json",{"status":"passed","unit_test_command":command,"unit_test_output":done.stdout,
        "initial_model_sha256":initial,"first_epoch_noise_sha256":noise_sha,"real_gpu_loss_components":gpu,
        "source_val0_5_matches_original_rng":True,"device":torch.cuda.get_device_name(),"torch_version":str(torch.__version__),
        "source_train_records":34,"source_val_records":15,"source_train_cycles":738,"source_val_cycles":302})
    print("XU_FULL_PREFLIGHT_PASSED",flush=True)


if __name__=="__main__":main()
