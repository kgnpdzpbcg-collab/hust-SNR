"""六模型可视化部署：新目录复用模型/输入读取代码，不安装依赖或训练。"""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[2]
CODE=ROOT/"code"
XU=BASE/"experiments/xu2025_full/bb2f219"


def sha(path):
    """源实现与实际字节一致，图分析代码另行记录。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """只准备读实测Dev的路径；Final-Holdout不复制到分析运行目录。"""
    assert not CODE.exists() and not (ROOT/"results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT/"code.tar") as archive:archive.extractall(CODE,filter="data")
    (CODE/"data/target_dev").mkdir(parents=True)
    (CODE/"outputs").mkdir()
    (CODE/"data/target_dev/clean").symlink_to(XU/"code/data/target_dev/clean",target_is_directory=True)
    (CODE/"outputs/gate0_v1").symlink_to(XU/"code/outputs/gate0_v1",target_is_directory=True)
    reference=json.loads((XU/"DEPLOYMENT_MANIFEST.json").read_text())
    names=["scripts/"+name for name in ("prepare.py","v2_experiment.py","v1_noise_consistency.py",
        "v2_paifilter.py","v4_hpasf.py","v5_nahsf.py","fixed_half_fusion.py","four_models_expanded_val.py",
        "fixed_half_experiment.py","xu2025_full_model.py","xu2025_full_experiment.py")]
    names.append("configs/splits_gate0_v1.json")
    assert all(sha(CODE/name)==reference["code_sha256"][name] for name in names)
    manifest={"code_commit":"f398d308f046f4b5bd7e001888007fbf85669556","python":sys.executable,
        "prior_failed_run":"fc4c487: CPU/CUDA state-equality guard; logs retained; formal run in new directory",
        "environment_reused":True,"base":str(BASE),"all_model_and_input_code_unchanged":True,
        "code_sha256":{p.relative_to(CODE).as_posix():sha(p) for p in CODE.rglob("*") if p.is_file() and p.suffix in (".py",".json",".md")},
        "config":json.loads((CODE/"configs/six_model_tsne_v1.json").read_text())}
    (ROOT/"DEPLOYMENT_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print("SIX_MODEL_VISUALIZATION_SETUP_PASSED",flush=True)


if __name__=="__main__":main()
