"""事后只读描述已选模型的训练/验证/实测Clean成绩，不再拟合或选模。

仅使用源训练特征缓存与已保存预测；不给分类器提供新的训练信息。
用于区分训练集拟合情况和跨记录/跨域泛化，不能据此证明错误因果。
"""
import hashlib
import json
from pathlib import Path
import os
import sys

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
REF = BASE / "experiments/architecture_suite/5c75bea"
sys.path.insert(0, str(ROOT / "code/scripts"))
sys.path.insert(0, str(REF / "code/scripts"))
import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from lp4_difficulty_models import SmallHarmonicMLP
from lp4_difficulty_experiment import mlp_predict, sha
from v1_noise_consistency import fix_randomness


def read(path):
    """读取实际已选模型记录与原预测，不改写原实验。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    """对9份固定分类器计算源Train Clean指标并描述预测分布。"""
    protected = read(ROOT / "results/reports/WEIGHTS_AND_RUN_FILES_SHA256.json")
    rows, distributions = [], []
    for seed in (42, 43, 44):
        fix_randomness(seed)
        with np.load(ROOT / f"results/cache/seed{seed}/source_features.npz", allow_pickle=False) as saved:
            clean, labels = saved["clean"].copy(), saved["labels"].copy()
        for method in ("lp4_logreg", "lp4_rbf_svm", "lp4_mlp"):
            folder = ROOT / f"results/models/{method}/seed{seed}"
            run = read(folder / "run.json")
            if method == "lp4_mlp":
                model = SmallHarmonicMLP().cuda().eval().requires_grad_(False)
                model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True)["state"], strict=True)
                scaler = joblib.load(folder / "scaler.joblib")
                predicted = mlp_predict(model, scaler.transform(clean).astype(np.float32), torch.device("cuda"))
            else:
                saved = joblib.load(folder / "best.joblib")
                predicted = saved["estimator"].predict(saved["scaler"].transform(clean).astype(np.float32))
            target = read(ROOT / f"results/evaluations/{method}/seed{seed}/target_dev_clean.json")
            rows.append({"method": method, "seed": seed, "source_train_clean_cycles": len(labels),
                         "source_train_clean_macro_f1_percent": 100 * f1_score(labels, predicted, labels=range(6), average="macro", zero_division=0),
                         "source_train_clean_accuracy_percent": 100 * accuracy_score(labels, predicted),
                         "selected_source_val_clean_macro_f1_percent": 100 * run["selected"]["clean_f1"],
                         "target_dev_clean_macro_f1_percent": 100 * target["metrics"]["macro_f1"]})
            for name in ("clean", "snr10", "snr5", "snr0", "snr-5"):
                obj = read(ROOT / f"results/evaluations/{method}/seed{seed}/target_dev_{name}.json")
                distributions.append({"method": method, "seed": seed, "condition": name,
                                      "predicted_label_counts": np.bincount(obj["predicted"], minlength=6).tolist()})
    assert all(sha(Path(path)) == value for path, value in protected.items())
    result = {"status": "passed", "analysis_type": "posthoc_descriptive_fixed_models_no_refit_or_reselection",
              "source_train_rows": rows, "target_prediction_distributions": distributions,
              "original_model_and_run_files_sha_unchanged": len(protected),
              "feature_or_hyperparameter_modified": False, "raw_waveforms_loaded": False,
              "script_sha256": sha(Path(__file__))}
    destination = ROOT / "results/reports/POSTHOC_SOURCE_FIT.json"
    assert not destination.exists()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False), flush=True)
    # 单独导出描述文件及脚本的原字节；不改先前结果包或其发布指纹。
    import tarfile
    export = {"results/reports/POSTHOC_SOURCE_FIT.json": destination, "DESCRIBE_SOURCE_FIT.py": Path(__file__)}
    manifest = ROOT / "POSTHOC_EXPORT_MANIFEST.json"
    manifest.write_text(json.dumps({"files": {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in export.items()}}, indent=2) + "\n", encoding="utf-8")
    export[manifest.name] = manifest
    with tarfile.open(ROOT / "posthoc_source_fit_records.tar.gz", "w:gz") as archive:
        for name, path in export.items():
            archive.add(path, arcname=name)


if __name__ == "__main__":
    main()
