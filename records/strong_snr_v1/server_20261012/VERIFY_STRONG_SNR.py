"""独立核对派生文件，并检查V4保存权重在源验证5/0 dB上的成绩。

原实测保留集只做生成数据的数值核验，不进行模型推理；源域验证
使用原302个周期，没有重新训练或改变先前保存的最佳权重。
"""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
V4 = BASE / "experiments/v4_hpasf/b4801df"
sys.path.insert(0, str(V4 / "code/scripts"))
import torch
import v2_experiment as data
from v1_noise_consistency import fix_randomness
from v4_hpasf import HPASFCNN, METHODS


def read(path):
    """读取已有配置/结果，不改写旧文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    """完整字节核对；旧权重与原输入文件均必须保持不变。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    """复算全部998份SNR，随后仅对仿真Val做24组固定权重推理。"""
    manifest = read(ROOT / "generated/GENERATION_MANIFEST.json")
    inputs = read(ROOT / "metadata/INPUT_MANIFEST.json")
    assert manifest["status"] == "passed"
    max_error = 0.
    for row in manifest["files"]:
        generated = ROOT / "generated" / row["file"]
        assert sha(generated) == row["file_sha256"]
        rid = row["record_id"]
        source = row["domain"] == "simulation"
        path = BASE / "data/source" / f"{rid}.mat" if source else ROOT / "inputs/target_clean" / f"{rid}.mat"
        field = "I_We" if source else "name"
        clean = loadmat(path)[field]
        noisy = loadmat(generated)["I_We" if source else "I_noisy"]
        assert noisy.shape == clean.shape and noisy.dtype == np.float64
        assert np.isfinite(noisy).all()
        columns = slice(0, 5) if source else slice(1, 6)
        measured = 10 * np.log10(np.mean(clean[:, columns] ** 2, axis=0) /
                                 np.mean((noisy[:, columns] - clean[:, columns]) ** 2, axis=0))
        max_error = max(max_error, float(np.max(np.abs(measured - row["snr_db"]))))
        assert np.array_equal(noisy[:, -1], clean[:, -1])
        if not source:
            assert np.array_equal(noisy[:, 0], clean[:, 0])
            original10 = loadmat(ROOT / "inputs/target_snr10" / f"{rid}.mat")["I_noisy"]
            residual = original10[:, 1:6] - clean[:, 1:6]
            assert np.allclose(noisy[:, 1:6] - clean[:, 1:6],
                               residual * row["noise_multiplier_from_author10"], atol=1e-13, rtol=1e-13)
    assert max_error < 1e-6
    for name, expected in inputs["source_clean_sha256"].items():
        assert sha(BASE / "data/source" / name) == expected
    for name, expected in inputs["target_files"].items():
        assert sha(ROOT / "inputs" / name) == expected["sha256"]
    print("DATA_REPLAY_PASSED: 998 generated MAT and 939 original MAT", flush=True)

    raw, meta, labels = data.source_data("val")
    views = {}
    for snr in (5, 0):
        def rng(rid):
            digest = hashlib.sha256(f"HUST-V1:val:{snr}:{rid}".encode()).digest()
            return np.random.default_rng(int.from_bytes(digest[:8], "little"))
        expected = data.calibrated_noise(raw, {rid: snr for rid in raw}, rng)
        new = {}
        for rid, (time, current, omega) in raw.items():
            a = loadmat(ROOT / "generated/source" / f"SNR_{snr}/{rid}.mat")["I_We"]
            assert np.array_equal(a[:, :5], expected[rid][1]), (rid, snr)
            new[rid] = time, a[:, :5], omega
        views[snr] = data.raw_cycles(new, meta)
    rows, summaries = [], []
    protected_weights = read(V4 / "results/VERIFICATION.json")["weights_sha256"]
    for method in METHODS:
        for seed in (42, 43, 44):
            fix_randomness(seed)
            key = f"{method}/seed{seed}"
            path = V4 / "results/models" / key / "best.pt"
            assert sha(path) == protected_weights[key]
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = HPASFCNN(method).cuda()
            model.load_state_dict(saved["state"], strict=True)
            for snr, values in views.items():
                predictions = data.inference(model, values, torch.device("cuda"))
                score = data.metrics(labels, predictions)
                assert abs(score["macro_f1"] - f1_score(labels, predictions, labels=list(range(6)), average="macro")) < 1e-12
                rows.append({"method": method, "seed": seed, "snr_db": snr,
                             "source_val_cycles": len(meta), "selected_original_epoch": saved["selected"]["epoch"],
                             "original_selected_val_snr10_macro_f1": saved["selected"]["noisy_f1"]["10"],
                             "metrics": score, "truth": labels.tolist(), "predicted": predictions.tolist()})
            for name, tensor in model.state_dict().items():
                assert torch.equal(tensor.cpu(), saved["state"][name])
            assert sha(path) == protected_weights[key]
            del model
    for method in METHODS:
        for snr in (5, 0):
            scores = [r["metrics"]["macro_f1"] * 100 for r in rows if r["method"] == method and r["snr_db"] == snr]
            summaries.append({"method": method, "snr_db": snr, "macro_f1_mean_percent": float(np.mean(scores)),
                              "macro_f1_sd_percent": float(np.std(scores, ddof=1))})
    output = {"status": "passed", "generated_mat_files_verified": 998,
              "original_mat_files_unchanged": 939, "max_snr_error_db": max_error,
              "source_val_fixed_rng_matches_original_policy": True,
              "source_val_evaluations": rows, "source_val_aggregates": summaries,
              "target_model_evaluations": 0, "models_retrained": False,
              "checkpoint_selection_changed": False, "original_weights_unchanged": 12,
              "verification_script_sha256": sha(Path(__file__))}
    (ROOT / "VERIFICATION.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "source_val_evaluations"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
