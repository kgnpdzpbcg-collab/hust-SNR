"""精确回放已有V5/固定α训练，仅增加快照保存并事后扫描Dev轮次。

逐轮loss/源验证和原最佳权重必须完全一致，否则停止诊断。目标Dev轮次
扫描发生在全部训练完成之后，不用于训练、选模或替换正式实验结果。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


def snapshot_class(base, folder):
    """只在每轮第一次切换到eval时保存状态，不改变网络或消耗随机数。

    共用原训练函数，仍然只有一次原训练更新流程。模型在每轮首个Clean
    验证前保存快照，后续噪声验证的eval调用不重复保存。不注册新参数。
    """
    class Snapshot(base):
        def __init__(self):
            super().__init__()
            self.diagnostic_epoch = 0

        def train(self, mode=True):
            if mode:
                self.diagnostic_epoch += 1
            elif self.training and self.diagnostic_epoch:
                folder.mkdir(parents=True, exist_ok=True)
                state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                torch.save({"epoch": self.diagnostic_epoch, "state": state}, folder/f"epoch{self.diagnostic_epoch:02d}.pt")
            return super().train(mode)
    return Snapshot


def main():
    """先回放两模型、核验完全一致，再逐轮Dev诊断；任何原文件保持不变。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-run", type=Path, required=True)
    parser.add_argument("--strong-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("拒绝覆盖回放诊断")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.extra_run/"code/scripts"))
    import five_models_extra_seeds as experiment
    import four_models_expanded_val as common
    import v2_experiment as data
    from fixed_half_fusion import FixedHalfCNN
    from v5_nahsf import NAHSFCNN
    from v1_noise_consistency import fix_randomness, metrics
    from v5_experiment import predict_with_gate, gate_summary
    from evaluate_v4_strong_snr import read, write, csv_write, frozen_ids
    from prepare import OUT, ROOT as DATA_ROOT
    if args.seed not in experiment.NEW_SEEDS:
        raise ValueError("回放种子必须属于当前新增三个种子")
    native_root = args.extra_run/"results"
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in native_root.rglob("*") if p.is_file()}
    parity = []
    for method, base in (("nahsf", NAHSFCNN), ("fixed_half", FixedHalfCNN)):
        snapshot_type = snapshot_class(base, args.output/f"snapshots/{method}")
        def factory(requested):
            """当前只构造请求的单一模型，CNN随机初始化顺序完全保留。"""
            if requested != method:
                raise ValueError(requested)
            return snapshot_type()
        common.train(method, args.seed, args.output/"replay", args.strong_data_root,
                     protocol=experiment.config(), model_factory=factory, version=experiment.VERSION)
        new_folder = args.output/f"replay/models/{method}/seed{args.seed}"
        old_folder = native_root/f"models/{method}/seed{args.seed}"
        history, original = read(new_folder/"history.json"), read(old_folder/"history.json")
        assert history == original, f"{method}逐轮loss/源验证未精确复现"
        a, b = (torch.load(p/"best.pt", map_location="cpu", weights_only=True) for p in (new_folder, old_folder))
        assert a["selected"] == b["selected"] and a["state"].keys() == b["state"].keys()
        assert all(torch.equal(v, b["state"][k]) for k, v in a["state"].items())
        assert len(list((args.output/f"snapshots/{method}").glob("*.pt"))) == 50
        parity.append({"method": method, "seed": args.seed, "loss_and_source_val_50_epochs_exact": True,
                       "selected_epoch": a["selected"]["epoch"], "original_best_all_tensors_exact": True})
        print(f"EXACT_REPLAY_PASSED {method}: 50 histories, original best tensors", flush=True)
    metadata = read(OUT/"target_dev_clean_windows.json")
    frozen_ids(read(DATA_ROOT/"configs/splits_gate0_v1.json"), metadata)
    truth = np.array([row["label"] for row in metadata])
    models = {}
    for method in ("nahsf", "fixed_half"):
        fix_randomness(args.seed)
        models[method] = experiment.make_model(method).cuda().eval()
    rows, details = [], []
    for level in (None, 10, 5, 0):
        condition = "clean" if level is None else f"snr{level}"
        root = args.strong_data_root/"generated/target" if level in (0, 5) else args.extra_run/"target_snr"
        raw, actual = data.target_dev_from_mat(level, root)
        assert actual == metadata
        inputs = data.raw_cycles(raw, metadata)
        for method, model in models.items():
            history = read(native_root/f"models/{method}/seed{args.seed}/history.json")
            selected_epoch = read(native_root/f"models/{method}/seed{args.seed}/run.json")["selected"]["epoch"]
            for epoch in range(1, 51):
                snapshot = torch.load(args.output/f"snapshots/{method}/epoch{epoch:02d}.pt", map_location="cpu", weights_only=True)
                assert snapshot["epoch"] == epoch
                model.load_state_dict(snapshot["state"], strict=True)
                if method == "nahsf":
                    prediction, alpha, descriptors = predict_with_gate(model, inputs, torch.device("cuda"))
                    gate = gate_summary(alpha, descriptors, truth)
                else:
                    prediction, gate = data.inference(model, inputs, torch.device("cuda")), None
                m = metrics(truth, prediction)
                if epoch == selected_epoch:
                    original = read(native_root/f"evaluations/{method}/seed{args.seed}/target_dev_{condition}.json")
                    assert np.array_equal(original["predicted"], prediction)
                h = history[epoch-1]
                rows.append({"method": method, "seed": args.seed, "epoch": epoch, "condition": condition,
                             "selected_official_epoch": epoch == selected_epoch, "train_loss_mean": h["train_loss_mean"],
                             "source_clean_f1_percent": 100*h["clean_f1"], "source_five_f1_percent": 100*h["noise_mean_f1"],
                             "target_macro_f1_percent": 100*m["macro_f1"], "target_accuracy_percent": 100*m["accuracy"],
                             "gate_alpha_mean": gate["mean"] if gate else .5})
                details.append({"method": method, "seed": args.seed, "epoch": epoch, "condition": condition,
                                "metrics": m, "predicted": prediction.tolist(), "gate": gate})
        print(f"EPOCH_SCAN_COMPLETED {condition}: 100 paired-model snapshots", flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected for p, expected in protected.items())
    csv_write(args.output/"EPOCH_TARGET_SCAN.csv", rows)
    write(args.output/"EPOCH_TARGET_DETAILS.json", {"truth": truth.tolist(), "results": details})
    write(args.output/"STATUS.json", {"status": "passed", "seed": args.seed, "parity": parity,
          "training_config_changed": False, "epoch_snapshot_count": 100, "posthoc_target_predictions": 400,
          "target_used_during_replay_training_or_source_selection": False, "posthoc_scan_used_to_replace_official": False,
          "original_results_and_weights_unchanged": True, "final_holdout_evaluated": False})


if __name__ == "__main__":
    main()
