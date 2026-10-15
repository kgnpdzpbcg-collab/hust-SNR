"""V4独立强噪声实验入口：扩展仿真训练/验证，再评估原88个实测编号。

    python scripts/v4_strong_noise_experiment.py train --strong-data-root DATA --output OUT
    python scripts/v4_strong_noise_experiment.py evaluate --snr-root SNR --output OUT
    python scripts/v4_strong_noise_experiment.py summarize --output OUT

模型与训练循环复用V4实现；新入口只传入冻结的强噪声协议。不修改旧CLI默认值，
评估前强制检查权重协议，防止拿旧权重冒充重训结果。新旧目录必须分开。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import v4_experiment as v4
from evaluate_v4_strong_snr import frozen_ids

CONFIG = Path(__file__).resolve().parents[1] / "configs/v4_strong_noise_v1.json"
PROTOCOL = "V4_strong_noise_v1"


def config():
    """只接受本轮确定的协议，运行中不搜索超参数或增加额外训练。"""
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {"protocol": PROTOCOL, "methods": list(v4.METHODS), "seeds": [42, 43, 44],
                "epochs": 50, "batch_size": 32, "learning_rate": .001,
                "training_noise_levels_db": [0, 5, 10, 15, 20, 25, 30, 35, 40],
                "validation_noise_levels_db": [0, 5, 10, 20, 30],
                "target_conditions": ["clean", 0, 5, 10, 15, 20, 25, 30, 35, 40],
                "target_used_for_training_or_selection": False, "final_holdout_evaluated": False}
    if any(value[key] != item for key, item in expected.items()):
        raise ValueError("配置与已确定的强噪声协议不一致")
    return value


def check_checkpoint(checkpoint, protocol):
    """拒绝旧权重或噪声档位不一致的权重；其余V4字段由原评估函数校验。"""
    if (checkpoint.get("train_protocol") != PROTOCOL or
            checkpoint.get("train_noise_levels_db") != protocol["training_noise_levels_db"] or
            checkpoint.get("val_noise_levels_db") != protocol["validation_noise_levels_db"]):
        raise ValueError("只能评估本轮强噪声重训权重")


def main():
    """四种方法共享相同协议；并发由服务器调度器负责，不在此处派生任务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    parser.add_argument("--methods", choices=v4.METHODS, nargs="+", default=list(v4.METHODS))
    parser.add_argument("--strong-data-root", type=Path)
    parser.add_argument("--snr-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = config()
    conditions = [None if c == "clean" else c for c in protocol["target_conditions"]]
    if args.stage == "train":
        if args.strong_data_root is None:
            parser.error("训练需要已核验的强噪声数据目录")
        for method in args.methods:
            for seed in protocol["seeds"]:
                v4.train(method, seed, args.output, protocol["epochs"], protocol["batch_size"],
                         protocol["learning_rate"],
                         train_noise_levels=tuple(protocol["training_noise_levels_db"]),
                         val_noise_levels=tuple(protocol["validation_noise_levels_db"]),
                         validation_mat_root=args.strong_data_root / "generated/source",
                         train_protocol=PROTOCOL)
    elif args.stage == "evaluate":
        if args.snr_root is None:
            parser.error("评估需要包含九档SNR目录的独立只读数据链接")
        from prepare import ROOT, OUT, read
        frozen_ids(read(ROOT / "configs/splits_gate0_v1.json"), read(OUT / "target_dev_clean_windows.json"))
        for method in args.methods:
            for seed in protocol["seeds"]:
                checkpoint = torch.load(args.output / f"models/{method}/seed{seed}/best.pt",
                                        map_location="cpu", weights_only=True)
                check_checkpoint(checkpoint, protocol)
                v4.evaluate(method, seed, args.output, args.snr_root, None, conditions)
    else:
        v4.summarize(args.output, args.methods, protocol["seeds"], conditions)


if __name__ == "__main__":
    main()
