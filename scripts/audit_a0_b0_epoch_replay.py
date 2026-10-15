"""按原A0/B0配置重放训练，仅增加第50轮快照，诊断早期源选模的影响。

不修改原模型、损失、训练循环或选模。以原50轮历史和原best.pt逐参数一致
作为有效重放条件；随后只比较预先指定的原best与第50轮，不按目标挑轮次。
输出单独保存，不能覆盖原正式实验或把诊断终轮替换为正式最佳权重。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
import numpy as np
import torch
from torch.nn import functional as F


def read(path):
    """只读原运行记录；全部新输出使用独立目录。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """完整文件指纹保护原权重和结果。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def score(exp, model, raw, truth):
    """计算F1、CE和置信度；与正式推理相同的128批大小和eval模式。"""
    output = torch.cat([model(torch.from_numpy(raw[pos:pos + 128]).cuda()).cpu()
                        for pos in range(0, len(raw), 128)])
    pred = output.argmax(1).numpy()
    return {"metrics": exp.metrics(truth, pred), "truth": truth.tolist(), "predicted": pred.tolist(),
            "cross_entropy": float(F.cross_entropy(output, torch.from_numpy(truth))),
            "mean_max_probability": float(output.softmax(1).max(1).values.mean())}


def main(args):
    """回调只保存原_snapshot返回的副本，不插入随机操作、优化或额外前向。"""
    reference = args.reference_root
    sys.path.insert(0, str(reference / "code/scripts"))
    import hust_prototypes_experiment as exp
    from prepare import OUT
    deploy = read(reference / "DEPLOYMENT_MANIFEST.json")
    old = reference / f"results/models/{args.method}/seed{args.seed}"
    protected = {str(p): sha(p) for p in old.iterdir() if p.is_file()}
    original_snapshot, calls = exp._snapshot, 0
    final_path = args.output / f"final_epochs/{args.method}/seed{args.seed}/epoch50.pt"
    assert not final_path.exists()

    def capture_final(model):
        """第50次快照另存一份；返回原始快照，原训练函数行为保持不变。"""
        nonlocal calls
        state = original_snapshot(model)
        calls += 1
        if calls == 50:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": state, "epoch": 50, "method": args.method, "seed": args.seed,
                        "config": exp.get_config(), "source_only": True, "diagnostic_snapshot_only": True}, final_path)
        return state

    exp._snapshot = capture_final
    exp.train(args.method, args.seed, args.output, Path(deploy["source_strong_root"]))
    assert calls == 50
    new = args.output / f"models/{args.method}/seed{args.seed}"
    assert read(new / "history.json") == read(old / "history.json"), "50轮历史未逐数值复现"
    old_run, new_run = read(old / "run.json"), read(new / "run.json")
    assert old_run == new_run, "原初始化、预训练或源选模未完全复现"
    old_best = torch.load(old / "best.pt", weights_only=True, map_location="cpu")
    new_best = torch.load(new / "best.pt", weights_only=True, map_location="cpu")
    assert old_best["selected"] == new_best["selected"]
    assert all(torch.equal(value, new_best["state"][name]) for name, value in old_best["state"].items())
    exp.fix_randomness(args.seed)
    model = exp.build_model(args.method).cuda().eval().requires_grad_(False)
    final = torch.load(final_path, weights_only=True, map_location="cpu")
    model.load_state_dict(final["state"], strict=True)
    (_, _, _, _), (_, _, vy, clean, noisy) = exp.source_views(Path(deploy["source_strong_root"]))
    records = {}
    for condition, raw in {"source_clean": clean, **{f"source_snr{k}": x for k, x in noisy.items()}}.items():
        value = score(exp, model, raw, vy)
        expected = read(old / "history.json")[-1]
        expected_f1 = expected["clean_f1"] if condition == "source_clean" else expected["noisy_f1"][condition[10:]]
        np.testing.assert_allclose(value["metrics"]["macro_f1"], expected_f1, atol=1e-12, rtol=0)
        records[condition] = value
    frozen = read(OUT / "target_dev_clean_windows.json")
    truth = np.asarray([r["label"] for r in frozen], dtype=np.int64)
    for snr in (None, 5, 0, -5, -10):
        raw, meta = exp.data.target_dev_from_mat(None, None) if snr is None else exp._target_cycles(snr,
                ordinary_root=Path(deploy["target_strong_root"]), extreme_root=Path(deploy["target_extreme_root"]), export_root=None)
        assert meta == frozen and len(raw) == 88
        value = score(exp, model, exp.data.raw_cycles(raw, meta), truth)
        condition = "target_clean" if snr is None else f"target_snr{snr}"
        records[condition] = value
        print(f"EPOCH50 {args.method} seed{args.seed} {condition}: F1={100*value['metrics']['macro_f1']:.4f}%", flush=True)
    assert all(torch.equal(v.detach().cpu(), final["state"][name]) for name, v in model.state_dict().items())
    assert all(sha(path) == value for path, value in protected.items())
    exp.write_json(args.output / f"epoch50_evaluations/{args.method}/seed{args.seed}.json", {
        "method": args.method, "seed": args.seed, "original_selected_epoch": old_best["selected"]["epoch"],
        "diagnostic_epoch": 50, "all_original_training_history_values_exactly_reproduced": True,
        "original_selected_weights_tensorwise_identical": True, "original_files_sha_unchanged": protected,
        "final_weight_sha256": sha(final_path), "training_protocol_or_selection_changed": False,
        "target_used_to_select_epoch": False, "final_holdout_accessed": False,
        "results": records, "script_sha256": sha(Path(__file__))})
    print(f"REPLAY_VERIFIED {args.method} seed{args.seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("a0", "b0"), required=True)
    parser.add_argument("--seed", choices=(42, 43, 44), type=int, required=True)
    main(parser.parse_args())
