"""用固定V1 M1权重诊断10dB误差：分类分布、归一化漂移及交叉输入。

不重训，不改V1实现，不读取Holdout。两项交叉输入使用配对Clean的
归一化参数，只作为诊断控制；不能作为部署方案或正式模型成绩。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

NAMES = ["Healthy", "OUS(A)", "OLS(A)", "OP(A)", "OUS(A)&OUS(B)", "OLS(A)&OP(C)"]
SEEDS = [42, 43, 44]


def read(path):
    """读取冻结的UTF-8标签、周期索引和预测记录。"""
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    """仅写新的诊断目录，不覆盖原实验记录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_save(path, rows):
    """将逐周期/逐相及分组统计保存为可检查的表格。"""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def sha(path):
    """核对最佳权重及脚本字节，诊断前后必须一致。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(time, currents, row):
    """精确展开原预处理：边界插值→极值中心→低电流分支→128点。

    返回原周期电流、中心、除数、分支标志及128点物理电流。原波形
    不修改；中心和除数来自完整原分辨率周期，不能先重采样后取极值。
    """
    start, end = row["start"], row["end"]
    knots = np.r_[start, time[(time > start) & (time < end)], end]
    raw = np.stack([np.interp(knots, time, currents[:, i]) for i in range(5)])
    center = (raw.max(1) + raw.min(1)) / 2
    centered = raw - center[:, None]
    meanabs = np.mean(np.abs(centered), axis=1)
    low = meanabs < .2
    divisor = np.where(low, 100., np.max(np.abs(centered), axis=1))
    grid = np.linspace(start, end, 128)
    sampled = np.stack([np.interp(grid, knots, channel) for channel in raw])
    return {"knots": knots, "raw": raw, "center": center, "meanabs": meanabs,
            "low": low, "divisor": divisor, "sampled": sampled}


def transform(signal, parameters):
    """独立控制电流内容与归一化参数；保持相同周期边界及128点网格。"""
    return ((signal["sampled"] - parameters["center"][:, None]) /
            parameters["divisor"][:, None]).astype(np.float32)


def snr(clean, difference):
    """计算周期功率比；零信号/零噪声时返回None，不虚构有限SNR。"""
    signal_power, noise_power = np.mean(clean ** 2), np.mean(difference ** 2)
    return float(10 * np.log10(signal_power / noise_power)) if signal_power > 0 and noise_power > 0 else None


def summary(values):
    """中位数和四分位数用于描述窗口分布，不作为独立采样置信区间。"""
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return {"n_defined": 0}
    return {"n_defined": len(values), "q25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, .75))}


def main(args):
    """按先分类、再输入变换、最后固定权重对照的顺序生成诊断证据。"""
    run = args.run_root.resolve();base = run.parents[2]
    out = args.output.resolve()
    assert not out.exists(), "不能覆盖已有诊断"
    out.mkdir(parents=True)
    sys.path.insert(0, str(run / "code/scripts"))
    import torch
    import v1_noise_consistency as v1
    from prepare import preprocess
    meta = read(base / "outputs/gate0_v1/target_dev_clean_windows.json")
    labels = read(base / "analysis/data_audit_v1/LABELS.json")["experimental"]
    split = read(run / "code/configs/splits_gate0_v1.json")
    assert len(meta) == 2474 and {r["record_id"] for r in meta} == set(split["target_dev_records"])
    assert set(split["target_dev_records"]).isdisjoint(split["target_holdout_records"])
    raw = {rid: [v1.target_condition_record(rid, condition, run / "target_snr") for condition in [None, 10]]
           for rid in split["target_dev_records"]}
    inputs = {k: [] for k in ["CC", "NN", "CN", "NC"]}
    phases, window_stats = [], []
    for i, row in enumerate(meta):
        rid = row["record_id"];(time, clean, omega), (nt, noisy, nw) = raw[rid]
        assert np.array_equal(time, nt) and np.array_equal(omega, nw)
        c, n = profile(time, clean, row), profile(nt, noisy, row)
        values = {"CC": transform(c, c), "NN": transform(n, n), "CN": transform(c, n), "NC": transform(n, c)}
        # CC/NN必须与原入口逐位一致，交叉参数只影响CN/NC两项诊断控制。
        assert np.allclose(values["CC"], preprocess(time, clean, row["start"], row["end"])[0], atol=2e-7, rtol=0)
        assert np.allclose(values["NN"], preprocess(nt, noisy, row["start"], row["end"])[0], atol=2e-7, rtol=0)
        for key in inputs:inputs[key].append(values[key])
        corr = []
        for p in range(5):
            rc, rn = c["sampled"][p], n["sampled"][p]
            coefficient = float(np.corrcoef(rc, rn)[0, 1]) if np.std(rc) > 0 and np.std(rn) > 0 else None
            corr.append(coefficient)
            phases.append({"window": i, "record_id": rid, "label": row["label"], "phase": "ABCDE"[p],
                "speed_label": labels[rid]["speed"], "load_label": labels[rid]["torque"],
                "clean_center": float(c["center"][p]), "noise_center": float(n["center"][p]),
                "center_shift_in_clean_scale": float((n["center"][p] - c["center"][p]) / c["divisor"][p]),
                "clean_divisor": float(c["divisor"][p]), "noise_divisor": float(n["divisor"][p]),
                "divisor_ratio": float(n["divisor"][p] / c["divisor"][p]),
                "clean_meanabs": float(c["meanabs"][p]), "noise_meanabs": float(n["meanabs"][p]),
                "clean_low": bool(c["low"][p]), "noise_low": bool(n["low"][p]),
                "branch_changed": bool(c["low"][p] != n["low"][p]),
                "raw_cycle_phase_snr_db": snr(c["raw"][p], n["raw"][p] - c["raw"][p]),
                "raw_resampled_correlation": coefficient,
                "clean_normalized_rms": float(np.sqrt(np.mean(values["CC"][p] ** 2))),
                "noise_normalized_rms": float(np.sqrt(np.mean(values["NN"][p] ** 2)))})
        active = ~c["low"]
        window_stats.append({"window": i, "record_id": rid, "label": row["label"],
            "speed_label": labels[rid]["speed"], "load_label": labels[rid]["torque"],
            "joint_raw_cycle_snr_db": snr(c["raw"], n["raw"] - c["raw"]),
            "max_active_divisor_ratio": float(np.max(n["divisor"][active] / c["divisor"][active])),
            "max_active_center_shift": float(np.max(np.abs(n["center"][active] - c["center"][active]) / c["divisor"][active])),
            "active_phase_correlation_mean": float(np.mean([corr[p] for p in range(5) if active[p] and corr[p] is not None])),
            "branch_changed": bool(np.any(c["low"] != n["low"]))})
    arrays = {k: np.stack(v) for k, v in inputs.items()}
    truth = np.asarray([r["label"] for r in meta])
    for key, condition in [("CC", "clean"), ("NN", "snr10")]:
        with np.load(run / f"dev_{condition}.npz") as f:
            assert np.allclose(arrays[key], f["x"], atol=2e-7, rtol=0) and np.array_equal(truth, f["y"])
            # 正式CC/NN用缓存原字节，以免独立代数展开的末位舍入干扰复评。
            arrays[key] = f["x"].copy()
    csv_save(out / "PHASE_PREPROCESSING.csv", phases)
    csv_save(out / "WINDOW_PREPROCESSING.csv", window_stats)
    cycle_rows, scores, predictions_by_seed = [], {}, {}
    weights = {}
    for seed in SEEDS:
        v1.fix_randomness(seed)
        checkpoint = run / f"results/models/augmentation/seed{seed}/best.pt"
        weights[str(seed)] = sha(checkpoint)
        model = v1.CNN().cuda();model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state"])
        model.eval();before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scores[seed], predictions_by_seed[seed], features = {}, {}, {}
        for key, x in arrays.items():
            logits, h = [], []
            with torch.no_grad():
                for start in range(0, len(x), 256):
                    z, feat = model(torch.from_numpy(x[start:start+256]).cuda(), return_features=True)
                    logits.append(z.cpu().numpy());h.append(feat.cpu().numpy())
            logits, h = np.concatenate(logits), np.concatenate(h)
            pred = logits.argmax(1);predictions_by_seed[seed][key] = pred;features[key] = (logits, h)
            scores[seed][key] = v1.metrics(truth, pred)
            if key in ["CC", "NN"]:
                original = read(run / f"results/evaluations/augmentation/seed{seed}/target_dev_{'clean' if key=='CC' else 'snr10'}.json")
                assert np.array_equal(pred, original["predicted"]), (seed, key)
        assert all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
        for i, row in enumerate(meta):
            pc, pn = predictions_by_seed[seed]["CC"][i], predictions_by_seed[seed]["NN"][i]
            clean_logits, hc = features["CC"][0][i], features["CC"][1][i]
            noisy_logits, hn = features["NN"][0][i], features["NN"][1][i]
            alternatives = clean_logits.copy();alternatives[row["label"]] = -np.inf
            margin = float(clean_logits[row["label"]] - np.max(alternatives))
            cosine = float(np.dot(hc, hn) / (np.linalg.norm(hc) * np.linalg.norm(hn)))
            item = {"seed": seed, **window_stats[i], "clean_prediction": int(pc), "noise_prediction": int(pn),
                "clean_with_noise_parameters_prediction": int(predictions_by_seed[seed]["CN"][i]),
                "noise_with_clean_parameters_prediction": int(predictions_by_seed[seed]["NC"][i]),
                "clean_correct": bool(pc == row["label"]), "noise_wrong": bool(pn != row["label"]),
                "new_error": bool(pc == row["label"] and pn != row["label"]),
                "clean_classifier_margin": margin, "feature_cosine_change": 1-cosine}
            cycle_rows.append(item)
        print('DIAGNOSED seed', seed, {k: round(v['macro_f1']*100, 3) for k,v in scores[seed].items()}, flush=True)
    csv_save(out / "CYCLE_ERRORS.csv", cycle_rows)
    per_class = []
    for label, name in enumerate(NAMES):
        clean_f1 = [scores[s]["CC"]["per_class_f1"][label]*100 for s in SEEDS]
        noise_f1 = [scores[s]["NN"]["per_class_f1"][label]*100 for s in SEEDS]
        per_class.append({"label": label, "name": name, "windows_per_seed": int(np.sum(truth == label)),
            "clean_f1_mean": float(np.mean(clean_f1)), "noise_f1_mean": float(np.mean(noise_f1)),
            "drop_pp": float(np.mean(clean_f1)-np.mean(noise_f1)),
            "contribution_to_macro_drop_pp": float((np.mean(clean_f1)-np.mean(noise_f1))/6)})
    csv_save(out / "PER_CLASS.csv", per_class)
    groups = []
    for kind in ["speed", "record", "speed_load"]:
        keys = sorted(set((str(r['speed_label']) if kind=='speed' else r['record_id'] if kind=='record'
                           else f"{r['speed_label']}:{r['load_label']}") for r in cycle_rows))
        for key in keys:
            rows = [r for r in cycle_rows if (str(r['speed_label']) if kind=='speed' else r['record_id'] if kind=='record'
                     else f"{r['speed_label']}:{r['load_label']}")==key]
            unique = {r['record_id'] for r in rows}
            class_error_rates = [np.mean([r['noise_wrong'] for r in rows if r['label']==label])
                                 for label in range(6) if any(r['label']==label for r in rows)]
            record_error_rates = [np.mean([r['noise_wrong'] for r in rows if r['record_id']==rid]) for rid in unique]
            groups.append({"group_kind": kind, "key": key, "record_families": len(unique),
                "window_seed_decisions": len(rows), "errors": sum(r['noise_wrong'] for r in rows),
                "new_errors": sum(r['new_error'] for r in rows), "cycle_weighted_error_percent": float(np.mean([r['noise_wrong'] for r in rows])*100),
                "record_equal_weight_error_percent": float(np.mean(record_error_rates)*100),
                "class_equal_weight_error_percent": float(np.mean(class_error_rates)*100),
                "represented_classes": len(class_error_rates)})
    csv_save(out / "GROUP_ERRORS.csv", groups)
    transitions = []
    for y in range(6):
        for p in range(6):
            if y==p:continue
            rows=[r for r in cycle_rows if r['label']==y and r['noise_prediction']==p]
            if rows:transitions.append({"truth": y,"prediction": p,"truth_name":NAMES[y],"prediction_name":NAMES[p],
                "window_seed_errors":len(rows),"new_errors":sum(r['new_error'] for r in rows),
                "unique_windows":len({r['window'] for r in rows}),"unique_record_families":len({r['record_id'] for r in rows})})
    csv_save(out / "CONFUSION_PAIRS.csv", sorted(transitions,key=lambda v:-v['new_errors']))
    selected=[]
    for y,p in [(4,1),(1,3),(2,0),(3,3)]:
        candidates=[]
        for i,row in enumerate(meta):
            if row['label']!=y:continue
            # NumPy比较产生np.bool_，写JSON前明确转为Python整数。
            support=int(sum(predictions_by_seed[s]['CC'][i]==y and predictions_by_seed[s]['NN'][i]==p for s in SEEDS))
            if support>=2:candidates.append((support,i))
        assert candidates,(y,p)
        candidates.sort(key=lambda z:(-z[0],meta[z[1]]['record_id'],z[1]))
        support,i=candidates[0];row=meta[i];rid=row['record_id']
        c,n=[profile(t,current,row) for t,current,_ in raw[rid]]
        selected.append({"index":i,"truth":y,"noisy_example_prediction":p,"paired_seed_support":support,
            "record_id":rid,"source_metadata":labels[rid],"window":row,
            "seed_predictions":{str(s):{k:int(predictions_by_seed[s][k][i]) for k in arrays} for s in SEEDS}})
        np.savez_compressed(out/f"example_{y}_to_{p}.npz",time_ms=(c['knots']-row['start'])*1000,
             raw_clean=c['raw'],raw_noisy=n['raw'],clean=arrays['CC'][i],noisy=arrays['NN'][i],
             clean_with_noise_parameters=arrays['CN'][i],noise_with_clean_parameters=arrays['NC'][i])
    save(out / "EXAMPLES.json", {"selection_rule":"predeclared pairs; >=2 clean-correct seeds predicting specified noisy class; maximum consensus then record ID/window index",
         "illustrative_only":True,"examples":selected})
    controls={}
    for key in arrays:
        values=[scores[s][key]['macro_f1']*100 for s in SEEDS]
        controls[key]={"macro_f1_mean_percent":float(np.mean(values)),"macro_f1_sample_sd_percent":float(np.std(values,ddof=1)),
          "per_seed_f1":dict(zip(map(str,SEEDS),values)),"windows_any_abs_gt1":int(np.sum(np.max(np.abs(arrays[key]),axis=(1,2))>1+1e-6))}
    signatures={}
    for state,rows in [('new_errors',[r for r in cycle_rows if r['new_error']]),
                       ('stable_correct',[r for r in cycle_rows if r['clean_correct'] and not r['noise_wrong']])]:
        signatures[state]={"window_seed_decisions":len(rows), **{k:summary([r[k] for r in rows]) for k in
            ['clean_classifier_margin','feature_cosine_change','joint_raw_cycle_snr_db','max_active_divisor_ratio','max_active_center_shift','active_phase_correlation_mean']}}
    phase_stats=[]
    for label in range(6):
        rows=[r for r in phases if r['label']==label]
        active=[r for r in rows if not r['clean_low']]
        phase_stats.append({"label":label,"name":NAMES[label],"phase_window_count":len(rows),
            "low_phase_windows":sum(r['clean_low'] for r in rows),'branch_changes':sum(r['branch_changed'] for r in rows),
            'active_divisor_ratio':summary([r['divisor_ratio'] for r in active]),
            'active_center_shift_abs':summary([abs(r['center_shift_in_clean_scale']) for r in active]),
            'active_raw_correlation':summary([r['raw_resampled_correlation'] for r in active]),
            'active_raw_cycle_phase_snr_db':summary([r['raw_cycle_phase_snr_db'] for r in active])})
    total_drop=controls['CC']['macro_f1_mean_percent']-controls['NN']['macro_f1_mean_percent']
    assert np.isclose(sum(r['contribution_to_macro_drop_pp'] for r in per_class),total_drop,atol=1e-12)
    assert all(sha(run/f'results/models/augmentation/seed{s}/best.pt')==weights[str(s)] for s in SEEDS)
    report={'status':'completed_and_verified','run_code_commit':'69e2926ebd09f0a9b8fe8dfdebfd15aa5a7a5481',
        'diagnostic_script_sha256':sha(Path(__file__)), 'weights_sha256':weights,'record_families':88,'windows':2474,'seeds':SEEDS,
        'control_definitions':{'CC':'Clean signal, Clean-derived center/divisor','NN':'Noisy signal, Noisy-derived center/divisor',
            'CN':'Clean signal, Noisy-derived center/divisor (diagnostic only)','NC':'Noisy signal, Clean-derived center/divisor (Clean-paired oracle diagnostic only)'},
        'controls':controls,'macro_f1_drop_pp':total_drop,'per_class':per_class,'phase_summary':phase_stats,
        'error_signatures':signatures,'noisy_error_decisions':sum(r['noise_wrong'] for r in cycle_rows),
        'new_error_decisions':sum(r['new_error'] for r in cycle_rows),'noisy_error_unique_windows':len({r['window'] for r in cycle_rows if r['noise_wrong']}),
        'noisy_error_record_families':len({r['record_id'] for r in cycle_rows if r['noise_wrong']}),
        'branch_changed_windows':sum(r['branch_changed'] for r in window_stats),'final_holdout_evaluated':False,
        'training_or_model_parameters_changed':False,'original_predictions_reproduced':True,'scope':'paired fixed-Dev sensitivity analysis; no causal attribution or deployable normalization claim'}
    save(out/'SUMMARY.json',report)
    save(out/'PER_SEED_CONTROLS.json',scores)
    print(json.dumps({k:report[k] for k in ['controls','macro_f1_drop_pp','noisy_error_decisions','new_error_decisions','noisy_error_unique_windows','noisy_error_record_families','branch_changed_windows','error_signatures']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
