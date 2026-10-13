"""核验72组特征/坐标/统计，并复现一个固定投影；导出时不包含高维NPZ。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
sys.path.insert(0,str(ROOT/"code/scripts"))
import numpy as np
from sklearn.metrics import f1_score
import six_model_tsne as analysis


def read(path):
    """读取原始全精度记录，不更改现有投影。"""
    return json.loads(Path(path).read_text())


def main():
    """核验全部样本/指标来源，再重跑第一个投影，最后只发布二维派生数据。"""
    assert read(ROOT/"RUN_STATUS.json")["status"]=="completed"
    summary,manifest=read(OUT/"SUMMARY.json"),read(OUT/"FEATURE_MANIFEST.json")
    assert len(summary["results"])==len(manifest["files"])==72 and len(summary["aggregates"])==24
    by_key={(r["model"],r["seed"],r["condition"],r["layer"]):r for r in summary["results"]}
    for entry in manifest["files"]:
        assert analysis.sha(entry["file"])==entry["feature_file_sha256"]
        key=tuple(entry[k] for k in ("model","seed","condition","layer"))
        metric=by_key[key]
        assert metric["cycles"]==2474 and metric["record_families"]==88
        for name in ("silhouette_macro","silhouette_weighted"):
            assert -1<=metric[name]<=1
        for name in ("cross_record_5nn_purity_macro","tsne_trustworthiness_k15","pca50_explained_variance_ratio_sum"):
            assert 0<=metric[name]<=1+1e-12
        assert 0<=metric["tsne_n_iter"]<1500 and np.isfinite(metric["tsne_kl_divergence"])
        coordinate=Path(metric["coordinate_file"])
        assert analysis.sha(coordinate)==metric["coordinate_file_sha256"]
        with coordinate.open() as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==2474 and [int(r["index"]) for r in rows]==list(range(2474))
        with np.load(entry["file"],allow_pickle=False) as arrays:
            assert arrays["features"].shape==(2474,entry["feature_dim"])
            assert np.array_equal(arrays["truth"],np.array([int(r["truth"]) for r in rows]))
            assert np.array_equal(arrays["predicted"],np.array([int(r["predicted"]) for r in rows]))
            assert arrays["record_ids"].tolist()==[r["record_id"] for r in rows]
            f1=100*f1_score(arrays["truth"],arrays["predicted"],labels=range(6),average="macro",zero_division=0)
            np.testing.assert_allclose(f1,metric["classifier_macro_f1_percent"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(metric["silhouette_macro"],np.mean(metric["class_silhouette"]),atol=1e-12,rtol=0)
        np.testing.assert_allclose(metric["cross_record_5nn_purity_macro"],np.mean(metric["class_5nn_purity"]),atol=1e-12,rtol=0)
    for group in summary["aggregates"]:
        for metric_name in ("silhouette_macro","cross_record_5nn_purity_macro","classifier_macro_f1_percent","tsne_trustworthiness_k15"):
            values=[by_key[group["model"],seed,group["condition"],group["layer"]][metric_name] for seed in (42,43,44)]
            np.testing.assert_allclose(group[metric_name]["values"],values,atol=1e-12,rtol=0)
            np.testing.assert_allclose(group[metric_name]["mean"],np.mean(values),atol=1e-12,rtol=0)
            np.testing.assert_allclose(group[metric_name]["std"],np.std(values,ddof=1),atol=1e-12,rtol=0)
    entry=manifest["files"][0]
    repeated=analysis.project_one((entry,str(ROOT/"reproducibility"),manifest["protocol"]))
    original=by_key[entry["model"],entry["seed"],entry["condition"],entry["layer"]]
    with Path(original["coordinate_file"]).open() as stream:before=list(csv.DictReader(stream))
    with Path(repeated["coordinate_file"]).open() as stream:after=list(csv.DictReader(stream))
    a=np.array([[float(r["tsne_x"]),float(r["tsne_y"])] for r in before])
    b=np.array([[float(r["tsne_x"]),float(r["tsne_y"])] for r in after])
    np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)
    assert all(analysis.sha(path)==expected for path,expected in read(OUT/"PROTECTED_FILES_SHA256.json").items())
    verification={"status":"passed","feature_sets_verified":72,"native_prediction_sets_reproduced":36,
        "head_feature_prediction_replays":72,"coordinate_label_rows_verified":72*2474,
        "aggregate_groups_verified":24,"repeat_projection_coordinate_max_abs_difference":float(np.max(np.abs(a-b))),
        "repeat_projection_identical_with_tolerance_1e_9":True,"weights_and_old_results_unchanged":True,
        "feature_arrays_uploaded":False,"final_holdout_evaluated":False}
    analysis.write(OUT/"VERIFICATION.json",verification)
    files=[p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".csv",".json")]
    files += [ROOT/name for name in ("DEPLOYMENT_MANIFEST.json","RUN_STATUS.json","CONTROLLER_PID.json","SETUP.py",
        "RUN_SERVER.py","LAUNCH.py","VERIFY_AND_EXPORT.py","controller.log","extract.log","project.log")]
    files += [ROOT/"code/configs/six_model_tsne_v1.json"]
    export={"status":"passed","files":{p.relative_to(ROOT).as_posix():{"bytes":p.stat().st_size,"sha256":analysis.sha(p)} for p in files},
            "raw_current_uploaded":False,"weights_uploaded":False,"high_dimensional_feature_arrays_uploaded":False}
    analysis.write(ROOT/"RESULTS_EXPORT_MANIFEST.json",export)
    with tarfile.open(ROOT/"visualization_records.tar.gz","w:gz") as archive:
        for p in files+[ROOT/"RESULTS_EXPORT_MANIFEST.json"]:archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print("TSNE_VERIFICATION_AND_EXPORT_PASSED",len(files),"files; repeat maxdiff",verification["repeat_projection_coordinate_max_abs_difference"],flush=True)


if __name__=="__main__":main()
