"""六模型固定权重的特征提取与t-SNE：352维编码器为主，50维分类前为辅。

原目标Dev全部2474周期、三种子42/43/44、Clean和10 dB。提取不更新权重；
降维不使用标签；模型分别拟合，不比较不同图坐标或簇间绝对距离。
高维特征只保存在服务器；发布二维坐标、指标、来源与绘图文件。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

CONFIG = Path(__file__).resolve().parents[1]/"configs/six_model_tsne_v1.json"


def read(path):
    """读取冻结配置/结果；不从预测分数选择样本或种子。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    """指标/来源记录写入独立分析目录，不覆盖原实验。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def sha(path):
    """流式SHA保护原权重、预测和已冻结输入。"""
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):digest.update(block)
    return digest.hexdigest()


def save_csv(path, rows):
    """发布点索引、标签/预测和二维坐标，不发布高维向量或原始电流。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def config():
    """实验设置固定为所有同种子/同周期模型，不接受挑选最佳图的参数搜索。"""
    p=read(CONFIG)
    assert p["models"]==["identity","lowpass3","lowpass4","fixed_half","nahsf","xu_full"]
    assert p["seeds"]==[42,43,44] and p["conditions"]==["clean","snr10"]
    assert p["feature_layers"]=={"encoder352":352,"classifier50":50}
    return p


def extract(base, output):
    """读取六模型各自最佳权重，钩取分类器首层/末层输入，重放原预测。

    A′系列首层是Flatten，输入[B,32,11]展开352维；Xu首层是Linear，
    输入已展开[B,352]。末层Linear输入均为50维隐藏特征，均不是六类logits。
    """
    import torch
    import four_models_expanded_val as common
    import fixed_half_experiment as fixed
    import xu2025_full_experiment as xu
    import v2_experiment as data
    from v1_noise_consistency import fix_randomness
    from evaluate_v4_strong_snr import frozen_ids
    from prepare import OUT,ROOT
    p=config()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    roots={m:base/"experiments/four_models_expanded_val/56513c7/results" for m in common.METHODS}
    roots["fixed_half"]=base/"experiments/fixed_half_fusion/3c38037/results"
    roots["xu_full"]=base/"experiments/xu2025_full/bb2f219/results/matched"
    for method, root in roots.items():
        verification=root/("reports/VERIFICATION.json" if method=="xu_full" else "VERIFICATION.json")
        assert read(verification)["status"]=="passed"
    protected=read(base/"experiments/xu2025_full/bb2f219/ORIGINAL_FILES_SHA256.json")
    for root in set(roots.values()):
        protected.update({str(file):sha(file) for file in root.rglob("*") if file.is_file()})
    assert all(sha(Path(path))==expected for path,expected in protected.items())
    write(output/"PROTECTED_FILES_SHA256.json",protected)
    metadata=read(OUT/"target_dev_clean_windows.json")
    frozen_ids(read(ROOT/"configs/splits_gate0_v1.json"),metadata)
    truth=np.array([r["label"] for r in metadata],dtype=np.int64)
    ids=np.array([r["record_id"] for r in metadata])
    manifest=[]
    for condition in p["conditions"]:
        raw,actual=data.target_dev_from_mat(None if condition=="clean" else 10,
            base/"experiments/four_models_expanded_val/56513c7/target_snr")
        assert actual==metadata
        x=data.raw_cycles(raw,metadata)
        for method in p["models"]:
            for seed in p["seeds"]:
                fix_randomness(seed)
                folder=roots[method]/f"models/{method}/seed{seed}"
                checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
                assert checkpoint["seed"]==seed and checkpoint["source_only"] and not checkpoint["final_holdout_evaluated"]
                if method=="xu_full":
                    assert checkpoint["mode"]==method and checkpoint["track"]=="matched"
                    model=xu.make_model(method,"matched").cuda().eval()
                    classifier=model.network.classifier
                else:
                    if method=="fixed_half":
                        common.check_checkpoint(checkpoint,method,seed,protocol=fixed.config(),version=fixed.VERSION)
                        model=fixed.make_model(method).cuda().eval()
                    else:
                        common.check_checkpoint(checkpoint,method,seed)
                        model=common.make_model(method).cuda().eval()
                    classifier=model.backbone.classifier
                model.load_state_dict(checkpoint["state"],strict=True)
                before={k:value.detach().cpu().clone() for k,value in model.state_dict().items()}
                captured={"encoder352":[],"classifier50":[]}
                def tap(name):
                    """只观察层输入，不改变数据或返回值。"""
                    def capture(_module, inputs):
                        value=inputs[0].detach().flatten(1).cpu().numpy().copy()
                        assert value.shape[1]==p["feature_layers"][name]
                        captured[name].append(value)
                    return capture
                hooks=[classifier[0].register_forward_pre_hook(tap("encoder352")),
                       classifier[-1].register_forward_pre_hook(tap("classifier50"))]
                predicted=[]
                with torch.no_grad():
                    for start in range(0,len(x),128):
                        result=model(torch.from_numpy(x[start:start+128]).cuda())
                        logits=result["logits"] if method=="xu_full" else result
                        predicted.append(logits.argmax(1).cpu().numpy())
                for hook in hooks:hook.remove()
                predicted=np.concatenate(predicted)
                source_prediction_path=roots[method]/f"evaluations/{method}/seed{seed}/target_dev_{condition}.json"
                original=read(source_prediction_path)
                assert np.array_equal(original["truth"],truth) and np.array_equal(original["predicted"],predicted)
                for layer,parts in captured.items():
                    features=np.concatenate(parts)
                    assert features.shape==(2474,p["feature_layers"][layer]) and np.isfinite(features).all()
                    # 从保存的层输入重新经过分类头，验证钩取的是当前模型的真实表示。
                    head=classifier if layer=="encoder352" and method=="xu_full" else (
                        classifier[1:] if layer=="encoder352" else classifier[-1])
                    replay=[]
                    with torch.no_grad():
                        for start in range(0,len(features),128):
                            replay.append(head(torch.from_numpy(features[start:start+128]).cuda()).argmax(1).cpu().numpy())
                    assert np.array_equal(predicted,np.concatenate(replay))
                    name=f"{method}_seed{seed}_{condition}_{layer}"
                    path=output/"features"/(name+".npz")
                    path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path,features=features,truth=truth,predicted=predicted,record_ids=ids)
                    manifest.append({"model":method,"seed":seed,"condition":condition,"layer":layer,
                        "feature_dim":p["feature_layers"][layer],"file":str(path),"feature_array_sha256":hashlib.sha256(features.tobytes()).hexdigest(),
                        "feature_file_sha256":sha(path),"weights":str(folder/"best.pt"),"weights_sha256":sha(folder/"best.pt"),
                        "original_prediction_file":str(source_prediction_path),"prediction_file_sha256":sha(source_prediction_path),
                        "selected_epoch":checkpoint["selected"]["epoch"],"macro_f1":original["metrics"]["macro_f1"],
                        "captured_head_predictions_exact":True})
                assert all(torch.equal(value,before[name]) for name,value in model.state_dict().items())
                del model
                print("FEATURE_EXTRACTED",method,seed,condition,"352+50 dims; original predictions exact",flush=True)
    assert len(manifest)==72 and all(sha(Path(path))==expected for path,expected in protected.items())
    write(output/"FEATURE_MANIFEST.json",{"protocol":p,"files":manifest,"original_weights_and_results_unchanged":True,
        "class_counts":np.bincount(truth,minlength=6).tolist(),"target_metadata_sha256":sha(OUT/"target_dev_clean_windows.json")})


def project_one(task):
    """单个模型/种子/条件/层的统一投影及原空间可分性，不用标签拟合降维。"""
    entry,output_str,p=task
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE,trustworthiness
    from sklearn.metrics import pairwise_distances,silhouette_samples
    output=Path(output_str)
    started=time.monotonic()
    path=Path(entry["file"])
    assert sha(path)==entry["feature_file_sha256"]
    with np.load(path,allow_pickle=False) as arrays:
        original=arrays["features"].astype(np.float64)
        truth,predicted,ids=(arrays[key] for key in ("truth","predicted","record_ids"))
    # 一次全局尺度归一化，不做逐维或逐样本归一化，保留原欧氏几何。
    scale=float(np.sqrt(np.mean(original**2)))
    assert np.isfinite(scale) and scale>0
    features=original/scale
    distances=pairwise_distances(features,metric="euclidean",n_jobs=1)
    np.fill_diagonal(distances,0.)
    silhouette=silhouette_samples(distances,truth,metric="precomputed")
    # 排除同一原始记录的全部周期，避免近重复周期人为提高邻居纯度。
    same=ids[:,None]==ids[None,:]
    distances[same]=np.inf
    neighbors=np.argpartition(distances,4,axis=1)[:,:5]
    assert not np.any(ids[neighbors]==ids[:,None])
    purity=(truth[neighbors]==truth[:,None]).mean(axis=1)
    class_sil=[float(silhouette[truth==label].mean()) for label in range(6)]
    class_purity=[float(purity[truth==label].mean()) for label in range(6)]
    del distances,same
    pca=PCA(n_components=50,svd_solver="full")
    reduced=pca.fit_transform(features)
    settings=p["tsne"].copy()
    settings.pop("initialization")
    initial=np.random.default_rng(settings["random_state"]).normal(0,1e-4,(len(features),2)).astype(np.float64)
    embedding=TSNE(**settings,init=initial,n_jobs=1)
    coords=embedding.fit_transform(reduced)
    assert coords.shape==(2474,2) and np.isfinite(coords).all()
    quality=float(trustworthiness(features,coords,n_neighbors=15,metric="euclidean"))
    name=f"{entry['model']}_seed{entry['seed']}_{entry['condition']}_{entry['layer']}"
    coordinate_path=output/"coordinates"/(name+".csv")
    save_csv(coordinate_path,[{"index":i,"record_id":str(ids[i]),"truth":int(truth[i]),"predicted":int(predicted[i]),
                              "tsne_x":float(coords[i,0]),"tsne_y":float(coords[i,1])} for i in range(len(truth))])
    result={"model":entry["model"],"seed":entry["seed"],"condition":entry["condition"],"layer":entry["layer"],
        "feature_dim":entry["feature_dim"],"cycles":2474,"record_families":88,"global_rms_scale":scale,
        "pca50_explained_variance_ratio_sum":float(pca.explained_variance_ratio_.sum()),
        "silhouette_weighted":float(silhouette.mean()),"silhouette_macro":float(np.mean(class_sil)),
        "cross_record_5nn_purity_macro":float(np.mean(class_purity)),"class_silhouette":class_sil,"class_5nn_purity":class_purity,
        "tsne_trustworthiness_k15":quality,"tsne_kl_divergence":float(embedding.kl_divergence_),
        "tsne_n_iter":int(embedding.n_iter_),"classifier_macro_f1_percent":100*entry["macro_f1"],
        "coordinate_file":str(coordinate_path),"coordinate_file_sha256":sha(coordinate_path),
        "elapsed_seconds":time.monotonic()-started,"labels_used_for_tsne_fit":False}
    write(output/"metrics"/(name+".json"),result)
    return result


def project(output):
    """四个独立CPU进程、各四线程，统一固定参数执行全部72张投影。"""
    p=config()
    if (output/"coordinates").exists() or (output/"metrics").exists():raise FileExistsError("拒绝覆盖投影")
    manifest=read(output/"FEATURE_MANIFEST.json")
    assert manifest["protocol"]==p and len(manifest["files"])==72
    started=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(project_one,(entry,str(output),p)) for entry in manifest["files"]]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print("TSNE_COMPLETED",len(results),"/72",result["model"],result["seed"],result["condition"],result["layer"],flush=True)
    results.sort(key=lambda r:(r["layer"],r["condition"],r["seed"],p["models"].index(r["model"])))
    rows=[{k:r[k] for k in ("model","seed","condition","layer","feature_dim","silhouette_weighted","silhouette_macro",
           "cross_record_5nn_purity_macro","tsne_trustworthiness_k15","classifier_macro_f1_percent","pca50_explained_variance_ratio_sum")} for r in results]
    save_csv(output/"FEATURE_SEPARABILITY_PER_SEED.csv",rows)
    aggregates=[]
    for layer in p["feature_layers"]:
        for condition in p["conditions"]:
            for method in p["models"]:
                group=[r for r in results if (r["layer"],r["condition"],r["model"])==(layer,condition,method)]
                assert len(group)==3 and sorted(r["seed"] for r in group)==p["seeds"]
                row={"layer":layer,"condition":condition,"model":method}
                for key in ("silhouette_macro","cross_record_5nn_purity_macro","classifier_macro_f1_percent","tsne_trustworthiness_k15"):
                    values=[next(r[key] for r in group if r["seed"]==seed) for seed in p["seeds"]]
                    row[key]={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"values":values}
                aggregates.append(row)
    write(output/"SUMMARY.json",{"protocol":p,"aggregates":aggregates,"results":results,"std_ddof":1,
                                  "projection_wall_seconds":time.monotonic()-started,"final_holdout_evaluated":False})
    protected=read(output/"PROTECTED_FILES_SHA256.json")
    assert all(sha(Path(path))==value for path,value in protected.items())
    print("SIX_MODEL_TSNE_COMPLETED",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("extract","project"))
    parser.add_argument("--base",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.stage=="extract":
        if args.base is None:parser.error("特征提取需要既有服务器项目根目录")
        extract(args.base,args.output)
    else:project(args.output)
