"""核验七模型单种子28组特征、坐标、指标及旧投影复用，再导出客观记录。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
sys.path.insert(0,str(ROOT/'code/scripts'))
import numpy as np
from sklearn.metrics import pairwise_distances,silhouette_samples,f1_score
from six_model_tsne import read,write,sha,project_one

def main():
    """原数组与当前坐标逐项核对；原空间指标独立重新计算，不从二维图推断。"""
    assert read(ROOT/'RUN_STATUS.json')['status'] == 'completed'
    summary = read(OUT/'SUMMARY.json')
    manifest = read(OUT/'FEATURE_MANIFEST.json')
    choice = read(OUT/'DISPLAY_SEED_SELECTION.json')
    assert summary['selected_seed'] == manifest['selected_seed'] == choice['selected_seed'] == 44
    assert sorted(choice['candidates'],key=lambda row:(-row['mean_macro_f1'],row['seed']))[0]['seed'] == 44
    assert len(summary['results']) == len(manifest['files']) == 28
    assert sum(row['reused_projection'] for row in manifest['files']) == 12
    for entry in manifest['files']:
        assert sha(entry['file']) == entry['feature_file_sha256']
        assert sha(entry['weights']) == entry['weights_sha256']
        assert sha(entry['original_prediction_file']) == entry['prediction_file_sha256']
        original = read(entry['original_prediction_file'])
        with np.load(entry['file'],allow_pickle=False) as arrays:
            features = arrays['features']
            truth,predicted,ids = (arrays[key] for key in ('truth','predicted','record_ids'))
            assert hashlib.sha256(features.tobytes()).hexdigest() == entry['feature_array_sha256']
        assert features.shape == (2474,128 if entry['model'] == 'resnet1d' else 352)
        assert np.array_equal(truth,original['truth']) and np.array_equal(predicted,original['predicted'])
        assert len(set(ids)) == 88 and np.bincount(truth,minlength=6).tolist() == [1106,307,297,298,237,229]
        result = next(row for row in summary['results'] if (row['model'],row['condition']) == (entry['model'],entry['condition']))
        path = Path(result['coordinate_file'])
        assert sha(path) == result['coordinate_file_sha256']
        with path.open() as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == 2474 and [int(row['index']) for row in rows] == list(range(2474))
        assert [row['record_id'] for row in rows] == ids.tolist()
        assert [int(row['truth']) for row in rows] == truth.tolist()
        assert [int(row['predicted']) for row in rows] == predicted.tolist()
        coords = np.array([[float(row['tsne_x']),float(row['tsne_y'])] for row in rows])
        assert np.isfinite(coords).all()
        actual_f1 = 100*f1_score(truth,predicted,labels=list(range(6)),average='macro',zero_division=0)
        np.testing.assert_allclose(result['classifier_macro_f1_percent'],actual_f1,atol=1e-12,rtol=0)
        # 使用原始向量独立重算宏轮廓系数和排除同文件的5NN同类率。
        features = features.astype(np.float64)
        features /= np.sqrt(np.mean(features**2))
        distance = pairwise_distances(features)
        np.fill_diagonal(distance,0)
        silhouette = silhouette_samples(distance,truth,metric='precomputed')
        s = np.mean([silhouette[truth == label].mean() for label in range(6)])
        np.testing.assert_allclose(result['silhouette_macro'],s,atol=1e-12,rtol=0)
        distance[ids[:,None] == ids[None,:]] = np.inf
        nearest = np.argpartition(distance,4,axis=1)[:,:5]
        assert not np.any(ids[nearest] == ids[:,None])
        purity = (truth[nearest] == truth[:,None]).mean(1)
        p = np.mean([purity[truth == label].mean() for label in range(6)])
        np.testing.assert_allclose(result['cross_record_5nn_purity_macro'],p,atol=1e-12,rtol=0)
        assert .0 <= result['tsne_trustworthiness_k15'] <= 1
        assert 0 < result['pca50_explained_variance_ratio_sum'] <= 1+1e-12
        if entry['reused_projection']:
            old = read(entry['previous_metric'])
            assert path.read_bytes() == Path(old['coordinate_file']).read_bytes()
        print('FEATURE_COORDINATES_VERIFIED',entry['model'],entry['condition'],flush=True)
    # 重复一组新增ResNet128维0 dB投影，以检查不同原生维数下仍可复现。
    repeated_entry = next(row for row in manifest['files'] if (row['model'],row['condition']) == ('resnet1d','snr0'))
    repeated = project_one((repeated_entry,str(ROOT/'reproducibility'),summary['protocol']))
    actual = next(row for row in summary['results'] if (row['model'],row['condition']) == ('resnet1d','snr0'))
    def coordinates(path):
        """只对二维数值比较，重复计算路径不参与正式结果。"""
        with Path(path).open() as stream:
            return np.array([[float(row['tsne_x']),float(row['tsne_y'])] for row in csv.DictReader(stream)])
    difference = float(np.abs(coordinates(repeated['coordinate_file'])-coordinates(actual['coordinate_file'])).max())
    assert difference <= 1e-9
    protected = read(OUT/'PROTECTED_FILES_SHA256.json')
    assert all(sha(path) == expected for path,expected in protected.items())
    write(OUT/'VERIFICATION.json',{'status':'passed','feature_and_coordinate_sets_verified':28,
        'new_native_and_feature_head_prediction_replays':16,'previous_verified_prediction_sets_reused':12,
        'original_space_metric_sets_independently_recomputed':28,'coordinate_label_rows_verified':69272,
        'previous_coordinate_files_byte_identical':12,'repeat_resnet0_projection_max_abs_difference':difference,
        'selected_seed':44,'display_seed_selected_posthoc_on_dev':True,
        'old_models_and_results_unchanged':True,'feature_arrays_uploaded':False,'final_holdout_evaluated':False})
    paths = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.log')]
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.csv')]
    paths += [ROOT/'code/configs/seven_model_tsne_v1.json']
    export = {'status':'passed','files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}}
    record = ROOT/'RESULTS_EXPORT_MANIFEST.json'
    write(record,export)
    with tarfile.open(ROOT/'seven_model_records.tar.gz','w:gz') as archive:
        for path in paths+[record]: archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    print('SEVEN_MODEL_VERIFIED_AND_EXPORTED',len(paths),'FILES',flush=True)

if __name__ == '__main__': main()
