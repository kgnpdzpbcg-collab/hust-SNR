"""部署冻结V4代码，在源域完成输入、噪声、梯度和轨迹检查。"""
from pathlib import Path
import argparse
import binascii
import hashlib
import json
import os
import subprocess
import struct
import sys
import tarfile
import zlib

RUN = Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v4/a0c78d8_20261019')
V3 = RUN.parents[1] / 'jung_hang_cycle_v3/853b1f6_20261019'
ARCHIVE = Path('/tmp/hust_jung_hang_v4_a0c78d8.tar')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf8', newline='\n')


def write_gray_png(path, pixels):
    """用标准库保存8位灰度检查图，无需给既有训练环境增加绘图依赖。"""
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', binascii.crc32(kind + payload) & 0xffffffff)
    height, width = pixels.shape
    scanlines = b''.join(b'\0' + row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 0, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def main():
    """只读取源域波形；历史文件指纹先核对，再保存新实验预检记录。"""
    resume = '--resume-after-missing-matplotlib' in sys.argv
    if resume:
        assert RUN.exists() and not (RUN/'DEPLOYMENT_MANIFEST.json').exists() and not (RUN/'results').exists()
        assert '26 passed' in (RUN/'logs/pytest.log').read_text()
    else:
        assert not RUN.exists(), '拒绝覆盖既有V4目录'
        RUN.mkdir(parents=True)
        (RUN / 'logs').mkdir()
        with tarfile.open(ARCHIVE) as archive:
            archive.extractall(RUN / 'code', filter='data')
    previous = read(V3 / 'DEPLOYMENT_MANIFEST.json')
    protected = read(V3 / 'PROTECTED_FILES_BEFORE.json')
    for name, expected in protected.items():
        assert sha(Path(name)) == expected, name
    for path in V3.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts:
            protected[str(path)] = sha(path)
    write(RUN / 'PROTECTED_FILES_BEFORE.json', protected)
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8', MPLBACKEND='Agg')
    scripts = RUN / 'code/scripts'
    compile_paths = [scripts / name for name in ('jung_hang_models_v4.py', 'compare_jung_hang_v4.py',
                                               'jung_hang_cycle_v4.py', 'jung_hang_v4_presets.py')]
    tests = ['tests/test_jung_hang_models.py', 'tests/test_jung_hang_models_v3.py',
             'tests/test_jung_hang_models_v4.py', 'tests/test_jung_hang_cycle_v3_noise.py']
    presets = RUN / 'preset_configs'
    if not resume:
        subprocess.run([sys.executable, '-m', 'py_compile', *map(str, compile_paths)], check=True)
        with (RUN / 'logs/pytest.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=RUN / 'code',
                                    stdout=log, stderr=subprocess.STDOUT)
        print((RUN / 'logs/pytest.log').read_text(), flush=True)
        assert result.returncode == 0, '单元测试失败，停止部署'
        subprocess.run([sys.executable, str(scripts / 'jung_hang_v4_presets.py'), '--output', str(presets)], check=True)
    assert len(list(presets.glob('*.json'))) == 9
    sys.path.insert(0, str(scripts))
    import numpy as np
    import torch
    import jung_hang_cycle_v4 as runner
    torch.set_num_threads(2)
    args = argparse.Namespace(config=RUN / 'code/configs/jung_hang_cycle_v4_v1.json',
                             reference_root=Path(previous['reference_root']), source_root=Path(previous['source_root']),
                             additional_root=Path(previous['additional_root']), previous_output=Path(previous['previous_output']),
                             output=RUN / 'results')
    command = [sys.executable, str(scripts / 'jung_hang_cycle_v4.py'), 'preflight',
               '--config', str(args.config), '--reference-root', str(args.reference_root),
               '--source-root', str(args.source_root), '--additional-root', str(args.additional_root),
               '--previous-output', str(args.previous_output), '--output', str(args.output)]
    with (RUN / 'logs/preflight.log').open('w') as log:
        subprocess.run(command, cwd=RUN / 'code', stdout=log, stderr=subprocess.STDOUT, check=True)
    print((RUN / 'logs/preflight.log').read_text(), flush=True)
    checks, max_snr_error = [], 0.0
    for method in ('jung', 'hang'):
        config, ref, paper, provenance, raw, metadata, y, clean, vy, val = runner.source_data(args, method)
        ids = [int(np.flatnonzero(y == label)[0]) for label in range(6)]
        model = paper.get_model(method, config).cuda().train()
        x = torch.from_numpy(clean[ids]).cuda()
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor(y[ids], device='cuda'))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert not any(np.any(~np.isfinite(values)) for values in val.values())
        checks.append({'method': method, 'train_shape': list(clean.shape), 'validation_shape': list(val['clean'].shape),
                       'parameters': sum(p.numel() for p in model.parameters()), 'finite_cuda_gradients': True})
        del model
        if method == 'hang':
            # 每类固定取清单中的首个训练周期；同时画类均值，避免按图形美观选样本。
            # 六列从左到右对应0~5类；四行分别为样本平面1/2、类均值平面1/2。
            canvas = np.full((720, 1072), 255, dtype=np.uint8)
            selected, centroids = [], []
            for label, index in enumerate(ids):
                mean_image = clean[y == label].mean(axis=0)
                centroids.append(mean_image.reshape(-1))
                selected.append({'label': label, 'metadata': metadata[index], 'train_index': index})
                for plane in range(2):
                    for row, image in ((plane, clean[index,plane]), (plane+2,mean_image[plane])):
                        tile = np.repeat(np.repeat((255*(1-image[::-1])).astype(np.uint8),4,axis=0),4,axis=1)
                        top, left = 16+row*176,16+label*176
                        canvas[top-1:top+161,left-1:left+161]=0
                        canvas[top:top+160,left:left+160]=tile
            write_gray_png(RUN / 'SOURCE_HANG_TRAJECTORIES.png', canvas)
            centroids = np.stack(centroids)
            hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in clean]
            by_hash = {}
            for label, value in zip(y, hashes):
                by_hash.setdefault(value, set()).add(int(label))
            write(RUN / 'SOURCE_HANG_INPUT_AUDIT.json', {
                'examples': selected, 'train_class_counts': np.bincount(y, minlength=6).tolist(),
                'image_layout': 'columns: labels0..5; rows: example plane1, example plane2, class mean plane1, class mean plane2',
                'val_class_counts': np.bincount(vy, minlength=6).tolist(),
                'all_zero_train_images': int(np.sum(np.all(clean == 0, axis=(1,2,3)))),
                'all_zero_val_images': int(np.sum(np.all(val['clean'] == 0, axis=(1,2,3)))),
                'identical_train_image_hashes_shared_between_labels': sum(len(v)>1 for v in by_hash.values()),
                'class_centroid_pairwise_l2': np.linalg.norm(centroids[:,None]-centroids[None,:], axis=-1).tolist(),
                'target_waveforms_read': False})
    # 在实际源记录上核对五相整记录SNR，10~40与0/5验证边界均覆盖。
    rid = sorted(raw)[0]
    time, currents, omega = raw[rid]
    snr_checks = []
    for snr in (0, 5, 10, 20, 40):
        noisy = paper.add_noise(time, currents, omega, snr, np.random.default_rng(42), mode=config['training_noise'])
        actual = 10*np.log10(np.mean(currents**2, axis=0) / np.mean((noisy-currents)**2, axis=0))
        error = float(np.max(np.abs(actual-snr)))
        assert error < 1e-9
        max_snr_error = max(max_snr_error, error)
        snr_checks.append({'requested_db':snr, 'actual_phase_db': actual.tolist()})
    manifest = {key: previous[key] for key in ('reference_root', 'source_root', 'additional_root', 'previous_output', 'reference_reader_commit')}
    manifest.update(model_branch='research/jung-hang-paper-fidelity-v4', model_commit='a0c78d8bf253082efd39b7b93428322dde5ccb67',
                    config=config, config_sha256=sha(args.config), source_split=provenance, interpreter=sys.executable,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0), previous_v3_root=str(V3), v2_root=previous['v2_root'],
                    tests_passed=26, preset_configs_generated=9, experiment_scope='default two-model config; presets not trained',
                    cuda_gradient_checks=checks, real_source_phase_snr_checks=snr_checks,
                    max_training_full_record_phase_snr_error_db=max_snr_error,
                    code_sha256={p.relative_to(RUN/'code').as_posix():sha(p) for p in (RUN/'code').rglob('*')
                                 if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts},
                    final_holdout_evaluated=False, workers=4, cpu_threads_per_worker=2,
                    upstream_code_modified=False, protected_files=len(protected))
    write(RUN / 'DEPLOYMENT_MANIFEST.json', manifest)
    print(json.dumps({'run':str(RUN), 'status':'passed', 'checks':checks, 'max_snr_error':max_snr_error,
                      'protected_files':len(protected)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
