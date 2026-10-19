"""Synthetic MAT tests for scripts/export_snr_data.py (no target Holdout access)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_snr_data.py"
spec = importlib.util.spec_from_file_location("export_snr_data", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def fake_clean_and_noisy(snr=10):
    # Real input shape and repeated time stamps: 10031 rows, 10011 unique.
    t = np.arange(10011, dtype=float) / 10000
    t = np.insert(t, np.arange(100, 120), t[np.arange(100, 120)])
    assert len(t) == 10031 and np.count_nonzero(np.diff(t) == 0) == 20
    current = np.stack([2 + np.sin(t * 2 * np.pi * (q + 1)) for q in range(5)], axis=1)
    omega = np.ones((len(t), 1), dtype=float) * 2 * np.pi * 50
    clean = np.column_stack([t, current, omega])
    g = np.random.default_rng(17)
    noise = g.standard_normal(current.shape)
    noise *= (np.sqrt(np.mean(current ** 2, axis=0) / np.mean(noise ** 2, axis=0)) / 10 ** (snr / 20))
    noisy = np.column_stack([t, current + noise, omega])
    return clean, noisy


def test_raw_pair_merge_and_float32_export(tmp_path):
    rid = "0081"
    root = tmp_path / "snr"
    (root / "SNR_10").mkdir(parents=True)
    clean, noisy = fake_clean_and_noisy()
    savemat(tmp_path / "clean.mat", {"name": clean})
    savemat(root / "SNR_10" / f"{rid}.mat", {"I_noisy": noisy})
    c = export.load_raw(tmp_path / "clean.mat", "name")
    n = export.load_raw(export.resolve_noisy(root, 10, rid), "I_noisy")
    assert export.assert_paired(c, n, 10, rid) < 1e-10
    t, currents, omega = export.merge_repeated_time(c)
    assert len(t) == 10011 and currents.shape == (10011, 5)
    assert np.allclose(omega, 2*np.pi*50)
    ref = {"time": t[None], "omega": omega[None]}
    curr, mismatch = export.collect_condition([rid], ref, {rid: c}, root, 10)
    assert curr.shape == (1, 10011, 5) and curr.dtype == np.float32
    assert mismatch < 1e-10
    outputs = export.write_chunks(tmp_path, "snr10", curr, 1)
    assert len(outputs) == 1
    with np.load(tmp_path / outputs[0]["file"], allow_pickle=False) as f:
        assert np.array_equal(f["currents"], curr)
        assert f["record_indices"].tolist() == [0]
    tampered = n.copy()
    tampered[0, 1] += 1
    with pytest.raises(ValueError, match="SNR mismatch"):
        export.assert_paired(c, tampered, 10, rid)


def test_frozen_allowlist_and_holdout_guard():
    dev = [f"ID{i}" for i in range(88)]
    holdout = [f"H{i}" for i in range(347)]
    split = {"target_dev_records": dev, "target_holdout_records": holdout,
             "target_canonical_records": dev + holdout, "target_duplicate_groups": []}
    labels = {"experimental": {rid: {"fault_label": 1} for rid in dev}}
    assert export.select_dev(split, labels) == sorted(dev)
    split["target_dev_records"] = [*dev[:-1], "H0"]
    with pytest.raises(ValueError, match="invalid"):
        export.select_dev(split, labels)
    split["target_dev_records"] = dev
    split["target_duplicate_groups"] = [["ID0", "H0"]]
    with pytest.raises(ValueError, match="Duplicate group"):
        export.select_dev(split, labels)


def test_invalid_mismatch_and_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        export.resolve_noisy(tmp_path, 10, "0081")
    clean, noisy = fake_clean_and_noisy()
    noisy[0, -1] += 1
    with pytest.raises(ValueError, match="Time/omega mismatch"):
        export.assert_paired(clean, noisy, 10, "0081")


def test_chunk_indices_are_global(tmp_path):
    arr = np.arange(5*7*5, dtype=np.float32).reshape(5, 7, 5)
    outputs = export.write_chunks(tmp_path, "snr20", arr, 2)
    assert [x["records"] for x in outputs] == [2, 2, 1]
    for item in outputs:
        with np.load(tmp_path / item["file"], allow_pickle=False) as npz:
            indexes = npz["record_indices"]
            assert np.array_equal(npz["currents"], arr[indexes])


def test_windows_reject_wrong_label():
    ids = [f"ID{i}" for i in range(88)]
    labels = {"experimental": {rid: {"fault_label": 1} for rid in ids}}
    windows = [{"record_id": rid, "label": 0, "start": 0.1, "end": 0.2} for rid in ids]
    windows += [{"record_id": ids[i % 88], "label": 1, "start": 0.6, "end": 0.61}
                for i in range(2474-88)]
    result = export.validate_windows(windows, ids, labels)
    assert result["total_windows"] == 2474
    windows[-1]["label"] = 0
    with pytest.raises(ValueError, match="Invalid window"):
        export.validate_windows(windows, ids, labels)


def test_end_to_end_export_synthetic_dev_pair(tmp_path, monkeypatch):
    """Exercise actual archive pipeline with 2 synthetic Dev records, not the official 88."""
    repo = tmp_path / "repo"
    snr_root = tmp_path / "SNR_parent"
    ids = ["0081", "4-300-1"]
    (repo / "configs").mkdir(parents=True)
    (repo / "analysis" / "data_audit_v1").mkdir(parents=True)
    (repo / "outputs" / "gate0_v1").mkdir(parents=True)
    (repo / "data" / "target_dev" / "clean").mkdir(parents=True)
    split = {"target_dev_records": ids, "target_holdout_records": ["H0"],
             "target_canonical_records": ids + ["H0"], "target_duplicate_groups": []}
    labels = {"experimental": {rid: {"fault_label": label, "speed": 300, "torque": 3}
                               for rid, label in zip(ids, (1, 4))}}
    windows = [{"record_id": rid, "label": 0, "start": 0.1, "end": 0.2} for rid in ids]
    for rid, label in zip(ids, (1, 4)):
        windows.append({"record_id": rid, "label": label, "start": 0.6, "end": 0.62})
    (repo / "configs" / "splits_gate0_v1.json").write_text(json.dumps(split))
    (repo / "analysis" / "data_audit_v1" / "LABELS.json").write_text(json.dumps(labels))
    (repo / "outputs" / "gate0_v1" / "target_dev_clean_windows.json").write_text(json.dumps(windows))
    for i, rid in enumerate(ids):
        clean, _ = fake_clean_and_noisy(10)
        clean[:, 1:6] *= (i + 1)
        savemat(repo / "data" / "target_dev" / "clean" / f"{rid}.mat", {"name": clean})
        for snr in export.SNR_LEVELS:
            dst = snr_root / f"SNR_{snr}"
            dst.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(17 + snr + i)
            noise = rng.normal(size=(len(clean), 5))
            power = np.mean(clean[:, 1:6] ** 2, axis=0)
            noise *= (np.sqrt(power / np.mean(noise ** 2, axis=0)) / 10 ** (snr / 20))
            noisy = clean.copy()
            noisy[:, 1:6] += noise
            savemat(dst / f"{rid}.mat", {"I_noisy": noisy})
    # Only bypass the 88/2474 count precondition for an isolated integration fixture.
    monkeypatch.setattr(export, "select_dev", lambda split, labels: sorted(ids))
    monkeypatch.setattr(export, "validate_windows", lambda windows, ids, labels: {"total_windows": len(windows)})
    out = repo / "outputs" / "tiny_export"
    m = export.export_dataset(repo, snr_root, out, records_per_file=1)
    assert m["final_holdout_evaluated_or_exported"] is False
    assert set(m["conditions"]) == {"clean", *[f"snr{s}" for s in export.SNR_LEVELS]}
    assert len(list(out.glob("dev_snr*_part*.npz"))) == 14
    with np.load(out / "dev_reference.npz", allow_pickle=False) as d:
        assert d["record_ids"].tolist() == sorted(ids)
        assert d["time"].shape == (2, 10011)
    with np.load(out / "dev_snr10_part02.npz", allow_pickle=False) as d:
        assert d["record_indices"].tolist() == [1]
        assert d["currents"].shape == (1, 10011, 5)
    with pytest.raises(FileExistsError):
        export.export_dataset(repo, snr_root, out, records_per_file=1)
