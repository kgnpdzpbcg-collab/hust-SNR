"""只读审计HUST五相AA数据，不训练模型、不生成最终训练/测试划分。

从两份Excel核对标签，用Windows自带tar解压原始记录到hust的分析缓存，
逐文件检查MAT字段、数值、时间轴，以及七档SNR与常规实测的对应关系。
输出记录清单和审计JSON；原数据目录保持只读，无需新建环境或安装依赖。
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, json, platform, subprocess, sys
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import scipy
from scipy.io import loadmat

SNR_LEVELS = (10, 15, 20, 25, 30, 35, 40)
FAULT_NAMES = {1:"OUS(A)", 2:"OLS(A)", 3:"OP(A)", 4:"OUS(A)&OUS(B)", 5:"OLS(A)&OP(C)"}
XML_NS = {"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def file_sha(path):
    """流式校验大压缩包，避免一次性读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    """按连续float64原字节识别完全相同的波形；不表示近似或物理独立性。"""
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def labels_from_xlsx(path):
    """直接读取XLSX的XML，按文件名列筛出有效记录，忽略工作表空白格式行。

    返回编号→工况/故障字典。Excel只有故障1–5，不能据此给完整记录贴标签。
    """
    with zipfile.ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            for item in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("m:si", XML_NS):
                strings.append("".join(t.text or "" for t in item.findall(".//m:t", XML_NS)))
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
        records = {}
        for row in sheet.findall(".//m:sheetData/m:row", XML_NS)[1:]:
            cells = {}
            for cell in row.findall("m:c", XML_NS):
                node = cell.find("m:v", XML_NS)
                value = None if node is None else node.text
                if cell.get("t") == "s" and value is not None:
                    value = strings[int(value)]
                if cell.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//m:t", XML_NS))
                cells["".join(c for c in cell.get("r") if c.isalpha())] = value
            if not cells.get("D"):
                continue
            name = cells["D"].strip()
            label = int(float(cells["E"]))
            assert name not in records and cells["C"].strip() == FAULT_NAMES[label]
            records[name] = {"speed":float(cells["A"]), "torque":float(cells["B"]),
                             "fault_label":label, "fault_name":cells["C"].strip()}
        return records


def archive_members(tar, archive):
    """先核查包内路径；本数据包必须是平铺的MAT文件，不能越出缓存目录。"""
    result = subprocess.run([str(tar), "-tf", str(archive)], capture_output=True, check=True, timeout=30)
    names = [line.strip() for line in result.stdout.decode("utf-8").splitlines() if line.strip()]
    assert len(names) == len(set(names))
    assert all(Path(n).name == n and n.lower().endswith(".mat") for n in names)
    return names


def cached_records(tar, archive, destination, names):
    """解压到hust下专用缓存；只向新目录解压，已有缓存完整时直接复用。

    不使用tar向管道输出二进制的方式，防止Windows归档工具的管道异常。
    """
    destination = destination.resolve()
    if destination.exists():
        assert {p.name for p in destination.iterdir()} == set(names), "缓存不完整，须先人工检查"
    else:
        destination.mkdir(parents=True)
        subprocess.run([str(tar), "-xf", str(archive), "-C", str(destination)], check=True, timeout=90)
    assert {p.name for p in destination.iterdir()} == set(names)
    return destination


def read_matrix(path, expected_field):
    """只接受已核实字段和二维有限浮点矩阵，不猜测缺失字段。"""
    value = loadmat(path, variable_names=[expected_field])[expected_field]
    assert value.ndim == 2 and value.dtype == np.float64 and np.isfinite(value).all(), str(path)
    return value


def summary_of_labels(records):
    """汇总原始记录数；它与后续切出的电气周期样本数不同。"""
    return {"records":len(records), "fault_record_counts":dict(Counter(r["fault_label"] for r in records.values())),
            "speeds":sorted({r["speed"] for r in records.values()}),
            "torques":sorted({r["torque"] for r in records.values()}),
            "condition_groups":len({(r["speed"],r["torque"]) for r in records.values()})}


def inspect(args):
    """源/常规实测逐条检查，含噪记录逐条与对应常规记录比较；峰值内存仅几条波形。"""
    repo, work = args.repo.resolve(), args.work.resolve()
    output = work/"analysis/data_audit_v1"
    assert not output.exists(), "审计输出已存在；不覆盖已归档证据"
    output.mkdir(parents=True)
    report = {"scope":"five-phase IM-AA only", "model_training":False,
              "runtime":{"python":sys.version, "numpy":np.__version__, "scipy":scipy.__version__, "platform":platform.platform()},
              "repository":str(repo), "source_git_commit":subprocess.check_output([str(args.git), "-C", str(repo), "rev-parse", "HEAD"]).decode().strip(),
              "snr_levels_db":list(SNR_LEVELS), "label_map":{"0":"Healthy: derived from prefault interval", **{str(k):v for k,v in FAULT_NAMES.items()}}}
    inputs = {}
    all_rows = []
    labels = {}
    for kind in ("simulation", "experimental"):
        archive = repo/f"Five-phase-IM-AA-{kind}-domain-data-set.rar"
        excel = repo/f"Five-phase-IM-AA-{kind}-domain.xlsx"
        for path in (archive, excel):
            inputs[path.name] = {"bytes":path.stat().st_size, "sha256":file_sha(path)}
        records = labels_from_xlsx(excel)
        labels[kind] = records
        names = archive_members(args.tar, archive)
        cache = cached_records(args.tar, archive, work/".cache/raw_records"/kind, names)
        report[kind] = {**summary_of_labels(records), "raw_archive_records":len(names),
                        "unlabelled_archive_members":sorted(set(names)-{name+".mat" for name in records}),
                        "labelled_missing_mat":sorted({name+".mat" for name in records}-set(names)), "cache":str(cache)}
        assert not report[kind]["labelled_missing_mat"]
        duplicate_health = defaultdict(list)
        shape_counts = Counter()
        dt_counts = Counter()
        duplicate_counts = Counter()
        omega_min, omega_max = float("inf"), float("-inf")
        for name, metadata in sorted(records.items()):
            path = cache/(name+".mat")
            value = read_matrix(path, "I_We" if kind == "simulation" else "name")
            assert value.shape[1] == (6 if kind == "simulation" else 7)
            shape_counts[str(value.shape)] += 1
            omega_min = min(omega_min, float(value[:,-1].min()))
            omega_max = max(omega_max, float(value[:,-1].max()))
            if kind == "simulation":
                assert len(value) == 10000
                # 前5000行健康的解释来自作者README；没有伪造仿真时间列。
                currents = value[:,:5]
                healthy = currents[:5000]
                repeated_time = "no_time_column"
            else:
                currents = value[:,1:6]
                time = value[:,0]
                diff = np.diff(time)
                assert np.all(diff >= 0)
                repeated_time = int(np.sum(diff == 0))
                duplicate_counts[repeated_time] += 1
                dt_counts.update({str(float(d)):int(c) for d,c in zip(*np.unique(np.round(diff,9),return_counts=True))})
                healthy = currents[time < .5]
            digest = array_sha(healthy)
            duplicate_health[digest].append(name)
            all_rows.append({"domain":kind,"record_id":name,"snr_db":"clean",**metadata,
                             "rows":len(value),"columns":value.shape[1],"mat_field":"I_We" if kind=="simulation" else "name",
                             "duplicate_time_count":repeated_time,"healthy_segment_sha256":digest,"file_sha256":file_sha(path)})
        report[kind].update({"matrix_shapes":dict(shape_counts), "duplicate_time_count_histogram":dict(duplicate_counts),
                            "time_differences_rounded_9_decimals":dict(dt_counts), "last_column_range":[omega_min,omega_max],
                            "exact_prefault_duplicate_groups":[names for names in duplicate_health.values() if len(names)>1]})
        print(kind, "检查完成", len(records), "条有标签记录", flush=True)
    # 同编号的Clean及七档SNR保持成组；不把3080个派生文件当3080次独立实测。
    noise_root = repo/"Five-phase-IM-AA-experimental-domain-different-SNR-data-set"
    target_ids = set(labels["experimental"])
    for snr in SNR_LEVELS:
        assert {p.stem for p in (noise_root/f"SNR_{snr}").glob("*.mat")} == target_ids
    comparisons = {str(snr):{"records":0,"max_abs_snr_error_db":0.,"min_whole_record_snr_db":float("inf"),
                             "max_whole_record_snr_db":float("-inf")} for snr in SNR_LEVELS}
    for index, (name, metadata) in enumerate(sorted(labels["experimental"].items()), 1):
        clean = read_matrix(work/".cache/raw_records/experimental"/(name+".mat"), "name")
        power = np.mean(clean[:,1:6]**2, axis=0)
        for snr in SNR_LEVELS:
            path = noise_root/f"SNR_{snr}"/(name+".mat")
            noisy = read_matrix(path,"I_noisy")
            assert noisy.shape == clean.shape
            assert np.array_equal(noisy[:,0],clean[:,0]) and np.array_equal(noisy[:,-1],clean[:,-1]), str(path)
            residual = noisy[:,1:6]-clean[:,1:6]
            measured = 10*np.log10(power/np.mean(residual**2,axis=0))
            item = comparisons[str(snr)]
            item["records"] += 1
            item["max_abs_snr_error_db"] = max(item["max_abs_snr_error_db"],float(np.max(np.abs(measured-snr))))
            item["min_whole_record_snr_db"] = min(item["min_whole_record_snr_db"],float(measured.min()))
            item["max_whole_record_snr_db"] = max(item["max_whole_record_snr_db"],float(measured.max()))
            all_rows.append({"domain":"experimental_noisy","record_id":name,"snr_db":snr,**metadata,
                             "rows":len(noisy),"columns":7,"mat_field":"I_noisy",
                             "duplicate_time_count":int(np.sum(np.diff(noisy[:,0])==0)),
                             "healthy_segment_sha256":array_sha(noisy[noisy[:,0]<.5,1:6]),"file_sha256":file_sha(path)})
        if index%100 == 0:
            print("七档SNR配对检查",index,"/",len(target_ids),flush=True)
    assert all(item["max_abs_snr_error_db"] < 1e-9 for item in comparisons.values())
    report["noise_pairs"] = {"base_records":len(target_ids),"noisy_files":len(target_ids)*len(SNR_LEVELS),
                             "all_time_and_last_column_exactly_equal":True,
                             "snr_power_definition":"10log10(mean(clean_phase^2)/mean((noisy_phase-clean_phase)^2)), each phase over full original record",
                             "noise_distribution_and_generation_code":"not supplied; not inferred solely from SNR", "per_snr":comparisons}
    for name, item in inputs.items():
        assert file_sha(repo/name) == item["sha256"], "原始输入发生变化"
    report["input_fingerprints"] = inputs
    report["status"] = "metadata_and_all_3080_noise_pairs_verified; training_protocol_not_frozen"
    (output/"DATA_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (output/"RECORD_MANIFEST.csv").open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (output/"LABELS.json").write_text(json.dumps(labels,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("审计完成：",len(all_rows),"条原始/派生记录元数据；没有训练或使用Test选模。",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--tar",type=Path,default=Path("C:/WINDOWS/system32/tar.exe"))
    parser.add_argument("--git",type=Path,default=Path("D:/APP/git/Git/cmd/git.exe"))
    inspect(parser.parse_args())
