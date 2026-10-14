"""修正核验记录相对文件引用并更新导出SHA，保留首份归档及所有数值。"""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parent
def sha(path):
    """记录修正前后真实字节，模型与预测文件完全不动。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

path = root/'results/reports/VERIFICATION.json'
before = json.loads(path.read_text())
assert before['status'] == 'passed' and before['target_prediction_sets_reproduced'] == 270
assert before['verification_settings_correction_record'] == '../../../REPLAY_SETTINGS_DIAGNOSIS.json'
before_sha = sha(path)
after = dict(before,verification_settings_correction_record='../../REPLAY_SETTINGS_DIAGNOSIS.json')
assert (path.parent/after['verification_settings_correction_record']).resolve().is_file()
path.write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n')
note = {'changed_field':'verification_settings_correction_record','before':before['verification_settings_correction_record'],
    'after':after['verification_settings_correction_record'],'before_sha256':before_sha,'after_sha256':sha(path),
    'numerical_results_or_models_modified':False,'original_archive_sha256':sha(root/'architecture_suite_records.tar.gz')}
(root/'EXPORT_REFERENCE_CORRECTION.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n')
files = [p for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.json','.log') and p.name != 'RESULTS_EXPORT_MANIFEST.json']
files += [p for p in (root/'results').rglob('*') if p.is_file() and p.suffix in ('.json','.csv')]
files += [p for p in (root/'logs').rglob('*') if p.is_file()]
files += [root/'code/configs/hust_arch_suite_v1.json']
manifest = root/'RESULTS_EXPORT_MANIFEST.json'
manifest.write_text(json.dumps({'status':'passed','files':{p.relative_to(root).as_posix():
    {'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}},ensure_ascii=False,indent=2)+'\n')
with tarfile.open(root/'architecture_suite_records_final.tar.gz','w:gz') as archive:
    for p in files+[manifest]: archive.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print('EXPORT_REFERENCES_AND_SHA_PASSED',len(files))
