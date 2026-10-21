"""只适配服务器数据路径的启动包装器，不改V6算法或读取器。"""
from pathlib import Path
import json,runpy,sys
root=Path(__file__).resolve().parent
paths=json.loads((root/'REFERENCE_PATHS.json').read_text())
sys.path.insert(0,str(Path(paths['reference_root'])/'code/scripts'))
import source_cycle_benchmark,source_cycle_split
sys.path.insert(0,str(root/'code/scripts'))
entry=root/'code/scripts/jung_hang_cycle_v6.py'
sys.argv=[str(entry),*sys.argv[1:]]
runpy.run_path(str(entry),run_name='__main__')
