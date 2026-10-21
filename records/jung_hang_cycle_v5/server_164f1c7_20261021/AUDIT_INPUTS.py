"""只读检查V5 Hang源训练图像的跨类重复，不训练或选择模型参数。"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse
import hashlib
import json
import os
import sys

RUN=Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v5/164f1c7_20261021')


def main():
    """统计同一二值轨迹被赋不同源标签的情况，并固定抽取每类首个样本。"""
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    m=json.loads((RUN/'DEPLOYMENT_MANIFEST.json').read_text())
    args=argparse.Namespace(config=RUN/'code/configs/jung_hang_v5_v1.json',reference_root=Path(m['reference_root']),
         source_root=Path(m['source_root']),additional_root=Path(m['additional_root']),previous_output=Path(m['previous_output']),output=RUN/'results')
    sys.path.insert(0,str(RUN/'code/scripts'))
    import numpy as np
    import jung_hang_cycle_v5 as runner
    config,ref,paper,prov,raw,meta,y,clean,vy,val=runner.source_data(args,'hang')
    buckets=defaultdict(list)
    for i,image in enumerate(clean):
        buckets[hashlib.sha256(image.tobytes()).hexdigest()].append(i)
    conflicts={h:ids for h,ids in buckets.items() if len({int(y[i]) for i in ids})>1}
    selected=[int(np.flatnonzero(y==label)[0]) for label in range(6)]
    # 使用上一轮已存在的标准库PNG写入函数，仅用于辅助图，不修改模型或图像输入。
    import importlib.util
    previous=Path(m['previous_v4_root'])/'PREFLIGHT.py'
    spec=importlib.util.spec_from_file_location('png_only_helper',previous)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    canvas=np.full((368,1072),255,dtype=np.uint8)
    for label,i in enumerate(selected):
        for row,image in enumerate((clean[i,0],clean[y==label,0].mean(axis=0))):
            tile=np.repeat(np.repeat((255*(1-image[::-1])).astype(np.uint8),4,axis=0),4,axis=1)
            top,left=16+row*176,16+label*176
            canvas[top-1:top+161,left-1:left+161]=0
            canvas[top:top+160,left:left+160]=tile
    helper.write_gray_png(RUN/'SOURCE_HANG_INPUTS.png',canvas)
    report={'source_train_samples':len(y),'all_unique_input_hashes':len(buckets),'cross_label_identical_input_hashes':len(conflicts),
        'samples_in_cross_label_identical_input_groups':sum(map(len,conflicts.values())),
        'affected_sample_counts_by_class':dict(Counter(str(int(y[i])) for ids in conflicts.values() for i in ids)),
        'cross_label_groups':[{'sha256':h,'sample_indices':ids,'labels':[int(y[i]) for i in ids]} for h,ids in conflicts.items()],
        'plot_layout':'columns0..5 are labels0..5; first row first training sample per class; second row class mean image',
        'fixed_examples':[{'label':label,'index':i,'metadata':meta[i]} for label,i in enumerate(selected)],
        'no_target_waveforms_read':True,'algorithm_or_configuration_modified':False}
    (RUN/'SOURCE_HANG_INPUT_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cross_label_groups','fixed_examples')},ensure_ascii=False))


if __name__=='__main__':
    main()
