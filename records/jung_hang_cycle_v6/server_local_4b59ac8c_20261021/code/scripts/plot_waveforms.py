"""绘制六类五相电流示例：归一化前电流及本轮128点模型输入。

仅从已冻结的源Train和目标Dev取例，不读Holdout。记录按编号/工况的
确定性规则选择，窗口取该记录对应类别的中间周期，不按模型分数选图。
原MAT和临时数组不提交仓库；PNG/SVG/PDF及来源编号、窗口、哈希可提交。
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

NAMES = {0:"Healthy", 1:"OUS(A)", 2:"OLS(A)", 3:"OP(A)",
         4:"OUS(A)&OUS(B)", 5:"OLS(A)&OP(C)"}
SLUGS = {0:"00_healthy",1:"01_ous_a",2:"02_ols_a",3:"03_op_a",4:"04_ous_a_ous_b",5:"05_ols_a_op_c"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333"]


def read(path):
    """读取实验冻结的窗口与标签元数据。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    """记录绘图所用原文件/张量的SHA256，不修改输入。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurements(path, field, source):
    """返回归一化前电流；实测同时间戳取平均，与实验预处理入口一致。

    仿真无时间列，依冻结配置10kHz重建时间；这里不加噪或修改幅值。
    """
    value=loadmat(path,variable_names=[field])[field]
    assert np.isfinite(value).all()
    if source:
        assert value.shape==(10000,6)
        return np.arange(len(value))/10000,value[:,:5]
    assert value.shape==(10031,7)
    time,start,count=np.unique(value[:,0],return_index=True,return_counts=True)
    current=np.add.reduceat(value[:,1:6],start,axis=0)/count[:,None]
    assert np.all(np.diff(time)>0)
    return time,current


def select_examples(source_rows,target_rows,labels):
    """目标每类取最小编号；源按转速差、负载差、编号顺序对应，取中间周期。"""
    selected=[]
    for label in range(6):
        target_id=min(r["record_id"] for r in target_rows if r["label"]==label)
        target=labels["experimental"][target_id]
        candidates={r["record_id"] for r in source_rows if r["label"]==label}
        source_id=min(candidates,key=lambda n:(abs(labels["simulation"][n]["speed"]-target["speed"]),
                                               abs(labels["simulation"][n]["torque"]-target["torque"]),n))
        indices=[]
        for rows,name in [(source_rows,source_id),(target_rows,target_id)]:
            eligible=[i for i,r in enumerate(rows) if r["record_id"]==name and r["label"]==label]
            indices.append(eligible[len(eligible)//2])
        selected.append({"label":label,"source_record":source_id,"target_record":target_id,
                         "source_window_index":indices[0],"target_window_index":indices[1]})
    return selected


def plot(args):
    """输出12张数据展示图；两种图采用相同记录和相同五相次序。"""
    project=args.project.resolve();result=project/"outputs/gate0_v1"
    source_rows=read(result/"source_train_windows.json");target_rows=read(result/"target_dev_clean_windows.json")
    assert target_rows==read(result/"target_dev_snr10_windows.json")
    labels=read(project/"analysis/data_audit_v1/LABELS.json")
    examples=select_examples(source_rows,target_rows,labels)
    tensors={}
    for key in ["source_train","target_dev_clean","target_dev_snr10"]:
        with np.load(result/(key+".npz")) as f:tensors[key]=f["x"].copy()
    out=project/"figures/waveforms";out.mkdir(parents=True,exist_ok=True)
    matplotlib.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,"axes.titlesize":7,
                                "axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,
                                "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})
    if args.qa_helper_dir:
        sys.path.insert(0,str(args.qa_helper_dir.resolve()))
        from audit_panel_alignment import require_matplotlib_panel_alignment
    manifest=[]
    for example in examples:
        label=example["label"];si,ti=example["source_window_index"],example["target_window_index"]
        windows=[source_rows[si],target_rows[ti],target_rows[ti]]
        source=project/".cache/raw_records/simulation"/(example["source_record"]+".mat")
        clean=project/".cache/raw_records/experimental"/(example["target_record"]+".mat")
        noisy=args.dataset_repo/"Five-phase-IM-AA-experimental-domain-different-SNR-data-set/SNR_10"/(example["target_record"]+".mat")
        paths=[source,clean,noisy]
        samples=[measurements(path,field,i==0) for i,(path,field) in enumerate(zip(paths,["I_We","name","I_noisy"]))]
        data=[]
        for (time,current),window in zip(samples,windows):
            start,end=window["start"],window["end"]
            knots=np.r_[start,time[(time>start)&(time<end)],end]
            value=np.stack([np.interp(knots,time,current[:,phase]) for phase in range(5)])
            data.append(((knots-start)*1000,value))
        processed=[tensors["source_train"][si],tensors["target_dev_clean"][ti],tensors["target_dev_snr10"][ti]]
        metadata={**example,"fault_name":NAMES[label],"source_window":windows[0],"target_window":windows[1],
                  "source_metadata":labels["simulation"][example["source_record"]],
                  "target_metadata":labels["experimental"][example["target_record"]],
                  "raw_input_sha256":dict(zip(["source","target_clean","target_snr10"],map(sha,paths)))}
        titles=["Simulation\n"+example["source_record"],"Measured\n"+example["target_record"],"Measured, 10 dB\n"+example["target_record"]]
        for kind in ("before_normalization","preprocessed_128"):
            width_mm=183
            height_mm=170
            fig,axes=plt.subplots(5,3,figsize=(width_mm/25.4,height_mm/25.4),sharey="row")
            fig.subplots_adjust(left=.115,right=.98,bottom=.105,top=.865,wspace=.25,hspace=.44)
            description="Before normalization" if kind=="before_normalization" else "Preprocessed, 128 points"
            fig.suptitle(f"Label {label}: {NAMES[label]} | {description}",fontsize=9,y=.975)
            for row in range(5):
                if kind=="before_normalization":
                    peak=max(float(np.max(np.abs(v[row]))) for _,v in data);limit=max(.1,peak*1.12)
                else:limit=1.1
                for col in range(3):
                    ax=axes[row,col]
                    x,y=data[col] if kind=="before_normalization" else (np.arange(128),processed[col])
                    ax.plot(x,y[row],color=COLORS[row],linewidth=.65)
                    ax.set_xlim(float(x[0]),float(x[-1]));ax.set_ylim(-limit,limit)
                    ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
                    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
                    ax.tick_params(width=.5,length=2,pad=2)
                    if row==0:ax.set_title(titles[col],pad=20)
                    if col==0:ax.set_ylabel(f"Phase {'ABCDE'[row]}\n"+("Current (A)" if kind=="before_normalization" else "Normalized current"))
                    if row==4:ax.set_xlabel("Time from window start (ms)" if kind=="before_normalization" else "Resampled point index")
                    # 面板标签固定在轴框外，与曲线无重叠；全部15个图区都接受几何检查。
                    ax.annotate(chr(ord('a')+row*3+col),xy=(0,1),xycoords='axes fraction',
                                xytext=(-5,4),textcoords='offset points',fontsize=8,fontweight='bold',
                                va='bottom',ha='right',annotation_clip=False)
            stem=out/(SLUGS[label]+"_"+kind)
            fig.canvas.draw()
            if args.qa_helper_dir:
                require_matplotlib_panel_alignment(fig,json_out=str(stem)+".alignment.json",tolerance_pt=1.5,gutter_tolerance_pt=1.5,
                        require_panel_labels=True,strict=True)
            for suffix in (".png",".svg",".pdf"):
                assert not Path(str(stem)+suffix).exists(),str(stem)
            fig.savefig(str(stem)+".png",dpi=600)
            fig.savefig(str(stem)+".svg")
            fig.savefig(str(stem)+".pdf")
            plt.close(fig)
            manifest.append({"figure_stem":str(stem.relative_to(project)).replace('\\','/'),"kind":kind,
                             "selection":metadata,"phase_order":["A","B","C","D","E"],
                             "columns":["simulation","measured_clean","measured_snr10"],
                             "width_mm":width_mm,"height_mm":height_mm,"png_dpi":600,
                             "alignment_checked":bool(args.qa_helper_dir)})
    report={"date":"2026-10-12","figure_count":12,"category_count":6,
            "selection_rule":"target smallest record ID per class in fixed Dev; source closest speed/torque then ID among fixed Train; middle matching cycle",
            "dataset_scope":"Source Train and Target Dev only; no Holdout records",
            "before_normalization_time_rule":"same-time measurements averaged, matching experiment parser",
            "preprocessed_arrays":"exact existing float32 model inputs, no new normalization",
            "observations_per_domain_per_class":1,"model_results_used_to_select_examples":False,"figures":manifest}
    (out/"FIGURE_SOURCE_MANIFEST.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"figures":12,"exports":36,"alignment_checked":bool(args.qa_helper_dir)},ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--dataset-repo",type=Path,required=True)
    p.add_argument("--qa-helper-dir",type=Path)
    plot(p.parse_args())
