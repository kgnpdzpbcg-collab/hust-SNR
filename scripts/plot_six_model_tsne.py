"""读取已计算的t-SNE坐标，统一绘制六模型三种子主图/补充图。

本脚本不重新降维、不选择样本或模型、不读取高维特征/电流。颜色和符号
表示真实类别，F1来自原分类器，S来自原高维特征空间的宏平均轮廓系数。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def main():
    """可显式复用已有绘图库目录，导出600dpi PNG、字体嵌入PDF及版式记录。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library-root",type=Path)
    args=parser.parse_args()
    if args.library_root is not None:sys.path.insert(0,str(args.library_root.resolve()))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if args.output.exists():raise FileExistsError("拒绝覆盖已有图")
    args.output.mkdir(parents=True)
    summary=json.loads((args.records/"SUMMARY.json").read_text(encoding="utf-8"))
    p=summary["protocol"]
    assert p["seeds"]==[42,43,44] and p["conditions"]==["clean","snr10"]
    methods=p["models"]
    names={"identity":"A′ no filter","lowpass3":"LP3","lowpass4":"LP4",
           "fixed_half":"Fixed α = 0.5","nahsf":"V5 NA-HSF","xu_full":"Xu-Full"}
    labels=["Healthy","OUS(A)","OLS(A)","OP(A)","OUS(A) & OUS(B)","OLS(A) & OP(C)"]
    # 六类语义在所有图中固定。不同符号辅助灰度打印识别，绝不按模型更换配色。
    colors=["#767676","#0F4D92","#B64342","#8BCF8B","#42949E","#9A4D8E"]
    markers=["o","s","^","D","P","v"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8,
        "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","text.usetex":False})
    cached={(r["model"],r["seed"],r["condition"],r["layer"]):r for r in summary["results"]}
    outputs,layouts=[],[]
    reference_truth=None
    for layer,dim in p["feature_layers"].items():
        for condition in p["conditions"]:
            for seed in p["seeds"]:
                fig,axes=plt.subplots(2,3,figsize=(180/25.4,132/25.4))
                fig.subplots_adjust(left=.025,right=.975,bottom=.165,top=.87,wspace=.13,hspace=.30)
                for index,(method,ax) in enumerate(zip(methods,axes.flat)):
                    metric=cached[method,seed,condition,layer]
                    name=f"{method}_seed{seed}_{condition}_{layer}.csv"
                    coordinate_path=args.records/"coordinates"/name
                    assert hashlib.sha256(coordinate_path.read_bytes()).hexdigest()==metric["coordinate_file_sha256"]
                    with coordinate_path.open(encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
                    assert len(rows)==2474 and [int(r["index"]) for r in rows]==list(range(2474))
                    coordinates=np.array([[float(r["tsne_x"]),float(r["tsne_y"])] for r in rows])
                    truth=np.array([int(r["truth"]) for r in rows])
                    if reference_truth is None:reference_truth=truth
                    assert np.array_equal(truth,reference_truth)
                    assert np.isfinite(coordinates).all() and len({r["record_id"] for r in rows})==88
                    for label in range(6):
                        selected=truth==label
                        ax.scatter(coordinates[selected,0],coordinates[selected,1],s=4.0,alpha=.70,
                            marker=markers[label],c=colors[label],edgecolors="none",linewidths=0,rasterized=False)
                    # 同一面板x/y使用同一尺度和正方形绘图区，不拉伸点间几何。
                    minimum,maximum=coordinates.min(axis=0),coordinates.max(axis=0)
                    center=(minimum+maximum)/2
                    radius=max(maximum-minimum)*.55
                    ax.set_xlim(center[0]-radius,center[0]+radius)
                    ax.set_ylim(center[1]-radius,center[1]+radius)
                    ax.set_aspect("equal",adjustable="box")
                    ax.set_xticks([]);ax.set_yticks([])
                    for spine in ax.spines.values():spine.set_visible(False)
                    ax.set_title(f"({chr(97+index)}) {names[method]}\n"
                        f"F1 {metric['classifier_macro_f1_percent']:.1f}%  |  S {metric['silhouette_macro']:.2f}",pad=5)
                condition_name="Clean" if condition=="clean" else "10 dB"
                layer_name="Encoder" if layer=="encoder352" else "Classifier hidden"
                fig.suptitle(f"{layer_name} features ({dim}D)  |  {condition_name}  |  Training seed {seed}",
                             x=.5,y=.974,fontsize=10)
                handles=[Line2D([],[],color=colors[k],marker=markers[k],linestyle="none",markersize=4,label=labels[k]) for k in range(6)]
                legend=fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.047),ncol=3,
                                  fontsize=7.5,columnspacing=1.9,handletextpad=.45)
                footer=fig.text(.5,.018,"All 2,474 cycles; true labels. Independent t-SNE fits. S: original-space macro silhouette.",
                               ha="center",va="center",fontsize=6.7,color="#444444")
                fig.canvas.draw()
                renderer=fig.canvas.get_renderer()
                rectangles=[]
                for ax in axes.flat:
                    box=ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
                    rectangles.append({"x_pt":box.x0*72,"y_pt":box.y0*72,"width_pt":box.width*72,"height_pt":box.height*72})
                assert np.ptp([r["width_pt"] for r in rectangles])<1.5 and np.ptp([r["height_pt"] for r in rectangles])<1.5
                for row in (rectangles[:3],rectangles[3:]):assert np.ptp([r["y_pt"] for r in row])<1.5
                legend_box=legend.get_window_extent(renderer)
                footer_box=footer.get_window_extent(renderer)
                assert footer_box.y1<legend_box.y0
                bottom_plots=min(ax.get_window_extent(renderer).y0 for ax in axes.flat)
                assert legend_box.y1<bottom_plots
                basename=f"{layer}_{condition}_seed{seed}"
                for extension in ("png","pdf"):
                    path=args.output/(basename+"."+extension)
                    fig.savefig(path,dpi=600,facecolor="white",transparent=False)
                    outputs.append({"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size})
                layouts.append({"figure":basename,"plot_rectangles":rectangles,"alignment_tolerance_pt":1.5,
                                "all_widths_heights_row_edges_passed":True,"legend_footer_overlap":False})
                plt.close(fig)
                print("TSNE_FIGURE_EXPORTED",basename,flush=True)
    (args.output/"FIGURE_MANIFEST.json").write_text(json.dumps({"status":"passed","source_records":str(args.records),
        "source_summary_sha256":hashlib.sha256((args.records/"SUMMARY.json").read_bytes()).hexdigest(),
        "matplotlib_version":matplotlib.__version__,"font":"DejaVu Sans","width_mm":180,"height_mm":132,
        "png_dpi":600,"labels":labels,"colors":colors,"markers":markers,"files":outputs,
        "layout_checks":layouts,"data_panels":72,"figures":12},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
