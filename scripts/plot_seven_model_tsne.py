"""七模型同一展示种子的四条件编码特征图，不重新降维或挑选样本。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

def main():
    """七个散点面板加独立图例，导出600dpi PNG和嵌入字体PDF。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library-root',type=Path)
    args = parser.parse_args()
    if args.library_root is not None: sys.path.insert(0,str(args.library_root.resolve()))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    summary = json.loads((args.records/'SUMMARY.json').read_text(encoding='utf-8'))
    protocol = summary['protocol']
    seed = summary['selected_seed']
    names = {'identity':'A′ no filter','lowpass3':'LP3','lowpass4':'LP4','fixed_half':'Fixed α = 0.5',
        'nahsf':'V5 NA-HSF','xu_full':'Xu-Full','resnet1d':'ResNet1D'}
    labels = ['Healthy','OUS(A)','OLS(A)','OP(A)','OUS(A) & OUS(B)','OLS(A) & OP(C)']
    colors = ['#767676','#0F4D92','#B64342','#8BCF8B','#42949E','#9A4D8E']
    markers = ['o','s','^','D','P','v']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7.1,
        'pdf.fonttype':42,'ps.fonttype':42,'text.usetex':False,'legend.frameon':False})
    files,layouts = [],[]
    reference = None
    for condition in protocol['conditions']:
        fig,axes = plt.subplots(2,4,figsize=(180/25.4,98/25.4))
        fig.subplots_adjust(left=.018,right=.982,bottom=.105,top=.82,wspace=.12,hspace=.43)
        for index,(method,ax) in enumerate(zip(protocol['models'],axes.flat)):
            record = next(row for row in summary['results'] if (row['model'],row['condition']) == (method,condition))
            assert record['seed'] == seed
            path = args.records/'coordinates'/f'{method}_seed{seed}_{condition}_encoder.csv'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['coordinate_file_sha256']
            with path.open(encoding='utf-8') as stream: rows = list(csv.DictReader(stream))
            assert len(rows) == 2474 and [int(row['index']) for row in rows] == list(range(2474))
            truth = np.array([int(row['truth']) for row in rows])
            if reference is None: reference = truth
            assert np.array_equal(truth,reference) and len({row['record_id'] for row in rows}) == 88
            coords = np.array([[float(row['tsne_x']),float(row['tsne_y'])] for row in rows])
            assert np.isfinite(coords).all()
            for label in range(6):
                selected = truth == label
                ax.scatter(coords[selected,0],coords[selected,1],s=3.4,alpha=.70,c=colors[label],
                    marker=markers[label],edgecolors='none',linewidths=0,rasterized=False)
            center = (coords.min(0)+coords.max(0))/2
            radius = np.ptp(coords,axis=0).max()*.55
            ax.set_xlim(center[0]-radius,center[0]+radius)
            ax.set_ylim(center[1]-radius,center[1]+radius)
            ax.set_aspect('equal',adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_title(f"({chr(97+index)}) {names[method]} ({record['feature_dim']}D)\n"
                f"F1 {record['classifier_macro_f1_percent']:.1f}%  |  S {record['silhouette_macro']:.2f}",pad=4)
        # 第八面板只用于图例，避免图例压住任何真实样本。
        legend_ax = axes.flat[-1]
        legend_ax.axis('off')
        handles = [Line2D([],[],color=colors[k],marker=markers[k],linestyle='none',markersize=3.4,
            label=labels[k]) for k in range(6)]
        legend = legend_ax.legend(handles=handles,loc='upper left',fontsize=6.6,
            borderaxespad=0,handletextpad=.35,labelspacing=.6)
        note = legend_ax.text(0,-.035,'F1: original classifier\nS: original-space\nmacro silhouette',
            transform=legend_ax.transAxes,fontsize=6.1,ha='left',va='bottom',color='#444444')
        name = 'Clean' if condition == 'clean' else condition[3:]+' dB'
        fig.suptitle(f'Encoder features  |  {name}  |  Selected V5 seed {seed}',y=.975,fontsize=9)
        footer = fig.text(.5,.025,'V5 Dev-selected illustration; all 2,474 cycles, true labels. Separate t-SNE fits.',
            ha='center',va='center',fontsize=6.2,color='#444444')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        rects = []
        for ax in list(axes.flat)[:7]:
            box = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            rects.append({'x_pt':box.x0*72,'y_pt':box.y0*72,'width_pt':box.width*72,'height_pt':box.height*72})
            assert abs(box.width-box.height)*72 < 1.5
        assert np.ptp([row['width_pt'] for row in rects]) < 1.5
        for group in (rects[:4],rects[4:]): assert np.ptp([row['y_pt'] for row in group]) < 1.5
        assert not legend.get_window_extent(renderer).overlaps(note.get_window_extent(renderer))
        assert footer.get_window_extent(renderer).y1 < note.get_window_extent(renderer).y0
        basename = f'encoder_{condition}_seed{seed}'
        for extension in ('png','pdf'):
            path = args.output/f'{basename}.{extension}'
            fig.savefig(path,dpi=600,facecolor='white',transparent=False)
            files.append({'file':path.name,'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        layouts.append({'figure':basename,'panels':7,'plot_rectangles':rects,
            'alignment_tolerance_pt':1.5,'square_and_aligned':True,'legend_note_footer_overlap':False})
        plt.close(fig)
        print('SEVEN_MODEL_FIGURE_EXPORTED',basename,flush=True)
    (args.output/'FIGURE_MANIFEST.json').write_text(json.dumps({'status':'passed','selected_seed':seed,
        'illustration_seed_selected_posthoc':True,'width_mm':180,'height_mm':98,'png_dpi':600,
        'font':'DejaVu Sans','matplotlib_version':matplotlib.__version__,'figures':4,'data_panels':28,
        'labels':labels,'colors':colors,'markers':markers,'files':files,'layout_checks':layouts},
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__ == '__main__': main()
