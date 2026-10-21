"""绘制M1错误周期的配对波形：原电流与归一化后模型输入。

示例由诊断脚本按预声明的混淆对和种子一致性选择；只展示A/B两相，
它们对应本次单管/双管故障的比较。全量统计使用全部五相、2474周期。
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main(args):
    """输出四张配对示例图及来源清单；调用外部QA辅助检查最终图区。"""
    source=args.input.resolve();out=args.output.resolve()
    assert not out.exists(), "不能覆盖已生成图像"
    out.mkdir(parents=True)
    sys.path.insert(0,str(args.qa_helper_dir.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    selected=json.loads((source/'EXAMPLES.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'legend.fontsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    manifest=[]
    for example in selected['examples']:
        y,p=example['truth'],example['noisy_example_prediction']
        with np.load(source/f'example_{y}_to_{p}.npz') as f:data={k:f[k].copy() for k in f.files}
        width_mm=183
        height_mm=125
        fig,axes=plt.subplots(2,2,figsize=(width_mm/25.4,height_mm/25.4),sharey='row')
        fig.subplots_adjust(left=.10,right=.975,bottom=.18,top=.82,wspace=.28,hspace=.60)
        predicted=','.join(str(example['seed_predictions'][str(s)]['NN']) for s in [42,43,44])
        fig.suptitle(f"Record {example['record_id']} | True {y} | 10 dB predictions [{predicted}]",fontsize=9,y=.975)
        fig.text(.5,.935,f"Paired window {example['index']} | Speed label {example['source_metadata']['speed']:g}, load label {example['source_metadata']['torque']:g}",ha='center',fontsize=7)
        peak=max(np.max(np.abs(data['raw_clean'][:2])),np.max(np.abs(data['raw_noisy'][:2])))
        handles=None
        for col in range(2):
            for row in range(2):
                ax=axes[row,col]
                if row==0:
                    x=data['time_ms'];clean=data['raw_clean'][col];noisy=data['raw_noisy'][col]
                    ax.set_ylim(-peak*1.1,peak*1.1)
                    ax.set_xlabel('Time in paired window (ms)')
                    ax.set_title(f"Phase {'AB'[col]}: before normalization",pad=5)
                else:
                    x=np.arange(128);clean=data['clean'][col];noisy=data['noisy'][col]
                    assert max(np.max(np.abs(clean)),np.max(np.abs(noisy)),np.max(np.abs(data['clean_with_noise_parameters'][col])))<=1.1
                    ax.set_ylim(-1.1,1.1)
                    ax.set_xlabel('Resampled point index')
                    ax.set_title(f"Phase {'AB'[col]}: model input",pad=5)
                a,=ax.plot(x,clean,color='#0072B2',linewidth=.9,label='Clean')
                b,=ax.plot(x,noisy,color='#D55E00',linewidth=.6,alpha=.85,label='10 dB')
                if row==1:
                    c,=ax.plot(x,data['clean_with_noise_parameters'][col],color='#009E73',linewidth=.9,
                              linestyle='--',label='Clean + noisy normalization')
                    handles=[a,b,c]
                ax.set_xlim(float(x[0]),float(x[-1]))
                ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
                ax.tick_params(width=.5,length=2,pad=2)
                if col==0:ax.set_ylabel('Current (A)' if row==0 else 'Normalized current')
                ax.annotate(chr(ord('a')+row*2+col),xy=(0,1),xycoords='axes fraction',
                    xytext=(-6,5),textcoords='offset points',fontsize=8,fontweight='bold',
                    ha='right',va='bottom',annotation_clip=False)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,frameon=False)
        stem=out/f'case_{y}_to_{p}'
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',
             tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
        fig.savefig(str(stem)+'.png',dpi=600)
        fig.savefig(str(stem)+'.svg')
        fig.savefig(str(stem)+'.pdf')
        plt.close(fig)
        manifest.append({'stem':stem.name,'example':example,'phases_shown':['A','B'],
            'phase_scope_reason':'A single-switch/phase faults and A+B compound-fault comparison; all five phases included in numerical diagnostics',
            'raw_columns':'Clean and 10dB before centering/scaling; same averaged time axis and period',
            'normalized_columns':'original CC and NN float32 inputs; diagnostic CN green dashed',
            'width_mm':width_mm,'height_mm':height_mm,'png_dpi':600,'illustrative_only':True})
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figures':len(manifest),'exports':len(manifest)*3,'alignment_checked':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qa-helper-dir',type=Path,required=True)
    main(parser.parse_args())
