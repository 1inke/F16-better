"""Full population, seed-deduplicated native rank utilities; Python only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
H=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(H))
import run
def main():
    run.guard();result=run.read(H/'RESULT.json');assert result['status']=='PASS_DIAGNOSTIC_ONLY_REMOVED_TAIL_UTILITY'
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,'legend.frameon':False})
    tags=('scene_family','target_track','recording_time');groups=('overall','stable','transition','change');colors=('#555555','#6D8FAB','#B08B62','#937CA5');styles=('-','--',':','-.')
    src=[];exports=[]
    for split in ('outer','inner','original_CAL'):
        fig,axs=plt.subplots(4,3,figsize=(7.20472440945,190/25.4),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.025,1,.94),h_pad=3/72,w_pad=2/72)
        for col,tag in enumerate(tags):
            rec=next(x for x in result['records'] if x['grouping']==tag and x['seed']==42 and x['split']==split)
            for group,color,style in zip(groups,colors,styles):
                s=rec['summary'][group];rank=s['rank_utility'];means=np.asarray([rank[str(i)]['mean'] for i in range(1,21)]);source=dict(grouping=tag,split=split,group=group,n=s['n'],tracks=s['tracks'],ranks=list(range(1,21)),mean_utility=means,positive_count=[rank[str(i)]['positive_count'] for i in range(1,21)],native_seed_repeats_deduplicated=True);src.append(source)
                for row,scale in enumerate((100,1000,1000,1)):
                    axs[row,col].plot(np.arange(1,21),means[:,row]*scale,color=color,linestyle=style,label=group,linewidth=1.2)
            for row in range(4):
                ax=axs[row,col];ax.axvspan(16.5,20.5,color='#D8D8D8',alpha=.3);ax.set_xlim(.5,20.5);ax.set_xticks([1,5,10,15,20]);ax.tick_params(labelsize=7)
                if row==0:ax.set_title(tag.replace('_',' ')+'; N='+str(rec['summary']['overall']['n']),fontsize=8)
                if row==3:ax.set_xlabel('Original logit rank (1 = highest)',fontsize=7)
                if col==0:ax.set_ylabel(('Coverage utility (pp)','ADE utility (mm)','FDE utility (mm)','J utility')[row],fontsize=7)
        axs[0,0].legend(fontsize=6.5,loc='upper right');fig.suptitle(split+': native20 leave-one-out utility; descriptive, no independent seed CI',fontsize=8,y=.985)
        (H/'figures').mkdir(parents=True,exist_ok=True)
        svg=H/'figures'/f'rank_utility_{split}.svg'
        pdf=H/'figures'/f'rank_utility_{split}.pdf'
        png=H/'figures'/f'rank_utility_{split}.png'
        tiff=H/'figures'/f'rank_utility_{split}.tiff'
        fig.savefig(svg)
        fig.savefig(pdf)
        fig.savefig(png,dpi=600)
        fig.savefig(tiff,dpi=600)
        exports.extend(map(str,(svg,pdf,png,tiff)))
        plt.close(fig);run.guard()
    run.write('figures/SOURCE_DATA.json',dict(records=src,utility_order=['coverage','ADE_m','FDE_m','J'],all_rows_including_zero_utility=True,no_p_values_or_independent_seed_CI=True))
    run.write('figures/QA.json',dict(status='SOURCE_AND_EXPORT_PASS_VISUAL_REVIEW_PENDING',backend='python',width_mm=183,height_mm=190,all_four_groups_three_splits=True,native_seed_deduplication_verified_by_worker=True,no_data_sampling=True,editable_svg_pdf_text=True,exports={x:run.sha(x) for x in exports},no_submission_claim=True))
if __name__=='__main__':main()
