"""KBS sensitivity boundary figure from complete diagnostic CSVs."""
from pathlib import Path
import sys,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reproduction/code/figure_checks'))
from audit_panel_alignment import require_matplotlib_panel_alignment
R=ROOT/'data/results/sensitivity'
F=ROOT/'KBS_submission/figures';F.mkdir(parents=True,exist_ok=True)
Q=ROOT/'reproduction/output';Q.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
fig,axes=plt.subplots(1,2,figsize=(7.1,3.3))
fig.subplots_adjust(left=.085,right=.99,bottom=.26,top=.82,wspace=.34)
radius=pd.read_csv(R/'radius_summary.csv')
styles=[('tabpfn_1','TabPFN (1)','#8595A1','o'),('tabpfn_8','TabPFN (8)','#285B80','s'),
 ('tabicl_8','TabICLv2','#7E6699','^'),('logistic','Logistic','#288B84','D'),('forest','Forest','#BE854E','v')]
for key,label,color,marker in styles:
    z=radius[radius.model==key]
    axes[0].plot(z.excess_tv,z.strict_count,marker=marker,color=color,ms=3.8,lw=1,label=label)
axes[0].set(xlabel='Excess pairwise TV allowance',ylabel='Strictly identified settings / 120',ylim=(-.7,19))
axes[0].set_xticks([0,.01,.025,.05]);axes[0].set_yticks([0,5,10,15])
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.30,.005),ncol=3,fontsize=8,handlelength=1.4,columnspacing=.9)
sizes=pd.read_csv(R/'calibration_size_summary.csv')
assert (sizes.n>0).all(), 'Logarithmic sample-size axis requires positive n.'
axes[1].plot(sizes.n,sizes.positive_pairs,color='#285B80',marker='o',ms=5,lw=1.3)
axes[1].set_xscale('log',base=2)
axes[1].set_xticks(sizes.n,[f'{n:,}' for n in sizes.n],rotation=25,ha='right',rotation_mode='anchor')
axes[1].set(xlabel='Calibration episodes per task',ylabel='Pairs with a positive mean bound / 12',ylim=(-.7,12))
axes[1].set_yticks([0,3,6,9,12])
for ax,letter,title in zip(axes,['a','b'],['Repair tolerance removes identification','Mean certificates require substantial data']):
    ax.text(-.12,1.12,letter,transform=ax.transAxes,fontsize=11,fontweight='bold')
    ax.set_title(title,fontsize=9,pad=13)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=str(Q/'sensitivity.alignment.json'),
 overlay_svg=str(Q/'sensitivity.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(F/'sensitivity.pdf')
(Q/'sensitivity.provenance.json').write_text(json.dumps({
 'question':'Where do repaired decision certificates lose usefulness?',
 'claim':'Strict identification is sensitive to extra repair tolerance, and mean-loss calibration needs enough historical episodes.',
 'archetype':'quantitative grid, boundary/failure evidence',
 'panel_a':{'source':'radius_summary.csv','rows':30,'unit':'120 overlapping benchmark splits per model','uncertainty':'exact count of configured cases; no sampling CI'},
 'panel_b':{'source':'calibration_size_summary.csv','rows':4,'unit':'12 task-baseline pairs, nested calibration prefixes','uncertainty':'count of positive simultaneous lower bounds; no independence across prefixes'},
 'backend':'Python/matplotlib','exclusions':[],'reuses':'palette only from supplied project; original data outputs unchanged'
},indent=2),encoding='utf8')
print('Exported KBS sensitivity figure; alignment passed.')
