"""Python vector figures for calibrated coverage and expanded validation."""
import environment
from environment import ROOT
import numpy as np,pandas as pd
from matplotlib.lines import Line2D
from ambiguity_figure import figure,save
R=ROOT.parent/'data/results/multiclass_and_calibration'
col={'tv':'#375E8C','kl':'#A76F52','raw':'#699884','joint':'#9A93AB'}
names={'tv':'TV value coverage','kl':'KL value coverage','raw':'Raw value coverage','joint':'Generating-joint containment'}

def main():
    marks=['o','s','^']
    d=pd.concat([pd.read_csv(R/f'multiclass_{m}_cases.csv') for m in ['tabpfn_8','logistic']])
    verified=pd.read_csv(R/'multiclass_verified_cases.csv')
    d=d.merge(verified[['dataset','model','seed','certified','raw_excluded','mean_width','possible_count']],on=['dataset','model','seed'],suffixes=('_floating',''))
    d['raw_possible']=~d.raw_excluded
    f,aa=figure(1,3,3.0)
    f.subplots_adjust(top=.78,left=.09,right=.975,wspace=.58)
    for i,m in enumerate(['tabpfn_8','logistic']):
        color=col['tv' if i==0 else 'raw'];dd=d[d.model==m]
        for z,t in enumerate(['iris','wine','digits_0_1_2']):
            g=dd[dd.dataset==t]
            aa[0,0].scatter(np.full(5,i)+.09*(z-1),g.mean_width,color=color,marker=marks[z],s=14)
            aa[0,1].scatter(np.full(5,i)+.09*(z-1),g.possible_count,color=color,marker=marks[z],s=14)
        aa[0,2].bar(i-.13,dd.certified.mean(),width=.25,color=color,alpha=.95)
        aa[0,2].bar(i+.13,1-dd.raw_possible.mean(),width=.25,color=color,alpha=.35)
    for ax in aa.flat:ax.set(xticks=[0,1],xticklabels=['TabPFN (8)','Logistic'])
    aa[0,0].set(ylabel='Mean interval width (bits)',title='Three-class minimum repairs')
    aa[0,1].set(ylabel='Possible recommendations (of 8)',ylim=(0,9),title='Outer local recommendation sets')
    aa[0,2].set(ylabel='Fraction of 15 contexts',ylim=(0,.55),title='Strict and excluded decisions')
    f.legend(handles=[Line2D([],[],color='.4',marker=marks[z],linestyle='None',label=t,markersize=4) for z,t in enumerate(['Iris','Wine','Digits 0/1/2'])]+[Line2D([],[],color='.3',lw=5,label='Strict certificate'),Line2D([],[],color='.7',lw=5,label='Raw choice excluded')],loc='upper center',ncol=5,bbox_to_anchor=(.53,.99),fontsize=6)
    save(f,'multiclass_decisions',d)

if __name__=='__main__':main()
