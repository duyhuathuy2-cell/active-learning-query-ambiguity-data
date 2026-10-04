"""Python-only vector figures; each panel is traced to exported source data."""
import environment
from environment import ROOT
import sys,json,string
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from method import interval,entropy
SKILL=ROOT/'code/figure_checks'
if not SKILL.exists(): SKILL=__import__('pathlib').Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
OUT=ROOT.parent/'KBS_submission/figures';OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT.parent/'data/figure_data';DATA.mkdir(exist_ok=True)
R=ROOT.parent/'data/results/benchmark'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
 'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
 'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.6,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
 'savefig.dpi':300,'pgf.texsystem':'xelatex','pgf.rcfonts':False,'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}'})
MODELS=['tabpfn_1','tabpfn_8','tabicl_8','logistic','forest']
LABELS=['TabPFN (1)','TabPFN (8)','TabICLv2 (8)','Logistic','Forest']
COLORS=['#8494AF','#375E8C','#A76F52','#699884','#9A93AB']
TASKS=['breast_cancer','iris_binary','wine_binary','digits_3_8','moons','mechanistic']
NAMES=['Breast cancer','Iris','Wine','Digits 3/8','Moons','Mechanistic']
TASK_MARKERS=['o','s','^','v','D','P']
TASK_SHORT=['BC (public)','Iris (public)','Wine (public)','Digits (public)','Moons (synthetic)','Mechanistic (synthetic)']
POLICIES=['raw','kl','lower','minimax','entropy','random','certify_entropy']
PNAMES=['Raw entropy reduction','KL repair','Lower bound','Minimax','Predictive entropy','Random','Certificate + entropy']
PC=['#375E8C','#A76F52','#A9967D','#699884','#956B91','#A0A0A0','#0E7C86']

def figure(rows=1,cols=2,height=2.6):
    f,aa=plt.subplots(rows,cols,figsize=(7.08661417,height),squeeze=False)
    f.subplots_adjust(left=.105,right=.975,bottom=.21 if rows==1 else .13,top=.87,hspace=.58,wspace=.42)
    for letter,a in zip(string.ascii_lowercase,aa.flat):
        a.text(0,1,letter,transform=a.transAxes+ScaledTranslation(-20/72,12/72,f.dpi_scale_trans),fontsize=8,fontweight='bold',va='bottom')
    return f,aa

def save(f,name,data):
    if '--only' in sys.argv and name!=sys.argv[sys.argv.index('--only')+1]:
        plt.close(f);return
    f.canvas.draw()
    require_matplotlib_panel_alignment(f,json_out=str(OUT/(name+'.alignment.json')),overlay_svg=str(OUT/(name+'.alignment.svg')),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    f.savefig(OUT/(name+'.pdf'))
    data.to_csv(DATA/(name+'.csv'),index=False)
    plt.close(f);print('FIGURE',name,flush=True)

def spread(ax,x,ys,color,label=None):
    ys=np.asarray(ys);avg=ys.mean(0);sd=ys.std(0,ddof=1)
    ax.plot(x,avg,color=color,label=label,lw=1.2);ax.fill_between(x,avg-sd,avg+sd,color=color,alpha=.12,lw=0)

def main():
    # Fig. 1. A constructive counterexample, not fitted data.
    f,aa=figure();a0,b=.5,.5;t=np.linspace(0,.5,401);q=[(.1,.6),(.8,.6)];records=[]
    for k,(q0,q1) in enumerate(q):
        L,U,lo,hi=interval(a0,np.array(b),np.array(q0),np.array(q1))
        cost=abs(t-a0*q1)+abs(t-(b-(1-a0)*q0))
        value=entropy(b)-(1-a0)*entropy((b-t)/(1-a0))-a0*entropy(t/a0)
        aa[0,0].plot(t,cost,color=COLORS[k],label=f'Candidate {k+1}')
        aa[0,0].plot([lo,hi],[abs(b-(1-a0)*q0-a0*q1)]*2,color=COLORS[k],lw=4)
        tt=np.linspace(lo,hi,101);vv=entropy(b)-(1-a0)*entropy((b-tt)/(1-a0))-a0*entropy(tt/a0)
        aa[0,1].plot(tt,vv,color=COLORS[k],label=f'Candidate {k+1}',lw=2)
        for x,c,v in zip(t,cost,value):records.append(dict(candidate=k+1,t=x,cost=c,value=v,lo=float(lo),hi=float(hi)))
    aa[0,0].set(xlabel='Joint probability t',ylabel='Weighted repair cost',title='Equal-cost repair freedom');aa[0,0].legend()
    aa[0,1].set(xlabel='Joint probability t',ylabel='Target information (bits)',title='Overlapping attainable values');aa[0,1].legend()
    save(f,'ambiguity_example',pd.DataFrame(records))

if __name__=='__main__':main()
