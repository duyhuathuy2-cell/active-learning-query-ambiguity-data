"""Redraw all eight KBS figures from archived data at final print dimensions.

PDF/SVG preserve vector text; PNG exports are 600 dpi. No outcome is discarded.
Colors follow scientific-figure-making; every figure records data and units.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from method import entropy, interval

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'data/results'; OUT=ROOT/'KBS_submission/figures'
QA=ROOT/'reproduction/output'; OUT.mkdir(exist_ok=True); QA.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'reproduction/code/figure_checks'))
from audit_panel_alignment import require_matplotlib_panel_alignment
BLUE='#0F4D92'; TEAL='#42949E'; RED='#B64342'; GRAY='#767676'; LIGHT='#CFCECE'; INK='#272727'
MODELS=['tabpfn_1','tabpfn_8','tabicl_8','logistic','forest']
MLAB=['TabPFN (1)','TabPFN (8)','TabICLv2 (8)','Logistic','Forest']
MCOL=[GRAY,BLUE,RED,TEAL,'#9A4D8E']; MARK=['o','s','^','D','v']
TASKS=['breast_cancer','iris_binary','wine_binary','digits_3_8','moons','mechanistic']
TLAB=['Cancer','Iris','Wine','Digits','Moons','Mechanistic']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.8,'lines.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white','axes.unicode_minus':False})
CONTRACTS={}

def title(ax,letter,text):
    ax.set_title(text,loc='left',pad=14)
    ax.text(-.13,1.085,letter,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')

def grid(rows=1,cols=2,height=3.25,bottom=.24):
    fig,axes=plt.subplots(rows,cols,figsize=(7.1,height),squeeze=False)
    fig.subplots_adjust(left=.11,right=.97,bottom=bottom,top=.84,wspace=.42,hspace=.7)
    return fig,axes

def save(fig,name,sources,unit,uncertainty,notes=''):
    fig.canvas.draw()
    alignment=require_matplotlib_panel_alignment(fig,json_out=QA/f'{name}.alignment.json',strict=True,
                                                  require_panel_labels=name!='principle')
    # Check text against the figure boundary. Legends are placed in reserved space.
    renderer=fig.canvas.get_renderer(); clipped=[]; unused_ticks=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-12<=tick.get_loc()<=hi+1e-12:
                    unused_ticks.update([id(tick.label1),id(tick.label2)])
    for txt in fig.findobj(matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text() or id(txt) in unused_ticks: continue
        bb=txt.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
            clipped.append(txt.get_text())
    if clipped: raise RuntimeError(f'{name}: clipped text {clipped}')
    for fmt in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{fmt}',dpi=600)
    CONTRACTS[name]={'sources':sources,'unit':unit,'uncertainty':uncertainty,'notes':notes,
                     'width_inches':7.1,'height_inches':fig.get_figheight(),'dpi_png':600,
                     'exports':['pdf','svg','png'],'excluded_observations':0,'font_pt':9,
                     'alignment':'passed','text_boundary_check':'passed'}
    plt.close(fig); print('Redrawn '+name,flush=True)

def principle():
    fig,ax=plt.subplots(figsize=(7.1,4.13))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,7.1),ylim=(0,4.13));ax.axis('off')
    offset=.62
    border='#CDD8E3';soft='#F1F6FB';muted='#536779'
    def box(x,y,w,h,fill='white',stroke=border,lw=.9):
        ax.add_patch(FancyBboxPatch((x,y+offset),w,h,boxstyle='round,pad=0,rounding_size=.055',
                                  facecolor=fill,edgecolor=stroke,lw=lw,zorder=2))
    def text(x,y,s,size=9,color=INK,weight='normal',ha='center'):
        ax.text(x,y+offset,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',linespacing=1.3,zorder=4)
    def arrow(points,color=muted):
        points=[(x,y+offset) for x,y in points]
        if len(points)>2:
            ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],color=color,lw=1.05,zorder=3)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,
                                    lw=1.05,color=color,shrinkA=0,shrinkB=0,zorder=3))
    # A single reading direction connects the paper's three contributions.
    ax.text(3.55,4.01,'Active-learning query selection',fontsize=11,weight='bold',ha='center',va='center',color=INK)
    headers=['Query predictions','Complete repair family','Query decision audit']
    details=['Labelled context + candidate pool\nCurrent / hypothetical predictions',
             'Fixed current marginals\nEvery minimum-TV repair',
             'Sharp utility bounds\nWinners + worst-case regret']
    for x,num,heading,detail in zip([.12,2.51,4.90],['01','02','03'],headers,details):
        box(x,2.47,2.08,.69,soft,BLUE)
        text(x+.15,2.94,num,8.3,BLUE,'bold','left')
        text(x+1.04,2.94,heading,9.7,BLUE,'bold')
        text(x+1.04,2.66,detail,8.3,muted)
    arrow([(2.20,2.82),(2.51,2.82)],BLUE)
    arrow([(4.59,2.82),(4.90,2.82)],BLUE)
    ax.plot([5.94,5.94],[2.47+offset,2.26+offset],color=muted,lw=1.05,zorder=1)
    arrow([(5.94,2.26),(1.76,2.26),(1.76,2.05)],BLUE)
    arrow([(5.94,2.26),(5.94,2.05)],TEAL)
    # The branches have distinct data requirements and guarantee endpoints.
    box(.12,.12,3.28,1.93,'#F5F8FC','#D8E2ED')
    box(3.65,.12,3.33,1.93,'#F3F9F8','#CFE4E2')
    text(.29,1.84,'Global information decision',10.2,BLUE,'bold','left')
    text(.29,1.61,'Over globally compatible repairs',8.3,muted,ha='left')
    for x,label in [(.27,'Joint\ncompletion'),(1.40,'Complete\npricing'),(2.53,'Verified\nminimax')]:
        box(x,.95,.72,.48,'white','#B9CCDF')
        text(x+.36,1.19,label,8.7,BLUE)
    arrow([(.99,1.19),(1.40,1.19)],BLUE)
    arrow([(2.12,1.19),(2.53,1.19)],BLUE)
    arrow([(2.89,.95),(2.89,.66),(1.76,.66),(1.76,.57)],BLUE)
    box(.36,.26,2.80,.31,BLUE,BLUE)
    text(1.76,.415,'Query + certified regret gap',9,'white','bold')
    text(3.82,1.84,'Calibrated prediction decision',10.2,TEAL,'bold','left')
    text(3.82,1.61,'Fully labelled historical episodes',8.3,muted,ha='left')
    for x,w,label in [(3.82,1.43,'Actual refit losses\n+ outcome-free envelope'),
                      (5.53,1.27,'Simultaneous\nmean-gain bound')]:
        box(x,.95,w,.48,'white','#AFCFCB')
        text(x+w/2,1.19,label,8.3,TEAL)
    arrow([(5.25,1.19),(5.53,1.19)],TEAL)
    ax.plot([6.165,6.165],[.95+offset,.74+offset],color=TEAL,lw=1.05,zorder=3)
    arrow([(6.165,.74),(4.55,.74),(4.55,.57)],TEAL)
    arrow([(6.165,.74),(6.165,.57)],muted)
    box(3.82,.26,1.46,.31,TEAL,TEAL)
    box(5.53,.26,1.27,.31,'white','#B9C7C7')
    text(4.55,.415,'Positive: switch',8.7,'white','bold')
    text(6.165,.415,'Else: baseline',8.7,muted,'bold')
    box(.12,-.48,6.86,.42,soft,border)
    text(3.55,-.27,'Acquire the selected label and update the predictor',9.5,INK,'bold')
    arrow([(1.76,.26),(1.76,-.06)],BLUE)
    arrow([(4.55,.26),(4.55,-.01),(5.45,-.01),(5.45,-.06)],TEAL)
    arrow([(6.165,.26),(6.165,-.01),(5.45,-.01),(5.45,-.06)],muted)
    save(fig,'principle',[],'conceptual article workflow','not a measured figure',
         'One active-learning acquisition: labelled context/candidate pool, predictive interface, alternative decision routes and annotation/refit action; no invented quantitative data.')

def ambiguity():
    fig,axs=grid(height=3.3,bottom=.22); a,b=axs[0]
    t=np.linspace(0,.5,401)
    for k,(q0,q1,color) in enumerate([(.1,.6,BLUE),(.8,.6,RED)],1):
        L,U,lo,hi=interval(.5,np.array(.5),np.array(q0),np.array(q1))
        a.plot(t,abs(t-.5*q1)+abs(t-(.5-.5*q0)),color=color,label=f'Candidate {k}')
        a.plot([lo,hi],[abs(.5-.5*q0-.5*q1)]*2,color=color,lw=4)
        tt=np.linspace(lo,hi,201); vv=1-.5*entropy(1-2*tt)-.5*entropy(2*tt)
        b.plot(tt,vv,color=color,lw=2)
        b.scatter([lo,hi],[vv[0],vv[-1]],s=25,facecolor='white',edgecolor=color,zorder=3)
    a.set(xlabel=r'Joint mass $t=P(A=1,B=1)$',ylabel='Weighted TV repair cost',xlim=(0,.5),ylim=(0,.85))
    b.set(xlabel=r'Joint mass $t=P(A=1,B=1)$',ylabel='Target information (bits)',xlim=(.06,.48),ylim=(-.015,.57))
    b.scatter([.3,.1],[.0290494055,.2780719051],marker='s',s=38,facecolor='white',edgecolor=INK,zorder=5)
    b.scatter([.45,.25],[.5310044064,0],marker='D',s=38,facecolor='white',edgecolor=INK,zorder=5)
    title(a,'a','Equal minimum cost');title(b,'b','Different optimal queries')
    fig.legend(handles=[Line2D([],[],color=BLUE,label='Candidate 1'),Line2D([],[],color=RED,label='Candidate 2'),
                        Line2D([],[],marker='s',color=INK,ls='',mfc='white',label='World 1'),
                        Line2D([],[],marker='D',color=INK,ls='',mfc='white',label='World 2')],loc='lower center',ncol=4)
    save(fig,'ambiguity_example',['data/figure_data/ambiguity_example.csv','data/results/identifiability/two_world_joint_atoms.csv'],
         'analytic candidate-target pair','attainable values, not confidence intervals')

def audit():
    fig,axs=grid(1,3,height=3.85,bottom=.36);a,b,c=axs[0]
    fig.subplots_adjust(left=.15,wspace=.65)
    summary=pd.read_csv(R/'identifiability/all_model_audit_summary.csv').set_index('model').loc[MODELS]
    for shift,col,color,label,hatch in [(-.24,'local_endpoint_disagreements',LIGHT,'Local disagreement',''),
          (0,'strict_global_reversal_witnesses',BLUE,'Global reversal',''),(.24,'strict_certificates',TEAL,'Strict local certificate','//')]:
        a.barh(np.arange(5)+shift,summary[col],height=.21,color=color,label=label,hatch=hatch,edgecolor='white',lw=.4)
    a.set_yticks(range(5),MLAB);a.invert_yaxis();a.set(xlim=(0,120),xlabel='Settings / 120',xticks=[0,40,80,120])
    cases=pd.read_csv(R/'benchmark/audit_cases.csv')
    for model,color,marker,label in zip(MODELS,MCOL,MARK,MLAB):
        g=cases[cases.model==model];x=np.arange(1,9)
        b.plot(x,[(g.possible<=v).mean() for v in x],color=color,marker=marker,ms=3,label=label)
    b.set(xlabel='Possible winners / 8',ylabel='Cumulative fraction',xticks=[1,4,8],ylim=(-.03,1.05))
    imp=pd.read_csv(R/'benchmark/ambiguity_impact.csv');g=imp[(imp.model=='tabpfn_8')&imp.endpoint_flip]
    bound=max(g.raw_internal_regret.max(),g.minimax_internal_regret.max())*1.06
    c.scatter(g.raw_internal_regret,g.minimax_internal_regret,s=15,color=BLUE,alpha=.65,edgecolors='none')
    c.plot([0,bound],[0,bound],color=GRAY,ls='--',lw=.9)
    c.set(xlabel='Raw regret (bits)',ylabel='Minimax regret (bits)',xlim=(-.001,bound),ylim=(-.001,bound))
    for ax,l,txt in [(a,'a','Decision ambiguity'),(b,'b','Decision-set size'),(c,'c','Paired regret')]:title(ax,l,txt)
    fig.legend(*a.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.07,.06),ncol=3,columnspacing=1)
    fig.legend(*b.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.07,0),ncol=5,columnspacing=.8,fontsize=7.6)
    save(fig,'audit_evidence',['data/results/identifiability/all_model_audit_summary.csv','data/results/benchmark/audit_cases.csv','data/results/benchmark/ambiguity_impact.csv'],
         '600 configured interfaces; panel c: 68 paired TabPFN(8) disagreement interfaces','descriptive counts and paired values; overlapping splits')

def global_results():
    fig,axs=grid(1,3,height=3.65,bottom=.31);a,b,c=axs[0];fig.subplots_adjust(wspace=.65)
    sizes=[(8,4),(16,8),(32,8),(64,8),(64,16)];names=['8×4','16×8','32×8','64×8','64×16']
    atom=pd.read_csv(R/'global_and_prediction/exact_global_summary.csv');block=pd.read_csv(R/'global_and_prediction/block_global_summary.csv')
    for src,shift,color,label in [(atom,-.12,GRAY,'Full-atom columns'),(block,.12,BLUE,'Complete block pricing')]:
        for x,(k,m) in enumerate(sizes):
            z=src[(src.candidates==k)&(src.targets==m)&src['case'].str.startswith('new_')]
            for jit,(_,row) in zip([-.04,0,.04],z.iterrows()):
                mk='o' if row.status in ['globally_optimal','numerical_optimum'] else '^'
                a.scatter(x+shift+jit,row.seconds,color=color,marker=mk,s=24,label=label if x==0 and jit==-.04 else None)
    a.set_yscale('log');a.set(xticks=range(5),xticklabels=names,ylabel='Solve time (s)',ylim=(.006,240));a.tick_params(axis='x',rotation=45)
    ver=pd.read_csv(R/'global_and_prediction/verified_linear_summary.csv');z=ver[ver.status=='verified_linear_enclosure']
    b.scatter(range(1,len(z)+1),z.gap,color=TEAL,s=22);b.set_yscale('log');b.axhline(1e-7,color=GRAY,ls='--',lw=.8)
    b.set(xlabel='Signed-linear case',ylabel='Primal-dual gap',ylim=(1e-13,2e-7),xticks=[1,8,15])
    reg=pd.read_csv(R/'global_and_prediction/verified_regret_summary.csv');raw=pd.read_csv(R/'global_and_prediction/global_regret_summary.csv')
    couple=pd.read_csv(R/'global_and_prediction/coupled_regret_summary.csv')
    vals=[]
    for name in reg['case']:
        rr=reg[reg['case']==name].iloc[0];old=raw[raw['case']==name].iloc[0]
        vals.append((rr.regret_lower,rr.solver_upper,old.local_regret))
    vals.append((couple.verified_lower.iloc[0],couple.verified_upper.iloc[0],couple.local_regret.iloc[0]))
    for y,(lo,hi,rect) in enumerate(vals):
        c.plot([lo,hi],[y,y],color=BLUE,lw=3);c.scatter((lo+hi)/2,y,s=22,color=BLUE)
        c.scatter(rect,y,marker='|',s=85,color=RED)
    c.set_yticks(range(6),names+['Coupled']);c.invert_yaxis();c.set(xlabel='Regret (bits)',xlim=(-.025,.94))
    for ax,l,txt in [(a,'a','Complete pricing'),(b,'b','Verified LP bounds'),(c,'c','Global regret bounds')]:title(ax,l,txt)
    fig.legend(*a.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.07,.065),ncol=2)
    fig.text(.08,.025,'Triangles: 120 s cap. Block cap: 180 s. Verification cost is separate.',fontsize=8,color=GRAY)
    c.legend(handles=[Line2D([],[],color=BLUE,marker='o',label='Global enclosure'),Line2D([],[],color=RED,marker='|',ls='',label='Local upper bound')],
             loc='upper left',bbox_to_anchor=(-.20,-.27),fontsize=7.7)
    save(fig,'complete_global',[f'data/results/global_and_prediction/{s}' for s in ['exact_global_summary.csv','block_global_summary.csv','verified_linear_summary.csv','verified_regret_summary.csv','global_regret_summary.csv','coupled_regret_summary.csv']],
         '15 matched linear systems; five fixed-action systems + coupled control','mathematical enclosures; differing resource caps are marked')

def calibration():
    fig,axs=grid(1,3,height=3.75,bottom=.36);a,b,c=axs[0];fig.subplots_adjust(wspace=.62)
    oracle=pd.read_csv(R/'policy_controls/oracle_policy_test.csv');cal=pd.read_csv(R/'policy_controls/learned_oracle_calibration.csv')
    families=['linear','interaction','radial','threshold'];flab=['Linear','Interaction','Radial','Threshold']
    ref=oracle[(oracle.objective=='uniform')&(oracle.policy=='raw')][['family','seed','true_regret']].rename(columns={'true_regret':'ref'})
    for pol,shift,color,label,mk in [('kl',-.24,GRAY,'KL','o'),('minimax',-.08,BLUE,'Minimax','s'),('tv_midpoint',.08,TEAL,'TV midpoint','D'),('endpoint_hi',.24,RED,'TV upper end','^')]:
        g=oracle[(oracle.objective=='uniform')&(oracle.policy==pol)].merge(ref,on=['family','seed'],validate='one_to_one');g['delta']=g.true_regret-g.ref
        stats=g.groupby('family').delta.agg(['mean','std','size']).loc[families]
        a.errorbar(np.arange(4)+shift,stats['mean']*1000,yerr=stats['std']/np.sqrt(stats['size'])*1000,fmt=mk,ms=3.5,color=color,lw=1,capsize=2,label=label)
    a.axhline(0,color=GRAY,lw=.8,ls='--');a.set(ylabel='Regret minus raw (millibits)')
    for p in [a,b,c]:p.set_xticks(range(4),flab,rotation=40,ha='right')
    for method,shift,color,mk,label in [('tv',-.2,BLUE,'s','TV set'),('kl',0,TEAL,'o','KL point'),('raw',.2,GRAY,'^','Raw point')]:
        for i,family in enumerate(families):
            g=cal[(cal.family==family)&(cal.method==method)&cal.calibrated]
            covered=int(g.coverage.sum());n=len(g);assert n==300
            p=covered/n;zz=1.95996398454;den=1+zz*zz/n
            center=(p+zz*zz/(2*n))/den;half=zz*np.sqrt(p*(1-p)/n+zz*zz/(4*n*n))/den
            b.errorbar(i+shift,p,yerr=np.array([[p-(center-half)],[(center+half)-p]]),fmt=mk,color=color,capsize=2,ms=3.5,label=label if i==0 else None)
    for i,family in enumerate(families):
        g=cal[(cal.family==family)&(cal.method=='tv')&~cal.calibrated]
        count=int(g.certified.sum());wrong=int(g.wrong_certificate.sum())
        c.bar(i-.15,count,width=.28,color=LIGHT,edgecolor=GRAY,lw=.6,label='Issued' if i==0 else None)
        c.bar(i+.15,wrong,width=.28,color=RED,label='Wrong' if i==0 else None)
    b.axhline(.95,color=GRAY,ls='--',lw=.8);b.set(ylabel='Simultaneous coverage',ylim=(.84,1.005))
    fig.legend(*b.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.08,.075),ncol=3,fontsize=8)
    c.set(ylabel='Uncalibrated strict certificates');c.legend(loc='upper left',fontsize=8)
    for ax,l,txt in [(a,'a','True information regret'),(b,'b','Oracle calibration'),(c,'c','Issuance diagnostic')]:title(ax,l,txt)
    fig.legend(*a.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.08,0),ncol=4)
    save(fig,'calibration_evidence',['data/results/policy_controls/oracle_policy_test.csv','data/results/policy_controls/learned_oracle_calibration.csv'],
         'four finite-hypothesis families; 300 independent test systems each','a: paired Monte Carlo SE; b: Wilson intervals conditional on a calibration batch',
         'Calibrated rules issue zero strict certificates; oracle targets differ from ordinary labels.')

def prediction():
    fig,axs=grid(1,3,height=3.5,bottom=.27);a,b,c=axs[0];fig.subplots_adjust(wspace=.62)
    df=pd.read_csv(R/'global_and_prediction/final_safe_summary.csv');assert len(df)==12 and df.deploy.sum()==9
    for baseline,shift,color,label,marker in [('entropy',-.15,BLUE,'Entropy baseline','o'),('raw',.15,TEAL,'Raw-information baseline','s')]:
        g=df[df.baseline==baseline].set_index('dataset').loc[TASKS];x=np.arange(6)+shift
        a.errorbar(x,g.test_mean_gain*1000,yerr=g.test_se_gain*1000,fmt=marker,color=color,ms=4,lw=1,capsize=2,label=label)
        b.scatter(x[g.deploy],g.certified_mean_lower_bound[g.deploy]*1000,color=color,marker=marker,s=26)
        b.scatter(x[~g.deploy],np.zeros((~g.deploy).sum()),color=GRAY,marker='x',s=32,lw=1.2)
        c.bar(x,g.issued/g.test_n*100,width=.26,color=color,alpha=.9)
        c.scatter(x,g.worse_queries/g.test_n*100,marker='_',s=55,color=RED,zorder=4)
    for ax in axs[0]:ax.set_xticks(range(6),['Cancer','Iris','Wine','Digits','Moons','Mech.'],rotation=45,ha='right');ax.set_xlim(-.55,5.55)
    a.axhline(0,color=GRAY,ls='--',lw=.8);a.set(ylabel='Test Brier gain × 1,000',ylim=(-1.4,5.5))
    b.set(ylabel='Mean lower bound × 1,000',ylim=(-.12,1.9))
    c.set(ylabel='Test episodes (%)',ylim=(0,60))
    for ax,l,txt in [(a,'a','Independent test'),(b,'b','Mean-gain certificate'),(c,'c','Switching frequency')]:title(ax,l,txt)
    fig.legend(*a.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.07,.01),ncol=2)
    fig.text(.65,.065,'×  Baseline fallback',fontsize=8,color=GRAY)
    fig.text(.65,.02,'Red ticks: harmful switches',fontsize=8,color=RED)
    save(fig,'prediction_safe',['data/results/global_and_prediction/final_safe_summary.csv'],
         '12 task-baseline comparisons; 2,048 test episodes per task','test: one conditional Monte Carlo SE; bound: simultaneous empirical Bernstein, F=72, n=16,384',
         'Fallback has exactly zero deployed gain; it is not a positive certificate. Individual harm remains visible.')

def sensitivity():
    fig,axs=grid(height=3.5,bottom=.29);a,b=axs[0]
    df=pd.read_csv(R/'sensitivity/radius_summary.csv')
    for model,label,color,marker in zip(MODELS,MLAB,MCOL,MARK):
        z=df[df.model==model];a.plot(z.excess_tv,z.strict_count,color=color,marker=marker,ms=4,label=label)
    a.set(xlabel='Excess pairwise TV allowance',ylabel='Strict winners / 120',ylim=(-.6,19),xticks=[0,.01,.025,.05],yticks=[0,5,10,15])
    n=pd.read_csv(R/'sensitivity/calibration_size_summary.csv');b.plot(n.n,n.positive_pairs,color=BLUE,marker='o',ms=5)
    b.set_xscale('log',base=2);b.set_xticks(n.n,[f'{v:,}' for v in n.n],rotation=25,ha='right')
    b.set(xlabel='Calibration episodes per task',ylabel='Positive mean bounds / 12',ylim=(-.6,12),yticks=[0,3,6,9,12])
    title(a,'a','Tolerance and identification');title(b,'b','Historical sample demand')
    fig.legend(*a.get_legend_handles_labels(),loc='lower center',ncol=3,columnspacing=1.5)
    save(fig,'sensitivity',['data/results/sensitivity/radius_summary.csv','data/results/sensitivity/calibration_size_summary.csv'],
         '120 settings per model; 12 calibration task-baseline pairs','descriptive counts; nested sample prefixes are dependent')

def multiclass():
    fig,axs=grid(1,3,height=3.3,bottom=.25);a,b,c=axs[0];fig.subplots_adjust(wspace=.57)
    df=pd.read_csv(R/'multiclass_and_calibration/multiclass_verified_cases.csv')
    for x,(model,color) in enumerate([('tabpfn_8',BLUE),('logistic',TEAL)]):
        g=df[df.model==model]
        for dx,task,marker in [(-.14,'iris','o'),(0,'wine','s'),(.14,'digits_0_1_2','^')]:
            z=g[g.dataset==task];offset=np.linspace(-.035,.035,len(z))
            a.scatter(x+dx+offset,z.mean_width,color=color,marker=marker,s=20,alpha=.85)
            b.scatter(x+dx+offset,z.possible_count,color=color,marker=marker,s=20,alpha=.85)
        c.bar(x-.15,g.certified.sum(),width=.28,color=color)
        c.bar(x+.15,g.raw_excluded.sum(),width=.28,color='white',edgecolor=color,hatch='//',lw=1)
    for ax in axs[0]:ax.set_xticks([0,1],['TabPFN (8)','Logistic']);ax.tick_params(axis='x',labelsize=8)
    a.set(ylabel='Mean information width (bits)');b.set(ylabel='Possible winners / 8',ylim=(0,9),yticks=[1,4,8])
    c.set(ylabel='Settings / 15',ylim=(0,6),yticks=[0,2,4,6])
    for ax,l,txt in [(a,'a','Finite-label repair'),(b,'b','Decision sets'),(c,'c','Verified decisions')]:title(ax,l,txt)
    handles=[Line2D([],[],color=GRAY,marker=mk,ls='',label=label) for mk,label in [('o','Iris'),('s','Wine'),('^','Digits 0/1/2')]]
    handles += [Patch(facecolor=GRAY,label='Strict winner'),Patch(facecolor='white',edgecolor=GRAY,hatch='//',label='Raw excluded')]
    fig.legend(handles=handles,loc='lower center',ncol=5,columnspacing=.9,fontsize=7.9)
    save(fig,'multiclass_decisions',['data/results/multiclass_and_calibration/multiclass_verified_cases.csv','data/results/multiclass_and_calibration/multiclass_verified_candidates.csv'],
         '30 three-class settings; five per dataset/model; 1,920 pairs','exact-rational feasibility and outward information enclosures; points are settings')

if __name__=='__main__':
    for draw in [principle,ambiguity,audit,global_results,calibration,prediction,sensitivity,multiclass]: draw()
    (QA/'publication_figure_contracts.json').write_text(json.dumps(CONTRACTS,indent=2),encoding='utf8')
