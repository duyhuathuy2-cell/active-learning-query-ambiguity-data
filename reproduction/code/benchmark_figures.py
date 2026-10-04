"""Manuscript evidence figures.

All configured observations are retained. PDF text remains editable. Error definitions differ only where the experimental unit differs.
"""
from pathlib import Path
import json,sys,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'data/results';F=R/'policy_controls'
OUT=ROOT/'KBS_submission/figures';QA=ROOT/'reproduction/output';DATA=ROOT/'data/figure_data/policy_controls'
DATA.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'reproduction/code/figure_checks'))
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.2,
 'axes.labelsize':7.2,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,
 'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
MODELS=['tabpfn_1','tabpfn_8','tabicl_8','logistic','forest']
LABELS=['TabPFN (1)','TabPFN (8)','TabICLv2','Logistic','Forest']
COLORS=['#8B99B2','#285B80','#7B6592','#288B84','#BD8B53']
POLICIES=['kl','js','lower','minimax','tv_midpoint','endpoint_lo','endpoint_hi','entropy','certify_entropy']
PLABEL=['KL','Branch JS','Max. lower','Minimax','TV midpoint','TV low endpoint','TV high endpoint','Entropy','Cert. + entropy']
TASKS=['breast_cancer','iris_binary','wine_binary','digits_3_8','moons','mechanistic']
TLABEL=['Cancer','Iris','Wine','Digits','Moons','Mechan.']
FAMILIES=['linear','interaction','radial','threshold']
CONTRACTS={}

def panel(ax,letter,title):
 ax.set_title(title,loc='left',pad=10,fontweight='normal')
 ax.text(0,1,letter,transform=ax.transAxes+ScaledTranslation(-15/72,12/72,ax.figure.dpi_scale_trans),
  ha='left',va='bottom',weight='bold',fontsize=8)

def setup(rows,cols,h):
 fig,axes=plt.subplots(rows,cols,figsize=(6.75,h),squeeze=False)
 fig.subplots_adjust(left=.12,right=.965,bottom=.33 if rows==1 else .17,top=.81 if rows==1 else .9,
  wspace=.62 if cols==3 else .48,hspace=.66)
 return fig,axes

def save(fig,name,claim,panels,sources):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=QA/f'{name}.alignment.json',strict=True,require_panel_labels=True)
 fig.savefig(OUT/f'{name}.pdf')
 CONTRACTS[name]={'claim':claim,'archetype':'quantitative grid','backend':'python','width_inches':6.75,
  'panels':panels,'source_data':sources,'all_configured_rows_retained':True,'exports':['PDF'],
  'policy':'Nature-skills typography, traceability and QA; Manuscript figure width'}
 plt.close(fig)

def audit():
 fig,axes=setup(1,3,2.7);a,b,c=axes[0]
 summary=pd.read_csv(R/'identifiability/all_model_audit_summary.csv')
 summary=summary.set_index('model').loc[MODELS];pos=np.arange(5)
 for shift,col,color,label in [(-.23,'local_endpoint_disagreements','#B6C2CE','Local disagreement'),
  (0,'strict_global_reversal_witnesses','#285B80','Global reversal'),(.23,'strict_certificates','#288B84','Strict local cert.')]:
  a.barh(pos+shift,summary[col],height=.20,color=color,label=label)
 a.set_yticks(pos,LABELS);a.invert_yaxis();a.set_xlim(0,125);a.set_xlabel('Settings (120 per model)')
 a.legend(loc='upper left',bbox_to_anchor=(-.02,-.28),frameon=False,fontsize=6.1,handlelength=1)
 panel(a,'a','Local versus global evidence')
 cases=pd.read_csv(R/'benchmark/audit_cases.csv')
 for model,color,label in zip(MODELS,COLORS,LABELS):
  g=cases[cases.model==model];x=np.arange(1,9);b.plot(x,[np.mean(g.possible<=v) for v in x],color=color,lw=1.1,marker='o',ms=2.8,label=label)
 b.set(xlabel='Possible-winner count',ylabel='Fraction of 120 settings',xticks=[1,2,4,6,8],ylim=(-.03,1.05));panel(b,'b','How many actions remain?')
 b.legend(loc='upper left',bbox_to_anchor=(0,-.28),frameon=False,fontsize=6.1,ncol=2,handlelength=1,columnspacing=.7)
 impact=pd.read_csv(R/'benchmark/ambiguity_impact.csv');g=impact[(impact.model=='tabpfn_8')&impact.endpoint_flip]
 c.scatter(g.raw_internal_regret,g.minimax_internal_regret,s=11,color='#285B80',alpha=.55,edgecolors='none')
 bound=max(g.raw_internal_regret.max(),g.minimax_internal_regret.max())*1.06;c.plot([0,bound],[0,bound],ls='--',lw=.65,color='#91A0AE')
 c.set(xlabel='Raw local regret (bits)',ylabel='Minimax local regret (bits)',xlim=(-.001,bound),ylim=(-.001,bound));panel(c,'c','Robustness within the set')
 summary.to_csv(DATA/'audit_model_counts.csv');g.to_csv(DATA/'audit_paired_regret.csv',index=False)
 cases.to_csv(DATA/'audit_all_contexts.csv',index=False)
 save(fig,'audit_evidence','Local ambiguity persists and global witnesses establish a subset of reversals.',
  {'a':'exact counts in all 600 settings; no sampling CI','b':'empirical CDF of all 600 contexts; no independent-domain inference',
   'c':'68 paired TabPFN(8) disagreement contexts; every point; diagonal is equal internal regret'},
  ['audit_model_counts.csv','audit_all_contexts.csv','audit_paired_regret.csv'])


def wilson(count,n):
 z=1.95996398454;p=count/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
 half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return center-half,center+half

def calibration():
 fig,axes=setup(1,3,2.95);a,b,c=axes[0];oracle=pd.read_csv(F/'oracle_policy_test.csv');cal=pd.read_csv(F/'learned_oracle_calibration.csv')
 for policy,shift,color,label in [('kl',-.24,'#7B6592','KL'),('minimax',-.08,'#285B80','Minimax'),('tv_midpoint',.08,'#288B84','TV midpoint'),('endpoint_hi',.24,'#BD8B53','TV high endpoint')]:
  baseline=oracle[(oracle.objective=='uniform')&(oracle.policy=='raw')][['family','seed','true_regret']].rename(columns={'true_regret':'ref'})
  g=oracle[(oracle.objective=='uniform')&(oracle.policy==policy)].merge(baseline,on=['family','seed'],validate='one_to_one');g['delta']=g.true_regret-g.ref
  means=[];ses=[]
  for family in FAMILIES:
   vals=g[g.family==family].delta;assert len(vals)==300;means.append(vals.mean()*1000);ses.append(vals.std(ddof=1)/np.sqrt(300)*1000)
  a.errorbar(np.arange(4)+shift,means,yerr=ses,fmt='o',color=color,ms=3,capsize=1.5,lw=.8,label=label)
 flabel=['Linear','Interact.','Radial','Thresh.']
 a.axhline(0,color='#91A0AE',ls='--',lw=.6);a.set_xticks(range(4),flabel,rotation=60,ha='right',rotation_mode='anchor')
 a.set_ylabel('Regret delta versus raw (millibits)');panel(a,'a','Exact-oracle paired evaluation')
 a.legend(loc='upper left',bbox_to_anchor=(0,-.31),frameon=False,fontsize=6.1,ncol=2,handlelength=1,columnspacing=.7)
 for method,shift,color,label in [('tv',-.2,'#285B80','TV set'),('kl',0,'#7B6592','KL point'),('raw',.2,'#999999','Raw point')]:
  means=[];errs=[]
  for family in FAMILIES:
   g=cal[(cal.family==family)&(cal.method==method)&cal.calibrated];count=int(g.coverage.sum());n=len(g);lo,hi=wilson(count,n)
   means.append(count/n);errs.append([count/n-lo,hi-count/n])
  b.errorbar(np.arange(4)+shift,means,yerr=np.asarray(errs).T,fmt='o',ms=3,capsize=1.5,lw=.8,color=color,label=label)
 b.axhline(.95,color='#91A0AE',ls='--',lw=.6);b.set_ylim(.84,1.005);b.set_xticks(range(4),flabel,rotation=60,ha='right',rotation_mode='anchor')
 b.set_ylabel('Simultaneous coverage');panel(b,'b','Calibration of learned predictions')
 b.legend(loc='upper left',bbox_to_anchor=(0,-.31),frameon=False,fontsize=6.1,ncol=2,handlelength=1,columnspacing=.7)
 for i,family in enumerate(FAMILIES):
  g=cal[(cal.family==family)&(cal.method=='tv')&~cal.calibrated];n=len(g);issued=int(g.certified.sum());wrong=int(g.wrong_certificate.sum())
  c.bar(i,issued/n,width=.6,color='#B6C2CE');c.bar(i,wrong/n,width=.6,color='#BD8B53')
  c.text(i,issued/n+.016,f'{wrong}/{issued}',ha='center',fontsize=6.5)
 c.set_xticks(range(4),flabel,rotation=60,ha='right',rotation_mode='anchor');c.set_ylim(0,.31)
 c.set_ylabel('Uncalibrated TV certificate rate');panel(c,'c','Wrong / issued certificates')
 c.text(.5,-.5,'Calibrated TV: 0 / 1,200 issued',ha='center',transform=c.transAxes,fontsize=6.2)
 oracle.to_csv(DATA/'oracle_policy_full.csv',index=False);cal.to_csv(DATA/'learned_calibration_full.csv',index=False)
 save(fig,'calibration_evidence','Oracle calibration curbs unsupported recommendations, while learned-model certificate precision remains limited.',
  {'a':'300 independent held-out systems per family; paired mean +/- one MC standard error; all policy results archived; exact EPIG control has zero oracle regret',
   'b':'300 systems after one separate 100-system calibration batch; Wilson 95% test Monte Carlo intervals conditional on that batch; no conditional conformal guarantee',
   'c':'exact wrong/issued counts, no sampling interval; grey = all issued, gold = wrong; zero calibrated issuance is not a conditional accuracy estimate'},
  ['oracle_policy_full.csv','learned_calibration_full.csv'])


if __name__=='__main__':
 audit();calibration()
