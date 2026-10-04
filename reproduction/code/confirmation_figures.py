"""Traceable prediction and complete-global figures using nature-figure rules."""
import environment
from pathlib import Path
import numpy as np,pandas as pd,json,shutil
import benchmark_figures as style
import matplotlib.pyplot as plt
R=style.R/'global_and_prediction';D=style.ROOT/'data/figure_data/global_and_prediction';D.mkdir(exist_ok=True)
TASKS=style.TASKS;LAB=['Cancer','Iris','Wine','Digits','Moons','Mech.']

def prediction():
 s=pd.read_csv(R/'final_safe_summary.csv');assert len(s)==12 and s.deploy.sum()==9
 fig,axes=style.setup(1,3,2.8);a,b,c=axes[0]
 for baseline,shift,color,label in [('entropy',-.14,'#285B80','Entropy'),('raw',.14,'#288B84','Raw information')]:
  z=s[s.baseline==baseline].set_index('dataset').loc[TASKS];pos=np.arange(6)+shift
  a.errorbar(pos,z.test_mean_gain*1000,yerr=z.test_se_gain*1000,fmt='o',ms=3,color=color,capsize=2,lw=.8,label=label)
  b.scatter(pos,np.maximum(z.certified_mean_lower_bound,0)*1000,s=19,c=color,marker='o')
  # Negative challenger bounds correspond to exact baseline fallback, not zero certificates.
  for p,dep in zip(pos,z.deploy):
   if not dep:b.scatter(p,0,s=20,marker='x',color='#8E9AA6',lw=.8)
  c.bar(pos,z.issued/2048*100,width=.24,color=color,alpha=.85)
  c.scatter(pos,z.worse_queries/2048*100,s=13,marker='_',color='#BE854E',zorder=5)
 for ax in [a,b,c]:ax.set_xticks(range(6),LAB,rotation=60,ha='right',rotation_mode='anchor');ax.set_xlim(-.5,5.5)
 a.axhline(0,color='#A5AFB8',lw=.6,ls='--');a.set_ylabel('Test mean Brier gain 脳 1,000');a.set_ylim(-1.5,5.4)
 a.legend(loc='upper left',bbox_to_anchor=(-.04,-.36),frameon=False,fontsize=6.2,handlelength=1)
 b.set_ylabel('Certified mean gain 脳 1,000');b.set_ylim(-.1,1.9)
 c.set_ylabel('Fraction of test episodes (%)');c.set_ylim(0,60)
 c.text(0,-.47,'Gold ticks: worse individual queries',transform=c.transAxes,fontsize=6.1)
 style.panel(a,'a','Independent final test')
 style.panel(b,'b','Simultaneous mean certificate')
 style.panel(c,'c','Switching and individual harm')
 s.to_csv(D/'final_safe_summary.csv',index=False)
 style.save(fig,'prediction_safe','A simultaneously certified noninferior mean-loss policy improves nine task-baseline pairs and retains three baselines.',
 {'a':'all 12 comparisons; means +/- one Monte Carlo SE over 2048 independent episodes conditional on each fixed table',
  'b':'positive simultaneous empirical-Bernstein lower bounds, family 72, alpha=.05, calibration n=16384; crosses show exact fallback, not certificates',
  'c':'switching and harmful-query fractions among all 2048 episodes; mean noninferiority is not individual-query safety'},['final_safe_summary.csv'])

def global_results():
 atom=pd.read_csv(R/'exact_global_summary.csv');block=pd.read_csv(R/'block_global_summary.csv');ver=pd.read_csv(R/'verified_linear_summary.csv')
 reg=pd.read_csv(R/'verified_regret_summary.csv');raw=pd.read_csv(R/'global_regret_summary.csv')
 coupled=pd.read_csv(R/'coupled_regret_summary.csv')
 fig,axes=style.setup(1,3,3.4);fig.subplots_adjust(bottom=.36);a,b,c=axes[0]
 names=['8脳4','16脳8','32脳8','64脳8','64脳16'];sizes=[(8,4),(16,8),(32,8),(64,8),(64,16)]
 for source,shift,color,label in [(atom,-.13,'#9A89AA','Full-atom columns'),(block,.13,'#285B80','Complete block pricing')]:
  for pos,(k,m) in enumerate(sizes):
   z=source[(source.candidates==k)&(source.targets==m)&source['case'].str.startswith('new_')]
   for jitter,(_,r) in zip([-.045,0,.045],z.iterrows()):
    marker='o' if r.status in ['globally_optimal','numerical_optimum'] else '^'
    a.scatter(pos+shift+jitter,r.seconds,s=15,color=color,marker=marker,label=label if pos==0 and jitter==-.045 else None)
 a.set_yscale('log');a.set_xticks(range(5),names,rotation=60,ha='right',rotation_mode='anchor');a.set_ylabel('Wall time (seconds)')
 a.tick_params(axis='y',labelsize=7.2)
 a.set_ylim(.006,220);a.legend(loc='upper left',bbox_to_anchor=(-.08,-.35),frameon=False,fontsize=6.4,handlelength=1)
 a.text(0,-.62,'Triangles: 120 s limit, incomplete',transform=a.transAxes,fontsize=6.4)
 z=ver[ver.status=='verified_linear_enclosure'];b.scatter(range(len(z)),z.gap,s=14,color='#288B84');b.set_yscale('log');b.axhline(1e-7,ls='--',lw=.6,color='#9CA9B4')
 b.tick_params(axis='y',labelsize=7.2)
 b.set(xlabel='Every signed-linear case',ylabel='Verified primal鈥揹ual gap',ylim=(1e-13,2e-7),xticks=[0,7,14]);b.text(0,-.43,'Exact rational feasibility +\nerror-aware complete dual bound',transform=b.transAxes,fontsize=6.1)
 values=[]
 for name in reg['case']:
  row=reg[reg['case']==name].iloc[0];rr=raw[raw['case']==name].iloc[0];values.append((row.regret_lower,row.solver_upper,rr.local_regret))
 values.append((coupled.verified_lower.iloc[0],coupled.verified_upper.iloc[0],coupled.local_regret.iloc[0]))
 for pos,(lo,hi,local) in enumerate(values):
  c.plot([lo,hi],[pos,pos],lw=2.5,color='#285B80');c.scatter(lo,pos,s=10,color='#285B80');c.scatter(local,pos,s=17,marker='|',color='#BE854E')
  c.text(.32,pos,f'gap {hi-lo:.1e}',fontsize=6.1,va='center',color='#526272')
 c.set_yticks(range(6),names+['Coupled 2脳2']);c.invert_yaxis();c.set_xlabel('Worst information regret (bits)')
 c.text(0,-.43,'Blue: verified global enclosure\nGold: rectangular upper bound',transform=c.transAxes,fontsize=6.1)
 for ax,l,t in zip([a,b,c],['a','b','c'],['Complete-polytope runtime','Verified linear optimization','Nonlinear global regret']):style.panel(ax,l,t)
 for file in ['exact_global_summary.csv','block_global_summary.csv','verified_linear_summary.csv','verified_regret_summary.csv','global_regret_summary.csv','coupled_regret_summary.csv']:shutil.copy2(R/file,D/file)
 style.save(fig,'complete_global','Complete smaller-side pricing resolves every benchmark, with exact feasible witnesses and bounded nonlinear regret.',
 {'a':'all 15 matched signed-linear inputs, three systems per size; differing caps explicitly shown; exact algorithm solves full polytope',
  'b':'all 15 signed-linear cases, exact rational feasible-joint construction versus error-aware full pricing lower bound; log axis',
  'c':'five fixed selected actions and a coupled branching control; bars are mathematical enclosures, not statistical intervals'},list(p.name for p in D.glob('*global*.csv')))

if __name__=='__main__':
 prediction();global_results()
 (style.QA/'resolution_figure_contracts.json').write_text(json.dumps(style.CONTRACTS,indent=2))
