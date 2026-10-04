"""Confirmatory fresh-seed mean-safety gate using ordinary observed losses.

The pre-fixed challenger is the robust Brier lower-bound rule. Selection across
six benchmark episode laws uses a Bonferroni empirical-Bernstein lower bound.
No search over favorable challengers or stopping times is performed.
"""
import environment
from environment import ROOT
import json,time,platform
import numpy as np,pandas as pd
from observable_calibration import episode
from gap_experiments import datasets,POLICIES
OUT=ROOT.parent/'data/results/global_and_prediction';OUT.mkdir(exist_ok=True)
DESIGN={'date':'2026-10-03','fixed_before_execution':True,'external_preregistration':False,
 'status':'confirmatory fresh-seed follow-up designed after the 512-episode observable pilot, not original preregistration',
 'tasks':['breast_cancer','iris_binary','wine_binary','digits_3_8','moons','mechanistic'],
 'challenger':'brier_lower','baseline':'entropy','calibration_n':8192,'test_n':2048,
 'alpha_family':.05,'family_size':6,'seed_offset':3100000,
 'radius':'sqrt(2*squared_sample_sd*log(2*T/alpha)/n)+14*log(2*T/alpha)/(3*(n-1))',
 'gain_support':[-1,1],'fixed_gate':'deploy challenger iff mean gain minus the stated radius is strictly positive; otherwise entropy',
 'unit':'independent seeded episodes conditional on fixed benchmark empirical population, not independent patients/domains',
 'development_rows':'same independently fixed 30% development rows excluded from every calibration/test episode',
 'target':'future mean one-query Brier risk under the same empirical episode law; not a sequence or unseen-domain guarantee'}
p=OUT/'mean_safe_design.json'
if p.exists():assert json.loads(p.read_text())==DESIGN
else:p.write_text(json.dumps(DESIGN,indent=2))

def run():
 start=time.perf_counter();rows=[];summary=[];data=datasets();N=8192;T=2048;ln=np.log(2*6/.05)
 for taskno,name in enumerate(DESIGN['tasks']):
  X,y=data[name];rng=np.random.default_rng(2700000+100000*taskno)
  dev=rng.choice(len(y),int(.3*len(y)),replace=False);pop=np.setdiff1d(np.arange(len(y)),dev)
  path=OUT/f'mean_safe_raw_{name}.npz'
  if path.exists():
   with np.load(path) as z:pack={k:z[k] for k in z.files}
  else:
   gains=[];losses=[];actions=[];ids=[];bases=[];branches=[]
   for stage,count,offset in [('calibration',N,0),('test',T,50000)]:
    for n in range(count):
     f,g,l,acts,ix,base,br=episode(X,y,pop,3100000+taskno*100000+offset+n)
     c=acts['brier_lower'];e=acts['entropy'];gains.append(g[c]);losses.append([l[e],l[c]]);actions.append([e,c]);ids.append(ix);bases.append(base);branches.append(br)
     if n%2048==2047:print('MEAN SAFE INFERENCE',name,stage,n+1,'seconds',round(time.perf_counter()-start,1),flush=True)
   pack={'gain':np.array(gains),'loss':np.array(losses),'actions':np.array(actions),'row_ids':np.array(ids),
    'base':np.array(bases),'branches':np.array(branches),'development_rows':dev,'evaluation_rows':pop}
   np.savez_compressed(path,**pack)
  gain=pack['gain'];mean=float(gain[:N].mean());variance=float(gain[:N].var(ddof=1))
  radius=float(np.sqrt(2*variance*ln/N)+14*ln/(3*(N-1)));lower=mean-radius;deploy=lower>0
  assert not np.intersect1d(dev,pack['row_ids'].ravel()).size
  for i,g in enumerate(gain):
   stage='calibration' if i<N else 'test'
   rows.append(dict(dataset=name,stage=stage,episode=i if i<N else i-N,entropy_brier=pack['loss'][i,0],
    challenger_brier=pack['loss'][i,1],challenger_gain=g,deploy_challenger=deploy,
    deployed_brier=pack['loss'][i,1 if deploy else 0],deployed_gain=g if deploy else 0))
  test=gain[N:];se=test.std(ddof=1)/np.sqrt(T)
  summary.append(dict(dataset=name,calibration_n=N,test_n=T,calibration_mean_gain=mean,sample_variance=variance,
   radius=radius,certified_mean_lower_bound=lower,deploy_challenger=deploy,
   test_challenger_gain=float(test.mean()),test_challenger_sd=float(test.std(ddof=1)),test_challenger_se=float(se),
   test_deployed_gain=float(test.mean()) if deploy else 0,entropy_test_brier=float(pack['loss'][N:,0].mean()),
   challenger_test_brier=float(pack['loss'][N:,1].mean()),gate_uses_only_calibration=True))
  pd.DataFrame(rows).to_csv(OUT/'mean_safe_episode_outcomes.csv',index=False)
  pd.DataFrame(summary).to_csv(OUT/'mean_safe_summary.csv',index=False)
  print('MEAN SAFE RESULT',name,'LCB',lower,'deploy',deploy,'test_gain',test.mean(),flush=True)
 (OUT/'mean_safe_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'design':DESIGN,'python':platform.python_version()},indent=2))
 print('MEAN SAFE COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':run()
