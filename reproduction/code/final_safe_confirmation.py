"""Reproduce the frozen 16,384-calibration / 2,048-test episode law directly."""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time,datetime
from loss_contrasts import contrasts
from observable_calibration import episode
from gap_experiments import datasets
OUT=ROOT.parent/'data/results/global_and_prediction';OUT.mkdir(exist_ok=True)
DESIGN={'date':'2026-10-03','fixed_before_execution':True,'external_preregistration':False,
 'status':'sample size increased based on the earlier threshold pilot before any bounded_confirmatory_summary existed',
 'tasks':['breast_cancer','iris_binary','wine_binary','digits_3_8','moons','mechanistic'],
 'baselines':['entropy','raw'],'thresholds':[.025,.05,.1,.2,.5,1.],
 'challenger':'brier_lower','calibration_n':16384,'final_test_n':2048,'seed_offset':5000000,
 'additional_calibration_offset':8192,'final_test_offset':70000,'alpha_family':.05,'family_size':72,
 'radius':'sqrt(2*variance*log(2*72/alpha)/n)+14*c*log(2*72/alpha)/(3*(n-1))',
 'selection':'largest positive calibration lower bound over fixed thresholds; fallback baseline otherwise',
 'boundary':'future mean loss under six fixed empirical episode laws, not per-query or unseen-domain benefit',
 'earlier_test_access':'earlier confirmatory test outcomes do not set final choices; entirely fresh final test batch',
 'study_history':'all exploratory, unbounded and earlier fixed-size outcomes retained without optional favorable-task removal'}
dp=OUT/'final_safe_design.json'
assert json.loads(dp.read_text())==DESIGN

def run():
 start=time.perf_counter();calrows=[];rows=[];summary=[];data=datasets();ln=np.log(2*72/.05);N=16384
 for taskno,name in enumerate(DESIGN['tasks']):
  X,y=data[name]
  path=OUT/f'final_safe_raw_{name}.npz'
  if path.exists():
   with np.load(path) as z:pack={key:z[key] for key in z.files}
   dev,pop=pack['development_rows'],pack['evaluation_rows']
  else:
   dev=np.random.default_rng(2700000+100000*taskno).choice(len(y),int(.3*len(y)),replace=False)
   pop=np.setdiff1d(np.arange(len(y)),dev)
   ids=[];bases=[];branches=[];actions=[]
   for stage,count,offset in [('calibration',16384,0),('final_test',2048,70000)]:
    for n in range(count):
     f,g,l,acts,ix,base,br=episode(X,y,pop,5000000+taskno*100000+offset+n)
     ids.append(ix);bases.append(base);branches.append(br);actions.append([acts['entropy'],acts['brier_lower']])
     if n%2048==2047:print('CONFIRMATION INFERENCE',name,stage,n+1,flush=True)
   pack=dict(row_ids=np.array(ids),base=np.array(bases),branches=np.array(branches),actions=np.array(actions),
             development_rows=dev,evaluation_rows=pop)
   np.savez_compressed(path,**pack)
  assert not np.intersect1d(dev,pack['row_ids'].ravel()).size
  for baseline in DESIGN['baselines']:
   gains,A,ref,challenger,ref_loss,ch_loss=contrasts(pack,y,baseline);options=[]
   for c in DESIGN['thresholds']:
    gg=np.where(A[:N]<=c,gains[:N],0);mean=float(gg.mean());variance=float(gg.var(ddof=1))
    radius=float(np.sqrt(2*variance*ln/N)+14*c*ln/(3*(N-1)));lower=mean-radius;options.append((lower,c,mean,variance,radius))
    calrows.append(dict(dataset=name,baseline=baseline,threshold=c,mean_gain=mean,sample_variance=variance,radius=radius,
     certified_mean_lower_bound=lower,eligible=int((A[:N]<=c).sum()),n=N))
   best=max(options);deploy=best[0]>0;c=best[1];eligible=A[N:]<=c;different=ref[N:]!=challenger[N:]
   issued=eligible&deploy&different;test_gain=np.where(eligible&deploy,gains[N:],0)
   for i in range(len(gains)):
    selected=gains[i] if deploy and A[i]<=c else 0
    rows.append(dict(dataset=name,baseline=baseline,stage='calibration' if i<N else 'test',episode=i if i<N else i-N,
     threshold=c,deploy=deploy,envelope=A[i],different_action=bool(ref[i]!=challenger[i]),
     issued=bool(deploy and A[i]<=c and ref[i]!=challenger[i]),observed_challenger_gain=gains[i],
     deployed_gain=selected,baseline_brier=ref_loss[i],deployed_brier=ref_loss[i]-selected))
   summary.append(dict(dataset=name,baseline=baseline,calibration_n=N,test_n=2048,selected_threshold=c,
    certified_mean_lower_bound=best[0],calibration_mean_gain=best[2],radius=best[4],deploy=deploy,
    test_mean_gain=float(test_gain.mean()),test_sd_gain=float(test_gain.std(ddof=1)),test_se_gain=float(test_gain.std(ddof=1)/np.sqrt(2048)),
    issued=int(issued.sum()),worse_queries=int(np.sum(issued&(gains[N:]<0))),
    baseline_brier=float(ref_loss[N:].mean()),deployed_brier=float((ref_loss[N:]-test_gain).mean())))
  outcomes=pd.DataFrame(rows)
  np.savez_compressed(OUT/'final_safe_outcomes_compact.npz',**{k:outcomes[k].to_numpy(dtype='U16' if k in ['dataset','baseline','stage'] else None) for k in outcomes})
  pd.DataFrame(calrows).to_csv(OUT/'final_safe_options.csv',index=False)
  pd.DataFrame(summary).to_csv(OUT/'final_safe_summary.csv',index=False)
  print('FINAL SAFE RESULT',name,[{k:r[k] for k in ['baseline','selected_threshold','certified_mean_lower_bound','deploy','test_mean_gain','issued']} for r in summary[-2:]],flush=True)
 (OUT/'final_safe_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'design':DESIGN,'all_six_tasks_complete':len(summary)==12},indent=2))
if __name__=='__main__':run()
