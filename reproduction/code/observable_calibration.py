"""Ordinary-label calibration of prospective realized Brier improvement.

All episode seeds are independent draws conditional on the fixed benchmark.
Development rows never occur in calibration or test episodes. Calibration and
test independently sample the same remaining empirical row population. The
guarantee is for this episode law, not for new patients or new domains.
"""
import environment
from environment import ROOT
import json,time,argparse,platform
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from gap_experiments import datasets,score_pack,choose,POLICIES
from method import interval
OUT=ROOT.parent/'data/results/global_and_prediction';OUT.mkdir(exist_ok=True)
DESIGN={'date':'2026-10-03','fixed_before_execution':True,'external_preregistration':False,
 'tasks':['breast_cancer','iris_binary','wine_binary','digits_3_8','moons','mechanistic'],
 'context':8,'candidates':4,'targets':12,'development_fraction':.3,
 'development_episodes':256,'calibration_episodes':512,'test_episodes':512,
 'seed_offset':2700000,'alpha':.05,'tolerances':[0,.01,.02],
 'predictor':'StandardScaler + LogisticRegression C=1 max_iter=1000; Beta(1,1) constant on single-class contexts',
 'meta_predictor':'RandomForestRegressor 300 trees min_samples_leaf=8 max_features=0.8 random_state=17',
 'endpoint':'realized one-query mean binary Brier improvement relative to the entropy policy',
 'calibration':'one-sided rank of predicted minus observed improvement for a frozen greedy meta policy; entropy has exactly zero contrast',
 'data_access':'development labels fit the meta predictor; calibration labels set the radius only; test candidate and target labels are used after all decisions',
 'unit':'episode seed conditional on a fixed empirical dataset; rows disjoint within episodes, overlaps across episodes retained; no domain/patient-population claim',
 'inference':'fresh classical inference for every episode; all static rules share exactly the same inference'}
dp=OUT/'observable_design.json'
if dp.exists():assert json.loads(dp.read_text())==DESIGN
else:dp.write_text(json.dumps(DESIGN,indent=2))

def brier_bounds(base,branches,k):
 a=base[:k,None];b=base[None,k:]
 _,_,lo,hi=interval(a,b,branches[:,0],branches[:,1]);ab=a*b
 denom=np.maximum(a*(1-a),np.finfo(float).tiny)
 f=lambda t:(t-ab)**2/denom
 lower=f(np.clip(ab,lo,hi));upper=np.maximum(f(lo),f(hi))
 return np.where((a==0)|(a==1),0,lower).mean(1),np.where((a==0)|(a==1),0,upper).mean(1)

def features(base,branches,s,k):
 a=base[:k];b=base[k:];L,U=brier_bounds(base,branches,k)
 drift=(1-a[:,None])*branches[:,0]+a[:,None]*branches[:,1]-b
 parts=[a, s['entropy'], s['raw'],s['kl'],s['js'],s['lower'],s['upper'],s['regret'],
  L,U,np.mean(drift,1),np.mean(abs(drift),1),np.max(abs(drift),1),
  np.mean((branches[:,1]-branches[:,0])**2,1),np.std(branches[:,0],1),np.std(branches[:,1],1),
  np.full(k,b.mean()),np.full(k,b.std())]
 return np.column_stack(parts)

def episode(X,y,rows,seed):
 rng=np.random.default_rng(seed);ix=rng.choice(rows,24,replace=False);tr,pool,target=ix[:8],ix[8:12],ix[12:]
 k=4;clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0))
 def fit(ctx,yy):
  if len(np.unique(yy))==1:return np.full(16,(yy.sum()+1)/(len(yy)+2))
  clf.fit(X[ctx],yy);return clf.predict_proba(X[np.r_[pool,target]])[:,1]
 base=fit(tr,y[tr]);branches=np.empty((k,2,12))
 for i in range(k):
  for label in [0,1]:branches[i,label]=fit(np.r_[tr,pool[i]],np.r_[y[tr],label])[k:]
 s=score_pack(base,branches,k);L,U=brier_bounds(base,branches,k)
 s['brier_lower']=L;s['brier_upper']=U
 s['brier_regret']=np.array([max(0,np.max(np.delete(U,i))-L[i]) for i in range(k)])
 # No actual candidate/target outcome is read until all interface quantities exist.
 static={p:choose(s,p) for p in POLICIES}
 static.update(brier_lower=int(L.argmax()),brier_minimax=int(s['brier_regret'].argmin()))
 losses=((branches[np.arange(k),y[pool]]-y[target])**2).mean(1)
 entropy_choice=static['entropy'];gain=losses[entropy_choice]-losses
 return features(base,branches,s,k),gain,losses,static,ix,base,branches

def run():
 start=time.perf_counter();allrows=[];calrows=[];summaries=[];data=datasets()
 for taskno,name in enumerate(DESIGN['tasks']):
  X,y=data[name];rng=np.random.default_rng(2700000+100000*taskno)
  dev=rng.choice(len(y),int(.3*len(y)),replace=False);remaining=np.setdiff1d(np.arange(len(y)),dev)
  assert len(dev)>=24 and len(remaining)>=24
  path=OUT/f'observable_{name}.npz'
  if path.exists():
   with np.load(path) as z:pack={k:z[k] for k in z.files}
  else:
   fs=[];gs=[];loss=[];actions=[];ids=[];bases=[];brs=[]
   for stage,count,population,offset in [('development',256,dev,0),('calibration',512,remaining,10000),('test',512,remaining,20000)]:
    for n in range(count):
     f,g,l,a,ix,b,br=episode(X,y,population,2700000+taskno*100000+offset+n)
     fs.append(f);gs.append(g);loss.append(l);actions.append(list(a.values()));ids.append(ix);bases.append(b);brs.append(br)
    print('OBSERVABLE INFERENCE',name,stage,'seconds',round(time.perf_counter()-start,1),flush=True)
   pack={'features':np.array(fs),'gain':np.array(gs),'loss':np.array(loss),'actions':np.array(actions),
    'row_ids':np.array(ids),'base':np.array(bases),'branches':np.array(brs),'development_rows':dev,'evaluation_rows':remaining,
    'action_names':np.array(POLICIES+['brier_lower','brier_minimax'])}
   np.savez_compressed(path,**pack)
  f=pack['features'];g=pack['gain'];a=pack['actions'];entropy_index=POLICIES.index('entropy')
  assert not np.intersect1d(pack['development_rows'],pack['row_ids'][256:].ravel()).size
  model=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,max_features=.8,random_state=17,n_jobs=1)
  model.fit(f[:256].reshape(-1,f.shape[-1]),g[:256].ravel())
  pred=model.predict(f[256:].reshape(-1,f.shape[-1])).reshape(-1,4)
  # The reference action's contrast is known exactly, not estimated.
  for n in range(len(pred)):pred[n,a[256+n,entropy_index]]=0
  greedy=pred.argmax(1);predicted=pred[np.arange(len(pred)),greedy]
  observed=g[256:][np.arange(len(pred)),greedy]
  residual=predicted[:512]-observed[:512];rank=int(np.ceil(513*.95));q=float(np.sort(residual)[rank-1])
  # Zero clipping is conservative and avoids an unjustified negative radius.
  q=max(0,q)
  np.savez_compressed(OUT/f'observable_calibration_{name}.npz',predicted_candidate_gain=pred,greedy=greedy,
   observed_candidate_gain=g[256:],residual=residual,q=q,rank=rank)
  for n in range(512):
   full=768+n;reference=int(a[full,entropy_index]);j=int(greedy[512+n]);pg=float(predicted[512+n]);gain=float(g[full,j])
   calrows.append(dict(dataset=name,episode=n,predicted_gain=pg,observed_gain=gain,q=q,lower_bound=pg-q,
    coverage=gain>=pg-q-1e-12,greedy_action=j,entropy_action=reference))
   for ip,p in enumerate(pack['action_names']):
    pos=int(a[full,ip]);allrows.append(dict(dataset=name,episode=n,policy=str(p),choice=pos,brier=float(pack['loss'][full,pos]),
     gain=float(g[full,pos]),issued=False,wrong=False))
   allrows.append(dict(dataset=name,episode=n,policy='meta_uncalibrated',choice=j,brier=float(pack['loss'][full,j]),gain=gain,issued=False,wrong=False))
   for delta in DESIGN['tolerances']:
    issued=bool(pg-q>-delta);choice=j if issued else reference
    allrows.append(dict(dataset=name,episode=n,policy=f'label_calibrated_{delta:g}',choice=choice,
     brier=float(pack['loss'][full,choice]),gain=float(g[full,choice]),issued=issued,wrong=bool(issued and g[full,choice]<-delta-1e-12)))
  pd.DataFrame(allrows).to_csv(OUT/'observable_policy_test.csv',index=False)
  pd.DataFrame(calrows).to_csv(OUT/'observable_coverage_test.csv',index=False)
  summaries.append(dict(dataset=name,calibration_n=512,test_n=512,rank=rank,q=q,
   coverage=np.mean(observed[512:]>=predicted[512:]-q-1e-12),development_rows=len(dev),evaluation_rows=len(remaining)))
  print('OBSERVABLE CALIBRATION',name,'q',q,'seconds',round(time.perf_counter()-start,1),flush=True)
 df=pd.DataFrame(allrows)
 summary=df.groupby(['dataset','policy']).agg(n=('episode','size'),mean_gain=('gain','mean'),sd_gain=('gain','std'),
  issued=('issued','sum'),wrong=('wrong','sum'),mean_brier=('brier','mean')).reset_index()
 summary['certification_applicable']=summary.policy.str.startswith('label_calibrated_')
 summary['wrong_definition']='issued gain below minus declared tolerance; not every negative query'
 summary.to_csv(OUT/'observable_policy_summary.csv',index=False)
 pd.DataFrame(summaries).to_csv(OUT/'observable_calibration_summary.csv',index=False)
 (OUT/'observable_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'python':platform.python_version(),
  'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'design':DESIGN},indent=2))
 print('OBSERVABLE COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':run()
