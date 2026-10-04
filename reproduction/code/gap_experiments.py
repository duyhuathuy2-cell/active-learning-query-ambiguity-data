"""Fixed supplementary design: matched baselines, oracle tasks, interventions and cost.

The design is saved before any outcome is generated. No population claims are
made for overlapping public-data splits. Oracle values belong to specified
finite-hypothesis simulators; ordinary benchmark labels are not oracle scores.
"""
import environment
from environment import ROOT
import argparse, itertools, json, os, platform, time, warnings
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.special import expit, logsumexp
from scipy.optimize import linprog
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from method import scores, entropy, interval

OUT=ROOT.parent/'data/results/policy_controls'; OUT.mkdir(exist_ok=True)
POLICIES=['raw','kl','js','lower','minimax','tv_midpoint','endpoint_lo','endpoint_hi','entropy','certify_entropy']
DESIGN={'date':'2026-10-02','fixed_before_execution':True,'externally_preregistered':False,
 'cached_comparison':'all 600 original settings and 60 expanded settings; ten selectors',
 'certificate_intervention':'every archived sequential state with a strict certificate; actual candidate labels revealed after scoring',
 'new_sequential':'original six fixed tasks, logistic, ten splits, five queries; all ten selectors freshly fitted in one environment',
 'oracle_families':['linear','interaction','radial','threshold'], 'oracle_systems_per_family':400,
 'calibration_systems':100,'test_systems':300,'oracle_candidates':8,'oracle_targets':16,'initial_labels':16,
 'oracle_learner':'StandardScaler + LogisticRegression(C=1,max_iter=1000)',
 'oracle_definition':'exact finite-hypothesis posterior, Bernoulli outcomes conditionally independent given shared hypothesis',
 'cost_sensitivity':'fixed geometric query costs 0.5..2; fixed target weights 1..2; information/cost objective',
 'lp_candidates':[2,4,6,8,10,12,14], 'lp_targets':16,'lp_repetitions':3,'lp_seconds_per_solve':10,
 'lp_resource_gate':'skip exact solve if candidates >12 or sparse stored arrays >128 MiB; report skipped status',
 'inference':'new classical-model inference; existing neural outputs reused explicitly',
 'reporting':'complete outcomes; matched per-system contrasts; no favourable task/seed filtering'}
design_path=OUT/'design.json'
if design_path.exists():
 assert json.loads(design_path.read_text())==DESIGN,'Change requires a new design namespace.'
else:design_path.write_text(json.dumps(DESIGN,indent=2),encoding='utf8')

def datasets():
 bc,ir,wi,di=load_breast_cancer(),load_iris(),load_wine(),load_digits()
 rng=np.random.default_rng(819);X=rng.uniform(-2,2,(600,5));p=expit(2*X[:,0]-X[:,1]+1.3*X[:,2]*X[:,3]);y=rng.binomial(1,p)
 return {'breast_cancer':(bc.data,bc.target),'iris_binary':(ir.data[ir.target>0],ir.target[ir.target>0]-1),
 'wine_binary':(wi.data[wi.target<2],wi.target[wi.target<2]),'digits_3_8':(di.data[np.isin(di.target,[3,8])],(di.target[np.isin(di.target,[3,8])]==8).astype(int)),
 'moons':make_moons(n_samples=600,noise=.22,random_state=731),'mechanistic':(X,y)}

def split(y,seed):
 rng=np.random.default_rng(1000+seed);per=[rng.permutation(np.flatnonzero(y==k)) for k in (0,1)]
 train=np.r_[per[0][:8],per[1][:8]];rest=rng.permutation(np.setdiff1d(np.arange(len(y)),train));return train,rest[:8],rest[8:56]

def score_pack(base,branches,nq,weights=None,costs=None):
 s=scores(base,branches,nq);a=base[:nq,None];b=base[None,nq:]
 L,U,lo,hi=interval(a,b,branches[:,0],branches[:,1]);t=(lo+hi)/2
 def value(tt):
  safe=np.clip(a,np.finfo(float).eps,1-np.finfo(float).eps)
  v=entropy(b)-(1-safe)*entropy((b-tt)/(1-safe))-safe*entropy(tt/safe)
  return np.where((a==0)|(a==1),0,v)
 mid=value(t)
 s['tv_midpoint']=mid.mean(1)
 if weights is not None:
  # Weighted/costed experiment uses exact pair formulas, including every target.
  q0,q1=branches[:,0],branches[:,1];m=(1-a)*q0+a*q1
  from method import kl_repair
  r0,r1=kl_repair(a,b,q0,q1)
  arrays={'raw':entropy(b)-(1-a)*entropy(q0)-a*entropy(q1),'js':entropy(m)-(1-a)*entropy(q0)-a*entropy(q1),
  'kl':entropy(b)-(1-a)*entropy(r0)-a*entropy(r1),'lower':L,'upper':U,'tv_midpoint':mid}
  for name,tt in [('endpoint_lo',lo),('endpoint_hi',hi)]:arrays[name]=value(tt)
  for name,v in arrays.items():s[name]=(v@weights)/costs
  s['entropy']=s['entropy']/costs
  LL,UU=s['lower'],s['upper'];s['regret']=np.array([max(0,np.max(np.delete(UU,i))-LL[i]) for i in range(nq)])
  s['certified']=np.array([LL[i]>np.max(np.delete(UU,i))+1e-9 for i in range(nq)])
 return s

def choose(s,policy):
 if policy=='minimax':return int(np.argmin(s['regret']))
 if policy=='certify_entropy':
  cert=np.flatnonzero(s['certified']);return int(cert[0]) if len(cert) else int(np.argmax(s['entropy']))
 return int(np.argmax(s[policy]))

def cached():
 start=time.perf_counter();rows=[];data=datasets()
 cases=pd.read_csv(ROOT.parent/'data/results/benchmark/audit_cases.csv')
 for r in cases.itertuples():
  with np.load(ROOT.parent/'data/results/benchmark/cache'/f'{r.dataset}_{r.model}_{r.seed}_initial.npz') as z:
   s=score_pack(z['base'],z['branches'],8);y=data[r.dataset][1];pool=z['pool'];target=z['target']
   loss=np.mean((z['branches'][np.arange(8),y[pool]]-y[target])**2,axis=1)
   for policy in POLICIES:
    p=choose(s,policy);rows.append(dict(dataset=r.dataset,model=r.model,seed=r.seed,policy=policy,
     choice=p,brier=loss[p],internal_regret=s['regret'][p],certificate_available=bool(s['certified'].any())))
 df=pd.DataFrame(rows);df.to_csv(OUT/'matched_one_step.csv',index=False)
 baseline=df[df.policy=='raw'][['dataset','model','seed','brier']].rename(columns={'brier':'raw_brier'})
 paired=df.merge(baseline,on=['dataset','model','seed'],validate='many_to_one');paired['delta_brier']=paired.brier-paired.raw_brier
 paired.to_csv(OUT/'matched_one_step_paired.csv',index=False)
 paired.groupby(['model','dataset','policy']).agg(n=('seed','size'),mean=('delta_brier','mean'),sd=('delta_brier','std'),
  better=('delta_brier',lambda x:int((x < -1e-12).sum())),worse=('delta_brier',lambda x:int((x>1e-12).sum()))).reset_index().to_csv(OUT/'matched_one_step_summary.csv',index=False)
 expanded=[]
 for kind in ['logistic','tabpfn_8']:
  df=pd.read_csv(ROOT.parent/f'data/results/multiclass_and_calibration/expanded_{kind}_audit.csv')
  for (task,seed),g in df.groupby(['dataset','seed']):
   with np.load(ROOT.parent/f'data/results/multiclass_and_calibration/expanded_cache/{task}_{kind}_{seed}_initial.npz') as z:s=score_pack(z['base'],z['branches'],32)
   g=g.sort_values('candidate')
   for policy in POLICIES:
    p=choose(s,policy);expanded.append(dict(dataset=task,model=kind,seed=seed,policy=policy,choice=p,brier=float(g.iloc[p].actual_brier),internal_regret=s['regret'][p]))
 pd.DataFrame(expanded).to_csv(OUT/'expanded_matched_one_step.csv',index=False)
 interventions=[];seq=pd.read_csv(ROOT.parent/'data/results/benchmark/sequential.csv')
 # Deduplicate by context, not by trajectory: repeated visits are not interventions.
 seen=set()
 for r in seq.itertuples():
  history=json.loads(r.selected_positions);prefix=tuple(sorted(history[:-1]));key=(r.dataset,r.model,r.seed,prefix)
  if key in seen:continue
  seen.add(key);suffix='-'.join(map(str,prefix)) if prefix else 'initial'
  with np.load(ROOT.parent/f'data/results/benchmark/cache/{r.dataset}_{r.model}_{r.seed}_{suffix}.npz') as z:
   n=len(z['remain']);s=score_pack(z['base'],z['branches'],n);cert=np.flatnonzero(s['certified'])
   if not len(cert):continue
   p=int(cert[0]);e=choose(s,'entropy');y=data[r.dataset][1];pool=z['pool'];target=z['target']
   losses=np.mean((z['branches'][np.arange(n),y[pool[z['remain']]]]-y[target])**2,axis=1)
   interventions.append(dict(dataset=r.dataset,model=r.model,seed=r.seed,prefix=json.dumps(prefix),remaining=n,
    certificate_position=int(z['remain'][p]),entropy_position=int(z['remain'][e]),different_choice=p!=e,
    certificate_brier=losses[p],entropy_brier=losses[e],delta_brier=losses[p]-losses[e],
    certificate_internal_regret=s['regret'][p],entropy_internal_regret=s['regret'][e]))
 pd.DataFrame(interventions).to_csv(OUT/'same_context_certificate_interventions.csv',index=False)
 print('CACHED',len(rows),'expanded',len(expanded),'certificate states',len(interventions),'seconds',time.perf_counter()-start,flush=True)

def classical_state(name,seed,X,y,tr,pool,target,selected):
 path=OUT/'classical_cache'/f'{name}_{seed}_{"-".join(map(str,sorted(selected))) or "initial"}.npz';path.parent.mkdir(exist_ok=True)
 if path.exists():
  with np.load(path) as z:return {k:z[k] for k in z.files}
 context=np.r_[tr,pool[sorted(selected)]];remain=np.array([i for i in range(len(pool)) if i not in selected]);query=pool[remain]
 clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0));clf.fit(X[context],y[context])
 base=clf.predict_proba(X[np.r_[query,target]])[:,1];branches=np.empty((len(query),2,len(target)))
 for i in range(len(query)):
  for label in (0,1):
   ctx=np.r_[tr,pool[sorted(selected+[int(remain[i])])]];yy=y[ctx].copy();yy[np.flatnonzero(ctx==query[i])]=label
   clf.fit(X[ctx],yy);branches[i,label]=clf.predict_proba(X[target])[:,1]
 z=dict(base=base,branches=branches,remain=remain,train=context,pool=pool,target=target);np.savez_compressed(path,**z);return z

def sequential():
 start=time.perf_counter();rows=[]
 for name,(X,y) in datasets().items():
  for seed in range(10):
   tr,pool,target=split(y,seed)
   for policy in POLICIES:
    selected=[]
    for budget in range(1,6):
     z=classical_state(name,seed,X,y,tr,pool,target,selected);s=score_pack(z['base'],z['branches'],len(z['remain']))
     p=choose(s,policy);pos=int(z['remain'][p]);selected.append(pos);pred=z['branches'][p,y[pool[pos]]]
     rows.append(dict(dataset=name,model='logistic',seed=seed,policy=policy,budget=budget,selected_positions=json.dumps(selected),
      brier=float(np.mean((pred-y[target])**2)),error=float(np.mean((pred>=.5)!=y[target])),internal_regret=float(s['regret'][p])))
   pd.DataFrame(rows).to_csv(OUT/'additional_sequential.csv',index=False)
  print('SEQUENTIAL',name,'seconds',round(time.perf_counter()-start,1),flush=True)

def oracle_system(family,seed):
 rng=np.random.default_rng(820000+10000*DESIGN['oracle_families'].index(family)+seed)
 n=16;k=8;m=16;h=12;X=rng.normal(size=(n+k+m,4));W=rng.normal(size=(h,4))*.65
 if family=='linear':features=X
 elif family=='interaction':features=np.c_[X[:,0],X[:,1],X[:,0]*X[:,1],X[:,2]*X[:,3]]
 elif family=='radial':features=np.c_[X[:,0],X[:,1],X[:,0]**2-1,X[:,1]**2-1]
 else:features=np.c_[X[:,0],X[:,1],(X[:,0]>0)*2-1,(X[:,1]>0)*2-1]
 theta=np.clip(expit(features@W.T+rng.normal(0,.25,size=h)),.01,.99).T
 hidden=int(rng.integers(h));y=rng.binomial(1,theta[hidden]);logw=(y[:n]*np.log(theta[:,:n])+(1-y[:n])*np.log1p(-theta[:,:n])).sum(1)
 pi=np.exp(logw-logsumexp(logw));aa=theta[:,n:n+k];bb=theta[:,n+k:];a=pi@aa;b=pi@bb
 t=np.einsum('h,hi,hj->ij',pi,aa,bb);q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None])
 true_pairs=entropy(b)[None,:]-(1-a[:,None])*entropy(q0)-a[:,None]*entropy(q1);true=true_pairs.mean(1)
 clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0))
 # If the random context is single-class, retain it with a smoothed intercept
 # predictor. This event is reported and never silently discarded.
 single=len(np.unique(y[:n]))==1
 if single:
  p=(y[:n].sum()+1)/(n+2);base=np.full(k+m,p);branches=np.empty((k,2,m))
  for label in (0,1):branches[:,label,:]=(y[:n].sum()+label+1)/(n+3)
 else:
  clf.fit(X[:n],y[:n]);base=clf.predict_proba(X[n:])[:,1];branches=np.empty((k,2,m))
  for i in range(k):
   for label in (0,1):
    clf.fit(np.r_[X[:n],X[n+i:n+i+1]],np.r_[y[:n],label]);branches[i,label]=clf.predict_proba(X[n+k:])[:,1]
 s=score_pack(base,branches,k)
 # Exact coherent EPIG control uses the same posterior, pool and target design.
 exact=scores(np.r_[a,b],np.stack([q0,q1],axis=1),k)
 assert np.max(abs(exact['raw']-true))<1e-12
 cost=np.geomspace(.5,2,k);weight=np.linspace(1,2,m);weight/=weight.sum()
 weighted=score_pack(base,branches,k,weight,cost);truth_cost=(true_pairs@weight)/cost
 return s,true,weighted,truth_cost,single

def oracle():
 start=time.perf_counter();rows=[];quantiles=[];cal_rows=[]
 for family in DESIGN['oracle_families']:
  cache=OUT/f'oracle_{family}_scores.npz'
  if cache.exists():
   with np.load(cache) as z:pack={k:z[k] for k in z.files}
  else:
   values=[];weighted=[];truth=[];costtruth=[];singles=[]
   for seed in range(400):
    s,t,ws,wt,single=oracle_system(family,seed);values.append(s);weighted.append(ws);truth.append(t);costtruth.append(wt);singles.append(single)
    if seed%100==99:print('ORACLE',family,seed+1,'seconds',round(time.perf_counter()-start,1),flush=True)
   pack={key:np.array([s[key] for s in values]) for key in values[0]}
   pack.update({'weighted_'+key:np.array([s[key] for s in weighted]) for key in weighted[0]})
   pack.update(truth=np.array(truth),costtruth=np.array(costtruth),single_class=np.array(singles));np.savez_compressed(cache,**pack)
  true=pack['truth'];N=100;rank=int(np.ceil((N+1)*.95))
  radii={}
  for method in ['tv','kl','raw']:
   lower=pack['lower'] if method=='tv' else pack[method];upper=pack['upper'] if method=='tv' else pack[method]
   e=np.maximum.reduce([lower[:N]-true[:N],true[:N]-upper[:N],np.zeros_like(true[:N])]).max(1)
   q=float(np.sort(e)[rank-1]);radii[method]=q;quantiles.append(dict(family=family,method=method,n=N,rank=rank,q=q))
   for seed in range(N,400):
    for calibrated in [False,True]:
     radius=q if calibrated else 0;L=np.clip(lower[seed]-radius,0,1);U=np.clip(upper[seed]+radius,0,1)
     assert np.all(L<=U+1e-12)
     regret=np.array([max(0,np.max(np.delete(U,i))-L[i]) for i in range(8)]);p=int(np.argmin(regret))
     actual=float(true[seed].max()-true[seed,p]);cert=bool(L[p]>np.max(np.delete(U,p))+1e-9)
     cal_rows.append(dict(family=family,seed=seed,method=method,calibrated=calibrated,coverage=bool(np.all((true[seed]>=L-1e-12)&(true[seed]<=U+1e-12))),
      certified=cert,wrong_certificate=bool(cert and actual>1e-9),true_regret=actual,regret_bound=regret[p],bound_violated=bool(actual>regret[p]+1e-9),width=float((U-L).mean())))
  for seed in range(N,400):
   for objective,prefix,tr in [('uniform','',true),('weighted_cost','weighted_',pack['costtruth'])]:
    s={key[len(prefix):]:v[seed] for key,v in pack.items() if (key.startswith(prefix) if prefix else not key.startswith('weighted_')) and v.ndim==2}
    for policy in POLICIES:
     p=choose(s,policy);rows.append(dict(family=family,seed=seed,objective=objective,policy=policy,choice=p,true_regret=float(tr[seed].max()-tr[seed,p]),
      internal_regret=float(s['regret'][p]),certified=bool(s['certified'].any()),single_class_context=bool(pack['single_class'][seed])))
    rows.append(dict(family=family,seed=seed,objective=objective,policy='exact_EPIG',choice=int(tr[seed].argmax()),true_regret=0.,internal_regret=0.,certified=True,single_class_context=bool(pack['single_class'][seed])))
  pd.DataFrame(rows).to_csv(OUT/'oracle_policy_test.csv',index=False);pd.DataFrame(cal_rows).to_csv(OUT/'learned_oracle_calibration.csv',index=False)
  pd.DataFrame(quantiles).to_csv(OUT/'learned_oracle_quantiles.csv',index=False)
 pd.DataFrame(rows).groupby(['family','objective','policy']).agg(n=('seed','size'),mean=('true_regret','mean'),sd=('true_regret','std')).reset_index().to_csv(OUT/'oracle_policy_summary.csv',index=False)
 pd.DataFrame(cal_rows).groupby(['family','method','calibrated']).agg(n=('seed','size'),coverage=('coverage','mean'),certified=('certified','sum'),wrong=('wrong_certificate','sum'),
  violations=('bound_violated','sum'),width=('width','mean'),true_regret=('true_regret','mean')).reset_index().to_csv(OUT/'learned_oracle_calibration_summary.csv',index=False)
 print('ORACLE COMPLETE',time.perf_counter()-start,flush=True)

def scaling():
 from global_completion import lp_template
 rows=[];m=16
 for k in DESIGN['lp_candidates']:
  for seed in range(3):
   build=time.perf_counter();states,E,U,template=lp_template(k,m);n=len(states)
   memory=sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in [E,U])+states.nbytes+template.nbytes
   rng=np.random.default_rng(910000+100*k+seed);pi=rng.dirichlet(np.ones(12));theta=rng.beta(.7,.7,(12,k+m))
   a=pi@theta[:,:k];b=pi@theta[:,k:];t=np.einsum('h,hi,hj->ij',pi,theta[:,:k],theta[:,k:]);q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None])
   # Nested K experiments are not implied; each design draws an independent system.
   pert=np.clip(np.stack([q0,q1],1)+.03*rng.normal(size=(k,2,m)),1e-6,1-1e-6)
   t0=time.perf_counter()
   for _ in range(20):scores(np.r_[a,b],pert,k)
   local=(time.perf_counter()-t0)/20
   _,_,lo,hi=interval(a[:,None],b[None,:],pert[:,0],pert[:,1]);bound=template.copy()
   for j in range(m):
    off=j*(n+2*k)+n;bound[off:off+2*k:2]=hi[:,j];bound[off+1:off+2*k:2]=-lo[:,j]
   row=dict(candidates=k,targets=m,seed=seed,variables=(m+1)*2**k,sparse_bytes=memory,construction_seconds=time.perf_counter()-build,
    local_seconds=local,status='resource_gate',solve_seconds=np.nan,residual=np.nan)
   if k<=12 and memory<=128*1024**2:
    t0=time.perf_counter()
    with warnings.catch_warnings():
     warnings.simplefilter('ignore',scipy.optimize.OptimizeWarning)
     res=linprog(np.zeros(E.shape[1]),A_ub=U,b_ub=bound,A_eq=E,b_eq=np.r_[1,a,b],bounds=(0,None),method='highs',
      options={'time_limit':10.,'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    row.update(status={0:'feasible',1:'time_limit',2:'infeasible'}.get(res.status,f'solver_{res.status}'),solve_seconds=time.perf_counter()-t0)
    if res.success:row['residual']=max(float(np.max(abs(E@res.x-np.r_[1,a,b]))),float(max(0,np.max(U@res.x-bound))))
   rows.append(row);pd.DataFrame(rows).to_csv(OUT/'scalability.csv',index=False)
   print('SCALING',k,seed,row['status'],row['solve_seconds'],flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['cached','sequential','oracle','scaling'],required=True);arg=ap.parse_args()
 t0=time.perf_counter();globals()[arg.stage]()
 (OUT/f'{arg.stage}_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-t0,'python':platform.python_version(),'numpy':np.__version__,
  'scipy':scipy.__version__,'sklearn':sklearn.__version__,'machine':platform.machine(),'processor':platform.processor(),'design':DESIGN},indent=2),encoding='utf8')
