"""New namespace; independently generated synthetic datasets; fixed design."""
import environment
from environment import ROOT
import argparse,json,time
import numpy as np
import pandas as pd
import run_experiments as old
from method import scores
OUT=ROOT.parent/'data/results/multiclass_and_calibration'; OUT.mkdir(exist_ok=True)
old.CACHE=OUT/'expanded_cache'; old.CACHE.mkdir(exist_ok=True)

def data(name,seed):
    if name in ('breast_cancer','digits_3_8'): return old.datasets()[name]
    rng=np.random.default_rng(410000+1000*['linear','interaction','radial','heterogeneous'].index(name)+seed)
    X=rng.normal(size=(1000,12))
    eta={'linear':lambda:2*X[:,0]-X[:,1]+.5*X[:,2],
         'interaction':lambda:1.8*X[:,0]*X[:,1]+X[:,2]-.5*X[:,3],
         'radial':lambda:2*(X[:,0]**2+X[:,1]**2-1.5),
         'heterogeneous':lambda:np.where(X[:,0]>0,2*X[:,1]-X[:,2],-1.5*X[:,1]+X[:,3])}[name]()
    y=rng.binomial(1,1/(1+np.exp(-eta)))
    return X,y

def split(y,seed):
    rng=np.random.default_rng(420000+seed)
    tr=np.concatenate([rng.permutation(np.flatnonzero(y==c))[:16] for c in (0,1)])
    rest=rng.permutation(np.setdiff1d(np.arange(len(y)),tr))
    return tr,rest[:32],rest[32:96]

def run(kind):
    rows=[]; audit=[]; start=time.perf_counter()
    names=['breast_cancer','digits_3_8','linear','interaction','radial','heterogeneous']
    for name in names:
      for seed in range(5):
        X,y=data(name,seed); tr,pool,target=split(y,seed)
        z=old.state(name,kind,seed,X,y,tr,pool,target)
        s=scores(z['base'],z['branches'],32)
        losses=np.mean((z['branches'][np.arange(32),y[pool]]-y[target])**2,axis=1)
        baseline=np.mean((z['base'][32:]-y[target])**2)
        for i in range(32):
            audit.append(dict(dataset=name,model=kind,seed=seed,candidate=i,baseline_brier=baseline,actual_brier=losses[i],**{k:float(v[i]) for k,v in s.items()}))
        for policy in ['raw','kl','minimax','entropy','certify_entropy']:
            selected=[]
            for step in range(1,(3 if kind=='logistic' else 1)+1):
                zz=z if step==1 else old.state(name,kind,seed,X,y,tr,pool,target,selected)
                ss=s if step==1 else scores(zz['base'],zz['branches'],len(zz['remain']))
                cert=np.flatnonzero(ss['certified'])
                pick=int(np.argmin(ss['regret'])) if policy=='minimax' else int(cert[0]) if policy=='certify_entropy' and len(cert) else int(np.argmax(ss['entropy' if policy=='certify_entropy' else policy]))
                pos=int(zz['remain'][pick]); selected.append(pos)
                actual=zz['branches'][pick,y[pool[pos]]]
                rows.append(dict(dataset=name,model=kind,seed=seed,policy=policy,budget=step,selected=json.dumps(sorted(selected)),brier=float(np.mean((actual-y[target])**2)),baseline_brier=baseline,certificate_trigger=bool(len(cert)),used_certificate=bool(policy=='certify_entropy' and len(cert)),internal_regret=float(ss['regret'][pick])))
        pd.DataFrame(rows).to_csv(OUT/f'expanded_{kind}_replay.csv',index=False)
        pd.DataFrame(audit).to_csv(OUT/f'expanded_{kind}_audit.csv',index=False)
        print(name,seed,kind,round(time.perf_counter()-start,1),flush=True)
    (OUT/f'expanded_{kind}_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,settings=30,rows=len(rows),candidate_rows=len(audit)),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);run(ap.parse_args().model)
