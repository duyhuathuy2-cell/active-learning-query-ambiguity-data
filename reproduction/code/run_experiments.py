"""Prospectively configured audit and closed-loop replay. No test-label selection."""
import environment
from pathlib import Path
from environment import ROOT,TABPFN_CACHE
import json,time,hashlib,sys,platform,argparse,os
import numpy as np
import pandas as pd
import torch, sklearn
from sklearn.datasets import load_breast_cancer,load_iris,load_wine,load_digits,make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from method import scores

OUT=ROOT.parent/'data/results/benchmark'; OUT.mkdir(exist_ok=True)
CACHE=OUT/'cache'; CACHE.mkdir(exist_ok=True)
torch.set_num_threads(4)

def datasets():
    bc,ir,wi,di=load_breast_cancer(),load_iris(),load_wine(),load_digits()
    moon=make_moons(n_samples=600,noise=.22,random_state=731)
    rng=np.random.default_rng(819); X=rng.uniform(-2,2,(600,5))
    p=1/(1+np.exp(-(2*X[:,0]-X[:,1]+1.3*X[:,2]*X[:,3])))
    y=rng.binomial(1,p)
    return {'breast_cancer':(bc.data,bc.target),'iris_binary':(ir.data[ir.target>0],ir.target[ir.target>0]-1),
      'wine_binary':(wi.data[wi.target<2],wi.target[wi.target<2]),'digits_3_8':(di.data[np.isin(di.target,[3,8])],(di.target[np.isin(di.target,[3,8])]==8).astype(int)),
      'moons':moon,'mechanistic':(X,y)}

def split(y,seed):
    rng=np.random.default_rng(1000+seed)
    per=[rng.permutation(np.flatnonzero(y==k)) for k in (0,1)]
    train=np.r_[per[0][:8],per[1][:8]]
    rest=rng.permutation(np.setdiff1d(np.arange(len(y)),train))
    return train,rest[:8],rest[8:56]

def model(kind):
    device=os.environ.get('KBS_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    if kind.startswith('tabpfn'):
        n=int(kind.split('_')[1])
        return TabPFNClassifier(n_estimators=n,device=device,random_state=0,n_jobs=1,
            softmax_temperature=.9,inference_precision=torch.float32,
            model_path=os.environ.get('TABPFN_CHECKPOINT') or str(next(TABPFN_CACHE.glob('*.ckpt'))))
    if kind=='tabicl_8':
        from tabicl import TabICLClassifier
        local=ROOT.parent/'models'/'tabicl'/'tabicl-classifier-v2-20260212.ckpt'
        checkpoint=os.environ.get('TABICL_CHECKPOINT') or (str(local) if local.exists() else None)
        return TabICLClassifier(n_estimators=8,device=device,random_state=0,n_jobs=4,use_amp=False,verbose=False,model_path=checkpoint)
    if kind=='logistic': return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0))
    if kind=='forest': return RandomForestClassifier(n_estimators=100,min_samples_leaf=2,random_state=0,n_jobs=4)
    raise ValueError(kind)

def state(name,kind,seed,X,y,train,pool,target,selected=()):
    selected=tuple(sorted(selected))
    key=f'{name}_{kind}_{seed}_'+('-'.join(map(str,selected)) or 'initial')
    path=CACHE/(key+'.npz')
    if path.exists():
        with np.load(path) as z: return {k:z[k] for k in z.files}
    context=np.r_[train,pool[list(selected)]].astype(int)
    remain=np.array([i for i in range(len(pool)) if i not in selected])
    query=pool[remain]; nq=len(query)
    clf=model(kind)
    clf.fit(X[context],y[context])
    base=clf.predict_proba(X[np.r_[query,target]])[:,1].astype(float)
    branches=np.empty((nq,2,len(target)))
    for i in range(nq):
        for label in (0,1):
            added=tuple(sorted(selected+(int(remain[i]),)))
            branch_context=np.r_[train,pool[list(added)]].astype(int)
            branch_labels=y[branch_context].copy()
            branch_labels[np.flatnonzero(branch_context==query[i])]=label
            clf.fit(X[branch_context],branch_labels)
            branches[i,label]=clf.predict_proba(X[target])[:,1]
    z=dict(base=base,branches=branches,remain=remain,train=context,pool=pool,target=target)
    np.savez_compressed(path,**z)
    return z

def audit(kinds,seeds):
    rows=[]; sums=[]; start=time.perf_counter()
    for name,(X,y) in datasets().items():
      for kind in kinds:
       for seed in range(seeds):
        tr,pool,target=split(y,seed); z=state(name,kind,seed,X,y,tr,pool,target)
        s=scores(z['base'],z['branches'],len(pool))
        actual=np.array([np.mean((z['branches'][i,y[pool[i]]]-y[target])**2) for i in range(len(pool))])
        selections={k:int(np.argmax(s[k])) for k in ['raw','kl','lower','entropy','endpoint_lo','endpoint_hi']}
        selections['minimax']=int(np.argmin(s['regret']))
        for i in range(len(pool)):
            row=dict(dataset=name,model=kind,seed=seed,position=i,query_index=int(pool[i]),pa=float(z['base'][i]),actual_brier=actual[i])
            row.update({k:float(v[i]) for k,v in s.items()}); rows.append(row)
        summary=dict(dataset=name,model=kind,seed=seed,base_brier=float(np.mean((z['base'][8:]-y[target])**2)),
          certified=int(s['certified'].sum()),possible=int(s['possible'].sum()),mean_tower=float(s['tower'].mean()),mean_width=float((s['upper']-s['lower']).mean()),
          raw_kl_flip=selections['raw']!=selections['kl'],endpoint_flip=selections['endpoint_lo']!=selections['endpoint_hi'],random_brier=float(actual.mean()),oracle_brier=float(actual.min()))
        summary.update({k+'_brier':float(actual[i]) for k,i in selections.items()})
        sums.append(summary)
        pd.DataFrame(rows).to_csv(OUT/'audit_candidates.csv',index=False)
        pd.DataFrame(sums).to_csv(OUT/'audit_cases.csv',index=False)
        print(f'AUDIT {name} {kind} {seed} width={summary["mean_width"]:.5f} cert={summary["certified"]}',flush=True)
    return time.perf_counter()-start

def sequential(kinds,seeds):
    previous=OUT/'sequential.csv'
    rows=pd.read_csv(previous).query('model not in @kinds').to_dict('records') if previous.exists() else []
    start=time.perf_counter()
    policies=['raw','kl','lower','minimax','entropy','random','certify_entropy']
    for name,(X,y) in datasets().items():
     for kind in kinds:
      for seed in range(seeds):
       tr,pool,target=split(y,seed)
       for policy in policies:
        selected=[]; rng=np.random.default_rng(70000+seed)
        for step in range(5):
         z=state(name,kind,seed,X,y,tr,pool,target,selected); nq=len(z['remain'])
         s=scores(z['base'],z['branches'],nq)
         cert=np.flatnonzero(s['certified']); abstain=False
         if policy=='random': choice=int(rng.integers(nq))
         elif policy=='minimax': choice=int(np.argmin(s['regret']))
         elif policy=='certify_entropy':
            abstain=len(cert)==0
            choice=int(cert[0]) if len(cert) else int(np.argmax(s['entropy']))
         else: choice=int(np.argmax(s[policy]))
         pos=int(z['remain'][choice]); selected.append(pos)
         pred=z['branches'][choice,y[pool[pos]]]
         rows.append(dict(dataset=name,model=kind,seed=seed,policy=policy,budget=step+1,selected_positions=json.dumps(selected),
           brier=float(np.mean((pred-y[target])**2)),error=float(np.mean((pred>=.5)!=y[target])),
           abstained=abstain,certified=len(cert)>0,candidate_count=nq,internal_regret=float(s['regret'][choice])))
        pd.DataFrame(rows).to_csv(OUT/'sequential.csv',index=False)
       print(f'SEQUENTIAL {name} {kind} {seed} cached_states={len(list(CACHE.glob("*.npz")))}',flush=True)
    return time.perf_counter()-start

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['audit','sequential'],required=True);ap.add_argument('--models',nargs='+',required=True);ap.add_argument('--seeds',type=int,default=20); args=ap.parse_args()
    config=dict(mode=args.mode,models=args.models,seeds=args.seeds,initial_labels=16,pool=8,target=48,budgets=[1,2,3,4,5],date='2026-09-30',
      prospective='Configuration written before executing this batch. Not externally preregistered.',python=sys.version,executable=Path(sys.executable).name,
      torch=torch.__version__,sklearn=sklearn.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    config_path=OUT/(args.mode+'_'+'-'.join(args.models)+'_config.json')
    config_path.write_text(json.dumps(config,indent=2))
    elapsed=audit(args.models,args.seeds) if args.mode=='audit' else sequential(args.models,args.seeds)
    config['elapsed_seconds']=elapsed
    config_path.write_text(json.dumps(config,indent=2))
    print('COMPLETE',elapsed,flush=True)
