import environment
from environment import ROOT
import argparse,json,time
import numpy as np,pandas as pd
from scipy.special import xlogy
from sklearn.datasets import load_iris,load_wine,load_digits
from multiclass_method import bounds
from run_experiments import model
OUT=ROOT.parent/'data/results/multiclass_and_calibration';OUT.mkdir(exist_ok=True)
CACHE=OUT/'multiclass_cache';CACHE.mkdir(exist_ok=True)

def entropy(p):return -np.sum(xlogy(p,p),axis=-1)/np.log(2)

def run(kind):
    rows=[];candidates=[];start=time.perf_counter()
    data={'iris':load_iris(),'wine':load_wine(),'digits_0_1_2':load_digits()}
    for name,d in data.items():
      X,y=d.data,d.target;mask=y<3;X=X[mask];y=y[mask]
      for seed in range(5):
        rng=np.random.default_rng(430000+seed)
        tr=np.concatenate([rng.permutation(np.flatnonzero(y==c))[:8] for c in range(3)])
        rest=rng.permutation(np.setdiff1d(np.arange(len(y)),tr));pool=rest[:8];target=rest[8:16]
        path=CACHE/f'{name}_{kind}_{seed}.npz'
        if path.exists():
            z=np.load(path);base=z['base'];branches=z['branches']
        else:
            clf=model(kind);clf.fit(X[tr],y[tr]);base=clf.predict_proba(X[np.r_[pool,target]])
            branches=np.empty((8,3,8,3))
            for i in range(8):
              for label in range(3):
                clf.fit(X[np.r_[tr,pool[i]]],np.r_[y[tr],label]);branches[i,label]=clf.predict_proba(X[target])
            np.savez_compressed(path,base=base,branches=branches,train=tr,pool=pool,target=target)
        # Softmax float32 sums can differ from one by ~1e-7. The transport
        # constraints require proper marginals: normalise in float64.
        base=np.asarray(base,dtype=np.float64);base=base/base.sum(-1,keepdims=True)
        branches=np.asarray(branches,dtype=np.float64);branches=branches/branches.sum(-1,keepdims=True)
        raw=entropy(base[8:])[None,:]-np.einsum('ic,icj->ij',base[:8],entropy(branches))
        L=np.empty((8,8));U=np.empty((8,8));quality=[]
        for i in range(8):
          for j in range(8):
            L[i,j],U[i,j],qa=bounds(base[i],base[8+j],branches[i,:,j,:]);quality.append(qa)
        lower=L.mean(1);upper=U.mean(1);possible=upper>=lower.max()-1e-9
        cert=np.array([lower[i]>np.delete(upper,i).max()+1e-9 for i in range(8)])
        costs=[v['min_cost'] for v in quality]
        actual=np.array([np.mean(np.sum((branches[i,y[pool[i]]]-np.eye(3)[y[target]])**2,axis=1)) for i in range(8)])
        for i in range(8):candidates.append(dict(dataset=name,model=kind,seed=seed,candidate=i,lower=lower[i],upper=upper[i],raw=raw[i].mean(),actual_brier=actual[i],possible=bool(possible[i]),certified=bool(cert[i])))
        rows.append(dict(dataset=name,model=kind,seed=seed,mean_width=float((upper-lower).mean()),possible_count=int(possible.sum()),certified=int(cert.sum()),raw_possible=bool(possible[np.argmax(raw.mean(1))]),max_lower_gap=max(v['lower_gap'] for v in quality),max_vertex_residual=max(v['constraint_residual'] for v in quality),max_cost_error=max(abs(v['min_cost']-v['observed_min_cost']) for v in quality),mean_cost=float(np.mean(costs))))
        pd.DataFrame(rows).to_csv(OUT/f'multiclass_{kind}_cases.csv',index=False)
        pd.DataFrame(candidates).to_csv(OUT/f'multiclass_{kind}_candidates.csv',index=False)
        print('multiclass',kind,name,seed,round(time.perf_counter()-start,1),flush=True)
    (OUT/f'multiclass_{kind}_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,settings=len(rows),pairs=len(rows)*64),indent=2))

def controls():
    rng=np.random.default_rng(440000);rows=[]
    for i in range(100):
        a=rng.dirichlet(np.ones(3));b=rng.dirichlet(np.ones(3));q=rng.dirichlet(np.ones(3),size=3)
        L,U,qa=bounds(a,b,q);assert -1e-8<=L<=U+1e-8
        assert qa['constraint_residual']<1e-8 and abs(qa['observed_min_cost']-qa['min_cost'])<1e-8
        rows.append(dict(seed=i,lower=L,upper=U,**qa))
    pd.DataFrame(rows).to_csv(OUT/'multiclass_controls.csv',index=False);print('100 multiclass controls passed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model');ap.add_argument('--controls',action='store_true');args=ap.parse_args()
    controls() if args.controls else run(args.model)
