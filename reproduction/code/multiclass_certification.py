import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time
from multiclass_verified import verified_bounds
R=ROOT.parent/'data/results/multiclass_and_calibration'
def run():
    rows=[];cases=[];start=time.perf_counter()
    for model in ['tabpfn_8','logistic']:
      old=pd.read_csv(R/f'multiclass_{model}_candidates.csv')
      for name in ['iris','wine','digits_0_1_2']:
       for seed in range(5):
        with np.load(R/'multiclass_cache'/f'{name}_{model}_{seed}.npz') as z:base=z['base'];branches=z['branches']
        L=np.empty((8,8));U=np.empty((8,8));nv=[]
        for i in range(8):
          for j in range(8):L[i,j],U[i,j],meta=verified_bounds(base[i],base[8+j],branches[i,:,j,:]);nv.append(meta['exact_vertices'])
        lo=np.zeros(8);hi=np.zeros(8)
        for j in range(8):
            lo=np.nextafter(lo+L[:,j],-np.inf);hi=np.nextafter(hi+U[:,j],np.inf)
        lo=np.maximum(0,np.nextafter(lo/8,-np.inf));hi=np.nextafter(hi/8,np.inf)
        g=old[(old.dataset==name)&(old.seed==seed)].sort_values('candidate');raw_pick=int(np.argmax(g.raw.to_numpy()))
        for i in range(8):
            competitor=np.delete(hi,i).max();cert=lo[i]>competitor+1e-9;exclusion=hi[i]<lo.max()-1e-9
            oldrow=old[(old.dataset==name)&(old.seed==seed)&(old.candidate==i)].iloc[0]
            rows.append(dict(dataset=name,model=model,seed=seed,candidate=i,lower_verified=lo[i],upper_verified=hi[i],strict_margin=float(lo[i]-competitor),excluded_margin=float(lo.max()-hi[i]),certified=bool(cert),raw_selected=bool(i==raw_pick),excluded=bool(exclusion),previous_lower=float(oldrow.lower),previous_upper=float(oldrow.upper)))
        certs=np.array([lo[i]>np.delete(hi,i).max()+1e-9 for i in range(8)])
        cases.append(dict(dataset=name,model=model,seed=seed,certified=int(certs.sum()),possible_count=int(np.sum(hi>=lo.max()-1e-9)),raw_excluded=bool(hi[raw_pick]<lo.max()-1e-9),maximum_exact_vertices=max(nv),mean_width=float(np.mean(hi-lo))))
        pd.DataFrame(rows).to_csv(R/'multiclass_verified_candidates.csv',index=False);pd.DataFrame(cases).to_csv(R/'multiclass_verified_cases.csv',index=False)
        print('Verified',model,name,seed,round(time.perf_counter()-start,1),flush=True)
    (R/'multiclass_certification_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,pairs=1920,exact_rational_membership=True,outward_interval_logarithms=True,mpmath_decimal_precision=60),indent=2))
if __name__=='__main__':run()
