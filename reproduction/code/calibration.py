"""Simultaneous score calibration; no assertion of probability-model coverage."""
import environment
from environment import ROOT
import json,time
import numpy as np
import pandas as pd
from method import scores,entropy
OUT=ROOT.parent/'data/results/multiclass_and_calibration';OUT.mkdir(exist_ok=True)

def system(seed,k,noise,marginal_noise):
    rng=np.random.default_rng(500000+seed)
    pi=rng.dirichlet(np.ones(16))
    # Independent, heterogeneous finite mechanisms, with no designated winner.
    theta=rng.beta(.45,.45,size=(16,k+32))
    a=pi@theta[:,:k]; b=pi@theta[:,k:]
    t=np.einsum('h,hi,hj->ij',pi,theta[:,:k],theta[:,k:])
    q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None])
    true=(entropy(b)[None,:]-(1-a[:,None])*entropy(q0)-a[:,None]*entropy(q1)).mean(1)
    branches=np.stack([q0,q1],axis=1)
    branches=np.clip(branches+noise*rng.normal(size=branches.shape),1e-6,1-1e-6)
    base=np.r_[a,b]
    if marginal_noise:base=np.clip(base+noise*rng.normal(size=base.shape),1e-6,1-1e-6)
    s=scores(base,branches,k)
    return s,true

def run():
    start=time.perf_counter();rows=[];quantiles=[]
    for k in [8,32,64]:
      for noise in [.01,.03,.1]:
       for marginal_noise in [False,True]:
        collected={key:[] for key in ['tv','kl','raw']}
        for seed in range(500):
            s,true=system(seed,k,noise,marginal_noise)
            for key in collected:
                L=s['lower'] if key=='tv' else s[key];U=s['upper'] if key=='tv' else s[key]
                collected[key].append(float(np.maximum.reduce([L-true,true-U,np.zeros(k)]).max()))
        q={key:float(np.sort(v)[int(np.ceil(501*.95))-1]) for key,v in collected.items()}
        for key,v in q.items():quantiles.append(dict(pool=k,noise=noise,marginal_noise=marginal_noise,method=key,quantile=v,rank=int(np.ceil(501*.95)),n=500,alpha=.05))
        for seed in range(500,2500):
            s,true=system(seed,k,noise,marginal_noise)
            for key in collected:
                L=s['lower'] if key=='tv' else s[key];U=s['upper'] if key=='tv' else s[key]
                for calibrated in [False,True]:
                    radius=q[key] if calibrated else 0.
                    lo=np.maximum(0,L-radius);hi=np.minimum(1,U+radius)
                    # Incoherent raw point may fall outside [0,1]: retain a
                    # nonempty conservative score interval after expansion.
                    lo=np.minimum(lo,1);hi=np.maximum(hi,0)
                    top=int(np.argmax(hi)); competitors=np.full(k,hi[top]);competitors[top]=np.partition(hi,-2)[-2]
                    reg=np.maximum(0,competitors-lo)
                    pick=int(np.argmin(reg))
                    cert=lo[pick]>np.max(np.delete(hi,pick))+1e-9
                    true_reg=float(true.max()-true[pick]);bound=float(reg[pick])
                    coverage=bool(np.all((true>=lo-1e-12)&(true<=hi+1e-12)))
                    rows.append(dict(pool=k,noise=noise,marginal_noise=marginal_noise,seed=seed,method=key,calibrated=calibrated,coverage=coverage,certified=bool(cert),certificate_wrong=bool(cert and true_reg>1e-9),true_regret=true_reg,regret_bound=bound,bound_violated=bool(true_reg>bound+1e-9),possible_count=int(np.sum(hi>=lo.max()-1e-9)),mean_width=float(np.mean(hi-lo))))
        pd.DataFrame(rows).to_csv(OUT/'calibration_test.csv',index=False)
        pd.DataFrame(quantiles).to_csv(OUT/'calibration_quantiles.csv',index=False)
        print(k,noise,marginal_noise,round(time.perf_counter()-start,1),flush=True)
    df=pd.DataFrame(rows)
    group=['pool','noise','marginal_noise','method','calibrated']
    summary=df.groupby(group).agg(n=('seed','size'),coverage=('coverage','mean'),certified=('certified','sum'),certificate_wrong=('certificate_wrong','sum'),true_regret=('true_regret','mean'),regret_bound=('regret_bound','mean'),bound_violated=('bound_violated','mean'),possible_count=('possible_count','mean'),mean_width=('mean_width','mean')).reset_index()
    summary.to_csv(OUT/'calibration_summary.csv',index=False)
    (OUT/'calibration_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,independent_calibration=500,independent_test=2000,rows=len(rows),unit='entire independently drawn system',coverage='simultaneous candidate objective intervals, not joint-distribution containment'),indent=2))

if __name__=='__main__':run()
