"""Conformal calibration of an entire pairwise-constrained joint neighbourhood."""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time
from method import interval,entropy
OUT=ROOT.parent/'data/results/multiclass_and_calibration';OUT.mkdir(exist_ok=True)

def one(seed,k,noise,marginal_noise):
    rng=np.random.default_rng(500000+seed); pi=rng.dirichlet(np.ones(16))
    theta=rng.beta(.45,.45,size=(16,k+32))
    a=pi@theta[:,:k];b=pi@theta[:,k:]
    t=np.einsum('h,hi,hj->ij',pi,theta[:,:k],theta[:,k:])
    q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None])
    true=(entropy(b)[None,:]-(1-a[:,None])*entropy(q0)-a[:,None]*entropy(q1)).mean(1)
    q=np.clip(np.stack([q0,q1],axis=1)+noise*rng.normal(size=(k,2,32)),1e-6,1-1e-6)
    base=np.r_[a,b]
    if marginal_noise:base=np.clip(base+noise*rng.normal(size=base.shape),1e-6,1-1e-6)
    aa=base[:k,None];bb=base[k:][None,:]
    L,U,lo,hi=interval(aa,bb,q[:,0],q[:,1])
    p=np.stack([1-a[:,None]-b[None,:]+t,b[None,:]-t,a[:,None]-t,t])
    roots=np.stack([p[0]-1+aa+bb,bb-p[1],aa-p[2],p[3]])
    # L1 minimisers form the median interval; any clipped median is optimal.
    best=np.clip(np.median(roots,axis=0),lo,hi)
    pair_distance=.5*np.abs(best[None,:,:]-roots).sum(0)
    return float(pair_distance.max()),L.mean(1),U.mean(1),true

def continuity(rho):
    # Audenaert/Fannes finite alphabet continuity; global maximum after 1-1/d.
    d2=min(rho,.5);d4=min(rho,.75)
    return float(2*entropy(d2)+entropy(d4)+d4*np.log2(3))

def run():
    rows=[];qrows=[];start=time.perf_counter()
    for k in [8,32,64]:
      for noise in [.01,.03,.1]:
       for marginal_noise in [False,True]:
        calib=[one(seed,k,noise,marginal_noise)[0] for seed in range(500)]
        radius=float(np.sort(calib)[475]);omega=continuity(radius)
        qrows.append(dict(pool=k,noise=noise,marginal_noise=marginal_noise,radius=radius,information_allowance=omega))
        for seed in range(500,2500):
            distance,L,U,true=one(seed,k,noise,marginal_noise)
            lo=np.maximum(0,L-omega);hi=np.minimum(1,U+omega)
            top=int(np.argmax(hi));other=np.full(k,hi[top]);other[top]=np.partition(hi,-2)[-2]
            regret=np.maximum(0,other-lo);pick=int(np.argmin(regret))
            actual=true.max()-true[pick]
            rows.append(dict(pool=k,noise=noise,marginal_noise=marginal_noise,seed=seed,joint_covered=distance<=radius+1e-12,score_covered=bool(np.all((lo<=true+1e-12)&(hi>=true-1e-12))),certified=bool(lo[pick]>other[pick]+1e-9),true_regret=float(actual),regret_bound=float(regret[pick]),bound_violated=actual>regret[pick]+1e-9,mean_width=float(np.mean(hi-lo))))
        print('joint',k,noise,marginal_noise,round(time.perf_counter()-start,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'joint_calibration_test.csv',index=False)
    pd.DataFrame(qrows).to_csv(OUT/'joint_calibration_quantiles.csv',index=False)
    df.groupby(['pool','noise','marginal_noise']).agg(n=('seed','size'),joint_coverage=('joint_covered','mean'),score_coverage=('score_covered','mean'),certified=('certified','sum'),true_regret=('true_regret','mean'),mean_width=('mean_width','mean')).reset_index().to_csv(OUT/'joint_calibration_summary.csv',index=False)
    (OUT/'joint_calibration_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,rows=len(rows)),indent=2))
if __name__=='__main__':run()
