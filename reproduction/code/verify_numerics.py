import environment
from environment import ROOT
import numpy as np,json
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
from method import kl_repair,interval

def main():
    rng=np.random.default_rng(9156);err=[];radius_err=[]
    for i in range(500):
        a,b,q0,q1=rng.uniform(.01,.99,4)
        r0,r1=kl_repair(a,np.array([b]),np.array([q0]),np.array([q1]))
        flo,fhi=max(0,a+b-1),min(a,b)
        def divergence(t):
            x=(b-t)/(1-a);y=t/a
            def d(p,q):return xlogy(p,p/q)+xlogy(1-p,(1-p)/(1-q))
            return (1-a)*d(x,q0)+a*d(y,q1)
        res=minimize_scalar(divergence,bounds=(flo,fhi),method='bounded',options={'xatol':1e-13})
        err.append(abs(a*r1[0]-res.x))
        for eps in [0,.001,.01,.1]:
            L,U,lo,hi=interval(a,np.array(b),np.array(q0),np.array(q1),eps)
            costmin=abs(b-(1-a)*q0-a*q1)
            for t in [float(lo),float(hi)]:
                radius_err.append(max(0,abs(t-a*q1)+abs(t-(b-(1-a)*q0))-costmin-eps))
    assert max(err)<1e-7 and max(radius_err)<1e-12
    observed=[]
    for p in (ROOT.parent/'data/results/benchmark'/'cache').glob('*_initial.npz'):
        with np.load(p) as z:
            observed.extend(z['base'][:8].tolist())
            assert not set(z['train']).intersection(z['pool'])
            assert not set(z['target']).intersection(np.r_[z['train'],z['pool']])
            assert z['branches'].shape==(8,2,48)
    result=dict(kl_independent_optimizer_cases=500,max_kl_joint_difference=max(err),
      max_radius_cost_violation=max(radius_err),candidate_probability_min=min(observed),candidate_probability_max=max(observed),
      all_initial_splits_disjoint=True,n_initial_cache_files=len(list((ROOT.parent/'data/results/benchmark'/'cache').glob('*_initial.npz'))))
    (ROOT.parent/'data/results/benchmark'/'numerical_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
