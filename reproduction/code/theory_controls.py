"""Numerical stress tests and a global-compatibility linear program."""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,itertools
from scipy.optimize import linprog
from method import interval,entropy,scores
OUT=ROOT.parent/'data/results/benchmark';OUT.mkdir(exist_ok=True)

def joint_lp(a,b,q0,q1,local_only=False):
    na,nb=len(a),len(b); states=np.array(list(itertools.product([0,1],repeat=na+nb)))
    n=len(states); pairs=[(i,j) for i in range(na) for j in range(nb)]; npair=len(pairs)
    # Joint atom probabilities followed by two absolute-value residuals per pair.
    objective=np.r_[np.zeros(n),np.ones(2*npair)]
    eq=[np.r_[np.ones(n),np.zeros(2*npair)]]; rhs=[1.]
    for i,p in enumerate(np.r_[a,b]): eq.append(np.r_[states[:,i],np.zeros(2*npair)]);rhs.append(p)
    ub=[]; lim=[]
    for k,(i,j) in enumerate(pairs):
        t=(states[:,i]*states[:,na+j]).astype(float)
        for h,v in enumerate([a[i]*q1[i,j], b[j]-(1-a[i])*q0[i,j]]):
            row=np.r_[t,np.zeros(2*npair)];row[n+2*k+h]=-1;ub.append(row);lim.append(v)
            row=np.r_[-t,np.zeros(2*npair)];row[n+2*k+h]=-1;ub.append(row);lim.append(-v)
    if local_only:
      L,U,lo,hi=interval(a[:,None],b[None,:],q0,q1)
      for k,(i,j) in enumerate(pairs):
        t=(states[:,i]*states[:,na+j]).astype(float)
        ub.extend([np.r_[t,np.zeros(2*npair)],np.r_[-t,np.zeros(2*npair)]])
        lim.extend([hi[i,j],-lo[i,j]])
    res=linprog(objective,A_ub=ub,b_ub=lim,A_eq=eq,b_eq=rhs,bounds=(0,None),method='highs')
    return res

def main():
    rng=np.random.default_rng(81239); n=10000
    a,b,q0,q1=rng.uniform(.001,.999,(4,n))
    low,high,lo,hi=interval(a,b,q0,q1)
    mincost=np.abs(b-(1-a)*q0-a*q1)
    c0=np.abs(lo-a*q1)+np.abs(lo-(b-(1-a)*q0))
    c1=np.abs(hi-a*q1)+np.abs(hi-(b-(1-a)*q0))
    maxcost=float(max(np.max(abs(c0-mincost)),np.max(abs(c1-mincost))))
    u=np.linspace(0,1,301)[:,None];t=lo+(hi-lo)*u
    values=entropy(b)-(1-a)*entropy((b-t)/(1-a))-a*entropy(t/a)
    enclosure=float(max(np.max(low-values),np.max(values-high)))
    assert maxcost<1e-10 and enclosure<1e-9
    # A finite Bayesian latent-state model is an exact tower-property control.
    prior=rng.dirichlet(np.ones(12)); pred=rng.uniform(.02,.98,(12,10))
    base=prior@pred; branches=np.empty((4,2,6))
    for i in range(4):
      for label in (0,1):
        w=prior*(pred[:,i] if label else 1-pred[:,i]);w/=w.sum();branches[i,label]=w@pred[:,4:]
    s=scores(base,branches,4)
    assert s['tower'].max()<1e-12 and (s['upper']-s['lower']).max()<1e-12
    control_tower=float(s['tower'].max());control_width=float((s['upper']-s['lower']).max())
    # Four pairwise coherent marginals: three perfect equalities and one inequality.
    aa=bb=np.array([.5,.5]); qq1=np.array([[1.,1.],[1.,0.]]);qq0=1-qq1
    impossible=joint_lp(aa,bb,qq0,qq1,local_only=True)
    relaxed=joint_lp(aa,bb,qq0,qq1)
    assert not impossible.success and relaxed.success and abs(relaxed.fun-1)<1e-9
    # A ranking reversal under the same margins and minimum repair costs.
    aa=np.array([.5,.5]);bb=np.array([.5]); q0=np.array([[.1],[.8]]);q1=np.array([[.6],[.6]])
    low,high,lo,hi=interval(aa[:,None],bb[None,:],q0,q1)
    assert low[0,0]<high[1,0] and low[1,0]<high[0,0]
    worlds=[];world_values=[]
    for world,ts in enumerate([[.3,.1],[.45,.3]],start=1):
        probs=[];states=np.array(list(itertools.product([0,1],repeat=3)))
        for A1,A2,B in states:
            ps=np.array(ts)/.5 if B else (.5-np.array(ts))/.5
            atom=.5*np.prod([ps[0] if A1 else 1-ps[0],ps[1] if A2 else 1-ps[1]])
            probs.append(atom);worlds.append(dict(world=world,A1=A1,A2=A2,B=B,probability=atom))
        probs=np.array(probs)
        assert np.max(np.abs(probs@states-.5))<1e-12 and abs(probs.sum()-1)<1e-12
        information=[]
        for i,t in enumerate(ts):
            assert abs(np.sum(probs*states[:,i]*states[:,2])-t)<1e-12
            cost=abs(t-.5*[.6,.6][i])+abs(t-(.5-.5*[.1,.8][i]))
            assert abs(cost-[.15,.2][i])<1e-12
            information.append(float(entropy(.5)-.5*entropy((.5-t)/.5)-.5*entropy(t/.5)))
        world_values.append(information)
    assert np.argmax(world_values[0])!=np.argmax(world_values[1])
    pd.DataFrame(worlds).to_csv(OUT/'globally_coherent_ranking_worlds.csv',index=False)
    rows=[]
    for trial in range(300):
        aa=rng.uniform(.05,.95,3);bb=rng.uniform(.05,.95,3);q0=rng.uniform(.01,.99,(3,3));q1=rng.uniform(.01,.99,(3,3))
        local=float(np.abs(bb[None,:]-(1-aa[:,None])*q0-aa[:,None]*q1).sum())
        g=joint_lp(aa,bb,q0,q1); f=joint_lp(aa,bb,q0,q1,True)
        assert g.success and g.fun>=local-1e-8
        rows.append(dict(trial=trial,local_cost=local,global_cost=g.fun,excess=g.fun-local,local_minima_globally_feasible=f.success))
    pd.DataFrame(rows).to_csv(OUT/'global_lp_controls.csv',index=False)
    # Known coherent simulator with deterministic probability perturbations.
    sims=[]
    for seed in range(200):
        rr=np.random.default_rng(20000+seed);prior=rr.dirichlet(np.ones(16));prob=rr.uniform(.03,.97,(16,16));base=prior@prob
        true=np.empty((8,2,8))
        for i in range(8):
          for label in (0,1):
            w=prior*(prob[:,i] if label else 1-prob[:,i]);w/=w.sum();true[i,label]=w@prob[:,8:]
        ss=scores(base,true,8); truth=ss['raw'];best=int(np.argmax(truth))
        noise=rr.normal(size=true.shape)
        for scale in [0,.01,.03,.05,.1,.2]:
            pert=np.clip(true+scale*noise,.0001,.9999);s=scores(base,pert,8)
            cert=np.flatnonzero(s['certified']);raw=int(np.argmax(s['raw']));mm=int(np.argmin(s['regret']))
            sims.append(dict(seed=seed,scale=scale,raw_regret=truth[best]-truth[raw],minimax_regret=truth[best]-truth[mm],
                certified=len(cert)>0,certified_correct=(int(cert[0])==best) if len(cert) else np.nan,
                width=float((s['upper']-s['lower']).mean()),tower=float(s['tower'].mean())))
    pd.DataFrame(sims).to_csv(OUT/'perturbation_simulation.csv',index=False)
    report=dict(n_random_cases=n,max_minimum_cost_error=maxcost,max_enclosure_violation=enclosure,
      bayes_max_tower_error=control_tower,bayes_max_interval_width=control_width,
      bayes_tower_metric='Maximum across candidates of target-mean absolute tower error',
      bayes_width_metric='Maximum candidate acquisition-interval width after target averaging',
      global_cycle_feasible=impossible.success,global_cycle_minimum_cost=relaxed.fun,
      ranking_counterexample={'a':[.5,.5],'b':[.5],'q0':[.1,.8],'q1':[.6,.6]},
      global_ranking_world_values=world_values,
      lp_trials=300,globally_infeasible_local_optima=int(sum(not r['local_minima_globally_feasible'] for r in rows)))
    (OUT/'theory_controls.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
