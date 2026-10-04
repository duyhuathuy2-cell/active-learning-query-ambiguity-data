"""Restricted-support inner completion: feasible => a checked global witness.

Failure is inconclusive. It is never converted into a proof of infeasibility.
"""
import environment
from environment import ROOT
import json,time,warnings
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from scipy.optimize import linprog,OptimizeWarning
from method import interval

OUT=ROOT.parent/'data/results/policy_controls';OUT.mkdir(exist_ok=True)
DESIGN={'candidates':[8,16,32,64],'targets':16,'systems_per_k':5,'atom_budgets':[256,1024],
 'seed_offset':950000,'perturbation':.03,'time_limit_seconds':5,'fixed_before_execution':True,
 'scope':'inner feasibility; unsuccessful search is inconclusive; independent latent-system draws'}
path=OUT/'scalable_completion_design.json'
if path.exists():assert json.loads(path.read_text())==DESIGN
else:path.write_text(json.dumps(DESIGN,indent=2))

def template(states,m):
 n,k=states.shape;v=n*(m+1);er=[];ec=[];ev=[]
 def eq(row,cols):er.extend([row]*len(cols));ec.extend(cols);ev.extend(np.ones(len(cols)))
 eq(0,np.arange(n))
 for i in range(k):eq(i+1,np.flatnonzero(states[:,i]))
 for j in range(m):eq(k+1+j,np.arange(n*(j+1),n*(j+2)))
 E=coo_matrix((ev,(er,ec)),shape=(k+1+m,v)).tocsr();ur=[];uc=[];uv=[];row=0
 for j in range(m):
  for ss in range(n):ur.extend([row,row]);uc.extend([ss,n*(j+1)+ss]);uv.extend([-1,1]);row+=1
  for i in range(k):
   cols=n*(j+1)+np.flatnonzero(states[:,i])
   for sign in [1,-1]:ur.extend([row]*len(cols));uc.extend(cols);uv.extend([sign]*len(cols));row+=1
 U=coo_matrix((uv,(ur,uc)),shape=(row,v)).tocsr();return E,U

def run():
 rows=[];m=16
 for k in DESIGN['candidates']:
  for seed in range(5):
   rng=np.random.default_rng(950000+k*100+seed);pi=rng.dirichlet(np.ones(16));theta=rng.beta(.8,.8,(16,k+m))
   a=pi@theta[:,:k];b=pi@theta[:,k:];t=np.einsum('h,hi,hj->ij',pi,theta[:,:k],theta[:,k:])
   q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None]);branches=np.clip(np.stack([q0,q1],1)+.03*rng.normal(size=(k,2,m)),1e-6,1-1e-6)
   _,_,lo,hi=interval(a[:,None],b[None,:],branches[:,0],branches[:,1])
   np.savez_compressed(OUT/f'scalable_input_{k}_{seed}.npz',a=a,b=b,lo=lo,hi=hi)
   for budget in DESIGN['atom_budgets']:
    start=time.perf_counter();states=np.unique(rng.binomial(1,a,size=(budget,k)),axis=0);n=len(states)
    E,U=template(states,m);bound=np.zeros(U.shape[0])
    for j in range(m):
     off=j*(n+2*k)+n;bound[off:off+2*k:2]=hi[:,j];bound[off+1:off+2*k:2]=-lo[:,j]
    build=time.perf_counter()-start;t0=time.perf_counter()
    with warnings.catch_warnings():
     warnings.simplefilter('ignore',OptimizeWarning)
     res=linprog(np.zeros(n*(m+1)),A_ub=U,b_ub=bound,A_eq=E,b_eq=np.r_[1,a,b],bounds=(0,None),method='highs',
      options={'time_limit':5,'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    residual=np.nan
    if res.success:
     residual=max(float(np.max(abs(E@res.x-np.r_[1,a,b]))),float(max(0,np.max(U@res.x-bound))),float(max(0,-res.x.min())))
     assert residual<1e-7
     np.savez_compressed(OUT/f'scalable_witness_{k}_{seed}_{budget}.npz',states=states,weights=res.x[:n],target_joint=res.x[n:].reshape(m,n))
    rows.append(dict(candidates=k,targets=m,seed=seed,atom_budget=budget,unique_atoms=n,variables=n*(m+1),
     construction_seconds=build,solve_seconds=time.perf_counter()-t0,sparse_bytes=sum(A.data.nbytes+A.indices.nbytes+A.indptr.nbytes for A in [E,U]),
     feasible_witness=bool(res.success),status='witness' if res.success else 'inconclusive_timeout' if res.status==1 else 'inconclusive_support',residual=residual))
    pd.DataFrame(rows).to_csv(OUT/'scalable_completion.csv',index=False)
   print('INNER COMPLETION',k,seed,'seconds',round(time.perf_counter()-start,2),flush=True)
 if not pd.DataFrame(rows).feasible_witness.any():print('No witness; all failures retained.',flush=True)

if __name__=='__main__':run()
