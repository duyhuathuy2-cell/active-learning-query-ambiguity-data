"""Full-polytope column generation with exhaustive smaller-side pricing.

No fixed atom support assumption. Numerical certificates expose primal residual,
the exhaustive pricing minimum and the resulting global primal-dual gap.
Worst-case dependence on min(K,M) remains exponential.
"""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time,warnings,itertools,argparse
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import csr_matrix,hstack,eye
from method import interval
OUT=ROOT.parent/'data/results/global_and_prediction';OUT.mkdir(exist_ok=True)
DESIGN={'date':'2026-10-03','fixed_before_execution':True,'external_preregistration':False,
 'sizes':[[8,4],[16,8],[32,8],[64,8],[64,16]],'systems_per_size':3,
 'seed_offset':2800000,'benchmark':'independent finite-mixture systems, same branch perturbation 0.03 as old scalable study',
 'initial_atoms':128,'pricing_batch':128,'numerical_tolerance':1e-8,'wall_seconds_per_system':120,
 'endpoint':'complete-polytope feasibility and random signed linear pair functional, not a claimed MI optimum',
 'controls':'embedded impossible cycle and all five former K64 support failures; independent full enumeration controls supplied separately for (2,2),(3,2),(4,3)',
 'pricing':'enumerate all states of smaller side and analytically minimize larger-side binary labels; complete, not sampled'}
dp=OUT/'exact_global_design.json'
if dp.exists():assert json.loads(dp.read_text())==DESIGN
else:dp.write_text(json.dumps(DESIGN,indent=2))

def column_matrices(atoms,k,m):
 pairs=(atoms[:,:k,None]*atoms[:,None,k:]).reshape(len(atoms),k*m).T.astype(float)
 return np.vstack([np.ones(len(atoms)),atoms.T]),np.vstack([pairs,-pairs])

def price(lam,beta,C,k,m,batch=128):
 # Reduced cost c - Aeq^T lam - Aub^T beta.
 pair=C-beta[:k*m].reshape(k,m)+beta[k*m:].reshape(k,m)
 ca=-lam[1:1+k];cb=-lam[1+k:];constant=-lam[0]
 swap=k<m
 if swap:pair=pair.T;ca,cb=cb,ca;k,m=m,k
 assert m<=20,'Complete pricing budget exceeded; do not certify without exact pricing.'
 active=np.flatnonzero(np.any(pair!=0,axis=0)|(cb!=0))
 codes=np.arange(2**len(active),dtype=np.uint32)
 states=np.zeros((len(codes),m),dtype=np.int8)
 states[:,active]=((codes[:,None]>>np.arange(len(active),dtype=np.uint32))&1).astype(np.int8)
 linear=states@pair.T+ca
 values=constant+states@cb+np.minimum(linear,0).sum(1)
 order=np.argsort(values,kind='stable')[:min(batch,len(values))]
 large=(linear[order]<0).astype(np.int8);small=states[order]
 atoms=np.c_[small,large] if swap else np.c_[large,small]
 return float(values.min()),atoms,values[order]

def solve(a,b,lo,hi,C=None,initial_atoms=None,seconds=120,batch=128,tol=1e-8):
 start=time.perf_counter();k=len(a);m=len(b);C=np.zeros((k,m)) if C is None else np.asarray(C)
 if initial_atoms is None:
  rng=np.random.default_rng(181);atoms=rng.binomial(1,np.r_[a,b],size=(128,k+m)).astype(np.int8)
 else:atoms=np.asarray(initial_atoms,dtype=np.int8).copy()
 atoms=np.unique(atoms,axis=0);seen={row.tobytes() for row in atoms};eqrhs=np.r_[1,a,b];ubrhs=np.r_[hi.ravel(),-lo.ravel()]
 trace=[];phase=1;iterations=0;pricing_min=np.nan;global_lower=np.nan
 while time.perf_counter()-start<seconds:
  E,U=column_matrices(atoms,k,m);n=len(atoms);rows=E.shape[0];ur=U.shape[0]
  if phase==1:
   EE=hstack([csr_matrix(E),eye(rows),-eye(rows),csr_matrix((rows,ur))],format='csr')
   UU=hstack([csr_matrix(U),csr_matrix((ur,2*rows)),-eye(ur)],format='csr')
   objective=np.r_[np.zeros(n),np.ones(2*rows+ur)]
  else:EE=csr_matrix(E);UU=csr_matrix(U);objective=(C.ravel()@U[:k*m]).ravel()
  remaining=seconds-(time.perf_counter()-start)
  if remaining<=0:break
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',OptimizeWarning)
   r=linprog(objective,A_ub=UU,b_ub=ubrhs,A_eq=EE,b_eq=eqrhs,bounds=(0,None),method='highs',
    options={'time_limit':remaining,'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  if not r.success:
   return {'status':'inconclusive_solver','solver_status':r.status,'seconds':time.perf_counter()-start,'atoms':len(atoms),'trace':trace}
  iterations+=1
  if phase==1 and r.fun<=tol:
   if not np.any(C):
    w=r.x[:n];return finish('globally_feasible',atoms,w,a,b,lo,hi,0,0,0,start,iterations,trace)
   phase=2;continue
  p0=time.perf_counter();rc,new,values=price(r.eqlin.marginals,r.ineqlin.marginals,np.zeros((k,m)) if phase==1 else C,k,m,batch)
  pricing_min=rc;global_lower=float(r.fun+min(0,rc))
  trace.append({'phase':phase,'atoms':n,'restricted_objective':float(r.fun),'minimum_reduced_cost':rc,
   'global_objective_lower_bound':global_lower,'pricing_seconds':time.perf_counter()-p0})
  if phase==1 and global_lower>tol:
   # A positive lower bound on minimum total constraint violation separates the entire polytope.
   return {'status':'globally_infeasible','violation_lower_bound':global_lower,'minimum_reduced_cost':rc,
    'seconds':time.perf_counter()-start,'atoms':n,'iterations':iterations,'trace':trace,
    'dual_eq':r.eqlin.marginals,'dual_ub':r.ineqlin.marginals}
  if rc>=-tol:
   if phase==1:
    return {'status':'inconclusive_tolerance','seconds':time.perf_counter()-start,'atoms':n,'trace':trace}
   return finish('globally_optimal',atoms,r.x,a,b,lo,hi,r.fun,global_lower,rc,start,iterations,trace,
                 r.eqlin.marginals,r.ineqlin.marginals)
  additions=[]
  for row,value in zip(new,values):
   code=row.tobytes()
   if value<-tol and code not in seen:seen.add(code);additions.append(row)
  if not additions:return {'status':'inconclusive_numerical_stall','seconds':time.perf_counter()-start,'atoms':n,'trace':trace}
  atoms=np.vstack([atoms,np.asarray(additions,dtype=np.int8)])
 # A timeout is not a certificate. Preserve any feasible phase-II incumbent and its last bound.
 if phase==2 and 'r' in locals() and r.success and len(r.x)==len(E[0]):
  result=finish('bounded_timeout',atoms[:len(r.x)],r.x,a,b,lo,hi,r.fun,global_lower,pricing_min,start,iterations,trace)
  return result
 return {'status':'inconclusive_timeout','seconds':time.perf_counter()-start,'atoms':len(atoms),'iterations':iterations,'trace':trace}

def finish(status,atoms,w,a,b,lo,hi,objective,lower,rc,start,iterations,trace,dual_eq=None,dual_ub=None):
 E,U=column_matrices(atoms,len(a),len(b));residual=max(np.max(abs(E@w-np.r_[1,a,b])),max(0,np.max(U@w-np.r_[hi.ravel(),-lo.ravel()])),max(0,-w.min()))
 assert residual<1e-7,residual
 keep=w>1e-12;active=atoms[keep];weights=w[keep];joint=np.einsum('h,hi,hj->ij',weights,active[:,:len(a)],active[:,len(a):])
 return dict(status=status,seconds=time.perf_counter()-start,atoms=len(atoms),active_atoms=int(keep.sum()),iterations=iterations,
  primal_residual=float(residual),objective=float(objective),global_lower=float(lower),gap=float(max(0,objective-lower)),
  minimum_reduced_cost=float(rc),states=active,weights=weights,pair_joint=joint,trace=trace,dual_eq=dual_eq,dual_ub=dual_ub)

def benchmark():
 rows=[];t0=time.perf_counter()
 for k,m in DESIGN['sizes']:
  for seed in range(3):
   rng=np.random.default_rng(2800000+1000*k+10*m+seed);pi=rng.dirichlet(np.ones(16));theta=rng.beta(.8,.8,(16,k+m))
   a=pi@theta[:,:k];b=pi@theta[:,k:];t=np.einsum('h,hi,hj->ij',pi,theta[:,:k],theta[:,k:])
   q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None]);branches=np.clip(np.stack([q0,q1],1)+.03*rng.normal(size=(k,2,m)),1e-6,1-1e-6)
   _,_,lo,hi=interval(a[:,None],b[None,:],branches[:,0],branches[:,1]);C=rng.uniform(-1,1,(k,m))/(k*m)
   run_case(f'new_{k}_{m}_{seed}',a,b,lo,hi,C,rows)
 for seed in range(5):
  with np.load(ROOT.parent/f'data/results/policy_controls/scalable_input_64_{seed}.npz') as z:
   run_case(f'former_failure_64_16_{seed}',z['a'],z['b'],z['lo'],z['hi'],np.zeros((64,16)),rows)
 # Incompatible parity cycle embedded into a large pool.
 a=np.full(64,.5);b=np.full(16,.5);lo=np.zeros((64,16));hi=np.full((64,16),.5)
 for i,j,t in [(0,0,.5),(0,1,.5),(1,0,.5),(1,1,0)]:lo[i,j]=hi[i,j]=t
 run_case('impossible_cycle_64_16',a,b,lo,hi,np.zeros((64,16)),rows)
 pd.DataFrame(rows).to_csv(OUT/'exact_global_summary.csv',index=False)
 (OUT/'exact_global_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-t0,'design':DESIGN},indent=2))

def run_case(name,a,b,lo,hi,C,rows):
 np.savez_compressed(OUT/f'exact_global_input_{name}.npz',a=a,b=b,lo=lo,hi=hi,C=C)
 r=solve(a,b,lo,hi,C,seconds=120)
 arrays={key:r.pop(key) for key in ['states','weights','pair_joint','dual_eq','dual_ub'] if key in r and r[key] is not None}
 if arrays:np.savez_compressed(OUT/f'exact_global_witness_{name}.npz',**arrays)
 (OUT/f'exact_global_trace_{name}.json').write_text(json.dumps(r,indent=2))
 row={key:value for key,value in r.items() if key!='trace'};row.update(case=name,candidates=len(a),targets=len(b));rows.append(row)
 pd.DataFrame(rows).to_csv(OUT/'exact_global_summary.csv',index=False)
 print('EXACT GLOBAL',name,r['status'],'seconds',round(r['seconds'],2),'gap',r.get('gap'),flush=True)
if __name__=='__main__':benchmark()
