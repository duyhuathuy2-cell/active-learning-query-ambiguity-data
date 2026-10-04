"""Smaller-side block column generation for the complete global LP.

Each generated target state carries continuous candidate joint masses, so it
represents all larger-side outcomes and does not sample them. Complete pricing
is analytic over that side and exhaustive over every smaller-side state.
"""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time,warnings
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import csr_matrix,hstack,vstack,eye,kron
from exact_global import price,OUT
from verified_global import dual_lower

DESIGN={'date':'2026-10-03','fixed_before_execution':True,'external_preregistration':False,
 'motivation':'block generation planned after the full-joint-column timing experiment; no result omitted',
 'cases':'all 15 signed-linear new systems, all five former K64 failures and embedded parity cycle',
 'initial_target_states':32,'pricing_batch':32,'wall_seconds_per_case':180,'numerical_tolerance':1e-8,
 'scope':'complete global-polytope optimization; bounded numerical gap, not rational arithmetic or polynomial time'}
dp=OUT/'block_global_design.json'
if dp.exists():assert json.loads(dp.read_text())==DESIGN
else:dp.write_text(json.dumps(DESIGN,indent=2))

def template(states,k):
 n,m=states.shape;ES=csr_matrix(np.vstack([np.ones(n),states.T]))
 E=vstack([hstack([ES,csr_matrix((m+1,n*k))]),hstack([csr_matrix((k,n)),kron(eye(k),csr_matrix(np.ones((1,n))))])],format='csr')
 cap=hstack([vstack([-eye(n)]*k),eye(n*k)],format='csr')
 pairs=hstack([csr_matrix((k*m,n)),kron(eye(k),csr_matrix(states.T))],format='csr')
 U=vstack([cap,pairs,-pairs],format='csr');return E,U,pairs

def solve_block(a,b,lo,hi,C=None,seconds=180,batch=32,tol=1e-8,initial_states=None):
 start=time.perf_counter();k,m=len(a),len(b);C=np.zeros((k,m)) if C is None else np.asarray(C)
 if k<m:
  raise ValueError('Caller must orient the smaller side as b for block generation.')
 if initial_states is None:
  rng=np.random.default_rng(191);states=np.unique(rng.binomial(1,b,size=(32,m)),axis=0)
 else:states=np.unique(initial_states,axis=0)
 seen={x.tobytes() for x in states};phase=1;trace=[];iterations=0;global_lower=np.nan;rc=np.nan
 rhs=np.r_[1,b,a]
 while time.perf_counter()-start<seconds:
  E,U,pairs=template(states,k);n=len(states);ne=E.shape[0];nu=2*k*m;nv=n*(k+1);caps=n*k
  ubrhs=np.r_[np.zeros(caps),hi.ravel(),-lo.ravel()]
  if phase==1:
   EE=hstack([E,eye(ne),-eye(ne),csr_matrix((ne,nu))],format='csr')
   UU=hstack([U,csr_matrix((caps+nu,2*ne)),vstack([csr_matrix((caps,nu)),-eye(nu)])],format='csr')
   objective=np.r_[np.zeros(nv),np.ones(2*ne+nu)]
  else:EE,UU=E,U;objective=np.r_[np.zeros(n),(states@C.T).T.ravel()]
  remaining=seconds-(time.perf_counter()-start)
  if remaining<=0:break
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',OptimizeWarning)
   r=linprog(objective,A_ub=UU,b_ub=ubrhs,A_eq=EE,b_eq=rhs,bounds=(0,None),method='highs',
    options={'time_limit':remaining,'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  if not r.success:return dict(status='inconclusive_solver',solver_status=r.status,seconds=time.perf_counter()-start,states=len(states),trace=trace)
  iterations+=1
  if phase==1 and r.fun<=tol:
   if not np.any(C):return finish_block('numerical_feasible',states,r.x[:nv],a,b,lo,hi,0,0,0,start,iterations,trace)
   phase=2;continue
  lam=r.eqlin.marginals;converted=np.r_[lam[0],lam[1+m:],lam[1:1+m]]
  t0=time.perf_counter();rc,atoms,values=price(converted,r.ineqlin.marginals[caps:],np.zeros((k,m)) if phase==1 else C,k,m,batch)
  global_lower,error_record=dual_lower(converted,r.ineqlin.marginals[caps:],a,b,lo,hi,np.zeros((k,m)) if phase==1 else C)
  trace.append(dict(phase=phase,target_states=n,restricted_objective=float(r.fun),minimum_reduced_cost=rc,
   global_lower=global_lower,pricing_seconds=time.perf_counter()-t0,numerical_error_accounting=error_record))
  if phase==1 and global_lower>tol:
   return dict(status='globally_infeasible',violation_lower_bound=global_lower,seconds=time.perf_counter()-start,
    states=len(states),iterations=iterations,trace=trace,dual_eq=converted,dual_ub=r.ineqlin.marginals[caps:])
  if rc>=-tol:
   if phase==1:return dict(status='inconclusive_tolerance',seconds=time.perf_counter()-start,states=n,trace=trace)
   return finish_block('numerical_optimum',states,r.x,a,b,lo,hi,r.fun,global_lower,rc,start,iterations,trace,converted,r.ineqlin.marginals[caps:])
  add=[]
  for atom,value in zip(atoms,values):
   row=atom[k:];code=row.tobytes()
   if value<-tol and code not in seen:seen.add(code);add.append(row)
  if not add:return dict(status='inconclusive_numerical_stall',seconds=time.perf_counter()-start,states=n,trace=trace)
  states=np.vstack([states,np.asarray(add,dtype=np.int8)])
 return dict(status='inconclusive_timeout',seconds=time.perf_counter()-start,states=len(states),iterations=iterations,trace=trace)

def finish_block(status,states,x,a,b,lo,hi,value,lower,rc,start,iterations,trace,dual_eq=None,dual_ub=None):
 E,U,pairs=template(states,len(a));n=len(states)
 residual=max(float(np.max(abs(E@x-np.r_[1,b,a]))),float(max(0,np.max(U@x-np.r_[np.zeros(n*len(a)),hi.ravel(),-lo.ravel()]))),float(max(0,-x.min())))
 assert residual<1e-7
 w=x[:n];v=x[n:].reshape(len(a),n)
 return dict(status=status,seconds=time.perf_counter()-start,states=n,active_states=int((w>1e-12).sum()),variables=len(x),iterations=iterations,
  primal_residual=residual,objective=float(value),global_lower=float(lower),gap=float(max(0,value-lower)),minimum_reduced_cost=float(rc),
  target_states=states,weights=w,candidate_joint=v,pair_joint=v@states,trace=trace,dual_eq=dual_eq,dual_ub=dual_ub)

def benchmark():
 rows=[];start=time.perf_counter()
 # Inputs are generated deterministically by the first benchmark, without waiting for its solve.
 from method import interval
 cases=[]
 for k,m in [[8,4],[16,8],[32,8],[64,8],[64,16]]:
  for seed in range(3):
   name=f'new_{k}_{m}_{seed}';path=OUT/f'exact_global_input_{name}.npz'
   if not path.exists():
    rng=np.random.default_rng(2800000+1000*k+10*m+seed);pi=rng.dirichlet(np.ones(16));theta=rng.beta(.8,.8,(16,k+m))
    a=pi@theta[:,:k];b=pi@theta[:,k:];t=np.einsum('h,hi,hj->ij',pi,theta[:,:k],theta[:,k:]);q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None])
    branches=np.clip(np.stack([q0,q1],1)+.03*rng.normal(size=(k,2,m)),1e-6,1-1e-6);_,_,lo,hi=interval(a[:,None],b[None,:],branches[:,0],branches[:,1]);C=rng.uniform(-1,1,(k,m))/(k*m)
    np.savez_compressed(path,a=a,b=b,lo=lo,hi=hi,C=C)
   cases.append((name,path))
 for seed in range(5):
  name=f'former_failure_64_16_{seed}';path=OUT/f'exact_global_input_{name}.npz'
  with np.load(ROOT.parent/f'data/results/policy_controls/scalable_input_64_{seed}.npz') as z:
   np.savez_compressed(path,a=z['a'],b=z['b'],lo=z['lo'],hi=z['hi'],C=np.zeros((64,16)))
  cases.append((name,path))
 a=np.full(64,.5);b=np.full(16,.5);lo=np.zeros((64,16));hi=np.full((64,16),.5)
 for i,j,t in [(0,0,.5),(0,1,.5),(1,0,.5),(1,1,0)]:lo[i,j]=hi[i,j]=t
 path=OUT/'exact_global_input_impossible_cycle_64_16.npz';np.savez_compressed(path,a=a,b=b,lo=lo,hi=hi,C=np.zeros((64,16)));cases.append(('impossible_cycle_64_16',path))
 for name,path in cases:
  with np.load(path) as z:r=solve_block(z['a'],z['b'],z['lo'],z['hi'],z['C']);k=len(z['a']);m=len(z['b'])
  arrays={key:r.pop(key) for key in ['target_states','weights','candidate_joint','pair_joint','dual_eq','dual_ub'] if key in r and r[key] is not None}
  if arrays:np.savez_compressed(OUT/f'block_global_witness_{name}.npz',**arrays)
  (OUT/f'block_global_trace_{name}.json').write_text(json.dumps(r,indent=2))
  row={key:value for key,value in r.items() if key!='trace'};row.update(case=name,candidates=k,targets=m);rows.append(row)
  pd.DataFrame(rows).to_csv(OUT/'block_global_summary.csv',index=False)
  print('BLOCK GLOBAL',name,r['status'],'seconds',round(r['seconds'],2),'gap',r.get('gap'),flush=True)
 (OUT/'block_global_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'design':DESIGN},indent=2))
if __name__=='__main__':benchmark()
