"""Independent full-atom controls and exactly feasible linear certificates."""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time,itertools
from scipy.optimize import linprog
from block_global import solve_block,OUT
from verified_global import fq,ends,ivq,dual_lower,anchor_for,rational_feasible,info_iv
from global_regret import worst_regret,info

def linear_certificates():
 rows=[]
 for name in pd.read_csv(OUT/'block_global_summary.csv')['case']:
  with np.load(OUT/f'exact_global_input_{name}.npz') as z:a,b,lo,hi,C=[z[k] for k in ['a','b','lo','hi','C']]
  with np.load(OUT/f'block_global_witness_{name}.npz') as z:w={k:z[k] for k in z.files}
  if 'impossible' in name:
   lower,accounting=dual_lower(w['dual_eq'],w['dual_ub'],a,b,lo,hi,np.zeros_like(C))
   assert lower>0
   cert=dict(case=name,status='verified_global_infeasibility',violation_lower=lower,accounting=accounting)
  else:
   if (OUT/f'verified_anchor_{name}.npz').exists():
    with np.load(OUT/f'verified_anchor_{name}.npz') as z:anchor={k:z[k] for k in z.files}
   else:
    anchor,_=anchor_for(a,b,lo,hi)
    np.savez_compressed(OUT/f'verified_anchor_{name}.npz',target_states=anchor['target_states'],weights=anchor['weights'],candidate_joint=anchor['candidate_joint'])
   t,cert=rational_feasible(a,b,lo,hi,w,anchor)
   value=sum(fq(C[i,j])*t[i][j] for i in range(len(a)) for j in range(len(b)))
   upper=ends(ivq(value))[1]
   if 'dual_eq' in w:
    lower,accounting=dual_lower(w['dual_eq'],w['dual_ub'],a,b,lo,hi,C)
    assert upper-lower<1e-7
    cert.update(case=name,status='verified_linear_enclosure',lower=lower,upper=upper,gap=upper-lower,accounting=accounting)
   else:cert.update(case=name,status='verified_global_feasibility',lower=0.,upper=upper,gap=0.)
  (OUT/f'verified_linear_certificate_{name}.json').write_text(json.dumps(cert,indent=2))
  rows.append({k:cert[k] for k in ['case','status','lower','upper','gap','violation_lower'] if k in cert})
  pd.DataFrame(rows).to_csv(OUT/'verified_linear_summary.csv',index=False)
  print('VERIFIED LINEAR',name,cert['status'],cert.get('gap'),flush=True)

def controls():
 rng=np.random.default_rng(91031);rows=[]
 for k,m in [(2,2),(3,2),(4,3)]:
  atoms=np.array(list(itertools.product([0,1],repeat=k+m)),dtype=np.int8);A=atoms[:,:k];B=atoms[:,k:]
  pairs=np.einsum('ni,nj->nij',A,B).reshape(len(atoms),-1).T
  w=rng.dirichlet(np.ones(len(atoms)));a=w@A;b=w@B;t=(pairs@w).reshape(k,m)
  lo=np.maximum(np.maximum(0,a[:,None]+b-1),t-.05);hi=np.minimum(np.minimum(a[:,None],b),t+.05)
  C=rng.uniform(-1,1,(k,m))
  full=linprog(C.ravel()@pairs,A_ub=np.vstack([pairs,-pairs]),b_ub=np.r_[hi.ravel(),-lo.ravel()],
    A_eq=np.vstack([np.ones(len(atoms)),A.T,B.T]),b_eq=np.r_[1,a,b],bounds=(0,None),method='highs',options={'threads':1})
  block=solve_block(a,b,lo,hi,C)
  assert full.success and block['status']=='numerical_optimum' and abs(full.fun-block['objective'])<1e-8
  rows.append(dict(candidates=k,targets=m,full_atoms=len(atoms),full_value=full.fun,block_value=block['objective'],difference=abs(full.fun-block['objective'])))
 pd.DataFrame(rows).to_csv(OUT/'independent_full_atom_controls.csv',index=False)
 # A nonrectangular feasible parity cycle forces actual nonlinear branching.
 a=b=np.full(2,.5);lo=np.array([[.35,.35],[.35,0.]]);hi=np.array([[.5,.5],[.5,.15]])
 np.savez_compressed(OUT/'coupled_regret_input.npz',a=a,b=b,lo=lo,hi=hi)
 r=worst_regret(a,b,lo,hi,epsilon=.001,seconds=240)
 witness=r.pop('witness');anchor,_=anchor_for(a,b,lo,hi);t,cert=rational_feasible(a,b,lo,hi,witness,anchor)
 chosen,comp=r['chosen'],r['worst_competitor'];val=sum(info_iv(t[comp][j],fq(.5),fq(.5))-info_iv(t[chosen][j],fq(.5),fq(.5)) for j in range(2))/2
 lower=max(0,ends(val)[0]);upper=np.nextafter(r['upper']+1e-8,np.inf)
 # Independent dense grid, with all eight CHSH facets of unbiased 2x2 pairs.
 # CHSH is a necessary condition; its use here only checks a relaxation upper.
 grid=[np.linspace(lo[i,j],hi[i,j],31) for i,j in itertools.product(range(2),repeat=2)]
 points=np.stack(np.meshgrid(*grid,indexing='ij'),axis=-1).reshape(-1,4);corr=4*points-1
 signs=np.array([s for s in itertools.product([-1,1],repeat=4) if np.prod(s)==-1])
 feasible=np.max(corr@signs.T,axis=1)<=2+1e-12
 values=info(points[:,2:4],.5,.5).mean(1)-info(points[:,:2],.5,.5).mean(1)
 if chosen==1:values=-values
 grid_max=max(0,float(values[feasible].max()))
 assert r['nodes']>2 and upper-lower<=.001+1e-8 and upper<r['local_regret']-.01
 assert grid_max<=upper+1e-8
 cert.update(case='coupled_2_2',chosen=chosen,competitor=comp,verified_lower=lower,verified_upper=upper,
  verified_gap=upper-lower,grid_max=grid_max,grid_points=len(points),grid_feasible_points=int(feasible.sum()))
 (OUT/'coupled_regret_certificate.json').write_text(json.dumps(cert,indent=2))
 (OUT/'coupled_regret_trace.json').write_text(json.dumps(r,indent=2))
 pd.DataFrame([{**{k:v for k,v in r.items() if k!='trace'},'verified_lower':lower,'verified_upper':upper,'verified_gap':upper-lower,'grid_max':grid_max}]).to_csv(OUT/'coupled_regret_summary.csv',index=False)
 print('COUPLED CONTROL',r['nodes'],'nodes',r['local_regret'],'local',lower,upper,'global',flush=True)

if __name__=='__main__':
 controls();linear_certificates()
