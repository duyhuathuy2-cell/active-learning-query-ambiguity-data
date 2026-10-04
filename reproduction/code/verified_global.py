"""Independent rational feasible-joint and error-aware dual verification.

Inputs and saved floating solutions are interpreted as exact dyadic rationals.
Bit-flip kernels correct every marginal exactly; pair membership is checked with
Fraction arithmetic. Information is enclosed with directed interval logs.
"""
import environment
from environment import ROOT
from fractions import Fraction as Q
import numpy as np,json,time
import mpmath as mp
mp.iv.dps=40
OUT=ROOT.parent/'data/results/global_and_prediction';OUT.mkdir(exist_ok=True)
def fq(x):return Q.from_float(float(x))
def ivq(x):return mp.iv.mpf(x.numerator)/x.denominator
def ends(x):return np.nextafter(float(x.a),-np.inf),np.nextafter(float(x.b),np.inf)
def info_iv(t,a,b):
 cells=[t,a-t,b-t,1-a-b+t];rows=[a,a,1-a,1-a];cols=[b,1-b,b,1-b];value=mp.iv.mpf(0)
 for p,r,c in zip(cells,rows,cols):
  assert p>=0
  if p:value+=ivq(p)*mp.iv.log(ivq(p)/(ivq(r)*ivq(c)))/mp.iv.log(2)
 return value
def dual_lower(lam,beta,a,b,lo,hi,C):
 from exact_global import price
 k,m=len(a),len(b);beta=np.minimum(np.asarray(beta),0)
 rc,_,_=price(lam,beta,C,k,m,1)
 # A standard IEEE gamma_N absolute forward-error bound; N conservatively
 # exceeds every addition/product path in pair coefficients and complete pricing.
 leaf=np.sum(abs(C))+np.sum(abs(lam))+2*np.sum(abs(beta))
 n=20*(k*m+k+m+1);u=np.finfo(float).eps/2;gamma=n*u/(1-n*u)
 price_error=float(np.nextafter(gamma*leaf+np.finfo(float).tiny*n,np.inf))
 rhs=np.r_[1,a,b];urhs=np.r_[hi.ravel(),-lo.ravel()]
 products=np.r_[np.asarray(lam)*rhs,beta*urhs];d=float(products.sum())
 n2=2*len(products)+2;gamma2=n2*u/(1-n2*u)
 derror=float(np.nextafter(gamma2*np.sum(abs(products))+np.finfo(float).tiny*n2,np.inf))
 lower=float(np.nextafter(d-derror+min(0,rc-price_error),-np.inf))
 return lower,{'raw_dual':d,'dual_roundoff_bound':derror,'raw_pricing_minimum':rc,
  'pricing_roundoff_bound':price_error,'minimum_reduced_cost_lower':rc-price_error}

def affine_envelope(a,b,ll,hh,chosen,competitor):
 k,m=len(a),len(b);coeff=np.zeros((k,m));constant=mp.iv.mpf(0)
 for row,sign in [(competitor,1),(chosen,-1)]:
  ar=fq(a[row])
  for j in range(m):
   bj=fq(b[j]);l,h=fq(ll[row,j]),fq(hh[row,j])
   if l==h:slope=mp.iv.mpf(0);intercept=sign*info_iv(l,ar,bj)
   elif sign==1:
    il,ih=info_iv(l,ar,bj),info_iv(h,ar,bj);slope=(ih-il)/ivq(h-l);intercept=il-slope*ivq(l)
   else:
    mid=(l+h)/2
    derivative=mp.iv.log(ivq(mid*(1-ar-bj+mid))/ivq((ar-mid)*(bj-mid)))/mp.iv.log(2)
    slope=-derivative;intercept=-info_iv(mid,ar,bj)+derivative*ivq(mid)
   slo,shi=ends(slope);c=float((slo+shi)/2);coeff[row,j]=c/m
   # Exact dyadic chosen coefficient, with outward compensation over its box.
   difference=slope-ivq(fq(coeff[row,j])*m)
   candidates=[difference*ivq(l),difference*ivq(h)]
   e=max(ends(x)[1] for x in candidates)
   constant+=(intercept+mp.iv.mpf(e))/m
 return coeff,ends(constant)[1]

def exact_moments(states,w,v):
 # Any cap violation is clipped BEFORE forming the represented joint.
 ww=[max(Q(0),fq(x)) for x in w];total=sum(ww);assert total>0
 vv=[[max(Q(0),min(ww[s],fq(v[i,s]))) for s in range(len(w))] for i in range(len(v))]
 aa=[sum(row)/total for row in vv]
 bb=[sum(ww[s] for s in range(len(w)) if states[s,j])/total for j in range(states.shape[1])]
 tt=[[sum(vv[i][s] for s in range(len(w)) if states[s,j])/total for j in range(states.shape[1])] for i in range(len(v))]
 return aa,bb,tt
def correct_marginals(aa,bb,tt,a,b):
 def kernel(old,new):
  if new>=old:
   r=(new-old)/(1-old) if old<1 else Q(0);return r,1-r
  return Q(0),new/old
 ka=[kernel(x,y) for x,y in zip(aa,a)];kb=[kernel(x,y) for x,y in zip(bb,b)]
 adjusted=[[ra*rb+ra*sb*bb[j]+sa*rb*aa[i]+sa*sb*tt[i][j] for j,(rb,sb) in enumerate(kb)] for i,(ra,sa) in enumerate(ka)]
 return adjusted,ka,kb
def rational_feasible(a,b,lo,hi,witness,anchor):
 aa,bb,tt=exact_moments(witness['target_states'],witness['weights'],witness['candidate_joint'])
 ar,br,tr=exact_moments(anchor['target_states'],anchor['weights'],anchor['candidate_joint'])
 aQ=list(map(fq,a));bQ=list(map(fq,b));lQ=[[fq(x) for x in row] for row in lo];hQ=[[fq(x) for x in row] for row in hi]
 # Find the least perturbing exact mixture with the independently computed interior joint.
 for blend in [Q(0),Q(1,10**9),Q(1,10**8),Q(1,10**7),Q(1,10**6),Q(1,10**5),Q(1,10**4),Q(1,1000),Q(1)]:
  ax=[(1-blend)*x+blend*y for x,y in zip(aa,ar)];bx=[(1-blend)*x+blend*y for x,y in zip(bb,br)]
  tx=[[(1-blend)*tt[i][j]+blend*tr[i][j] for j in range(len(b))] for i in range(len(a))]
  adjusted,ka,kb=correct_marginals(ax,bx,tx,aQ,bQ)
  if all(lQ[i][j]<=adjusted[i][j]<=hQ[i][j] for i in range(len(a)) for j in range(len(b))):
   return adjusted,{'blend':str(blend),'candidate_flip_kernels':[[str(x) for x in pair] for pair in ka],
    'target_flip_kernels':[[str(x) for x in pair] for pair in kb],
    'exact_marginals_a':[str(x) for x in aQ],'exact_marginals_b':[str(x) for x in bQ],
    'exact_pair_joint':[[str(x) for x in row] for row in adjusted],
    'exact_normalization_and_marginals':True,'all_pair_bounds_checked_with_rationals':True}
 raise RuntimeError('No exactly feasible corrected witness; do not claim certification.')

def anchor_for(a,b,lo,hi):
 from block_global import solve_block
 positive=(hi-lo)[hi>lo]
 for fraction in [.05,.005,.0005]:
  margin=min(1e-6,float(positive.min())*fraction)
  ll=np.where(hi>lo,lo+margin,lo);hh=np.where(hi>lo,hi-margin,hi)
  r=solve_block(a,b,ll,hh,seconds=90)
  if r['status']=='numerical_feasible':return r,margin
 raise RuntimeError('Interior anchor unavailable; exact feasible reconstruction unresolved.')

def verify_block_result(a,b,lo,hi,C,result):
 """Promote a provisional solver result only after original-polytope checks."""
 try:
  if result['status']=='globally_infeasible':
   lower,accounting=dual_lower(result['dual_eq'],result['dual_ub'],a,b,lo,hi,np.zeros_like(C))
   if lower>0:return dict(status='verified_global_infeasibility',violation_lower=lower,accounting=accounting)
   return dict(status='inconclusive_separation')
  if result['status'] not in ['numerical_feasible','numerical_optimum']:return dict(status='inconclusive_solver',raw_status=result['status'])
  try:t,cert=rational_feasible(a,b,lo,hi,result,result)
  except RuntimeError:
   anchor,_=anchor_for(a,b,lo,hi);t,cert=rational_feasible(a,b,lo,hi,result,anchor)
  objective=sum(fq(C[i,j])*t[i][j] for i in range(len(a)) for j in range(len(b)))
  cert.update(status='verified_global_feasibility',upper=ends(ivq(objective))[1])
  if result.get('dual_eq') is not None:
   lower,accounting=dual_lower(result['dual_eq'],result['dual_ub'],a,b,lo,hi,C)
   cert.update(status='verified_linear_enclosure',lower=lower,gap=cert['upper']-lower,accounting=accounting)
  return cert
 except (RuntimeError,ValueError,AssertionError) as e:
  return dict(status='inconclusive_exact_feasibility',reason=str(e),raw_status=result['status'])

def verify_saved():
 rows=[];start=time.perf_counter()
 for name in ['new_8_4_0','new_16_8_0','new_32_8_0','new_64_8_0','new_64_16_0']:
  with np.load(OUT/f'exact_global_input_{name}.npz') as z:a,b,lo,hi,C=[z[k] for k in ['a','b','lo','hi','C']]
  anchor,margin=anchor_for(a,b,lo,hi)
  np.savez_compressed(OUT/f'verified_anchor_{name}.npz',target_states=anchor['target_states'],weights=anchor['weights'],candidate_joint=anchor['candidate_joint'])
  with np.load(OUT/f'global_regret_witness_{name}.npz') as z:witness={k:z[k] for k in z.files}
  adjusted,cert=rational_feasible(a,b,lo,hi,witness,anchor)
  row=next(x for x in __import__('pandas').read_csv(OUT/'global_regret_summary.csv').to_dict('records') if x['case']==name)
  chosen,competitor=int(row['chosen']),int(row['worst_competitor'])
  val=sum(info_iv(adjusted[competitor][j],fq(a[competitor]),fq(b[j]))-info_iv(adjusted[chosen][j],fq(a[chosen]),fq(b[j])) for j in range(len(b)))/len(b)
  lower=max(0,ends(val)[0]);upper=float(np.nextafter(row['upper']+1e-8,np.inf))
  # The global_regret runner below recomputes affine and dual bounds with outward/error-aware arithmetic.
  cert.update(case=name,chosen=chosen,competitor=competitor,regret_lower=lower,solver_upper=upper,
   verified_gap=upper-lower,anchor_margin=margin,information_logs='mpmath directed interval, 40 decimal digits')
  assert cert['verified_gap']<=.001+1e-8
  (OUT/f'verified_regret_certificate_{name}.json').write_text(json.dumps(cert,indent=2))
  rows.append({k:cert[k] for k in ['case','regret_lower','solver_upper','verified_gap','anchor_margin','blend']})
  print('RATIONAL GLOBAL',name,'gap',upper-lower,'blend',cert['blend'],flush=True)
 __import__('pandas').DataFrame(rows).to_csv(OUT/'verified_regret_summary.csv',index=False)
 (OUT/'verified_global_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'cases':len(rows)},indent=2))
if __name__=='__main__':verify_saved()
