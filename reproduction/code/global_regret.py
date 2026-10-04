"""Global information-regret branch and bound with complete LP pricing.

Every active competitor/box carries a valid analytic or LP upper bound. A
successful run encloses the global worst regret within the declared bit gap.
"""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time,heapq
from method import entropy
from block_global import solve_block,OUT
from verified_global import affine_envelope,info_iv,fq,ends
DESIGN={'date':'2026-10-03','fixed_before_execution':True,'external_preregistration':False,
 'cases':['new_8_4_0','new_16_8_0','new_32_8_0','new_64_8_0','new_64_16_0'],
 'objective':'complete-polytope worst MI regret of the rectangular-minimax action against every other candidate',
 'epsilon_bits':.001,'wall_seconds_per_case':240,'LP_seconds_per_node':40,
 'bound':'positive MI secant and negative MI tangent, complete globally priced linear support; all unresolved boxes retained',
 'scope':'absolute certified numerical enclosure, not zero-gap rational nonlinear arithmetic'}
dp=OUT/'global_regret_design.json'
if dp.exists():assert json.loads(dp.read_text())==DESIGN
else:dp.write_text(json.dumps(DESIGN,indent=2))

def info(t,a,b):
 if a==0 or a==1:return np.zeros_like(t)
 return entropy(b)-(1-a)*entropy((b-t)/(1-a))-a*entropy(t/a)

def derivative(t,a,b):
 num=t*(1-a-b+t);den=(a-t)*(b-t)
 return np.log(np.maximum(num,1e-300)/np.maximum(den,1e-300))/np.log(2)

def candidate_bounds(a,b,lo,hi):
 L=[];U=[]
 for i in range(len(a)):
  lower=0;upper=0;ar=fq(a[i])
  for j in range(len(b)):
   bj=fq(b[j]);l=max(fq(lo[i,j]),0,ar+bj-1);h=min(fq(hi[i,j]),ar,bj)
   center=max(l,min(h,ar*bj));lower+=info_iv(center,ar,bj)
   upper+=max(ends(info_iv(l,ar,bj))[1],ends(info_iv(h,ar,bj))[1])
  L.append(ends(lower/len(b))[0]);U.append(float(np.nextafter(upper/len(b),np.inf)))
 return np.array(L),np.array(U)

def envelope(a,b,ll,hh,chosen,competitor):
 k,m=len(a),len(b);coeff=np.zeros((k,m));constant=0.;mid=(ll[chosen]+hh[chosen])/2
 for j in range(m):
  vlow=info(ll[competitor,j],a[competitor],b[j]);vhigh=info(hh[competitor,j],a[competitor],b[j])
  width=hh[competitor,j]-ll[competitor,j]
  slope=0. if width<1e-14 else float((vhigh-vlow)/width)
  coeff[competitor,j]=slope/m;constant+=float(vlow-slope*ll[competitor,j])/m
  width0=hh[chosen,j]-ll[chosen,j]
  tangent=0. if width0<1e-14 else float(derivative(mid[j],a[chosen],b[j]))
  coeff[chosen,j]=-tangent/m;constant+=float(-info(mid[j],a[chosen],b[j])+tangent*mid[j])/m
 return coeff,constant,mid

def worst_regret(a,b,lo,hi,epsilon=.001,seconds=240,warm=None,chosen_override=None):
 start=time.perf_counter();L,U=candidate_bounds(a,b,lo,hi)
 local=np.array([max(0,np.max(np.delete(U,i))-L[i]) for i in range(len(a))]);chosen=int(local.argmin()) if chosen_override is None else int(chosen_override)
 initial=solve_block(a,b,lo,hi,seconds=min(40,seconds),initial_states=warm)
 if initial['status'] not in ['numerical_feasible','numerical_optimum']:return dict(status='inconclusive_nonempty',initial_status=initial['status'])
 states=initial['target_states'];t0=initial['pair_joint'];val0=np.array([info(t0[i],a[i],b).mean() for i in range(len(a))])
 lower=float(max(0,val0.max()-val0[chosen]));witness=initial;best_comp=int(val0.argmax())
 queue=[];counter=0;trace=[]
 for competitor in range(len(a)):
  if competitor==chosen:continue
  ub=float(np.nextafter(max(0,U[competitor]-L[chosen]),np.inf));heapq.heappush(queue,(-ub,counter,competitor,lo.copy(),hi.copy()));counter+=1
 nodes=0;unresolved=False;pruned_upper=lower
 while queue and -queue[0][0]-lower>epsilon and time.perf_counter()-start<seconds:
  negub,_,competitor,ll,hh=heapq.heappop(queue)
  if -negub<=lower+epsilon:
   pruned_upper=max(pruned_upper,-negub);continue
  C,constant=affine_envelope(a,b,ll,hh,chosen,competitor);mid=(ll[chosen]+hh[chosen])/2
  r=solve_block(a,b,ll,hh,-C,seconds=min(40,max(.1,seconds-(time.perf_counter()-start))),initial_states=states)
  nodes+=1
  if r['status']=='globally_infeasible':
   trace.append(dict(node=nodes,competitor=competitor,status='pruned_infeasible'));continue
  if r['status']!='numerical_optimum':
   heapq.heappush(queue,(negub,counter,competitor,ll,hh));counter+=1;unresolved=True;break
  states=np.unique(np.vstack([states,r['target_states']]),axis=0)
  t=r['pair_joint'];actual=float(info(t[competitor],a[competitor],b).mean()-info(t[chosen],a[chosen],b).mean())
  if actual>lower:lower=actual;witness=r;best_comp=competitor
  upper=min(-negub,float(np.nextafter(constant-r['global_lower'],np.inf)))
  trace.append(dict(node=nodes,competitor=competitor,status='bounded',primal_regret=actual,upper_bound=upper,
   LP_gap=r['gap'],incumbent=lower))
  if upper<=lower+epsilon:
   pruned_upper=max(pruned_upper,upper);continue
  errors=[]
  for row in [competitor,chosen]:
   for j in range(len(b)):
    if hh[row,j]-ll[row,j]<1e-12:continue
    if row==competitor:
     l,h=ll[row,j],hh[row,j];slope=(info(h,a[row],b[j])-info(l,a[row],b[j]))/(h-l)
     gap=info(l,a[row],b[j])+slope*(t[row,j]-l)-info(t[row,j],a[row],b[j])
    else:
     derivative_mid=derivative(mid[j],a[row],b[j]);gap=info(t[row,j],a[row],b[j])-info(mid[j],a[row],b[j])-derivative_mid*(t[row,j]-mid[j])
    errors.append((float(gap),row,j))
  if not errors:
   heapq.heappush(queue,(-upper,counter,competitor,ll,hh));counter+=1;unresolved=True;break
  _,row,j=max(errors);cut=(ll[row,j]+hh[row,j])/2
  for side in [0,1]:
   l2,h2=ll.copy(),hh.copy()
   if side==0:h2[row,j]=cut
   else:l2[row,j]=cut
   heapq.heappush(queue,(-upper,counter,competitor,l2,h2));counter+=1
 upper=max(lower,pruned_upper,-queue[0][0]) if queue else max(lower,pruned_upper)
 status='numerical_enclosure' if upper-lower<=epsilon+1e-10 else 'bounded_timeout' if not unresolved else 'bounded_solver_limit'
 return dict(status=status,chosen=chosen,worst_competitor=best_comp,lower=lower,upper=upper,gap=upper-lower,
  local_regret=float(local[chosen]),nodes=nodes,remaining_boxes=len(queue),seconds=time.perf_counter()-start,trace=trace,witness=witness)

def run():
 rows=[]
 for name in DESIGN['cases']:
  with np.load(OUT/f'exact_global_input_{name}.npz') as z:
   r=worst_regret(z['a'],z['b'],z['lo'],z['hi']);k=len(z['a']);m=len(z['b'])
  if 'witness' in r:
   w=r.pop('witness');np.savez_compressed(OUT/f'global_regret_witness_{name}.npz',target_states=w['target_states'],weights=w['weights'],candidate_joint=w['candidate_joint'],pair_joint=w['pair_joint'])
  (OUT/f'global_regret_trace_{name}.json').write_text(json.dumps(r,indent=2))
  row={key:value for key,value in r.items() if key!='trace'};row.update(case=name,candidates=k,targets=m);rows.append(row)
  pd.DataFrame(rows).to_csv(OUT/'global_regret_summary.csv',index=False)
  print('GLOBAL REGRET',name,r['status'],'gap',r.get('gap'),'nodes',r.get('nodes'),'seconds',r.get('seconds'),flush=True)
if __name__=='__main__':run()
