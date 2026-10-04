"""Globally epsilon-minimax action selection with verified bounds for every action."""
import environment
from environment import ROOT
from fractions import Fraction as Q
import numpy as np,pandas as pd,json,time
from verified_global import fq,ends,info_iv,rational_feasible,OUT
from global_regret import worst_regret,candidate_bounds
DESIGN={'date':'2026-10-03','fixed_before_execution':True,'external_preregistration':False,
 'cases':['new_8_4_0','new_16_8_0','new_32_8_0','new_64_8_0','new_64_16_0'],
 'objective':'min over all candidate actions of complete-polytope worst information regret',
 'action_gap_bits':.001,'evaluation_gap_bits':.0004,'seconds_per_case':600,
 'bounds':'every action retains rectangular upper and rational feasible-world lower; complete global support evaluates only unresolved contenders',
 'no_unresolved_action_excluded':True}
p=OUT/'global_minimax_design.json'
if p.exists():assert json.loads(p.read_text())==DESIGN
else:p.write_text(json.dumps(DESIGN,indent=2))
def world_bounds(t,a,b):
 vals=np.array([ends(sum(info_iv(t[i][j],fq(a[i]),fq(b[j])) for j in range(len(b)))/len(b)) for i in range(len(a))])
 return np.maximum(0,np.nextafter(vals[:,0].max()-vals[:,1],-np.inf)),vals
def run():
 rows=[]
 for name in DESIGN['cases']:
  start=time.perf_counter()
  with np.load(OUT/f'exact_global_input_{name}.npz') as z:a,b,lo,hi=[z[k] for k in ['a','b','lo','hi']]
  L,U=candidate_bounds(a,b,lo,hi);upper=np.array([max(0,np.max(np.delete(U,i))-L[i]) for i in range(len(a))]);upper=np.nextafter(upper,np.inf);lower=np.zeros(len(a))
  with np.load(OUT/f'verified_anchor_{name}.npz') as z:anchor={k:z[k] for k in z.files}
  warm=anchor['target_states'];evaluated=[];trace=[];worlds=[]
  for prefix in ['verified_linear_certificate','verified_regret_certificate']:
   cert=json.loads((OUT/f'{prefix}_{name}.json').read_text());t=[[Q(x) for x in rr] for rr in cert['exact_pair_joint']]
   lb,vals=world_bounds(t,a,b);lower=np.maximum(lower,lb);worlds.append(dict(source=prefix,values=vals.tolist()))
   if 'regret' in prefix:upper[cert['chosen']]=min(upper[cert['chosen']],cert['solver_upper'])
  original=int(np.argmin(np.array([max(0,np.max(np.delete(U,i))-L[i]) for i in range(len(a))])))
  while upper.min()-lower.min()>.001 and time.perf_counter()-start<600:
   action=int(lower.argmin())
   remaining=600-(time.perf_counter()-start)
   r=worst_regret(a,b,lo,hi,epsilon=.0004,seconds=min(240,remaining),warm=warm,chosen_override=action)
   if 'witness' not in r:break
   witness=r.pop('witness');t,cert=rational_feasible(a,b,lo,hi,witness,anchor);lb,vals=world_bounds(t,a,b)
   lower=np.maximum(lower,lb);upper[action]=min(upper[action],float(np.nextafter(r['upper']+1e-8,np.inf)))
   np.savez_compressed(OUT/f'global_minimax_witness_{name}_action{action}.npz',target_states=witness['target_states'],weights=witness['weights'],candidate_joint=witness['candidate_joint'])
   cert.update(case=name,action=action,regret_lower=float(lower[action]),regret_upper=float(upper[action]),solver_trace=r)
   (OUT/f'global_minimax_action_certificate_{name}_{action}.json').write_text(json.dumps(cert,indent=2))
   evaluated.append(action);worlds.append(dict(source=f'action{action}',values=vals.tolist()))
   trace.append(dict(action=action,regret_lower=float(lower[action]),regret_upper=float(upper[action]),optimal_gap=float(upper.min()-lower.min()),nodes=r['nodes']))
   print('GLOBAL MINIMAX',name,'action',action,'gap',upper.min()-lower.min(),flush=True)
   warm=np.unique(np.vstack([warm,witness['target_states']]),axis=0)
   if len(evaluated)>2*len(a):break
  selected=int(upper.argmin());gap=float(upper[selected]-lower.min());status='verified_epsilon_global_minimax' if gap<=.001 else 'bounded_inconclusive'
  cert=dict(case=name,status=status,selected=selected,rectangular_choice=original,optimal_regret_lower=float(lower.min()),optimal_regret_upper=float(upper.min()),
   action_suboptimality_upper=gap,lower_all_actions=lower.tolist(),upper_all_actions=upper.tolist(),evaluated_actions=evaluated,worlds=worlds,trace=trace,
   criterion='upper[selected] - min(lower over ALL actions) <= .001 bits',seconds=time.perf_counter()-start)
  (OUT/f'global_minimax_certificate_{name}.json').write_text(json.dumps(cert,indent=2))
  rows.append({k:cert[k] for k in ['case','status','selected','rectangular_choice','optimal_regret_lower','optimal_regret_upper','action_suboptimality_upper','seconds']})
  pd.DataFrame(rows).to_csv(OUT/'global_minimax_summary.csv',index=False)
  print('GLOBAL MINIMAX FINAL',name,status,'choice',selected,'local',original,'gap',gap,flush=True)
if __name__=='__main__':run()
