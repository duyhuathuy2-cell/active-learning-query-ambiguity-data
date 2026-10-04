"""Boundary, interruption, pruning and no-outcome-selection controls."""
import environment
from environment import ROOT
import numpy as np,json
from block_global import solve_block,OUT
from verified_global import verify_block_result
import global_regret as gr

def run():
 a=b=np.full(2,.5);lo=hi=np.full((2,2),.5);C=np.array([[1.,-.5],[-.2,.4]])
 cases=[]
 for delta in [0.,5e-9]:
  lo=np.full((2,2),.5);hi=lo.copy();lo[1,1]=hi[1,1]=.5-delta
  r=solve_block(a,b,lo,hi,C,initial_states=np.array([[0,0],[1,1]]))
  cert=verify_block_result(a,b,lo,hi,C,r)
  if delta:assert not cert['status'].startswith('verified_global_feasibility') and cert['status']!='verified_linear_enclosure'
  else:assert cert['status']=='verified_linear_enclosure'
  cases.append(dict(delta=delta,raw_status=r['status'],verification_status=cert['status']))
 boundary=solve_block(a,b,np.full((2,2),.5),np.full((2,2),.5),C,initial_states=np.array([[0,0],[1,1]]))
 corrupted=dict(boundary);corrupted['weights']=np.full_like(boundary['weights'],-1.)
 rejected=verify_block_result(a,b,np.full((2,2),.5),np.full((2,2),.5),C,corrupted)
 assert rejected['status']=='inconclusive_exact_feasibility'
 lo=np.full((2,2),.35);hi=np.full((2,2),.5);lo[1,1]=0;hi[1,1]=.15
 coarse=gr.worst_regret(a,b,lo,hi,epsilon=.01);fine=gr.worst_regret(a,b,lo,hi,epsilon=.0005)
 assert fine['upper']-fine['lower']<=.0005 and fine['upper']<=coarse['upper']+1e-8
 assert any(x['status']=='pruned_infeasible' for x in fine['trace'])
 original=gr.solve_block;calls=[0]
 def interrupt(*args,**kwargs):
  calls[0]+=1
  if calls[0]>1:return dict(status='inconclusive_timeout')
  return original(*args,**kwargs)
 gr.solve_block=interrupt
 try:stopped=gr.worst_regret(a,b,lo,hi,epsilon=.0005)
 finally:gr.solve_block=original
 assert stopped['status']=='bounded_solver_limit' and stopped['upper']>=fine['upper'] and stopped['remaining_boxes']>0
 record=dict(boundary_cases=cases,coarse_epsilon=.01,coarse_gap=coarse['gap'],fine_epsilon=.0005,fine_gap=fine['gap'],
  fine_nodes=fine['nodes'],globally_infeasible_children=sum(x['status']=='pruned_infeasible' for x in fine['trace']),
  forced_interruption_status=stopped['status'],forced_interruption_upper=stopped['upper'],unresolved_bounds_retained=True,
  corrupted_negative_joint_rejected=rejected['status'],
  raw_solver_statuses_are_explicitly_provisional=True)
 (OUT/'numerical_path_controls.json').write_text(json.dumps(record,indent=2))
 (OUT/'fine_coupled_path_trace.json').write_text(json.dumps({k:v for k,v in fine.items() if k!='witness'},indent=2))
 (OUT/'interrupted_path_trace.json').write_text(json.dumps({k:v for k,v in stopped.items() if k!='witness'},indent=2))
 print(json.dumps(record,indent=2))
if __name__=='__main__':run()
