"""Independent residual validation and aggregation of six disjoint numerical jobs."""
from pathlib import Path
import numpy as np,pandas as pd,json,itertools
import sys
from method import interval
ROOT=Path(__file__).resolve().parents[1];R=ROOT.parent/'data/results/benchmark'
df=pd.read_csv(R/'global_fullpool.csv')
assert len(df)==1800 and not df.duplicated(['dataset','model','seed','repair']).any()
assert set(df.status)<= {0,2},'Timeout or solver-error status must remain explicitly unresolved.'
df=df.sort_values(['model','dataset','seed','repair']);df.to_csv(R/'global_fullpool.csv',index=False)
states=np.array(list(itertools.product([0,1],repeat=8)));maxres=0.;nchecked=0
for r in df[df.feasible].itertuples():
    with np.load(R/'cache'/f'{r.dataset}_{r.model}_{r.seed}_initial.npz') as z:
        a=z['base'][:8];b=z['base'][8:];_,_,lo,hi=interval(a[:,None],b[None,:],z['branches'][:,0],z['branches'][:,1])
    if r.repair=='endpoint_lo':hi=lo
    if r.repair=='endpoint_hi':lo=hi
    with np.load(R/'global_completions'/f'{r.dataset}_{r.model}_{r.seed}_{r.repair}.npz') as z:
        w,u,t=z['weights'],z['target_joint'],z['pair_joint'];tt=states.T@u.T
        residual=max(abs(w.sum()-1),np.max(abs(states.T@w-a)),np.max(abs(u.sum(1)-b)),np.max(abs(t-tt)),max(0,-w.min(),-u.min(),(u-w).max(),(lo-t).max(),(t-hi).max()))
        assert residual<1.1e-8,(r,residual);maxres=max(maxres,float(residual));nchecked+=1
rows=[]
for model,g in df.groupby('model'):
    pivot=g.pivot(index=['dataset','seed'],columns='repair',values=['feasible','winner','winner_margin'])
    witness=(pivot['feasible']['endpoint_lo']&pivot['feasible']['endpoint_hi']&(pivot['winner']['endpoint_lo']!=pivot['winner']['endpoint_hi'])&(pivot['winner_margin']['endpoint_lo']>1e-7)&(pivot['winner_margin']['endpoint_hi']>1e-7))
    local=g[g.repair=='minima'].set_index(['dataset','seed']).endpoint_flip
    assert not (witness&~local).any()
    rows.append(dict(model=model,settings=len(pivot),minima_feasible=int(pivot['feasible']['minima'].sum()),lower_endpoints_feasible=int(pivot['feasible']['endpoint_lo'].sum()),upper_endpoints_feasible=int(pivot['feasible']['endpoint_hi'].sum()),local_endpoint_disagreements=int(local.sum()),strict_global_reversal_witnesses=int(witness.sum()),local_disagreements_without_strict_global_witness=int((local&~witness).sum())))
summary=pd.DataFrame(rows);summary.to_csv(R/'global_fullpool_summary.csv',index=False)
(R/'global_fullpool_qa.json').write_text(json.dumps(dict(programs=1800,settings=600,feasible_completions_checked=nchecked,max_constraint_residual=maxres,status_counts=df.status.value_counts().to_dict(),primal_dual_feasibility_tolerance=1e-9,winner_margin_threshold_bits=1e-7,interpretation='Numerical sufficient witnesses; failure of these endpoint witnesses is not proof of global ranking uniqueness.'),indent=2));print(summary.to_string(index=False));print('Maximum independently checked residual',maxres)
