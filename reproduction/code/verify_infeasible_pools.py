"""Quantify minimum interval expansion needed by both infeasible full-pool cases."""
from global_completion import lp_template,ROOT,interval
from scipy.sparse import hstack,csr_matrix
from scipy.optimize import linprog
import numpy as np,pandas as pd,json,warnings
R=ROOT.parent/'data/results/benchmark';states,E,U,template=lp_template();n=256;k=8;m=48;column=np.zeros(U.shape[0])
for j in range(m):column[j*(n+2*k)+n:(j+1)*(n+2*k)]=-1
UU=hstack([U,csr_matrix(column[:,None])]).tocsr();EE=hstack([E,csr_matrix((E.shape[0],1))]).tocsr();objective=np.zeros(EE.shape[1]);objective[-1]=1;rows=[]
df=pd.read_csv(R/'global_fullpool.csv')
for r in df[(df.repair=='minima')&~df.feasible].itertuples():
    with np.load(R/'cache'/f'{r.dataset}_{r.model}_{r.seed}_initial.npz') as z:
        a=z['base'][:8];b=z['base'][8:];_,_,lo,hi=interval(a[:,None],b[None,:],z['branches'][:,0],z['branches'][:,1])
    bound=template.copy()
    for j in range(m):
        off=j*(n+2*k)+n;bound[off:off+2*k:2]=hi[:,j];bound[off+1:off+2*k:2]=-lo[:,j]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res=linprog(objective,A_ub=UU,b_ub=bound,A_eq=EE,b_eq=np.r_[1,a,b],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'threads':1})
    assert res.success
    rows.append(dict(dataset=r.dataset,model=r.model,seed=int(r.seed),minimum_uniform_pair_interval_expansion=float(res.fun),max_constraint_residual=float(max(abs(EE@res.x-np.r_[1,a,b]).max(),max(0,(UU@res.x-bound).max())))))
(R/'global_failures_slack.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
