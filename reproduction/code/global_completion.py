"""Diagnostic analyses of all saved settings; no new neural inference or seed selection."""
import environment
from environment import ROOT
from pathlib import Path
import numpy as np,pandas as pd,json,itertools,time,warnings
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from method import scores,interval,entropy
R=ROOT.parent/'data/results/benchmark'




def lp_template(k=8,m=48):
    states=np.array(list(itertools.product([0,1],repeat=k)),dtype=float);n=len(states);v=n*(m+1)
    er=[];ec=[];ev=[]
    def add_e(row,cols,vals):er.extend([row]*len(cols));ec.extend(cols);ev.extend(vals)
    add_e(0,np.arange(n),np.ones(n))
    for i in range(k):idx=np.flatnonzero(states[:,i]);add_e(i+1,idx,np.ones(len(idx)))
    for j in range(m):add_e(1+k+j,np.arange(n*(j+1),n*(j+2)),np.ones(n))
    E=coo_matrix((ev,(er,ec)),shape=(1+k+m,v)).tocsr()
    ur=[];uc=[];uv=[];ub=[];row=0
    for j in range(m):
        for s in range(n):ur.extend([row,row]);uc.extend([s,n*(j+1)+s]);uv.extend([-1,1]);ub.append(0.);row+=1
        for i in range(k):
            idx=np.flatnonzero(states[:,i]);cols=n*(j+1)+idx
            for sign in [1,-1]:ur.extend([row]*len(idx));uc.extend(cols);uv.extend([sign]*len(idx));ub.append(0.);row+=1
    U=coo_matrix((uv,(ur,uc)),shape=(row,v)).tocsr()
    return states,E,U,np.asarray(ub)

def global_checks(limit=None,shard=None,shards=6):
    states,E,U,template=lp_template();n=len(states);m=48;k=8
    out=R/('global_fullpool.csv' if shard is None else f'global_fullpool_part{shard}.csv');rows=pd.read_csv(out).to_dict('records') if out.exists() else [];done={(r['dataset'],r['model'],r['seed'],r['repair']) for r in rows}
    cases=pd.read_csv(R/'audit_cases.csv').sort_values(['model','dataset','seed']);cases=cases[cases.model.isin(['tabpfn_1','tabpfn_8','tabicl_8','logistic','forest'])]
    if shard is not None:cases=cases.iloc[np.arange(len(cases))%shards==shard]
    files=R/'global_completions';files.mkdir(exist_ok=True);start=time.monotonic();count=0
    for _,case in cases.iterrows():
        name=f'{case.dataset}_{case.model}_{case.seed}_initial.npz'
        with np.load(R/'cache'/name) as z:a=z['base'][:8];b=z['base'][8:];q0=z['branches'][:,0];q1=z['branches'][:,1]
        _,_,lo,hi=interval(a[:,None],b[None,:],q0,q1)
        for kind,ll,hh in [('minima',lo,hi),('endpoint_lo',lo,lo),('endpoint_hi',hi,hi)]:
            key=(case.dataset,case.model,int(case.seed),kind)
            if key in done:continue
            bound=template.copy()
            for j in range(m):
                offset=j*(n+2*k)+n;bound[offset:offset+2*k:2]=hh[:,j];bound[offset+1:offset+2*k:2]=-ll[:,j]
            t0=time.monotonic()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',category=__import__('scipy').optimize.OptimizeWarning)
                res=linprog(np.zeros(E.shape[1]),A_ub=U,b_ub=bound,A_eq=E,b_eq=np.r_[1,a,b],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':30.,'threads':1})
            row=dict(dataset=case.dataset,model=case.model,seed=int(case.seed),repair=kind,feasible=bool(res.success),status=int(res.status),seconds=time.monotonic()-t0,endpoint_flip=bool(case.endpoint_flip))
            if res.success:
                w=res.x[:n];u=res.x[n:].reshape(m,n);t=states.T@u.T
                vals=(entropy(b)[None,:]-(1-a[:,None])*entropy(np.divide(b[None,:]-t,1-a[:,None],out=np.zeros_like(t),where=(1-a[:,None])!=0))-a[:,None]*entropy(np.divide(t,a[:,None],out=np.zeros_like(t),where=a[:,None]!=0))).mean(1)
                order=np.argsort(vals);row.update(eq_residual=float(np.max(abs(E@res.x-np.r_[1,a,b]))),inequality_violation=float(max(0,np.max(U@res.x-bound))),winner=int(order[-1]),winner_margin=float(vals[order[-1]]-vals[order[-2]]))
                np.savez_compressed(files/f'{case.dataset}_{case.model}_{case.seed}_{kind}.npz',weights=w,target_joint=u,pair_joint=t,information=vals)
            rows.append(row);done.add(key);count+=1
            if count%3==0:
                pd.DataFrame(rows).to_csv(out,index=False);print('FULLPOOL',len(rows),'LPs',round(time.monotonic()-start,1),'seconds',flush=True)
            if limit and count>=limit:pd.DataFrame(rows).to_csv(out,index=False);return
    pd.DataFrame(rows).to_csv(out,index=False)

if __name__=='__main__':
 global_checks()
