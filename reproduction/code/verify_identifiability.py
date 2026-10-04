"""Verify the quantitative two-world result and aggregate every configured model."""
from pathlib import Path
import itertools, json, sys, platform
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from method import interval, entropy

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'data/results/identifiability';OUT.mkdir(exist_ok=True)

def mutual_information(p):
    pa=p.sum(1);pb=p.sum(0)
    keep=p>0
    return float(np.sum(p[keep]*np.log2((p/(pa[:,None]*pb[None,:]))[keep])))

def main():
    a=b=.5
    branches=[(.1,.6),(.8,.6)]
    limits=[tuple(float(x) for x in interval(a,b,q0,q1)) for q0,q1 in branches]
    L1,U1,lo1,hi1=limits[0];L2,U2,lo2,hi2=limits[1]
    states=list(itertools.product([0,1],repeat=3))
    rows=[];world_values=[]
    for name,ts in [('W1',(lo1,lo2)),('W2',(hi1,a*b))]:
        masses=[]
        for A1,A2,B in states:
            p=.5
            for A,t in zip([A1,A2],ts):
                prob=2*t if B else 1-2*t
                p*=prob if A else 1-prob
            masses.append(p);rows.append({'world':name,'A1':A1,'A2':A2,'B':B,'mass':p})
        assert abs(sum(masses)-1)<1e-14 and min(masses)>=0
        values=[]
        for i,t in enumerate(ts):
            pair=np.zeros((2,2))
            for st,p in zip(states,masses):pair[st[i],st[2]]+=p
            assert np.max(abs(pair.sum(0)-.5))<1e-14
            assert np.max(abs(pair.sum(1)-.5))<1e-14
            assert abs(pair[1,1]-t)<1e-14
            q0,q1=branches[i]
            cost=abs(t-a*q1)+abs(t-(b-(1-a)*q0))
            assert abs(cost-abs(b-(1-a)*q0-a*q1))<1e-14
            values.append(mutual_information(pair))
        world_values.append(values)
    assert np.max(abs(np.array(world_values)-[[L1,U2],[U1,L2]]))<1e-14
    d1=U2-L1;d2=U1-L2;pstar=d2/(d1+d2);bound=d1*d2/(d1+d2)
    opt=minimize_scalar(lambda p:max(p*d1,(1-p)*d2),bounds=(0,1),method='bounded',options={'xatol':1e-14})
    assert abs(opt.x-pstar)<1e-7 and abs(opt.fun-bound)<1e-8
    pd.DataFrame(rows).to_csv(OUT/'two_world_joint_atoms.csv',index=False)
    result={'binary_information_bounds':limits,'world_information_values':world_values,
            'world_gaps':[d1,d2],'deterministic_minimax_regret_bits':min(d1,d2),
            'randomized_minimax_regret_bits':bound,'optimal_probability_candidate_1':pstar,
            'independent_optimizer_value':float(opt.fun),'joint_atom_checks_passed':True,
            'scope':'Minimum-TV compatible two-world information objective; no prediction-loss guarantee.',
            'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
    (OUT/'two_world_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    audit=pd.read_csv(ROOT.parent/'data/results/benchmark/audit_cases.csv')
    global_summary=pd.read_csv(ROOT.parent/'data/results/benchmark/global_fullpool_summary.csv').set_index('model')
    summary=[]
    for model,g in audit.groupby('model'):
        summary.append({'model':model,'settings':len(g),'local_endpoint_disagreements':int(g.endpoint_flip.sum()),
           'strict_certificates':int(g.certified.sum()),'globally_feasible':int(global_summary.loc[model,'minima_feasible']),
           'strict_global_reversal_witnesses':int(global_summary.loc[model,'strict_global_reversal_witnesses'])})
    pd.DataFrame(summary).to_csv(OUT/'all_model_audit_summary.csv',index=False)
    print(json.dumps(result,indent=2));print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
