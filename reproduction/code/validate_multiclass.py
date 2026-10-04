import environment
from environment import ROOT
import json,itertools,time
import numpy as np,pandas as pd
from scipy.optimize import linprog
from method import interval,entropy,scores
from expanded import data,split
from joint_calibration import continuity
R=ROOT.parent/'data/results/multiclass_and_calibration'

def main():
    start=time.perf_counter();rng=np.random.default_rng(460000);distance_error=0.;continuity_excess=0.
    for rep in range(1000):
        a,b,q0,q1=rng.uniform(.001,.999,4);p=rng.dirichlet(np.ones(4));_,_,lo,hi=interval(a,np.array(b),np.array(q0),np.array(q1))
        roots=np.array([p[0]-1+a+b,b-p[1],a-p[2],p[3]])
        t=np.clip(np.median(roots),lo,hi);distance=.5*np.abs(t-roots).sum()
        # Independent epigraph program for the four absolute deviations.
        A=[];rhs=[]
        for i,z in enumerate(roots):
            row=np.zeros(5);row[0]=1;row[1+i]=-1;A.append(row);rhs.append(z)
            row=np.zeros(5);row[0]=-1;row[1+i]=-1;A.append(row);rhs.append(-z)
        opt=linprog([0,.5,.5,.5,.5],A_ub=A,b_ub=rhs,bounds=[(float(lo),float(hi))]+[(0,None)]*4,method='highs')
        assert opt.success;distance_error=max(distance_error,abs(distance-opt.fun))
        Q=np.array([1-a-b+t,b-t,a-t,t]);true_marg_a=p[2]+p[3];true_marg_b=p[1]+p[3]
        def info(cells):
            from scipy.special import xlogy
            cells=np.maximum(cells,0);row=np.array([cells[0]+cells[1],cells[2]+cells[3]]);col=np.array([cells[0]+cells[2],cells[1]+cells[3]])
            return float(np.sum(xlogy(cells,cells/np.outer(row,col).ravel()))/np.log(2))
        continuity_excess=max(continuity_excess,abs(info(p)-info(Q))-continuity(distance))
    assert distance_error<1e-10 and continuity_excess<1e-10
    losses=0.;score_error=0.;steps=0
    for kind in ['tabpfn_8','logistic']:
        audit=pd.read_csv(R/f'expanded_{kind}_audit.csv');seq=pd.read_csv(R/f'expanded_{kind}_replay.csv')
        assert len(audit)==960 and not audit.duplicated(['dataset','seed','candidate']).any()
        assert len(seq)==(150 if kind=='tabpfn_8' else 450)
        for (name,seed),g in audit.groupby(['dataset','seed']):
            X,y=data(name,int(seed));tr,pool,target=split(y,int(seed))
            with np.load(R/'expanded_cache'/f'{name}_{kind}_{seed}_initial.npz') as z:
                assert np.array_equal(z['train'],tr) and np.array_equal(z['pool'],pool) and np.array_equal(z['target'],target)
                assert len(set(tr)&set(pool))==len(set(tr)&set(target))==len(set(pool)&set(target))==0
                ss=scores(z['base'],z['branches'],32);gg=g.sort_values('candidate')
                for key in ['raw','kl','lower','upper','regret']:
                    score_error=max(score_error,float(np.max(np.abs(ss[key]-gg[key]))))
                actual=np.mean((z['branches'][np.arange(32),y[pool]]-y[target])**2,1)
                losses=max(losses,float(np.max(abs(actual-gg.actual_brier))))
        for (name,seed,policy),g in seq.groupby(['dataset','seed','policy']):
            X,y=data(name,int(seed));tr,pool,target=split(y,int(seed));previous=[]
            for row in g.sort_values('budget').itertuples():
                key='-'.join(map(str,sorted(previous))) or 'initial'
                with np.load(R/'expanded_cache'/f'{name}_{kind}_{seed}_{key}.npz') as z:
                    ss=scores(z['base'],z['branches'],len(z['remain']));cert=np.flatnonzero(ss['certified'])
                    pick=int(np.argmin(ss['regret'])) if policy=='minimax' else int(cert[0]) if policy=='certify_entropy' and len(cert) else int(np.argmax(ss['entropy' if policy=='certify_entropy' else policy]))
                    pos=int(z['remain'][pick]);actual=z['branches'][pick,y[pool[pos]]];previous.append(pos)
                    assert json.loads(row.selected)==sorted(previous) and len(set(previous))==row.budget
                    assert np.array_equal(z['train'],np.r_[tr,pool[sorted(previous[:-1])]])
                    losses=max(losses,abs(float(np.mean((actual-y[target])**2))-row.brier));steps+=1
    assert losses<1e-12 and score_error<1e-12
    mult=pd.concat([pd.read_csv(R/f'multiclass_{m}_cases.csv') for m in ['tabpfn_8','logistic']])
    control=pd.read_csv(R/'multiclass_controls.csv')
    assert len(mult)==30 and len(control)==100
    assert mult.max_lower_gap.max()<1.1e-8 and mult.max_vertex_residual.max()<1e-10 and mult.max_cost_error.max()<1e-10
    cal=pd.read_csv(R/'calibration_test.csv');joint=pd.read_csv(R/'joint_calibration_test.csv')
    assert len(cal)==216000 and len(joint)==36000
    assert not cal.duplicated(['pool','noise','marginal_noise','seed','method','calibrated']).any()
    assert np.isfinite(cal.select_dtypes('number')).all().all()
    # Any covered score vector must satisfy the selected-action regret bound.
    assert not cal[cal.coverage].bound_violated.any()
    assert not joint[joint.joint_covered].bound_violated.any()
    assert not (joint.joint_covered & ~joint.score_covered).any()
    qa=dict(expanded_settings=60,expanded_candidate_rows=1920,expanded_replayed_steps=steps,multiclass_settings=30,multiclass_pairs=1920,multiclass_random_controls=100,calibration_test_records=len(cal),joint_calibration_test_records=len(joint),maximum_replayed_loss_error=losses,maximum_recomputed_score_error=score_error,maximum_joint_distance_LP_error=distance_error,maximum_continuity_bound_excess=continuity_excess,multiclass_maximum_lower_gap=float(mult.max_lower_gap.max()),multiclass_maximum_probability_residual=float(mult.max_vertex_residual.max()),multiclass_maximum_cost_error=float(mult.max_cost_error.max()),all_covered_actions_obey_regret_bounds=True,seconds=time.perf_counter()-start)
    (R/'validate_multiclass.json').write_text(json.dumps(qa,indent=2));print(json.dumps(qa,indent=2))
if __name__=='__main__':main()
