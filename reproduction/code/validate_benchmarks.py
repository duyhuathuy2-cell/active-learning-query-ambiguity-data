"""Cache-only independent review audit. Does not import neural packages."""
from pathlib import Path
import itertools,json,ast,types
import numpy as np,pandas as pd
from method import scores,interval,entropy
ROOT=Path(__file__).resolve().parents[1];R=ROOT.parent/'data/results/benchmark'

def data_functions(filename,names,extra):
    # Load only these reviewed deterministic data definitions, excluding all
    # neural imports and execution from the original inference runners.
    tree=ast.parse((Path(__file__).resolve().parent/filename).read_text(encoding='utf8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(selected)==len(names)
    space=dict(np=np,**extra)
    exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),space)
    return space

def replay_check():
    from sklearn.datasets import load_breast_cancer,load_iris,load_wine,load_digits,make_moons
    definitions=data_functions('run_experiments.py',{'datasets','split'},locals())
    data=definitions['datasets']();split=definitions['split'];maximum=0.;steps=0
    seq=pd.read_csv(R/'sequential.csv')
    for (task,model,seed,policy),g in seq.groupby(['dataset','model','seed','policy']):
        X,y=data[task];tr,pool,target=split(y,int(seed));previous=[];rng=np.random.default_rng(70000+int(seed))
        for row in g.sort_values('budget').itertuples():
            key='-'.join(map(str,sorted(previous))) or 'initial'
            with np.load(R/'cache'/f'{task}_{model}_{seed}_{key}.npz') as z:
                s=scores(z['base'],z['branches'],len(z['remain']));cert=np.flatnonzero(s['certified'])
                if policy=='random':pick=int(rng.integers(len(z['remain'])))
                elif policy=='minimax':pick=int(np.argmin(s['regret']))
                elif policy=='certify_entropy':pick=int(cert[0]) if len(cert) else int(np.argmax(s['entropy']))
                else:pick=int(np.argmax(s[policy]))
                pos=int(z['remain'][pick]);assert json.loads(row.selected_positions)==previous+[pos]
                assert np.array_equal(z['train'],np.r_[tr,pool[sorted(previous)]])
                assert not(set(tr)&set(pool) or set(tr)&set(target) or set(pool)&set(target))
                maximum=max(maximum,abs(float(np.mean((z['branches'][pick,y[pool[pos]]]-y[target])**2))-row.brier));previous.append(pos);steps+=1
    expanded=data_functions('expanded.py',{'data','split'},{'old':types.SimpleNamespace(datasets=definitions['datasets'])})
    for model in ['tabpfn_8','logistic']:
        seq=pd.read_csv(R.parent/'multiclass_and_calibration'/f'expanded_{model}_replay.csv')
        for (task,seed,policy),g in seq.groupby(['dataset','seed','policy']):
            X,y=expanded['data'](task,int(seed));tr,pool,target=expanded['split'](y,int(seed));previous=[]
            for row in g.sort_values('budget').itertuples():
                key='-'.join(map(str,sorted(previous))) or 'initial'
                with np.load(R.parent/'multiclass_and_calibration/expanded_cache'/f'{task}_{model}_{seed}_{key}.npz') as z:
                    s=scores(z['base'],z['branches'],len(z['remain']));cert=np.flatnonzero(s['certified'])
                    pick=int(np.argmin(s['regret'])) if policy=='minimax' else int(cert[0]) if policy=='certify_entropy' and len(cert) else int(np.argmax(s['entropy' if policy=='certify_entropy' else policy]))
                    pos=int(z['remain'][pick]);assert np.array_equal(z['train'],np.r_[tr,pool[sorted(previous)]])
                    previous.append(pos);assert json.loads(row.selected)==sorted(previous)
                    maximum=max(maximum,abs(float(np.mean((z['branches'][pick,y[pool[pos]]]-y[target])**2))-row.brier));steps+=1
    assert steps==4800 and maximum<1e-12
    return steps,maximum
def main():
    candidates=pd.read_csv(R/'audit_candidates.csv');max_difference=0.
    for (task,model,seed),g in candidates.groupby(['dataset','model','seed']):
        with np.load(R/'cache'/f'{task}_{model}_{seed}_initial.npz') as z:
            ss=scores(z['base'],z['branches'],8);gg=g.sort_values('position')
            for key in ['raw','kl','lower','upper','regret']:
                max_difference=max(max_difference,float(np.max(abs(ss[key]-gg[key]))))
    completion=pd.read_csv(R/'global_fullpool.csv');states=np.array(list(itertools.product([0,1],repeat=8)),float)
    residual=0.;checked=0
    for row in completion[completion.feasible].itertuples():
        with np.load(R/'cache'/f'{row.dataset}_{row.model}_{row.seed}_initial.npz') as z:a=z['base'][:8];b=z['base'][8:];q0=z['branches'][:,0];q1=z['branches'][:,1]
        _,_,lo,hi=interval(a[:,None],b[None,:],q0,q1)
        if row.repair=='endpoint_lo':hi=lo
        if row.repair=='endpoint_hi':lo=hi
        with np.load(R/'global_completions'/f'{row.dataset}_{row.model}_{row.seed}_{row.repair}.npz') as z:
            w=z['weights'];v=z['target_joint'];t=states.T@v.T
            violations=[abs(w.sum()-1),np.max(abs(states.T@w-a)),np.max(abs(v.sum(1)-b)),np.max(v-w[None,:]),-w.min(),-v.min(),np.max(lo-t),np.max(t-hi),np.max(abs(t-z['pair_joint']))]
            residual=max(residual,*map(float,violations));checked+=1
    cal=pd.read_csv(R.parent/'multiclass_and_calibration/calibration_test.csv');joint=pd.read_csv(R.parent/'multiclass_and_calibration/joint_calibration_test.csv')
    assert not cal[cal.coverage].bound_violated.any()
    assert not (joint.joint_covered&~joint.score_covered).any()
    cert=pd.read_csv(R.parent/'multiclass_and_calibration/multiclass_verified_candidates.csv')
    assert int(cert.certified.sum())==6 and int((cert.raw_selected&cert.excluded).sum())==4
    assert max_difference<1e-12 and residual<1e-9
    steps,loss_error=replay_check()
    result=dict(audit_cases=600,candidates=4800,maximum_score_difference=max_difference,feasible_fullpool_completions_checked=checked,maximum_independent_completion_residual=residual,calibrated_covered_values_obey_regret_bounds=True,verified_multiclass_certificates=6,verified_raw_exclusions=4,replayed_query_steps=steps,maximum_replayed_loss_error=loss_error,mode='Cached-result validation; no model inference')
    (ROOT/'output/benchmark_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
