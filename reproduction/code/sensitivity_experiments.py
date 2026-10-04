"""Frozen KBS diagnostic extension; never changes original outcomes or policies."""
from pathlib import Path
import json,platform,sys,time,importlib.metadata as meta
import numpy as np
import pandas as pd
from method import entropy,interval,decision_bounds

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'data/results/sensitivity'
OUT.mkdir(parents=True,exist_ok=True)
DESIGN={
 'frozen_before_outputs':True,'external_preregistration':False,
 'purpose':'cached sensitivity and measured local postprocessing; no new neural inference or final-test tuning',
 'radius_grid':[0,.001,.005,.01,.025,.05],
 'cached_cases':'all 600 original interfaces, all five model configurations',
 'calibration_prefixes':[512,2048,8192,16384],
 'family_size':72,'alpha':.05,'thresholds':[.025,.05,.1,.2,.5,1.],
 'scale_K':[8,32,64],'scale_M':[8,48,128],
 'scale_seeds':[951000,951001,951002],'scale_timed_batches':21,
 'scale_calls_per_batch':50,'scale_warmup_calls':5,
 'storage':'actual input plus returned output array bytes; not process peak RAM',
 'timing':'repair intervals, information averaging and local decision bounds; excludes model calls and KL comparator',
 'statistical_units':'nested calibration prefixes and overlapping cached splits are descriptive; no new simultaneous family or deployment claim'}
dp=OUT/'design.json'
if dp.exists():assert json.loads(dp.read_text())==DESIGN
else:dp.write_text(json.dumps(DESIGN,indent=2),encoding='utf8')

def local(base,branches,k,epsilon=0):
    L,U,lo,hi=interval(base[:k,None],base[None,k:],branches[:,0],branches[:,1],epsilon)
    lower,upper=L.mean(1),U.mean(1)
    regret,strict,possible=decision_bounds(lower,upper)
    return L,U,lo,hi,lower,upper,regret,strict,possible

def checks_and_radius():
    cases=pd.read_csv(ROOT.parent/'data/results/benchmark/audit_cases.csv')
    rows=[];eq=[]
    for r in cases.itertuples():
        with np.load(ROOT.parent/'data/results/benchmark/cache'/f'{r.dataset}_{r.model}_{r.seed}_initial.npz') as z:
            base,br=z['base'],z['branches']
        p=base[:8];he=entropy(p);lc=1-np.maximum(p,1-p);margin=-np.abs(2*p-1)
        # Numerical decisions and analytic monotonicity together justify equivalence.
        picks=[int(np.argmax(x)) for x in (he,lc,margin)]
        assert len(set(picks))==1
        eq.append(dict(dataset=r.dataset,model=r.model,seed=r.seed,entropy=picks[0],least_confidence=picks[1],margin=picks[2]))
        for eps in DESIGN['radius_grid']:
            *_,L,U,R,S,P=local(base,br,8,eps)
            brute=np.array([max(0,np.max(np.delete(U,i))-L[i]) for i in range(8)])
            assert np.array_equal(R,brute)
            rows.append(dict(dataset=r.dataset,model=r.model,seed=r.seed,excess_tv=eps,
              possible_winners=int(P.sum()),strict_winner=bool(S.any()),mean_width=float((U-L).mean()),
              minimax_regret=float(R.min())))
    df=pd.DataFrame(rows);df.to_csv(OUT/'radius_cases.csv',index=False)
    df.groupby(['model','excess_tv']).agg(settings=('seed','size'),strict_count=('strict_winner','sum'),
      median_possible=('possible_winners','median'),mean_possible=('possible_winners','mean'),
      median_width=('mean_width','median'),median_minimax_regret=('minimax_regret','median')).reset_index().to_csv(OUT/'radius_summary.csv',index=False)
    pd.DataFrame(eq).to_csv(OUT/'baseline_equivalence.csv',index=False)
    # Degenerate, singleton and tied upper endpoints for the implementation change.
    for k in (1,2,8,64):
        rng=np.random.default_rng(128+k)
        for _ in range(100):
            L=rng.random(k);U=L+rng.random(k)
            if k>1:U[:2]=max(U)
            R,S,P=decision_bounds(L,U)
            if k>1:
                B=np.array([max(0,np.max(np.delete(U,i))-L[i]) for i in range(k)])
                BS=np.array([L[i]>np.max(np.delete(U,i))+1e-9 for i in range(k)])
                assert np.array_equal(R,B) and np.array_equal(S,BS)
            else:assert R[0]==0 and S[0] and P[0]
    return len(eq)

def calibration_prefixes():
    with np.load(ROOT.parent/'data/results/global_and_prediction/final_safe_outcomes_compact.npz') as z:
        df=pd.DataFrame({key:z[key] for key in z.files})
    original=pd.read_csv(ROOT.parent/'data/results/global_and_prediction/final_safe_options.csv')
    rows=[];err=0.;lam=np.log(2*72/.05)
    cal=df[df.stage=='calibration']
    for (task,baseline),g in cal.groupby(['dataset','baseline']):
        g=g.sort_values('episode')
        for n in DESIGN['calibration_prefixes']:
            s=g.iloc[:n]
            for c in DESIGN['thresholds']:
                gain=np.where(s.envelope<=c,s.observed_challenger_gain,0)
                variance=gain.var(ddof=1);rad=np.sqrt(2*variance*lam/n)+14*c*lam/(3*(n-1))
                lb=float(gain.mean()-rad)
                rows.append(dict(dataset=task,baseline=baseline,n=n,threshold=c,
                  mean_gain=float(gain.mean()),lower_bound=lb,radius=float(rad),positive=lb>0))
                if n==16384:
                    op=original[(original.dataset==task)&(original.baseline==baseline)&(original.threshold==c)].iloc[0]
                    err=max(err,abs(lb-op.certified_mean_lower_bound))
    assert err<1e-12
    tab=pd.DataFrame(rows);tab.to_csv(OUT/'calibration_prefix_options.csv',index=False)
    best=tab.groupby(['dataset','baseline','n']).lower_bound.max().reset_index()
    best['positive']=best.lower_bound>0;best.to_csv(OUT/'calibration_prefix_best.csv',index=False)
    best.groupby('n').agg(pairs=('positive','size'),positive_pairs=('positive','sum')).reset_index().to_csv(OUT/'calibration_size_summary.csv',index=False)
    return err

def synthetic_interface(k,m,seed):
    rng=np.random.default_rng(seed+1000*k+m)
    weights=rng.dirichlet(np.ones(16));p=rng.beta(.8,.8,(16,k+m))
    base=weights@p;a,b=base[:k],base[k:]
    t=np.einsum('s,si,sj->ij',weights,p[:,:k],p[:,k:])
    q1=t/a[:,None];q0=(b[None,:]-t)/(1-a[:,None])
    br=np.stack((q0,q1),axis=1)
    br=np.clip(br+rng.normal(0,.03,br.shape),1e-6,1-1e-6)
    return base,br

def scale():
    rows=[]
    for k in DESIGN['scale_K']:
        for m in DESIGN['scale_M']:
            for seed in DESIGN['scale_seeds']:
                base,br=synthetic_interface(k,m,seed)
                for _ in range(DESIGN['scale_warmup_calls']):result=local(base,br,k)
                for batch in range(DESIGN['scale_timed_batches']):
                    start=time.perf_counter_ns()
                    for _ in range(DESIGN['scale_calls_per_batch']):result=local(base,br,k)
                    ns=(time.perf_counter_ns()-start)/DESIGN['scale_calls_per_batch']
                    rows.append(dict(K=k,M=m,seed=seed,batch=batch,milliseconds=ns/1e6,
                      input_bytes=base.nbytes+br.nbytes,output_bytes=sum(x.nbytes for x in result)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'local_runtime_batches.csv',index=False)
    df.groupby(['K','M']).agg(measurements=('milliseconds','size'),median_ms=('milliseconds','median'),
      q25_ms=('milliseconds',lambda x:x.quantile(.25)),q75_ms=('milliseconds',lambda x:x.quantile(.75)),
      input_bytes=('input_bytes','first'),output_bytes=('output_bytes','first')).reset_index().to_csv(OUT/'local_runtime_summary.csv',index=False)

def environment_record():
    packages={}
    for p in ['numpy','scipy','pandas','scikit-learn','matplotlib','torch','tabpfn','tabicl','mpmath']:
        try:packages[p]=meta.version(p)
        except meta.PackageNotFoundError:packages[p]='not installed in scn; archived model inference only'
    import torch,os
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),cpu_threads=os.cpu_count(),
      cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda,
      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,packages=packages,new_dependencies=[])

def main():
    environment=environment_record();start=time.perf_counter()
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf8')
    count=checks_and_radius();error=calibration_prefixes();scale()
    report=dict(cached_interfaces=count,baseline_equivalence_passed=True,linear_decision_bounds_checked=True,
      calibration_original_max_error=error,seconds=time.perf_counter()-start,
      outputs='all conditions retained; no original metric modified; no new test outcomes inspected')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--check-cached',action='store_true');args=parser.parse_args()
    if args.check_cached:
        count=checks_and_radius();error=calibration_prefixes()
        result=dict(cached_interfaces=count,baseline_equivalence_passed=True,linear_decision_bounds_checked=True,
          calibration_original_max_error=error,timing_rerun=False)
        (OUT/'cached_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result,indent=2))
    else:main()
