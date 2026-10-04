"""Check every final option, exact global certificate and retained adverse result.

Review mode needs compact outcome arrays and certificates, not bulky inference
matrices. Full mode additionally recomputes every contrast from saved branches.
"""
import environment
from environment import ROOT
from pathlib import Path
import numpy as np,pandas as pd,json,itertools,argparse
from fractions import Fraction as Q
from verified_global import exact_moments,info_iv,fq,ends,dual_lower
OUT=ROOT.parent/'data/results/global_and_prediction'

def build_compact():
 p=OUT/'final_safe_outcomes.csv';rows=pd.read_csv(p)
 np.savez_compressed(OUT/'final_safe_outcomes_compact.npz',**{c:rows[c].to_numpy() for c in rows.columns if c not in ['dataset','baseline','stage']},
  dataset=np.array(rows.dataset,dtype='U16'),baseline=np.array(rows.baseline,dtype='U7'),stage=np.array(rows.stage,dtype='U11'))
 (OUT/'final_safe_compact_schema.json').write_text(json.dumps({'rows':len(rows),'columns':list(rows.columns),
  'representation':'Lossless numeric float64, int64 and booleans; fixed-width Unicode labels; no pickle',
  'numeric_outcomes_retained_losslessly':True,'no_outcomes_filtered':True},indent=2))

def run(review=False):
 if not (OUT/'final_safe_outcomes_compact.npz').exists():build_compact()
 with np.load(OUT/'final_safe_outcomes_compact.npz') as z:rows=pd.DataFrame({k:z[k] for k in z.files})
 s=pd.read_csv(OUT/'final_safe_summary.csv');opts=pd.read_csv(OUT/'final_safe_options.csv')
 assert len(s)==12 and len(opts)==72 and len(rows)==221184
 assert s.deploy.sum()==9 and (s[s.deploy].test_mean_gain>0).all()
 assert (s[~s.deploy].test_mean_gain==0).all() and (s[~s.deploy].issued==0).all()
 option_error=0.;contrast_error=0.;max_envelope_violation=0.;leakage_checks=0
 ln=np.log(2*72/.05);datasets=list(s.dataset.drop_duplicates());full=None
 if not review:
  from gap_experiments import datasets as load
  from loss_contrasts import contrasts
  full=load()
 for name in datasets:
  pack=None
  if not review:
   with np.load(OUT/f'final_safe_raw_{name}.npz') as z:pack={k:z[k] for k in z.files}
   assert len(pack['row_ids'])==18432
   assert not np.intersect1d(pack['development_rows'],pack['row_ids'].ravel()).size
   assert all(len(set(x))==24 for x in pack['row_ids']);leakage_checks+=1
  for baseline in ['entropy','raw']:
   g=rows[(rows.dataset==name)&(rows.baseline==baseline)]
   cal=g[g.stage=='calibration'].sort_values('episode');test=g[g.stage=='test'].sort_values('episode')
   assert len(cal)==16384 and len(test)==2048
   assert np.max(abs(g.observed_challenger_gain)-g.envelope)<=1e-12
   if pack is not None:
    gain,A,*_=contrasts(pack,full[name][1],baseline);ordered=pd.concat([cal,test])
    contrast_error=max(contrast_error,float(np.max(abs(gain-ordered.observed_challenger_gain))),float(np.max(abs(A-ordered.envelope))))
    _,alteredA,altered_ref,altered_challenger,*_=contrasts(pack,1-full[name][1],baseline)
    _,_,ref,challenger,*_=contrasts(pack,full[name][1],baseline)
    assert np.array_equal(A,alteredA) and np.array_equal(ref,altered_ref) and np.array_equal(challenger,altered_challenger)
   choices=[]
   for op in opts[(opts.dataset==name)&(opts.baseline==baseline)].itertuples():
    x=np.where(cal.envelope<=op.threshold,cal.observed_challenger_gain,0)
    radius=np.sqrt(2*x.var(ddof=1)*ln/16384)+14*op.threshold*ln/(3*16383);lower=x.mean()-radius
    option_error=max(option_error,abs(lower-op.certified_mean_lower_bound),abs(radius-op.radius),abs(x.mean()-op.mean_gain))
    choices.append((lower,op.threshold))
   best=max(choices);r=s[(s.dataset==name)&(s.baseline==baseline)].iloc[0]
   assert best[1]==r.selected_threshold and (best[0]>0)==r.deploy
   d=np.where((test.envelope<=best[1])&r.deploy,test.observed_challenger_gain,0)
   assert np.max(abs(d-test.deployed_gain))<1e-12
   assert abs(d.mean()-r.test_mean_gain)<1e-12
   assert abs(d.std(ddof=1)/np.sqrt(2048)-r.test_se_gain)<1e-12
   assert int(test.issued.sum())==r.issued and int((test.issued&(test.deployed_gain<0)).sum())==r.worse_queries
 assert option_error<1e-12 and contrast_error<1e-12
 # Independently validate the explicit rational joint construction and every pair.
 checked=0;gaps=[]
 for name in pd.read_csv(OUT/'verified_linear_summary.csv')['case']:
  cert=json.loads((OUT/f'verified_linear_certificate_{name}.json').read_text())
  with np.load(OUT/f'exact_global_input_{name}.npz') as z:a,b,lo,hi,C=[z[k] for k in ['a','b','lo','hi','C']]
  with np.load(OUT/f'block_global_witness_{name}.npz') as z:w={k:z[k] for k in z.files}
  if cert['status']=='verified_global_infeasibility':
   lower,_=dual_lower(w['dual_eq'],w['dual_ub'],a,b,lo,hi,np.zeros_like(C));assert lower>0 and lower>=cert['violation_lower']-1e-12
  else:
   with np.load(OUT/f'verified_anchor_{name}.npz') as z:anchor={k:z[k] for k in z.files}
   aw,bw,tw=exact_moments(w['target_states'],w['weights'],w['candidate_joint']);aa,ba,ta=exact_moments(anchor['target_states'],anchor['weights'],anchor['candidate_joint'])
   blend=Q(cert['blend']);am=[(1-blend)*x+blend*y for x,y in zip(aw,aa)];bm=[(1-blend)*x+blend*y for x,y in zip(bw,ba)]
   ka=[[Q(x) for x in pair] for pair in cert['candidate_flip_kernels']];kb=[[Q(x) for x in pair] for pair in cert['target_flip_kernels']]
   assert all(r+s*old==fq(new) and 0<=r<=1 and 0<=r+s<=1 for (r,s),old,new in zip(ka,am,a))
   assert all(r+s*old==fq(new) and 0<=r<=1 and 0<=r+s<=1 for (r,s),old,new in zip(kb,bm,b))
   t=[[Q(x) for x in rr] for rr in cert['exact_pair_joint']]
   for i,(ra,sa) in enumerate(ka):
    for j,(rb,sb) in enumerate(kb):
     tm=(1-blend)*tw[i][j]+blend*ta[i][j]
     assert ra*rb+ra*sb*bm[j]+sa*rb*am[i]+sa*sb*tm==t[i][j]
     assert fq(lo[i,j])<=t[i][j]<=fq(hi[i,j])
   if cert['status']=='verified_linear_enclosure':
    objective=sum(fq(C[i,j])*t[i][j] for i in range(len(a)) for j in range(len(b)))
    assert float(objective)<=cert['upper'] and cert['gap']<1e-7;gaps.append(cert['gap'])
    lower,_=dual_lower(w['dual_eq'],w['dual_ub'],a,b,lo,hi,C);assert lower>=cert['lower']-1e-12
  checked+=1
 regret=pd.read_csv(OUT/'verified_regret_summary.csv');coupled=pd.read_csv(OUT/'coupled_regret_summary.csv')
 assert len(regret)==5 and regret.verified_gap.max()<.001 and coupled.verified_gap.iloc[0]<.001
 assert coupled.nodes.iloc[0]==55 and coupled.verified_upper.iloc[0]<coupled.local_regret.iloc[0]-.5
 # 90 independent brute-force reduced-cost comparisons on both orientations.
 from exact_global import price
 rng=np.random.default_rng(19073);pricing_error=0.
 for k,m in [(2,3),(3,2),(3,3)]:
  atoms=np.array(list(itertools.product([0,1],repeat=k+m)));A=atoms[:,:k];B=atoms[:,k:]
  for _ in range(30):
   lam=rng.normal(size=1+k+m);beta=-rng.random(2*k*m);C=rng.normal(size=(k,m))
   brute=(np.einsum('ni,ij,nj->n',A,C,B)-lam[0]-A@lam[1:1+k]-B@lam[1+k:]
    -np.einsum('ni,ij,nj->n',A,beta[:k*m].reshape(k,m)-beta[k*m:].reshape(k,m),B)).min()
   rc,*_=price(lam,beta,C,k,m,1);pricing_error=max(pricing_error,abs(rc-brute))
 assert pricing_error<1e-12
 actions=pd.read_csv(OUT/'global_minimax_summary.csv');assert len(actions)==5
 action_gap=0.
 for row in actions.itertuples():
  cert=json.loads((OUT/f'global_minimax_certificate_{row.case}.json').read_text())
  with np.load(OUT/f'exact_global_input_{row.case}.npz') as z:a,b,lo,hi=[z[k] for k in ['a','b','lo','hi']]
  with np.load(OUT/f'verified_anchor_{row.case}.npz') as z:anchor={k:z[k] for k in z.files}
  for action in cert['evaluated_actions']:
   ac=json.loads((OUT/f'global_minimax_action_certificate_{row.case}_{action}.json').read_text())
   with np.load(OUT/f'global_minimax_witness_{row.case}_action{action}.npz') as z:w={k:z[k] for k in z.files}
   aw,bw,tw=exact_moments(w['target_states'],w['weights'],w['candidate_joint']);aa,ba,ta=exact_moments(anchor['target_states'],anchor['weights'],anchor['candidate_joint'])
   blend=Q(ac['blend']);am=[(1-blend)*x+blend*y for x,y in zip(aw,aa)];bm=[(1-blend)*x+blend*y for x,y in zip(bw,ba)]
   ka=[[Q(x) for x in pair] for pair in ac['candidate_flip_kernels']];kb=[[Q(x) for x in pair] for pair in ac['target_flip_kernels']]
   assert all(r+s*old==fq(new) and 0<=r<=1 and 0<=r+s<=1 for (r,s),old,new in zip(ka,am,a))
   assert all(r+s*old==fq(new) and 0<=r<=1 and 0<=r+s<=1 for (r,s),old,new in zip(kb,bm,b))
   t=[[Q(x) for x in rr] for rr in ac['exact_pair_joint']]
   for i,(ra,sa) in enumerate(ka):
    for j,(rb,sb) in enumerate(kb):
     assert ra*rb+ra*sb*bm[j]+sa*rb*am[i]+sa*sb*((1-blend)*tw[i][j]+blend*ta[i][j])==t[i][j]
     assert fq(lo[i,j])<=t[i][j]<=fq(hi[i,j])
  recomputed=np.zeros(len(a))
  for world in cert['worlds']:
   source=world['source']
   cp=OUT/f'{source}_{row.case}.json' if source.startswith('verified_') else OUT/f'global_minimax_action_certificate_{row.case}_{source.removeprefix("action")}.json'
   cc=json.loads(cp.read_text());t=[[Q(x) for x in rr] for rr in cc['exact_pair_joint']]
   values=np.array([ends(sum(info_iv(t[i][j],fq(a[i]),fq(b[j])) for j in range(len(b)))/len(b)) for i in range(len(a))])
   assert np.max(abs(values-np.array(world['values'])))<1e-12
   recomputed=np.maximum(recomputed,np.maximum(0,np.nextafter(values[:,0].max()-values[:,1],-np.inf)))
  assert np.max(abs(recomputed-np.array(cert['lower_all_actions'])))<1e-12
  lower=np.array(cert['lower_all_actions']);upper=np.array(cert['upper_all_actions']);selected=cert['selected']
  assert len(lower)==len(a)==len(upper) and selected==upper.argmin()
  gap=upper[selected]-lower.min();assert gap<=.001 and abs(gap-cert['action_suboptimality_upper'])<1e-12
  assert cert['status']=='verified_epsilon_global_minimax';action_gap=max(action_gap,gap)
 result=dict(all_72_calibration_choices_recomputed=True,final_outcomes_checked=len(rows),final_positive_pairs=int(s.deploy.sum()),
  exact_baseline_fallback_pairs=int((~s.deploy).sum()),maximum_calibration_recomputation_error=option_error,
  raw_contrast_checks='omitted bulky matrices; compact retained outcomes' if review else 'all 110592 inference episodes, two baselines each',
  maximum_raw_contrast_error=contrast_error,development_row_exclusion_tasks=leakage_checks,
  candidate_target_outcome_perturbation_changes_no_actions_or_envelopes=None if review else True,
  exact_global_construction_certificates_checked=checked,linear_objective_certificates=len(gaps),maximum_verified_linear_gap=max(gaps),
  complete_pricing_bruteforce_cases=90,maximum_pricing_error=pricing_error,
  nonlinear_enclosures_checked=6,coupled_branch_nodes=55,mode='review' if review else 'full')
 result.update(global_minimax_action_certificates_checked=5,maximum_global_action_gap=action_gap)
 (OUT/'resolution_validation_review.json' if review else OUT/'resolution_validation.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review',action='store_true');p.add_argument('--compact',action='store_true');args=p.parse_args()
 if args.compact:build_compact()
 run(args.review)
