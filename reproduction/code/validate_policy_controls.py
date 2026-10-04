"""Independent saved-result checks for the additional KBS evidence."""
import environment
from environment import ROOT
import json
import numpy as np,pandas as pd
from method import entropy,interval

R=ROOT.parent/'data/results/policy_controls'
def main():
 one=pd.read_csv(R/'matched_one_step_paired.csv');assert len(one)==6000
 assert len(one.groupby(['dataset','model','seed']))==600
 assert one.groupby(['dataset','model','seed']).policy.nunique().eq(10).all()
 assert np.isfinite(one[['brier','delta_brier','internal_regret']]).all().all()
 large=pd.read_csv(R/'expanded_matched_one_step.csv');assert len(large)==600
 seq=pd.read_csv(R/'additional_sequential.csv');assert len(seq)==3000
 assert seq.groupby(['dataset','seed','policy']).budget.nunique().eq(5).all()
 assert seq.groupby(['dataset','seed']).policy.nunique().eq(10).all()
 for r in seq.itertuples():
  selected=json.loads(r.selected_positions);assert len(selected)==r.budget and len(set(selected))==len(selected)
 assert np.isfinite(seq[['brier','error','internal_regret']]).all().all()
 inter=pd.read_csv(R/'same_context_certificate_interventions.csv');assert len(inter)==169
 assert not inter.duplicated(['dataset','model','seed','prefix']).any()
 assert np.max(abs(inter.delta_brier-(inter.certificate_brier-inter.entropy_brier)))<1e-14
 assert np.max(inter.certificate_internal_regret)<1e-9
 # All repaired values and point selectors are finite, including deterministic
 # forest candidate probabilities. Midpoint information is computed in t space.
 from gap_experiments import score_pack
 for p in (ROOT.parent/'data/results/benchmark/cache').glob('*forest*_initial.npz'):
  with np.load(p) as z:
   s=score_pack(z['base'],z['branches'],len(z['remain']))
   assert np.isfinite(s['tv_midpoint']).all()
   assert np.all(s['tv_midpoint']>=s['lower']-1e-12) and np.all(s['tv_midpoint']<=s['upper']+1e-12)
 cal=pd.read_csv(R/'learned_oracle_calibration.csv');assert len(cal)==7200
 assert not cal[cal.coverage].bound_violated.any(),'Coverage must imply selected-action regret bound.'
 oracle=pd.read_csv(R/'oracle_policy_test.csv');assert len(oracle)==26400
 assert oracle[oracle.policy=='exact_EPIG'].true_regret.eq(0).all()
 assert (oracle.true_regret>=-1e-12).all()
 assert cal.groupby(['family','method','calibrated']).seed.nunique().eq(300).all()
 # Calibration rank and score selection recomputed from held-out-disjoint scores.
 qtable=pd.read_csv(R/'learned_oracle_quantiles.csv')
 single=0
 for family in ['linear','interaction','radial','threshold']:
  with np.load(R/f'oracle_{family}_scores.npz') as z:
   true=z['truth'];single+=int(z['single_class'].sum())
   for method in ['tv','kl','raw']:
    L=z['lower'] if method=='tv' else z[method];U=z['upper'] if method=='tv' else z[method]
    e=np.maximum.reduce([L[:100]-true[:100],true[:100]-U[:100],np.zeros((100,8))]).max(1)
    radius=float(np.sort(e)[95]);expected=qtable[(qtable.family==family)&(qtable.method==method)].q.iloc[0]
    assert abs(radius-expected)<1e-14
 inner=pd.read_csv(R/'scalable_completion.csv');assert len(inner)==40
 count=0;maxres=0
 for row in inner[inner.feasible_witness].itertuples():
  with np.load(R/f'scalable_witness_{row.candidates}_{row.seed}_{row.atom_budget}.npz') as z,np.load(R/f'scalable_input_{row.candidates}_{row.seed}.npz') as inp:
   states,w,v=z['states'],z['weights'],z['target_joint'];a,b,lo,hi=[inp[s] for s in ['a','b','lo','hi']]
   t=states.T@v.T
   residual=max(abs(w.sum()-1),float(np.max(abs(states.T@w-a))),float(np.max(abs(v.sum(1)-b))),
    float(max(0,-w.min(),-v.min(),(v-w).max(),(lo-t).max(),(t-hi).max())))
   assert residual<1e-7;maxres=max(maxres,residual);count+=1
 scaling=pd.read_csv(R/'scalability.csv');assert len(scaling)==21
 assert (scaling[scaling.candidates==14].status=='resource_gate').all()
 assert (scaling[scaling.candidates==12].status=='time_limit').all()
 report={'matched_original_contexts':600,'matched_original_selector_rows':6000,'expanded_contexts':60,
  'fresh_logistic_trajectories':600,'fresh_logistic_decisions':3000,'distinct_certificate_contexts':169,
  'certificate_contexts_by_model':inter.groupby('model').size().to_dict(),
  'different_certificate_vs_entropy_choices':inter.groupby('model').different_choice.sum().to_dict(),
  'mean_certificate_minus_entropy_brier':inter.groupby('model').delta_brier.mean().to_dict(),
  'independent_oracle_systems':1600,'oracle_calibration_systems':400,'oracle_test_systems':1200,'single_class_contexts_retained':single,
  'coverage_implies_true_regret_bound_checked':True,'calibration_rank_recomputed':96,
  'tv_calibrated_issued':int(cal[(cal.method=='tv')&cal.calibrated].certified.sum()),
  'tv_uncalibrated_issued':int(cal[(cal.method=='tv')&~cal.calibrated].certified.sum()),
  'tv_uncalibrated_wrong':int(cal[(cal.method=='tv')&~cal.calibrated].wrong_certificate.sum()),
  'full_scaling_programs':21,'inner_scaling_programs':40,'inner_witnesses_independently_checked':count,'max_inner_witness_residual':maxres,
  'scientific_limits':['no universal predictive improvement','no historical oracle for ordinary public tasks',
   'no physical deployment evaluation','restricted-support failure is inconclusive','dependent public splits do not support population causal inference']}
 (R/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
