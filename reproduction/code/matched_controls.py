"""Matched diagnostics from the already frozen final family; no new policy selection."""
import environment
from environment import ROOT
import json,numpy as np,pandas as pd
R=ROOT.parent/'data/results/global_and_prediction';D=ROOT.parent/'data/figure_data/global_and_prediction'
opts=pd.read_csv(R/'final_safe_options.csv');summary=pd.read_csv(R/'final_safe_summary.csv')
with np.load(R/'final_safe_outcomes_compact.npz') as z:episodes=pd.DataFrame({k:z[k] for k in z.files})
rows=[]
for r in summary.itertuples():
 fixed=opts[(opts.dataset==r.dataset)&(opts.baseline==r.baseline)&(opts.threshold==1)].iloc[0]
 test=episodes[(episodes.dataset==r.dataset)&(episodes.baseline==r.baseline)&(episodes.stage=='test')]
 assert len(test)==2048 and fixed.n==16384
 g=np.where(test.envelope<=1,test.observed_challenger_gain,0)
 rows.append(dict(dataset=r.dataset,baseline=r.baseline,selected_threshold=r.selected_threshold,
  selected_calibration_mean=r.calibration_mean_gain,selected_lower=r.certified_mean_lower_bound,
  selected_test_gain=r.test_mean_gain,selected_switches=r.issued,
  c1_calibration_mean=fixed.mean_gain,c1_lower=fixed.certified_mean_lower_bound,
  c1_test_challenger_gain=float(g.mean()),c1_different_action_fraction=float(test.different_action.mean()),
  c1_deployed_test_gain=float(g.mean()) if fixed.certified_mean_lower_bound>0 else 0,
  c1_calibration_n=int(fixed.n),family_size=72))
matched=pd.DataFrame(rows);matched.to_csv(R/'matched_envelope_control.csv',index=False)
all_actions=[];traces=[]
for r in pd.read_csv(R/'global_minimax_summary.csv').itertuples():
 cert=json.loads((R/f'global_minimax_certificate_{r.case}.json').read_text())
 assert min(cert['upper_all_actions'])-min(cert['lower_all_actions'])<=.001
 for i,(lo,hi) in enumerate(zip(cert['lower_all_actions'],cert['upper_all_actions'])):
  all_actions.append(dict(case=r.case,action=i,lower=lo,upper=hi,selected=i==r.selected,further_evaluated=i in cert['evaluated_actions']))
 traces.append(dict(case=r.case,selected=r.selected,evaluated=cert['evaluated_actions'],seconds=r.seconds,gap=r.action_suboptimality_upper))
pd.DataFrame(all_actions).to_csv(R/'global_all_action_bounds.csv',index=False)
(R/'global_action_trace_summary.json').write_text(json.dumps(traces,indent=2),encoding='utf8')
names={'breast_cancer':'Cancer','iris_binary':'Iris','wine_binary':'Wine','digits_3_8':'Digits','moons':'Moons','mechanistic':'Mechanistic'}
print(json.dumps({'matched_comparisons':len(matched),'all_action_bounds':len(all_actions),'trace_cases':len(traces)},indent=2))
