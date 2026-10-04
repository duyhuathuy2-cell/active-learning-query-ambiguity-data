"""Post-confirmation descriptive comparison of every static rule on common tests.

No new policy or threshold is chosen from these results. These diagnostics are
not additional simultaneous certificates; every configured rule is retained.
"""
import environment
from environment import ROOT
import numpy as np,pandas as pd,json,time
from gap_experiments import datasets,score_pack,choose,POLICIES
from observable_calibration import brier_bounds
OUT=ROOT.parent/'data/results/global_and_prediction'
def run():
 start=time.perf_counter();rows=[]
 for no,(name,(X,y)) in enumerate(datasets().items()):
  with np.load(OUT/f'final_safe_raw_{name}.npz') as z:ids=z['row_ids'][16384:];base=z['base'][16384:];br=z['branches'][16384:]
  for n in range(2048):
   s=score_pack(base[n],br[n],4);L,U=brier_bounds(base[n],br[n],4);actions={p:choose(s,p) for p in POLICIES}
   actions.update(brier_lower=int(L.argmax()),brier_minimax=int(np.argmin([max(0,np.max(np.delete(U,i))-L[i]) for i in range(4)])))
   # Every action was formed before the actual labels evaluate counterfactual loss.
   loss=((br[n,np.arange(4),y[ids[n,8:12]]]-y[ids[n,12:]])**2).mean(1)
   for p,act in actions.items():rows.append(dict(dataset=name,episode=n,policy=p,choice=act,brier=loss[act],gain_vs_raw=loss[actions['raw']]-loss[act]))
  print('ALL STATIC FINAL',name,flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'final_static_diagnostic.csv',index=False)
 summary=df.groupby(['dataset','policy']).agg(n=('episode','size'),mean_brier=('brier','mean'),mean_gain_vs_raw=('gain_vs_raw','mean'),sd_gain_vs_raw=('gain_vs_raw','std')).reset_index()
 summary['se_gain_vs_raw']=summary.sd_gain_vs_raw/np.sqrt(summary.n);summary.to_csv(OUT/'final_static_diagnostic_summary.csv',index=False)
 (OUT/'final_static_diagnostic_provenance.json').write_text(json.dumps({'status':'post-confirmation descriptive diagnostic; not a new confirmatory family',
  'same_final_tests':True,'policies':POLICIES+['brier_lower','brier_minimax'],'tasks':6,'test_episodes_per_task':2048,
  'policy_or_threshold_selection_from_this_diagnostic':False,'seconds':time.perf_counter()-start},indent=2))
if __name__=='__main__':run()
