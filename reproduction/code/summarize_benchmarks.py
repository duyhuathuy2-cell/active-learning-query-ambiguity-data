import environment
from environment import ROOT
import numpy as np,pandas as pd,json,hashlib,sys
OUT=ROOT.parent/'data/results/benchmark'

def main():
    cases=pd.read_csv(OUT/'audit_cases.csv');seq=pd.read_csv(OUT/'sequential.csv')
    grouped=cases.groupby(['dataset','model']).agg(n=('seed','count'),tower=('mean_tower','mean'),width=('mean_width','mean'),
        raw_kl_flip=('raw_kl_flip','mean'),endpoint_flip=('endpoint_flip','mean'),certified=('certified','mean'),possible=('possible','mean'),base_brier=('base_brier','mean')).reset_index()
    grouped.to_csv(OUT/'audit_summary.csv',index=False)
    paired=[]
    for (dataset,model,budget),g in seq.groupby(['dataset','model','budget']):
      pivot=g.pivot(index='seed',columns='policy',values='brier')
      for policy in pivot.columns:
        diff=(pivot[policy]-pivot['raw']).to_numpy()
        paired.append(dict(dataset=dataset,model=model,budget=int(budget),policy=policy,n=len(diff),mean_brier=float(pivot[policy].mean()),
          mean_delta_vs_raw=float(diff.mean()),sd_delta=float(diff.std(ddof=1)),wins=int((diff< -1e-10).sum()),ties=int((abs(diff)<=1e-10).sum()),losses=int((diff>1e-10).sum())))
    pd.DataFrame(paired).to_csv(OUT/'sequential_paired.csv',index=False)
    stats={}
    for kind,g in cases.groupby('model'):
        stats[kind]={k:int(g[k].sum()) for k in ['raw_kl_flip','endpoint_flip','certified']};stats[kind]['n']=len(g)
        stats[kind]['mean_width']=float(g.mean_width.mean());stats[kind]['mean_tower']=float(g.mean_tower.mean())
    stats['sequential_n']=len(seq);stats['sequential_trajectories']=len(seq[['dataset','model','seed','policy']].drop_duplicates())
    stats['sequential_abstention_rate']=float(seq.loc[seq.policy=='certify_entropy','abstained'].mean())
    stats['conditional_split_sd_note']='SD and paired differences summarize algorithm variability across splits of fixed datasets. No population CI or significance claim.'
    (OUT/'headline.json').write_text(json.dumps(stats,indent=2))
    weights=[]
    for directory in [ROOT.parent/'models'/'tabpfn',ROOT.parent/'models'/'tabicl']:
      for f in directory.rglob('*.ckpt'):
        weights.append(dict(name=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (OUT/'weight_manifest.json').write_text(json.dumps(weights,indent=2))
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
