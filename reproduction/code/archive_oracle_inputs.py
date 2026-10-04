"""Archive full simulator inputs, learned interfaces and independently checked oracles."""
import environment
from environment import ROOT
import numpy as np,json,time
from scipy.special import expit,logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from method import entropy,scores

OUT=ROOT.parent/'data/results/policy_controls'
def run():
 start=time.perf_counter();maxerror=0
 for family in ['linear','interaction','radial','threshold']:
  full={key:[] for key in ['X','y','theta','posterior','base','branches','true_pair_joint','true_information']}
  with np.load(OUT/f'oracle_{family}_scores.npz') as z:stored={key:z[key] for key in z.files}
  for seed in range(400):
   rng=np.random.default_rng(820000+10000*['linear','interaction','radial','threshold'].index(family)+seed)
   X=rng.normal(size=(40,4));W=rng.normal(size=(12,4))*.65
   features={'linear':lambda:X,'interaction':lambda:np.c_[X[:,0],X[:,1],X[:,0]*X[:,1],X[:,2]*X[:,3]],
    'radial':lambda:np.c_[X[:,0],X[:,1],X[:,0]**2-1,X[:,1]**2-1],
    'threshold':lambda:np.c_[X[:,0],X[:,1],(X[:,0]>0)*2-1,(X[:,1]>0)*2-1]}[family]()
   theta=np.clip(expit(features@W.T+rng.normal(0,.25,size=12)),.01,.99).T
   hidden=int(rng.integers(12));y=rng.binomial(1,theta[hidden]);ll=(y[:16]*np.log(theta[:,:16])+(1-y[:16])*np.log1p(-theta[:,:16])).sum(1)
   posterior=np.exp(ll-logsumexp(ll));a=posterior@theta[:,16:24];b=posterior@theta[:,24:]
   joint=np.einsum('h,hi,hj->ij',posterior,theta[:,16:24],theta[:,24:])
   # Evaluate MI directly from the four joint-table cells, independently of
   # conditional-entropy formulas in the experiment runner.
   cells=np.stack([1-a[:,None]-b[None,:]+joint,b[None,:]-joint,a[:,None]-joint,joint],axis=-1)
   independent=np.stack([(1-a[:,None])*(1-b[None,:]),(1-a[:,None])*b[None,:],a[:,None]*(1-b[None,:]),a[:,None]*b[None,:]],axis=-1)
   assert cells.min()>0
   true=(cells*np.log2(cells/independent)).sum(-1).mean(1)
   clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0))
   if len(np.unique(y[:16]))==1:
    base=np.full(24,(y[:16].sum()+1)/18);branches=np.empty((8,2,16))
    for label in (0,1):branches[:,label,:]=(y[:16].sum()+label+1)/19
   else:
    clf.fit(X[:16],y[:16]);base=clf.predict_proba(X[16:])[:,1];branches=np.empty((8,2,16))
    for i in range(8):
     for label in (0,1):clf.fit(np.r_[X[:16],X[16+i:17+i]],np.r_[y[:16],label]);branches[i,label]=clf.predict_proba(X[24:])[:,1]
   computed=scores(base,branches,8)
   error=max(float(np.max(abs(true-stored['truth'][seed]))),float(np.max(abs(computed['raw']-stored['raw'][seed]))))
   assert error<1e-12;maxerror=max(maxerror,error)
   for key,val in {'X':X,'y':y,'theta':theta,'posterior':posterior,'base':base,'branches':branches,'true_pair_joint':joint,'true_information':true}.items():full[key].append(val)
  np.savez_compressed(OUT/f'oracle_{family}_full_inputs.npz',**{key:np.array(val) for key,val in full.items()})
  print('FULL ORACLE INPUTS',family,round(time.perf_counter()-start,1),flush=True)
 (OUT/'oracle_input_verification.json').write_text(json.dumps({'systems':1600,'max_direct_joint_information_or_refit_score_error':maxerror,
  'full_inputs_and_learned_interfaces_saved':True,'seconds':time.perf_counter()-start},indent=2))
if __name__=='__main__':run()
