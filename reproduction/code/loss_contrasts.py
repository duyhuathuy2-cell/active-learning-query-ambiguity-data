"""Outcome-free refit contrast envelopes and observed paired losses."""
import numpy as np
from method import entropy

def contrasts(pack,y,baseline):
 base,br=pack['base'],pack['branches'];n=len(base);k=4
 challenger=pack['actions'][:,1].astype(int)
 if baseline=='entropy':ref=pack['actions'][:,0].astype(int)
 else:
  a=base[:,:4];raw=(entropy(base[:,None,4:])-(1-a[:,:,None])*entropy(br[:,:,0])-a[:,:,None]*entropy(br[:,:,1])).mean(2)
  ref=raw.argmax(1)
 pe=br[np.arange(n),ref];pc=br[np.arange(n),challenger]
 dmax=np.full(n,-np.inf);dmin=np.full(n,np.inf)
 for le in [0,1]:
  for lc in [0,1]:
   u0=pe[:,le]**2-pc[:,lc]**2;u1=(1-pe[:,le])**2-(1-pc[:,lc])**2
   dmax=np.maximum(dmax,np.maximum(u0,u1).mean(1));dmin=np.minimum(dmin,np.minimum(u0,u1).mean(1))
 envelope=np.maximum(abs(dmax),abs(dmin));envelope[ref==challenger]=0
 ids=pack['row_ids'];labels=y[ids[:,8:12]];truth=y[ids[:,12:]]
 rpred=br[np.arange(n),ref,labels[np.arange(n),ref]];cpred=br[np.arange(n),challenger,labels[np.arange(n),challenger]]
 rloss=((rpred-truth)**2).mean(1);closs=((cpred-truth)**2).mean(1);gain=rloss-closs
 assert np.all(abs(gain)<=envelope+1e-12)
 return gain,envelope,ref,challenger,rloss,closs
