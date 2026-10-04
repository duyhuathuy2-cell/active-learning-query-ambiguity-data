import numpy as np
from scipy.special import xlogy, expit, logit

def entropy(p):
    p = np.clip(p, 0, 1)
    return -(xlogy(p,p)+xlogy(1-p,1-p))/np.log(2)

def interval(a,b,q0,q1,epsilon=0):
    original_a=np.asarray(a)
    degenerate=(original_a==0)|(original_a==1)
    a = np.clip(original_a, np.finfo(float).eps, 1-np.finfo(float).eps)
    x, z = a*q1, b-(1-a)*q0
    lo = np.maximum(np.maximum(0,a+b-1),np.minimum(x,z)-epsilon/2)
    hi = np.minimum(np.minimum(a,b),np.maximum(x,z)+epsilon/2)
    assert np.all(lo<=hi+1e-9)
    def val(t):
        return entropy(b)-(1-a)*entropy((b-t)/(1-a))-a*entropy(t/a)
    lower=val(np.clip(a*b,lo,hi))
    upper=np.maximum(val(lo),val(hi))
    # Deterministic queries convey zero information. Their zero-weight branch
    # is unidentifiable, but cannot affect the information objective.
    exact_t=np.where(original_a==0,0,b)
    return np.where(degenerate,0,lower),np.where(degenerate,0,upper),np.where(degenerate,exact_t,lo),np.where(degenerate,exact_t,hi)

def kl_repair(a,b,q0,q1):
    z0,z1=logit(np.clip(q0,1e-9,1-1e-9)),logit(np.clip(q1,1e-9,1-1e-9))
    lo=np.full_like(b,-80.); hi=np.full_like(b,80.)
    for _ in range(70):
        mid=(lo+hi)/2; r0,r1=expit(z0+mid),expit(z1+mid)
        mask=(1-a)*r0+a*r1 < b
        lo=np.where(mask,mid,lo); hi=np.where(mask,hi,mid)
    assert np.max(np.abs((1-a)*r0+a*r1-b))<1e-8
    return r0,r1

def decision_bounds(lower,upper):
    """Rectangular decisions in linear time, preserving first-index tie rules."""
    lower,upper=np.asarray(lower),np.asarray(upper)
    if len(lower)==1:
        return np.zeros(1),np.ones(1,dtype=bool),np.ones(1,dtype=bool)
    top=int(np.argmax(upper))
    other=upper.copy()
    other[top]=-np.inf
    competing=np.full(len(lower),upper[top])
    competing[top]=np.max(other)
    return (np.maximum(0,competing-lower),
            lower>competing+1e-9,
            upper>=lower.max()-1e-9)

def scores(base,branches,nq):
    a=base[:nq,None]; b=base[nq:][None,:]
    q0,q1=branches[:,0,:],branches[:,1,:]
    m=(1-a)*q0+a*q1
    raw=entropy(b)-(1-a)*entropy(q0)-a*entropy(q1)
    js=entropy(m)-(1-a)*entropy(q0)-a*entropy(q1)
    drift=entropy(b)-entropy(m)
    assert np.max(np.abs(raw-js-drift))<1e-10
    lower,upper,lo,hi=interval(a,b,q0,q1)
    r0,r1=kl_repair(a,b,q0,q1)
    kl=entropy(b)-(1-a)*entropy(r0)-a*entropy(r1)
    def v(t):
        safe_a=np.clip(a,np.finfo(float).eps,1-np.finfo(float).eps)
        vv=entropy(b)-(1-safe_a)*entropy((b-t)/(1-safe_a))-safe_a*entropy(t/safe_a)
        return np.where((a==0)|(a==1),0,vv).mean(1)
    result={k:x.mean(1) for k,x in dict(raw=raw,js=js,drift=drift,lower=lower,upper=upper,kl=kl,tower=np.abs(b-m)).items()}
    result.update(entropy=entropy(base[:nq]), endpoint_lo=v(lo), endpoint_hi=v(hi))
    L,U=result['lower'],result['upper']
    result['regret'],result['certified'],result['possible']=decision_bounds(L,U)
    return result
