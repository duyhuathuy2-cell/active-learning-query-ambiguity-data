"""Three-class sharp repair bounds; all vertex maxima, certified convex minima."""
import itertools
import numpy as np
from scipy.special import xlogy
from scipy.optimize import minimize,linprog,minimize_scalar
D=np.array([[1,0,0,0],[0,1,0,0],[-1,-1,0,0],[0,0,1,0],[0,0,0,1],[0,0,-1,-1],[-1,0,-1,0],[0,-1,0,-1],[1,1,1,1]],float)
A=np.r_[D,-D];comb=[];inv=[]
for ids in itertools.combinations(range(18),4):
    mat=A[list(ids)]
    if abs(np.linalg.det(mat))>.1:comb.append(ids);inv.append(np.linalg.inv(mat))
COMB=np.array(comb);INV=np.array(inv)

def mi(P,a,b):
    P=np.maximum(0,P)
    return np.sum(xlogy(P,P/(a[:,None]*b[None,:])),axis=(-2,-1))/np.log(2)

def bounds(a,b,q):
    assert a.shape==b.shape==(3,) and q.shape==(3,3)
    original=a[:,None]*q;m=original.sum(0);delta=b-m
    lower=np.where(delta[None,:]>1e-12,original,0.)
    upper=np.where(delta[None,:]<-1e-12,original,np.minimum(a[:,None],b[None,:]))
    zero=np.abs(delta)<1e-12
    lower[:,zero]=original[:,zero];upper[:,zero]=original[:,zero]
    c=np.array([0,0,a[0],0,0,a[1],b[0],b[1],a[2]-b[0]-b[1]])
    rhs=np.r_[upper.ravel()-c,c-lower.ravel()]
    xs=np.einsum('nij,nj->ni',INV,rhs[COMB])
    mask=np.all(xs@A.T<=rhs+1e-10,axis=1)
    verts=xs[mask]
    if not len(verts):raise RuntimeError('No vertex found')
    P=(verts@D.T+c).reshape(-1,3,3)
    values=mi(P,a,b);U=float(values.max())
    def fun(x):return float(mi((D@x+c).reshape(3,3),a,b))
    def grad(x):
        pp=np.maximum(D@x+c,1e-15)
        return D.T@(np.log(pp/(a[:,None]*b[None,:]).ravel())+1)/np.log(2)
    opt=minimize(fun,verts.mean(0),jac=grad,method='SLSQP',constraints={'type':'ineq','fun':lambda x:rhs-A@x,'jac':lambda x:-A},options={'ftol':1e-13,'maxiter':1000})
    x=opt.x
    # A linear supporting hyperplane is a valid global lower bound for the
    # fixed-marginal convex information objective; report its optimisation gap.
    if np.max(A@x-rhs)>1e-10:x=verts.mean(0)
    anchor=(1-1e-9)*x+1e-9*verts.mean(0)
    pp=D@anchor+c
    # Positive feasible anchor gives a genuine supporting hyperplane, even
    # when the optimiser is on a zero-cell boundary. Fixed zero support has
    # no varying coordinate and contributes no derivative.
    varying=np.max(P.reshape(-1,9),axis=0)>1e-13
    if np.any(pp[varying]<=0):anchor=verts.mean(0);pp=D@anchor+c
    for iteration in range(300):
        gcell=np.zeros(9);gcell[varying]=(np.log(pp[varying]/(a[:,None]*b[None,:]).ravel()[varying])+1)/np.log(2)
        gradient=D.T@gcell;v=verts[np.argmin(verts@gradient)]
        dual_gap=float(gradient@(anchor-v))
        if dual_gap<1e-8:break
        direction=v-anchor
        line=minimize_scalar(lambda step:fun(anchor+step*direction),bounds=(0,1-1e-12),method='bounded',options={'xatol':1e-14})
        anchor=anchor+line.x*direction;pp=D@anchor+c
    gcell=np.zeros(9);gcell[varying]=(np.log(pp[varying]/(a[:,None]*b[None,:]).ravel()[varying])+1)/np.log(2)
    gradient=D.T@gcell
    lp=linprog(gradient,A_ub=A,b_ub=rhs,bounds=[(None,None)]*4,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert lp.success,lp.message
    linear_min=float(np.min(verts@gradient))
    assert abs(lp.fun-linear_min)<1e-7
    f=fun(anchor);L=max(0,float(f+linear_min-gradient@anchor)-1e-9)
    gap=float(f-L)
    residual=float(np.max(np.maximum(A@verts.T-rhs[:,None],0)))
    return L,U,dict(lower_gap=gap,vertex_count=len(verts),constraint_residual=residual,min_cost=.5*np.abs(delta).sum(),observed_min_cost=.5*np.abs(P-original).sum((1,2)).min(),anchor_cells=np.maximum(D@anchor+c,1e-30).tolist())
