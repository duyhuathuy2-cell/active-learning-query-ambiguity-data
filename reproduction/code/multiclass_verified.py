"""Exact rational vertex membership and outward interval information bounds.

IEEE prediction cells are treated as exact input rationals, normalized exactly.
Directed float intervals only discard provably infeasible active bases; every
remaining basis is checked with Fraction arithmetic. mpmath.iv encloses logs.
"""
from fractions import Fraction as F
import itertools
import numpy as np
from mpmath import iv
from multiclass_method import D,A,COMB,INV,bounds
iv.dps=60
permutations=list(itertools.permutations(range(4)))
parity=[(-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4)) for p in permutations]
active_set={tuple(ids) for ids in COMB}
for ids in itertools.combinations(range(18),4):
    matrix=A[list(ids)].astype(int)
    determinant=sum(sign*int(np.prod([matrix[i,p[i]] for i in range(4)])) for sign,p in zip(parity,permutations))
    assert (determinant!=0)==(ids in active_set),'Incomplete active-basis enumeration'
COEFF=[[[F(float(value)).limit_denominator(4) for value in row] for row in matrix] for matrix in INV]
for ids,matrix in zip(COMB,COEFF):
    original=A[ids].astype(int)
    assert all(sum(matrix[i][h]*int(original[h,j]) for h in range(4))==int(i==j) for i in range(4) for j in range(4))
    assert all(F(float(c))==c for row in matrix for c in row)

def rational_row(values):
    row=[F(float(x)) for x in values];total=sum(row)
    return [x/total for x in row]

def mpq(q):return iv.mpf(q.numerator)/iv.mpf(q.denominator)
def outward_low(x):return float(np.nextafter(float(x.a),-np.inf))
def outward_high(x):return float(np.nextafter(float(x.b),np.inf))

def verified_bounds(a_raw,b_raw,q_raw):
    a=rational_row(a_raw);b=rational_row(b_raw);q=[rational_row(row) for row in q_raw]
    original=[a[r]*q[r][c] for r in range(3) for c in range(3)]
    delta=[b[c]-sum(original[3*r+c] for r in range(3)) for c in range(3)]
    lower=[];upper=[]
    for r in range(3):
      for c in range(3):
        z=original[3*r+c]
        lower.append(z if delta[c]>=0 else F(0))
        upper.append(z if delta[c]<=0 else min(a[r],b[c]))
    offset=[F(0),F(0),a[0],F(0),F(0),a[1],b[0],b[1],a[2]-b[0]-b[1]]
    rhs=[upper[i]-offset[i] for i in range(9)]+[offset[i]-lower[i] for i in range(9)]
    rfloat=np.array([float(z) for z in rhs]);rlo=np.nextafter(rfloat,-np.inf);rhi=np.nextafter(rfloat,np.inf)
    xlo=np.zeros((len(COMB),4));xhi=xlo.copy()
    for term in range(4):
        coefficient=INV[:,:,term];low=np.where(coefficient>=0,rlo[COMB[:,term]][:,None],rhi[COMB[:,term]][:,None]);high=np.where(coefficient>=0,rhi[COMB[:,term]][:,None],rlo[COMB[:,term]][:,None])
        xlo=np.nextafter(xlo+np.nextafter(coefficient*low,-np.inf),-np.inf)
        xhi=np.nextafter(xhi+np.nextafter(coefficient*high,np.inf),np.inf)
    lhslo=np.zeros((len(COMB),18))
    for term in range(4):
        coefficient=A[:,term][None,:];low=np.where(coefficient>=0,xlo[:,term,None],xhi[:,term,None])
        lhslo=np.nextafter(lhslo+np.nextafter(coefficient*low,-np.inf),-np.inf)
    possible=np.flatnonzero(np.all(lhslo<=rhi[None,:],axis=1));vertices=set()
    for basis in possible:
        x=[sum(COEFF[basis][i][h]*rhs[COMB[basis,h]] for h in range(4)) for i in range(4)]
        P=tuple(offset[i]+sum(F(int(D[i,h]))*x[h] for h in range(4)) for i in range(9))
        if all(lower[i]<=P[i]<=upper[i] for i in range(9)):
            assert sum(P)==1
            assert all(sum(P[3*r+c] for c in range(3))==a[r] for r in range(3))
            assert all(sum(P[3*r+c] for r in range(3))==b[c] for c in range(3))
            assert sum(abs(P[i]-original[i]) for i in range(9))/2==sum(abs(z) for z in delta)/2
            vertices.add(P)
    assert vertices,'Exact rational polytope has no vertex'
    # The floating optimiser supplies only a positive tangent anchor; neither
    # its feasibility nor solver status is used to certify the interval.
    aa=np.array([float(z) for z in a]);bb=np.array([float(z) for z in b]);qq=np.array([[float(z) for z in row] for row in q])
    _,_,meta=bounds(aa,bb,qq)
    anchor=[F(float(max(z,1e-30))) for z in meta['anchor_cells']]
    denominator=[a[r]*b[c] for r in range(3) for c in range(3)]
    logarithms=[iv.ln(mpq(anchor[i]/denominator[i]))/iv.ln(2) for i in range(9)]
    correction=mpq(F(1)-sum(anchor))/iv.ln(2)
    lower_bounds=[];upper_bounds=[]
    for P in vertices:
        tangent=correction+sum(mpq(P[i])*logarithms[i] for i in range(9))
        information=sum(mpq(P[i])*iv.ln(mpq(P[i]/denominator[i]))/iv.ln(2) for i in range(9) if P[i]>0)
        lower_bounds.append(outward_low(tangent));upper_bounds.append(outward_high(information))
    L=max(0,min(lower_bounds));U=max(upper_bounds)
    return L,U,dict(exact_vertices=len(vertices),active_bases_retained=len(possible),objective_interval_decimal_precision=60)
