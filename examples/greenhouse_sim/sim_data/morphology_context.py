"""Rigid/scale/name-invariant context distances; not release approval.

Compare curved petioles *and* their parent/leaf context. World placement, file
names, seeds and colors cannot create novelty. Geometry is expected to come
from replayed assets. This module alone grants neither new donor families nor
new view-cap budgets; native/image/review gates remain separate.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

SCHEMA='greenhouse.normalized_morphology_context.v1'


def _array(value, shape=None):
    a=np.asarray(value,dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape!=shape):
        raise ValueError('Finite geometry with declared dimensions required')
    return a


def _unit(value):
    a=_array(value,(3,));n=np.linalg.norm(a)
    if n<=1e-12:raise ValueError('Nondegenerate axis required')
    return a/n


def _chain(value):
    a=_array(value)
    if a.ndim!=2 or a.shape[1]!=4 or len(a)<2 or np.any(a[:,3]<=0):
        raise ValueError('At least two xyz-radius points required')
    lengths=np.linalg.norm(np.diff(a[:,:3],axis=0),axis=1)
    if np.any(lengths<=1e-12):raise ValueError('Zero-length chain segment')
    return a,np.r_[0,np.cumsum(lengths)]


def _sample(value,count):
    chain,arc=_chain(value)
    s=np.linspace(0,arc[-1],count)
    return np.column_stack([np.interp(s,arc,chain[:,i]) for i in range(4)])


def descriptor(reference_petiole,parent,current_petiole,leaves):
    """Inputs are plant-frame xyz/radius chains and actual leaf mesh summaries.

    Leaf fields: attachment[3], centroid[3], axis[3], covariance[3,3]. The
    covariance is in squared metric units. Additional metadata is ignored.
    The unchanged donor proximal tangent disambiguates the parent-axis frame.
    """
    donor,arc=_chain(reference_petiole);parent,_=_chain(parent)
    origin=donor[0,:3];length=arc[-1]
    choices=[]
    for a,b in zip(parent,parent[1:]):
        d=b[:3]-a[:3];t=np.clip(np.dot(origin-a[:3],d)/np.dot(d,d),0,1)
        choices.append((np.linalg.norm(origin-a[:3]-t*d),d))
    z=_unit(min(choices,key=lambda q:q[0])[1]);d=_unit(donor[1,:3]-origin)
    x=_unit(d-z*np.dot(d,z));frame=np.column_stack((x,np.cross(z,x),z))
    def mapped(chain,count):
        sampled=_sample(chain,count)
        return np.column_stack(((sampled[:,:3]-origin)@frame/length,sampled[:,3]/length)).ravel()
    fixed=np.r_[mapped(current_petiole,21),mapped(parent,5)]
    vectors=[]
    for leaf in leaves:
        attachment=_array(leaf['attachment'],(3,));center=_array(leaf['centroid'],(3,))
        axis=_unit(leaf['axis']);cov=_array(leaf['covariance'],(3,3))
        if not np.allclose(cov,cov.T,atol=1e-12,rtol=0) or np.min(np.linalg.eigvalsh(cov)) < -1e-12:
            raise ValueError('Positive semidefinite symmetric leaf covariance required')
        local_cov=frame.T@cov@frame/(length*length)
        # Square-root principal covariance encodes size in normalized length
        # units while preserving orientation (not just eigenvalue magnitudes).
        eig,rot=np.linalg.eigh(local_cov)
        root=(rot*np.sqrt(np.maximum(eig,0)))@rot.T
        vectors.append(np.r_[(attachment-origin)@frame/length,
            (center-origin)@frame/length,axis@frame,root[np.triu_indices(3)]].tolist())
    vectors.sort(key=lambda row:tuple(np.round(row,12)))
    value=dict(schema=SCHEMA,fixed=fixed.tolist(),leaves=vectors,leaf_count=len(vectors))
    validate(value)
    return value


def validate(value):
    if value.get('schema')!=SCHEMA:raise ValueError('Unknown morphology descriptor')
    _array(value['fixed'],(104,))
    n=value['leaf_count']
    if type(n) is not int or n<0 or n!=len(value['leaves']):raise ValueError('Leaf count mismatch')
    if n:_array(value['leaves'],(n,15))


def fingerprint(value):
    """Rounded provenance aid only; tolerance matching remains authoritative."""
    validate(value)
    def rounded(values):
        a=np.round(values,10)
        a[a==0]=0.0  # Signed floating zero must not encode a rigid transform.
        return a.tolist()
    v=dict(schema=SCHEMA,fixed=rounded(value['fixed']),
           leaves=sorted(rounded(value['leaves'])),leaf_count=value['leaf_count'])
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def equivalent(a,b,tolerance):
    """Conservative context near-duplicate test with unordered leaf matching.

    A perfect bipartite matching avoids name/order artifacts and greedy errors.
    Different component cardinality is outside this equivalence contract.
    """
    validate(a);validate(b)
    if isinstance(tolerance,bool) or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Finite nonnegative normalized tolerance required')
    if a['leaf_count']!=b['leaf_count']:return False
    if np.max(np.abs(np.asarray(a['fixed'])-b['fixed']))>tolerance:return False
    n=a['leaf_count']
    if n==0:return True
    distances=np.max(np.abs(np.asarray(a['leaves'])[:,None,:]-np.asarray(b['leaves'])[None,:,:]),axis=2)
    neighbours=[np.flatnonzero(row<=tolerance).tolist() for row in distances]
    matched=[-1]*n
    def augment(i,seen):
        for j in neighbours[i]:
            if j in seen:continue
            seen.add(j)
            if matched[j]<0 or augment(matched[j],seen):matched[j]=i;return True
        return False
    return all(augment(i,set()) for i in sorted(range(n),key=lambda k:len(neighbours[k])))


class ContextInventory:
    """Global, explicit-tolerance candidate clustering, not training admission."""
    def __init__(self,tolerance):
        if isinstance(tolerance,bool) or not np.isfinite(tolerance) or tolerance<0:
            raise ValueError('Finite nonnegative normalized tolerance required')
        self.tolerance=float(tolerance);self.records=[];self.keys=set()

    def add(self,key,value):
        if not isinstance(key,str) or not key or key in self.keys:raise ValueError('Unique record key required')
        validate(value);self.keys.add(key)
        # Compare against every previous record, including cluster followers:
        # otherwise an interposed near-duplicate could evade cross-batch review.
        matches=[r['key'] for r in self.records if equivalent(value,r['descriptor'],self.tolerance)]
        record=dict(key=key,descriptor=json.loads(json.dumps(value)),matches=matches)
        self.records.append(record)
        return dict(near_duplicate_of=matches,novel_context_candidate=not matches,
            training_approved=False,source_cap_reset=False,new_biological_family=False)
