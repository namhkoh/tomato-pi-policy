"""Deterministic bounded joint-space detours; caller supplies collision checks."""
import numpy as np


def dense_segment(start,end,step=1.):
    count=max(2,int(np.ceil(np.max(np.abs(end-start))/step))+1)
    return np.linspace(start,end,count)


def connect_path(start,goal,lower,upper,valid,*,iterations=300,seed=0,max_checks=4000):
    """Bidirectional bounded search; every returned interval checked at <=1 deg.

    None means no path was found within the budget, not proof of infeasibility.
    This does not confer a whole-scene/dynamic collision certificate.
    """
    start,goal,lower,upper=[np.asarray(v,float) for v in (start,goal,lower,upper)]
    if (start.ndim!=1 or any(v.shape!=start.shape for v in (goal,lower,upper))
            or not np.isfinite([start,goal,lower,upper]).all() or np.any(lower>=upper)
            or np.any(start<=lower) or np.any(start>=upper)
            or np.any(goal<=lower) or np.any(goal>=upper)
            or not isinstance(iterations,int) or not 1<=iterations<=1000
            or not isinstance(max_checks,int) or not 2<=max_checks<=20000):
        raise ValueError('Invalid bounded joint-path request')
    cache={}
    def checked(q):
        key=tuple(np.round(q,8))
        if key not in cache and len(cache)>=max_checks: return False
        if key not in cache: cache[key]=bool(valid(q))
        return cache[key]
    def edge(a,b): return all(checked(q) for q in dense_segment(a,b))
    if not checked(start) or not checked(goal): return None
    if edge(start,goal): return dense_segment(start,goal)
    rng=np.random.default_rng(seed)
    trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
    def extend(tree,target):
        nodes,parents=tree
        nearest=int(np.argmin([np.linalg.norm(q-target) for q in nodes]))
        delta=target-nodes[nearest];length=np.linalg.norm(delta)
        if length<1e-8: return nearest,True
        q=nodes[nearest]+delta*min(1.,12./length)
        if not edge(nodes[nearest],q): return None,False
        nodes.append(q);parents.append(nearest)
        return len(nodes)-1,length<=12.
    def trace(tree,index):
        result=[]
        while index>=0:
            result.append(tree[0][index]);index=tree[1][index]
        return result[::-1]
    for iteration in range(iterations):
        if len(cache)>=max_checks: return None
        side=iteration%2;a,b=trees[side],trees[1-side]
        target=b[0][-1] if iteration%5==0 else rng.uniform(lower+1e-6,upper-1e-6)
        ia,_=extend(a,target)
        if ia is None: continue
        for _ in range(100):
            ib,reached=extend(b,a[0][ia])
            if ib is None: break
            if reached:
                vertices=trace(a,ia)+trace(b,ib)[-2::-1]
                if side: vertices=vertices[::-1]
                # Deterministic shortcuts are checked with the same predicate.
                compact=[vertices[0]];i=0
                while i<len(vertices)-1:
                    j=len(vertices)-1
                    while j>i+1 and not edge(vertices[i],vertices[j]): j-=1
                    compact.append(vertices[j]);i=j
                chunks=[dense_segment(x,y) for x,y in zip(compact[:-1],compact[1:])]
                path=np.concatenate([c if i==0 else c[1:] for i,c in enumerate(chunks)])
                if not all(checked(q) for q in path): raise RuntimeError('Unchecked path reconstruction')
                return path
    return None
