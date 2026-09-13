"""Total physical knife load, independent of cutting-edge eligibility.

Wrong face/direction still cannot cut. Rejected rows and their friction must
nevertheless slow/stop the tool: no-cut is not the same as no physical contact.
"""
import math


def physical_load(pairs,rows,*,root,dt):
    if not isinstance(root,str) or not root.startswith('/') or root.endswith('/') or root in ('/World','/'):
        raise ValueError('Exact knife assembly root required')
    if isinstance(dt,bool) or not math.isfinite(dt) or dt<=0:raise ValueError('Positive native timestep required')
    prefix=root+'/'
    def owns(path):return path==root or path.startswith(prefix)
    loads=[];minimum=0.
    for (a,b),impulse in pairs.items():
        if owns(a) or owns(b):
            if isinstance(impulse,bool) or not math.isfinite(impulse) or impulse<0:
                raise RuntimeError('Finite noncancelling knife normal-plus-friction impulse required')
            loads.append(impulse)
    for row in rows:
        if owns(row['collider0']) or owns(row['collider1']):
            separation=row['separation']
            if not math.isfinite(separation):raise RuntimeError('Finite raw knife separation required')
            minimum=min(minimum,separation)
    upper=math.fsum(loads)/dt
    if not math.isfinite(upper):raise RuntimeError('Physical knife load overflow')
    return dict(upper_bound_n=upper,minimum_separation_m=minimum,
        source='all_knife_native_pairs_normal_plus_friction_regardless_of_cut_eligibility',
        separation_scope='original_order_tool_candidate_rows_self_pairs_not_available_here',
        cut_evidence=False)
