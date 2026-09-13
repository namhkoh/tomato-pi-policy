"""Geometric escape bounds for the TWO already severed faces only.

This never permits release, main-stem/leaf contact, increased cutting load or
an uncleared endpoint. Nonpositive bounds are not claims of actual penetration.
"""
import numpy as np


def bind(screen):
    shapes=[s for s in screen.shapes if s[0]==screen.blade_path]
    if len(shapes)!=1 or shapes[0][3]!='box':raise ValueError('Exact original knife box required')
    paths=set(screen.seam_paths)
    if len(paths)!=2:raise ValueError('Exactly two preauthored cut-face colliders required')
    records=[s for s in screen.obstacles if s[0] in paths]
    if len(records)!=2 or any(s[1]!='capsule' for s in records):
        raise ValueError('Complete original cut-face cylinder bounds required')
    return shapes[0],records


def gaps(binding,world):
    from greenhouse_sim.robot_kinematics import _segment_aabb_distance
    shape,records=binding;_,_,link,_,(c,axes,half)=shape
    m=np.asarray(world[link],float);c=m[:3,:3]@c+m[:3,3];axes=m[:3,:3]@axes
    # Capsule distance is a conservative separation bound even for flat caps.
    return {p:float(_segment_aabb_distance(axes.T@(a-c),axes.T@(b-c),half)-radius)
            for p,_,(a,b,radius),_,_ in records}


def monotone(previous,current):
    if set(previous)!=set(current) or len(current)!=2:raise ValueError('Both exact cut-face bounds required')
    if not np.isfinite([*previous.values(),*current.values()]).all():raise ValueError('Finite separation bounds required')
    # Once a face is already outside the original margin it may approach only
    # while remaining fully clear. An uncleared face must move away, not in.
    return all(current[p]>=min(previous[p],.001)-1e-8 for p in previous)


def execution_guard(record):
    load=record.get('knife',{}).get('tool_contact_upper_bound_n')
    if (record.get('cut') is not True or record.get('native_guards_passed') is not True
            or isinstance(load,(bool,np.bool_)) or not isinstance(load,(int,float))
            or not np.isfinite(load) or not 0<=load<.01):
        raise RuntimeError('Post-cut escape must remain unloaded under 0.01 N')
