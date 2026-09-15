"""Reference pose constraints apply independently of old generic orbit limits."""
import math
import numpy as np
import pytest
from sim_data.native_view_pose import bounded_reference_root
from sim_data.native_view_plan import propose_specs


def reference(yaw=240):
    t=math.radians(yaw);m=np.eye(4)
    m[:2,:2]=[[math.cos(t),math.sin(t)],[-math.sin(t),math.cos(t)]]
    m[3,:3]=[.49,.51,.101]
    return dict(robot_root_to_world_usd_row_vectors=m.tolist(),visual_bound_screen={'passed':True})


@pytest.mark.parametrize('yaw',[197,240,95,275])
def test_preserves_proven_orientation_and_only_offsets_xy(yaw):
    r=reference(yaw);target=[.13,.63,1.58]
    for s in propose_specs(r['robot_root_to_world_usd_row_vectors'],target,'qualified'):
        result=bounded_reference_root(r,s,target)
        np.testing.assert_allclose(result[:3,:3],np.asarray(r['robot_root_to_world_usd_row_vectors']).T[:3,:3])
        assert result[2,3]==.101 and np.allclose(result[3],[0,0,0,1])


@pytest.mark.parametrize('fault',['inward','too_far','lateral','yaw','unscreened','nan','tilted'])
def test_refuses_to_expand_reference_permission(fault):
    r=reference();target=[.13,.63,1.58]
    s=propose_specs(r['robot_root_to_world_usd_row_vectors'],target,'qualified')[0]
    if fault=='inward':s['root_x_m']-=.001
    elif fault=='too_far':s['root_x_m']+=.041
    elif fault=='lateral':s['y_offset_m']+=.041
    elif fault=='yaw':s['root_yaw_degrees']+=1
    elif fault=='unscreened':r['visual_bound_screen']['passed']=False
    elif fault=='nan':s['root_x_m']=float('nan')
    else:r['robot_root_to_world_usd_row_vectors'][2][0]=.1
    with pytest.raises(ValueError):bounded_reference_root(r,s,target)
