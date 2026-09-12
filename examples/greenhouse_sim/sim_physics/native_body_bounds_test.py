from types import SimpleNamespace as S
import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim_physics.native_body_bounds import collect_synchronous,countertransform_box


def collect(query,**kwargs):
    return collect_synchronous(query,stage_id=1,body_id=2,expected_paths={'/leaf'},
        decode_path=lambda i:'/leaf' if i==3 else '/unexpected',valid_result='VALID',**kwargs)


def response(**kwargs):
    return S(**dict(dict(result='VALID',stage_id=1,path_id=3,aabb_local_min=[-1]*3,
        aabb_local_max=[1]*3,local_pos=[0,0,0],local_rot=[0,0,0,1],volume=8),**kwargs))


def test_complete_response_keeps_all_native_parts():
    def query(**k):
        assert k['timeout_ms']==100
        k['rigid_body_fn'](S(result='VALID',stage_id=1,path_id=2))
        k['collider_fn'](response());k['collider_fn'](response(volume=4))
        k['finished_fn']()
    rows=collect(query)
    assert len(rows)==2 and rows[1]['volume']==4


@pytest.mark.parametrize('fault',['late','missing','body','stage','path','error','nan','quaternion','negative_volume','inverted','duplicate_finish','callback_after_finish'])
def test_incomplete_unknown_or_invalid_native_data_is_not_empty_clearance(fault):
    saved=[]
    def query(**k):
        saved.append(k)
        if fault=='late':return
        k['rigid_body_fn'](S(result='VALID',stage_id=1,path_id=99 if fault=='body' else 2))
        changes=dict(stage={'stage_id':99},path={'path_id':99},error={'result':'ERROR'},
            nan={'local_pos':[np.nan,0,0]},quaternion={'local_rot':[0,0,0,2]},
            negative_volume={'volume':-1},inverted={'aabb_local_min':[2,2,2]})
        if fault=='callback_after_finish':k['finished_fn']()
        if fault!='missing':k['collider_fn'](response(**changes.get(fault,{})))
        if fault!='callback_after_finish':k['finished_fn']()
        if fault=='duplicate_finish':k['finished_fn']()
    with pytest.raises(RuntimeError):collect(query)
    # Late responses have no way to turn the failed request into accepted data.
    saved[0]['collider_fn'](response());saved[0]['finished_fn']()


def test_epoch_failure_is_not_swallowed():
    with pytest.raises(RuntimeError,match='changed'):
        collect(lambda **k:None,guard=lambda:(_ for _ in ()).throw(RuntimeError('changed')))


def test_explicitly_disabled_property_response_is_audited_not_an_active_shape():
    log=[]
    def query(**k):
        k['rigid_body_fn'](S(result='VALID',stage_id=1,path_id=2))
        k['collider_fn'](response());k['collider_fn'](response(path_id=99));k['finished_fn']()
    rows=collect(query,disabled_paths={'/unexpected'},ignored_disabled=log)
    assert [r['path'] for r in rows]==['/leaf']
    assert [r['path'] for r in log]==['/unexpected']
    with pytest.raises(ValueError):collect(query,disabled_paths={'/unexpected'})
    with pytest.raises(ValueError):collect(query,disabled_paths={'/leaf'},ignored_disabled=[])


def test_disabled_status_does_not_mask_wrong_stage_or_missing_enabled_geometry():
    def query(**k):
        k['rigid_body_fn'](S(result='VALID',stage_id=1,path_id=2))
        k['collider_fn'](response(path_id=99,stage_id=9));k['finished_fn']()
    with pytest.raises(RuntimeError):collect(query,disabled_paths={'/unexpected'},ignored_disabled=[])


def test_countertransform_preserves_all_corners_in_tool_local_coordinates():
    rng=np.random.default_rng(21)
    for _ in range(20):
        a,b=np.eye(4),np.eye(4)
        for f in (a,b):f[:3,:3]=Rotation.random(random_state=rng).as_matrix();f[:3,3]=rng.normal(size=3)
        axes=Rotation.random(random_state=rng).as_matrix();centre=rng.normal(size=3);half=rng.uniform(.001,.2,3)
        c,r,h=countertransform_box(a,b,centre,axes,half)
        for sign in itertools.product((-1,1),repeat=3):
            old=centre+axes@(half*sign);new=c+r@(h*sign)
            np.testing.assert_allclose((np.linalg.inv(a)@np.r_[new,1]),(np.linalg.inv(b)@np.r_[old,1]),atol=1e-12)


@pytest.mark.parametrize('fault',['nan','scale','reflection','bottom','box_axes','half'])
def test_countertransform_rejects_nonrigid_or_unbounded_inputs(fault):
    a,b=np.eye(4),np.eye(4);centre=np.zeros(3);axes=np.eye(3);half=np.ones(3)
    if fault=='nan':centre[0]=np.nan
    if fault=='scale':a[0,0]=2
    if fault=='reflection':b[0,0]=-1
    if fault=='bottom':b[3,0]=1
    if fault=='box_axes':axes[0,0]=0
    if fault=='half':half[0]=0
    with pytest.raises(ValueError):countertransform_box(a,b,centre,axes,half)
