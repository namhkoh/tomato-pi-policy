from copy import deepcopy
import numpy as np
import pytest
from .grasp_contact_geometry import compile_contacts
from .shaft_grasp import ShaftCapsule,FingerPad


def fixture():
    cap=ShaftCapsule('/Plant/Stem','/Plant/Stem/StemCollider',np.eye(4),.003,.02,.0005)
    pads=[FingerPad('/Robot/L','/Robot/L/pad',np.eye(4),[.01,.02,.03],0,1,.0005),
          FingerPad('/Robot/R','/Robot/R/pad',np.eye(4),[.01,.02,.03],0,-1,.0005)]
    robot=np.tile(np.eye(4),(2,1,1));robot[:,0,3]=[-.0132,.0132]
    jac=np.zeros((1,6,7));jac[0,:,:6]=np.eye(6)
    value=dict(source_target='p/t',step_id=0,native_com_jacobian_velocity_check_passed=True,
        body_paths=['/Plant/Stem'],robot_body_paths=['/Robot/L','/Robot/R'],
        generalized_velocity=[0.]*7,body_frames_world=[np.eye(4).tolist()],
        robot_body_frames_world=robot.tolist(),body_world_com_jacobians=jac.tolist(),
        native_com_local_poses=[[0,0,0,0,0,0,1]],robot_body_velocities_world=np.zeros((2,6)).tolist(),
        robot_com_local_poses=[[0,0,0,0,0,0,1]]*2)
    rows=[]
    for z in (-.01,.01):
        rows.append(dict(collider0=cap.collider,collider1=pads[0].collider,kind='normal',
            point_world_m=[-.003,0,z],normal_on_0=[1,0,0],separation_m=.0002))
        rows.append(dict(collider0=cap.collider,collider1=pads[0].collider,kind='friction',point_world_m=[-.003,0,z]))
    return dict(chain=[cap],pads=pads,source_target='p/t',all_plant_colliders=[cap.collider],
        reference=deepcopy(value),current=deepcopy(value),rows=rows,mu=.5)


def test_geometry_signed_gap_and_point_jacobian():
    args=fixture();J,g,s,r=compile_contacts(**args)
    np.testing.assert_allclose(g,[.0002,.0002],atol=1e-16)
    np.testing.assert_allclose(J[:,0],1)
    np.testing.assert_allclose(J[:,4],[-.01,.01])
    assert s.tolist()==[0,0]
    assert r['patches']['normal_indices']==[[0,1]]
    assert r['contact_impulses_read'] is False


def test_original_order_reverse_headers_does_not_change_geometry():
    args=fixture();initial=compile_contacts(**args)
    for row in args['rows']:
        row['collider0'],row['collider1']=row['collider1'],row['collider0']
        if row['kind']=='normal':row['normal_on_0']=[-1,0,0]
    actual=compile_contacts(**args)
    for a,b in zip(initial[:3],actual[:3]):np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(initial[3]['patches']['tangent_jacobians'],actual[3]['patches']['tangent_jacobians'])


def test_no_impulse_access_and_real_single_anchor_not_duplicated():
    class NoImpulse(dict):
        def __getitem__(self,key):
            assert 'impulse' not in key
            return super().__getitem__(key)
    args=fixture();args['rows']=[NoImpulse(r) for r in args['rows'][:2]]
    _,_,_,r=compile_contacts(**args)
    assert len(r['patches']['tangent_jacobians'][0])==1
    assert r['patches']['normal_indices']==[[0]]


def test_moving_surface_and_nonzero_COM_use_actual_velocity():
    args=fixture();args['current']['robot_body_velocities_world'][0]=[.1,.2,.3,0,0,2.]
    args['current']['robot_com_local_poses'][0][0]=.001
    J,g,s,r=compile_contacts(**args)
    np.testing.assert_allclose(s,[.1,.1])
    np.testing.assert_allclose(r['patches']['surface_speeds_m_s'][0],[[.2184,.3],[.2184,.3]])
    args['current']['native_com_local_poses'][0][2]=.005
    new=compile_contacts(**args)[0]
    np.testing.assert_allclose(new[:,4],[-.015,.005])


def test_one_step_transport_recomputes_gap_not_replays_native_separation():
    args=fixture();args['current']['step_id']=1
    args['current']['body_frames_world'][0][0][3]=.0001
    _,g,_,r=compile_contacts(**args)
    np.testing.assert_allclose(g,[.0003,.0003],atol=1e-16)
    assert r['reference_step']==0 and r['current_step']==1


def test_supported_root_contact_is_reported_outside_dynamic_plant():
    args=fixture();support='/Plant/Support/StemCollider';args['all_plant_colliders'].append(support)
    args['rows'].append(dict(collider0=support,collider1='/Robot/Knife',kind='normal'))
    result=compile_contacts(**args)[3]
    assert len(result['ignored_outside_dynamic_system'])==1


@pytest.mark.parametrize('fault', ['knife','leaf','internal','missing_anchor','duplicate','gap','plane','cap','face','stale','source'])
def test_unsupported_or_incomplete_geometry_never_returns_force_input(fault):
    args=fixture()
    if fault=='knife':args['rows'][0]['collider1']='/Robot/Knife'
    elif fault=='leaf':args['rows'][0]['collider0']='/Plant/Stem/Leaf';args['all_plant_colliders'].append('/Plant/Stem/Leaf')
    elif fault=='internal':args['rows'][0]['collider1']='/Plant/Stem/Other'
    elif fault=='missing_anchor':args['rows']=[r for r in args['rows'] if r['kind']=='normal']
    elif fault=='duplicate':args['rows'].append(deepcopy(args['rows'][0]))
    elif fault=='gap':args['rows'][0]['separation_m']=.0004
    elif fault=='plane':args['rows'][0]['normal_on_0']=[0,1,0]
    elif fault=='cap':args['rows'][0]['point_world_m'][2]=.03
    elif fault=='face':args['rows'][0]['point_world_m'][2]=.035
    elif fault=='stale':args['current']['step_id']=2
    elif fault=='source':args['current']['source_target']='wrong'
    with pytest.raises(ValueError):compile_contacts(**args)


def test_empty_observed_stream_stays_empty_not_a_contact_certificate():
    args=fixture();args['rows']=[]
    J,g,s,r=compile_contacts(**args)
    assert J.shape==(0,7) and g.size==s.size==0
    assert not r['feature_observed'] and r['native_contact_law_parity'] is False

def _ry(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def _reverse(args):
    for row in args['rows']:
        row['collider0'],row['collider1']=row['collider1'],row['collider0']
        if row['kind']=='normal':row['normal_on_0']=(-np.array(row['normal_on_0'])).tolist()


def test_single_anchor_compiler_model_is_accepted_by_generalized_solver():
    from .contact_patch_prediction import GENERALIZED_MODEL,solve_generalized
    from .contact_coupled_prediction import MaterialLaw
    args=fixture();args['rows']=args['rows'][:2]
    J,g,s,r=compile_contacts(**args)
    assert r['patches']['model']==GENERALIZED_MODEL
    result=solve_generalized(np.eye(7),np.zeros(7),np.zeros(7),np.zeros(7),
        np.zeros(7),J,g,s,np.full(1,1000.),np.full(1,1.0573968428135787),
        1/240,np.zeros(7),law=MaterialLaw('unilateral_kv_v1'),patches=r['patches'],
        feature_observed=r['feature_observed'],root_dofs=6)
    assert result['status']=='resolved' and result['anchor_counts']==[1]
    assert len(r['friction_features'])==1
    assert not r['points_projected'] and not result['native_qualified']


def test_transported_normal_outside_current_finite_face_fails_without_projection():
    args=fixture()
    args['reference']['robot_body_frames_world'][0][1][3]=-.0199
    args['current']['robot_body_frames_world'][0][1][3]=-.0201
    args['current']['step_id']=1
    # A 0.2 mm relative translation carries an initially interior point 0.1 mm
    # outside the pad; reference-only footprint checking used to accept it.
    with pytest.raises(ValueError,match='current normal.*outside finite pad face'):
        compile_contacts(**args)


@pytest.mark.parametrize('angle', [.001,.05])
def test_rotated_current_capsule_does_not_repair_normal_feature_inside_solid(angle):
    args=fixture();frame=np.eye(4);frame[:3,:3]=_ry(angle)
    args['current']['body_frames_world']=[frame.tolist()];args['current']['step_id']=1
    # .05 rad gives 3.749 um radial error; .001 rad has tiny radial error
    # but its plane normal is still incompatible with a cylindrical side.
    with pytest.raises(ValueError,match='current normal.*(off cylindrical|not aligned)'):
        compile_contacts(**args)


@pytest.mark.parametrize('kind', ['normal','friction'])
@pytest.mark.parametrize('phase', ['reference','current'])
@pytest.mark.parametrize('sign', [-1,1])
def test_true_spherical_endcap_is_not_accepted_as_cylinder(kind,phase,sign):
    args=fixture();angle=.5*sign;R=_ry(angle);frame=np.eye(4);frame[:3,:3]=R
    # End-sphere support with outward direction -X. Both signs select a
    # genuine outward hemisphere, not a side-cylinder point.
    z=-sign*.02;point=R@np.array([0.,0.,z])-np.array([.003,0.,0.])
    if kind=='normal' and phase=='reference':
        for name in ('reference','current'):
            args[name]['body_frames_world']=[frame.tolist()]
            args[name]['robot_body_frames_world'][0][0][3]=float(point[0]-.01-.0002)
        args['rows']=args['rows'][:2]
        for row in args['rows']:row['point_world_m']=point.tolist()
    elif kind=='friction' and phase=='reference':
        # A spherical endcap with radial direction nearly side-on can be
        # mistaken for the cylinder under radial tolerance alone.
        axial=sign*1e-4
        point=np.array([-np.sqrt(.003**2-axial**2),0,sign*.02+axial])
        args['rows'][1]['point_world_m']=point.tolist()
    elif kind=='friction':
        # A pad-owned material anchor moves onto/beyond the end while normal
        # features on the stationary cylinder remain geometrically valid.
        _reverse(args);args['current']['step_id']=1
        args['current']['robot_body_frames_world'][0][2][3]=sign*.011
    else:
        # The normal's finite axial identity cannot change under exact rigid
        # transport alone. Tip-ward axis motion from a changed plane must be
        # rejected (alignment/end checks), never clamped back to the cylinder.
        args['rows']=args['rows'][:2]
        for row in args['rows']:row['point_world_m'][2]=-sign*(.02-4e-6)
        args['current']['step_id']=1
        args['current']['body_frames_world']=[frame.tolist()]
    with pytest.raises(ValueError,match=phase+' '+kind):
        compile_contacts(**args)


@pytest.mark.parametrize('kind', ['normal','friction'])
@pytest.mark.parametrize('sign', [-1,1])
@pytest.mark.parametrize('clearance', [0.,1e-6,-1e-6])
def test_end_ring_existing_two_micron_uncertainty_band_fails_closed(kind,sign,clearance):
    args=fixture()
    indices=[0] if kind=='normal' else [1]
    for i in indices:args['rows'][i]['point_world_m'][2]=sign*(.02-clearance)
    with pytest.raises(ValueError,match='reference '+kind+'.*endcap/end-ring'):
        compile_contacts(**args)


@pytest.mark.parametrize('kind', ['normal','friction'])
@pytest.mark.parametrize('sign', [-1,1])
def test_supported_side_point_outside_end_uncertainty_band_is_retained(kind,sign):
    args=fixture()
    i=0 if kind=='normal' else 1
    args['rows'][i]['point_world_m'][2]=sign*(.02-4e-6)
    _,_,_,r=compile_contacts(**args)
    evidence=r['features'] if kind=='normal' else r['friction_features']
    assert evidence[0]['reference_geometry']['axial_clearance_m']>2e-6
    assert evidence[0]['point_world_m']==args['rows'][i]['point_world_m']


@pytest.mark.parametrize('kind', ['normal','friction'])
@pytest.mark.parametrize('offset', [-3e-6,3e-6])
def test_reference_radial_tolerance_is_not_widened(kind,offset):
    args=fixture();i=0 if kind=='normal' else 1
    args['rows'][i]['point_world_m'][0]+=offset
    if kind=='normal':args['rows'][i]['separation_m']+=offset
    with pytest.raises(ValueError,match='reference '+kind+'.*off cylindrical'):
        compile_contacts(**args)


@pytest.mark.parametrize('offset', [-1e-6,1e-6])
def test_small_supported_raw_radial_residual_is_reported_not_projected(offset):
    args=fixture();args['rows']=args['rows'][:2]
    for row in args['rows']:
        row['point_world_m'][0]+=offset
        if row['kind']=='normal':row['separation_m']+=offset
    before=deepcopy(args['rows']);_,_,_,r=compile_contacts(**args)
    assert args['rows']==before
    for evidence in (r['features'][0],r['friction_features'][0]):
        np.testing.assert_array_equal(evidence['point_world_m'],before[0]['point_world_m'])
        assert evidence['reference_geometry']['radial_error_m']==pytest.approx(abs(offset))
        assert evidence['current_geometry']['radial_error_m']==pytest.approx(abs(offset))
    assert r['point_tolerance_m']==2e-6 and r['normal_tolerance']==2e-4
    assert r['points_projected'] is False


def test_friction_reference_off_surface_fails_even_when_normal_rows_are_valid():
    args=fixture();args['rows'][1]['point_world_m']=[-.003,.04,-.01]
    with pytest.raises(ValueError,match='reference friction.*off cylindrical'):
        compile_contacts(**args)


def test_friction_reference_finite_face_is_checked_independently():
    from dataclasses import replace
    args=fixture();half=args['pads'][0].half_extents_m.copy();half[2]=.012
    args['pads'][0]=replace(args['pads'][0],half_extents_m=half)
    args['rows'][1]['point_world_m'][2]=.013
    with pytest.raises(ValueError,match='reference friction.*outside finite pad face'):
        compile_contacts(**args)


def test_friction_current_finite_face_is_checked_after_original_owner_transport():
    args=fixture();args['rows'][1]['point_world_m'][2]=-.015
    args['current']['step_id']=1;args['current']['robot_body_frames_world'][0][2][3]=.016
    with pytest.raises(ValueError,match='current friction.*outside finite pad face'):
        compile_contacts(**args)


def test_pad_owned_friction_anchor_transport_is_not_projected_onto_moving_shaft():
    args=fixture();_reverse(args)
    args['current']['step_id']=1;args['current']['body_frames_world'][0][0][3]=.0001
    with pytest.raises(ValueError,match='current friction.*off cylindrical'):
        compile_contacts(**args)


@pytest.mark.parametrize('reverse', [False,True])
def test_common_rigid_motion_preserves_supported_reference_and_current_features(reverse):
    args=fixture()
    if reverse:_reverse(args)
    R=_ry(.37);translation=np.array([.2,-.1,.3]);H=np.eye(4)
    H[:3,:3]=R;H[:3,3]=translation
    args['current']['step_id']=1
    for key in ('body_frames_world','robot_body_frames_world'):
        args['current'][key]=[(H@np.array(frame)).tolist() for frame in args['current'][key]]
    _,g,_,r=compile_contacts(**args)
    np.testing.assert_allclose(g,.0002,atol=1e-16)
    for feature,row in zip(r['friction_features'],args['rows'][1::2],strict=True):
        np.testing.assert_allclose(feature['point_world_m'],R@row['point_world_m']+translation,atol=1e-16)
        assert feature['owner_body']==('/Robot/L' if reverse else '/Plant/Stem')
        assert feature['current_geometry']['radial_error_m']<1e-15
    assert r['native_contact_law_parity'] is False
