import numpy as np
import pytest
from sim_physics.blade_loading import LoadingWindow,slide_parameters,joint_anchors,joint_angular_error,leading_face_normal,loading_sample_safe


def test_effort_is_bounded_without_position_override():
    p=slide_parameters(.35)
    assert p['maximum_force_n']==.35 and p['stiffness_n_m']==0.
    assert p['target_velocity_m_s']==.001 and p['damping_ns_m']==pytest.approx(350.)
    for force in (0,.5,float('nan')):
        with pytest.raises(ValueError):slide_parameters(force)


def test_two_native_anchors_not_rest_point_displacement():
    frames=np.tile(np.eye(4),(2,1,1));frames[:,0,3]=[1,2]
    anchors=joint_anchors(frames,[[1,0,0],[0,0,0]])
    np.testing.assert_allclose(anchors,[[2,0,0],[2,0,0]])
    frames[1,1,3]=.0002
    anchors=joint_anchors(frames,[[1,0,0],[0,0,0]])
    assert np.linalg.norm(anchors[1]-anchors[0])==pytest.approx(.0002)


def test_loading_window_never_claims_grasp_or_release():
    window=LoadingWindow()
    for i in range(12):window.sample(.005,.3,i*.0001,True)
    report=window.report()
    assert report['force_displacement_window_observed']
    assert not report['robot_grasp_verified'] and not report['seam_release_authorized']


def test_loading_needs_contact_force_and_net_travel_not_approach_or_jitter():
    for force,eligible in ((.1,True),(.6,True),(.3,False)):
        window=LoadingWindow()
        for i in range(20):window.sample(.005,force,i*.0001,eligible)
        assert not window.report()['force_displacement_window_observed']
    window=LoadingWindow();window.sample(.005,0,0,False)
    for i in range(1000):window.sample(.005,.3,.01+(i%2)*.0000008,True)
    assert not window.report()['force_displacement_window_observed']
    assert window.report()['maximum_net_loaded_advance_m']<.000001


def test_coupon_uses_source_capsules_approved_knife_and_intact_seam():
    from pxr import Usd,UsdPhysics
    from sim_physics.blade_loading_probe import author
    stage=Usd.Stage.CreateInMemory();fixture=author(stage)
    assert fixture['joint'].GetJointEnabledAttr().Get()
    assert fixture['joint'].GetPrim().IsA(UsdPhysics.FixedJoint)
    assert fixture['metadata']['cut_arc_m']==.01
    assert fixture['metadata']['full_robot'] is False
    assert fixture['metadata']['full_plant'] is False
    assert fixture['metadata']['release_enabled'] is False
    assert fixture['drive'].GetMaxForceAttr().Get()==pytest.approx(.35)
    assert fixture['drive'].GetTargetVelocityAttr().Get()==0.
    assert fixture['knife'].collider.endswith('/BladePlateContact')
    assert fixture['metadata']['knife_mount']['flat_edge_faces']=='wrist_minus_y'
    anchors=joint_anchors(fixture['frames'],fixture['local_anchors'])
    assert np.linalg.norm(anchors[1]-anchors[0])<1e-8


def test_contact_normal_rejects_broad_face_and_side_load():
    assert leading_face_normal([1,0,0],[1,0,0])
    assert leading_face_normal([-1,0,0],[1,0,0])
    assert not leading_face_normal([0,0,1],[1,0,0])
    assert not leading_face_normal([0,0,0],[1,0,0])
    assert not leading_face_normal([.3,2,0],[1,0,0])
    # Full impulse norm guards an otherwise allowable 0.3 N projected load.
    assert not loading_sample_safe(np.linalg.norm([.3,2,0]),0,0,0,True)


def test_unsafe_sample_cannot_latch_success():
    window=LoadingWindow()
    for i in range(8):window.sample(.005,.3,i*.00002,True)
    safe=loading_sample_safe(.3,-.002,0,0,True)
    window.sample(.005,.3,.0004,safe)
    assert not window.report()['force_displacement_window_observed']
    assert loading_sample_safe(.3,-.0001,.001,1e-8,True)
    assert not loading_sample_safe(.3,0,0,0,False)


def test_zero_anchor_gap_does_not_hide_angular_seam_failure():
    from scipy.spatial.transform import Rotation
    frames=np.tile(np.eye(4),(2,1,1));frames[1,:3,:3]=Rotation.from_euler('z',10,degrees=True).as_matrix()
    angle=joint_angular_error(frames,np.tile(np.eye(3),(2,1,1)))
    assert angle==pytest.approx(np.radians(10))
    assert not loading_sample_safe(.3,0,0,0,True,angle)


def test_compliant_coupon_is_explicit_nondefault_experiment():
    from pxr import Usd,UsdShade
    from sim_physics.blade_loading_probe import author
    stage=Usd.Stage.CreateInMemory();fixture=author(stage,contact_stiffness_n_m=1000.)
    assert fixture['metadata']['applied_material_compliance']
    assert fixture['metadata']['contact_compliance']['calibrated'] is False
    assert not fixture['metadata']['release_enabled']
    for path in fixture['shapes']:
        material,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')
        assert material.GetPrim().GetAttribute('physxMaterial:compliantContactStiffness').Get()==1000.
    assert joint_angular_error(fixture['frames'],fixture['local_rotations'])<1e-6


def test_native_friction_anchors_are_decoded_separately_and_fail_closed(monkeypatch):
    from types import SimpleNamespace as S
    import pxr
    from sim_physics.blade_loading_probe import NativeContacts
    monkeypatch.setattr(pxr,'PhysicsSchemaTools',S(intToSdfPath=lambda value:value),raising=False)
    h=S(collider0='blade',collider1='stem',contact_data_offset=0,num_contact_data=0,
        friction_anchors_offset=0,num_friction_anchors_data=1)
    c=S(position=S(x=0.,y=0.,z=0.),impulse=S(x=0.,y=.001,z=0.))
    contacts=NativeContacts();contacts.callback([h],[],[c])
    assert contacts.error is None and not contacts.rows
    assert contacts.friction[0]['impulse']==[0.,.001,0.]
    h.num_friction_anchors_data=2;contacts.callback([h],[],[c])
    assert 'friction span' in contacts.error
    contacts=NativeContacts();h.num_friction_anchors_data=0;h.friction_anchors_offset=4294967295
    h.contact_data_offset=4294967295;contacts.callback([h],[],[])
    assert contacts.error is None and not contacts.friction and not contacts.rows
