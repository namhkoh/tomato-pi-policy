from copy import deepcopy
import numpy as np
import pytest
from .cut_section import clearance,cut_face_contact
from .through_stroke import ThroughStroke
from .through_stroke_test import sample,command


KNIFE='/R/ee_right/attachments/DeleafKnife/CrossbarContact'
FACES=['/P/Support/Segment_000/StemCollider','/P/Branch/Segment_001/StemCollider']


def section(edge,axis=(1,0,0),centre=(0,0,0)):
    return clearance(edge,centre,axis,radius=.003,tip_offset=.0005,half_span=.014)


def test_full_tip_clearance_not_arbitrary_waypoint_or_blade_side_unloading():
    e=np.asarray(sample(z=-.0031)['knife']['edge_frame'])
    r=section(e)
    assert r['sharp_edge_cleared']
    assert r['sharp_edge_clearance_m']==pytest.approx(.0006)
    assert r['blade_end_clearance_m']==pytest.approx(.011)
    e[2,3]=-.0025
    assert not section(e)['sharp_edge_cleared']
    e[2,3]=-.0031;e[1,3]=.02
    assert not section(e)['sharp_edge_cleared']  # Edge end misses part of shaft.


@pytest.mark.parametrize('tilt',[0.,10.,15.])
def test_ellipse_projection_matches_dense_exact_cylinder_plane_intersection(tilt):
    from scipy.spatial.transform import Rotation
    e=np.asarray(sample(z=-.0031)['knife']['edge_frame'])
    e[:3,:3]=Rotation.from_euler('z',tilt,degrees=True).as_matrix()@e[:3,:3]
    axis=np.array([np.sqrt(1-.25**2),0,.25]);r=section(e,axis)
    u=np.cross(axis,[0,1,0]);u/=np.linalg.norm(u);v=np.cross(axis,u)
    theta=np.linspace(0,2*np.pi,100000,endpoint=False)
    radial=.003*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
    n=e[:3,2];p=e[:3,3]
    axial=((p-radial)@n)/(n@axis)
    points=radial+axial[:,None]*axis
    tip=.0005-np.max((points-p)@(-e[:3,0]))
    ends=.014-np.max(np.abs((points-p)@e[:3,1]))
    assert r['sharp_edge_clearance_m']==pytest.approx(tip,abs=1e-10)
    assert r['blade_end_clearance_m']==pytest.approx(ends,abs=1e-10)
    assert r['maximum_section_axial_distance_m']==pytest.approx(np.max(abs(axial)),abs=1e-10)


def contact(z=.0041,upper=.12):
    r=sample(z=z,upper=upper)
    r['native_contact_pairs_n']=[[KNIFE,FACES[0],upper]]
    r['knife'].update(physical_knife_contact=dict(minimum_separation_m=-.0002),
        raw_normal_rows=[dict(collider0=KNIFE,collider1=FACES[0],separation=-.0002,eligible=False)])
    return r


def test_side_contact_is_post_release_sliding_never_new_cut_evidence():
    r=contact()
    assert cut_face_contact(r,knife=KNIFE,faces=FACES)
    for key in ('cut','native_guards_passed'):
        bad=deepcopy(r);bad[key]=False
        with pytest.raises(ValueError):cut_face_contact(bad,knife=KNIFE,faces=FACES)


@pytest.mark.parametrize('other',[KNIFE.replace('CrossbarContact','ArcContacts/Part_0'),'/R/Camera','/Neighbor/StemCollider'])
def test_wrong_tool_or_other_structure_cannot_borrow_cut_face_permission(other):
    r=contact();r['native_contact_pairs_n']=[[KNIFE,other,.12]]
    assert not cut_face_contact(r,knife=KNIFE,faces=FACES)


def test_friction_only_loaded_pair_with_no_current_separation_does_not_advance():
    r=contact();r['knife']['raw_normal_rows']=[]
    assert not cut_face_contact(r,knife=KNIFE,faces=FACES)
    r=contact();r['native_contact_pairs_n']=[]
    with pytest.raises(RuntimeError):cut_face_contact(r,knife=KNIFE,faces=FACES)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.00101])
def test_invalid_or_excess_penetration_refused(value):
    r=contact();r['knife']['physical_knife_contact']['minimum_separation_m']=value
    with pytest.raises(RuntimeError):cut_face_contact(r,knife=KNIFE,faces=FACES)


def controller():
    return ThroughStroke([-.008,0,.005],fraction=.3,endpoint=[0,0,-.005],
        direction=[0,0,-1],physics_hz=480,step=20,
        material_section=dict(radius=.003,tip_offset=.0005,half_span=.014,knife=KNIFE,faces=FACES))


def observe(f,r):
    return f.observe(r,step=f.step+1,arc_up=[0,0,1],support_ready=True,
        section_pose=dict(centre=[0,0,0],axis=[1,0,0]))


def test_section_traversal_requires_actual_motion_dwell_and_fresh_section():
    f=controller();observe(f,contact());command(f)
    assert f.receipt['command_state']=='guarded_cut_face_sliding'
    for i in range(12):
        r=observe(f,contact(z=-.0031));assert r['complete'] is (i==11)
        if i<11:command(f)
    assert r['remaining_m']==pytest.approx(.0019) # Not arbitrary waypoint shortening by time.
    assert r['withdrawal_unloading_still_required']
    assert not r['cut_authorized']
    with pytest.raises(RuntimeError):controller().observe(contact(),step=20,arc_up=[0,0,1],support_ready=True)


def test_section_still_refuses_wrong_pair_and_force_cap():
    f=controller();r=contact();r['native_contact_pairs_n']=[[KNIFE,'/Neighbor/StemCollider',.12]]
    observe(f,r);command(f)
    assert f.receipt['command_state']=='hold_unverified_cut_face'
    with pytest.raises(RuntimeError):observe(controller(),contact(upper=.501))


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_explicit_complete_native_profile_validates_before_launch(tmp_path,monkeypatch,mode):
    from pathlib import Path
    from .ground_truth_trial import main
    class Validated(Exception):pass
    def stop(*a,**k):raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(['--output',str(tmp_path/'new'),'--mode',mode,
        '--process-zone-trial','--milestone','cut_action','--through-stroke-trial',
        '--material-clearance-trial','--blade-aim-offset-m','.0015'])


def test_missing_through_profile_never_creates_output(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Material-clearance'):
        main(['--output',str(tmp_path/'new'),'--material-clearance-trial'])
    assert not (tmp_path/'new').exists()
