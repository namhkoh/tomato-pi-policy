import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from .cylinder_bounds import separation


def test_actual_flat_end_separates_where_capsule_end_does_not():
    # Cylinder ends at z=10 mm; box begins at 12 mm. A 3 mm
    # hemispherical planning extension incorrectly intersects that box.
    gap=separation([0,0,0],[0,0,.01],.003,[0,0,.013],np.eye(3),[.002,.002,.001])
    assert gap==pytest.approx(.002-1e-8)
    assert gap>.001


def test_original_one_mm_margin_is_not_relaxed():
    gap=separation([0,0,0],[0,0,.01],.003,[0,0,.012],np.eye(3),[.002,.002,.001])
    assert gap<.001


def test_side_collision_and_cap_contact_cannot_clear():
    for centre in ([.002,0,.005],[0,0,.011],[0,0,.005]):
        assert separation([0,0,0],[0,0,.01],.003,centre,np.eye(3),[.002,.002,.001])<=0


def test_rigid_motion_does_not_change_separation():
    rng=np.random.default_rng(14)
    a=np.array([0,0,0.]);b=np.array([0,0,.01]);c=np.array([.002,.001,.014]);half=np.array([.002,.002,.001])
    expected=separation(a,b,.003,c,np.eye(3),half)
    for _ in range(30):
        r=Rotation.random(random_state=rng).as_matrix();t=rng.uniform(-2,2,3)
        assert separation(r@a+t,r@b+t,.003,r@c+t,r,half)==pytest.approx(expected,abs=1e-12)


def test_projection_certificate_never_exceeds_independent_sample_pair_distance():
    rng=np.random.default_rng(15);angles=np.linspace(0,2*np.pi,129)
    # Dense actual-cylinder surface samples. Every sampled pair is an
    # independent upper bound on true distance; a false separating plane
    # cannot exceed even the closest sampled pair. Analytic cap/side cases
    # above independently cover the capsule-versus-cylinder distinction.
    cylinder=np.array([[.003*np.cos(v),.003*np.sin(v),z] for z in np.linspace(0,.01,11) for v in angles])
    for _ in range(50):
        r=Rotation.random(random_state=rng).as_matrix();c=rng.uniform(-.01,.02,3)
        half=np.array([.002,.003,.001])
        box=c+np.array(list(itertools.product((-1.,0.,1.),repeat=3)))*half@r.T
        upper=np.linalg.norm(cylinder[:,None]-box[None],axis=2).min()
        assert separation([0,0,0],[0,0,.01],.003,c,r,half)<=upper+1e-12


def screen(flat):
    from .held_plant_screen import HeldPlantScreen
    s=object.__new__(HeldPlantScreen);s.arm='right';s.workspace=None
    s.shapes=[('/robot/tool','/robot','ee_right','box',(np.array([0,0,.013]),np.eye(3),np.array([.002,.002,.001])))]
    s.obstacles=[('/plant/stem','capsule',(np.zeros(3),np.array([0,0,.01]),.003),None,None)]
    s.lower=np.array([[-.003,-.003,-.003]]);s.upper=np.array([[.003,.003,.013]])
    s.flat_cylinders={'/plant/stem'} if flat else set();s.flat_cylinder_refinements=0
    s.seam_paths=set();s.blade_path='/robot/tool'
    return s


def test_refinement_only_for_explicit_flat_geometry_not_real_capsules():
    a=screen(True);b=screen(False);world={'ee_right':np.eye(4)}
    assert a.check(world) and a.flat_cylinder_refinements==1
    assert not b.check(world) and b.flat_cylinder_refinements==0
    assert b.last_failure['plant_collider']=='/plant/stem'


@pytest.mark.parametrize('fault',['nan','zero_length','radius','half','rotation'])
def test_bad_geometry_refused(fault):
    a=np.zeros(3);b=np.array([0,0,.01]);radius=.003;c=np.ones(3);r=np.eye(3);half=np.ones(3)
    if fault=='nan':c[0]=np.nan
    if fault=='zero_length':b=a.copy()
    if fault=='radius':radius=0
    if fault=='half':half[0]=0
    if fault=='rotation':r[0,0]=2
    with pytest.raises(ValueError):separation(a,b,radius,c,r,half)
