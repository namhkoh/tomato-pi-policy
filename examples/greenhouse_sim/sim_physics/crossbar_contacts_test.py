"""Source-preservation/edge mapping tests; no native success claimed."""
import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics
from greenhouse_sim.robot_model import DEFAULT_ASSET
from .blade_contacts import refine_blade_contacts,LOWER_EDGE,CROSSBAR_EDGE
from .arc_contacts import refine_arc_contacts
from .knife import KnifeGeometry,mount_forward


def source():
    s=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s,'/Robot').GetPrim().GetReferences().AddReference(str(DEFAULT_ASSET))
    return s,'/Robot/ee_right/attachments/DeleafKnife'


def test_actual_beveled_bar_not_mounting_plate_is_cutting_surface():
    s,root=source();before=s.GetRootLayer().ExportToString()
    mount_forward(s,'/Robot',alignment='camera')
    refine_blade_contacts(s,'/Robot',edge_mode=LOWER_EDGE)
    result=refine_arc_contacts(s,'/Robot',crossbar_edge=True)
    k=KnifeGeometry(s,'/Robot')
    assert k.collider==root+'/CrossbarContact'
    assert k.edge_mode==CROSSBAR_EDGE
    assert result['edge_from_source_bevel'] is True
    assert result['legacy_component']=='Arc'
    assert result['physical_part']=='straight_beveled_lower_crossbar'
    assert .025<k.size[1]<.035
    assert k.size[[0,2]]==pytest.approx([.001,.001])
    assert k.source_crossbar_half_thickness_m==pytest.approx(.0015,abs=1e-9)
    for normal in ([1,0,0],[-1,0,0]):
        wrist=k.wrist_for_edge([0,0,1],[0,0,-1],normal)
        edge=k.frame(wrist)
        assert k.arc_up(wrist)[2]>np.cos(np.radians(3))
        assert k.on_edge(edge[:3,3]-.0005*edge[:3,0],edge)
        assert not k.on_edge(edge[:3,3]+.008*edge[:3,0],edge)  # broad side/back
    assert s.GetRootLayer().ExportToString()==before
    assert all(UsdPhysics.CollisionAPI(s.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get()
        for p in result['collider_paths'])


def test_all_original_arc_and_blade_surfaces_retained_without_window_hull():
    from scipy.spatial import ConvexHull
    from .crossbar_contacts import geometry
    s,root=source();mesh=UsdGeom.Mesh.Get(s,root+'/Arc')
    p=np.asarray(mesh.GetPointsAttr().Get());c=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    result=geometry(p,c,ids);pieces=[*result['support'],result['bar']]
    def area(t):return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()/2
    assert sum(area(t.reshape(-1,3,3)) for t in pieces)==pytest.approx(area(p[ids].reshape(-1,3,3)),rel=2e-7)
    assert np.ptp(result['bar'][:,0])==pytest.approx(.003)
    assert result['bar'][:,2].max()<.02
    for vertices in pieces:
        hull=ConvexHull(vertices)
        assert not np.all(hull.equations[:,:3]@np.array([0.,-.09,.026])+hull.equations[:,3]<=0)
    original_blade=np.asarray(UsdGeom.Mesh.Get(s,root+'/Blade').GetPointsAttr().Get()).copy()
    refine_blade_contacts(s,'/Robot',edge_mode=LOWER_EDGE)
    refine_arc_contacts(s,'/Robot',crossbar_edge=True)
    np.testing.assert_array_equal(UsdGeom.Mesh.Get(s,root+'/Blade').GetPointsAttr().Get(),original_blade)


def test_unknown_or_duplicate_crossbar_geometry_fails_closed():
    from .crossbar_contacts import geometry
    with pytest.raises(ValueError):geometry(np.zeros((3,3)),[3],[0,1,2])
    s,_=source();refine_arc_contacts(s,'/Robot')
    with pytest.raises(ValueError,match='before authoring'):refine_arc_contacts(s,'/Robot',crossbar_edge=True)
    s,_=source();refine_arc_contacts(s,'/Robot',crossbar_edge=True)
    with pytest.raises(ValueError,match='already authored'):refine_arc_contacts(s,'/Robot')


@pytest.mark.parametrize('tilt,load,expected',[(0.,.25,True),(10.,.25,False),
    (90.,.25,False),(180.,.25,False),(0.,.1,False),(0.,.6,False)])
def test_source_crossbar_requires_downward_alignment_and_bounded_load(tilt,load,expected):
    from scipy.spatial.transform import Rotation
    from .knife import ShearGate,ShearParameters,DOWNWARD_CUT_MODEL,KNIFE_IMPULSE_CONTRACT
    s,_=source();refine_blade_contacts(s,'/Robot',edge_mode=LOWER_EDGE)
    refine_arc_contacts(s,'/Robot',crossbar_edge=True);k=KnifeGeometry(s,'/Robot')
    g=ShearGate('target',ShearParameters(model=DOWNWARD_CUT_MODEL))
    rotation=Rotation.from_euler('y',tilt,degrees=True).as_matrix()
    events=[]
    for i in range(20):
        w=k.wrist_for_edge([0,0,1-i*.0001],[0,0,-1],[1,0,0])
        w[:3,:3]=rotation@w[:3,:3]
        edge=k.frame(w);normal=edge[:3,0]
        events.append(g.observe(dt=.005,edge=edge,centre=[0,0,1],axis=[1,0,0],
            points=[edge[:3,3]],normals=[normal],impulses=[normal*load*.005],
            impulse_contract=KNIFE_IMPULSE_CONTRACT,edge_contact_verified=True,
            tool_contact_upper_bound_n=load,held=True,slip=.001,
            arc_up=k.arc_up(w),edge_mode=CROSSBAR_EDGE))
    assert any(e is not None for e in events)==expected
    if tilt:assert 'arc_up_and_world_down' in g.diagnostic['failed_conditions']


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_crossbar_native_profile_validates_without_launching(tmp_path,monkeypatch,mode):
    from pathlib import Path
    from .ground_truth_trial import arguments
    from .benchmark import main
    from .knife import DOWNWARD_CUT_MODEL
    out=tmp_path/'not_launched'
    args=arguments(out,mode,'cut_action')+['--knife-edge-mode',CROSSBAR_EDGE,'--cut-model',DOWNWARD_CUT_MODEL]
    class Validated(Exception):pass
    def stop(self,*a,**k):assert self==out;raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(args)
