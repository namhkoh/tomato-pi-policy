"""Source geometry and fail-closed contact tests; not native cut qualification."""
import copy
import numpy as np
import pytest
from pxr import Usd,UsdGeom
from greenhouse_sim.robot_model import DEFAULT_ASSET
from .blade_contacts import refine_blade_contacts,LOWER_EDGE,SIDE_EDGE
from .knife import KnifeGeometry,ShearGate,ShearParameters,DOWNWARD_CUT_MODEL,KNIFE_IMPULSE_CONTRACT
from .plant_test import native


def test_lower_rim_keeps_every_original_contact_surface_and_arc_is_up():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/Robot').GetPrim().GetReferences().AddReference(str(DEFAULT_ASSET))
    original=stage.GetRootLayer().ExportToString()
    side=refine_blade_contacts(stage,'/Robot')
    snapshots=[UsdGeom.Mesh.Get(stage,p).GetPointsAttr().Get() for p in side['collider_paths']]
    lower=refine_blade_contacts(stage,'/Robot',edge_mode=LOWER_EDGE)
    for p,points in zip(lower['collider_paths'],snapshots,strict=True):
        np.testing.assert_array_equal(UsdGeom.Mesh.Get(stage,p).GetPointsAttr().Get(),points)
    k=object.__new__(KnifeGeometry);k.local=np.array(lower['edge_frame']);k.size=np.array(lower['edge_size_m'])
    assert np.linalg.det(k.local[:3,:3])==pytest.approx(1)
    assert k.size[[0,2]]==pytest.approx([.002,.002])
    assert k.local[2,3]-k.size[0]/2==pytest.approx(lower['plate_z_range_m'][0])
    for normal in ([1,0,0],[-1,0,0],[0,1,0],[0,-1,0]):
        wrist=k.wrist_for_edge([0,0,1],[0,0,-1],normal)
        # Here wrist == source frame. Actual +Z arc and plate normal are up.
        np.testing.assert_allclose(wrist[:3,2],[0,0,1],atol=1e-10)
        edge=k.frame(wrist)
        np.testing.assert_allclose(-edge[:3,0],[0,0,-1],atol=1e-10)
        assert np.linalg.det(wrist[:3,:3])==pytest.approx(1)
        assert not k.on_edge(edge[:3,3]+.01*edge[:3,2],edge)  # broad plate centre
    assert stage.GetRootLayer().ExportToString()==original


def gate():return ShearGate('petiole',ShearParameters(model=DOWNWARD_CUT_MODEL))


def sample(g,i,*,world=True,relative=True,load=.25,arc=(0,0,1),sideways=False):
    edge=np.eye(4);edge[:3,:3]=np.array([[0,0,1],[0,-1,0],[1,0,0]])
    edge[2,3]=-i*.0001 if world else 0.
    centre=np.array([0.,0.,edge[2,3]+i*.0001 if relative else edge[2,3]])
    if sideways:edge[:3,:3]=np.eye(3)
    direction=-edge[:3,0];normal=-direction
    return g.observe(dt=.005,edge=edge,centre=centre,axis=np.array([1.,0,0]),
        points=[centre],normals=[normal],impulses=[normal*load*.005],
        impulse_contract=KNIFE_IMPULSE_CONTRACT,edge_contact_verified=True,
        tool_contact_upper_bound_n=abs(load),held=True,slip=.001,arc_up=arc,edge_mode=LOWER_EDGE)


def test_measured_world_and_relative_downward_loaded_motion_are_both_required():
    g=gate();events=[sample(g,i) for i in range(20)]
    assert sum(e is not None for e in events)==1
    e=next(e for e in events if e)
    assert e['measured_world_downward_travel_m']>=.0003
    assert e['measured_relative_loading_travel_m']>=.0003
    assert e['minimum_source_arc_up_cosine']==1


@pytest.mark.parametrize('kwargs',[dict(world=False),dict(relative=False),dict(load=.1),
    dict(load=.6),dict(arc=(0,0,-1)),dict(arc=(1,0,0)),dict(sideways=True)])
def test_pressure_only_moving_target_wrong_face_and_wrong_force_never_release(kwargs):
    g=gate();assert all(sample(g,i,**kwargs) is None for i in range(30))


def test_micro_motion_is_not_a_cut():
    g=gate();assert all(sample(g,(i%2)*.008) is None for i in range(1000))


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_both_explicit_native_modes_validate_without_loosening_other_guards(tmp_path,monkeypatch,mode):
    from pathlib import Path
    from .ground_truth_trial import arguments
    from .benchmark import main
    out=tmp_path/'not_launched';args=arguments(out,mode,'cut_action')
    args+=['--knife-edge-mode',LOWER_EDGE,'--cut-model',DOWNWARD_CUT_MODEL]
    class Validated(Exception):pass
    def stop(self,*a,**k):assert self==out;raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(args)
    for bad in (['--knife-edge-mode',SIDE_EDGE],['--cut-style','legacy']):
        with pytest.raises(ValueError):main(args+bad)


@pytest.mark.parametrize('key,value',[
    ('edge_mode',SIDE_EDGE),('measured_world_downward_travel_m',0.),
    ('minimum_world_downward_cosine',0.),('minimum_source_arc_up_cosine',-1.),
    ('world_travel_definition','commanded'),('loading_travel_required',False)])
def test_release_api_independently_rejects_incomplete_downward_evidence(native,key,value):
    from .plant import build
    stage,record=native;rig=build(stage,record,'SubStem_41')
    g=gate();e=next(e for i in range(20) if (e:=sample(g,i)))
    e['target']=rig.source_target;bad=copy.deepcopy(e);bad[key]=value
    with pytest.raises(ValueError):rig.release_from_blade(bad)
    assert not rig.cut
    assert rig.release_from_blade(e)['physical_cut_verified'] is False
