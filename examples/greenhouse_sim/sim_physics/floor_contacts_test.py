import numpy as np
import pytest
from .floor_contacts import boxes,apply,tile_near


def mesh(cells):
    """Oriented boundary of unit cells; supports real holes and steps."""
    cells=set(cells);tri=[]
    for cell in cells:
        for axis in range(3):
            for sign in (-1,1):
                n=np.zeros(3,int);n[axis]=sign
                if tuple(np.array(cell)+n) in cells:continue
                u=np.zeros(3,int);u[(axis+1)%3]=1
                v=np.zeros(3,int);v[(axis+2)%3]=1
                p=np.array(cell);p=p+(n if sign>0 else 0)
                quad=np.array([p,p+u,p+u+v,p+v],float)
                if sign<0:quad=quad[::-1]
                tri.extend([quad[[0,1,2]],quad[[0,2,3]]])
    p=np.array(tri).reshape(-1,3)
    return p,np.full(len(tri),3),np.arange(len(p))


def test_exact_cuboid_coalesces_without_filling_holes_or_raised_steps():
    cells={(x,y,0) for x in range(3) for y in range(3)}-{(1,1,0)}
    cells.add((0,0,1))
    b,r=boxes(*mesh(cells))
    assert r['geometry_equivalent'] and not r['holes_or_steps_filled']
    assert r['box_union_volume_local3']==pytest.approx(9)
    for cell in cells|{(1,1,0),(1,1,1)}:
        centre=np.array(cell)+.5
        inside=sum(np.all(centre>lo)&np.all(centre<hi) for lo,hi in b)
        assert inside==int(cell in cells)


def test_closed_rectangular_solid_not_only_matching_bounding_box():
    b,r=boxes(*mesh({(0,0,0),(1,0,0)}))
    assert len(b)==1 and r['source_triangle_count']==20
    np.testing.assert_array_equal(b[0][0],[0,0,0]);np.testing.assert_array_equal(b[0][1],[2,1,1])


@pytest.mark.parametrize('fault',['open','slope','nan','winding'])
def test_unsupported_source_is_refused(fault):
    p,c,i=mesh({(0,0,0)})
    if fault=='open':c=c[:-1];i=i[:-3]
    elif fault=='slope':p[:,2]+=.1*p[:,0]
    elif fault=='nan':p[0,0]=float('nan')
    else:i[:3]=i[:3][::-1]
    with pytest.raises(ValueError):boxes(p,c,i)


def stage():
    from pxr import Usd,UsdGeom,UsdPhysics,Vt
    s=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(s,'/Floor')
    m=UsdGeom.Mesh.Define(s,'/Floor/Original');p,c,i=mesh({(0,0,0)})
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(c.astype(np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(i.astype(np.int32)))
    UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('none')
    return s,m


def test_session_collision_replacement_preserves_render_mesh_and_material():
    from pxr import UsdGeom,UsdPhysics,UsdShade,Sdf
    s,m=stage();material=UsdShade.Material.Define(s,'/Material')
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(.63)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material,materialPurpose='physics')
    m.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    m.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.002)
    before=s.GetRootLayer().ExportToString();points=np.asarray(m.GetPointsAttr().Get()).copy()
    r=apply(s,'/Floor');p=s.GetPrimAtPath(r['collider_paths'][0])
    assert s.GetRootLayer().ExportToString()==before
    np.testing.assert_array_equal(m.GetPointsAttr().Get(),points)
    assert m.ComputeVisibility()=='inherited'
    assert not UsdPhysics.CollisionAPI(m).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
    assert UsdGeom.Imageable(p).ComputeVisibility()=='invisible'
    bound,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose='physics')
    assert bound.GetPath()==material.GetPath()
    assert p.GetAttribute('physxCollision:contactOffset').Get()==pytest.approx(.002)
    assert r['native_requalification_required']


def test_failed_preflight_does_not_mutate_earlier_valid_shape():
    from pxr import UsdGeom,UsdPhysics
    s,m=stage();bad=UsdGeom.Cube.Define(s,'/Floor/Other')
    UsdPhysics.CollisionAPI.Apply(bad.GetPrim()).CreateCollisionEnabledAttr(True)
    before=s.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):apply(s,'/Floor')
    assert s.GetSessionLayer().ExportToString()==before


def test_actual_package_floor_three_exact_boxes_preserves_every_source_surface():
    from pxr import Usd,UsdGeom
    from sim_data.audit import DEFAULT_PACK
    from sim_data.floor_alignment import PACKAGE_FLOOR
    s=Usd.Stage.Open(str(DEFAULT_PACK/'house/green_house_base.usd'),Usd.Stage.LoadNone);s.Load(PACKAGE_FLOOR)
    before=s.GetRootLayer().ExportToString()
    r=apply(s,PACKAGE_FLOOR)
    assert len(r['collider_paths'])==3
    assert r['sources'][0]['source_triangle_count']==44
    assert s.GetRootLayer().ExportToString()==before


def test_local_tiles_retain_complete_exterior_union_and_small_station_cells():
    original=[(np.array([-70100.,-31570.,0.]),np.array([70100.,31570.,100.]))]
    scale=np.full(3,.001);translation=np.array([0.,0.,.001]);centre=[.3,-6.2]
    tiles=tile_near(original,scale,translation,centre)
    assert len(tiles)==64
    assert sum(np.prod(b-a) for a,b in tiles)==pytest.approx(np.prod(original[0][1]-original[0][0]))
    for i,(a,b) in enumerate(tiles):
        assert np.all(a>=original[0][0]) and np.all(b<=original[0][1])
        for c,d in tiles[i+1:]:assert not np.all(np.minimum(b,d)>np.maximum(a,c))
        mid=(a+b)/2*scale+translation
        if np.all(np.abs(mid[:2]-centre)<3):assert np.all((b-a)[:2]*scale[:2]<=1.+1e-12)


@pytest.mark.parametrize('centre',[[float('nan'),0],[0],[0,0,0]])
def test_local_tiles_invalid_centre_refused(centre):
    with pytest.raises(ValueError):tile_near([(np.zeros(3),np.ones(3))],np.ones(3),np.zeros(3),centre)


def test_actual_package_local_floor_partition_retains_source_visuals_and_volume():
    from pxr import Usd,UsdGeom
    from sim_data.audit import DEFAULT_PACK
    from sim_data.floor_alignment import PACKAGE_FLOOR
    s=Usd.Stage.Open(str(DEFAULT_PACK/'house/green_house_base.usd'),Usd.Stage.LoadNone);s.Load(PACKAGE_FLOOR)
    before=s.GetRootLayer().ExportToString()
    r=apply(s,PACKAGE_FLOOR,tile_centre_xy=[.3,-6.2])
    assert len(r['collider_paths'])==80
    assert s.GetRootLayer().ExportToString()==before
    assert r['sources'][0]['local_partition']['exterior_solid_retained']
    assert not r['sources'][0]['local_partition']['numerical_stability_verified']


def test_local_partition_requires_explicit_fixed_station_before_write(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Local floor partition'):
        main(['--output',str(tmp_path/'new'),'--local-floor-tiles-trial'])
    assert not (tmp_path/'new').exists()


def test_cli_requires_full_package_robot_before_creation(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Exact floor'):
        main(['--output',str(tmp_path/'new'),'--rectilinear-floor-contacts'])
    assert not (tmp_path/'new').exists()
