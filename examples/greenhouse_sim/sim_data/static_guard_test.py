import pytest

from sim_data.static_guard import StaticSceneMonitor,relevant_path


@pytest.mark.parametrize('path', ['/World','/','/World/Plant.points','/Looks/Material.inputs:color','/World/RBY1/link_head_2.xformOp:transform'])
def test_scene_and_material_paths_are_watched(path):
    assert relevant_path(path)


@pytest.mark.parametrize('path',['/Render','/Render/Product.camera','/Replicator','/Replicator/Graph','/Orchestrator','/Orchestrator/Graph.node:type'])
def test_only_renderer_bookkeeping_is_excluded(path):
    assert not relevant_path(path)


def test_notice_guard_allows_explicit_snapshot_only_between_frames():
    from pxr import Usd,UsdGeom,Gf
    stage=Usd.Stage.CreateInMemory()
    robot=UsdGeom.Xform.Define(stage,'/World/RBY1')
    pose=robot.AddTranslateOp()
    plant=UsdGeom.Cube.Define(stage,'/World/Plant')
    render=UsdGeom.Xform.Define(stage,'/Render/Bookkeeping')
    monitor=StaticSceneMonitor(stage)
    try:
        a=monitor.begin()
        pose.Set(Gf.Vec3d(.2,0,0))
        assert monitor.token()!=a  # Mid-frame robot movement must fail too.
        b=monitor.begin()          # Only caller-controlled between-frame pose acceptance.
        assert b==monitor.token()
        render.AddTranslateOp().Set(Gf.Vec3d(0,0,1))
        assert monitor.token()==b
        plant.GetSizeAttr().Set(.8)
        assert monitor.token()!=b
        with pytest.raises(ValueError,match='Unapproved static scene change'): monitor.begin()
    finally: monitor.close()


def test_notice_guard_rejects_non_world_material_changes():
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World')
    shader=UsdShade.Shader.Define(stage,'/Looks/Shader')
    colour=shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f)
    colour.Set(Gf.Vec3f(1))
    monitor=StaticSceneMonitor(stage)
    try:
        before=monitor.begin()
        colour.Set(Gf.Vec3f(0))
        assert monitor.token()!=before
        with pytest.raises(ValueError): monitor.begin()
    finally: monitor.close()
