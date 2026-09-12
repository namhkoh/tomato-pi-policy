import pytest
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
from sim_physics.isolated_station import isolate
from sim_physics.benchmark import main, parser


def test_matched_source_plant_floor_resources_and_transforms_survive():
    stage=Usd.Stage.CreateInMemory()
    for path in ('/World/Plant/Stem','/World/Plant/Leaf',
                 '/World/Environment/floor','/World/Environment/wall','/World/Gutters/gutter'):
        UsdGeom.Cube.Define(stage,path)
    plant=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant'))
    plant.AddTranslateOp().Set(Gf.Vec3d(-.005,0,.9))
    UsdShade.Material.Define(stage,'/World/Looks/PlantMaterial')
    UsdLux.DomeLight.Define(stage,'/World/Lighting/Dome')
    before=stage.GetRootLayer().ExportToString()
    matrix=plant.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    r=isolate(stage,floor_root='/World/Environment/floor')
    assert stage.GetRootLayer().ExportToString()==before
    assert not r['full_greenhouse_qualified'] and not r['training_eligible']
    assert r['removed_scene_roots']==['/World/Environment/wall','/World/Gutters']
    for path in ('/World/Plant/Stem','/World/Plant/Leaf','/World/Environment/floor',
                 '/World/Looks/PlantMaterial','/World/Lighting/Dome'):
        assert stage.GetPrimAtPath(path).IsActive()
    assert plant.ComputeLocalToWorldTransform(Usd.TimeCode.Default())==matrix
    assert not stage.GetPrimAtPath('/World/Gutters').IsActive()


@pytest.mark.parametrize('floor',['/','/World','/Missing','/World/Plant'])
def test_invalid_scope_fails_without_edits(floor):
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/Plant')
    before=stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):isolate(stage,floor_root=floor)
    assert stage.GetSessionLayer().ExportToString()==before


@pytest.mark.parametrize('extra',[[],['--scene','package'],
    ['--scene','package','--full-robot-probe','--local-wire-physics'],
    ['--scene','package','--full-robot-probe','--context-gutters','3'],
    ['--scene','package','--full-robot-probe','--batch-gutter-visuals'],
    ['--scene','package','--full-robot-probe','--scene-profile']])
def test_invalid_cli_fails_before_output_or_kit(tmp_path,extra):
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='Isolated station requires'):
        main(['--output',str(output),'--isolate-station',*extra])
    assert not output.exists()


def test_normal_package_default_keeps_all_surroundings():
    assert not parser().parse_args(['--output','unused','--scene','package']).isolate_station
