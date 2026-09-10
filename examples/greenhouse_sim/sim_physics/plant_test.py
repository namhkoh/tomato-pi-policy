from pathlib import Path

import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics

from sim_data.audit import DEFAULT_PACK,audit_manifest
from sim_data.geometry import assemble_plant
from sim_physics.plant import build


@pytest.fixture
def native():
    manifest=DEFAULT_PACK/'plants/components/seed101_full/manifest.json'
    if not manifest.is_file(): pytest.skip('Supplied greenhouse package not installed')
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    report=audit_manifest(manifest)
    paths=assemble_plant(stage,'/World/Plant',report)
    return stage,dict(manifest_path=str(manifest),component_paths=paths,plant_root='/World/Plant')


def test_native_visuals_seam_and_physics_are_session_only(native):
    stage,record=native; before=stage.GetRootLayer().ExportToString()
    rig=build(stage,record,'SubStem_56')
    assert stage.GetRootLayer().ExportToString()==before
    assert rig.report()['cut_material_arc_m']==.01
    assert len(rig.visuals)>=2 and len(rig.body_paths)>3
    assert UsdPhysics.Joint.Get(stage,rig.cut_joint_path).GetExcludeFromArticulationAttr().Get()
    assert stage.GetPrimAtPath(rig.cut_joint_path).IsA(UsdPhysics.FixedJoint)
    assert stage.GetPrimAtPath(rig.body_paths[rig.cut_index]).HasAPI(UsdPhysics.ArticulationRootAPI)
    for i,path in enumerate(rig.body_paths):
        assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get()==(i<rig.cut_index)
        # Codeless schema is authored even when offline USD has no PhysX plugin.
        assert 'PhysxContactReportAPI' in stage.GetPrimAtPath(path).GetMetadata('apiSchemas').GetAddedOrExplicitItems()
    assert rig.report()['training_eligible'] is False
    source=stage.GetPrimAtPath(record['component_paths']['SubStem_56'])
    assert all(not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
               for p in Usd.PrimRange(source) if p.HasAPI(UsdPhysics.CollisionAPI))
    parent=stage.GetPrimAtPath(record['component_paths']['MainStem_37'])
    assert any(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
               for p in Usd.PrimRange(parent) if p.HasAPI(UsdPhysics.CollisionAPI))


def test_deleafed_stub_and_duplicate_rig_rejected(native):
    stage,record=native
    with pytest.raises(ValueError): build(stage,record,'SubStem_00')
    assert not stage.GetPrimAtPath('/World/InteractionPhysics/Target')
    build(stage,record,'SubStem_56')
    with pytest.raises(ValueError): build(stage,record,'SubStem_56')


def test_release_does_not_claim_contact_cut_and_reset_restores_it(native):
    stage,record=native;rig=build(stage,record,'SubStem_56')
    before=stage.GetRootLayer().ExportToString()
    event=rig.diagnostic_release()
    assert event['physical_cut_verified'] is False and event['material_arc_m']==.01
    assert not UsdPhysics.Joint.Get(stage,rig.cut_joint_path).GetJointEnabledAttr().Get()
    with pytest.raises(ValueError): rig.diagnostic_release()
    rig.restore_authored_state()
    assert stage.GetRootLayer().ExportToString()==before
    assert UsdPhysics.Joint.Get(stage,rig.cut_joint_path).GetJointEnabledAttr().Get()


def test_fixed_base_diagnostic_does_not_silently_release(native):
    stage,record=native
    rig=build(stage,record,'SubStem_56',constraint_mode='fixed_articulation')
    joint=UsdPhysics.Joint.Get(stage,rig.cut_joint_path)
    assert joint.GetBody0Rel().GetTargets()==[]
    assert joint.GetPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not joint.GetExcludeFromArticulationAttr().Get()
    np.testing.assert_allclose(joint.GetLocalPos0Attr().Get(),rig.chain_world[rig.cut_index],atol=1e-7)
    with pytest.raises(ValueError,match='topology transition'):
        rig.diagnostic_release()
    assert not rig.cut and joint.GetJointEnabledAttr().Get()


def test_invalid_constraint_mode_has_no_stage_side_effects(native):
    stage,record=native
    before=stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError,match='constraint mode'):
        build(stage,record,'SubStem_56',constraint_mode='unqualified')
    assert stage.GetSessionLayer().ExportToString()==before
