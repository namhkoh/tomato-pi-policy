import pytest
from pxr import Usd,UsdGeom,UsdPhysics
from sim_physics.solver_configuration import uniform_iterations,verify_iterations
from sim_physics.benchmark import main


def stage():
    s=Usd.Stage.CreateInMemory()
    for root in ('/World/Plant','/World/Robot'):
        for i in range(3):UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(s,root+f'/link{i}').GetPrim())
        UsdPhysics.ArticulationRootAPI.Apply(s.GetPrimAtPath(root+'/link0'))
    return s


def test_all_bodies_and_articulations_receive_exact_pair_session_only():
    s=stage();before=s.GetRootLayer().ExportToString()
    r=uniform_iterations(s,('/World/Plant','/World/Robot'),(128,0))
    assert len(r['bindings'])==8 and not r['native_effective_iterations_verified']
    assert s.GetRootLayer().ExportToString()==before
    verify_iterations(s,r)
    p=r['bindings'][-1]
    with Usd.EditContext(s,s.GetSessionLayer()):
        s.GetPrimAtPath(p['path']).GetAttribute(p['prefix']+':solverVelocityIterationCount').Set(8)
    with pytest.raises(RuntimeError):verify_iterations(s,r)


@pytest.mark.parametrize('pair',[(128,True),(128,-1),(1000,0),(128,),('128',0)])
def test_invalid_pair_cannot_edit_stage(pair):
    s=stage();before=s.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):uniform_iterations(s,('/World/Plant','/World/Robot'),pair)
    assert s.GetSessionLayer().ExportToString()==before


def test_cli_cannot_apply_iterations_to_production_or_cutting(tmp_path):
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='Uniform iterations require'):
        main(['--output',str(output),'--uniform-solver-iterations','128','0'])
    assert not output.exists()
