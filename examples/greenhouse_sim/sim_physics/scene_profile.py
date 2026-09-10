"""Explicitly non-qualifying scene-removal timing controls. Never a demo."""
import json
import time
import numpy as np


def run(sim,rig,fixture,output):
    from pxr import Usd,UsdPhysics
    from .runtime import PlantRuntime
    from .implicit_springs import ImplicitJointSprings
    roots=['/World/Environment','/World/Gutters','/World/GutterWires','/World/PhysicsBackdrop']
    original={p:rig.stage.GetPrimAtPath(p).IsActive() for p in roots if rig.stage.GetPrimAtPath(p)}
    session=rig.stage.GetSessionLayer();snapshot=session.ExportToString()
    cases={'all':[], 'without_gutter_modules':[], 'without_gutter_body_apis':[], 'without_wires':['/World/GutterWires'],
           'without_gutters':['/World/Gutters'],
           'without_backdrop':['/World/PhysicsBackdrop'],
           'without_environment':['/World/Environment'],
           'without_background':roots}
    rows=[]
    try:
        for name,off in cases.items():
            sim.stop();rig.restore_authored_state();fixture.restore_authored_state()
            with Usd.EditContext(rig.stage,rig.stage.GetSessionLayer()):
                session.ImportFromString(snapshot)
                for p,active in original.items(): rig.stage.GetPrimAtPath(p).SetActive(active and p not in off)
                if name=='without_gutter_modules':
                    for gutter in rig.stage.GetPrimAtPath('/World/Gutters').GetChildren():
                        for prim in gutter.GetChildren():
                            if prim.GetName().startswith('Gutter_Module_'): prim.SetActive(False)
                if name=='without_gutter_body_apis':
                    for prim in Usd.PrimRange(rig.stage.GetPrimAtPath('/World/Gutters')):
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                                raise ValueError('Only disabled infrastructure APIs may be removed')
                            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                            prim.RemoveAPI(UsdPhysics.MassAPI)
            sim.reset();sim.step(render=False)
            runtime=PlantRuntime(rig,sim.physics_sim_view);springs=ImplicitJointSprings(runtime.articulation)
            fixture.bind(sim.physics_sim_view);times=[]
            for _ in range(120):
                fixture.target_palm(fixture.start[:3,3]);fixture.close(0.)
                springs.step(1/240,root_constrained=True)
                start=time.perf_counter();sim.step(render=False);times.append((time.perf_counter()-start)*1000)
                runtime.sample()
            row=dict(case=name,native_step_median_ms=float(np.median(times)),
                native_step_p95_ms=float(np.percentile(times,95)),removed_roots=off)
            rows.append(row);print('SCENE_ABLATION '+json.dumps(row),flush=True)
    finally:
        sim.stop()
        with Usd.EditContext(rig.stage,rig.stage.GetSessionLayer()):
            session.ImportFromString(snapshot)
            for p,active in original.items(): rig.stage.GetPrimAtPath(p).SetActive(active)
    result=dict(state='scene_ablation_diagnostic_not_qualification',cases=rows,
        physical_execution_approved=False,training_eligible=False)
    (output/'scene_profile.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
