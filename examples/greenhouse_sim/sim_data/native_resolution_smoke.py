"""Native higher-resolution sensor qualification; never a training collector.

Run with Isaac Python only, after other native capture workers have exited.
Diagnostic cube/occluder images are not greenhouse training observations.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import traceback
import time
import numpy as np
from .capture_contract import jsonable
from .capture_sensor import HIRES_RESOLUTION,sensor_profile,calibration_for_native_resolution,checked_resolution
from .dataset_review import require,write_json
from .native_sensor_payload import validate_native_static,decode_native_instances


def known_surface(depth,valid,ids,mapping,resolution,expected_path,expected_z):
    width,height=checked_resolution(resolution)
    require(depth.shape==valid.shape==ids.shape==(height,width),"Native diagnostic buffer dimensions mismatch")
    require(np.isfinite(expected_z) and expected_z>0,"Positive metric reference required")
    probes=[(width//2,height//2),(width//2+int(round(46*width/848)),height//2)]
    checks=[]
    for x,y in probes:
        require(bool(valid[y,x]),"Known diagnostic surface has invalid native depth")
        measured=float(depth[y,x])
        require(abs(measured-expected_z)<=.002,"Known optical-Z surface mismatch")
        require(mapping.get(int(ids[y,x]))==expected_path,"Known native instance identity mismatch")
        checks.append(dict(pixel_xy=[x,y],native_z_m=measured,expected_z_m=expected_z,
                           native_prim_path=mapping[int(ids[y,x])]))
    return checks


def smoke(app,output):
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,UsdGeom,UsdLux
    from .capture_scene import calibration,scene_guard
    from .capture_pilot import make_writer,step_payload
    from .review_camera import HEAD_CAMERA
    from PIL import Image
    resolution=HIRES_RESOLUTION
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,"Z")
    cube=UsdGeom.Cube.Define(stage,"/World/CalibrationCube");cube.CreateSizeAttr(1)
    cube_op=cube.AddTranslateOp();cube_op.Set(Gf.Vec3d(0,0,-2.5))
    UsdLux.DomeLight.Define(stage,"/World/Light").CreateIntensityAttr(1000)
    camera=UsdGeom.Camera.Define(stage,HEAD_CAMERA)
    camera.CreateFocalLengthAttr(24)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955*408/848)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    camera_op=camera.AddTranslateOp();camera_op.Set(Gf.Vec3d(0))
    product=rep.create.render_product(HEAD_CAMERA,resolution)
    # The fast mapper is still fixed-resolution production code. Qualify the
    # native legacy instance annotator here without patching running workers.
    writer=make_writer(rep,include_instances=True,instance_backend="legacy")
    writer.attach([product]);checks=[];previous=None
    def capture(name,expected_path,expected_z,previous_token=None):
        cal=calibration_for_native_resolution(calibration(stage),resolution)
        # Established fixed 56-subframe reference; no short-budget claim.
        for _ in range(6):step_payload(rep,writer,subframes=8)
        before=scene_guard(stage)
        payload=step_payload(rep,writer,subframes=8)
        rgb,z,valid,ref,token=validate_native_static(
            payload,cal,before,scene_guard(stage),writer.sequence,previous_token)
        ids,mapping=decode_native_instances(payload,resolution)
        evidence=known_surface(z,valid,ids,mapping,resolution,expected_path,expected_z)
        folder=output/name;folder.mkdir()
        Image.fromarray(rgb).save(folder/"rgb_diagnostic.png")
        np.save(folder/"native_optical_z_m.npy",z,allow_pickle=False)
        Image.fromarray(valid.astype(np.uint8)*255).save(folder/"validity_diagnostic.png")
        np.save(folder/"native_instance_ids.npy",ids,allow_pickle=False)
        rec=dict(name=name,calibration=cal,native_camera_params=jsonable(payload["camera_params"]),
                 writer_reference=ref,native_render_frame=jsonable(payload["pilot_render_frame"]),
                 freshness=token,known_surface=evidence,scene_guard=before,
                 no_training_input=True,native_resolution=list(resolution))
        write_json(folder/"diagnostic.json",rec)
        checks.append(rec)
        print("NATIVE_HIRES_SENSOR_CHECK",name,evidence,flush=True)
        return token
    try:
        for i,(offset,z) in enumerate(((0.,2.),(.15,2.4))):
            camera_op.Set(Gf.Vec3d(offset,0,0));cube_op.Set(Gf.Vec3d(0,0,-z-.5))
            previous=capture(f"pose_{i}","/World/CalibrationCube",z,previous)
        blocker=UsdGeom.Cube.Define(stage,"/World/KnownOccluder")
        # Large enough to cover the off-axis optical-Z probe at 1 m.
        blocker.CreateSizeAttr(.4);blocker.AddTranslateOp().Set(Gf.Vec3d(.15,0,-1.2))
        for visible in (True,False):
            blocker.CreateVisibilityAttr("inherited" if visible else "invisible")
            capture("occluder_visible" if visible else "occluder_hidden",
                    "/World/KnownOccluder" if visible else "/World/CalibrationCube",
                    1. if visible else 2.4)
    finally:
        writer.detach();product.destroy()
    return dict(state="native_resolution_smoke_passed_not_training_data",
                sensor=sensor_profile(resolution),checks=checks,
                rgb_upscaling_performed=False,native_depth_reconstructed=False,
                greenhouse_capture_qualified=False,mounted_robot_pose_qualified=False,
                training_approved=False,instance_backend="legacy",
                render_budget="established_7_steps_x_8_subframes")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);output=a.output.resolve()
    require(not output.exists(),"Choose a new diagnostic output")
    output.mkdir(parents=True)
    from sim_physics.host_memory import preflight as check_memory
    preflight=check_memory()
    write_json(output/"request.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        sensor=sensor_profile(HIRES_RESOLUTION),host_memory_preflight=preflight,
        training_started=False,source_assets_changed=False))
    app=None
    succeeded=False
    try:
        require(preflight["allowed"],"Native launch memory reserve unavailable; no override")
        from isaacsim import SimulationApp
        started=time.perf_counter()
        app=SimulationApp(dict(headless=True,width=HIRES_RESOLUTION[0],height=HIRES_RESOLUTION[1],multi_gpu=False))
        result=smoke(app,output)
        result["elapsed_including_startup_s"]=time.perf_counter()-started
        write_json(output/"result.json",result)
        succeeded=True
    except BaseException:
        write_json(output/"failure.json",dict(state="native_resolution_smoke_failed",
            traceback=traceback.format_exc(),training_approved=False))
        raise
    finally:
        if app is not None:app.close(exit_code=0 if succeeded else 1)


if __name__=="__main__":main()
