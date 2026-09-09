"""One isolated full-greenhouse native-plant capture job; no lab robot access."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from .collection_plan import load_plan
from .dataset_review import require, verify_bindings, write_json
from .depth_preview import sha256


def prepare_views(stage, robot, variants, rows, count, *, grounding=False, vary_torso=False, view_offset=0):
    """Search disposable poses, then restore the full root/link snapshot even on failure."""
    import numpy as np
    from pxr import Usd, UsdGeom
    from greenhouse_sim.robot_hardware import _set_transform
    root = stage.GetPrimAtPath(robot['root'])
    require(bool(root), 'Missing robot root')
    # set_snapshot_pose changes the root and its flat URDF link children only.
    snapshots = [(prim, np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T.copy())
                 for prim in [root, *root.GetChildren()] if prim.IsA(UsdGeom.Xformable)]
    try:
        if grounding:
            return _prepare_views(stage, robot, variants, rows, count, grounding=True, vary_torso=vary_torso, view_offset=view_offset)
        return _prepare_views(stage, robot, variants, rows, count)
    finally:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for prim, matrix in snapshots:
                _set_transform(prim, matrix[:3,:3], matrix[:3,3])


def _prepare_views(stage, robot, variants, rows, count, *, grounding=False, vary_torso=False, view_offset=0):
    # All USD imports stay AFTER SimulationApp startup in this module.
    import numpy as np
    from pxr import UsdGeom
    from .capture_contract import project
    from .capture_scene import calibration, set_snapshot_pose, target_world_geometry
    from .capture_viewpoints import static_obstacles, scene_triangle_refiner, screen_bounds, visible_bounds
    from .viewpoint_plan import focus_specs, select_planned
    original = float(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot['root']))[3][0])
    obstacles, refine = static_obstacles(stage, robot['root']), scene_triangle_refiner(stage)
    fast_screen=None
    if grounding:
        from .training_screen import StaticBoundScreen
        fast_screen=StaticBoundScreen(obstacles,refine)
    by_variant = {v['variant_id']: v for v in variants}
    result = {'schema_version': 'greenhouse.native_job_view_plan.v1', 'original_base_x_m': original,
              'selected': {}, 'target_world_m': {}, 'decisions': [], 'rendered_visibility_checked': False}
    if grounding:
        result.update(schema_version='greenhouse.grounding_job_view_plan.v1', requested_views_per_target=count)
        if vary_torso: result['vary_torso']=True
        if view_offset: result['view_offset']=view_offset
    for row in rows:
        world = target_world_geometry(stage, by_variant[row['variant_id']], row)
        result['target_world_m'][row['draft_id']] = world['nominal_world_m']
        viable = []
        if grounding:
            from .training_plan import view_specs
            specs = view_specs(original, world['nominal_world_m'][0], row['draft_id'], count,
                               vary_torso=vary_torso, view_offset=view_offset)
        else:
            specs = focus_specs(original, world['nominal_world_m'][0])
        for spec in specs:
            decision = {'target_review_id': row['draft_id'], **spec}
            result['decisions'].append(decision)
            try:
                from .training_views import robot_for_spec
                pose = set_snapshot_pose(stage, robot_for_spec(robot,spec), world['nominal_world_m'], spec['y_offset_m'],
                    spec['desired_pixel_xy'], root_x_m=spec['root_x_m'], root_yaw_degrees=spec['root_yaw_degrees'])
                cal = calibration(stage)
                nominal, interval = project([world['nominal_world_m']], cal)[0], project(world['interval_world_m'], cal)
                diameter = 2*row['cut_region_proposal']['nominal']['petiole_radius_m']*cal['intrinsics'][0][0]/nominal['camera_optical_xyz_m'][2]
                length = float(np.linalg.norm(np.diff([p['pixel_xy'] for p in interval], axis=0), axis=1).sum())
                decision.update(predicted_diameter_px=diameter, predicted_interval_px=length,
                    base_xy_m=np.asarray(pose['robot_root_to_world_usd_row_vectors'])[3, :2].tolist())
                if diameter < (4 if vary_torso else 3) or length < (6 if vary_torso else 5) or not all(p['projection_status']=='in_frame' for p in interval):
                    decision['state'] = 'rejected_projected_sampling'
                    continue
                bounds=visible_bounds(stage,robot['root'])
                screen = fast_screen(bounds) if fast_screen else screen_bounds(bounds, obstacles, refinement=refine)
                decision['visual_bound_screen'] = screen
                decision['state'] = 'geometry_screen_passed_visibility_unknown' if screen['passed'] else 'rejected_possible_geometry_overlap'
                if screen['passed']:
                    viable.append(decision)
                    if grounding and len(viable) >= count:
                        break
            except ValueError as exc:
                decision.update(state='rejected_pose', reason=str(exc))
        result['selected'][row['draft_id']] = select_planned(viable, count)
        print('NATIVE_VIEW_PLAN', row['draft_id'], 'viable', len(viable), 'selected', len(result['selected'][row['draft_id']]), flush=True)
    return result


def capture_job(app, args, plan, reports, job, manifest):
    import carb
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom
    from launch_sim_data import load_local_payloads, populate
    from .capture_pilot import calibration_smoke, source_hashes
    from .capture_scene import capture_root, freeze_rigid_bodies
    from .capture_search import run_refined_capture
    from .floor_alignment import PACKAGE_FLOOR
    from .robot_preview import add_robot_preview
    manifest['instance_annotation_backend']=args.instance_backend
    manifest['render_budget_profile']=args.render_budget
    manifest['calibration_smoke'] = calibration_smoke(app, rep, include_instances=True, instance_backend=args.instance_backend)
    rep.set_global_seed(plan['configuration']['seed'])
    package = Path(plan['package'])
    scene = package/'house/green_house_base.usd'
    context = omni.usd.get_context()
    wrapper = capture_root(scene)
    context.new_stage()
    stage = context.get_stage()
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = list(wrapper.subLayerPaths)
    require(stage.GetRootLayer().anonymous and bool(stage.GetPrimAtPath('/World/Gutters')), 'Expected original greenhouse wrapper')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = load_local_payloads(stage)
    records = []
    gutter_x, counts = populate(stage, package, app, records, review_plant=job['plant_family'])
    variants = []
    for record in records:
        family = Path(record['manifest_path']).parent.name
        # Native catalogue entries, explicitly not generated branch variants.
        variants.append({'variant_id': family, 'source_plant_id': family, 'split_group': family,
            'plant_root': record['plant_root'], 'added_components': {}, 'added_component_paths': {},
            'source_geometry_modified': False})
    require(job['plant_family'] in {v['variant_id'] for v in variants}, 'Target plant did not load')
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool='gripper')
    freeze_rigid_bodies(stage)
    sys.path.insert(0, str(package/'env_panel'))
    from tomato_env import daylight
    lighting = daylight.apply(stage, day=172, minutes=13*60, intensity=1500, dome_intensity=1200)
    settings=carb.settings.get_settings()
    requested_mode=plan['configuration'].get('renderer_mode')
    if requested_mode is not None:
        manifest['scene_renderer_before_explicit_selection']=settings.get('/rtx/rendermode')
        if requested_mode=='RaytracedLighting':
            key='/persistent/rtx/modes/rt2/enabled'
            original=settings.get(key)
            require(type(original) is bool,'Unknown persistent RT2 switch; refuse to change it')
            manifest['temporary_renderer_switch_restore']={'path':key,'original':original}
            settings.set_bool(key,False)
        settings.set('/rtx/rendermode',requested_mode)
        app.update()
        require(settings.get('/rtx/rendermode')==requested_mode,'Scene overrode the explicitly requested renderer')
    renderer_settings={name:settings.get(name) for name in ('/rtx/rendermode','/rtx/post/aa/op',
        '/rtx/post/histogram/enabled','/rtx/post/histogram/tau','/rtx/post/tonemap/op',
        '/rtx/pathtracing/spp','/rtx/rtpt/maxBounces','/persistent/rtx/modes/rt2/enabled','/app/settings/persistent')}
    manifest['observed_renderer_settings']=renderer_settings
    print('OBSERVED_RENDERER_SETTINGS',renderer_settings,flush=True)
    context.get_selection().clear_selected_prim_paths()
    omni.timeline.get_timeline_interface().pause()
    carb.settings.get_settings().set_bool('/app/runLoops/main/rateLimitEnabled', False)
    manifest.update(package=str(package), scene=str(scene), scene_counts=counts, robot=robot,
        source_catalogue_kind='native_collection_plan.v1', source_collection_plan_path=str(args.plan.resolve()),
        source_collection_plan_sha256=sha256(args.plan), collection_job_id=job['job_id'],
        source_usd_sha256=source_hashes(stage), cut_rule=plan['cut_rule'], cut_rule_sha256=plan['cut_rule_sha256'],
        variants=variants, lighting=lighting, renderer=renderer_settings['/rtx/rendermode'], unbundled_external_prop_roots_excluded=excluded,
        target_family_split=job['split'], split_scope=plan['split_scope'], source_geometry_modified=False)
    prepared = prepare_views(stage, robot, variants, job['targets'], job['max_rendered_views_per_target'],
        grounding=plan['schema_version']=='greenhouse.grounding_collection_plan.v1',
        vary_torso=plan['configuration'].get('vary_torso',False),view_offset=plan['configuration'].get('view_offset',0))
    write_json(args.output/'planned_views.json', prepared)
    # Keep the existing validated native writer, masks, freshness gates and output format.
    args.package, args.view_plan = package, None
    args.max_samples = len(job['targets'])*job['max_rendered_views_per_target']
    run_refined_capture(stage, rep, args, manifest, robot, records, reports, variants, job['targets'], prepared_plan=prepared)
    current_hashes = source_hashes(stage)
    require(all(current_hashes.get(p)==h for p,h in manifest['source_usd_sha256'].items()), 'Capture source USD changed')
    require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()), 'Capture dirtied source layers')
    verify_bindings(plan['source_bindings_sha256'])
    require(sha256(args.plan)==manifest['source_collection_plan_sha256'], 'Collection plan changed during capture')
    manifest['source_assets_unchanged'] = True
    lines = ['# Native multi-plant robot-head pilot', '',
        'Full greenhouse, unchanged source components and lighting, mounted RB-Y1 A v1.2 head D405 at 848x408.',
        'Prototype labels only. No human confirmation, training eligibility, arm reach or physical cutting approval.',
        'Depth is native optical-axis Z; colour maps and overlays are review-only. No difficulty labels assigned.', '',
        '| Sample | Target | Numerical clear-view gate |', '|---|---|---|']
    for sample in manifest['samples']:
        lines.append(f"| {sample['sample_id']} | {sample['target_review_id']} | {sample['quality']['clear_view_gate_passed']} |")
    (args.output/'review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-reference-check', action='store_true')
    parser.add_argument('--profile-first-render', action='store_true')
    parser.add_argument('--instance-backend',choices=['legacy','fast','compare'],default='legacy')
    parser.add_argument('--render-budget',choices=['established','single56','single56_compare','noise_probe','warm56_then8'],default='established')
    args = parser.parse_args(argv)
    plan, reports = load_plan(args.plan)
    job = next((j for j in plan['jobs'] if j['job_id']==args.job), None)
    require(job is not None, 'Unknown scheduled job')
    args.output = args.output.resolve()
    require(not args.output.exists() and not args.output.is_relative_to(Path(plan['package'])), 'Choose a new output outside sources')
    args.output.mkdir(parents=True)
    manifest = {'schema_version': 'greenhouse.rgbd_pilot.v1', 'state': 'initializing', 'samples': [],
        'training_dataset_approved': False, 'human_review_performed': False, 'physics_enabled': False,
        'implementation_sha256': {name:sha256(Path(__file__).with_name(name)) for name in
            ('collection_worker.py','collection_plan.py','capture_pilot.py','capture_search.py','capture_scene.py',
            'capture_contract.py','capture_visibility.py','capture_viewpoints.py','viewpoint_plan.py')}}
    if plan.get('schema_version')=='greenhouse.grounding_collection_plan.v1':
        manifest['implementation_sha256'].update({name:sha256(Path(__file__).with_name(name))
                                                 for name in ('training_plan.py','training_views.py','training_screen.py','static_guard.py','static_render.py','native_instances.py')})
    app, code = None, 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 848, 'height': 408, 'multi_gpu': False,
                             'renderer': 'RaytracedLighting', 'sync_loads': False,
                             'disable_viewport_updates':plan.get('schema_version')=='greenhouse.grounding_collection_plan.v1',
                             'extra_args':['--/app/settings/persistent=false']})
        capture_job(app, args, plan, reports, job, manifest)
        manifest['state'] = 'pilot_ready_for_review' if manifest['samples'] else 'blocked_no_screened_viewpoints'
        code = 0 if manifest['samples'] else 3
    except Exception:
        manifest.update(state='failed_do_not_train', error=traceback.format_exc())
        print(manifest['error'], flush=True)
    finally:
        restore=manifest.get('temporary_renderer_switch_restore')
        if restore is not None:
            import carb
            carb.settings.get_settings().set_bool(restore['path'],restore['original'])
            restore['restored_before_exit']=carb.settings.get_settings().get(restore['path'])==restore['original']
        write_json(args.output/'manifest.json', manifest)
        # Kit fast shutdown can terminate the process: completion must be durable BEFORE close.
        print('NATIVE_CAPTURE_RESULT', manifest['state'], str(args.output), flush=True)
        if app is not None:
            app.close(exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
