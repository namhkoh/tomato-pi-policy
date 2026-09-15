"""Full-greenhouse scene seam with actual CLEAR context, never a fake old_manifest.

This module does not create a SimulationApp, render product, writer or capture.
Main must provide an explicitly authorized/bound producer and post-exit audit
before invoking these scene hooks inside Kit. No source layer is edited.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from ..native_original_capture.contracts import require, digest, canonical
from ..native_original_capture.scene import restore_pose
from ..native_greenhouse_pair import assert_same_camera


def current_scene_evidence(plan, *, lighting, counts, renderer):
    """Metadata comes from actual scene construction, checked against fresh proof."""
    authority = plan['scene_authority']
    require(lighting == authority['actual_lighting']
        and counts == authority['expected_scene_counts']
        and renderer == authority['actual_renderer'] == 'RealTimePathTracing',
        'Actual clear scene differs from original1696 reference')
    return dict(lighting=deepcopy(lighting), scene_counts=deepcopy(counts), renderer=renderer,
        clear_plan=deepcopy(authority['clear_plan']), policy=deepcopy(authority['policy']),
        renderer_settings_status='mode_checked_full_settings_not_previously_recorded',
        historical_lighting_reproduced=False, source_cap_reset=False, training_approved=False)


def prepare_scene(app, plan):
    # Only called AFTER the owner has initialized Kit; no pip USD import before it.
    import sys
    import carb
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom
    from launch_sim_data import load_local_payloads, populate
    from .prepare import check_plan
    from ..native_original_capture import contracts as oc
    from ..capture_scene import (capture_root, calibration, freeze_rigid_bodies,
                                 target_world_geometry, mounted_camera_to_head)
    from ..capture_pilot import source_hashes
    from ..capture_sensor import calibration_for_native_resolution
    from ..collection_plan import load_plan
    from ..floor_alignment import PACKAGE_FLOOR
    from ..robot_preview import add_robot_preview

    started = time.perf_counter()
    generated = check_plan(plan)
    authority = plan['scene_authority']
    original_plan, reports = load_plan(authority['clear_plan']['path'])
    rows = [r for j in original_plan['jobs'] for r in j['targets'] if r['target_id'] == plan['source_row']['target_id']]
    require(rows == [plan['source_row']], 'Current clear source row changed')
    package = Path(authority['package'])
    require(Path(original_plan['package']).resolve() == package.resolve(), 'Current scene package changed')
    wrapper = capture_root(package/'house/green_house_base.usd')
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = list(wrapper.subLayerPaths)
    require(stage.GetRootLayer().anonymous and stage.GetPrimAtPath('/World/Gutters'), 'Full anonymous greenhouse required')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = load_local_payloads(stage)
    records = []
    gutter_x, counts = populate(stage, package, app, records, review_plant=plan['source_family'])
    require(excluded == authority['excluded_external_roots'], 'External-prop scope changed')
    variants = deepcopy(authority['scene_variants'])
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool='gripper')
    freeze_rigid_bodies(stage)
    sys.path.insert(0, str(package/'env_panel'))
    from tomato_env import daylight
    require(Path(daylight.__file__).resolve() == package/'env_panel/tomato_env/daylight.py',
            'Daylight import escaped pinned package')
    lighting = daylight.apply(stage, day=172, minutes=13*60, intensity=1500, dome_intensity=6000)
    settings = carb.settings.get_settings()
    settings.set('/rtx/rendermode', 'RealTimePathTracing')
    app.update()
    scene_evidence = current_scene_evidence(plan, lighting=lighting, counts=counts,
                                             renderer=settings.get('/rtx/rendermode'))
    rep.set_global_seed(original_plan['configuration']['seed'])
    pose = plan['expected_robot_snapshot']
    restore_pose(stage, robot, pose)
    prior_cal = plan['pose_prior']['calibration']
    assert_same_camera(calibration(stage), prior_cal)
    require(np.allclose(mounted_camera_to_head(stage), pose['camera_to_head_column_vectors'], atol=1e-9, rtol=0),
            'Mounted camera transform changed')
    native_cal = calibration_for_native_resolution(calibration(stage), oc.RESOLUTION)
    assert_same_camera(native_cal, plan['expected_calibration'])
    original_variant = next(v for v in variants if v['variant_id'] == plan['source_family'])
    require(original_variant == authority['original_variant'], 'Original foreground placement changed')
    original_world = target_world_geometry(stage, original_variant, plan['source_row'])
    for key, value in original_world.items():
        require(np.allclose(value, plan['expected_original_world'][key], atol=1e-9, rtol=0),
                'Fresh original1696 physical target changed: ' + key)
    manifest_pin = authority['source_asset_manifest']
    manifest = oc.read_json(oc.pin(manifest_pin['path'], manifest_pin['sha256']))
    loaded_hashes = source_hashes(stage)
    require(loaded_hashes == manifest['source_usd_sha256'], 'Loaded source asset set changed')
    usd_context.get_selection().clear_selected_prim_paths()
    omni.timeline.get_timeline_interface().pause()
    return dict(stage=stage, records=records, reports=reports, variants=variants, generated=generated,
        robot=robot, original_variant=original_variant, original_world=original_world,
        calibration=native_cal, pose=deepcopy(pose), settings=settings, counts=counts,
        scene_evidence=scene_evidence, source_hashes=loaded_hashes,
        prepared_plan_sha256=digest(canonical(plan)),
        setup_seconds=time.perf_counter()-started, native_capture_performed=False)


def substitute_generated(context, plan):
    """Session-only existing substitution, preserving full population/placement."""
    require(context['prepared_plan_sha256'] == digest(canonical(plan)), 'Context belongs to another prepared plan')
    from ..generated_capture import substitute_plant
    from ..capture_viewpoints import component_catalogue
    replacement = substitute_plant(context['stage'], context['original_variant'],
        context['records'], context['variants'], context['generated'])
    reports = [*context['reports'], replacement['report']]
    catalogue = component_catalogue(context['stage'], replacement['records'], reports, replacement['variants'])
    require(len(catalogue) == context['counts']['components'], 'Generated substitution changed full population')
    require(np.allclose(replacement['plant_to_world'],
        plan['expected_original_world']['plant_to_world_usd_row_vectors'], atol=1e-9, rtol=0),
        'Generated substitution changed original world placement')
    return dict(replacement=replacement, reports=reports, catalogue=catalogue,
                scene_evidence=deepcopy(context['scene_evidence']), native_capture_performed=False)
