"""Opt-in two-pose 1/2/4/8/16-subframe accumulation-reset diagnostic. Never admits TRAIN samples."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import importlib.util
import hashlib
import time
import traceback
import numpy as np
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256
from . import native848_direct_plan_v2 as direct
from . import native848_pair_plan_v2 as pair_api
from .native848_pair_audit_v2 import expected_catalogue, world_from_row, verify_camera, image_array
from .native_sensor_payload import validate_native_static, validate_native_payload, decode_native_instances
from .native_render_probe import comparison
from .static_render import LONG_RGB_LIMITS
from .capture_contract import jsonable, fingerprint, project, depth_evidence, write_sample
from .capture_visibility import component_masks, interval_visibility, view_quality, write_visibility
from .capture_sensor import LEGACY_RESOLUTION, sensor_profile, calibration_for_native_resolution
from .native_greenhouse_pair import assert_same_camera
from . import native848_clear_labels_v1 as labels, native848_query_selection_v1 as queries

SCHEMA = 'greenhouse.native848_reset_probe_plan.v2'
SAMPLE = 'greenhouse.native848_reset_probe_sample.v2'
STATE = 'native848_reset_probe_v2_complete_unqualified_diagnostic_only'
CASES = ['original_reviewed_clear', 'generated_known_fruit_occlusion']
BRANCHES = {'reset56_reference': [8]*7, 'reset56_repeat': [8]*7,
            'reset1': [1], 'reset2': [2], 'reset4': [4], 'reset8': [8], 'reset16': [16]}
SETTINGS = ['/omni/replicator/RTSubframes', '/omni/replicator/captureMotionBlur', '/rtx/rendermode', '/rtx/post/aa/op', '/rtx/post/dlss/execMode',
            '/rtx-transient/post/aa/limitedOps', '/rtx/post/dlss/enabled',
            '/rtx/pathtracing/spp', '/rtx/pathtracing/totalSpp']


def checked_subframe_floor(settings,requested):
    raw=settings.get('/omni/replicator/RTSubframes')
    require(raw is None or (type(raw) is int and raw>=0),'Unknown Replicator subframe floor')
    require(settings.get('/omni/replicator/captureMotionBlur') in (None,False),
        'Motion-blur subframes are outside this static diagnostic')
    effective=max(1,int(raw or 1),requested)
    require(effective==requested,'Global Replicator subframe floor overrides requested diagnostic budget')
    return dict(requested_rt_subframes=requested,observed_global_floor=raw,
        effective_rt_subframes_from_installed_policy=effective,
        gpu_sample_count_inferred=False)


def implementation_bindings():
    root=Path(__file__).resolve().parent
    roots=(root.parent,root.parents[1]); pending=[Path(__file__).resolve()]; found={}
    while pending:
        path=pending.pop().resolve()
        if str(path) in found: continue
        found[str(path)]=sha256(path)
        base=next(p for p in roots if path.is_relative_to(p))
        package='.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(path.read_bytes())):
            names=[]
            if isinstance(node,ast.Import): names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                name='.'*node.level+(node.module or '')
                name=importlib.util.resolve_name(name,package) if node.level else name
                names=[name]+[name+'.'+a.name for a in node.names if a.name!='*']
            for name in names:
                for folder in roots:
                    candidate=folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'),candidate/'__init__.py') if p.is_file())
    return dict(sorted(found.items()))


def prepare(direct_path, native_evidence, original_pair_path, clear_admission_path, visual_review_path):
    path = Path(direct_path).resolve()
    cache, plans, _ = direct.check(read_json(path))
    record = next(r for r in cache['records'] if r['sample_id'] == 'SubStem_44_view_002')
    pair = plans[record['pair_plan_path']]
    evidence = Path(native_evidence).resolve()
    plan = dict(schema=SCHEMA, direct_plan_path=str(path), direct_plan_sha256=sha256(path),
        pair_plan_path=record['pair_plan_path'], pair_plan_sha256=record['pair_plan_sha256'],
        original_pair_plan_path=str(Path(original_pair_path).resolve()),original_pair_plan_sha256=sha256(original_pair_path),
        clear_admission_path=str(Path(clear_admission_path).resolve()),clear_admission_sha256=sha256(clear_admission_path),
        clear_visual_review_path=str(Path(visual_review_path).resolve()),clear_visual_review_sha256=sha256(visual_review_path),
        generated_sample_id=record['sample_id'], native_negative_sample_path=str(evidence),
        native_negative_sample_sha256=sha256(evidence), cases=CASES, branches=BRANCHES,
        resolution=[848,408], instance_backend='fast', maximum_poses=2,
        maximum_saved_observations=14, short_branches_require_stable_long_reference=True,
        rgb_limits=dict(LONG_RGB_LIMITS), depth_tolerance_m=.0002,
        reset_api='omni.usd.get_context().reset_renderer_accumulation',
        reset_return_interpretation='returned_without_exception_not_a_convergence_receipt',
        source_cap_reset=False, training_approved=False, accepted_training_increment=0,
        implementation_bindings=implementation_bindings())
    check(plan)
    return plan


def check(plan, full=False):
    require(plan['schema'] == SCHEMA and plan['cases'] == CASES and plan['branches'] == BRANCHES
        and plan['resolution'] == [848,408] and plan['instance_backend'] == 'fast'
        and plan['maximum_poses'] == 2 and plan['maximum_saved_observations'] == 14
        and plan['short_branches_require_stable_long_reference'] is True
        and plan['rgb_limits'] == LONG_RGB_LIMITS and plan['depth_tolerance_m'] == .0002
        and plan['reset_api'] == 'omni.usd.get_context().reset_renderer_accumulation'
        and plan['reset_return_interpretation'] == 'returned_without_exception_not_a_convergence_receipt'
        and plan['source_cap_reset'] is False and plan['training_approved'] is False
        and plan['accepted_training_increment'] == 0, 'Changed reset diagnostic scope')
    require(plan['implementation_bindings'] == implementation_bindings(), 'Changed diagnostic implementation')
    verify_bindings(plan['implementation_bindings'])
    require(sha256(plan['direct_plan_path']) == plan['direct_plan_sha256'], 'Changed source direct plan')
    cache, plans, generated = direct.check(read_json(plan['direct_plan_path']), full=full)
    record = next(r for r in cache['records'] if r['sample_id'] == plan['generated_sample_id'])
    require(record['pair_plan_path'] == plan['pair_plan_path']
        and record['pair_plan_sha256'] == plan['pair_plan_sha256'], 'Changed cached negative pose')
    pair = plans[record['pair_plan_path']]
    require(sha256(plan['original_pair_plan_path'])==plan['original_pair_plan_sha256'], 'Changed clear original plan')
    original_pair=read_json(plan['original_pair_plan_path']); pair_api.check(original_pair,full=full)
    require(sha256(plan['clear_admission_path'])==plan['clear_admission_sha256'], 'Changed fresh clear admission')
    admission=read_json(plan['clear_admission_path']); verify_bindings(admission['source_bindings'])
    clear=next(r for r in admission['records'] if r['mode']=='original_control')
    require(clear['local_clarity_passed'] is True and clear['full_trace_and_anchored_grid_passed'] is True
        and clear['workspace_passed'] is True and clear['source_family']==original_pair['source_family'], 'Clear original source role unsupported')
    for path,pin in [(clear['source_sample_path'],clear['source_sample_sha256']),
        (clear['rgb_path'],clear['rgb_sha256']),(clear['label_path'],clear['label_sha256']),
        (Path(clear['label_path']).parent/'query_trace.json',clear['query_trace_sha256'])]:
        require(sha256(path)==pin, 'Changed actual clear source evidence')
    clear_meta=read_json(clear['source_sample_path'])
    assert_same_camera(clear_meta['calibration'],original_pair['expected_calibration'])
    require(clear_meta['robot_snapshot']==original_pair['expected_robot_snapshot']
        and clear_meta['supervision']['target_id']==original_pair['source_row']['target_id'], 'Clear source embodiment/target differs')
    require(sha256(plan['clear_visual_review_path'])==plan['clear_visual_review_sha256'], 'Changed clear visual review')
    review=read_json(plan['clear_visual_review_path'])
    require(review['rgb_sha256']==clear['rgb_sha256'] and review['decision']=='clear_for_render_diagnostic'
        and review['training_approved'] is False, 'Fresh original visual review required')
    require(sha256(plan['native_negative_sample_path']) == plan['native_negative_sample_sha256'], 'Changed native negative evidence')
    negative = read_json(plan['native_negative_sample_path'])
    require(negative['schema_version'] == direct.SAMPLE_SCHEMA
        and negative['sample_id'] == 'SubStem_44_view_002'
        and negative['supervision']['target_id'] == pair['generated_row']['target_id'], 'Wrong observed negative')
    assert_same_camera(negative['calibration'],record['calibration'])
    negative_folder = Path(plan['native_negative_sample_path']).parent
    for name, info in negative['files'].items():
        require(sha256(negative_folder/name) == info['sha256'], 'Changed native negative buffer')
    ids=np.load(negative_folder/'supervision/renderer_instance_id.npy',allow_pickle=False)
    mapping={int(k):v for k,v in read_json(negative_folder/'supervision/identities.json')['renderer_id_to_prim'].items()}
    xy=np.floor([p['pixel_xy'] for p in negative['supervision']['projected_interval']]).astype(int)
    require(any('/Fruit_07' in mapping.get(int(ids[y,x]),'') for x,y in xy),
        'Saved negative no longer demonstrates Fruit_07 interval occlusion')
    return cache, pair, record, generated


def annotations(meta, report, rgb, depth, valid, components, catalogue, mask):
    baseline = labels.derive(meta, report, rgb, depth, valid, components, catalogue)
    label, trace, selection = queries.annotate_v1(meta, report, rgb, depth, valid,
        components, catalogue, target_mask=mask.astype(np.uint8)*255, expected_label=baseline)
    # Same sample ID in all branches preserves the deterministic query seed.
    decisions = dict(eligible=label['eligible'], reason=label['reason'], answer=label.get('answer'),
        selected_query_uv=label.get('query_pixel_uv'), trace_passed=None if trace is None else trace['passed'],
        candidates=[dict(index=r['candidate_index'], local=r['local_passed'], both=r['both_passed'],
            rejections=r['rejections']) for r in selection['candidates']])
    return dict(baseline_label=baseline, label=label, query_trace=trace,
        query_selection=selection, decisions=decisions)


def interval_probes(interval, depth, valid, ids, mapping):
    rows = []
    for p in interval:
        if p['projection_status'] != 'in_frame':
            rows.append(dict(projection=p, in_frame=False)); continue
        x,y = np.floor(p['pixel_xy']).astype(int)
        identifier = int(ids[y,x])
        rows.append(dict(projection=p, in_frame=True, pixel_xy=[int(x),int(y)],
            instance_id=identifier, prim_path=mapping.get(identifier), valid=bool(valid[y,x]),
            native_z_m=float(depth[y,x]) if valid[y,x] else None))
    return rows


def load_payload(folder):
    native = read_json(folder/'native_payload.json')
    native['rgb'] = np.load(folder/'native_rgba.npy', allow_pickle=False)
    native['distance_to_image_plane'] = np.load(folder/'inputs/depth_m.npy', allow_pickle=False)
    native['instance_id_segmentation'] = dict(
        data=np.load(folder/'supervision/renderer_instance_id.npy', allow_pickle=False),
        info=read_json(folder/'native_instance_info.json'))
    return native


def compare_saved(left, right):
    lm, rm = read_json(left/'sample.json'), read_json(right/'sample.json')
    require(lm['calibration'] == rm['calibration'] and lm['sample_id'] == rm['sample_id'], 'Comparison pose/query seed changed')
    report = comparison(load_payload(left), load_payload(right), lm['calibration'],
        roi_pixel=lm['supervision']['nominal_projected']['pixel_xy'])
    report['query_and_trace_decisions_equal'] = (read_json(left/'annotation.json')['decisions']
        == read_json(right/'annotation.json')['decisions'])
    report['passed'] = report['passed'] and report['query_and_trace_decisions_equal']
    return report


def _capture_one(app, output, plan, pair, record, selected_case):
    import omni.usd
    import omni.replicator.core as rep
    from .native_scene import prepare_native_scene
    from .native848_direct_worker_v2 import apply_cached_pose
    from .generated_capture import substitute_plant
    from .capture_pilot import calibration_smoke, make_writer, step_payload, source_hashes
    from .capture_scene import calibration, target_world_geometry
    from .capture_viewpoints import component_catalogue, static_obstacles, scene_triangle_refiner, visible_bounds
    from .training_screen import StaticBoundScreen
    from .static_guard import StaticSceneMonitor
    from .review_camera import HEAD_CAMERA
    scene = prepare_native_scene(app, pair)
    stage, robot = scene['stage'], scene['robot']
    original_hashes = source_hashes(stage)
    reports_by_family = {r['plant_id']:r for r in scene['reports']}
    product = rep.create.render_product(HEAD_CAMERA, LEGACY_RESOLUTION)
    writer = make_writer(rep, include_instances=True, instance_backend='fast')
    writer.attach([product])
    cases = []
    try:
        for case_id in [selected_case]:
            is_original = case_id == CASES[0]
            mode = 'original_control' if is_original else 'generated_variant'
            row = pair['source_row' if is_original else 'generated_row']
            if is_original:
                records, reports, variants = scene['records'], scene['reports'], scene['variants']
                pose = scene['pose']; cal = calibration_for_native_resolution(calibration(stage), LEGACY_RESOLUTION)
                assert_same_camera(cal, pair['expected_calibration'])
                report = reports_by_family[pair['source_family']]
                sample_id = pair['source_sample']
            else:
                changed = substitute_plant(stage, scene['original_variant'], scene['records'], scene['variants'], scene['generated'])
                records, reports, variants = changed['records'], [*scene['reports'],changed['report']], changed['variants']
                pose, cal = apply_cached_pose(stage, robot, record)
                report = scene['generated']['report']; sample_id = record['sample_id']
            catalogue = component_catalogue(stage, records, reports, variants)
            require(catalogue == expected_catalogue(pair, mode, reports_by_family, scene['generated'])
                and len(catalogue) == scene['counts']['components'], 'Full scene catalogue changed')
            variant = next(v for v in variants if v['variant_id'] == row['variant_id'])
            world = target_world_geometry(stage, variant, row)
            require(all(np.allclose(world[k],world_from_row(pair,row)[k],atol=1e-9,rtol=0) for k in world), 'Target world geometry changed')
            target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
            radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
            nominal = project([world['nominal_world_m']],cal)[0]; interval = project(world['interval_world_m'],cal)
            case = output/case_id; case.mkdir()
            screen = StaticBoundScreen(static_obstacles(stage,robot['root']),
                scene_triangle_refiner(stage,include_generated_plants=True))(visible_bounds(stage,robot['root']))
            write_json(case/'geometry_screen.json',screen)
            require(screen['passed'], 'Current snapshot intersects robot/environment geometry')
            phase_hashes=source_hashes(stage)
            # Flush authored pose/product before the reset, never between reset and first request.
            for _ in range(2): app.update()
            monitor = StaticSceneMonitor(stage,robot['root'])
            branches = []
            try:
                for branch, steps in BRANCHES.items():
                    if branch == 'reset1':
                        stable = compare_saved(case/'reset56_reference',case/'reset56_repeat')
                        write_json(case/'reference_stability.json',stable)
                        if not stable['passed']:
                            break
                    before = monitor.begin()
                    seq, requests = writer.sequence, writer.request_index
                    settings = {k:jsonable(scene['settings'].get(k)) for k in SETTINGS}
                    require(settings['/rtx/rendermode'] == scene['old_manifest']['renderer'], 'Renderer changed')
                    effective_budgets=[checked_subframe_floor(settings,n) for n in steps]
                    reset_start = time.perf_counter()
                    reset_return = omni.usd.get_context().reset_renderer_accumulation()
                    reset = dict(api=plan['reset_api'], returned_without_exception=True,
                        returned_value=jsonable(reset_return), elapsed_seconds=time.perf_counter()-reset_start,
                        convergence_or_temporal_cleanliness_verified=False)
                    started = time.perf_counter()
                    request_evidence = []
                    for subframes in steps:
                        before_request, before_callback = writer.request_index, writer.sequence
                        payload = step_payload(rep,writer,subframes=subframes)
                        validate_native_payload(payload,cal)
                        request_evidence.append(dict(subframes=subframes, native_requests=writer.request_index-before_request,
                            callback_sequence_before=before_callback,callback_sequence_after=writer.sequence))
                    render_seconds = time.perf_counter()-started
                    rgb,depth,valid,reference,fresh = validate_native_static(payload,cal,before,monitor.token(),writer.sequence)
                    require(writer.sequence > seq, 'No fresh branch callback')
                    assert_same_camera(calibration_for_native_resolution(calibration(stage),LEGACY_RESOLUTION),cal)
                    require(settings == {k:jsonable(scene['settings'].get(k)) for k in SETTINGS}, 'Renderer settings changed during branch')
                    ids,mapping = decode_native_instances(payload,LEGACY_RESOLUTION)
                    old_root=scene['original_variant']['plant_root']
                    old_ids=[i for i,path in mapping.items() if path==old_root or path.startswith(old_root+'/')]
                    old_pixels=int(np.isin(ids,old_ids).sum())
                    require(is_original or old_pixels==0, 'Original foreground identity remains after substitution')
                    components,organs,owners = component_masks(ids,mapping,catalogue)
                    visibility,mask = interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,target,radius)
                    meta = dict(schema_version=SAMPLE,sample_id=sample_id,case_id=case_id,branch=branch,
                        training_sample_approved=False,sensor=sensor_profile(LEGACY_RESOLUTION),calibration=cal,
                        robot_snapshot=pose,scene_counts=scene['counts'],lighting=scene['old_manifest']['lighting'],
                        renderer=scene['old_manifest']['renderer'],render_settings=settings,geometry_screen=screen,
                        rendered_camera_params=jsonable(payload['camera_params']),
                        input_policy=dict(clean_full_scene=True,isolation=False,diagnostic_overlays=False),
                        quality=view_quality(cal,nominal,interval,radius,visibility,rgb,mask),
                        native_instance_backend='fast',native_instance_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                        native_mapping_sha256=fingerprint(mapping),native_target_pixels=int(mask.sum()),
                        old_plant_native_pixels=old_pixels,
                        synchronization=dict(method='frozen_scene_single_native_writer_payload',
                            scene_unchanged_during_capture=True,dynamic_recording_supported=False,
                            reference_time=reference,freshness=fresh,static_guard=before,
                            native_render_frame=jsonable(payload['pilot_render_frame']),engine_frame_id_verified=False,
                            render_budget_subframes=sum(steps),reset=reset,request_evidence=request_evidence,
                            actual_native_requests=writer.request_index-requests,effective_subframe_policy=effective_budgets,
                            render_seconds=render_seconds),
                        supervision=dict(**world,target_id=row['target_id'],source_target_id=pair['source_row']['target_id'],
                            split_group=pair['split_group'],conservative_view_cap_group=pair['conservative_view_cap_group'],
                            cut_region_proposal=row['cut_region_proposal'],nominal_projected=nominal,projected_interval=interval,
                            depth_evidence=depth_evidence(nominal,depth,valid,radius),visibility_evidence=visibility,
                            cut_safety_validated=False))
                    folder = case/branch
                    write_sample(folder,rgb,depth,valid,meta)
                    write_visibility(folder,ids,mapping,catalogue,components,organs,mask)
                    np.save(folder/'native_rgba.npy',np.asarray(payload['rgb']),allow_pickle=False)
                    native = {k:jsonable(v) for k,v in payload.items() if k in
                        ('camera_params','ReferenceTime','reference_time','pilot_render_frame') or k.startswith('rp_')}
                    write_json(folder/'native_payload.json',native)
                    write_json(folder/'native_instance_info.json',jsonable(payload['instance_id_segmentation']['info']))
                    saved_meta = read_json(folder/'sample.json')
                    annotation = annotations(saved_meta,report,rgb,depth,valid,components,catalogue,mask)
                    write_json(folder/'annotation.json',annotation)
                    write_json(folder/'interval_probes.json',interval_probes(interval,depth,valid,ids,mapping))
                    write_json(folder/'bindings.json',{str(p.resolve()):sha256(p) for p in folder.rglob('*') if p.is_file()})
                    require(all(r['native_requests'] == 1 for r in request_evidence),
                        'Requested branch exceeded budget due to callback retry; evidence preserved')
                    branches.append(branch)
                    print('RESET848_BRANCH',case_id,branch,round(render_seconds,3),flush=True)
            finally:
                monitor.close()
            checks = {'reference_repeat':compare_saved(case/'reset56_reference',case/'reset56_repeat')}
            for name in branches[2:]:
                checks[name+'_reference'] = compare_saved(case/name,case/'reset56_reference')
                checks[name+'_repeat'] = compare_saved(case/name,case/'reset56_repeat')
            result = dict(case_id=case_id,branches=branches,comparisons=checks,
                reference_stable=checks['reference_repeat']['passed'],training_approved=False,
                accepted_training_increment=0,production_profile_qualified=False)
            write_json(case/'result.json',result); cases.append(result)
            require(source_hashes(stage)==phase_hashes, 'Source asset changed during pose experiment')
    finally:
        writer.detach(); product.destroy()
    verify_bindings(original_hashes)
    require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()), 'Source layer dirtied')
    check(plan)
    return dict(state=STATE,cases=cases,resolution=[848,408],source_assets_unchanged=True,
        source_cap_reset=False,training_approved=False,accepted_training_increment=0,
        production_profile_qualified=False,workspace_approval_inherited=False)


def capture(app,output,plan,pair,record):
    import omni.replicator.core as rep
    from .capture_pilot import calibration_smoke
    smoke=calibration_smoke(app,rep,include_instances=True,instance_backend='fast')
    require(smoke['passed'] and smoke['native_identity_and_occluder_smoke_passed'],'Sensor smoke failed')
    write_json(output/'calibration_smoke.json',smoke)
    cases=[]
    for case_id in CASES:
        active_pair=read_json(plan['original_pair_plan_path']) if case_id==CASES[0] else pair
        result=_capture_one(app,output,plan,active_pair,record,case_id)
        cases.extend(result['cases'])
    return dict(state=STATE,cases=cases,resolution=[848,408],source_assets_unchanged=True,
        full_greenhouse_stage_count=2,source_cap_reset=False,training_approved=False,
        accepted_training_increment=0,production_profile_qualified=False,workspace_approval_inherited=False)


def audit(output, plan):
    """CPU replay of every saved native buffer, identity, label and comparison."""
    from .collection_plan import load_plan
    _,direct_pair,record,direct_generated = check(plan,full=True)
    result = read_json(output/'result.json')
    request=read_json(output/'request.json')
    require(sha256(request['plan_path'])==request['plan_sha256']==result['plan_sha256']
        and read_json(request['plan_path'])==plan, 'Changed executed plan binding')
    require(result['state'] == STATE and [r['case_id'] for r in result['cases']] == CASES,
        'Incomplete reset experiment')
    verified = []
    for case_result in result['cases']:
        case_id=case_result['case_id']; case=output/case_id; original=case_id==CASES[0]
        pair=read_json(plan['original_pair_plan_path']) if original else direct_pair
        generated=pair_api.check(pair,full=True) if original else direct_generated
        _,reports=load_plan(pair['source_collection_plan']);reports={r['plant_id']:r for r in reports}
        source_manifest=read_json(Path(pair['source_capture'])/'manifest.json')
        mode='original_control' if original else 'generated_variant'
        row=pair['source_row' if original else 'generated_row']
        report=reports[pair['source_family']] if original else generated['report']
        expected_cal=pair['expected_calibration'] if original else record['calibration']
        names=case_result['branches']
        require(names == list(BRANCHES)[:len(names)] and len(names) in (2,7), 'Invalid branch execution order')
        for name in names:
            folder=case/name; verify_bindings(read_json(folder/'bindings.json'))
            meta=read_json(folder/'sample.json'); payload=load_payload(folder)
            require(meta['schema_version']==SAMPLE and meta['case_id']==case_id and meta['branch']==name
                and meta['scene_counts']==pair['expected_scene_counts']
                and meta['supervision']['target_id']==row['target_id']
                and meta['lighting']==source_manifest['lighting'] and meta['renderer']==source_manifest['renderer']
                and meta['input_policy']==dict(clean_full_scene=True,isolation=False,diagnostic_overlays=False),
                'Saved scene/camera/target changed')
            if original:
                require(meta['robot_snapshot']==pair['expected_robot_snapshot'], 'Original embodiment changed')
            else:
                require(meta['robot_snapshot']['joint_degrees']==record['joint_degrees']
                    and np.allclose(meta['robot_snapshot']['robot_root_to_world_usd_row_vectors'],
                        record['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0), 'Cached embodiment changed')
            assert_same_camera(meta['calibration'],expected_cal)
            cal=meta['calibration']
            verify_camera(meta)
            rgb,depth,valid,_=validate_native_payload(payload,cal)
            require(np.array_equal(rgb,image_array(folder/'inputs/rgb.png'))
                and np.array_equal(valid,image_array(folder/'inputs/depth_valid.png')==255), 'Saved native RGB/validity differ')
            sync=meta['synchronization']; reset=sync['reset']
            require(reset['api']==plan['reset_api'] and reset['returned_without_exception'] is True
                and reset['convergence_or_temporal_cleanliness_verified'] is False
                and [r['subframes'] for r in sync['request_evidence']]==BRANCHES[name]
                and all(r['native_requests']==1 for r in sync['request_evidence'])
                and sync['actual_native_requests']==len(BRANCHES[name]), 'Reset/request evidence changed')
            require(sync['effective_subframe_policy']==[checked_subframe_floor(meta['render_settings'],n) for n in BRANCHES[name]],
                'Effective subframe floor evidence changed')
            ids,mapping=decode_native_instances(payload,LEGACY_RESOLUTION)
            require(meta['native_instance_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
                and meta['native_mapping_sha256']==fingerprint(mapping), 'Native identity fingerprints changed')
            oldroot=pair['original_variant']['plant_root']
            oldids=[i for i,path in mapping.items() if path==oldroot or path.startswith(oldroot+'/')]
            oldpixels=int(np.isin(ids,oldids).sum())
            require(meta['old_plant_native_pixels']==oldpixels and (original or oldpixels==0), 'Old foreground pixels differ')
            catalogue=read_json(folder/'supervision/identities.json')['component_catalogue']
            require(catalogue==expected_catalogue(pair,mode,reports,generated), 'Changed saved full scene catalogue')
            components,organs,owners=component_masks(ids,mapping,catalogue)
            world=world_from_row(pair,row)
            require(all(np.allclose(meta['supervision'][k],v,atol=1e-9,rtol=0) for k,v in world.items()), 'Saved world geometry differs')
            nominal=project([world['nominal_world_m']],cal)[0]; interval=project(world['interval_world_m'],cal)
            target=next(c for c in catalogue if c['variant_id']==row['variant_id'] and c['component_id']==row['component_id'])
            radius=row['cut_region_proposal']['nominal']['petiole_radius_m']
            vis,mask=interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,target,radius)
            require(vis==meta['supervision']['visibility_evidence']
                and np.array_equal(components,np.load(folder/'supervision/component_id.npy',allow_pickle=False))
                and np.array_equal(mask,image_array(folder/'supervision/target_visible.png')==255), 'Native derived visibility differs')
            require(annotations(meta,report,rgb,depth,valid,components,catalogue,mask)==read_json(folder/'annotation.json')
                and interval_probes(interval,depth,valid,ids,mapping)==read_json(folder/'interval_probes.json'), 'Native trace/probe replay differs')
            verified.append(str(folder))
        actual={'reference_repeat':compare_saved(case/names[0],case/names[1])}
        for name in names[2:]:
            actual[name+'_reference']=compare_saved(case/name,case/names[0])
            actual[name+'_repeat']=compare_saved(case/name,case/names[1])
        require(actual==case_result['comparisons'] and (len(names)==7)==actual['reference_repeat']['passed'],
            'Comparison replay or short-branch gate differs')
    return dict(state='native848_reset_saved_buffers_independently_replayed',verified_branches=verified,
        production_profile_qualified=False,training_approved=False,accepted_training_increment=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True); parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    require(sha256(args.plan)==args.plan_sha256,'Changed reset plan')
    plan=read_json(args.plan); _,pair,record,_=check(plan)
    output=args.output.resolve()
    require(not output.exists(), 'Create-only reset output required')
    original_pair=read_json(plan['original_pair_plan_path'])
    protected=[Path(pair['source_capture']).resolve(),Path(pair['variant_directory']).resolve(),args.plan.parent.resolve(),
        Path(original_pair['source_capture']).resolve(),Path(original_pair['variant_directory']).resolve()]
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),'Output overlaps source')
    from .native_generated_pair import windows_worker_admission
    from sim_physics.host_memory import preflight
    import shutil
    memory=preflight(); disk=shutil.disk_usage(output.parent).free
    require(memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30 and disk>=60*2**30,'Native resource reserves unavailable')
    process=windows_worker_admission(None)
    output.mkdir(parents=True)
    write_json(output/'request.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(args.plan.resolve()),plan_sha256=args.plan_sha256,host_memory_preflight=memory,
        available_disk_bytes=disk,process_admission=process,training_approved=False))
    app=None; success=False
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RaytracedLighting',
            sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false']))
        check(plan,full=True)
        result=capture(app,output,plan,pair,record)
        result['plan_sha256']=args.plan_sha256
        require(sha256(args.plan)==args.plan_sha256,'Plan changed during capture')
        write_json(output/'result.json',result); success=True
    except BaseException:
        write_json(output/'failure.json',dict(error=traceback.format_exc(),training_approved=False)); raise
    finally:
        if app is not None: app.close(exit_code=0 if success else 1)


if __name__=='__main__': main()
