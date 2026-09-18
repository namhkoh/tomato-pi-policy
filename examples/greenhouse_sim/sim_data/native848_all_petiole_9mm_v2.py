"""CPU candidate labels for ALL petioles; clean full-frame inputs, no query.

This is a separate annotation epoch, not an acceptance/export tool. Inputs must
come from an independently authenticated native frame and complete scene census.
The 9--19 mm arc is visibility support only: the output is exactly 9 mm.
"""
from copy import deepcopy
import numpy as np
from scipy.ndimage import distance_transform_edt
from .capture_contract import transform_points, project, depth_evidence
from .cut_regions import _oriented_chain, _sample
from .dataset_review import require
from . import native848_joint_ownership_v1 as joint_ownership

ANNOTATION_EPOCH = 'greenhouse.native848_all_petiole_9mm.v2'
NOMINAL_ARC_M = .009
SUPPORT_ARC_M = (.009, .019)
# Numerical local gates retained from native848_clear_labels_v1.clear_screen.
POLICY = dict(minimum_support_px=12., minimum_width_proxy_px=8.,
              minimum_local_median_luma=40., maximum_local_dark_fraction=.10,
              minimum_proximal_visible_fraction=.95,
              minimum_visible_parent_pixels=6, minimum_cut_to_parent_distance_px=3.)


def geometry_9mm(report, component_id, matrix, calibration):
    component = report['components'][component_id]
    require(component.get('parent') in report['components'], 'Missing anatomical parent')
    require(report['components'][component['parent']]['type'] == 'main_stem',
            'not_direct_main_stem_petiole')
    matrix = np.asarray(matrix, float)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
            and np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-7, rtol=0)
            and abs(np.linalg.det(matrix[:3, :3])-1.) < 1e-7
            and np.allclose(matrix[:, 3], [0,0,0,1], atol=1e-9, rtol=0),
            'Rigid metre-scale plant transform required')
    chain, lengths, reverse, endpoint_error = _oriented_chain(component, 1e-6)
    require(lengths[-1] >= .030, 'Petiole too short for unchanged proximal evidence')
    origin = component['translation_plant_m']

    def at(distance):
        sample = _sample(chain, lengths, float(distance), origin)
        world = transform_points([sample['point_plant_m']], matrix)[0]
        return dict(arc_m=float(distance), point_plant_m=sample['point_plant_m'],
                    world_m=world.tolist(), radius_m=sample['petiole_radius_m'],
                    projected=project([world], calibration)[0])

    # Retain every anatomical bend and never exceed 1 mm between probes.
    knots = [.009] + [float(x) for x in lengths if .009 < x < .019] + [.019]
    distances = []
    for low, high in zip(knots, knots[1:]):
        count = max(1, int(np.ceil((high-low)/.001)))
        distances.extend(float(x) for x in np.linspace(low, high, count, endpoint=False))
    distances.append(.019)
    return dict(nominal=at(.009), legacy_10mm_comparison=at(.010),
                oriented_centerline_world_m=transform_points(
                    [np.asarray(origin)+np.asarray(p[:3]) for p in chain], matrix).tolist(),
                centerline_arc_distances_m=[float(x) for x in lengths],
                visibility_support=[at(x) for x in distances],
                proximal=[at(x) for x in np.linspace(.004, .030, 27)],
                junction_to_cut=[at(x) for x in np.linspace(0., .009, 19)],
                attachment_projected=project(transform_points([component['attachment_plant_m']], matrix), calibration)[0],
                centerline_total_length_m=float(lengths[-1]), source_chain_reversed=bool(reverse),
                attachment_endpoint_error_m=float(endpoint_error))


def local_screen(rgb, mask, support_uv, proximal_fraction):
    line = np.asarray(support_uv, float)
    require(line.ndim == 2 and line.shape[1] == 2 and len(line) >= 2
            and np.isfinite(line).all() and ((line >= 0) & (line < [848, 408])).all(),
            'Invalid native visibility support')
    xy = np.floor(line).astype(int)
    x, y = xy.T
    x0, y0 = np.maximum(xy.min(axis=0)-8, 0)
    x1, y1 = np.minimum(xy.max(axis=0)+9, [848, 408])
    pixels = rgb[y0:y1, x0:x1][mask[y0:y1, x0:x1]].astype(float)
    luma = pixels @ np.asarray([.2126, .7152, .0722]) if len(pixels) else np.asarray([0.])
    metrics = dict(support_length_px=float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()),
                   width_proxy_px=float(np.median(2*distance_transform_edt(mask)[y, x])),
                   local_median_luma=float(np.median(luma)), local_dark_fraction=float(np.mean(luma < 30)),
                   all_support_samples_on_target=bool(mask[y, x].all()),
                   proximal_visible_fraction=float(proximal_fraction))
    reasons = []
    for key, bound in [('support_length_px', 'minimum_support_px'), ('width_proxy_px', 'minimum_width_proxy_px'),
                       ('local_median_luma', 'minimum_local_median_luma'),
                       ('proximal_visible_fraction', 'minimum_proximal_visible_fraction')]:
        if not np.isfinite(metrics[key]) or metrics[key] < POLICY[bound]:
            reasons.append(key)
    if metrics['local_dark_fraction'] > POLICY['maximum_local_dark_fraction']:
        reasons.append('local_dark_fraction')
    if not metrics['all_support_samples_on_target']:
        reasons.append('support_not_fully_visible')
    return dict(metrics, passed=not reasons, reasons=reasons)


def evaluate_target(metadata, rgb, depth, valid, components, catalogue, entry, workspace_checker):
    target_id = entry['target_id']
    variant, component_id = target_id.split('/')
    report = entry['report']
    require(report['plant_id'] == entry['source_family'], 'Report/source family differs')
    target_matches = [c for c in catalogue if c['variant_id'] == variant and c['component_id'] == component_id]
    require(len(target_matches) == 1 and target_matches[0]['organ_type'] == 'sub_stem', 'Exact petiole catalogue identity required')
    target = target_matches[0]
    base = dict(target_id=target_id, source_family=entry['source_family'],
                status='unknown', reason=None, annotation_epoch=ANNOTATION_EPOCH,
                nominal_arc_m=NOMINAL_ARC_M, visibility_support_arc_m=list(SUPPORT_ARC_M),
                support_is_permissible_cut_interval=False, actual_visual_review=False,
                automated_pass=False, training_approved=False)
    if not np.any(components == target['component_index']):
        return dict(base, status='excluded', reason='not_visible_zero_authenticated_component_pixels')
    if entry.get('semantic_leaf_petiole_candidate') is False:
        return dict(base, status='excluded', reason='not_anatomically_eligible_leaf_petiole',
                    anatomy_reason_codes=entry.get('anatomy_reason_codes', []))
    anatomy_targets = [t for t in report.get('targets', []) if t['component_id'] == component_id]
    if anatomy_targets:
        require(len(anatomy_targets) == 1, 'Ambiguous anatomical target')
        anatomy = anatomy_targets[0]
        leaves = [key for key in anatomy['expected_detached_component_ids']
                  if report['components'][key]['type'] == 'leaf']
        if anatomy['protected_descendant_ids'] or not leaves:
            return dict(base, status='excluded', reason='protected_descendant_or_no_leaf_petiole')
    elif entry.get('semantic_leaf_petiole_candidate') is not True:
        return dict(base, reason='missing_anatomical_leaf_petiole_classification')
    component = report['components'][component_id]
    if component.get('type') != 'sub_stem' or component.get('deleafed') is not False:
        return dict(base, status='excluded', reason='not_intact_petiole')
    if component.get('parent') not in report['components']:
        return dict(base, reason='missing_parent_anatomy')
    if report['components'][component['parent']]['type'] != 'main_stem':
        return dict(base, status='excluded', reason='not_direct_main_stem_petiole')
    parents = [c for c in catalogue if c['variant_id'] == variant and c['component_id'] == component['parent']]
    require(len(parents) == 1, 'Exact main stem catalogue identity required')
    parent = parents[0]
    geo = geometry_9mm(report, component_id, entry['plant_to_world_usd_row_vectors'], metadata['calibration'])
    base.update(geometry=geo, cut_world_m=geo['nominal']['world_m'],
                cut_point_uv=geo['nominal']['projected']['pixel_xy'],
                cut_optical_xyz_m=geo['nominal']['projected']['camera_optical_xyz_m'])
    target_y, target_x = np.nonzero(components == target['component_index'])
    base['target_bbox_convention'] = 'xmin_ymin_inclusive_xmax_ymax_exclusive_native_pixels'
    base['target_bbox_xyxy'] = ([int(target_x.min()), int(target_y.min()), int(target_x.max()+1), int(target_y.max()+1)]
                                if len(target_x) else None)

    def probe(point, allowed):
        p = point['projected']
        evidence = depth_evidence(p, depth, valid, point['radius_m'])
        visible, observed = False, None
        if p['projection_status'] == 'in_frame':
            x, y = np.floor(p['pixel_xy']).astype(int)
            observed = int(components[y, x])
            visible = observed in allowed and evidence['status'] == 'depth_consistent_not_visibility_verified'
        return dict(arc_m=point['arc_m'], projected=p, depth=evidence,
                    observed_component_index=observed, visible=bool(visible))

    # Only an authenticated mesh seam can add a same-instance joint owner.
    additional_parent_indices, joint_proof = joint_ownership.verified_additional_parent(entry, catalogue)
    parent_indices = {parent['component_index']} | additional_parent_indices
    base['attachment_joint_ownership'] = joint_proof
    own = {target['component_index']}
    nominal = probe(geo['nominal'], own)
    support = [probe(p, own) for p in geo['visibility_support']]
    proximal = [probe(p, own if p['arc_m'] >= .008 else own | parent_indices) for p in geo['proximal']]
    junction = [probe(p, own if p['arc_m'] >= .008 else own | parent_indices) for p in geo['junction_to_cut']]
    base['junction_continuity'] = dict(attachment_visible=junction[0]['visible'],
        all_attachment_to_9mm_probes_visible=all(p['visible'] for p in junction),
        maximum_arc_step_m=.0005, probes=junction)
    base['visibility'] = dict(nominal=nominal, support=support, proximal=proximal)
    if not nominal['visible'] or not all(p['visible'] for p in support):
        uncertain = any(p['projected']['projection_status'] == 'in_frame'
                        and p['depth']['status'].startswith('unknown') for p in [nominal, *support])
        return dict(base, status='unknown' if uncertain else 'excluded', reason='cut_or_support_not_native_visible')
    if not all(p['visible'] for p in junction):
        uncertain = any(p['projected']['projection_status'] == 'in_frame'
                        and p['depth']['status'].startswith('unknown') for p in junction)
        return dict(base, status='unknown' if uncertain else 'excluded', reason='attachment_to_9mm_continuity_not_verified')
    unique = {}
    for p in proximal:
        if p['projected']['projection_status'] == 'in_frame':
            unique.setdefault(tuple(np.floor(p['projected']['pixel_xy']).astype(int)), []).append(p['visible'])
    fraction = sum(all(values) for values in unique.values())/len(unique) if unique else 0.
    mask = components == target['component_index']
    screen = local_screen(rgb, mask, [p['projected']['pixel_xy'] for p in support], fraction)
    base['local_clarity'] = screen
    if not screen['passed']:
        return dict(base, status='excluded', reason='strict_native_local_clarity_failed')
    attach = geo['attachment_projected']
    if attach['projection_status'] != 'in_frame':
        return dict(base, status='excluded', reason='attachment_out_of_frame')
    x, y = np.floor(attach['pixel_xy']).astype(int)
    # Apply the same clearance formula to every proven joint-forming stem.
    parent_mask = np.isin(components, sorted(parent_indices))
    count = int(parent_mask[max(0, y-12):min(408, y+13), max(0, x-12):min(848, x+13)].sum())
    yy, xx = np.nonzero(parent_mask)
    uv = np.asarray(base['cut_point_uv'])
    distance = float(np.min(np.hypot(xx+.5-uv[0], yy+.5-uv[1]))) if len(xx) else None
    z = geo['nominal']['projected']['camera_optical_xyz_m'][2]
    cal = metadata['calibration']
    diameter = 2*geo['nominal']['radius_m']*min(cal['intrinsics'][0][0], cal['intrinsics'][1][1])/z
    base['parent_context'] = dict(parent_component_indices=sorted(parent_indices),
                                  parent_scope='declared_parent_and_authenticated_joint_forming_segment',
                                  visible_parent_pixels=count, cut_to_parent_distance_px=distance,
                                  required_distance_px=max(3., diameter*.65), estimated_diameter_px=float(diameter))
    if count < 6 or distance is None or distance < max(3., diameter*.65):
        return dict(base, status='excluded', reason='insufficient_visible_parent_clearance')
    # The solver's exact cache key includes this NEW target. No 10 mm IK reuse.
    meta = deepcopy(metadata)
    meta['supervision'] = dict(target_id=target_id, nominal_world_m=base['cut_world_m'])
    workspace = workspace_checker.check(meta)
    require(workspace['target_id'] == target_id
            and np.allclose(workspace['nominal_world_m'], base['cut_world_m'], atol=1e-12, rtol=0),
            'Workspace proof must bind the new 9 mm point')
    base['workspace'] = workspace
    if workspace['result']['workspace_passed'] is not True:
        return dict(base, status='excluded', reason='workspace_9mm_failed')
    return dict(base, status='candidate_pending_visual_review', reason='9mm_local_visibility_and_workspace_passed',
                automated_pass=True)


def evaluate_frame(metadata, rgb, depth, valid, components, catalogue, target_inventory, workspace_checker, *, scene_coverage=None):
    """Enumerate the complete catalogue; omitted geometry becomes UNKNOWN, never absent."""
    cal = metadata['calibration']
    require(cal['resolution'] == [848, 408] and cal['crop_resize'] is None
            and cal['depth_convention'] == 'optical_axis_z_metres_not_ray_range', 'Native unscaled calibration required')
    require(rgb.shape == (408, 848, 3) and rgb.dtype == np.uint8
            and depth.shape == valid.shape == components.shape == (408, 848)
            and depth.dtype == np.float32 and valid.dtype == bool
            and np.issubdtype(components.dtype, np.integer), 'Exact native RGB/depth/identity arrays required')
    near, far = cal['clipping_range_m']
    require(np.array_equal(valid, np.isfinite(depth) & (depth > 0) & (depth >= near) & (depth <= far)), 'Invalid depth validity')
    require(metadata['synchronization']['scene_unchanged_during_capture'] is True
            and metadata['synchronization']['dynamic_recording_supported'] is False, 'Authenticated static frame required')
    expected = [c['variant_id']+'/'+c['component_id'] for c in catalogue if c['organ_type'] == 'sub_stem']
    require(len(expected) == len(set(expected)), 'Duplicate catalogue petiole identity')
    supplied = [e['target_id'] for e in target_inventory]
    require(len(supplied) == len(set(supplied)) and set(supplied) <= set(expected), 'Duplicate or foreign inventory target')
    entries = {e['target_id']: e for e in target_inventory}
    rows = []
    for target_id in sorted(expected):
        if target_id not in entries:
            rows.append(dict(target_id=target_id, status='unknown', reason='missing_authenticated_target_geometry',
                             automated_pass=False, actual_visual_review=False, training_approved=False))
            continue
        try:
            rows.append(evaluate_target(metadata, rgb, depth, valid, components, catalogue, entries[target_id], workspace_checker))
        except (ValueError, KeyError, AssertionError) as exc:
            rows.append(dict(target_id=target_id, status='unknown', reason='invalid_target_evidence: '+str(exc),
                             automated_pass=False, actual_visual_review=False, training_approved=False))
    census = dict(complete=False, catalogue_complete=set(supplied) == set(expected), catalogue_petiole_ids=sorted(expected),
                  full_scene_coverage=deepcopy(scene_coverage),
                  full_scene_coverage_validation_required=True,
                  candidate_target_ids=[r['target_id'] for r in rows if r['status'] == 'candidate_pending_visual_review'],
                  excluded_target_ids=[r['target_id'] for r in rows if r['status'] == 'excluded'],
                  unknown_target_ids=[r['target_id'] for r in rows if r['status'] == 'unknown'])
    return dict(schema=ANNOTATION_EPOCH, annotation_epoch=ANNOTATION_EPOCH, frame_id=metadata['sample_id'],
                state='all_petiole_9mm_candidates_pending_authentication_and_actual_QA', targets=rows,
                target_census=census, frame_blocked_by_unknown_targets=bool(census['unknown_target_ids']),
                frame_blocked_by_unverified_full_scene_coverage=True,
                input_contract=dict(full_rgb=True, aligned_optical_z_depth=True, query_input=False,
                                    target_id_input=False, target_mask_input=False, target_crop_input=False),
                private_robot_context=dict(robot_snapshot=deepcopy(metadata['robot_snapshot']), calibration=deepcopy(cal)),
                training_approved=False, accepted_training_increment=0)
