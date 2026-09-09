"""Versioned RGB/query-pixel grounding labels. No execution or human-review claims.

Source geometry supplies synthetic labels, not observations. Conservative native
visibility checks either localize, label a proved occlusion, or exclude ambiguity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .capture_contract import project, transform_points, depth_evidence
from .cut_regions import _oriented_chain, _sample
from .dataset_review import read_json, require

TASK_ID='greenhouse.target_conditioned_cutpoint_rgb.v2'
CONTRACT={
    'task_id':TASK_ID, 'input':'unaltered_full_robot_head_rgb_plus_visible_petiole_query_pixel',
    'resolution':[848,408], 'coordinates':'continuous_pixels_top_left_edge_origin_x_right_y_down',
    'nominal_arc_m':.01, 'accepted_arc_m':[.01,.02],
    'target_query':'visible_petiole_centerline_at_least_45mm_from_attachment_and_18px_from_nominal',
    'positive_query_association':'same_8_connected_native_visible_petiole_region_as_the_nominal_cut_no_gap_filling',
    'label_source':'automatic_manifest_geometry_plus_native_instance_and_camera_Z',
    'human_review_claim':'none_unless_separate_explicit_record',
    'release_scope':'synthetic_perception_finetuning_not_agronomic_or_physical_cut_approval',
    'depth':'native_optical_axis_Z_metres_sidecar_not_RGB_model_input',
    'xyz':'anatomical_centerline_camera_optical_and_world_sidecar_not_visible_surface_or_RGB_output',
    'difficulty':'operational_visible_proximal_centerline_bins_not_human_validated_agronomic_difficulty',
    'no_training_images':['overlays','masks','review_cards','depth_heatmaps'],
    'unsupported':['autonomous_target_selection','grasp_trajectory','blade_trajectory','physical_safety',
                   'dynamic_RGBD_synchronization','real_world_generalization','cosmos_real_hard'],
    'thresholds':{'minimum_query_distance_px':18.,'minimum_diameter_px':4.,'minimum_interval_px':6.,
                  'minimum_parent_pixels':6,'minimum_proximal_visible_fraction':.8,
                  'easy_proximal_fraction':.95,'minimum_interval_visible_fraction':.8,
                  'maximum_target_dark_fraction':.5},
}

SYSTEM_PROMPT=(
    'You localize a synthetic tomato petiole cut point in a robot-head RGB image. '
    'The user supplies a query pixel on the target petiole, NOT the cut point. '
    'Trace that petiole to its junction with the main stem. A localized target '
    'must be traceable through visible petiole pixels from the query to the cut region. The dataset convention '
    'places the nominal cut 10 mm along the petiole toward the leaves from its '
    'attachment; 10-20 mm is the evaluation interval, not a spherical tolerance. '
    'Use visual evidence. If the cut region is occluded or cannot be distinguished, '
    'abstain instead of inventing coordinates. Coordinates are continuous pixels '
    'in the ORIGINAL 848x408 image, top-left edge origin, x right and y down. '
    'Return ONLY JSON with exactly these keys: status (localized or abstain), '
    'cut_point_uv (a two-number pixel array or null), visibility (clear, partial, '
    'or occluded), next_action (inspect_cut_region or change_viewpoint). '
    'This is perception only; never claim that a blade motion is safe or executable.'
)


def contract_hash():
    return hashlib.sha256(json.dumps(CONTRACT,sort_keys=True,allow_nan=False).encode()).hexdigest()


def user_prompt(query):
    require(isinstance(query,list) and len(query)==2 and np.isfinite(query).all()
            and 0<=query[0]<848 and 0<=query[1]<408, 'Invalid target query')
    return (f'The target petiole passes through pixel ({query[0]:.1f}, {query[1]:.1f}). '
            'Locate its nominal cut point if the junction and cut region are visually '
            'distinguishable; otherwise abstain. Use the original full image.')


def validate_answer(answer):
    require(isinstance(answer,dict) and set(answer)=={'status','cut_point_uv','visibility','next_action'},'Invalid answer keys')
    if answer['status']=='localized':
        p=answer['cut_point_uv']
        require(isinstance(p,list) and len(p)==2 and all(type(v) in (int,float) for v in p)
                and np.isfinite(p).all() and 0<=p[0]<848 and 0<=p[1]<408, 'Invalid cut pixel')
        require(answer['visibility'] in ('clear','partial') and answer['next_action']=='inspect_cut_region', 'Invalid localization decision')
    else:
        require(answer['status']=='abstain' and answer['cut_point_uv'] is None
                and answer['visibility']=='occluded' and answer['next_action']=='change_viewpoint', 'Invalid abstention')
    return answer


def derive_label(directory, metadata, report):
    """Reconstruct query and proximal evidence from ORIGINAL source chains + buffers.

    Run only after the independent RGB/Z/mask/FK audit. Unknown visibility and
    unresolved attachment context are exclusions, never automatic negatives.
    """
    directory=Path(directory)
    sup,cal=metadata['supervision'],metadata['calibration']
    component=report['components'][sup['target_id'].split('/')[-1]]
    require(sup['split_group']==report['plant_id'], 'Wrong source family')
    chain,lengths,_,_=_oriented_chain(component,1e-6)
    origin=component['translation_plant_m']
    matrix=sup['plant_to_world_usd_row_vectors']
    depth=np.load(directory/'inputs/depth_m.npy',allow_pickle=False)
    valid=np.asarray(Image.open(directory/'inputs/depth_valid.png'))==255
    components=np.load(directory/'supervision/component_id.npy',allow_pickle=False)
    identities=read_json(directory/'supervision/identities.json')
    cat=identities['component_catalogue']
    target=next(c for c in cat if c['component_id']==component['id'] and c['variant_id']==sup['variant_id'])
    parent=next(c for c in cat if c['component_id']==component['parent'] and c['variant_id']==sup['variant_id'])

    def probe(distance, allowed):
        g=_sample(chain,lengths,float(distance),origin)
        p=project(transform_points([g['point_plant_m']],matrix),cal)[0]
        evidence=depth_evidence(p,depth,valid,g['petiole_radius_m'])
        visible=False
        if p['projection_status']=='in_frame':
            x,y=np.floor(p['pixel_xy']).astype(int)
            visible=int(components[y,x]) in allowed and evidence['status']=='depth_consistent_not_visibility_verified'
        return dict(arc_m=float(distance),projected=p,visible=bool(visible),depth_status=evidence['status'])

    nominal=np.asarray(sup['nominal_projected']['pixel_xy'])
    nominal_visible=sup['visibility_evidence']['nominal']['visible_target_evidence']
    connectivity=None
    nominal_component=0
    if nominal_visible:
        from scipy.ndimage import label as connected_components
        connectivity,_=connected_components(components==target['component_index'],structure=np.ones((3,3),bool))
        nx,ny=np.floor(nominal).astype(int)
        if 0<=nx<848 and 0<=ny<408: nominal_component=int(connectivity[ny,nx])
    query_candidates=[]
    unconnected_visible_candidates=0
    if lengths[-1]>=.045:
        for d in np.linspace(.045,min(lengths[-1]*.85,.25),41):
            if d<.045: continue
            p=probe(d,{target['component_index']})
            if p['visible'] and np.linalg.norm(np.asarray(p['projected']['pixel_xy'])-nominal)>=18:
                # Rounded query must still refer to the exact same visible petiole.
                query=[round(float(v),1) for v in p['projected']['pixel_xy']]
                x,y=np.floor(query).astype(int)
                if 0<=x<848 and 0<=y<408 and int(components[y,x])==target['component_index']:
                    if nominal_visible and (not nominal_component or int(connectivity[y,x])!=nominal_component):
                        unconnected_visible_candidates+=1
                        continue
                    query_candidates.append((p,query))
    base=dict(task_id=TASK_ID,contract_sha256=contract_hash(),eligible=False,human_review_performed=False,
              target_id=sup['target_id'],source_plant_family=report['plant_id'],label_origin=CONTRACT['label_source'])
    if not query_candidates:
        return {**base,'reason':'no_visible_query_connected_to_cut' if unconnected_visible_candidates else 'no_unambiguous_visible_distal_query'}
    # Vary query distance; do not teach a fixed query-to-cut vector.
    index=int.from_bytes(hashlib.sha256((sup['target_id']+metadata['sample_id']).encode()).digest()[:4],'little')%len(query_candidates)
    query_evidence,query=query_candidates[index]
    proximal=[probe(d,{target['component_index']} if d>=.008 else {target['component_index'],parent['component_index']})
              for d in np.linspace(.004,.030,27)]
    unique={}
    for p in proximal:
        if p['projected']['projection_status']=='in_frame':
            unique.setdefault(tuple(np.floor(p['projected']['pixel_xy']).astype(int)),[]).append(p['visible'])
    fraction=sum(all(v) for v in unique.values())/len(unique) if unique else 0.
    attach=project(transform_points([component['attachment_plant_m']],matrix),cal)[0]
    parent_pixels=0
    parent_distance=None
    if attach['projection_status']=='in_frame':
        x,y=np.floor(attach['pixel_xy']).astype(int)
        patch=components[max(0,y-12):min(408,y+13),max(0,x-12):min(848,x+13)]
        parent_pixels=int((patch==parent['component_index']).sum())
        yy,xx=np.nonzero(components==parent['component_index'])
        if len(xx): parent_distance=float(np.min(np.hypot(xx+.5-nominal[0],yy+.5-nominal[1])))
    visibility=sup['visibility_evidence']
    q=metadata['quality']
    base.update(query_pixel_uv=query,query_evidence=query_evidence,proximal_evidence=proximal,
                query_cut_visible_connection_verified=bool(nominal_visible and nominal_component),
                query_association_scope='8_connected_native_visible_target_pixels_no_morphological_gap_filling' if nominal_visible else 'hidden_cut_abstention_no_visible_connection_claim',
                proximal_visible_pixel_fraction=fraction,attachment_parent_visible_pixels=parent_pixels,
                nominal_distance_to_visible_parent_px=parent_distance,
                nominal_optical_xyz_m=sup['nominal_projected']['camera_optical_xyz_m'],nominal_world_m=sup['nominal_world_m'])
    t=CONTRACT['thresholds']
    if (q['estimated_petiole_diameter_px']<t['minimum_diameter_px'] or q['projected_interval_length_px']<t['minimum_interval_px']
            or q['target_mask_dark_fraction'] is None or q['target_mask_dark_fraction']>t['maximum_target_dark_fraction']):
        return {**base,'reason':'insufficient_visual_sampling_or_exposure'}
    if nominal_visible:
        if (fraction<t['minimum_proximal_visible_fraction'] or parent_pixels<t['minimum_parent_pixels']
                or parent_distance is None or parent_distance<max(3.,q['estimated_petiole_diameter_px']*.65)
                or not visibility['interval_fully_in_frame']
                or (visibility['sampled_interval_visible_pixel_fraction'] or 0)<t['minimum_interval_visible_fraction']):
            return {**base,'reason':'ambiguous_or_insufficient_junction_context'}
        easy=fraction>=t['easy_proximal_fraction'] and visibility['sampled_interval_visible_pixel_fraction']>=.95
        answer=dict(status='localized',cut_point_uv=[round(float(v),1) for v in nominal],
                    visibility='clear' if easy else 'partial',next_action='inspect_cut_region')
        difficulty='easy' if easy else 'medium'
    elif (visibility['nominal']['status']=='foreground_occluder_identified'
          and visibility['nominal'].get('observed_component') is not None):
        # A known foreground organ with depth evidence, not a failed positive gate.
        answer=dict(status='abstain',cut_point_uv=None,visibility='occluded',next_action='change_viewpoint')
        difficulty='hard'
    else:
        return {**base,'reason':'unknown_visibility_not_a_negative_label'}
    validate_answer(answer)
    return {**base,'eligible':True,'reason':'synthetic_task_contract_passed','difficulty':difficulty,'answer':answer,
            'user_prompt':user_prompt(query),'physical_execution_approved':False,'horticultural_validation':'pending'}
