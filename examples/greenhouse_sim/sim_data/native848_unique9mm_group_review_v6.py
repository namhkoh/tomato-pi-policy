"""Predeclared perspective-based visual sampling for complete unique9mm frames.

Every member retains unchanged automated checks. Only selected frames are claimed
individually viewed; a bad representative holds its whole group. No exporter.
"""
from pathlib import Path
import json,math
import numpy as np
from PIL import Image
from . import native848_fully_labeled_frame_validation_v5 as dataset
from . import native848_data_roots_v1 as roots
from . import native848_unique9mm_ambiguity_v2 as ambiguity
AMBIGUITY_SHA256='8152cd4194b2544badf3886f35be7e9bff2a68b4df65aa9fe620108408a99ec5'

DATASET_SHA256='a51ec6745ede143c7b14359f45a38ef66d54e3ee61e691faaea6c889b9954fc6'
SCHEMA='greenhouse.native848_fully_labeled_group_inventory.v6'
REVIEW_SCHEMA='greenhouse.native848_fully_labeled_group_review.v6'
PROTOCOL=dict(schema='greenhouse.native848_unique9mm_perspective_sampling.v1',
    max_group_members=64,max_pairwise_view_ray_degrees=2.,max_distance_ratio=1.05,
    individual_risk_normalized_margin_below=.10,
    grouping='exact_scene_physical_target_profile_sensor_body_then_contiguous_capture_order',
    representatives='first_last_cut_uv_min_max_and_every_weakest_normalized_margin',
    group_hold_on_any_bad_representative=True,first_qualification_individual=True,
    automatic_visibility_or_workspace_threshold_changes=False)
require=dataset.require


def read(spec):return json.loads(dataset.read_pin(spec).read_text())


def normalized_margins(target):
    c=target['local_clarity'];p=target['parent_context']
    result=dict(support_length=c['support_length_px']/12.-1.,width=c['width_proxy_px']/8.-1.,
        luma=c['local_median_luma']/40.-1.,dark_fraction=(.10-c['local_dark_fraction'])/.10,
        proximal_fraction=(c['proximal_visible_fraction']-.95)/.05,
        parent_pixels=p['visible_parent_pixels']/6.-1.,
        own_parent_clearance=(p['cut_to_parent_distance_px']-p['required_distance_px'])/p['required_distance_px'])
    depths=[q['depth'] for q in target['junction_continuity']['probes']]
    result['junction_depth']=min((q['tolerance_m']-abs(q['foreground_gap_m']))/q['tolerance_m'] for q in depths)
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=-1e-9 for v in result.values()),
        'Missing, nonfinite or failing automatic margin cannot enter a group')
    return result


def describe(spec):
    require(dataset.digest(dataset.__file__)==DATASET_SHA256,'Frozen per-frame validator changed')
    meta,annotation=read(spec['metadata']),read(spec['annotation'])
    target=dataset.validate_census(annotation)[0]
    dataset.target_answer(target,meta['calibration'])
    identity=dataset.validate_source_identity(spec,meta,annotation,target)
    require(dataset.digest(ambiguity.__file__)==AMBIGUITY_SHA256,'Independent ambiguity guard changed')
    assessment=ambiguity.validate_assessment(spec['ambiguity'],spec['metadata'],spec['annotation'],annotation)
    context=read(identity['context']);observation=read(annotation['observation'])
    robot=meta['robot_snapshot'];cal=meta['calibration'];cut=np.asarray(target['cut_world_m'],float)
    camera=np.asarray(cal['camera_to_world_usd_row_vectors'],float)[3,:3];ray=camera-cut
    distance=float(np.linalg.norm(ray));require(math.isfinite(distance) and distance>0,'Invalid actual view distance')
    margins=normalized_margins(target)
    flags=['near_threshold_'+k for k,v in sorted(margins.items()) if v<.10]
    if assessment['continuously_visible_alternative_ids']:flags.append('visible_alternative_with_objective_workspace_exclusion')
    for source in (annotation,target):
        extra=source.get('visual_review_flags',[])
        require(isinstance(extra,list) and all(isinstance(x,str) and x for x in extra),'Malformed explicit review flags')
        flags.extend(extra)
    nonpose=dict(body_joints={k:v for k,v in robot['joint_degrees'].items() if not k.startswith('head_')},
        camera_mount=robot['camera_to_head_column_vectors'],sensor=dataset.sensor_calibration(cal))
    key=dict(context_sha256=identity['context']['sha256'],scene=context['full_scene_census']['deterministic_census_sha256'],
        scene_policy=context['scene_policy'],source_family=target['source_family'],physical_target=target['target_id'],
        split=identity['split'],profile=context['profile_evidence'],renderer=context['renderer'],render_settings=context['render_settings'],
        target_centerline=dataset.canonical_sha(target['geometry']['oriented_centerline_world_m']),nonpose=nonpose)
    return dict(frame_id=annotation['frame_id'],annotation=spec['annotation'],ambiguity=spec['ambiguity'],group_key=dataset.canonical_sha(key),
        private_group_identity=key,context_path=identity['context']['path'],capture_index=observation['capture_index'],
        ray_unit=(ray/distance).tolist(),camera_to_cut_distance_m=distance,cut_uv=target['cut_point_uv'],
        normalized_margins=margins,individual_review_flags=sorted(set(flags)))


def fits(group,item):
    if not group:return True
    if len(group)>=64 or group[0]['group_key']!=item['group_key']:return False
    distances=[x['camera_to_cut_distance_m'] for x in group]+[item['camera_to_cut_distance_m']]
    if max(distances)/min(distances)>1.05+1e-12:return False
    threshold=math.cos(math.radians(2.))
    return all(float(np.dot(x['ray_unit'],item['ray_unit']))>=threshold-1e-12 for x in group)


def group_descriptors(descriptors):
    require(len({x['frame_id'] for x in descriptors})==len(descriptors),'Repeated candidate frame')
    ordered=sorted(descriptors,key=lambda x:(x['context_path'],x['capture_index'],x['frame_id']))
    groups=[];active=[]
    for item in ordered:
        if not fits(active,item):groups.append(active);active=[]
        active.append(item)
    if active:groups.append(active)
    output=[]
    for rows in groups:
        chosen={rows[0]['frame_id'],rows[-1]['frame_id']};reasons={x['frame_id']:[] for x in rows}
        reasons[rows[0]['frame_id']].append('first');reasons[rows[-1]['frame_id']].append('last')
        def pick(name,key):
            row=min(rows,key=lambda x:(key(x),x['capture_index'],x['frame_id']));chosen.add(row['frame_id']);reasons[row['frame_id']].append(name)
        for axis in (0,1):
            pick('cut_uv_min_'+str(axis),lambda x:x['cut_uv'][axis]);pick('cut_uv_max_'+str(axis),lambda x:-x['cut_uv'][axis])
        keys=set(rows[0]['normalized_margins'])
        require(all(set(x['normalized_margins'])==keys for x in rows),'Different margin inventory')
        for key in sorted(keys):pick('weakest_'+key,lambda x:x['normalized_margins'][key])
        for row in rows:
            if row['individual_review_flags']:chosen.add(row['frame_id']);reasons[row['frame_id']].extend(row['individual_review_flags'])
        ids=[x['frame_id'] for x in rows]
        output.append(dict(group_id=dataset.canonical_sha(dict(protocol=PROTOCOL,key=rows[0]['group_key'],members=ids)),
            group_key=rows[0]['group_key'],members=ids,selected=[x for x in ids if x in chosen],
            selection_reasons={x:reasons[x] for x in ids if x in chosen},
            maximum_pairwise_ray_degrees=max([0.]+[math.degrees(math.acos(float(np.clip(np.dot(a['ray_unit'],b['ray_unit']),-1,1)))) for a in rows for b in rows]),
            distance_ratio=max(x['camera_to_cut_distance_m'] for x in rows)/min(x['camera_to_cut_distance_m'] for x in rows)))
    return output


def inventory(frames):
    require(frames,'Empty group population')
    require(all('review' not in f or f['review'] is None for f in frames),'Predeclared population must not depend on outcomes')
    descriptors=[describe(f) for f in frames]
    groups=group_descriptors(descriptors)
    return dict(schema=SCHEMA,protocol=PROTOCOL,implementation=dataset.pin(__file__),per_frame_validator=dataset.pin(dataset.__file__),ambiguity_validator=dataset.pin(ambiguity.__file__),
        frames=frames,descriptors=descriptors,groups=groups,selected_frame_ids=[f for g in groups for f in g['selected']],
        candidate_count=len(frames),actual_visual_reviews=0,accepted_increment=0)


def prepare(frames,output):
    output=roots.diagnostic(output);require(not output.exists(),'Create-only review packet required')
    value=inventory(frames);output.mkdir(parents=True);dataset.write(output/'inventory.json',value)
    specs={d['frame_id']:f for d,f in zip(value['descriptors'],frames)}
    selected=[]
    for frame_id in value['selected_frame_ids']:
        spec=specs[frame_id];ann=read(spec['annotation']);target=dataset.validate_census(ann)[0]
        points=np.asarray([target['geometry']['attachment_projected']['pixel_xy'],target['cut_point_uv']]
            +[x['projected']['pixel_xy'] for x in target['geometry']['visibility_support']],float)
        lo=np.maximum(0,np.floor(points.min(axis=0)).astype(int)-16)
        hi=np.minimum([848,408],np.ceil(points.max(axis=0)).astype(int)+17)
        box=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
        crop_path=output/(dataset.canonical_sha(frame_id)[:24]+'_native_crop.png')
        with Image.open(dataset.read_pin(spec['rgb'])) as im:im.crop(tuple(box)).save(crop_path)
        selected.append(dict(frame_id=frame_id,**{k:spec[k] for k in ('metadata','rgb','buffers','annotation','ambiguity')},
            native_crop=dict(**dataset.pin(crop_path),box_xyxy=box,unscaled=True,source_rgb_sha256=spec['rgb']['sha256'])))
    packet=dict(schema='greenhouse.native848_unique9mm_group_packet.v1',inventory=dataset.pin(output/'inventory.json'),
        selected=selected,
        instruction='Inspect full native RGB and exact unscaled attachment/9mm crop; no resampling. Any ambiguity holds group.',accepted_increment=0)
    dataset.write(output/'visual_packet.json',packet);return dict(inventory=dataset.pin(output/'inventory.json'),packet=dataset.pin(output/'visual_packet.json'))


def validate_group_review(inventory_path,inventory_sha256,review_path,review_sha256):
    inv=read(dict(path=str(inventory_path),sha256=inventory_sha256));review=read(dict(path=str(review_path),sha256=review_sha256))
    require(inv==inventory(inv['frames']),'Changed group population, descriptors or deterministic representatives')
    require(review['schema']==REVIEW_SCHEMA and review['inventory']==dict(path=str(Path(inventory_path).resolve()),sha256=inventory_sha256)
        and review.get('CPU_fixture_only') is not True,'Exact actual group review required')
    packet_path=Path(inventory_path).resolve().parent/'visual_packet.json'
    require(review['packet']==dataset.pin(packet_path),'Exact unchanged visual packet required')
    packet=read(review['packet']);require(packet['inventory']==review['inventory'],'Packet inventory differs')
    packet_rows={x['frame_id']:x for x in packet['selected']}
    require(len(packet_rows)==len(packet['selected']) and set(packet_rows)==set(inv['selected_frame_ids']),'Packet selected population differs')
    rows=review['reviews'];by={x['frame_id']:x for x in rows}
    require(len(by)==len(rows) and set(by)==set(inv['selected_frame_ids']),'Every exact representative/flag requires actual review')
    specs={d['frame_id']:f for d,f in zip(inv['descriptors'],inv['frames'])}
    for frame_id,row in by.items():
        spec=specs[frame_id];ann=read(spec['annotation']);packet_row=packet_rows[frame_id]
        require(all(packet_row[k]==spec[k] for k in ('metadata','rgb','buffers','annotation','ambiguity')),'Packet asset substitution')
        crop=packet_row['native_crop'];crop_path=dataset.read_pin({k:crop[k] for k in ('path','sha256')})
        require(crop['unscaled'] is True and crop['source_rgb_sha256']==spec['rgb']['sha256'],'Crop source differs')
        with Image.open(dataset.read_pin(spec['rgb'])) as full,Image.open(crop_path) as cut:
            require(np.array_equal(np.asarray(full.crop(tuple(crop['box_xyxy']))),np.asarray(cut)),'Crop is not exact native pixels')
        require(row['schema']==dataset.REVIEW_SCHEMA and row['annotation']==spec['annotation'] and row['frame_id']==ann['frame_id']
            and row['actual_full_native_rgb_viewed'] is True and row['no_query_task_reviewed'] is True
            and row['actual_native_junction_crop_viewed'] is True and row['native_crop']==crop
            and row['independent_ambiguity_assessment']==spec['ambiguity'] and row['near_eligible_alternatives_inspected'] is True
            and row['decision'] in ('accept','hold','reject') and isinstance(row['reason'],str) and row['reason']
            and row.get('CPU_fixture_only') is not True,'Actual selected-image review missing')
        if row['decision']=='accept':dataset.validate_actual_review(row,ann,spec['annotation'])
    decisions=review['group_decisions'];gd={x['group_id']:x for x in decisions}
    require(len(gd)==len(decisions) and set(gd)=={g['group_id'] for g in inv['groups']},'Exact group decisions required')
    result={}
    for group in inv['groups']:
        decision=gd[group['group_id']]
        require(decision['decision'] in ('accept','hold') and isinstance(decision['reason'],str) and decision['reason'],'Explicit group decision required')
        bad=any(by[x]['decision']!='accept' for x in group['selected'])
        require(not bad or decision['decision']=='hold','Bad representative cannot admit its group')
        for frame_id in group['members']:
            result[frame_id]=dict(decision=decision['decision'],group_id=group['group_id'],
                individual_full_frame_review=frame_id in by,review_scope='individual' if frame_id in by else 'target_view_group',
                annotation=specs[frame_id]['annotation'],actual_review=by.get(frame_id),
                group_review=dict(path=str(Path(review_path).resolve()),sha256=review_sha256))
    return dict(records=result,source_bindings={str(Path(__file__).resolve()):dataset.digest(__file__),
        str(Path(inventory_path).resolve()):inventory_sha256,str(Path(review_path).resolve()):review_sha256})
