"""Independent saved-buffer/geometry replay for native848 pair trials.

This audit does not derive training labels, approve visual clarity, certify arm
workspace or motion, or qualify independent geometry. An owned zero-exit receipt
must be checked by the serial owner before this audit is an executed result.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from . import native848_pair_plan_v2 as ap
from .capture_sensor import LEGACY_RESOLUTION,sensor_profile
from .capture_contract import fingerprint,project,transform_points,depth_evidence
from .capture_visibility import component_masks,interval_visibility,view_quality,ORGAN_IDS
from .native_sensor_payload import decode_native_instances
from .native_greenhouse_pair import assert_same_camera
from .generated_capture import assert_pair_fresh
from .dataset_review import require,read_json,safe_file,verify_bindings
from .depth_preview import sha256

def image_array(p):
    with Image.open(p) as im:return np.asarray(im).copy()

def verify_camera(metadata):
    """Native-size adaptation of dataset_review's mounted FK/projection checks."""
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    cal, pose = metadata["calibration"], metadata["robot_snapshot"]
    root = np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T
    links = Rby1Kinematics().all_link_transforms(pose["joint_degrees"])
    expected = root @ links["link_head_2"] @ np.asarray(pose["camera_to_head_column_vectors"])
    observed = np.asarray(cal["camera_to_world_usd_row_vectors"])
    require(np.allclose(expected, observed.T, atol=1e-7, rtol=0), "Fresh camera disagrees with mounted FK")
    params = metadata["rendered_camera_params"]
    require(params["renderProductResolution"] == list(LEGACY_RESOLUTION) and params["metersPerSceneUnit"] == 1,
            "Native renderer dimensions/units changed")
    require(np.allclose(np.asarray(params["cameraViewTransform"]).reshape(4, 4), np.linalg.inv(observed),
                        atol=5e-5, rtol=0), "Native renderer camera differs from FK")
    probes = np.asarray([[0, 0, -1], [.1, .15, -1.2], [-.2, -.1, -2]])
    clip = np.column_stack((probes, np.ones(3))) @ np.asarray(params["cameraProjection"]).reshape(4, 4)
    require(np.isfinite(clip).all() and np.all(np.abs(clip[:, 3]) > 1e-8), "Invalid native camera projection")
    pixels = (clip[:, :2] / clip[:, 3, None] * [1, -1] + 1) * (np.asarray(list(LEGACY_RESOLUTION)) / 2)
    projected = [p["pixel_xy"] for p in project(transform_points(probes, observed), cal)]
    require(np.allclose(pixels, projected, atol=.01, rtol=0), "Native projection differs from original optics")

def world_from_row(plan, row):
    from .capture_contract import transform_points
    matrix = plan['expected_original_world']['plant_to_world_usd_row_vectors']
    proposal = row['cut_region_proposal']
    samples = proposal['accepted_centerline_interval']['samples']; points = []
    for a,b in zip(samples,samples[1:]):
        steps = max(1,int(np.ceil((b['arc_distance_m']-a['arc_distance_m'])/.001)))
        for t in np.linspace(0,1,steps,endpoint=False):
            points.append(np.asarray(a['point_plant_m'])*(1-t)+np.asarray(b['point_plant_m'])*t)
    points.append(samples[-1]['point_plant_m'])
    return dict(plant_to_world_usd_row_vectors=deepcopy(matrix),
        nominal_world_m=transform_points([proposal['nominal']['point_plant_m']],matrix)[0].tolist(),
        interval_world_m=transform_points(points,matrix).tolist())


def expected_catalogue(plan, mode, reports, generated):
    items = []
    for variant in plan['scene_variants']:
        require(not variant['added_components'] and not variant['added_component_paths'], 'Unexpected added background anatomy')
        family, root = variant['variant_id'], variant['plant_root']
        source = variant['source_plant_id']
        report = reports[family]
        if mode == 'generated_variant' and family == plan['source_family']:
            family = generated['variant_id']; root = '/World/GeneratedNativePilot/'+family
            report = generated['report']
        components = report['components']
        for key, component in components.items():
            chain, parent = [key], component['parent']
            while parent is not None:
                require(parent in components and parent not in chain, 'Invalid component ancestry')
                chain.append(parent); parent = components[parent]['parent']
            items.append(dict(component_id=key,prim_path=root+'/'+'/'.join(reversed(chain)),
                organ_type=component['type'],variant_id=family,source_plant_id=source,split_group=variant['split_group']))
    return [dict(item,component_index=i) for i,item in enumerate(sorted(items,key=lambda x:x['prim_path']),1)]


def audit_capture(capture,*,plan_path,plan_sha256,result_sha256):
    from .collection_plan import load_plan
    capture=Path(capture).resolve();plan_path=Path(plan_path).resolve()
    require(sha256(plan_path)==plan_sha256 and sha256(capture/'result.json')==result_sha256
        and not (capture/'failure.json').exists(),'Changed or failed native848 trial')
    plan=read_json(plan_path);generated=ap.check(plan,full=True)
    request=read_json(capture/'request.json');result=read_json(capture/'result.json')
    require(request['plan_sha256']==result['plan_sha256']==plan_sha256
        and Path(request['plan_path'])==plan_path and result['state']==ap.RESULT_STATE
        and result['resolution']==[848,408] and result['source_assets_unchanged'] is True
        and all(result[k] is v for k,v in ap.FLAGS.items()),'Wrong native848 result scope')
    require(request['process_admission']['no_unrelated_kit_process_at_admission'] is True
        and request['host_memory_preflight']['commit_headroom_bytes']>=20*2**30
        and request['available_disk_bytes']>=60*2**30,'Changed native process/resource admission')
    manifest=read_json(Path(plan['source_capture'])/'manifest.json')
    _,loaded=load_plan(plan['source_collection_plan']);reports={r['plant_id']:r for r in loaded}
    pins={str(plan_path):plan_sha256,str(capture/'result.json'):result_sha256,str(capture/'request.json'):sha256(capture/'request.json')}
    samples=[];records=[]
    require([r['sample_id'] for r in result['samples']]==plan['modes'],'Incomplete pair')
    for mode in plan['modes']:
        folder=capture/mode;mp=folder/'sample.json';meta=read_json(mp);pins[str(mp)]=sha256(mp)
        require(meta==next(r for r in result['samples'] if r['sample_id']==mode),'Saved/result sample mismatch')
        require(meta['schema_version']==ap.SAMPLE_SCHEMA and meta['training_sample_approved'] is False
            and meta['native_instance_backend']=='legacy' and meta['historical_labels_inherited'] is False
            and meta['sensor']==sensor_profile(LEGACY_RESOLUTION)
            and meta['robot_snapshot']==plan['expected_robot_snapshot']
            and meta['scene_counts']==plan['expected_scene_counts']
            and meta['lighting']==manifest['lighting'] and meta['renderer']==manifest['renderer']
            and meta['input_policy']==dict(clean_full_scene=True,isolation=False,diagnostic_overlays=False),'Changed actual camera/scene policy')
        sync=meta['synchronization']
        require(sync['scene_unchanged_during_capture'] is True and sync['dynamic_recording_supported'] is False
            and sync['render_budget_subframes']==56,'Changed native temporal contract')
        gp=capture/('geometry_screen_'+mode+'.json');pins[str(gp)]=sha256(gp)
        require(meta['geometry_screen']==read_json(gp)['screen'] and meta['geometry_screen']['passed'] is True,'Missing geometry screen')
        for relative,info in meta['files'].items():
            file=safe_file(folder,relative);require(sha256(file)==info['sha256'],'Changed saved native artifact');pins[str(file)]=info['sha256']
        cal=meta['calibration'];assert_same_camera(cal,plan['expected_calibration']);verify_camera(meta)
        rgb=image_array(folder/'inputs/rgb.png');depth=np.load(folder/'inputs/depth_m.npy',allow_pickle=False)
        rawvalid=image_array(folder/'inputs/depth_valid.png');valid=rawvalid==255
        require(rgb.shape==(408,848,3) and rgb.dtype==np.uint8 and depth.shape==(408,848)
            and depth.dtype==np.float32 and rawvalid.shape==(408,848) and set(np.unique(rawvalid))<={0,255},'Invalid848 native arrays')
        near,far=cal['clipping_range_m']
        require(np.array_equal(valid,np.isfinite(depth)&(depth>0)&(depth>=near)&(depth<=far)),'Depth validity differs')
        fresh=sync['freshness']
        require(fresh['rgb_sha256']==hashlib.sha256(rgb.tobytes()).hexdigest()
            and fresh['depth_sha256']==hashlib.sha256(depth.tobytes()).hexdigest()
            and fresh['camera_sha256']==fingerprint(cal),'Native callback fingerprint mismatch')
        ids=np.load(folder/'supervision/renderer_instance_id.npy',allow_pickle=False)
        identities=read_json(folder/'supervision/identities.json');catalogue=identities['component_catalogue']
        require(catalogue==expected_catalogue(plan,mode,reports,generated)
            and len(catalogue)==meta['scene_counts']['components'] and identities['organ_ids']==ORGAN_IDS,'Full original/generated hierarchy differs')
        mapping={int(k):v for k,v in identities['renderer_id_to_prim'].items()}
        decode_native_instances({'instance_id_segmentation':{'data':ids,'info':{'idToLabels':mapping}}},LEGACY_RESOLUTION)
        require(meta['native_instance_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
            and meta['native_mapping_sha256']==fingerprint(mapping),'Native identity fingerprint mismatch')
        components,organs,owners=component_masks(ids,mapping,catalogue)
        require(np.array_equal(components,np.load(folder/'supervision/component_id.npy',allow_pickle=False))
            and np.array_equal(organs,image_array(folder/'supervision/organ_type.png')),'Derived native masks differ')
        row=plan['source_row' if mode=='original_control' else 'generated_row'];world=world_from_row(plan,row);sup=meta['supervision']
        require(sup['target_id']==row['target_id'] and sup['source_target_id']==sup['conservative_view_cap_group']==plan['conservative_view_cap_group']
            and sup['split_group']==plan['source_family'] and sup['cut_region_proposal']==row['cut_region_proposal']
            and sup['cut_safety_validated'] is False,'Changed target ancestry')
        for key,value in world.items():require(np.allclose(sup[key],value,atol=1e-9,rtol=0),'Changed metric geometry '+key)
        target=next(c for c in catalogue if c['variant_id']==row['variant_id'] and c['component_id']==row['component_id'])
        nominal=project([world['nominal_world_m']],cal)[0];interval=project(world['interval_world_m'],cal)
        radius=row['cut_region_proposal']['nominal']['petiole_radius_m']
        visibility,mask=interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,target,radius)
        require(sup['nominal_projected']==nominal and sup['projected_interval']==interval
            and sup['depth_evidence']==depth_evidence(nominal,depth,valid,radius) and sup['visibility_evidence']==visibility
            and meta['quality']==view_quality(cal,nominal,interval,radius,visibility,rgb,mask)
            and meta['native_target_pixels']==int(mask.sum())
            and np.array_equal(image_array(folder/'supervision/target_visible.png'),mask.astype(np.uint8)*255),'Native visibility replay differs')
        oldroot=plan['original_variant']['plant_root'];oldids=[i for i,path in mapping.items() if path==oldroot or path.startswith(oldroot+'/')]
        require(meta['old_plant_native_pixels']==int(np.isin(ids,oldids).sum()),'Old foreground identity differs')
        samples.append(meta);records.append(dict(mode=mode,target_id=row['target_id'],sample_path=str(mp),sample_sha256=sha256(mp),rgb_sha256=sha256(folder/'inputs/rgb.png'),same_callback_buffers_replayed=True,full_native_identity_masks_replayed=True,metric_cut_geometry_recomputed=True,accepted_training_increment=0))
    pair=assert_pair_fresh(*samples)
    verify_bindings(pins);ap.check(plan)
    return dict(schema='greenhouse.native848_pair_buffer_geometry_audit.v2',state='saved_native848_pair_buffers_geometry_replayed_pending_clarity_workspace_novelty',records=records,pair_checks=pair,source_bindings=pins,strict_clarity_qualified=False,generated_workspace_qualified=False,accepted_training_increment=0,**ap.FLAGS)
