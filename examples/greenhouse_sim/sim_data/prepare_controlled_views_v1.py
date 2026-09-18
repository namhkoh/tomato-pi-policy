"""Prepare mounted-camera poses for one controlled plant, without rendering.

Generated geometry is recomputed; original acceptance is never inherited.
The three saved survey anchors only supply an authenticated robot embodiment.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import math
import numpy as np

from .audit import audit_manifest
from .capture_contract import project
from .capture_scene import plan_head_pose
from .dataset_review import read_json, require, verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json
from .native848_all_petiole_9mm_v2 import geometry_9mm
from .native848_pilot_plan_v1 import raw_anchor
from .native_view_pose import bounded_reference_root

ROOT = Path(__file__).resolve().parents[3]
DIAGNOSTICS = ROOT/'data/sim_data/diagnostics'
SURVEY = DIAGNOSTICS/'native848_fully_labeled_bulk_prepare_20260917_v1/central_body_survey_all_v1/survey.json'
SURVEY_SHA = '3d3b2b559184c427101ba1046fb0bcf001ffbc2a36ba826ec16851bf6c28177e'
PROFILE = DIAGNOSTICS/'native848_bulk_original_reset8_profile_20260916_v2/profile.json'
PROFILE_SHA = '253cf90516f65400896bc5c20502676434d5996f8c81c43dd341def28657134b'


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def prepare(directory, output):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.exists(), 'New output required')
    require(sha256(SURVEY) == SURVEY_SHA and sha256(PROFILE) == PROFILE_SHA, 'Source survey/profile changed')
    q = read_json(directory/'qualification.json')
    family, cid = q['source_family'], q['recipes'][0]['component_id']
    require(q['split'] == 'train' and q['source_assets_unchanged'] is True, 'TRAIN controlled donor required')
    report = audit_manifest(directory/'manifest.json')
    require(report['status'] != 'blocked' and report['plant_id'] == directory.name, 'Generated anatomy failed')
    entries = [r for r in read_json(SURVEY)['results'] if r['target_id'] == family+'/'+cid]
    require(len(entries) == 1, 'One survey target required')
    output.mkdir(parents=True)
    records, holds, anchors, bindings = [], [], {}, {str(SURVEY):SURVEY_SHA, str(PROFILE):PROFILE_SHA,
        str(Path(__file__).resolve()):sha256(__file__), str(directory/'qualification.json'):sha256(directory/'qualification.json')}
    for relative, expected in q['output_hashes'].items():
        bindings[str(directory/relative)] = expected
    model = Rby1Kinematics()
    # One proposal per original body anchor; select at most two after fresh scene checks.
    for index, entry in enumerate(entries[0]['best'][:3]):
        source = Path(entry['native_source_sample']['path'])
        require(sha256(source) == entry['native_source_sample']['sha256'], 'Source camera changed')
        sample = read_json(source)
        anchor = raw_anchor(source.parent.parent, sample['sample_id'])
        require(anchor['source_family'] == family and anchor['split'] == 'train', 'Survey body donor differs')
        ap = output/('anchor_'+str(index)+'.json'); save_json(ap, anchor)
        anchors[str(index)] = pin(ap); bindings.update(anchor['source_bindings']); bindings[str(ap)] = sha256(ap)
        ref = anchor['expected_robot_snapshot']; root = np.asarray(ref['robot_root_to_world_usd_row_vectors']).T
        mount = np.asarray(ref['camera_to_head_column_vectors'])
        matrix = anchor['expected_original_world']['plant_to_world_usd_row_vectors']
        initial = geometry_9mm(report, cid, matrix, anchor['expected_calibration'])
        point = initial['nominal']['world_m']; uv = entry['desired_9mm_pixel_xy']
        try:
            joints, error = plan_head_pose(model, ref['joint_degrees'], root, mount, point,
                anchor['expected_calibration']['intrinsics'], uv)
            cal = deepcopy(anchor['expected_calibration'])
            cal['camera_to_world_usd_row_vectors'] = (root @ model.all_link_transforms(joints)['link_head_2'] @ mount).T.tolist()
            geo = geometry_9mm(report, cid, matrix, cal)
            probes = [geo['nominal'], *geo['visibility_support'], *geo['junction_to_cut'], *geo['proximal']]
            require(all(p['projected']['projection_status']=='in_frame' for p in probes), '9mm detail outside frame')
            width = 2*geo['nominal']['radius_m']*min(cal['intrinsics'][0][0],cal['intrinsics'][1][1])/geo['nominal']['projected']['camera_optical_xyz_m'][2]
            length = float(np.linalg.norm(np.diff([p['projected']['pixel_xy'] for p in geo['visibility_support']],axis=0),axis=1).sum())
            require(width >= 8 and length >= 12, 'Insufficient projected detail')
            spec = dict(root_x_m=float(root[0,3]), y_offset_m=float(root[1,3]-point[1]),
                root_yaw_degrees=math.degrees(math.atan2(root[1,0],root[0,0])), base_heading_preserved=True, desired_pixel_xy=uv)
            require(np.allclose(bounded_reference_root(ref,spec,point),root,rtol=0,atol=1e-9), 'Body bound differs')
            records.append(dict(sample_id=directory.name+'_view_'+str(index), source_family=family,
                target_id=directory.name+'/'+cid, component_id=cid, geometry_source_id=directory.name,
                source_target_id=family+'/'+cid, anchor=pin(ap), robot_root_to_world_usd_row_vectors=root.T.tolist(),
                camera_to_head_column_vectors=mount.tolist(), joint_degrees=joints, framing_error_degrees=error,
                requested_spec=spec, calibration=cal, pilot_9mm_geometry=geo,
                projected_width_px=float(width), projected_support_px=length,
                body_unchanged=True, geometry_and_workspace_pending=True, training_approved=False))
        except ValueError as exc:
            holds.append(dict(anchor_index=index, reason=str(exc)))
    verify_bindings(bindings)
    value = dict(schema='greenhouse.controlled_mounted_camera_candidates.v1',
        qualification=pin(directory/'qualification.json'), manifest=pin(directory/'manifest.json'),
        source_family=family, geometry_source_id=directory.name, target_component_id=cid,
        source_plan=dict(path=q['source_plan_path'],sha256=q['source_plan_sha256']), profile=pin(PROFILE),
        records=records, holds=holds, source_bindings=bindings, native_launched=False,
        training_approved=False, accepted_training_increment=0)
    save_json(output/'candidates.json',value)
    print(dict(candidates=pin(output/'candidates.json'),poses=len(records),holds=holds),flush=True)
    return value


def prepare_existing(directory, output, camera_plan, sample_ids):
    """Reuse authenticated camera poses exactly; recompute the generated label."""
    from .native848_fully_labeled_plan_v3 import check
    from .native848_controlled_capture_v1 import validate_pose
    directory,output,camera_plan=map(lambda p:Path(p).resolve(),(directory,output,camera_plan))
    require(not output.exists(),'New output required')
    plan=read_json(camera_plan);data=check(plan)
    ids=list(sample_ids)
    require(1<=len(ids)<=2 and len(set(ids))==len(ids),'One or two distinct source cameras required')
    by={r['sample_id']:r for r in data['records']}
    require(set(ids)<=by.keys(),'Selected camera absent from authenticated plan')
    q=read_json(directory/'qualification.json');family=q['source_family'];cid=q['recipes'][0]['component_id']
    require(q['split']=='train' and q['source_assets_unchanged'] is True,'TRAIN controlled donor required')
    report=audit_manifest(directory/'manifest.json')
    require(report['status']!='blocked' and report['plant_id']==directory.name,'Generated anatomy failed')
    bindings={**data['source_bindings'],str(camera_plan):sha256(camera_plan),
        str(Path(__file__).resolve()):sha256(__file__),str(directory/'qualification.json'):sha256(directory/'qualification.json')}
    bindings.update({str(directory/r):h for r,h in q['output_hashes'].items()})
    output.mkdir(parents=True);records=[]
    for index,sample_id in enumerate(ids):
        source=by[sample_id]
        require(source['source_family']==family and source['source_row']['component_id']==cid,'Source camera target differs')
        anchor_path=Path(source['anchor_reference_path'])
        require(sha256(anchor_path)==source['anchor_reference_sha256'],'Source camera anchor changed')
        anchor=read_json(anchor_path)
        require(raw_anchor(anchor['source_capture'],anchor['source_sample'])==anchor,'Original robot anchor changed')
        bindings.update(anchor['source_bindings']);bindings[str(anchor_path)]=sha256(anchor_path)
        cal=deepcopy(source['calibration']);root=np.asarray(source['robot_root_to_world_usd_row_vectors']).T
        geo=geometry_9mm(report,cid,anchor['expected_original_world']['plant_to_world_usd_row_vectors'],cal)
        probes=[geo['nominal'],*geo['visibility_support'],*geo['junction_to_cut'],*geo['proximal']]
        require(all(p['projected']['projection_status']=='in_frame' for p in probes),'Generated detail outside source camera')
        width=2*geo['nominal']['radius_m']*min(cal['intrinsics'][0][0],cal['intrinsics'][1][1])/geo['nominal']['projected']['camera_optical_xyz_m'][2]
        length=float(np.linalg.norm(np.diff([p['projected']['pixel_xy'] for p in geo['visibility_support']],axis=0),axis=1).sum())
        require(width>=8 and length>=12,'Insufficient projected detail')
        point=geo['nominal']['world_m']
        spec=dict(root_x_m=float(root[0,3]),y_offset_m=float(root[1,3]-point[1]),
            root_yaw_degrees=math.degrees(math.atan2(root[1,0],root[0,0])),base_heading_preserved=True,
            desired_pixel_xy=geo['nominal']['projected']['pixel_xy'])
        record=dict(sample_id=directory.name+'_view_'+str(index),source_family=family,
            target_id=directory.name+'/'+cid,component_id=cid,geometry_source_id=directory.name,
            source_target_id=family+'/'+cid,anchor=pin(anchor_path),
            robot_root_to_world_usd_row_vectors=deepcopy(source['robot_root_to_world_usd_row_vectors']),
            camera_to_head_column_vectors=deepcopy(source['camera_to_head_column_vectors']),
            joint_degrees=deepcopy(source['joint_degrees']),framing_error_degrees=0.,
            requested_spec=spec,calibration=cal,pilot_9mm_geometry=geo,
            projected_width_px=float(width),projected_support_px=length,
            camera_pose_exactly_preserved=True,source_camera_plan=pin(camera_plan),source_camera_sample_id=sample_id,
            source_framing_error_degrees=source['framing_error_degrees'],
            geometry_and_workspace_pending=True,training_approved=False)
        validate_pose(record,anchor);records.append(record)
    verify_bindings(bindings)
    value=dict(schema='greenhouse.controlled_mounted_camera_candidates.v1',
        qualification=pin(directory/'qualification.json'),manifest=pin(directory/'manifest.json'),
        source_family=family,geometry_source_id=directory.name,target_component_id=cid,
        source_plan=dict(path=q['source_plan_path'],sha256=q['source_plan_sha256']),
        profile=plan['profile_evidence'],records=records,holds=[],source_bindings=bindings,
        source_camera_plan=pin(camera_plan),native_launched=False,training_approved=False,accepted_training_increment=0)
    save_json(output/'candidates.json',value)
    print(dict(candidates=pin(output/'candidates.json'),poses=len(records),source_cameras_exact=True),flush=True)
    return value


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera-plan',type=Path);p.add_argument('--sample-id',action='append')
    a=p.parse_args()
    if a.camera_plan:prepare_existing(a.directory,a.output,a.camera_plan,a.sample_id or [])
    else:
        require(not a.sample_id,'--sample-id requires --camera-plan')
        prepare(a.directory,a.output)
