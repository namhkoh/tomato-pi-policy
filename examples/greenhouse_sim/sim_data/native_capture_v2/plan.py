"""Opt-in wider plans wrapping a checked existing multi-target plan.

No existing plan/split/cap/review edits. Every new snapshot needs actual native
pose, scene-overlap, camera, depth, target-mask and annotation qualification.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,math
import numpy as np
from ..dataset_review import read_json,require,verify_bindings
from ..depth_preview import sha256
from ..native_multitarget_plan import check as source_check
from .pose import POLICY,bounded_reference_root

SCHEMA='greenhouse.wider_native_multitarget_plan.v1'


def specs(reference,target_world,seed_key):
    row=np.asarray(reference['robot_root_to_world_usd_row_vectors'],float)
    target=np.asarray(target_world,float)
    require(row.shape==(4,4) and target.shape==(3,),'Robot/target shape mismatch')
    side=1 if row[3,0]>target[0] else -1
    heading=math.degrees(math.atan2(row[0,1],row[0,0]))
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8],'little'))
    result=[]
    for i,(outward,lateral) in enumerate(( (o,l) for o in (.08,.16,.24) for l in (-.08,0,.08)),1):
        x=float(row[3,0]+outward*side);y=float(row[3,1]+lateral)
        spec=dict(candidate_id=f'wide_{i:03d}',root_x_m=x,y_offset_m=float(y-target[1]),
            root_yaw_degrees=heading,desired_pixel_xy=[float(rng.uniform(.30,.70)*848),float(rng.uniform(.30,.65)*408)],
            framing_reference_resolution=[848,408],opposite_aisle=side<0,orbit_clear=False,near_clear=False,
            base_target_radius_m=float(np.linalg.norm([x-target[0],y-target[1]])),
            base_displacement_from_reference_m=math.hypot(outward,lateral),
            base_heading_preserved=True,view_policy=POLICY)
        bounded_reference_root(reference,spec,target);result.append(spec)
    return result


def build(source_path,*,replay_geometry=True):
    source_path=Path(source_path).resolve();source=read_json(source_path)
    anchor=source_check(source,replay_geometry=replay_geometry)
    bindings=dict(source['source_bindings']);bindings[str(source_path)]=sha256(source_path)
    implementation=dict(source['implementation_bindings'])
    for path in Path(__file__).parent.glob('*.py'):implementation[str(path.resolve())]=sha256(path)
    cases=[]
    for old in source['target_cases']:
        case=deepcopy(old);base=read_json(case['base_pair_plan'])
        require(sha256(case['base_pair_plan'])==case['base_pair_plan_sha256'],'Changed target reference')
        proposed=specs(base['expected_robot_snapshot'],case['expected_nominal_world_m'],sha256(case['base_pair_plan'])+'wide_v1')
        for spec in proposed:spec['candidate_id']=base['source_row']['component_id']+'_'+spec['candidate_id']
        case['views']=proposed;cases.append(case)
    return dict(schema=SCHEMA,source_multitarget_plan=str(source_path),source_plan_sha256=sha256(source_path),
        anchor_pair_plan=source['anchor_pair_plan'],source_bindings=bindings,
        implementation_bindings=implementation,prerequisite_bindings=source['prerequisite_bindings'],
        target_cases=cases,maximum_native_frames=sum(len(c['views']) for c in cases),
        candidate_views_per_target=9,source_family=source['source_family'],split='train',resolution=[1696,816],
        requested_render_subframes_per_view=56,pose_policy=POLICY,
        source_cap_reset=False,training_approved=False,physical_motion_commanded=False,
        hidden_cut_coordinates_executable=False,screened_reference=anchor['source_sample'])


def check(plan,*,replay_geometry=True):
    require(plan.get('schema')==SCHEMA,'Unknown wider-view plan')
    verify_bindings(plan['source_bindings']);verify_bindings(plan['implementation_bindings'])
    verify_bindings(plan['prerequisite_bindings'])
    expected=build(plan['source_multitarget_plan'],replay_geometry=replay_geometry)
    require(plan==expected,'Wider-view plan, policy or source changed')
    return read_json(plan['anchor_pair_plan'])
