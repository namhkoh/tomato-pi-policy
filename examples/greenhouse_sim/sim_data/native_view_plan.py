"""Bounded actual-robot view proposals for one qualified generated plant.

GT geometry chooses static capture snapshots, not blind VLM evaluation or robot
actions. No camera-mount edits, global cap resets, split edits or data approval.
"""
from pathlib import Path
import argparse
import hashlib
import math
import numpy as np
from .dataset_review import require,read_json,write_json,verify_bindings
from .depth_preview import sha256
from .capture_contract import transform_points

SCHEMA='greenhouse.generated_native_multiview_plan.v3'


def implementation_paths():
    code_root=Path(__file__).parent
    paths=sorted(code_root.glob('*.py'))
    paths.append(code_root.parent/'launch_sim_data.py')
    paths.extend(code_root.parent/'greenhouse_sim'/n for n in ('robot_hardware.py','robot_kinematics.py'))
    return paths


def propose_specs(root_row, target_world, seed_key, count=6):
    require(type(count) is int and 1<=count<=6,'Bounded one-to-six-view pilot required')
    matrix=np.asarray(root_row,float);target=np.asarray(target_world,float)
    require(matrix.shape==(4,4) and target.shape==(3,) and np.isfinite(matrix).all()
            and np.isfinite(target).all(),'Finite robot/target geometry required')
    delta=matrix[3,:2]-target[:2]
    require(abs(delta[0])>.02,'Ambiguous reference aisle side')
    side=1 if delta[0]>0 else -1
    require(np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6),
            'Rigid source robot transform required')
    heading=math.degrees(math.atan2(matrix[0,1],matrix[0,0]))
    if side>0 and heading<0:heading+=360
    # The bound is the proven reference orientation, not a new target-facing
    # orbit constraint. Native geometry and head-joint checks still apply.
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8],'little'))
    rows=[]
    for i,(outward,lateral) in enumerate([(0,0),(0,-.04),(0,.04),(.04,0),(.04,-.04),(.04,.04)][:count]):
        # Preserve the already-clear body orientation. Facing the whole robot
        # at the target can swing stowed arms into neighbouring stems.
        dx=delta[0]+side*outward;dy=delta[1]+lateral
        # Continuous framing distribution: do not encode a fixed centre answer.
        pixel=[float(rng.uniform(.30,.70)*848),float(rng.uniform(.30,.65)*408)]
        rows.append(dict(candidate_id=f'view_{i+1:03d}',root_x_m=float(target[0]+dx),
            y_offset_m=float(dy),root_yaw_degrees=heading,desired_pixel_xy=pixel,
            framing_reference_resolution=[848,408],opposite_aisle=side<0,
            orbit_clear=False,near_clear=False,base_target_radius_m=float(math.hypot(dx,dy)),
            base_displacement_from_reference_m=float(math.hypot(outward,lateral)),
            base_heading_preserved=True,view_policy='reference_outward_lateral_real_head.v1'))
    return rows


def build(base_path,count=6):
    from .generated_capture import check_plan,verify_sensor_prerequisite
    from .native_greenhouse_pair import load_source
    base_path=Path(base_path).resolve();base=read_json(base_path);check_plan(base)
    prerequisite=verify_sensor_prerequisite(base)
    _,sample,source_bindings=load_source(base['source_capture'],base['source_sample'])
    row=base['generated_row']
    world=transform_points([row['cut_region_proposal']['nominal']['point_plant_m']],
                           sample['supervision']['plant_to_world_usd_row_vectors'])[0]
    specs=propose_specs(base['expected_robot_snapshot']['robot_root_to_world_usd_row_vectors'],
                        world,sha256(base_path),count)
    code_paths=implementation_paths()
    return dict(schema=SCHEMA,base_pair_plan=str(base_path),base_pair_plan_sha256=sha256(base_path),
        source_bindings=source_bindings,prerequisite_bindings=prerequisite['bindings'],
        implementation_bindings={str(p.resolve()):sha256(p) for p in code_paths},
        expected_nominal_world_m=world.tolist(),views=specs,maximum_native_frames=count,
        resolution=[1696,816],requested_render_subframes_per_view=56,split='train',
        source_family=base['source_family'],conservative_view_cap_group=base['conservative_view_cap_group'],
        source_cap_reset=False,training_approved=False,physical_motion_commanded=False,
        hidden_cut_coordinates_executable=False,geometry_guided_capture_not_blind_evaluation=True)


def check(plan):
    from .generated_capture import check_plan
    require(plan.get('schema')==SCHEMA and plan.get('resolution')==[1696,816]
            and plan.get('requested_render_subframes_per_view')==56,'Unknown view/sensor/render plan')
    require(all(plan.get(k) is False for k in ('source_cap_reset','training_approved',
        'physical_motion_commanded','hidden_cut_coordinates_executable')),'Unexpected execution or data approval')
    base_path=Path(plan['base_pair_plan'])
    require(sha256(base_path)==plan['base_pair_plan_sha256'],'Base native plan changed')
    base=read_json(base_path);check_plan(base)
    require(plan['split']==base['split']=='train' and plan['source_family']==base['source_family']
            and plan['conservative_view_cap_group']==base['conservative_view_cap_group'],'Source/cap/split changed')
    verify_bindings(plan['source_bindings']);verify_bindings(plan['prerequisite_bindings'])
    verify_bindings(plan['implementation_bindings'])
    expected=build(base_path,plan['maximum_native_frames'])
    require(plan==expected,'Frozen view plan or code coverage changed')
    return base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--views',type=int,default=6)
    a=p.parse_args();require(not a.output.exists(),'New view plan only')
    plan=build(a.base_plan,a.views);check(plan)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,plan)
    print('NATIVE_GENERATED_VIEWS_PLANNED',len(plan['views']),flush=True)


if __name__=='__main__':main()
