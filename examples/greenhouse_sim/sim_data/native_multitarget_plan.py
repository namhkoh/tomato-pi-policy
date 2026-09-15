"""Frozen multi-target pilot using one plant and shared native sensor proof.

The sensor proof qualifies mounted optics/native annotators, not visibility of
new targets. Every new pose still needs actual calibration, geometry, native
component/depth continuity and image review. No split/cap/release approval.
"""
from pathlib import Path
import argparse
import numpy as np
from .dataset_review import read_json,write_json,require,verify_bindings
from .depth_preview import sha256
from .capture_contract import transform_points
from .native_view_plan import propose_specs,implementation_paths

SCHEMA='greenhouse.native_multitarget_plan.v1'
OPTICS=('camera_path','resolution','intrinsics','focal_length_mm','apertures_mm',
        'aperture_offsets_mm','clipping_range_m','depth_convention','crop_resize')


def assert_compatible(anchor,case):
    for key in ('source_family','split','split_group','variant_directory',
                'source_collection_plan','expected_scene_counts','original_variant'):
        require(case[key]==anchor[key],'Multi-target scene/lineage mismatch: '+key)
    require(case['split']=='train','TRAIN-only pilot')
    for key in OPTICS:
        require(case['expected_calibration'][key]==anchor['expected_calibration'][key],
                'Multi-target mounted optics mismatch: '+key)
    require(np.allclose(case['expected_robot_snapshot']['camera_to_head_column_vectors'],
                        anchor['expected_robot_snapshot']['camera_to_head_column_vectors'],
                        atol=1e-9,rtol=0),'Multi-target camera mount changed')
    require(case['expected_robot_snapshot'].get('visual_bound_screen',{}).get('passed') is True,
            'Screened original robot reference required')
    require(case['source_row']['component_id']==case['generated_row']['component_id'],
            'Generated/source target mismatch')


def merge_bindings(destination,source):
    for path,value in source.items():
        require(path not in destination or destination[path]==value,'Conflicting source hashes: '+path)
        destination[path]=value


def build(anchor_path,case_paths,views=6):
    from .generated_capture import check_plan,verify_sensor_prerequisite
    from .native_greenhouse_pair import load_source
    from .plant_variant_catalogue import load_for_inspection
    anchor_path=Path(anchor_path).resolve();paths=[Path(p).resolve() for p in case_paths]
    require(1<=len(paths)<=12 and len(set(paths))==len(paths),'One-to-twelve distinct target bases required')
    anchor=read_json(anchor_path);check_plan(anchor)
    proof=verify_sensor_prerequisite(anchor)
    catalogue=load_for_inspection(anchor['variant_directory'],anchor['source_collection_plan'])
    generated_rows={r['target_id']:r for r in catalogue['rows']}
    original_manifest,_,anchor_bindings=load_source(anchor['source_capture'],anchor['source_sample'])
    bindings=dict(anchor_bindings);merge_bindings(bindings,anchor['source_bindings'])
    bindings[str(anchor_path)]=sha256(anchor_path)
    cases=[];identities=set()
    for path in paths:
        base=read_json(path);check_plan(base);assert_compatible(anchor,base)
        identity=base['generated_row']['target_id']
        require(identity not in identities,'Repeated target cannot increase pilot volume')
        identities.add(identity)
        require(base['generated_row']==generated_rows.get(identity),'Generated row differs from replayed geometry')
        manifest,sample,source_bindings=load_source(base['source_capture'],base['source_sample'])
        require(all(manifest[k]==original_manifest[k] for k in
                ('lighting','renderer','scene_counts','unbundled_external_prop_roots_excluded')),
                'Source scene/lighting differs between targets')
        world=transform_points([base['generated_row']['cut_region_proposal']['nominal']['point_plant_m']],
                               sample['supervision']['plant_to_world_usd_row_vectors'])[0]
        specs=propose_specs(base['expected_robot_snapshot']['robot_root_to_world_usd_row_vectors'],
                            world,sha256(path),views)
        component=base['source_row']['component_id']
        for spec in specs:spec['candidate_id']=component+'_'+spec['candidate_id']
        cases.append(dict(base_pair_plan=str(path),base_pair_plan_sha256=sha256(path),
            target_id=identity,expected_nominal_world_m=world.tolist(),views=specs,
            conservative_view_cap_group=base['conservative_view_cap_group']))
        merge_bindings(bindings,source_bindings);merge_bindings(bindings,base['source_bindings'])
        bindings[str(path)]=sha256(path)
    result=dict(schema=SCHEMA,anchor_pair_plan=str(anchor_path),
        source_bindings=bindings,prerequisite_bindings=proof['bindings'],
        implementation_bindings={str(p.resolve()):sha256(p) for p in implementation_paths()},
        target_cases=cases,views_per_target=views,maximum_native_frames=sum(len(c['views']) for c in cases),
        source_family=anchor['source_family'],split='train',resolution=[1696,816],
        requested_render_subframes_per_view=56,source_cap_reset=False,
        training_approved=False,physical_motion_commanded=False,hidden_cut_coordinates_executable=False,
        sensor_qualification_scope='mounted_camera_optics_and_native_annotators_not_each_target_visibility',
        new_targets_require_native_visibility_and_geometry_checks=True)
    verify_bindings(bindings)
    return result


def check(plan,*,replay_geometry=True):
    require(plan.get('schema')==SCHEMA,'Unknown multi-target plan')
    verify_bindings(plan['source_bindings']);verify_bindings(plan['prerequisite_bindings'])
    verify_bindings(plan['implementation_bindings'])
    # In an Isaac worker, importing USD before SimulationApp can load the pip
    # build beside Kit's ABI-incompatible USD. Hash-only preflight must not
    # replay the geometry catalogue; full replay follows application startup.
    if not replay_geometry:
        from .generated_capture import check_plan
        anchor=read_json(plan['anchor_pair_plan']);check_plan(anchor)
        require(plan['resolution']==[1696,816] and plan['split']=='train'
                and plan['requested_render_subframes_per_view']==56,'Unexpected native capture profile')
        require(all(plan.get(k) is False for k in ('source_cap_reset','training_approved',
            'physical_motion_commanded','hidden_cut_coordinates_executable')),'Unexpected approval')
        return anchor
    expected=build(plan['anchor_pair_plan'],[c['base_pair_plan'] for c in plan['target_cases']],
                   plan['views_per_target'])
    require(plan==expected,'Frozen multi-target plan changed')
    return read_json(plan['anchor_pair_plan'])


def capture_jobs(plan,anchor):
    if 'target_cases' not in plan:
        return [(anchor,plan,spec) for spec in plan['views']]
    result=[]
    for case in plan['target_cases']:
        base=read_json(case['base_pair_plan'])
        require(sha256(case['base_pair_plan'])==case['base_pair_plan_sha256'],'Target base changed')
        assert_compatible(anchor,base)
        result.extend((base,case,spec) for spec in case['views'])
    require(len(result)==plan['maximum_native_frames'],'Capture job count differs from frozen plan')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor',type=Path,required=True)
    p.add_argument('--targets',type=Path,nargs='+',required=True)
    p.add_argument('--views',type=int,default=6)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'New multi-target plan only')
    plan=build(a.anchor,a.targets,a.views);check(plan)
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,plan)
    print('NATIVE_MULTITARGET_PLANNED',plan['maximum_native_frames'],flush=True)


if __name__=='__main__':main()
