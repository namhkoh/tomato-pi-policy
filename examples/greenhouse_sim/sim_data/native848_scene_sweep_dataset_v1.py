"""Materialize reviewed fixed-greenhouse sweeps with explicit capture provenance."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from PIL import Image
from . import native848_unique_petiole_dataset_v8 as base
from . import native848_scene_sweep_9mm_v1 as evaluated
from . import native848_scene_sweep_raw_9mm_v1 as raw_evaluated
from . import native848_unique9mm_joint_contract_v1 as strict
from . import native848_unique9mm_ambiguity_v3 as ambiguity
from . import background_scoring_v1 as background
from .dataset_review import verify_bindings

SCHEMA='greenhouse.native848_reviewed_scene_sweep_dataset.v1'
FRAME_SCHEMA=base.FRAME_SCHEMA
STATE='complete_reviewed_scene_sweep_dataset'
REVIEW_SCHEMA='greenhouse.native848_scene_sweep_review.v1'
require=base.require
pin=base.pin
write=base.write
load_model_inputs=base.load_model_inputs
EVALUATORS={evaluated.SCHEMA:evaluated,raw_evaluated.SCHEMA:raw_evaluated}
PINS={'D:\\research\\tomato-pi-policy\\examples\\greenhouse_sim\\sim_data\\native848_unique_petiole_dataset_v8.py': 'f3934b4f89b717866e58415c6700184d77ec497b6886005ff0c7e642ea41414a', 'D:\\research\\tomato-pi-policy\\examples\\greenhouse_sim\\sim_data\\native848_scene_sweep_9mm_v1.py': '96adb168d8dbacb7240888e3e8e4875f0472592f0dd2004cdd0063141d567721', 'D:\\research\\tomato-pi-policy\\examples\\greenhouse_sim\\sim_data\\native848_scene_sweep_raw_9mm_v1.py': 'eea45d63bbbcc72d61c723fa608a4b648ab4654577b9ad39c2b393276902f0ac', 'D:\\research\\tomato-pi-policy\\examples\\greenhouse_sim\\sim_data\\native848_unique9mm_joint_contract_v1.py': '166f6e18ca50721137b7d431027fbb4fdd4c9de948ee849c00842e598c8efe84', 'D:\\research\\tomato-pi-policy\\examples\\greenhouse_sim\\sim_data\\native848_unique9mm_ambiguity_v3.py': '2f9936a03bf2b405dd97fac19988afcbb4abe13612055c65c13098291d44754d', 'D:\\research\\tomato-pi-policy\\examples\\greenhouse_sim\\sim_data\\background_scoring_v1.py': 'fce68b2343e90b744aac4ed3d9eea2b18ad191f2e98e016670fce795ddb2dec6'}

def checked_review(review,evaluation_pin,record,assessment):
    require(review['schema']==REVIEW_SCHEMA and review['evaluation']==evaluation_pin
        and review.get('human_review_claimed') is False,'Exact actual assistant review required')
    rows={r['frame_id']:r for r in review['records']}
    require(len(rows)==len(review['records']),'Duplicate review frame')
    r=rows[record['frame_id']]
    require(r['decision']=='accept' and all(r.get(k) is True for k in (
        'actual_full_native_rgb_viewed','junction_and_first_leaf_reviewed','no_obvious_ghosting',
        'single_answer_uniqueness_reviewed','background_vines_visible','all_flagged_alternatives_inspected')),
        'Actual complete visual review required')
    require(all(r[k]==record[k] for k in ('annotation','ambiguity','rgb','background')),'Review evidence differs')
    required={a['target_id'] for a in assessment['alternative_assessments'] if a['workspace_recheck_required']}
    inspected=r['inspected_alternative_ids']
    require(len(inspected)==len(set(inspected)) and required<=set(inspected),'Unreviewed flagged alternative')
    return r

def materialize(evaluation_pin,selected,review_pin,output,*,authorization_pin):
    selected=list(selected);require(selected and len(selected)==len(set(selected)),'Nonempty unique selection required')
    output=base.roots.checkpoint_output(output);bound={}
    def read(spec,*,within=None):
        spec={k:spec[k] for k in ('path','sha256')};path=Path(spec['path']).resolve()
        require(within is None or path.is_relative_to(Path(within).resolve()),'Source escaped actual capture')
        require(base.digest(path)==spec['sha256'],'Changed pinned evidence')
        bound[str(path)]=spec['sha256'];return json.loads(path.read_text())
    def local(path):return pin(path)
    verify_bindings(PINS)
    value=read(evaluation_pin);review=read(review_pin);authorization=read(authorization_pin)
    require(authorization['schema']=='greenhouse.native848_scene_sweep_export_authorization.v1'
        and authorization['evaluation']==evaluation_pin and authorization['review']==review_pin
        and authorization['selected_frame_ids']==selected and authorization['exporter']==pin(__file__)
        and authorization['dataset_only'] is True and authorization['actual_capture_exit_code']==0
        and authorization['actual_annotation_exit_code']==0 and authorization['blocking_findings']==[]
        and authorization['reviewer']==review['reviewer']=='assistant_native_recovery2',
        'Exact root-reviewed completed capture, annotation and visual selection required')
    module=EVALUATORS[value['schema']]
    require(value['completed_whole_capture_authenticated'] is True and value['all_committed_frames_evaluated'] is True
        and value['frames_evaluated']==len(value['records']) and value['accepted_training_increment']==0,
        'Complete offline capture evaluation required')
    require(value['source_bindings'].get(str(Path(module.__file__).resolve()))==base.digest(module.__file__),
        'Evaluation source changed')
    verify_bindings(value['source_bindings']);verify_bindings(value['workspace_stats']['source_bindings'])
    capture=Path(value['source_capture']).resolve()
    complete,result,request,checked,context,observations=module.authenticate_capture(capture,read,local)
    require(value['capture_result']==complete['result'] and value['context']==result['context']
        and value['owner_complete']==pin(capture.parent/'owner_complete.json'),'Evaluation capture differs')
    records=module.unique(value['records'],'frame_id');observations=module.unique(observations,'sample_id')
    require(set(records)==set(observations) and set(selected)<=set(records),'Evaluation omits actual frames')
    census=read(context['census']);catalogue=read(context['catalogue']);reports=read(context['reports'])
    inventory,_=module.prepare_inventory(context,census,reports,catalogue,read(context['source_collection_plan']))
    by={r['variant_id']+'/'+r['component_id']:r for r in catalogue if r['organ_type']=='sub_stem'}
    require(len(by)==len(inventory),'Complete petiole catalogue required')
    checked_rows=[]
    for name in selected:
        r=records[name];obs=observations[name]
        require(r['unique_automated_candidate'] is True and r['background_criterion_passed'] is True
            and r['banned_source_target_instance_ids']==[],'Held frame cannot export')
        meta=read(r['metadata']);ann=read(r['annotation']);cov=read(r['coverage']);bg=read(r['background'])
        target=strict.validate_census(ann)[0]
        require(set(ann['target_census']['catalogue_petiole_ids'])==set(by),'Incomplete native petiole partition')
        for t in ann['targets']:
            c=by[t['target_id']]
            require(t['source_family']==c['source_plant_id'] and t['source_component_id']==c['component_id']
                and t['plant_instance_id']==c['variant_id']
                and t['source_target_id']==c['source_plant_id']+'/'+c['component_id'],'Relabeled native source target')
        require(target['source_target_id']!=base.BANNED and target['source_family']==r['source_family']
            and r['source_target_ids']==[target['source_target_id']],'Banned or changed source target')
        require(ann['frame_id']==meta['sample_id']==obs['sample_id']==name
            and ann['observation']==cov['observation']==pin(capture/'frames'/name/'observation.json')
            and obs['context']==cov['context']==result['context']
            and meta['calibration']==obs['calibration'] and meta['robot_snapshot']==obs['robot_snapshot']
            and ann['private_robot_context']==dict(calibration=meta['calibration'],robot_snapshot=meta['robot_snapshot']),
            'Actual sensor/robot/census lineage differs')
        require(r['rgb']==obs['files']['rgb'] and r['buffers']==obs['files']['buffers'],'Sensor substitution')
        assessment=ambiguity.validate_assessment(r['ambiguity'],r['metadata'],r['annotation'],ann)
        visual=checked_review(review,evaluation_pin,r,assessment)
        answer=base.target_answer(target,meta['calibration'])
        rgb_path=base.read_pin(r['rgb']);rgb=np.asarray(Image.open(rgb_path).convert('RGB'))
        with np.load(base.read_pin(r['buffers']),allow_pickle=False) as a:
            depth=a['depth_m'].copy();valid=a['depth_valid'].copy();ids=a['renderer_instance_id'].copy()
        base.validate_arrays(rgb,depth,valid,meta['calibration'])
        score=background.score(ids,read(obs['mapping']),catalogue,[target['plant_instance_id']],depth=depth,valid=valid)
        require(score==bg['score'] and bg['excluded_foreground_variant']==target['plant_instance_id']
            and bg['foreground_basis']=='actual_sole_strict_candidate'
            and score['background_plant_pixel_fraction']>=.40 and score['unknown_unmapped_fraction']==0,
            'Actual background quality differs')
        checked_rows.append(dict(record=r,meta=meta,ann=ann,assessment=assessment,visual=visual,target=target,
            answer=answer,rgb=rgb,rgb_bytes=rgb_path.read_bytes(),depth=depth,valid=valid))
    hashes=[hashlib.sha256(x['rgb'].tobytes()).hexdigest() for x in checked_rows]
    cameras=[(x['target']['source_family'],base.camera_signature(x['meta'],x['target']['source_family'])) for x in checked_rows]
    require(len(set(hashes))==len(hashes) and len(set(cameras))==len(cameras),'Repeated RGB or camera')
    output.mkdir();rows=[]
    try:
        for x in checked_rows:
            r=x['record'];t=x['target'];name=r['frame_id'];folder=output/'frames'/hashlib.sha256(name.encode()).hexdigest()[:24];folder.mkdir(parents=True)
            with (folder/'rgb.png').open('xb') as f:f.write(x['rgb_bytes'])
            require(base.digest(folder/'rgb.png')==r['rgb']['sha256'],'RGB bytes changed')
            for filename,key in [('depth_m.npy','depth'),('depth_valid.npy','valid')]:
                with (folder/filename).open('xb') as f:np.save(f,x[key],allow_pickle=False)
            write(folder/'calibration.json',base.sensor_calibration(x['meta']['calibration']))
            write(folder/'workspace_context.json',base.workspace_context(x['meta']))
            write(folder/'annotation_private.json',x['ann']);write(folder/'ambiguity_private.json',x['assessment']);write(folder/'review.json',x['visual'])
            def asset(n):return dict(path=(folder/n).relative_to(output).as_posix(),sha256=base.digest(folder/n))
            row=dict(schema=FRAME_SCHEMA,task_id=base.TASK,annotation_epoch=base.EPOCH,frame_id=name,split='train',
                source_family=t['source_family'],source_target_id=t['source_target_id'],plant_instance_id=t['plant_instance_id'],
                decoded_rgb_sha256=hashlib.sha256(x['rgb'].tobytes()).hexdigest(),
                conservative_camera_signature=base.camera_signature(x['meta'],t['source_family']),
                answer=dict(cut_point_uv=x['answer']['cut_point_uv']),answer_audit={k:v for k,v in x['answer'].items() if k!='cut_point_uv'},
                inputs={k:asset(n) for k,n in [('rgb','rgb.png'),('depth_m','depth_m.npy'),('depth_valid','depth_valid.npy'),('calibration','calibration.json')]},
                workspace_filter_context=asset('workspace_context.json'),private_annotation=asset('annotation_private.json'),
                independent_ambiguity_assessment=asset('ambiguity_private.json'),actual_review=asset('review.json'),
                model_instruction=base.INSTRUCTION,individual_full_frame_review=True,human_review_claimed=False,
                source_provenance=dict(evaluation=evaluation_pin,review=review_pin,**r,capture_kind=context['schema'],
                    owner_complete=value['owner_complete'],capture_result=value['capture_result'],context=value['context']),
                source_geometry_modified=False,new_independent_source_family=False)
            load_model_inputs(row,output);rows.append(row)
        with (output/'index.jsonl').open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        verify_bindings(bound);verify_bindings(value['source_bindings']);verify_bindings(PINS)
        require(pin(__file__)==authorization['exporter'],'Exporter changed during materialization')
        manifest=dict(schema=SCHEMA,task_id=base.TASK,annotation_epoch=base.EPOCH,unique_images=len(rows),
            fully_populated_greenhouse=True,plant_instance_count=144,native_resolution=[848,408],nominal_arc_m=.009,
            implementation_sha256=base.digest(__file__),source_evaluation=evaluation_pin,actual_review=review_pin,root_authorization=authorization_pin,
            source_bindings=bound,index=pin(output/'index.jsonl'),source_family_counts=dict(Counter(r['source_family'] for r in rows)),
            source_target_counts=dict(Counter(r['source_target_id'] for r in rows)),input_query=False,
            target_hint_used_to_select_answer=False,training_started=False,training_approved=False)
        write(output/'manifest.json',manifest);write(output/'result.json',dict(schema=SCHEMA,state=STATE,
            unique_images=len(rows),manifest=pin(output/'manifest.json'),training_started=False))
        (output/'README.md').write_text('Reviewed original-plant scene sweep. Use native848_unique_petiole_dataset_v8.load_model_inputs. Native848x408 RGB-D; one9mm cut point; no query cue. Each row retains its original donor TRAIN lineage.\n',encoding='utf-8')
        return manifest
    except BaseException as exc:
        write(output/'failure.json',dict(error=repr(exc),partial_output_not_admissible=True));raise
