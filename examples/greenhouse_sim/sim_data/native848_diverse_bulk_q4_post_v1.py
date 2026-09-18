"""Apply a separate conditional Q4 background policy to fresh bulk Q3 holds."""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json
import numpy as np
from PIL import Image
from .dataset_review import require,read_json,write_json,verify_bindings
from .depth_preview import sha256
from .capture_visibility import component_masks
from . import native848_query_selection_v4 as selector
from . import native848_bulk_delivery_v1 as delivery
from .native848_diverse_q3_post_admission_v1 import artifact,bound_json
ROOT=Path('D:/research/tomato-pi-policy')
SCHEMA='greenhouse.native848_diverse_bulk_q4_background_post.v1'
STATE='fresh_original_bulk_background_exception_pending_individual_review'
ADAPTER_SHA='cb43398abec73122d570589de3600c9ca6081a57c16eeb00132169e86043addc'


def run(q3_result,*,q3_sha256,hold_ledger,output,policy_review=None,policy_review_sha256=None):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),'New Q4 output required')
    pins={};original=bound_json(q3_result,pins,q3_sha256)
    require(original['schema']=='greenhouse.native848_diverse_bulk_q3_adapter.v1'
        and original['state']=='original_bulk_q3_normalized_pending_individual_review'
        and original['implementation']['sha256']==ADAPTER_SHA
        and original['training_approved'] is False,'Genuine frozen Q3 bulk normalization required')
    pins.update(original['source_bindings']);artifact(original['implementation']['path'],pins,ADAPTER_SHA)
    for module in [selector,delivery]:artifact(module.__file__,pins)
    for name,pin in [('native848_query_selection_v3.py','69bc1babbe462d7dee107c1a70832faaa688579cebe3493928119e98a7b5a462'),
                     ('native848_query_selection_v1.py','896c81e5900ed31eb5a52989e15517a47d367b9aacb452bd8580e47260fe3964'),
                     ('native848_diverse_q3_post_admission_v1.py','9e9dc67df6addc2f33ef02665e57763b35066c3d827cbb5353a8ccb03ea6c3a7')]:
        artifact(Path(__file__).with_name(name),pins,pin)
    artifact(__file__,pins)
    trial=Path(original['native_result']['path']).resolve().parent
    require(trial.is_relative_to(ROOT/'data/sim_data/diagnostics/native848_diverse_noon_native_20260917_v1'),
        'Only authorized fresh 20260917 diverse-pilot native frames may use Q4')
    require(original['input_route']=='bulk_q3','No historical short/direct reannotation')
    root_review=None
    if policy_review is not None:
        value=bound_json(policy_review,pins,policy_review_sha256)
        require(value['selector_sha256']==sha256(selector.__file__) and value['post_wrapper_sha256']==sha256(__file__)
            and value['background_exception_policy_approved'] is True and value['training_approved'] is False
            and value['blocking_findings']==[],'Independent exact-source policy review required')
        root_review=artifact(policy_review,pins,policy_review_sha256)
    else:require(policy_review_sha256 is None,'Review pin without receipt')
    ledger=Path(hold_ledger).resolve();raw=ledger.read_bytes();ledger_hash=hashlib.sha256(raw).hexdigest()
    require(not raw or raw.endswith(b'\n'),'Incomplete hold ledger')
    holds=[json.loads(x) for x in raw.splitlines() if x];held=set()
    for row in holds:held.update(delivery.aliases(row))
    context=bound_json(original['context']['path'],pins,original['context']['sha256'])
    reports=bound_json(context['source_reports_path'],pins,context['source_reports_sha256'])
    report=next(r for r in reports if r['plant_id']==context['source_family'])
    catalogue=bound_json(context['catalogue_path'],pins,context['catalogue_sha256'])
    verify_bindings(pins);output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'preserved_visual_holds.snapshot.jsonl'
    with snapshot.open('xb') as stream:stream.write(raw)
    hold_snapshot=artifact(snapshot,pins,ledger_hash)
    candidates=[];excluded=[];q3_unchanged=deepcopy(original['records'])
    for prior in original['excluded_records']:
        row=deepcopy(prior);name=row['sample_id']
        row.update(input_route='bulk_q4',q4_passed=False,Q1passed=False,protected_foreground_passed=False,
            conditional_background_passed=False,background_exception_applied=False,
            candidate_for_individual_visual_review=False,individual_visual_review=False,training_approved=False,
            original_q3_decision=prior['decision'])
        if prior.get('original_decision')!='hold_query_trace_or_route_separation':
            row['decision']='preserved_non_exception_'+prior['decision'];excluded.append(row);continue
        require(prior['q3_passed'] is False and prior['workspace_passed'] is True
            and prior['local_clarity_passed'] is True and prior['render_profile_qualified'] is True,
            'Exception may not relax original local/workspace/render checks')
        if delivery.aliases(prior)&held:
            row['decision']='preserved_manual_or_historical_hold';excluded.append(row);continue
        assets=row['artifacts'];saved={}
        for key in ['sample','label','query_trace','query_selection','mapping']:
            saved[key]=bound_json(assets[key]['path'],pins,assets[key]['sha256'])
        rgb=np.asarray(Image.open(assets['rgb']['path']).convert('RGB'))
        with np.load(assets['buffers']['path'],allow_pickle=False) as buffers:
            require(set(buffers.files)=={'depth_m','renderer_instance_id','depth_valid','rgba_alpha'},'Exact native buffers required')
            depth=buffers['depth_m'];ids=buffers['renderer_instance_id'];valid=buffers['depth_valid']
        mapping={int(k):v for k,v in saved['mapping']['renderer_id_to_prim'].items()}
        components,_,_=component_masks(ids,mapping,catalogue)
        with Image.open(assets['target_mask']['path']) as mask_image:mask=np.asarray(mask_image).copy()
        label,trace,selection=selector.annotate_v4(saved['sample'],report,rgb,depth,valid,components,catalogue,
            target_mask=mask,renderer_ids=ids,renderer_mapping=mapping,
            expected_q3_label=saved['label'],expected_q3_trace=saved['query_trace'],expected_q3_selection=saved['query_selection'])
        for key in ['label','query_trace','query_selection']:assets['original_q3_'+key]=deepcopy(assets[key])
        dest=output/name;dest.mkdir()
        for key,value in [('label',label),('query_trace',trace),('query_selection',selection)]:
            path=dest/(key+'.json');write_json(path,value);assets[key]=artifact(path,pins)
            row[key+'_path']=str(path);row[key+'_sha256']=assets[key]['sha256']
        chosen=selection['selected_query'];proof=(trace or {}).get('conditional_background');passed=bool(trace and trace['passed'])
        require(not passed or (chosen['both_passed'] is True and proof['Q1passed'] is True
            and proof['protected_foreground_passed'] is True and proof['conditional_background_passed'] is True
            and proof['q3_passed'] is False and proof['background_exception_applied'] is True),
            'Q4 condition bypassed preserved gates or changed Q3 result')
        row.update(annotation_epoch=selector.ANNOTATION_EPOCH,q3_passed=False,q4_passed=passed,
            Q1passed=bool(passed and proof['Q1passed']),protected_foreground_passed=bool(passed and proof['protected_foreground_passed']),
            conditional_background_passed=bool(passed and proof['conditional_background_passed']),
            background_exception_applied=bool(passed and proof['background_exception_applied']),
            full_trace_and_anchored_grid_passed=bool(passed),candidate_for_individual_visual_review=passed,
            prior_candidate_for_individual_visual_review=False,
            decision='conditional_background_candidate_pending_actual_individual_review' if passed else 'preserved_q3_hold_q4_also_failed',
            original_q3_artifacts={k:assets['original_q3_'+k] for k in ['label','query_trace','query_selection']},
            original_q3_label_sha256=assets['original_q3_label']['sha256'])
        (candidates if passed else excluded).append(row)
    require(sha256(ledger)==ledger_hash,'Hold ledger changed during conditional screening')
    require(len(candidates)+len(excluded)==len(original['excluded_records']),'Lost original Q3 held population')
    verify_bindings(pins)
    primary={k:deepcopy(original[k]) for k in ['original_admission','native_result','owner_complete','cpu_owner_result','native_owned_exit','context','manifest']}
    result=dict(schema=SCHEMA,state=STATE,input_route='bulk_q4',q3_normalization=artifact(q3_result,pins,q3_sha256),
        implementation=artifact(__file__,pins),selector=artifact(selector.__file__,pins),root_policy_review=root_review,
        root_policy_review_required_before_export=True,**primary,annotation_epoch=selector.ANNOTATION_EPOCH,
        policy=selector.POLICY,source_bindings=pins,records=candidates,excluded_records=excluded,
        unchanged_q3_candidates=q3_unchanged,preserved_hold_snapshot=hold_snapshot,
        original_hold_ledger_path=str(ledger),current_holds_rechecked_at_export=True,
        candidate_count=len(candidates),excluded_count=len(excluded),original_q3_hold_population=len(original['excluded_records']),
        q3_outputs_unchanged=True,manual_or_historical_holds_revived=False,
        individual_visual_review=False,all_candidates_require_individual_visual_review=True,
        training_approved=False,accepted_training_increment=0)
    write_json(output/'result.json',result)
    print('CONDITIONAL_BACKGROUND_Q4',len(candidates),len(excluded),bool(root_review))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['q3-result','q3-sha256','hold-ledger','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--policy-review');p.add_argument('--policy-review-sha256');a=p.parse_args()
    run(a.q3_result,q3_sha256=a.q3_sha256,hold_ledger=a.hold_ledger,output=a.output,
        policy_review=a.policy_review,policy_review_sha256=a.policy_review_sha256)

