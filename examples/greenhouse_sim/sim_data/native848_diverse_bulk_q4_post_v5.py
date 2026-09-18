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
from . import native848_q4_group_margin_v1 as group_margin
from . import native848_bulk_delivery_v1 as delivery
from . import native848_data_roots_v1 as roots
from .native848_diverse_q3_post_admission_v1 import artifact,bound_json
ROOT=Path('D:/research/tomato-pi-policy')
SCHEMA='greenhouse.native848_diverse_bulk_q4_background_post.v5'
STATE='fresh_original_bulk_background_exception_pending_group_or_individual_review'
SELECTOR_SHA='ec8ac3900457ee4358ad62da0ab4743e67f5731e8f1405bb073c813869efd7a8'
ADAPTERS={
 'greenhouse.native848_diverse_bulk_q3_adapter.v5':('native848_diverse_bulk_adapter_v5.py','04b997f71abfcd75402c1a64bc1f9570c75dc0a0f761e6c32b11e554cdad6759','original_bulk_q3_normalized_pending_group_or_individual_review'),
 'greenhouse.native848_diverse_bulk_q3_adapter.v4':('native848_diverse_bulk_adapter_v4.py','fd244adc37b6170ad5e29417e0d43ac1f18708c7d946f04e6d0fd68632cd9e4b','original_bulk_q3_normalized_pending_group_or_individual_review'),
 'greenhouse.native848_diverse_bulk_q3_adapter.v1':('native848_diverse_bulk_adapter_v1.py','cb43398abec73122d570589de3600c9ca6081a57c16eeb00132169e86043addc','original_bulk_q3_normalized_pending_individual_review'),
 'greenhouse.native848_diverse_bulk_q3_adapter.v2':('native848_diverse_bulk_adapter_v2.py','123a266a802e0866a4314838b2672e92a0f847b0c51145b91988e0e0d4c74262','original_bulk_q3_normalized_pending_group_or_individual_review'),
 'greenhouse.native848_diverse_bulk_q3_adapter.v3':('native848_diverse_bulk_adapter_v3.py','3a71e289dd96a2104f146b9e3f965fa8e0a9003f8783f41d8cfdb0e485354512','original_bulk_q3_normalized_pending_group_or_individual_review'),
}


def validate_adapter(original,pins):
    require(original['schema'] in ADAPTERS,'Unsupported genuine Q3 adapter')
    name,expected,state=ADAPTERS[original['schema']]
    path=Path(__file__).with_name(name).resolve()
    require(original['state']==state and original['implementation']==dict(path=str(path),sha256=expected)
        and original['input_route']=='bulk_q3' and original['training_approved'] is False,
        'Genuine exact-source Q3 normalization required')
    for source,h in original['source_bindings'].items():
        require(source not in pins or pins[source]==h,'Conflicting original normalization binding')
        pins[source]=h
    require(pins.get(str(path))==expected,'Unbound Q3 adapter implementation')
    artifact(path,pins,expected)
    parallel=original['schema'].endswith(('.v4','.v5'))
    multi=parallel or original['schema'].endswith('.v3')
    if original['schema'].endswith('.v5'):
        sources=[(ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_native848_multianchor_parallel_v2.py',
            'bd0e3d54ef08f7189366f845962e7deda1ff45e29aa6de3f17dbf7dbe06972fa'),
            (Path(__file__).with_name('native848_multianchor_sibling_worker_v2.py'),'5abb477b94f8d32ff1f52969e38fd1a102b1d9956bb0c245147f11d517abe793'),
            (Path(__file__).with_name('native848_multianchor_parallel_predecessor_v2.py'),'b2c9e49a2afb8a951064ad1bbbe66ed4039f14f34e5e27b5c2dcebf0aebe17ee'),
            (Path(__file__).with_name('native848_bulk_sibling_gate_v5.py'),'3766afb02212dfa0bb432bb61c6dc9b3e45d7fbd259d778be6e7bc2b9a7150d4'),
            (Path(__file__).with_name('native848_storage_environment_v1.py'),'826f3e3c85e886d9c6b4e21fbdf8164d9eb3e1b50c395fb91c3ef90ed456667b'),
            (ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_bulk_cpu_v6.py','54d077837ece88c6612bdef5b2b10076c167b7939daa73e6b251f8be4df8e955')]
        sources += [(Path(p),h) for p,h in roots.bindings().items()]
    elif parallel:
        sources=[(ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_native848_multianchor_parallel_v1.py',
            '526f6cd95bc405b72c4e2f14af48df989d57e029f7e2efd87518ccd562df2566'),
            (Path(__file__).with_name('native848_multianchor_sibling_worker_v1.py'),
            '160089f6fb87628ca7d56962cb2552eee3fc4be22eaa75c9569190aa88311485'),
            (Path(__file__).with_name('native848_multianchor_parallel_predecessor_v1.py'),
            'd906cbedbd909be5d1981992d1db326680998e17c8fb1493a0c57504ae13bc9e'),
            (Path(__file__).with_name('native848_bulk_sibling_gate_v5.py'),
            '3766afb02212dfa0bb432bb61c6dc9b3e45d7fbd259d778be6e7bc2b9a7150d4')]
    if parallel:
        require(original['native_source_route']=='completed_parallel_multianchor_slot','Parallel normalization scope differs')
        for key in ('parallel_parent_result','parallel_parent_complete','parallel_parent_intent','parallel_slot_ticket','parallel_slot_release'):
            spec=original[key];require(pins.get(str(Path(spec['path']).resolve()))==spec['sha256'],'Unbound genuine parent/slot evidence')
            artifact(spec['path'],pins,spec['sha256'])
    else:
        owner_name='run_native848_multianchor_v1.py' if multi else 'run_native848_bulk_v2.py'
        owner_sha=('c92ec4cd9409dfbaed657bd5344ea9cd78380deb1c7cc77c62c135ebea1237b8' if multi
                   else '8e9131cc92a64d2e6c83ba7a5e0d52c8a1dd4f8b81adb6b2f0ede3cf6fa5bf0c')
        worker_name='native848_multianchor_worker_v1.py' if multi else 'native848_bulk_worker_v1.py'
        worker_sha=('261a7f7374de738a17edd0235637dfb0501ca56905782d4127863b4b98164fae' if multi
                    else '5306fe59b579aa8e7bfc021ac39639ef7e0e62b06a851b12a6ab79a96daa204a')
        sources=[(ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'/owner_name,owner_sha),
                 (Path(__file__).with_name(worker_name),worker_sha)]
    if multi:sources.append((Path(__file__).with_name('native848_multianchor_plan_v1.py'),
        '5fc1e15bd493435fb2db50f11ab9a5c67c82d52daaae7caa3d7b31244f61e31c'))
    for p,h in sources:
        p=p.resolve();require(pins.get(str(p))==h,'Unbound source owner/worker/plan chain');artifact(p,pins,h)
    return dict(schema=original['schema'],implementation=dict(path=str(path),sha256=expected))



def run(q3_result,*,q3_sha256,hold_ledger,output,policy_review=None,policy_review_sha256=None):
    output=roots.diagnostic(output);require(not output.exists(),'New Q4 output required')
    pins={};original=bound_json(q3_result,pins,q3_sha256)
    artifact(roots.__file__,pins,'b3b5d6c77385b2776a6997dd7dbe14f1aae8e1d093a45140d0f2c033fc2c8d83')
    for p,h in roots.bindings().items():artifact(p,pins,h)
    adapter_dispatch=validate_adapter(original,pins)
    artifact(selector.__file__,pins,SELECTOR_SHA)
    artifact(group_margin.__file__,pins,'c815d6e11fc208a2de427f2662a4b5a539fdc9e54a6f1a36504c9141784149ea')
    user_policy=artifact(ROOT/'data/sim_data/diagnostics/native848_group_review_user_policy_20260917_v1/policy.json',pins)
    authorized=read_json(user_policy['path'])
    require(authorized['preserve_per_image_automated_gates'] is True
        and authorized['group_representatives_and_flagged_visual_review_required'] is True
        and authorized['individual_review_of_every_image_required'] is False,'Exact user group-review authorization required')
    predeclaration=artifact(ROOT/'data/sim_data/diagnostics/native848_q4_group_sampling_predeclaration_20260917_v1/result.json',pins,
        '9f83885858a7a8a3a69488c3881c364252f43ab705835e431282c9a2e6fd0ab2')
    artifact(delivery.__file__,pins)
    for name,pin in [('native848_query_selection_v3.py','69bc1babbe462d7dee107c1a70832faaa688579cebe3493928119e98a7b5a462'),
                     ('native848_query_selection_v1.py','896c81e5900ed31eb5a52989e15517a47d367b9aacb452bd8580e47260fe3964'),
                     ('native848_diverse_q3_post_admission_v1.py','9e9dc67df6addc2f33ef02665e57763b35066c3d827cbb5353a8ccb03ea6c3a7')]:
        artifact(Path(__file__).with_name(name),pins,pin)
    artifact(__file__,pins)
    trial=Path(original['native_result']['path']).resolve().parent
    require(roots.fresh_native(trial)==trial,
        'Only authorized fresh 20260917 diverse-pilot native frames may use Q4')
    require(original['input_route']=='bulk_q3','No historical short/direct reannotation')
    root_review=None
    if policy_review is not None:
        value=bound_json(policy_review,pins,policy_review_sha256)
        require(value['selector_sha256']==sha256(selector.__file__) and value['post_wrapper_sha256']==sha256(__file__)
            and value['background_exception_policy_approved'] is True and value['training_approved'] is False
            and value['blocking_findings']==[] and value['group_Q4_review_policy_approved'] is True
            and value['group_margin_sha256']==sha256(group_margin.__file__)
            and value['sampling_predeclaration']==predeclaration,'Independent exact-source group Q4 policy review required')
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
        if passed:
            row.update(q4_review_margins=group_margin.summarize(proof),q4_group_review_authorized=True,
                effective_visual_review_requirement='target_view_group_representatives_and_all_flagged',
                legacy_selector_individual_hint_superseded=True,
                decision='conditional_background_candidate_pending_actual_group_or_individual_review')
        (candidates if passed else excluded).append(row)
    require(sha256(ledger)==ledger_hash,'Hold ledger changed during conditional screening')
    require(len(candidates)+len(excluded)==len(original['excluded_records']),'Lost original Q3 held population')
    verify_bindings(pins)
    primary={k:deepcopy(original[k]) for k in ['original_admission','native_result','owner_complete','cpu_owner_result','native_owned_exit','context','manifest']}
    for key in ('parallel_parent_result','parallel_parent_complete','parallel_parent_intent','parallel_slot_ticket','parallel_slot_release'):
        if key in original:primary[key]=deepcopy(original[key])
    result=dict(schema=SCHEMA,state=STATE,input_route='bulk_q4',q3_normalization=artifact(q3_result,pins,q3_sha256),
        implementation=artifact(__file__,pins),selector=artifact(selector.__file__,pins),root_policy_review=root_review,
        root_policy_review_required_before_export=True,**primary,annotation_epoch=selector.ANNOTATION_EPOCH,
        policy=selector.POLICY,source_adapter_dispatch=adapter_dispatch,source_bindings=pins,records=candidates,excluded_records=excluded,
        unchanged_q3_candidates=q3_unchanged,preserved_hold_snapshot=hold_snapshot,
        original_hold_ledger_path=str(ledger),current_holds_rechecked_at_export=True,
        candidate_count=len(candidates),excluded_count=len(excluded),original_q3_hold_population=len(original['excluded_records']),
        q3_outputs_unchanged=True,manual_or_historical_holds_revived=False,
        individual_visual_review=False,all_candidates_require_individual_visual_review=False,
        authorized_group_or_individual_review_required=True,review_policy=user_policy,
        group_margin_implementation=artifact(group_margin.__file__,pins),sampling_predeclaration=predeclaration,
        effective_visual_review_contract=dict(schema='greenhouse.native848_Q4_group_review_contract.v1',
            user_policy=user_policy,sampling_policy=group_margin.POLICY,
            legacy_selector_individual_hint_superseded_for_new_group_post_only=True,
            label_and_query_bytes_unchanged=True,all_flagged_require_individual_review=True,
            any_bad_representative_holds_whole_group=True,unsampled_individual_visual_review=False),
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

