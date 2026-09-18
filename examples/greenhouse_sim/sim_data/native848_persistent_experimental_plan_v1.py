"""Finite seven/one warmup experiment with candidate views and one paired control.

The unchanged source profile remains the optical-settings reference. Shortened
warmups are explicitly unqualified; all images still require full evaluation and
actual visual review. The paired replay is never eligible for export.
"""
from copy import deepcopy
from .dataset_review import require
from .native848_bulk_plan_v1 import check_profile

PROFILE_SCHEMA='greenhouse.native848_persistent_experimental_profile.v1'
ACTUAL_SCHEMA='greenhouse.native848_persistent_segment_optical_experiment.v1'
COUNTS=(7,1,1,1)
PROFILE_ROLE='source_optical_settings_and_baseline_warmup_reference_only'
CANDIDATE='experimental_candidate'
CONTROL='paired_control_export_prohibited'


def make_profile(source_profile):
    return dict(schema=PROFILE_SCHEMA,source_optical_profile=source_profile,
        warmup_requests_per_segment=list(COUNTS),warmup_subframes=8,production_subframes=8,
        max_segments=4,max_frames=12,max_candidate_frames=11,paired_segment_indices=[[0,3]],
        shortened_warmup_qualified=False,experimental_profile=True,
        training_approved=False,accepted_training_increment=0)


def actual_profile(segment,source_profile_pin,source_profile):
    check_profile(source_profile)
    require(source_profile['warmup_steps']==[8]*7 and source_profile['request_subframes']==8,
        'Unchanged7x8 source baseline and production8 required')
    index=int(segment['segment_id'][1:])
    require(0<=index<4 and segment['segment_id']==f's{index:03d}'
        and type(segment['warmup_request_count']) is int and segment['warmup_request_count']==COUNTS[index],
        'Finite experimental segment warmup schedule differs')
    return dict(schema=ACTUAL_SCHEMA,source_optical_profile=deepcopy(source_profile_pin),
        warmup_steps=[8]*COUNTS[index],request_subframes=8,
        delta_time_seconds=source_profile['delta_time_seconds'],render_settings=deepcopy(source_profile['render_settings']),
        reset_before_production=True,shortened_warmup_qualified=False,experimental_profile=True,
        training_approved=False,accepted_training_increment=0)


def validate_schedule(request,source_candidates):
    specs=request['segments'];candidates=[x[1] for x in source_candidates]
    require(request['experimental_profile'] is True and request['training_approved'] is False
        and request['accepted_training_increment']==0 and len(specs)==len(candidates)==4,
        'Exactly four finite experimental segments required')
    require(request['experimental_profile_policy']==make_profile(candidates[0]['profile']),
        'Exact explicit unqualified experimental profile required')
    require(specs[0]['source_request']==specs[3]['source_request']
        and len({(s['source_request']['path'],s['source_request']['sha256']) for s in specs[:3]})==3,
        'Three candidate sources followed by first-source replay required')
    require(len({(c['manifest']['path'],c['manifest']['sha256']) for c in candidates[:3]})==3,
        'Three distinct generated shapes required')
    total=0
    for i,(spec,(source,candidate)) in enumerate(zip(specs,source_candidates)):
        require(set(spec)=={'segment_id','source_request','selected_sample_ids','warmup_request_count','purpose','paired_control_reference'}
            and spec['segment_id']==f's{i:03d}' and type(spec['warmup_request_count']) is int
            and spec['warmup_request_count']==COUNTS[i], 'Canonical finite experimental segment required')
        ids=spec['selected_sample_ids'];available=[r['sample_id'] for r in candidate['records']]
        require(type(ids) is list and 1<=len(ids)<=4 and len(ids)==len(set(ids))
            and len(available)==len(set(available)) and set(ids)<=set(available)
            and len(ids)<=source['max_frames'], 'Finite unique selected source cameras required')
        require(ids==[name for name in available if name in ids],'Preserve exact source camera order')
        if i<3:
            require(spec['purpose']==CANDIDATE and spec['paired_control_reference'] is None,
                'Candidate segment purpose differs')
        else:
            require(spec['purpose']==CONTROL and ids==specs[0]['selected_sample_ids'][:1]
                and spec['paired_control_reference']==dict(segment_id='s000',sample_id=ids[0]),
                'Last segment must replay the exact first candidate camera, export prohibited')
        total+=len(ids)
    require(type(request['max_frames']) is int and request['max_frames']==total<=12
        and request['potential_candidate_frames']==total-1<=11,'Bounded frame totals differ')


def validate_evidence(segment,actual,context,value,observations):
    count=COUNTS[int(segment['segment_id'][1:])];ids=segment['selected_sample_ids']
    observed=[o['sample_id'] for o in observations]
    require(actual['schema']==ACTUAL_SCHEMA and actual['warmup_steps']==[8]*count
        and actual['request_subframes']==8 and actual['reset_before_production'] is True
        and actual['shortened_warmup_qualified'] is False
        and actual['source_optical_profile']==context['profile'] and context['profile_role']==PROFILE_ROLE
        and context['actual_experimental_profile']==value['actual_experimental_profile']==actual
        and value['selected_experimental_sample_ids']==context['selected_experimental_sample_ids']==ids
        and observed==[name for name in ids if name in observed] and len(observed)==len(set(observed)),
        'Actual experimental settings or selected cameras differ')
    for item in (context,value,*observations):
        require(item['capture_purpose']==segment['purpose']
            and item['paired_control_reference']==segment['paired_control_reference']
            and item['export_prohibited']==(segment['purpose']==CONTROL),
            'Candidate/control authority differs; paired control cannot be exported')
    for item in (actual,context,value,*observations):
        require(item['experimental_profile'] is True and item['training_approved'] is False
            and item['accepted_training_increment']==0,'Unreviewed experiment cannot approve training')
