"""Diagnostic A7/B7/A1/B1: source optics unchanged, explicit actual warmup policy."""
from copy import deepcopy
from .dataset_review import require
from .native848_bulk_plan_v1 import check_profile

PROFILE_SCHEMA='greenhouse.native848_persistent_shortwarm_profile.v1'
ACTUAL_SCHEMA='greenhouse.native848_persistent_segment_optical_diagnostic.v1'
COUNTS=(7,7,1,1)
PROFILE_ROLE='source_optical_settings_and_baseline_warmup_reference_only'


def make_profile(source_profile):
    return dict(schema=PROFILE_SCHEMA,source_optical_profile=source_profile,
        warmup_requests_per_segment=list(COUNTS),warmup_subframes=8,production_subframes=8,
        frames_per_segment=1,paired_segment_indices=[[0,2],[1,3]],
        diagnostic_only=True,training_approved=False,accepted_training_increment=0)


def validate_profile(profile,source_profile):
    require(profile==make_profile(source_profile),'Exact bounded diagnostic warmup profile required')


def actual_profile(segment,source_profile_pin,source_profile):
    check_profile(source_profile)
    require(source_profile['warmup_steps']==[8]*7 and source_profile['request_subframes']==8,
        'Diagnostic requires the unchanged7x8 baseline and production8')
    index=int(segment['segment_id'][1:])
    require(0<=index<4 and segment['segment_id']==f's{index:03d}'
        and type(segment['warmup_request_count']) is int and segment['warmup_request_count']==COUNTS[index],
        'Diagnostic segment warmup schedule differs')
    return dict(schema=ACTUAL_SCHEMA,source_optical_profile=deepcopy(source_profile_pin),
        warmup_steps=[8]*COUNTS[index],request_subframes=8,
        delta_time_seconds=source_profile['delta_time_seconds'],
        render_settings=deepcopy(source_profile['render_settings']),
        reset_before_production=True,diagnostic_only=True,training_approved=False,accepted_training_increment=0)


def validate_schedule(request,source_candidates):
    require(request['diagnostic_only'] is True and request['training_approved'] is False
        and request['accepted_training_increment']==0 and request['max_frames']==4
        and len(request['segments'])==len(source_candidates)==4,'Exactly4 diagnostic segments required')
    specs=request['segments'];candidates=[x[1] for x in source_candidates]
    require(specs[0]['source_request']==specs[2]['source_request']
        and specs[1]['source_request']==specs[3]['source_request']
        and specs[0]['source_request']!=specs[1]['source_request'], 'Exact A-B-A-B source pairing required')
    require(candidates[0]['manifest']!=candidates[1]['manifest'],'Two distinct generated shapes required')
    validate_profile(request['diagnostic_profile'],candidates[0]['profile'])
    for i,(spec,(source,candidate)) in enumerate(zip(specs,source_candidates)):
        require(set(spec)=={'segment_id','source_request','selected_sample_id','warmup_request_count'}
            and spec['segment_id']==f's{i:03d}' and type(spec['warmup_request_count']) is int
            and spec['warmup_request_count']==COUNTS[i], 'Canonical diagnostic segment fields required')
        ids=[r['sample_id'] for r in candidate['records']]
        require(source['max_frames']>=1 and len(ids)==len(set(ids)) and spec['selected_sample_id'] in ids,
            'Diagnostic selected sample outside source candidate schedule')
        if i>=2:require(spec['selected_sample_id']==specs[i-2]['selected_sample_id'],
                        'Paired morphology must use the exact same source camera record')


def validate_evidence(segment,actual,context,value,observations):
    """Require separately typed actual settings, never relabel the baseline profile."""
    count=COUNTS[int(segment['segment_id'][1:])]
    require(actual['schema']==ACTUAL_SCHEMA and actual['warmup_steps']==[8]*count
        and actual['request_subframes']==8 and actual['reset_before_production'] is True
        and actual['source_optical_profile']==context['profile']
        and context['profile_role']==PROFILE_ROLE
        and context['actual_diagnostic_profile']==value['actual_diagnostic_profile']==actual
        and value['selected_diagnostic_sample_id']==context['selected_diagnostic_sample_id']==segment['selected_sample_id']
        and len(observations)<=1
        and all(o['sample_id']==segment['selected_sample_id'] for o in observations),
        'Diagnostic actual optical policy or selected camera differs')
    for item in (actual,context,value,*observations):
        require(item['diagnostic_only'] is True and item['training_approved'] is False
            and item['accepted_training_increment']==0,'Diagnostic evidence cannot approve training')

