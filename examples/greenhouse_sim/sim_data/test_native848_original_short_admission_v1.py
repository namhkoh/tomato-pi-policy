"""CPU contract tamper fixtures, not capture evidence or qualification."""
from copy import deepcopy
from pathlib import Path
import pytest
from . import native848_original_short_admission_v1 as admission


def plan_fixture(mode='production'):
    root = Path.cwd().resolve()
    return dict(schema=admission.PLAN_SCHEMA,mode=mode,profile=admission.PROFILE,
        render_budget_subframes=8,resolution=[848,408],generated_geometry_used=False,
        source_cap_reset=False,training_approved=False,selected_sample_ids=['pose_A','pose_B'],
        qualification_evidence=dict(path=str(root/'unit_only_Q.json'),sha256='a'*64),
        original_adaptation_evidence=(dict(path=str(root/'unit_only_Q2.json'),sha256='b'*64)
            if mode=='production' else None))


def audit_fixture(mode='production'):
    plan = plan_fixture(mode)
    root = Path.cwd().resolve()
    pins = {plan['qualification_evidence']['path']:'a'*64}
    def row(name,pose,role):
        sample = root/'unit_only_native'/name/'sample.json'
        pins[str(sample)]='c'*64
        pins[str(sample.parent/'inputs/rgb.png')]='d'*64
        return dict(sample_id=name,source_pose_id=pose,capture_role=role,
            sample_path=str(sample),sample_sha256='c'*64,rgb_sha256='d'*64,
            render_profile_qualified=True,profile=admission.PROFILE,render_budget_subframes=8,
            qualification_evidence_path=plan['qualification_evidence']['path'],
            qualification_evidence_sha256='a'*64)
    control_ids=['control_A1','control_B','control_A2']
    pose_ids=['pose_A','pose_B','pose_A']
    if mode=='production':
        records=[row('production_A','pose_A','production')]
        controls=[]
        pins[plan['original_adaptation_evidence']['path']]='b'*64
        evidence=plan['original_adaptation_evidence']
    else:
        controls=[row(n,p,'adaptation_control') for n,p in zip(control_ids,pose_ids)]
        records=[]
        evidence=dict(path=None,sha256=None)
    actual=dict(schema=admission.AUDIT_SCHEMA,training_approved=False,source_cap_reset=False,
        geometry_novelty_qualified=False,profile=admission.PROFILE,render_budget_subframes=8,
        source_bindings=pins,records=records,adaptation_records=controls,
        runtime_adaptation=dict(passed=True,sequence=['A','B','A'],control_sample_ids=control_ids,
            source_pose_ids=pose_ids,evidence_path=evidence['path'],evidence_sha256=evidence['sha256']))
    return plan,actual


@pytest.mark.parametrize('mode',['adaptation','production'])
def test_explicit_modes_preserve_inputs(mode):
    plan,actual=audit_fixture(mode)
    before=deepcopy((plan,actual))
    assert admission.verify_short_audit_contract(plan,actual)==plan['qualification_evidence']
    assert (plan,actual)==before
    if mode=='adaptation': assert actual['records']==[]


@pytest.mark.parametrize('mutation',['profile','budget16','budget_bool','generated','capreset',
    'mode','relative_qualification','bad_pin','training'])
def test_changed_plan_cannot_inherit_short_qualification(mutation):
    p=plan_fixture()
    if mutation=='profile':p['profile']='established56'
    elif mutation=='budget16':p['render_budget_subframes']=16
    elif mutation=='budget_bool':p['render_budget_subframes']=True
    elif mutation=='generated':p['generated_geometry_used']=True
    elif mutation=='capreset':p['source_cap_reset']=True
    elif mutation=='mode':p['mode']='automatic'
    elif mutation=='relative_qualification':p['qualification_evidence']['path']='relative.json'
    elif mutation=='bad_pin':p['qualification_evidence']['sha256']='g'*64
    else:p['training_approved']=True
    with pytest.raises(ValueError):admission.qualification_binding(p)


@pytest.mark.parametrize('mutation',['audit_schema','profile','budget','qualification_pin','adaptation_failed',
    'sequence','repeated_pose','missing_Q2_pin','Q2_hash','Q2_path','control_as_production',
    'duplicate_production','source_pose','render_false','record_budget','record_qualification',
    'sample_pin','rgb_pin','local_controls_in_production'])
def test_production_requires_bound_adaptation_and_actual_record_evidence(mutation):
    p,a=audit_fixture();r=a['records'][0]
    if mutation=='audit_schema':a['schema']='old.fixed56'
    elif mutation=='profile':a['profile']='other'
    elif mutation=='budget':a['render_budget_subframes']=16
    elif mutation=='qualification_pin':a['source_bindings'][p['qualification_evidence']['path']]='e'*64
    elif mutation=='adaptation_failed':a['runtime_adaptation']['passed']=False
    elif mutation=='sequence':a['runtime_adaptation']['sequence']=['A','A','A']
    elif mutation=='repeated_pose':a['runtime_adaptation']['source_pose_ids']=['pose_A']*3
    elif mutation=='missing_Q2_pin':del a['source_bindings'][p['original_adaptation_evidence']['path']]
    elif mutation=='Q2_hash':a['runtime_adaptation']['evidence_sha256']='e'*64
    elif mutation=='Q2_path':a['runtime_adaptation']['evidence_path']=str(Path.cwd()/'other.json')
    elif mutation=='control_as_production':r['sample_id']='control_A1';r['capture_role']='adaptation_control'
    elif mutation=='duplicate_production':a['records'].append(deepcopy(r))
    elif mutation=='source_pose':r['source_pose_id']='unknown_pose'
    elif mutation=='render_false':r['render_profile_qualified']=False
    elif mutation=='record_budget':r['render_budget_subframes']=16
    elif mutation=='record_qualification':r['qualification_evidence_sha256']='e'*64
    elif mutation=='sample_pin':del a['source_bindings'][r['sample_path']]
    elif mutation=='rgb_pin':del a['source_bindings'][str(Path(r['sample_path']).parent/'inputs/rgb.png')]
    else:a['adaptation_records']=[dict(sample_id='control_A1',capture_role='adaptation_control')]
    with pytest.raises(ValueError):admission.verify_short_audit_contract(p,a)


@pytest.mark.parametrize('mutation',['promoted_control','missing_control','wrong_pose_order','self_sealed','wrong_role'])
def test_adaptation_never_exposes_candidates(mutation):
    p,a=audit_fixture('adaptation')
    if mutation=='promoted_control':a['records']=[deepcopy(a['adaptation_records'][0])]
    elif mutation=='missing_control':a['adaptation_records'].pop()
    elif mutation=='wrong_pose_order':a['adaptation_records'][1]['source_pose_id']='pose_A'
    elif mutation=='self_sealed':a['runtime_adaptation']['evidence_path']=str(Path.cwd()/'self.json')
    else:a['adaptation_records'][0]['capture_role']='production'
    with pytest.raises(ValueError):admission.verify_short_audit_contract(p,a)


def sample_fixture():
    p,a=audit_fixture();r=a['records'][0]
    m=dict(schema_version=admission.SAMPLE_SCHEMA,sample_id=r['sample_id'],capture_role='production',
        original_geometry_only=True,generated_plant_native_pixels=0,
        direct_camera=dict(cached_sample_id=r['source_pose_id']),
        synchronization=dict(profile=admission.PROFILE,render_budget_subframes=8))
    return m,r,p,a


def test_actual_production_sample_preserves_cached_source_pose():
    admission.verify_production_sample(*sample_fixture())


@pytest.mark.parametrize('mutation',['sample_schema','control_role','wrong_pose','generated_pixels','budget'])
def test_control_or_relabelled_frame_cannot_reach_annotation(mutation):
    m,r,p,a=sample_fixture()
    if mutation=='sample_schema':m['schema_version']='old.fixed56'
    elif mutation=='control_role':m['capture_role']='adaptation_control'
    elif mutation=='wrong_pose':m['direct_camera']['cached_sample_id']='pose_B'
    elif mutation=='generated_pixels':m['generated_plant_native_pixels']=1
    else:m['synchronization']['render_budget_subframes']=56
    with pytest.raises(ValueError):admission.verify_production_sample(m,r,p,a)
