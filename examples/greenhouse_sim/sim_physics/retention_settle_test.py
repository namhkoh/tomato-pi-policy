from copy import deepcopy
import pytest
from .retention_settle import PreloadSettle


def record(step,support=.24):
    return dict(t=step/240,cut=False,native_guards_passed=True,slip_m=.0001,
        contact=dict(step_id=step,adapter_valid=True,bilateral=True,stem_only=True,
            compressive_support_n=[support,support]),
        force_closure=dict(load_profile='retention_preload_v1',desired_support_n=.24,
            symmetric_aperture_command=True),
        grasp_dynamics=dict(step_id=step,dt_s=1/240,model='grasp_dynamics_post_fetch_telemetry_v1',
            finger_velocities_m_s=[0.,0.]))


def observe(s,step,**kw):
    r=record(step,**kw)
    return s.observe(r,step=step,time_s=step/240,start_time_s=4.)


def test_first_touch_cannot_substitute_for_preload_then_full_measured_dwell():
    s=PreloadSettle()
    for step in range(961,1001):assert observe(s,step,support=.03)['state']=='waiting'
    for step in range(1001,1048):assert observe(s,step)['state']=='waiting'
    ready=observe(s,1048)
    assert ready['state']=='ready' and ready['consecutive_stable_steps']==48
    assert not ready['retention_verified'] and not ready['cut_authorized'] and not ready['force_limits_changed']


@pytest.mark.parametrize('fault',['weak','excess','motion','bilateral','stem'])
def test_nonsettled_sample_resets_dwell_without_commanding_extra_force(fault):
    s=PreloadSettle()
    for step in range(961,1000):observe(s,step)
    r=record(1000)
    if fault in ('weak','excess'):r['contact']['compressive_support_n'][0]=.03 if fault=='weak' else .32
    if fault=='motion':r['grasp_dynamics']['finger_velocities_m_s'][0]=.00201
    if fault=='bilateral':r['contact']['bilateral']=False
    if fault=='stem':r['contact']['stem_only']=False
    before=deepcopy(r)
    result=s.observe(r,step=1000,time_s=1000/240,start_time_s=4.)
    assert result['state']=='waiting' and result['consecutive_stable_steps']==0 and r==before


def test_unsettled_grasp_cannot_extend_the_existing_acquisition_deadline():
    s=PreloadSettle()
    for step in range(961,3240):assert observe(s,step,support=.03)['state']=='waiting'
    assert observe(s,3240,support=.03)['state']=='timeout'


@pytest.mark.parametrize('fault',['clock','contact_step','telemetry','dt','guards','cut','adapter','profile','symmetric','nan','slip'])
def test_unbound_or_unsafe_observation_fails_closed(fault):
    s=PreloadSettle();r=record(961)
    if fault=='clock':r['t']+=.01
    if fault=='contact_step':r['contact']['step_id']-=1
    if fault=='telemetry':r['grasp_dynamics']['model']='unknown'
    if fault=='dt':r['grasp_dynamics']['dt_s']=1/60
    if fault=='guards':r['native_guards_passed']=False
    if fault=='cut':r['cut']=True
    if fault=='adapter':r['contact']['adapter_valid']=False
    if fault=='profile':r['force_closure']['desired_support_n']=.5
    if fault=='symmetric':r['force_closure']['symmetric_aperture_command']=False
    if fault=='nan':r['contact']['compressive_support_n'][0]=float('nan')
    if fault=='slip':r['slip_m']=.003
    with pytest.raises(ValueError):s.observe(r,step=961,time_s=961/240,start_time_s=4.)


def test_skipped_repeated_or_changed_start_time_rejected():
    for step,start in [(961,4.),(963,4.),(962,3.9)]:
        s=PreloadSettle();observe(s,961)
        with pytest.raises(ValueError):s.observe(record(step),step=step,time_s=step/240,start_time_s=start)


def test_delay_rebuilds_full_original_schedule_without_jump_to_stroke():
    from .bimanual_probe import schedule_after_grasp
    before=schedule_after_grasp(6.7625,0);after=schedule_after_grasp(8.1,0)
    assert after['approach']-8.1==pytest.approx(.5)
    for first,last in [('approach','stroke'),('stroke','end')]:
        assert after[last]-after[first]==pytest.approx(before[last]-before[first])


def test_cli_rejects_missing_original_profile_before_output(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Preload settling'):
        main(['--output',str(tmp_path/'unused'),'--settle-retention-preload'])
    assert not (tmp_path/'unused').exists()


def test_settle_tracks_the_existing_symmetric_mean_not_two_independent_targets():
    s=PreloadSettle()
    for step in range(961,1009):
        r=record(step);r['contact']['compressive_support_n']=[.20,.24]
        receipt=s.observe(r,step=step,time_s=step/240,start_time_s=4.)
    assert receipt['state']=='ready' and receipt['regulated_quantity']=='mean_support_original_symmetric_controller'
