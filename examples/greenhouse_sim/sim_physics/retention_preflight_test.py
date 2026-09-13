from copy import deepcopy
import numpy as np
import pytest
from .retention_preflight import assess
from .benchmark import main


def case():
    paths=['/P/Support','/P/Branch'];frames=np.tile(np.eye(4),(2,1,1))
    contacts=[]
    for i,sign in enumerate((-1,1)):
        for x in (-.02,.02):
            normal=np.array([0.,sign,0.])
            contacts.append(dict(finger_index=i,finger_path='/F'+str(i),other_collider='/P/Branch/StemCollider',
                point_world_m=[x,sign*.003,0],normal_on_finger_world=normal.tolist(),
                impulse_on_finger_world_ns=(.001*normal).tolist()))
    r=dict(t=2.,cut=False,native_guards_passed=True,slip_m=.0001,grasp_point=[0,0,0],
        contact=dict(adapter_valid=True,bilateral=True),grasp_dynamics=dict(step_id=480,dt_s=1/240,
            model='grasp_dynamics_post_fetch_telemetry_v1',
            contact_row_contract='shaft_grasp_core_oriented_signed_normal_rows_v1',body_paths=paths,
            body_frames_world_m=frames.tolist(),finger_paths=['/F0','/F1'],contacts=contacts))
    kw=dict(step=480,time_s=2.,body_paths=paths,cut_index=1,masses=[100,.005],
        local_coms=[[0,0,0],[.01,0,0]],current_frames=frames)
    return r,kw


def test_balanced_patch_is_only_a_static_prerequisite_and_ignores_attached_mass():
    r,kw=case();before=deepcopy(r);result=assess(r,**kw)
    assert result['prerequisite_passed'] and result['detached_mass_kg']==.005
    assert result['gravity_wrench_n_nm'][2]==pytest.approx(-.04905)
    assert not result['cut_authorized'] and not result['dynamic_retention_verified']
    assert r==before


def test_inadequate_moment_does_not_increase_force_cap():
    r,kw=case();kw['local_coms'][1][0]=.8
    result=assess(r,**kw)
    assert not result['prerequisite_passed']
    assert result['capacity']['contact_budgets_n']==[.5,.5]


@pytest.mark.parametrize('fault',['step','time','dt','guards','cut','adapter','bilateral','inventory','pose','mass','com','slip','finger','normal','nan'])
def test_stale_or_invalid_native_binding_rejected(fault):
    r,kw=case()
    if fault=='step':kw['step']-=1
    elif fault=='time':kw['time_s']-=.01
    elif fault=='dt':r['grasp_dynamics']['dt_s']=1/60
    elif fault=='guards':r['native_guards_passed']=False
    elif fault=='cut':r['cut']=True
    elif fault in ('adapter','bilateral'):r['contact']['adapter_valid' if fault=='adapter' else 'bilateral']=False
    elif fault=='inventory':kw['body_paths']=['/Q/Support','/Q/Branch']
    elif fault=='pose':kw['current_frames'][1,0,3]=.001
    elif fault=='mass':kw['masses'][1]=0
    elif fault=='com':kw['local_coms']=[[0,0,0]]
    elif fault=='slip':r['slip_m']=.003
    elif fault=='finger':r['grasp_dynamics']['contacts'][0]['finger_path']='/Wrong'
    elif fault=='normal':r['grasp_dynamics']['contacts'][0]['normal_on_finger_world']=[0,2,0]
    elif fault=='nan':r['grasp_dynamics']['contacts'][0]['point_world_m']=[np.nan,0,0]
    with pytest.raises(ValueError):assess(r,**kw)


@pytest.mark.parametrize('kind',['zero','tensile','support','leaf'])
def test_noncompressive_and_unrelated_rows_never_support_retention(kind):
    r,kw=case()
    for row in r['grasp_dynamics']['contacts']:
        if kind=='zero':row['impulse_on_finger_world_ns']=[0,0,0]
        if kind=='tensile':row['impulse_on_finger_world_ns']=(-np.array(row['impulse_on_finger_world_ns'])).tolist()
        if kind=='support':row['other_collider']='/P/Support/StemCollider'
        if kind=='leaf':row['other_collider']='/P/Branch/Leaf/Mesh'
    result=assess(r,**kw)
    assert not result['prerequisite_passed'] and result['compressive_rows']==0


def test_cli_rejects_outside_isolated_trial_before_output(tmp_path):
    out=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='Retention preflight'):
        main(['--output',str(out),'--require-retention-screen'])
    assert not out.exists()
