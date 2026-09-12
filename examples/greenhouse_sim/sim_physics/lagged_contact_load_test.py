import numpy as np
import pytest
from .lagged_contact_load import LaggedContactLoad, generalized_contact_load


def binding():
    jac=np.zeros((2,6,8));jac[0,:,:6]=np.eye(6);jac[1,:,:6]=np.eye(6)
    jac[1,1,5]=2; jac[1,2,4]=-2  # root rotation moves the displaced second COM
    jac[0,5,6]=1; jac[1,5,7]=1
    return dict(collider_bodies={'/A':0,'/B':1},body_com_world=[[0,0,0],[2,0,0]],com_jacobians=jac)


def row(a='/A',b='/Outside',kind='normal'):
    return dict(collider0=a,collider1=b,kind=kind,point_world_m=[1,0,0],impulse_on_0_ns=[0,.001,0])


def test_signed_normal_and_friction_virtual_work_and_internal_pair():
    x=generalized_contact_load([row(),row(kind='friction')],dt=.001,**binding())
    np.testing.assert_array_equal(x,[0,2,0,0,0,2,2,0])
    r=row('/Outside','/A'); r['impulse_on_0_ns']=[0,-.001,0]
    np.testing.assert_array_equal(generalized_contact_load([r],dt=.001,**binding()),x/2)
    internal=generalized_contact_load([row('/A','/B')],dt=.001,**binding())
    # Root generalized wrench cancels; opposing body-frame reactions remain.
    np.testing.assert_array_equal(internal,[0,0,0,0,0,0,1,1])


def test_one_step_lag_and_empty_fetch_clears_load():
    f=LaggedContactLoad(8);load,r=f.command(step=1,dt=.001)
    assert not load.any() and r['bootstrap_zero_load']
    f.observe([row()],step=1,dt=.001,guards_passed=True,full_normal_friction_stream=True,**binding())
    load,r=f.command(step=2,dt=.001)
    assert load[1]==1 and r['observation_step']==1 and r['estimate_age_steps']==1
    load[:]=999
    assert f.load[1]==1
    f.observe([],step=2,dt=.001,guards_passed=True,full_normal_friction_stream=True,**binding())
    assert not f.command(step=3,dt=.001)[0].any()


@pytest.mark.parametrize('bad',['stale','skipped','dt','guard','incomplete'])
def test_bad_observation_cannot_update_estimate(bad):
    f=LaggedContactLoad(8);f.command(step=1,dt=.001)
    kwargs=dict(step=1,dt=.001,guards_passed=True,full_normal_friction_stream=True)
    if bad=='stale':kwargs['step']=0
    elif bad=='skipped':kwargs['step']=2
    elif bad=='dt':kwargs['dt']=.002
    elif bad=='guard':kwargs['guards_passed']=False
    elif bad=='incomplete':kwargs['full_normal_friction_stream']=False
    with pytest.raises(ValueError):f.observe([row()],**kwargs,**binding())
    assert not f.load.any()
    with pytest.raises(ValueError):f.command(step=2,dt=.001)


@pytest.mark.parametrize('bad',['unknown','nan','oversize','kind','shape'])
def test_bad_contact_geometry_rejected(bad):
    rows=[row()]; b=binding()
    if bad=='unknown':rows=[row('/Other','/NotBound')]
    elif bad=='nan':rows[0]['impulse_on_0_ns']=[float('nan'),0,0]
    elif bad=='oversize':rows=rows*257
    elif bad=='kind':rows[0]['kind']='combined'
    elif bad=='shape':b['com_jacobians']=np.zeros((2,5,8))
    with pytest.raises(ValueError):generalized_contact_load(rows,dt=.001,**b)


def test_cli_rejects_wrong_model_before_startup(tmp_path):
    from .contact_spring_native_probe import main
    with pytest.raises(ValueError,match='implicit/PGS/240'):
        main(['--output',str(tmp_path/'unused'),'--source-report','missing','--lagged-contact-load'])
    assert not (tmp_path/'unused').exists()
