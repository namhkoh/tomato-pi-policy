import numpy as np
import pytest

from .contact_spring_native_probe import ContactRows,main,finite_native,read_state,read_prediction_state


def monitor():
    return ContactRows(robot_root='/World/Coupon',target_root='/World/Coupon',fingers=[],floor_root=None)


def test_signed_original_order_and_one_copy_per_friction_anchor():
    m=monitor()
    m.consume('/World/Coupon/B','/World/Coupon/A',[(0,0,-.01),(0,0,.002)],
        [(1,0,0),(2,0,0)],[(0,0,1),(0,0,1)],[-.001,0],
        friction_impulses=[(.003,0,0)],friction_points=[(1.5,0,0)])
    assert len(m.rows)==3
    assert [r['kind'] for r in m.rows]==['normal','normal','friction']
    assert all(r['collider0']=='/World/Coupon/B' for r in m.rows)
    assert m.rows[0]['impulse_on_0_ns']==[0,0,-.01]
    assert m.rows[0]['normal_on_0']==[0,0,1]
    assert m.rows[0]['separation_m']==-.001
    assert m.rows[1]['separation_m']==0
    assert m.rows[-1]['point_world_m']==[1.5,0,0]
    m.begin_step()
    assert m.rows==[] and m.pairs=={}


def test_friction_only_and_numpy_points_supported():
    m=monitor()
    m.consume('/World/Coupon/A','/World/Coupon/B',[],[],[],[],
        friction_impulses=np.array([[.001,0,0]]),friction_points=np.array([[1,2,3]]))
    assert len(m.rows)==1 and m.rows[0]['kind']=='friction'


def test_missing_positions_and_row_overflow_fail_closed():
    m=monitor()
    with pytest.raises(ValueError,match='Full signed contact points'):
        m.consume('/World/Coupon/A','/World/Coupon/B',[(.1,0,0)])
    m.begin_step()
    with pytest.raises(RuntimeError,match='overflow'):
        m.consume('/World/Coupon/A','/World/Coupon/B',[(.1,0,0)]*257,[(0,0,0)]*257,[(1,0,0)]*257)


@pytest.mark.parametrize('seconds',['nan','inf','0.9','10.1'])
def test_bad_duration_rejected_before_app_or_output(tmp_path,seconds):
    out=tmp_path/'unused'
    with pytest.raises(ValueError,match='Bounded'):
        main(['--output',str(out),'--source-report','missing','--seconds',seconds])
    assert not out.exists()


def test_existing_output_never_reused(tmp_path):
    with pytest.raises(ValueError,match='New diagnostic'):
        main(['--output',str(tmp_path),'--source-report','missing'])


@pytest.mark.parametrize('options,iterations,solver,hz',[
    ([],(16,4),'PGS',240),
    (['--model','native','--iterations','128/0','--solver','TGS',
      '--physics-hz','1920','--seconds','1','--diagnostic-contact-geometry'],(128,0),'TGS',1920),
])
def test_iteration_cli_reaches_preparation_with_defaults_unchanged(
        tmp_path,monkeypatch,options,iterations,solver,hz):
    from . import contact_spring_native_probe as runner
    captured={}
    class StopBeforeNative(RuntimeError): pass
    def prepare(path,**kwargs):
        captured.update(kwargs)
        raise StopBeforeNative('CPU preparation sentinel')
    monkeypatch.setattr(runner,'from_report',prepare)
    out=tmp_path/'unused'
    with pytest.raises(StopBeforeNative):
        runner.main(['--output',str(out),'--source-report','missing',*options])
    assert captured == dict(rotation=((1.,0.,0),(0.,1.,0),(0,0,1)),
        iterations=iterations,dt=1/hz,solver=solver,held_contacts=True)
    assert not out.exists()


@pytest.mark.parametrize('iterations',['128/1','128/4','128/-1','127/0'])
def test_other_iteration_choices_remain_rejected_before_preparation(tmp_path,monkeypatch,iterations):
    from . import contact_spring_native_probe as runner
    def unexpected(*args,**kwargs):
        pytest.fail('Invalid CLI must not reach source preparation or native import')
    monkeypatch.setattr(runner,'from_report',unexpected)
    out=tmp_path/'unused'
    with pytest.raises(SystemExit) as error:
        runner.main(['--output',str(out),'--source-report','missing','--iterations',iterations])
    assert error.value.code == 2 and not out.exists()


@pytest.mark.parametrize('options',[[],['--model','coupled_contact_prediction',
    '--contact-law','signed_overlap_kv_v1']])
def test_patch_friction_requires_explicit_compatible_coupon_before_app(tmp_path,options):
    out=tmp_path/'unused'
    with pytest.raises(ValueError,match='Patch friction'):
        main(['--output',str(out),'--source-report','missing','--contact-patch-friction',*options])
    assert not out.exists()


@pytest.mark.parametrize('value',[[[float('nan'),0]],[[float('inf'),0]],[0,0]])
def test_nonfinite_or_wrong_shape_native_diagnostics_rejected(value):
    with pytest.raises(RuntimeError,match='Invalid native'):
        finite_native(value,(1,2),'diagnostic')


def test_native_diagnostic_copy_and_section_cli(tmp_path):
    original=np.array([[1.,2.]])
    copy=finite_native(original,(1,2),'diagnostic')
    original[:]=99
    assert copy.tolist()==[[1,2]]
    with pytest.raises(ValueError,match='New diagnostic'):
        main(['--output',str(tmp_path),'--source-report','missing','--model','section_springs'])


def test_maximal_state_uses_only_native_body_readbacks():
    from types import SimpleNamespace
    poses=np.array([[0,0,-.03,0,0,0,1],[0,0,0,0,0,0,1],[0,0,.03,0,0,0,1]],float)
    velocity=np.zeros((3,6));velocity[:,4]=[1,2,4]
    view=SimpleNamespace(get_transforms=lambda:poses,get_velocities=lambda:velocity)
    frames,v,q,qdot=read_state(view,maximal=True)
    np.testing.assert_allclose(frames[:,:3,3],poses[:,:3])
    np.testing.assert_allclose(v,velocity)
    np.testing.assert_allclose(q,[0,0])
    np.testing.assert_allclose(qdot,[1,2])


def test_maximal_cannot_use_articulation_friction_setter(tmp_path):
    with pytest.raises(ValueError,match='unavailable for maximal'):
        main(['--output',str(tmp_path/'unused'),'--source-report','missing',
              '--model','section_springs_maximal','--zero-joint-friction'])


@pytest.mark.parametrize('separations',[[float('nan')],[],[0,0]])
def test_bad_separations_cannot_become_prediction_evidence(separations):
    m=monitor()
    with pytest.raises(ValueError,match='separation'):
        m.consume('/World/Coupon/A','/World/Coupon/B',[(.1,0,0)],[(0,0,0)],
                  [(1,0,0)],separations)


def prediction_view():
    from types import SimpleNamespace
    return SimpleNamespace(get_root_velocities=lambda:np.zeros((1,6)),
        get_jacobians=lambda:np.zeros((1,3,6,8)),
        get_generalized_mass_matrices=lambda:np.eye(8)[None],
        get_coms=lambda:np.tile([0,0,0,0,0,0,1],(1,3,1)),
        get_gravity_compensation_forces=lambda:np.zeros((1,8)),
        get_coriolis_and_centrifugal_compensation_forces=lambda:np.zeros((1,8)))


def test_prediction_state_is_read_only_and_checks_native_mapping():
    v=prediction_view()
    result=read_prediction_state(v,np.zeros((3,6)),np.zeros(2))
    assert result['point_velocity_max_error']==0
    assert result['contact_force_included'] is False
    with pytest.raises(RuntimeError,match='convention check failed'):
        read_prediction_state(v,np.ones((3,6)),np.zeros(2))


def test_prediction_state_rejects_singular_mass_without_regularizing():
    v=prediction_view();v.get_generalized_mass_matrices=lambda:np.zeros((1,8,8))
    with pytest.raises(RuntimeError,match='convention check failed'):
        read_prediction_state(v,np.zeros((3,6)),np.zeros(2))


def test_contact_order_control_is_explicit_preparse_usd_only():
    from pxr import Usd
    from sim_physics.contact_spring_native_probe import author_contact_order
    stage=Usd.Stage.CreateInMemory();scene=stage.DefinePrim('/World/Physics','PhysicsScene')
    before=stage.GetRootLayer().ExportToString()
    report=author_contact_order(scene,enabled=False)
    assert stage.GetRootLayer().ExportToString()==before and not report['authored']
    report=author_contact_order(scene,enabled=True)
    assert report['observed_usd_value'] is True and report['previous_usd_value'] is None
    assert not report['native_effective_value_verified']
    with pytest.raises(ValueError):author_contact_order(scene,enabled=1)


@pytest.mark.parametrize('fault',['close','detach','exit','receipt',None])
def test_cleanup_fault_does_not_prevent_other_cleanup_or_failure_evidence(fault):
    from types import SimpleNamespace
    from sim_physics.contact_spring_native_probe import cleanup_native
    calls=[]
    def call(name):
        calls.append(name)
        if name==fault:raise RuntimeError('injected '+name)
        return {'faulted':False}
    result=cleanup_native(SimpleNamespace(close=lambda:call('close')),
        SimpleNamespace(detach_stage=lambda:call('detach')),
        SimpleNamespace(__exit__=lambda *args:call('exit'),report=lambda:call('receipt')))
    assert calls==['close','detach','exit','receipt']
    assert len(result['failures'])==int(fault is not None)
    if fault:assert 'injected '+fault in result['failures'][0]
    assert (result['native_errors'] is None)==(fault=='receipt')
