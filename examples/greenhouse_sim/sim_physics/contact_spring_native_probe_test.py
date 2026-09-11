import numpy as np
import pytest

from .contact_spring_native_probe import ContactRows,main,finite_native,read_state


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
