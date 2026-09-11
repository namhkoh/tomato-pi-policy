"""Submission contract tests with an explicit fake view, not native evidence."""
from copy import deepcopy

import numpy as np
import pytest

from sim_physics.contact_coupled_prediction import MaterialLaw, _kinematic_jacobian
from sim_physics.contact_spring_probe import bind, layout
from sim_physics.contact_spring_probe_test import coupon, SplitFake


class Fake(SplitFake):
    def __init__(self,c):
        super().__init__(c)
        self.root_v = np.zeros((1,6)); self.joint_v = np.zeros((1,2))
    def get_root_velocities(self): return self.root_v.copy()
    def get_dof_velocities(self): return self.joint_v.copy()


def setup():
    c = coupon(held_contacts=False); a = Fake(c)
    report, p = bind(a,c,model='coupled_contact_prediction',contact_law=MaterialLaw('unilateral_kv_v1'))
    frames = layout(c)['frames']
    state = dict(native_coordinate_convention_verified=True, contact_force_included=False,
        generalized_velocity=np.zeros(8), mass_matrix=np.eye(8)*.001,
        body_world_com_jacobians=_kinematic_jacobian(frames), native_known_external_force=np.zeros(8))
    args = dict(dt=c.dt, state=state, frames=frames, velocities=np.zeros((3,6)),
        native_rows=[], reference_frames=frames, step_id=1, reference_step_id=0)
    return a,p,report,args


def test_binding_and_submission_only_changes_K_C_and_joint_spring_command():
    a,p,report,args = setup()
    assert a.writes == ['stiffness','damping']
    assert report['native_drives_disabled'] and not report['native_si_coefficients_verified']
    before = deepcopy(args)
    cmd,evidence = p.step(**args)
    assert len(a.commands) == 1 and cmd.shape == (2,)
    np.testing.assert_array_equal(cmd,a.commands[0][0])
    assert evidence['prediction']['tau_spring'][:6] == [0]*6
    assert not evidence['contact_effort_submitted'] and not evidence['root_actuated']
    assert not report['coupled_prediction']['native_qualified']
    np.testing.assert_array_equal(args['frames'],before['frames'])
    with pytest.raises(ValueError,match='contiguous'):
        p.step(**args)
    assert len(a.commands) == 1


@pytest.mark.parametrize('bad',['dt','step','root_v','q','K','C','caps','type','target','paths','mass','snapshot','force'])
def test_faults_abort_before_any_effort(bad):
    a,p,_,args = setup()
    if bad == 'dt': args['dt'] *= 2
    elif bad == 'step': args['step_id'] = True
    elif bad == 'root_v': a.root_v[0,0] = .1
    elif bad == 'q': a.q[0,0] = .05
    elif bad == 'K': a.k[0,0] = 1e-9
    elif bad == 'C': a.c[0,0] = 1e-9
    elif bad == 'caps': a.caps[0,0] = 1
    elif bad == 'type': a.drive_types[0,0] = 2
    elif bad == 'target': a.velocity_targets[0,0] = .1
    elif bad == 'paths': a.link_paths[0].reverse()
    elif bad == 'mass': args['state']['mass_matrix'][:] = 0
    elif bad == 'snapshot': args['state']['native_coordinate_convention_verified'] = False
    else: args['state']['contact_force_included'] = True
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        p.step(**args)
    assert a.commands == [] and p.last_step == 0


def test_law_is_never_silently_chosen_or_applied_to_other_models():
    c = coupon(); a = Fake(c)
    with pytest.raises(ValueError,match='Explicit contact material'):
        bind(a,c,model='coupled_contact_prediction')
    with pytest.raises(ValueError,match='only valid'):
        bind(a,c,model='native',contact_law=MaterialLaw('unilateral_kv_v1'))
    assert a.writes == [] and a.commands == []
