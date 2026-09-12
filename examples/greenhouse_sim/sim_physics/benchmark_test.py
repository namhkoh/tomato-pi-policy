import json
from pathlib import Path

import pytest

from sim_physics.benchmark import main, parser, report_configuration


def test_fixed_articulation_requires_attached_only_before_kit_or_output_creation(tmp_path):
    output=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='attached-only'):
        main(['--output',str(output),'--constraint-mode','fixed_articulation'])
    assert not output.exists()


def test_default_probe_includes_release_and_does_not_use_gpu():
    args=parser().parse_args(['--output','unused'])
    assert args.constraint_mode=='articulation'
    assert not args.attached_only and not args.gui and args.render_hz==0
    assert not args.solve_articulation_contact_last
    assert not args.measured_withdrawal
    assert not args.experimental_contact_springs
    assert not args.force_closure
    assert args.torso_degrees is None and args.physics_window_half_m==2.
    assert args.right_ready_degrees is None and args.left_ik_seed_degrees is None


@pytest.mark.parametrize('option',['--right-ready-degrees','--left-ik-seed-degrees'])
def test_initial_joint_proposals_need_downward_fixture(tmp_path,option):
    with pytest.raises(ValueError,match='downward bimanual fixture'):
        main(['--output',str(tmp_path/'unused'),option,*['0']*7])
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('extra', [[], ['--bimanual-cut','--cut-style','downward'],
    ['--bimanual-cut','--cut-style','downward','--scene','package','--torso-yaw','1']])
def test_torso_proposal_cannot_override_legacy_or_yaw_fixture(tmp_path,extra):
    with pytest.raises(ValueError,match='Explicit torso requires'):
        main(['--output',str(tmp_path/'unused'),'--torso-degrees',*['0']*6,*extra])
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('joint,value', [(i,'nan') for i in range(6)] +
    [(i,'10000') for i in range(6)] + [(i,'-10000') for i in range(6)])
def test_torso_proposal_limits_fail_before_kit(tmp_path,joint,value):
    angles=['0']*6;angles[joint]=value
    with pytest.raises(ValueError,match='Explicit torso must obey exact URDF limits'):
        main(['--output',str(tmp_path/'unused'),'--bimanual-cut','--cut-style','downward',
              '--scene','package','--torso-degrees',*angles])
    assert not (tmp_path/'unused').exists()


def test_initial_torso_proposal_is_preserved_in_configuration():
    angles=[-8,-4,1,3,-20,-5]
    args=parser().parse_args(['--output','unused','--torso-degrees',*map(str,angles)])
    assert report_configuration(args,args.output)['torso_degrees']==angles


@pytest.mark.parametrize('extra',[[],['--bimanual-cut','--full-robot-probe',
    '--sparse-contacts','--finger-gravity','--seconds','28'],
    ['--bimanual-cut','--full-robot-probe','--sparse-contacts',
     '--finger-gravity','--compliant-fingers','--seconds','20']])
def test_force_closure_requires_compliance_and_sufficient_duration(tmp_path,extra):
    with pytest.raises(ValueError):main(['--output',str(tmp_path/'unused'),'--force-closure',*extra])
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('value',['nan','inf','.9','2.1','1.25'])
def test_workspace_size_cannot_silently_change_unbounded_scene(tmp_path,value):
    with pytest.raises(ValueError,match='Physics window'):
        main(['--output',str(tmp_path/'unused'),'--physics-window-half-m',value])
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('extra',[[],['--bimanual-cut'],['--bimanual-cut','--bimanual-hold-control'],
    ['--bimanual-cut','--bimanual-hold-control','--full-robot-probe','--diagnostic-contact-prediction','--diagnostic-grasp-dynamics']])
def test_experimental_contact_springs_cannot_enter_cut_or_uninstrumented_run(tmp_path,extra):
    output=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='Experimental contact springs'):
        main(['--output',str(output),'--experimental-contact-springs',*extra])
    assert not output.exists()


@pytest.mark.parametrize('extra',[[],['--bimanual-cut'],
    ['--bimanual-cut','--native-static-clearance','--bimanual-hold-control']])
def test_measured_withdrawal_needs_cut_and_native_clearance_before_launch(tmp_path,extra):
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='Measured withdrawal'):
        main(['--output',str(output),'--measured-withdrawal',*extra])
    assert not output.exists()


def test_contact_solver_order_is_explicit_and_recorded_without_changing_other_defaults():
    baseline=parser().parse_args(['--output','unused'])
    enabled=parser().parse_args(['--output','unused','--solve-articulation-contact-last'])
    differences={k for k in vars(baseline) if getattr(baseline,k)!=getattr(enabled,k)}
    assert differences=={'solve_articulation_contact_last'}
    assert report_configuration(enabled,enabled.output)['solve_articulation_contact_last'] is True


@pytest.mark.parametrize('extra',[[],['--bimanual-cut']])
def test_raw_grasp_diagnostic_restricted_to_right_parked_control(tmp_path,extra):
    output=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='right-parked'):
        main(['--output',str(output),'--diagnostic-grasp-contacts',*extra])
    assert not output.exists()


@pytest.mark.parametrize('proposal', [None, 'review inputs/single_proposal.json'])
def test_report_configuration_serializes_optional_cut_proposal(proposal):
    # Exercise the real CLI types and the actual report configuration helper.
    # No source file, output directory, or native runtime is opened here.
    argv=['--output','unused']
    if proposal is not None:argv+=['--cut-proposal-json',proposal]
    args=parser().parse_args(argv)
    configuration=report_configuration(args,args.output.resolve())
    expected=None if proposal is None else str(Path(proposal))
    for state in ('initializing','failed','passed_bimanual_mechanism_not_robot_task'):
        report=dict(state=state,configuration=configuration,training_eligible=False)
        restored=json.loads(json.dumps(report,allow_nan=False))
        assert restored['configuration']['cut_proposal_json']==expected
        assert restored['configuration']['output']==configuration['output']


@pytest.mark.parametrize('missing', [('native_static_clearance',), ('cut_proposal_json',),
    ('native_static_clearance','cut_proposal_json')])
def test_legacy_namespace_without_opt_ins_reaches_existing_validation(monkeypatch,tmp_path,missing):
    from types import SimpleNamespace
    import sim_physics.benchmark as module
    output=tmp_path/'must_not_create'
    args=parser().parse_args(['--output',str(output),'--constraint-mode','fixed_articulation'])
    values=vars(args).copy()
    for name in missing:del values[name]
    legacy=SimpleNamespace(**values)
    monkeypatch.setattr(module,'parser',lambda:SimpleNamespace(parse_args=lambda argv:legacy))
    # This later, unchanged validation prevents any output/native creation.
    with pytest.raises(ValueError,match='attached-only'):module.main([])
    assert not output.exists()
    assert all(not hasattr(legacy,name) for name in missing)


@pytest.mark.parametrize('extra,message', [
    (['--native-static-clearance'],'Native static clearance requires'),
    (['--cut-proposal-json','proposal.json'],'Single cut proposal requires')])
def test_explicit_opt_ins_still_require_bimanual_before_native(tmp_path,extra,message):
    output=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match=message):main(['--output',str(output),*extra])
    assert not output.exists()


@pytest.mark.parametrize('extra', [[], ['--gui'], ['--gui','--render-hz','30'],
    ['--gui','--render-hz','30','--spring-mode','implicit_effort','--solver','PGS','--scene','package']])
def test_interactive_rejects_unqualified_configuration_before_kit(tmp_path, extra):
    output=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='Interactive demo requires'):
        main(['--output',str(output),'--interactive',*extra])
    assert not output.exists()
def test_contact_prediction_snapshot_requires_existing_grasp_diagnostic(tmp_path):
    from sim_physics.benchmark import main
    import pytest
    with pytest.raises(ValueError,match='Contact prediction snapshots'):
        main(['--output',str(tmp_path/'unused'),'--diagnostic-contact-prediction'])
