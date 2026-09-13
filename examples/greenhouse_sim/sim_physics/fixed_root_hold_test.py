import pytest

from .benchmark import parser, main, validate_fixed_root_hold, report_configuration


def arguments():
    return ['--output','unused','--fixed-root-contact-hold','--isolated-cut-contact-trial',
        '--bimanual-hold-control','--attached-only','--constraint-mode','fixed_articulation',
        '--spring-mode','implicit_effort','--diagnostic-grasp-dynamics']


def test_scope_is_default_off_and_serializable():
    default=parser().parse_args(['--output','unused'])
    assert validate_fixed_root_hold(default) is False
    args=parser().parse_args(arguments())
    assert validate_fixed_root_hold(args) is True
    assert report_configuration(args,args.output)['fixed_root_contact_hold'] is True


@pytest.mark.parametrize('flag',['--isolated-cut-contact-trial','--bimanual-hold-control',
    '--attached-only','--diagnostic-grasp-dynamics'])
def test_required_hold_scope(flag):
    argv=arguments();argv.remove(flag)
    with pytest.raises(ValueError,match='Fixed root contact HOLD'):
        validate_fixed_root_hold(parser().parse_args(argv))


@pytest.mark.parametrize('extra',[
    ['--constraint-mode','articulation'],['--spring-mode','native'],
    ['--bimanual-reposition-m','.002'],['--native-torsion-trial'],
    ['--native-spring-cut-trial'],['--measured-withdrawal']])
def test_forbidden_variant(extra):
    with pytest.raises(ValueError,match='Fixed root contact HOLD'):
        validate_fixed_root_hold(parser().parse_args(arguments()+extra))


def test_fixed_hold_does_not_bypass_complete_contact_fixture():
    with pytest.raises(ValueError,match='complete explicit flat'):
        main(arguments())


def test_native_hold_requires_both_explicit_comparison_and_native_mode():
    args=parser().parse_args(arguments()+['--native-spring-cut-trial','--spring-mode','native'])
    assert validate_fixed_root_hold(args) is True
    assert args.attached_only and args.bimanual_hold_control


def test_unqualified_fixed_cut_still_rejected(tmp_path):
    out=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='attached-only'):
        main(['--output',str(out),'--constraint-mode','fixed_articulation'])
    assert not out.exists()
