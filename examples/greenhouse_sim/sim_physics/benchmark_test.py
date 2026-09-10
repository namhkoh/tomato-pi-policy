import pytest

from sim_physics.benchmark import main, parser


def test_fixed_articulation_requires_attached_only_before_kit_or_output_creation(tmp_path):
    output=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='attached-only'):
        main(['--output',str(output),'--constraint-mode','fixed_articulation'])
    assert not output.exists()


def test_default_probe_includes_release_and_does_not_use_gpu():
    args=parser().parse_args(['--output','unused'])
    assert args.constraint_mode=='articulation'
    assert not args.attached_only and not args.gui and args.render_hz==0


@pytest.mark.parametrize('extra', [[], ['--gui'], ['--gui','--render-hz','30'],
    ['--gui','--render-hz','30','--spring-mode','implicit_effort','--solver','PGS','--scene','package']])
def test_interactive_rejects_unqualified_configuration_before_kit(tmp_path, extra):
    output=tmp_path/'must_not_create'
    with pytest.raises(ValueError,match='Interactive demo requires'):
        main(['--output',str(output),'--interactive',*extra])
    assert not output.exists()
