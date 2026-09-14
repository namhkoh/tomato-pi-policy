import json
import pytest
from .startup_report import start


def test_startup_error_has_a_report_and_still_raises(tmp_path):
    def fail(config):raise RuntimeError('engine refused startup')
    report=dict(state='initializing',configuration={'target':'seed19/SubStem_41'})
    with pytest.raises(RuntimeError,match='engine refused'):
        start(fail,{},tmp_path,report)
    saved=json.loads((tmp_path/'report.json').read_text())
    assert saved['state']=='failed_application_startup'
    assert not saved['task_scene_constructed'] and not saved['training_eligible']
    assert 'engine refused startup' in saved['error']
    assert saved['configuration']==report['configuration']


def test_success_returns_app_without_writing_a_premature_result(tmp_path):
    app=object();config={'headless':True}
    assert start(lambda c:app,config,tmp_path,{}) is app
    assert not (tmp_path/'report.json').exists()


def test_failure_never_overwrites_an_existing_report(tmp_path):
    p=tmp_path/'report.json';p.write_text('preserved')
    def fail(config):raise RuntimeError('failed')
    with pytest.raises(FileExistsError):start(fail,{},tmp_path,{})
    assert p.read_text()=='preserved'


@pytest.mark.parametrize('phase',['before_native_parse','before_reset','after_reset'])
def test_checkpoint_is_diagnostic_exclusive_and_does_not_mutate_report(tmp_path,phase):
    from .startup_report import checkpoint
    report={'state':'initializing','configuration':{'solver':'PGS'}}
    before=json.dumps(report)
    checkpoint(tmp_path,report,phase)
    path=tmp_path/('startup_'+phase+'.json');saved=json.loads(path.read_text())
    assert saved['state']=='startup_checkpoint_not_qualification'
    assert not saved['motion_authorized'] and not saved['training_eligible']
    assert saved['phase']==phase and saved['report_snapshot']==report
    assert json.dumps(report)==before and not (tmp_path/'report.json').exists()
    with pytest.raises(FileExistsError):checkpoint(tmp_path,report,phase)
    assert json.loads(path.read_text())==saved


def test_unknown_phase_and_nonfinite_snapshot_do_not_create_checkpoint(tmp_path):
    from .startup_report import checkpoint
    with pytest.raises(ValueError):checkpoint(tmp_path,{},'../report')
    with pytest.raises(ValueError):checkpoint(tmp_path,{'bad':float('nan')},'before_reset')
    assert not list(tmp_path.iterdir())
