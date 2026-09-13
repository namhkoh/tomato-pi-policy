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
