"""Test durable completion/error reporting before Kit's potentially terminal close."""
import json
import sys
from types import SimpleNamespace

import pytest

from sim_data import collection_worker as worker


@pytest.mark.parametrize('fails',[False,True])
def test_capture_manifest_and_result_are_written_before_close(tmp_path,monkeypatch,capsys,fails):
    output=tmp_path/'capture'
    plan={'package':str(tmp_path/'sources'),'jobs':[{'job_id':'job_001'}]}
    monkeypatch.setattr(worker,'load_plan',lambda p:(plan,[]))
    def capture(app,args,plan,reports,job,manifest):
        if fails: raise ValueError('deliberate fixture failure')
        manifest['samples']=[{'fixture':True}]
    monkeypatch.setattr(worker,'capture_job',capture)
    closed=[]
    class App:
        def __init__(self,config): pass
        def close(self,exit_code):
            saved=json.loads((output/'manifest.json').read_text())
            assert saved['state']==('failed_do_not_train' if fails else 'pilot_ready_for_review')
            assert 'NATIVE_CAPTURE_RESULT' in capsys.readouterr().out
            closed.append(exit_code)
    monkeypatch.setitem(sys.modules,'isaacsim',SimpleNamespace(SimulationApp=App))
    assert worker.main(['--plan',str(tmp_path/'plan.json'),'--job','job_001','--output',str(output)]) == int(fails)
    assert closed == [int(fails)]
