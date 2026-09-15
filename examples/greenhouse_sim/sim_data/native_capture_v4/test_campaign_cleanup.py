"""Mocked Windows tree cleanup only; never terminate actual processes."""
import subprocess
import pytest
from . import campaign as c
from .test_campaign_guards import controls,write,FakeProcess


def test_control_plan_removed_from_both_maps_is_rejected(tmp_path):
    root,request,result=controls(tmp_path)
    request['bindings'].pop(request['plans'][0]);result['bindings']=request['bindings']
    write(root/'request.json',request);write(root/'result.json',result)
    with pytest.raises(ValueError,match='lacks immutable'):c.completed_controls(root)


def test_already_exited_parent_does_not_claim_tree_cleanup(monkeypatch):
    process=FakeProcess();process.alive=False
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:pytest.fail('Do not target an exited parent'))
    with pytest.raises(ValueError,match='already exited'):c._terminate_owned(process)


def test_failed_taskkill_is_not_ignored(monkeypatch):
    def fail(command,**kwargs):
        assert kwargs['check'] is True
        assert command==['taskkill.exe','/PID','999999999','/T','/F']
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(c.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):c._terminate_owned(FakeProcess())


def test_remaining_native_process_keeps_cleanup_uncertain(monkeypatch):
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(c,'native_processes',lambda:[{'ProcessId':123}])
    with pytest.raises(ValueError,match='remain'):c._terminate_owned(FakeProcess())


def test_confirmed_taskkill_and_no_native_process(monkeypatch):
    calls=[]
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:calls.append((a,k)))
    monkeypatch.setattr(c,'native_processes',lambda:[])
    c._terminate_owned(FakeProcess())
    assert calls[0][1]['check'] is True
