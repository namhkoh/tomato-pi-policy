import json
import math
import numpy as np
import pytest
from . import strict_json as module


@pytest.mark.parametrize('fallback',[False,True])
def test_recursive_finite_records_round_trip_without_numeric_loss(monkeypatch,fallback):
    if fallback:monkeypatch.setattr(module,'_fast',None)
    rng=np.random.default_rng(182)
    values=[float(v) for v in rng.normal(size=2000)]
    values += [float(v) for v in np.nextafter([0.,1.,-1.,1e300],[1.,2.,-2.,math.inf])]
    r={'values':values,'strings':['Korean: \uac00','quotation "','slash \\','\n'],
       'flags':[False,True,None], 'repeated':(1,2,3),'nested':{'tiny':5e-324,'minuszero':-0.}}
    expected=json.loads(json.dumps(r,allow_nan=False))
    packet=module.encode(r);assert packet.endswith(b'\n') and packet.count(b'\n')==1
    assert json.loads(packet)==expected
    assert math.copysign(1,json.loads(packet)['nested']['minuszero'])==-1


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),np.float64('nan')])
@pytest.mark.parametrize('location',['root','nested','key'])
def test_nonfinite_cannot_silently_turn_into_null(value,location):
    r=value if location=='root' else {'deep':[{'value':value}]} if location=='nested' else {value:1}
    with pytest.raises(ValueError):module.encode(r)


def test_supported_stdlib_key_and_big_integer_behavior_is_preserved():
    for r in ({1:'int',False:'bool',None:'none'}, {'big':10**100}, {'numpy':np.float64(.123)}):
        assert json.loads(module.encode(r))==json.loads(json.dumps(r,allow_nan=False))


def test_cycles_rejected_but_shared_children_are_not_cycles():
    shared=[1.,2.];assert json.loads(module.encode([shared,shared]))==[shared,shared]
    r=[];r.append(r)
    with pytest.raises(ValueError):module.encode(r)


@pytest.mark.parametrize('r',[{'set':{1,2}},{'array':np.ones(2)},{'object':object()}])
def test_unsupported_data_stays_an_error(r):
    with pytest.raises(TypeError):module.encode(r)


def test_fast_library_cannot_expand_input_contract_to_unchecked_dataclasses():
    from dataclasses import dataclass
    from datetime import datetime
    @dataclass
    class Hidden:
        force:float=float('nan')
    for r in ({'hidden':Hidden()},{'time':datetime(2026,9,14)}):
        with pytest.raises(TypeError):module.encode(r)


def test_archive_rejects_bad_final_tick_instead_of_approving_missing_measurement(tmp_path):
    from .probe_records import ProbeRecords
    r=ProbeRecords(tmp_path/'trace.gz');r.append({'force':float('nan')})
    with pytest.raises(ValueError):r.close()
    assert r.closed and r.written==0
