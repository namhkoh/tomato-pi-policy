from copy import deepcopy
import gzip
import json
import pytest
from .probe_records import ProbeRecords,summary


def sample(i):return dict(t=i/480,cut=i>2,slip_m=.001,detached_seam_gap_m=.005,
    contact=dict(bilateral=True,raw=[list(range(100))]),native_guards_passed=True,
    robot=dict(released_debris_contact_n=0.,large=list(range(100))),cut_only_park={'passed':True})


def test_lossless_all_rows_including_mutation_and_failure_tick(tmp_path):
    path=tmp_path/'trace.jsonl.gz';records=ProbeRecords(path);expected=[]
    for i in range(10):
        r=sample(i);records.append(r);r['late_mutation']={'fault':i==9};expected.append(deepcopy(r))
        assert records[-1] is r and len(records)==i+1
        if i:assert 'raw' not in records[-2]['contact']
    records.close();records.close()
    with gzip.open(path,'rt',encoding='utf8') as stream:actual=[json.loads(line) for line in stream]
    assert actual==expected and records.report()['samples']==10
    assert records.report()['closed'] and records.report()['full_sample_retention_in_memory']==1
    assert all(r['contact']['bilateral'] for r in records)
    assert records[0]==summary(expected[0])
    with pytest.raises(RuntimeError):records.append(sample(12))


def test_no_overwrite(tmp_path):
    p=tmp_path/'trace.jsonl.gz';p.write_bytes(b'preserved')
    with pytest.raises(FileExistsError):ProbeRecords(p)
    assert p.read_bytes()==b'preserved'


def test_empty_archive_and_indexing(tmp_path):
    r=ProbeRecords(tmp_path/'trace.gz')
    assert not r and r[:]==[]
    with pytest.raises(IndexError):r[-1]
    r.close()
    assert r.report()['samples']==0


def test_missing_guard_not_promoted(tmp_path):
    r=ProbeRecords(tmp_path/'trace.gz');r.append(dict(t=1,contact={'bilateral':False}))
    r.append(sample(2))
    assert 'native_guards_passed' not in r[0] and not r[0]['contact']['bilateral']
    r.close()
