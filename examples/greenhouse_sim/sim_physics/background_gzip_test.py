from copy import deepcopy
import gzip
import json
import threading
import pytest
from .background_gzip import BackgroundGzip
from .probe_records import ProbeRecords


class Sink:
    def __init__(self):self.packets=[]
    def write(self,packet):self.packets.append(packet)


def test_fifo_immutable_packets_and_joined_close():
    sink=Sink();worker=BackgroundGzip(sink)
    for i in range(200):worker.write(str(i).encode())
    worker.close();worker.close()
    assert sink.packets==[str(i).encode() for i in range(200)]
    r=worker.report()
    assert r['compressed_samples']==200 and r['worker_joined']
    assert r['queued_packet_limit']==2 and not r['native_objects_in_worker']
    with pytest.raises(RuntimeError):worker.write(b'late')


@pytest.mark.parametrize('bad',[{},[],bytearray(b'x'),memoryview(b'x'),'x'])
def test_worker_never_receives_mutable_records_or_native_objects(bad):
    w=BackgroundGzip(Sink())
    try:
        with pytest.raises(TypeError):w.write(bad)
    finally:w.close()


def test_bounded_queue_applies_backpressure_without_dropping():
    entered=threading.Event();release=threading.Event();completed=threading.Event()
    class Blocked(Sink):
        def write(self,packet):
            entered.set();assert release.wait(5)
            super().write(packet)
    sink=Blocked();w=BackgroundGzip(sink);producer=None
    try:
        w.write(b'a');assert entered.wait(5)
        w.write(b'b');w.write(b'c');assert w.queue.full()
        def produce():w.write(b'd');completed.set()
        producer=threading.Thread(target=produce);producer.start()
        assert not completed.wait(.03)
    finally:
        release.set()
        if producer is not None:producer.join(5)
        w.close()
    assert completed.is_set() and sink.packets==[b'a',b'b',b'c',b'd']


def test_compression_error_stays_sticky_and_close_does_not_deadlock():
    failed=threading.Event()
    class Broken:
        def write(self,packet):failed.set();raise OSError('disk failure')
    w=BackgroundGzip(Broken())
    try:w.write(b'a')
    except RuntimeError:pass  # Worker may report the fault before write returns.
    assert failed.wait(5);w.thread.join(.01)
    with pytest.raises(RuntimeError):w.close()
    with pytest.raises(RuntimeError):w.close()
    with pytest.raises(RuntimeError):w.write(b'b')
    assert w.written==0 and not w.thread.is_alive()
    assert w.report()['worker_error']=='disk failure'


@pytest.mark.parametrize('background',[False,True])
def test_exact_records_with_mutated_final_failure_tick(tmp_path,background):
    p=tmp_path/'trace.gz';r=ProbeRecords(p,background_compression=background);expected=[]
    for i in range(150):
        row=dict(t=i/480,contact=dict(bilateral=i>3),values=[i*.03]*120,
                 native_guards_passed=i!=149)
        r.append(row);row['late_fault']=i==149;expected.append(deepcopy(row))
    r.close()
    with gzip.open(p,'rt') as f:assert [json.loads(line) for line in f]==expected
    assert r.report()['samples']==150 and r.closed and r.report()['archive_complete']
    if background:assert r.report()['worker_joined']


def test_nonfinite_final_tick_remains_synchronous_error_and_worker_is_joined(tmp_path):
    r=ProbeRecords(tmp_path/'trace.gz',background_compression=True)
    r.append(dict(t=1));r.append(dict(t=2,force=float('nan')))
    with pytest.raises(ValueError):r.close()
    assert r.closed and r.report()['samples']==1 and r.report()['worker_joined']
    assert not r.report()['archive_complete']


def test_worker_failure_cannot_produce_successful_archive_close(tmp_path,monkeypatch):
    r=ProbeRecords(tmp_path/'trace.gz',background_compression=True)
    def fail(packet):raise OSError('no space')
    monkeypatch.setattr(r.archive,'write',fail)
    r.append(dict(t=1))
    with pytest.raises(RuntimeError):r.close()
    assert r.closed and r.file.closed and r.report()['samples']==0
    assert not r.report()['archive_complete']
    assert r.report()['worker_joined'] and r.report()['worker_error']=='no space'


def test_trailer_flush_failure_cannot_claim_a_complete_archive(tmp_path,monkeypatch):
    r=ProbeRecords(tmp_path/'trace.gz',background_compression=True);r.append(dict(t=1))
    original=r.archive.close
    def fail():original();raise OSError('trailer flush failed')
    monkeypatch.setattr(r.archive,'close',fail)
    with pytest.raises(OSError):r.close()
    assert r.closed and r.file.closed and not r.report()['archive_complete']
    assert r.report()['worker_joined']


def test_public_flag_is_explicit_and_requires_streaming(tmp_path,monkeypatch):
    from . import benchmark,ground_truth_trial
    with pytest.raises(ValueError,match='streaming'):
        benchmark.main(['--output',str(tmp_path/'bad'),'--background-evidence-compression'])
    assert not (tmp_path/'bad').exists()
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda argv:captured.append(benchmark.parser().parse_args(argv)))
    ground_truth_trial.main(['--output',str(tmp_path/'new'),'--mode','right_only',
        '--milestone','cut_action','--process-zone-trial','--through-stroke-trial',
        '--background-evidence-compression'])
    assert captured[0].stream_trajectory and captured[0].background_evidence_compression
    assert not benchmark.parser().parse_args(['--output','old']).background_evidence_compression
