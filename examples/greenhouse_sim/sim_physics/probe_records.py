"""Lossless compressed per-step evidence with small in-memory gate summaries.

All native checks still run every step. The current mutable failure sample
stays complete until the next append/close; prior full samples go to JSONL.
This is diagnostic storage, not a training dataset or observation stream.
"""
import gzip
import time
from collections.abc import Sequence
from .strict_json import encode,backend


def summary(record):
    result={k:record[k] for k in ('t','slip_m','cut','detached_seam_gap_m','native_guards_passed') if k in record}
    if 'contact' in record:result['contact']={'bilateral':record['contact']['bilateral']}
    if 'robot' in record:result['robot']={'released_debris_contact_n':record['robot'].get('released_debris_contact_n',0.)}
    if 'cut_only_park' in record:result['cut_only_park']=True
    return result


class ProbeRecords(Sequence):
    def __init__(self,path,*,background_compression=False):
        if type(background_compression) is not bool:raise ValueError('Explicit compression option required')
        # Exclusive creation: never overwrite another run's evidence.
        self.file=path.open('xb');self.archive=gzip.GzipFile(fileobj=self.file,mode='wb',compresslevel=1,mtime=0)
        self.path=path;self.past=[];self.current=None;self.closed=False
        self.seconds=0.;self.maximum_s=0.;self.written=0
        self.archive_complete=False
        self.worker=None
        if background_compression:
            from .background_gzip import BackgroundGzip
            self.worker=BackgroundGzip(self.archive)

    def _write_current(self):
        if self.current is None:return
        start=time.perf_counter()
        packet=encode(self.current)
        (self.worker if self.worker is not None else self.archive).write(packet);self.written+=1
        elapsed=time.perf_counter()-start;self.seconds+=elapsed;self.maximum_s=max(self.maximum_s,elapsed)

    def append(self,record):
        if self.closed:raise RuntimeError('Evidence archive is closed')
        if self.worker is not None:self.worker.check()
        if self.current is not None:
            self._write_current();self.past.append(summary(self.current))
        self.current=record

    def __len__(self):return len(self.past)+(self.current is not None)

    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return self.current if index==len(self.past) else self.past[index]

    def close(self):
        if not self.closed:
            try:self._write_current()
            finally:
                try:
                    if self.worker is not None:self.worker.close()
                finally:
                    try:self.archive.close()
                    finally:self.file.close();self.closed=True
            self.archive_complete=True
        elif self.worker is not None:self.worker.check()

    def report(self):
        result=dict(format='lossless_jsonl_gzip_v1',path=self.path.name,
            samples=self.written if self.worker is None else self.worker.written,
            json_encoder=backend(),
            closed=self.closed,archive_complete=self.archive_complete,
            bytes=self.path.stat().st_size,serialization_wall_s=self.seconds,
            serialization_wall_scope=('main_thread_encode_and_compress' if self.worker is None
                else 'main_thread_encode_and_bounded_enqueue'),
            maximum_serialization_s=self.maximum_s,full_sample_retention_in_memory=1,
            original_per_step_guards_unchanged=True,training_eligible=False)
        if self.worker is not None:result.update(self.worker.report())
        return result
