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
    def __init__(self,path):
        # Exclusive creation: never overwrite another run's evidence.
        self.file=path.open('xb');self.archive=gzip.GzipFile(fileobj=self.file,mode='wb',compresslevel=1,mtime=0)
        self.path=path;self.past=[];self.current=None;self.closed=False
        self.seconds=0.;self.maximum_s=0.;self.written=0

    def _write_current(self):
        if self.current is None:return
        start=time.perf_counter()
        packet=encode(self.current)
        self.archive.write(packet);self.written+=1
        elapsed=time.perf_counter()-start;self.seconds+=elapsed;self.maximum_s=max(self.maximum_s,elapsed)

    def append(self,record):
        if self.closed:raise RuntimeError('Evidence archive is closed')
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
                self.archive.close();self.file.close();self.closed=True

    def report(self):
        return dict(format='lossless_jsonl_gzip_v1',path=self.path.name,samples=self.written,
            json_encoder=backend(),
            closed=self.closed,bytes=self.path.stat().st_size,serialization_wall_s=self.seconds,
            maximum_serialization_s=self.maximum_s,full_sample_retention_in_memory=1,
            original_per_step_guards_unchanged=True,training_eligible=False)
