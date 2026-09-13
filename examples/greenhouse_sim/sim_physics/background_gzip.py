"""Bounded diagnostic compression worker. Never receives simulator objects.

The controller serializes/validates immutable bytes before enqueueing. The
worker only writes gzip data. Backpressure blocks rather than discarding any
sample; errors are sticky and close joins before a result can be published.
"""
import queue
import threading
import time


class BackgroundGzip:
    def __init__(self, archive):
        self.archive=archive
        self.queue=queue.Queue(maxsize=2)
        self.error=None;self.closed=False;self.written=0;self.seconds=0.
        self.thread=threading.Thread(target=self._run,name='physics-evidence-gzip',daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Background evidence compression failed') from self.error

    def _run(self):
        while True:
            packet=self.queue.get()
            try:
                if packet is None:return
                if self.error is None:
                    try:
                        start=time.perf_counter();self.archive.write(packet)
                        self.seconds+=time.perf_counter()-start;self.written+=1
                    except Exception as exc:self.error=exc
                # Drain after failure so close cannot deadlock; failure stays
                # sticky, and this archive can never be reported complete.
            finally:self.queue.task_done()

    def write(self,packet):
        if self.closed:raise RuntimeError('Background evidence writer is closed')
        if type(packet) is not bytes:raise TypeError('Immutable serialized bytes required')
        while True:
            self.check()
            try:self.queue.put(packet,timeout=.01);break
            except queue.Full:pass
        self.check()

    def close(self):
        if not self.closed:
            self.closed=True
            self.queue.put(None)
            self.thread.join()
        self.check()

    def report(self):
        return dict(background_compression=True,queued_packet_limit=2,
            maximum_worker_and_queued_packets=3,compression_wall_s=self.seconds,
            compressed_samples=self.written,worker_joined=not self.thread.is_alive(),
            worker_error=None if self.error is None else str(self.error),
            backpressure='block_without_dropping',native_objects_in_worker=False)
