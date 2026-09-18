"""Bounded lossless bulk-frame persistence, independent of Isaac/Usd imports."""
from pathlib import Path
from threading import Condition, Thread, Lock
from queue import Queue
import hashlib
import json
import time
import numpy as np
from PIL import Image


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
    # Windows rename is atomic and fails if the destination already exists.
    # Consumers poll only the final name, never a partially written document.
    temporary.rename(path)


class FrameSink:
    """Backpressure bounds both queued frame count and uncompressed array bytes.

    The caller hands over owned, immutable arrays. observation.json is published
    last; incomplete folders are never committed observations. Writer failures
    are raised by submit/close, and never silently discard a successful batch.
    """
    def __init__(self, output, *, max_frames=8, max_bytes=256*2**20, workers=2):
        if type(max_frames) is not int or max_frames < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Positive queue limits required')
        if type(workers) is not int or not 1 <= workers <= max_frames:
            raise ValueError('Bounded worker count required')
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)
        self.queue = Queue(maxsize=max_frames)
        self.budget = max_bytes
        self.pending_bytes = 0
        self.peak_bytes = 0
        self.cv = Condition()
        self.lock = Lock()
        self.error = None
        self.closed = False
        self.ids = set()
        self.records = []
        self.threads = [Thread(target=self._run, name='bulk-frame-writer-'+str(i), daemon=False) for i in range(workers)]
        for thread in self.threads:
            thread.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Bulk frame writer failed; batch must remain incomplete') from self.error

    def submit(self, observation, rgb, depth, ids, valid, alpha):
        self._check()
        if self.closed:
            raise RuntimeError('Frame sink is closed')
        name = observation['observation_id']
        if not name or Path(name).name != name or name in ('.', '..') or name in self.ids:
            raise ValueError('Unique safe observation ID required')
        if rgb.shape != (408,848,3) or rgb.dtype != np.uint8:
            raise ValueError('Native848 RGB required')
        if depth.shape != (408,848) or depth.dtype != np.float32 or ids.shape != depth.shape or ids.dtype != np.uint32:
            raise ValueError('Native optical depth and renderer IDs required')
        if valid.shape != depth.shape or valid.dtype != np.bool_ or alpha.shape != depth.shape or alpha.dtype != np.uint8:
            raise ValueError('Native validity and original RGBA alpha required')
        size = sum(a.nbytes for a in (rgb,depth,ids,valid,alpha))
        if size > self.budget:
            raise ValueError('One frame exceeds the explicit queue byte budget')
        started = time.perf_counter()
        with self.cv:
            while self.pending_bytes + size > self.budget:
                self._check()
                self.cv.wait(timeout=0.25)
            self._check()
            self.pending_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
        self.ids.add(name)
        # Queue count applies even when the byte budget has room.
        self.queue.put((observation,rgb,depth,ids,valid,alpha,size))
        return time.perf_counter()-started

    def _run(self):
        while True:
            job = self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            observation,rgb,depth,ids,valid,alpha,size = job
            try:
                if self.error is None:
                    started = time.perf_counter()
                    folder = self.output/observation['observation_id']
                    folder.mkdir(exist_ok=False)
                    rgb_path, buffers_path = folder/'rgb.png', folder/'buffers.npz'
                    Image.fromarray(rgb).save(rgb_path, compress_level=1)
                    # Store each native array once. Alpha preserves the original
                    # callback RGBA losslessly without duplicating RGB pixels.
                    np.savez_compressed(buffers_path, depth_m=depth, renderer_instance_id=ids,
                        depth_valid=valid, rgba_alpha=alpha)
                    observation['files'] = {
                        'rgb':dict(path=str(rgb_path.resolve()),sha256=sha256(rgb_path)),
                        'buffers':dict(path=str(buffers_path.resolve()),sha256=sha256(buffers_path))}
                    observation['timing']['lossless_write_seconds'] = time.perf_counter()-started
                    path = folder/'observation.json'
                    save_json(path,observation)
                    receipt = dict(observation_id=observation['observation_id'],path=str(path.resolve()),
                        sha256=sha256(path),request_index=observation['synchronization']['request_index'])
                    with self.lock:
                        self.records.append(receipt)
            except BaseException as exc:
                with self.cv:
                    if self.error is None:
                        self.error = exc
                    self.cv.notify_all()
            finally:
                with self.cv:
                    self.pending_bytes -= size
                    self.cv.notify_all()
                self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            for _ in self.threads:
                self.queue.put(None)
            self.queue.join()
            for thread in self.threads:
                thread.join()
        self._check()
        return sorted(self.records,key=lambda r:r['request_index'])
