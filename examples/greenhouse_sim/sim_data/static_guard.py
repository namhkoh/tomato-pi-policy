"""One full static scan plus USD notices; no scene traversal on every frame.

Only explicit robot snapshot edits are allowed BETWEEN captures. Any other USD
scene edit, root resync, or edit DURING capture invalidates the frame. This is a
static capture guard, not a native engine-frame synchronization substitute.
"""
from __future__ import annotations

from .capture_contract import fingerprint


def relevant_path(path):
    text=str(path)
    # Only renderer bookkeeping is excluded. Materials outside /World are watched.
    return not any(text==p or text.startswith(p+'/') or text.startswith(p+'.')
                   for p in ('/Render','/Replicator','/Orchestrator'))


class StaticSceneMonitor:
    def __init__(self,stage,robot_root='/World/RBY1'):
        from pxr import Tf,Usd
        from .capture_scene import scene_guard
        self.baseline=scene_guard(stage)
        self.robot_root=robot_root
        self.generation=0
        self.changed=set()
        self.subscription=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self._notice,stage)

    def _notice(self,notice,sender):
        paths=[*notice.GetResyncedPaths(),*notice.GetChangedInfoOnlyPaths()]
        relevant={str(p) for p in paths if relevant_path(p)}
        if relevant:
            self.changed.update(relevant)
            self.generation+=1

    def begin(self):
        unexpected=[p for p in self.changed if not(p==self.robot_root or p.startswith(self.robot_root+'/')
                                                   or p.startswith(self.robot_root+'.'))]
        if unexpected:
            raise ValueError('Unapproved static scene change: '+repr(sorted(unexpected)[:8]))
        self.changed.clear()
        return self.token()

    def token(self):
        return fingerprint(dict(method='initial_full_static_scan_and_USD_change_notice.v1',
                                baseline=self.baseline,generation=self.generation))

    def close(self):
        self.subscription.Revoke()
