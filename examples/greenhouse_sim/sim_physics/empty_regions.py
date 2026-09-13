"""Bounded same-epoch native-miss certificates, not approximate collision data."""
import numpy as np


class EmptyRegions:
    def __init__(self):
        self.regions={};self.hits=0;self.created=0

    def contains(self,path,centre,axes,half):
        # A query box, INCLUDING its original margin, must fit entirely in a
        # larger box that a real native overlap query proved empty for this
        # exact collider. Rotation and all eight corners are accounted for.
        # The outward numerical reserve is intentionally against reuse.
        for c,r,h in reversed(self.regions.get(path,())):
            extent=np.abs(r.T@axes)@half+np.abs(r.T@(centre-c))
            if np.all(extent+1e-7<=h):
                self.hits+=1
                return True
        return False

    def add(self,path,centre,axes,half):
        if path not in self.regions and len(self.regions)>=128:return
        rows=self.regions.setdefault(path,[])
        rows.append(tuple(np.array(x,dtype=float,copy=True) for x in (centre,axes,half)))
        if len(rows)>16:del rows[0]
        self.created+=1

    def clear(self):self.regions.clear()

    def report(self):
        return dict(model='same_epoch_native_empty_box_containment_v1',hits=self.hits,
            certificates_created=self.created,retained=sum(map(len,self.regions.values())),
            maximum_paths=128,maximum_per_path=16,extra_half_extent_m=.005,
            containment_reserve_m=1e-7,exact_collider_identity=True,
            physics_geometry_or_margin_changed=False)
