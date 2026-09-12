"""Conservative full-cell native overlap refinement, not point sampling.

The caller must provide COMPLETE enclosing obstacle boxes in the current
native epoch. Positive actor controls alone do not prove that enclosure.
For each cell: reject empty obstacle space using the actual obstacle query;
otherwise query the countertransformed complete cell against the actual tool.
Only all-clear leaves prove separation. An unresolved cell stays blocked.
No collider, native body pose, source mesh or contact permission is changed.
"""
import numpy as np
from .native_body_bounds import countertransform_box


class NativeOccupancyClearance:
    def __init__(self,overlap,bounds,*,guard,positive_control,min_cell_m=.001,max_nodes=4096):
        if (not callable(overlap) or not callable(guard) or not callable(positive_control)
                or isinstance(min_cell_m,(bool,np.bool_)) or not np.isfinite(min_cell_m)
                or not .0005<=min_cell_m<=.002 or type(max_nodes) is not int or not 1<=max_nodes<=4096):
            raise ValueError('Bounded native cell queries and epoch/actor controls required')
        self.overlap=overlap;self.guard=guard;self.positive_control=positive_control
        self.min_cell_m=float(min_cell_m);self.max_nodes=max_nodes
        self.bounds={}
        for path,parts in bounds.items():
            if not isinstance(path,str) or not path.startswith('/') or not parts:
                raise ValueError('Exact obstacle paths and nonempty complete bounds required')
            self.bounds[path]=[]
            for centre,axes,half in parts:
                values=countertransform_box(np.eye(4),np.eye(4),centre,axes,half)
                self.bounds[path].append(values)
        if not self.bounds:raise ValueError('Complete native bounds required')
        self.occupancy={};self.nodes=0;self.calls=0;self.clearances=0;self.blocked=0

    def _query(self,path,centre,axes,half):
        self.guard();self.calls+=1
        # Native coordinates/quaternions use float32. Keep a conservative
        # scale-dependent roundoff reserve in addition to the requested margin.
        reserve=1e-6+16*np.finfo(np.float32).eps*(np.linalg.norm(centre)+np.linalg.norm(half)+1)
        if not np.isfinite(reserve):raise RuntimeError('Unrepresentable native query reserve')
        result=self.overlap(path,centre,axes,half+reserve)
        self.guard()
        if type(result) is not bool:raise RuntimeError('Unavailable native overlap is not empty space')
        return result

    def clear(self,tool,actual_body,proposed_body,obstacle,margin):
        self.guard()
        if (isinstance(margin,(bool,np.bool_)) or not np.isfinite(margin) or not .001<=margin<=.05):
            raise ValueError('Preserve finite 1..50 mm clearance margin')
        if obstacle not in self.bounds:return False
        # Validate poses even if every obstacle cell turns out empty.
        countertransform_box(actual_body,proposed_body,*self.bounds[obstacle][0])
        if self.positive_control(tool) is not True or self.positive_control(obstacle) is not True:
            raise RuntimeError('Missing exact native actor positive control')
        stack=[(i,(),*box) for i,box in enumerate(self.bounds[obstacle])]
        while stack:
            self.guard();self.nodes+=1
            if self.nodes>self.max_nodes:raise RuntimeError('Native occupancy node budget exhausted')
            root,address,centre,axes,half=stack.pop()
            key=(obstacle,root,address)
            if key not in self.occupancy:
                # Padding prevents float32/native query cracks between cells.
                self.occupancy[key]=self._query(obstacle,centre,axes,half+2e-6)
            if not self.occupancy[key]:continue
            c,a,h=countertransform_box(actual_body,proposed_body,centre,axes,half)
            if not self._query(tool,c,a,h+margin+2e-6):continue
            axis=int(np.argmax(half))
            if 2*half[axis]<=self.min_cell_m:
                self.blocked+=1;return False
            child=half.copy();child[axis]/=2
            # A gap-free binary partition of the whole parent, not samples.
            for sign in (-1,1):
                stack.append((root,address+(sign,),centre+sign*child[axis]*axes[:,axis],axes,child.copy()))
        self.guard();self.clearances+=1;return True

    def report(self):
        return dict(method='native_full_cell_countertransform_overlap',node_visits=self.nodes,
            native_query_calls=self.calls,cached_obstacle_cells=len(self.occupancy),
            pair_clearances=self.clearances,retained_rejections=self.blocked,
            minimum_cell_side_m=self.min_cell_m,max_nodes=self.max_nodes,
            requires_complete_enclosing_bounds=True,point_sampling=False,
            native_poses_changed=False,colliders_changed=False,whole_path_certified=False)
