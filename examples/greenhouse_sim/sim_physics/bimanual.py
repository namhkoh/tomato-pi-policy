"""Guarded full-robot knife test. Privileged fixture, not a learned controller."""
import numpy as np

from .full_robot import FullRobotGripper
from .knife import KnifeGeometry,ShearGate,mount_forward


class BimanualRobot(FullRobotGripper):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('station_offset',(0.,0.))
        kwargs.setdefault('approach_side',1)
        super().__init__(*args,**kwargs)
        if not self.sparse_contacts: raise ValueError('Bimanual test requires sparse native contacts')
        self.knife_mount=mount_forward(self.stage,self.root)
        self.knife=KnifeGeometry(self.stage,self.root)
        self.cut_gate=ShearGate(self.rig.source_target)
        self.cut_authorized=False;self.edge_points=[];self.edge_impulses=[]
        self.cut_contacts=0;self.cut_event=None;self.plan=None
        self.expected_right=self.kin.forward('right',self.right,self.base)

    def bind(self,simulation_view):
        super().bind(simulation_view)
        self.right_indices=[self.names.index(f'right_arm_{i}') for i in range(7)]
        self.right_palm=simulation_view.create_rigid_body_view(self.knife.wrist_path)
        if self.right_palm.count!=1: raise RuntimeError('Missing native right wrist')
        self.event_monitor.tool_contact=self._tool_contact

    def seam(self,frames):
        i=self.rig.cut_index
        half=np.linalg.norm(self.rig.chain_world[i+1]-self.rig.chain_world[i])/2
        return frames[i,:3,3]-half*frames[i,:3,2],frames[i,:3,2]

    def _tool_contact(self,robot,other,point,impulse):
        # Only the flat leading strip contacting the two seam-adjacent shaft
        # capsules is an expected tool load. Arc, camera, main stem and leaves
        # remain unwanted contacts. Positions come from the native callback.
        eligible=[self.rig.body_paths[i]+'/StemCollider'
            for i in (self.rig.cut_index-1,self.rig.cut_index)]
        if (not self.cut_authorized or robot!=self.knife.collider or other not in eligible
                or not self.knife.on_edge(point,self.edge_frame)):
            return False
        centre,axis=self.cut_frame
        if abs(np.dot(np.asarray(point)-centre,axis))>.003: return False
        self.edge_points.append(point);self.edge_impulses.append(impulse)
        self.cut_contacts+=1
        return True

    def prepare_step(self,frames):
        from .runtime import pose_matrices
        self.edge_frame=self.knife.frame(pose_matrices(self.right_palm.get_transforms())[0])
        self.cut_frame=self.seam(frames)
        self.edge_points=[];self.edge_impulses=[]

    def plan_cut(self,frames,left_q):
        """Bounded orientation search, then dense arm-pair screened IK paths.

        Native whole-scene guards remain essential: arm capsules alone do not
        certify the knife, cameras, foliage or gutter swept volume.
        """
        centre,axis=self.seam(frames)
        direction=-self.goal[:3,2];direction-=axis*np.dot(direction,axis);direction/=np.linalg.norm(direction)
        candidates=[]
        for degrees,normal_sign,wing in [(a,s,w) for w in (0.,-.028,.028) for s in (1,-1) for a in (0,45,-45,90,-90,135,-135,180)]:
            # Keep the curved support on the upward side during cutting;
            # rotating the parked assembly is not permission to cut with it.
            if normal_sign*axis[2]<0: continue
            angle=np.radians(degrees)
            d=direction*np.cos(angle)+np.cross(axis,direction)*np.sin(angle)
            desired=self.knife.wrist_for_edge(centre-.025*d,d,normal_sign*axis,wing)
            solution=self.kin.solve_pose('right',desired,self.right,self.base,maximum_evaluations=250)
            if not solution.succeeded: continue
            q=np.array(solution.joint_degrees)
            clearance=self.kin.inter_arm_clearance(left_q,q,self.base).clearance_m
            if clearance>=.01: candidates.append((np.linalg.norm(q-self.right),degrees,d,q,normal_sign,wing))
        failures=[]
        for _,angle,d,q,normal_sign,wing in sorted(candidates,key=lambda v:v[0]):
            approach=np.linspace(self.right,q,161)
            minimum=min(self.kin.inter_arm_clearance(left_q,row,self.base).clearance_m for row in approach)
            if minimum<.01:
                failures.append([angle,'approach_arm_clearance',minimum]);continue
            stroke=[];seed=q
            for offset in np.linspace(-.025,.012,75):
                desired=self.knife.wrist_for_edge(centre+offset*d,d,normal_sign*axis,wing)
                solution=self.kin.solve_pose('right',desired,seed,self.base,maximum_evaluations=250)
                if not solution.succeeded: break
                seed=np.asarray(solution.joint_degrees)
                clearance=self.kin.inter_arm_clearance(left_q,seed,self.base).clearance_m
                minimum=min(minimum,clearance)
                if clearance<.01: break
                stroke.append(seed)
            if len(stroke)!=75:
                failures.append([angle,'stroke_IK_or_arm_clearance',minimum]);continue
            self.plan=dict(approach=approach,stroke=np.asarray(stroke),direction=d,
                centre=centre.copy(),axis=axis.copy(),angle=angle,normal_sign=normal_sign,wing_m=wing,minimum_interarm_m=minimum)
            return
        raise RuntimeError('No bimanual arm-clearance path: '+str(failures))

    def command_right(self,phase,fraction):
        if phase=='park': q=self.right
        else:
            if self.plan is None or not 0<=fraction<=1: raise RuntimeError('Unplanned right command')
            path=self.plan[phase];where=fraction*(len(path)-1)
            low=min(int(where),len(path)-2);alpha=where-low
            q=(1-alpha)*path[low]+alpha*path[low+1]
        self.expected_right=self.kin.forward('right',q,self.base)
        self.targets[0,self.right_indices]=np.radians(q)
        self.robot.set_dof_position_targets(self.targets,self.index)

    def inspect_cut(self,dt,frames,held,slip):
        from .runtime import pose_matrices
        actual=pose_matrices(self.right_palm.get_transforms())[0]
        error=float(np.linalg.norm(actual[:3,3]-self.expected_right[:3,3]))
        if error>.012: raise RuntimeError(f'Right wrist tracking error {error:.5f} m')
        edge=self.knife.frame(actual);centre,axis=self.seam(frames)
        decision=self.cut_gate.observe(dt=dt,edge=edge,centre=centre,axis=axis,
            points=self.edge_points,impulses=self.edge_impulses,held=held and self.cut_authorized,slip=slip)
        if decision:
            self.cut_event=self.rig.release_from_blade(decision)
        return dict(edge_frame=edge.tolist(),right_tracking_error_m=error,
            edge_contact_count=len(self.edge_points),edge_force_n=sum(np.linalg.norm(v) for v in self.edge_impulses)/dt,
            gate_dwell_s=self.cut_gate.dwell,gate_travel_m=self.cut_gate.travel,cut_event=self.cut_event)

    def restore_authored_state(self):
        super().restore_authored_state()
        self.cut_gate=ShearGate(self.rig.source_target);self.cut_authorized=False
        self.cut_event=None;self.plan=None;self.cut_contacts=0
        self.expected_right=self.kin.forward('right',self.right,self.base)

    def setup_views(self,viewport):
        from pxr import Gf,UsdGeom
        super().setup_views(viewport)
        edge=self.knife.frame(self.kin.forward('right',self.right,self.base))
        centre=edge[:3,3]
        eye=centre+.23*edge[:3,2]+.18*edge[:3,1]-.12*edge[:3,0]
        path='/World/RightKnifeMountInspection'
        camera=UsdGeom.Camera.Define(self.stage,path)
        camera.CreateFocalLengthAttr(28.);camera.CreateClippingRangeAttr(Gf.Vec2f(.003,10))
        xf=UsdGeom.Xformable(camera);xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*centre),Gf.Vec3d(0,0,1)).GetInverse())
        self.views['Right knife mount']=path

    def report(self):
        result=super().report()
        result.update(right_arm='original_fitted_knife_guarded_native_joint_drives',
            knife_mount=self.knife_mount,
            cut_model='measured_contact_seam_failure_not_calibrated_tissue_cutting')
        if self.plan is not None:
            result['cut_plan']={k:(v.tolist() if isinstance(v,np.ndarray) else v)
                for k,v in self.plan.items() if k not in ('approach','stroke')}
        return result
