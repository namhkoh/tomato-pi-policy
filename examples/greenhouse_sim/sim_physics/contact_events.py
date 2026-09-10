"""Sparse native contact accounting. Never cancels opposing contact impulses."""
import math


class ContactEvents:
    def __init__(self,*,robot_root,target_root,fingers,floor_root):
        self.robot_root=robot_root;self.target_root=target_root
        self.fingers=set(fingers);self.floor_root=floor_root
        self.subscription=None;self.paths={};self.total_events=0;self.error=None
        self.begin_step()

    def begin_step(self):
        self.pairs={}
        self.allowed_target_impulse=0.;self.unwanted_impulse=0.;self.self_impulse=0.

    def consume(self,first,second,impulses):
        """Pure accounting entry point also used by regression tests."""
        r0=first.startswith(self.robot_root+'/');r1=second.startswith(self.robot_root+'/')
        if not r0 and not r1: return
        magnitude=0.
        for value in impulses:
            if len(value)!=3 or not all(math.isfinite(v) for v in value):
                raise ValueError('Nonfinite native contact impulse')
            magnitude+=math.hypot(*value)
        if not math.isfinite(magnitude): raise ValueError('Overflowing native contact impulse')
        if not magnitude: return
        self.total_events+=1
        key=tuple(sorted((first,second)))
        self.pairs[key]=self.pairs.get(key,0.)+magnitude
        if r0 and r1:
            self.self_impulse+=magnitude
            return
        robot,other=(first,second) if r0 else (second,first)
        body=self.robot_root+'/'+robot[len(self.robot_root)+1:].split('/')[0]
        if body in self.fingers and other.startswith(self.target_root+'/'):
            self.allowed_target_impulse+=magnitude
        elif self.floor_root and (other==self.floor_root or other.startswith(self.floor_root+'/')) and body in {
                self.robot_root+'/base',self.robot_root+'/wheel_l',self.robot_root+'/wheel_r'}:
            pass
        else:
            # Includes finger hits on neighbors/gutters and non-finger target
            # contact, not only the selected plant or a pre-enumerated subset.
            self.unwanted_impulse+=magnitude

    def subscribe(self):
        from omni.physx import get_physx_simulation_interface
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(self._callback)

    def _callback(self,headers,data):
        from pxr import PhysicsSchemaTools
        def path(value):
            if value not in self.paths: self.paths[value]=str(PhysicsSchemaTools.intToSdfPath(value))
            return self.paths[value]
        try:
            for header in headers:
                if header.num_contact_data==0: continue
                start=header.contact_data_offset;stop=start+header.num_contact_data
                if start<0 or stop>len(data): raise ValueError('Invalid native contact buffer range')
                impulses=[(float(c.impulse.x),float(c.impulse.y),float(c.impulse.z)) for c in data[start:stop]]
                self.consume(path(header.collider0),path(header.collider1),impulses)
        except Exception as exc:
            # Callback exceptions must propagate to the explicit control guard.
            self.error=str(exc)

    def measurements(self,dt):
        if dt<=0 or not math.isfinite(dt): raise ValueError('Invalid contact timestep')
        if self.error: raise RuntimeError(self.error)
        return dict(unwanted_contact_n=self.unwanted_impulse/dt,
            self_contact_n=self.self_impulse/dt,allowed_target_contact_n=self.allowed_target_impulse/dt,
            native_contact_events=self.total_events)

    def close(self):
        self.subscription=None
