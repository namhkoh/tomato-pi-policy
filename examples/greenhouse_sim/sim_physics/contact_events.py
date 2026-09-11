"""Native normal + friction impulse upper bounds, never cancelling contacts."""
import math
import operator
from dataclasses import dataclass


NATIVE_NORMAL_ROW_CONTRACT='physx_contact_data_original_order_v1'


@dataclass(frozen=True)
class NativeNormalContact:
    """Copied NORMAL row: n and J act on collider0; n points 1 -> 0.

    J may have a negative projection on n. Neither this container nor the
    legacy load classifier changes that sign or turns friction into normal J.
    """
    collider0: str
    collider1: str
    point: tuple
    normal: tuple
    impulse: tuple
    separation: float

    def __post_init__(self):
        if not all(isinstance(p,str) and p.startswith('/') for p in (self.collider0,self.collider1)):
            raise ValueError('Exact original collider paths required')
        for name in ('point','normal','impulse'):
            value=_vectors([getattr(self,name)],name)[0]
            object.__setattr__(self,name,tuple(float(v) for v in value))
        if not math.isfinite(self.separation): raise ValueError('Nonfinite native contact separation')
        object.__setattr__(self,'separation',float(self.separation))


def original_order_tool_contact(callback):
    """Explicit opt-in to one immutable raw row, NEVER signature inference.

    Unmarked six-argument callbacks retain legacy load classification only:
    their sorted names do not establish the frame/sign of their raw vectors.
    """
    callback.normal_row_contract=NATIVE_NORMAL_ROW_CONTRACT
    return callback


def _sum(values):
    try:
        total=math.fsum(values)
    except OverflowError as exc:
        raise ValueError('Overflowing native contact impulse') from exc
    if not math.isfinite(total): raise ValueError('Overflowing native contact impulse')
    return total


def _vectors(values,label):
    result=[]
    for value in values:
        if len(value)!=3 or not all(math.isfinite(v) for v in value):
            raise ValueError('Nonfinite native '+label)
        result.append(tuple(value))
    return result


def _span(offset,count,size,label):
    if isinstance(count,bool):
        raise ValueError('Invalid native '+label+' buffer range')
    try:
        count=operator.index(count)
    except TypeError as exc:
        raise ValueError('Invalid native '+label+' buffer range') from exc
    if count<0: raise ValueError('Invalid native '+label+' buffer range')
    # Lost-contact headers may leave an unused offset set to a sentinel.
    # Validate the count first and never interpret an empty span's offset.
    if count==0: return slice(0,0)
    if isinstance(offset,bool): raise ValueError('Invalid native '+label+' buffer range')
    try:
        start=operator.index(offset)
    except TypeError as exc:
        raise ValueError('Invalid native '+label+' buffer range') from exc
    stop=start+count
    if start<0 or stop>size:
        raise ValueError('Invalid native '+label+' buffer range')
    return slice(start,stop)


class ContactEvents:
    _buckets=('allowed_target','allowed_tool','allowed_floor','unwanted','self')

    def __init__(self,*,robot_root,target_root,fingers,floor_root,full_contact_observer=None):
        self.robot_root=robot_root;self.target_root=target_root
        self.fingers=set(fingers);self.floor_root=floor_root
        self.subscription=None;self.paths={};self.total_events=0;self.error=None
        self.native_full_contact_reporting=False;self.native_friction_type=None
        self.tool_contact=None
        self.normal_contact_observer=None
        self.full_contact_observer=full_contact_observer
        self.begin_step()

    def _full_contact_fault(self,exc):
        if self.error is None:self.error=str(exc) or type(exc).__name__
        # Even validation failures before row delivery invalidate a partial stream.
        try:self.full_contact_observer.invalidate(exc)
        except Exception:pass  # The monitor fault remains authoritative and latched.

    def begin_step(self):
        if self.full_contact_observer is not None:
            try:
                if self.error is not None:raise RuntimeError(self.error)
                self.full_contact_observer.begin_step()
            except Exception as exc:
                self._full_contact_fault(exc)
                raise
        if self.normal_contact_observer is not None:
            try: self.normal_contact_observer.begin_step()
            except Exception as exc:
                self.error=str(exc)
                if self.full_contact_observer is not None:self._full_contact_fault(exc)
                raise
        self.pairs={};self.normal_pairs={};self.friction_pairs={};self._pair_states={}
        self.normal_impulse=0.;self.friction_impulse=0.
        for kind in self._buckets:
            setattr(self,kind+'_impulse',0.)
            setattr(self,kind+'_friction_impulse',0.)
        self.minimum_tool_separation_m=0.

    def _kind(self,first,second):
        r0=first.startswith(self.robot_root+'/');r1=second.startswith(self.robot_root+'/')
        if not r0 and not r1: return None,None,None
        if r0 and r1: return 'self',None,None
        robot,other=(first,second) if r0 else (second,first)
        body=self.robot_root+'/'+robot[len(self.robot_root)+1:].split('/')[0]
        if body in self.fingers and other.startswith(self.target_root+'/'):
            return 'allowed_target',robot,other
        if self.floor_root and (other==self.floor_root or other.startswith(self.floor_root+'/')) and body in {
                self.robot_root+'/base',self.robot_root+'/wheel_l',self.robot_root+'/wheel_r'}:
            return 'allowed_floor',robot,other
        return 'tool_candidate',robot,other

    def consume(self,first,second,impulses,points=None,normals=None,separations=None,
                *,friction_impulses=(),friction_points=None):
        """Pure accounting; friction is NEVER passed to the normal-edge callback."""
        try:
            if self.full_contact_observer is not None and self.error is not None:
                raise RuntimeError(self.error)
            return self._consume(first,second,impulses,points,normals,separations,
                friction_impulses=friction_impulses,friction_points=friction_points)
        except Exception as exc:
            if self.full_contact_observer is not None:self._full_contact_fault(exc)
            raise

    def _consume(self,first,second,impulses,points,normals,separations,
                 *,friction_impulses,friction_points):
        impulses=_vectors(impulses,'contact impulse')
        friction_impulses=_vectors(friction_impulses,'friction impulse')
        for values,count,label in ((points,len(impulses),'contact position'),
                (normals,len(impulses),'contact normal'),
                (friction_points,len(friction_impulses),'friction position')):
            if values is not None:
                if len(values)!=count: raise ValueError('Native '+label+'/impulse count mismatch')
                _vectors(values,label)
        if separations is not None:
            if len(separations)!=len(impulses):
                raise ValueError('Native separation/impulse count mismatch')
            if not all(math.isfinite(v) for v in separations):
                raise ValueError('Nonfinite native contact separation')
        if self.full_contact_observer is not None:
            if impulses and (points is None or normals is None or separations is None):
                raise ValueError('Full contact observer requires complete normal geometry')
            if friction_impulses and friction_points is None:
                raise ValueError('Full contact observer requires friction anchor positions')
            # Separate COPIED original-order rows, before robot-only classification.
            # Never infer anchor normals, flip vectors, or borrow normal eligibility.
            for i,impulse in enumerate(impulses):
                self.full_contact_observer.add_contact(dict(collider0=first,collider1=second,
                    kind='normal',point_world_m=tuple(points[i]),
                    normal_on_0=tuple(normals[i]),separation_m=separations[i],
                    impulse_on_0_ns=tuple(impulse)))
            for i,impulse in enumerate(friction_impulses):
                self.full_contact_observer.add_contact(dict(collider0=first,collider1=second,
                    kind='friction',point_world_m=tuple(friction_points[i]),
                    impulse_on_0_ns=tuple(impulse)))
        # Preserve original collider order and pass NORMAL rows only. This
        # evidence observer cannot reclassify loads or authorize a tool cut.
        if self.normal_contact_observer is not None and impulses:
            try:
                if points is None or normals is None or separations is None:
                    raise ValueError('Normal grasp observer requires full native contact rows')
                for point,normal,impulse,separation in zip(points,normals,impulses,separations,strict=True):
                    self.normal_contact_observer.add_contact(first,second,point,normal,impulse,separation)
            except Exception as exc:
                self.error=str(exc)
                raise
        magnitudes=[math.hypot(*v) for v in impulses]
        friction=_sum(math.hypot(*v) for v in friction_impulses)
        magnitude=_sum([*magnitudes,friction])
        kind,robot,other=self._kind(first,second)
        if kind is None or not (impulses or friction_impulses): return
        key=tuple(sorted((first,second)))
        state=self._pair_states.setdefault(key,dict(kind=kind,
            normal={k:0. for k in self._buckets},friction=0.,normal_count=0,all_tool_normals=True))
        if magnitude: self.total_events+=1
        state['friction']=_sum((state['friction'],friction))
        if kind!='tool_candidate':
            state['normal'][kind]=_sum([state['normal'][kind],*magnitudes])
        else:
            evidence=points is not None and normals is not None and separations is not None and self.tool_contact is not None
            contract=getattr(self.tool_contact,'normal_row_contract',None)
            if contract not in (None,NATIVE_NORMAL_ROW_CONTRACT):
                raise ValueError('Unknown tool normal-row contract')
            for i,impulse in enumerate(impulses):
                # Even a rejected zero-impulse normal prevents friction from
                # borrowing eligibility from another contact on this pair.
                accepted=False
                if evidence:
                    if contract==NATIVE_NORMAL_ROW_CONTRACT:
                        accepted=bool(self.tool_contact(NativeNormalContact(first,second,
                            points[i],normals[i],impulse,separations[i])))
                    else:
                        accepted=bool(self.tool_contact(robot,other,points[i],impulse,normals[i],separations[i]))
                state['normal_count']+=1
                state['all_tool_normals'] &= accepted
                bucket='allowed_tool' if accepted else 'unwanted'
                state['normal'][bucket]=_sum((state['normal'][bucket],magnitudes[i]))
                if accepted:
                    self.minimum_tool_separation_m=min(self.minimum_tool_separation_m,separations[i])
        self._refresh_totals()

    def _refresh_totals(self):
        # Recompute from sparse pair state so later headers/callbacks can revoke
        # earlier tool-friction permission without subtraction/cancellation.
        normal={k:[] for k in self._buckets};friction={k:[] for k in self._buckets}
        for key,state in self._pair_states.items():
            for kind,value in state['normal'].items(): normal[kind].append(value)
            kind=state['kind']
            if kind=='tool_candidate':
                kind='allowed_tool' if state['normal_count'] and state['all_tool_normals'] else 'unwanted'
            friction[kind].append(state['friction'])
            self.normal_pairs[key]=_sum(state['normal'].values())
            self.friction_pairs[key]=state['friction']
            self.pairs[key]=_sum((self.normal_pairs[key],self.friction_pairs[key]))
        for kind in self._buckets:
            f=_sum(friction[kind]);n=_sum(normal[kind])
            setattr(self,kind+'_friction_impulse',f)
            setattr(self,kind+'_impulse',_sum((n,f)))
        self.normal_impulse=_sum(self.normal_pairs.values())
        self.friction_impulse=_sum(self.friction_pairs.values())
        _sum((self.normal_impulse,self.friction_impulse))

    def subscribe(self):
        observer=self.full_contact_observer
        self.close()
        self.full_contact_observer=observer
        try:
            from omni.physx import get_physx_simulation_interface
            from omni.usd import get_context
            from pxr import PhysxSchema,UsdPhysics
            stage=get_context().get_stage()
            if stage is None: raise RuntimeError('Missing native contact stage')
            scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
            if not scenes: raise RuntimeError('Missing native contact physics scene')
            # Installed PhysX schema defaults to patch, the only mode with
            # friction-anchor reporting. Do not silently fall back to normals.
            for scene in scenes:
                mode=PhysxSchema.PhysxSceneAPI(scene).GetFrictionTypeAttr().Get()
                if mode!='patch': raise RuntimeError('Native full contact reporting requires patch friction')
            self.subscription=get_physx_simulation_interface().subscribe_full_contact_report_events(self._callback)
            if self.subscription is None: raise RuntimeError('Missing native full contact subscription')
            self.native_friction_type='patch';self.native_full_contact_reporting=True
        except Exception as exc:
            self.error=str(exc)
            raise

    def _callback(self,headers,data,friction_anchors):
        def vector(value): return (float(value.x),float(value.y),float(value.z))
        try:
            from pxr import PhysicsSchemaTools
            def path(value):
                if value not in self.paths: self.paths[value]=str(PhysicsSchemaTools.intToSdfPath(value))
                return self.paths[value]
            for header in headers:
                contacts=data[_span(header.contact_data_offset,header.num_contact_data,len(data),'contact')]
                anchors=friction_anchors[_span(header.friction_anchors_offset,header.num_friction_anchors_data,len(friction_anchors),'friction')]
                if not contacts and not anchors: continue
                # Friction-only headers still carry load and must not be skipped.
                self.consume(path(header.collider0),path(header.collider1),
                    [vector(c.impulse) for c in contacts],[vector(c.position) for c in contacts],
                    [vector(c.normal) for c in contacts],[float(c.separation) for c in contacts],
                    friction_impulses=[vector(c.impulse) for c in anchors],
                    friction_points=[vector(c.position) for c in anchors])
        except Exception as exc:
            # Callback exceptions must propagate to the explicit control guard.
            if self.full_contact_observer is not None:
                self._full_contact_fault(exc)
                raise
            self.error=str(exc)

    def measurements(self,dt):
        if dt<=0 or not math.isfinite(dt): raise ValueError('Invalid contact timestep')
        if self.error is not None: raise RuntimeError(self.error)
        result={kind+'_contact_n':getattr(self,kind+'_impulse')/dt for kind in self._buckets}
        result.update({kind+'_friction_n':getattr(self,kind+'_friction_impulse')/dt for kind in self._buckets})
        result.update(normal_contact_n=self.normal_impulse/dt,friction_contact_n=self.friction_impulse/dt,
            total_contact_upper_bound_n=_sum((self.normal_impulse,self.friction_impulse))/dt)
        if not all(math.isfinite(v) for v in result.values()): raise ValueError('Overflowing native contact force')
        result.update(minimum_tool_separation_m=self.minimum_tool_separation_m,
            native_contact_events=self.total_events,native_full_contact_reporting=self.native_full_contact_reporting,
            native_patch_friction=self.native_friction_type=='patch',
            contact_magnitude_is_upper_bound=True,friction_in_edge_evidence=False)
        return result

    def close(self):
        self.subscription=None
        self.normal_contact_observer=None
        self.full_contact_observer=None
        self.native_full_contact_reporting=False;self.native_friction_type=None
