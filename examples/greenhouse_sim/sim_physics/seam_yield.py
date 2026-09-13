"""Opt-in, uncalibrated local cutting process-zone experiment.

The linear 1000 N/m obstacle cannot sustain a 0.3 mm loaded stroke inside the
existing force envelope. After qualified edge loading, an irreversible secant
softening law permits measured advance at approximately the yield traction.
This is NOT volumetric tissue fracture or a clean-cut quality measurement.
It cannot authorize release, move bodies, edit collision geometry, or advance
damage from time, a desired pose, an unaligned blade or a moving stem alone.
"""
import math
import numpy as np


class EdgeYield:
    def __init__(self):
        self.step=0;self.origin=None;self.progress=0.;self.stiffness=1000.

    def observe(self,record,*,step):
        from .knife import DOWNWARD_CUT_MODEL
        if type(step) is not int or step!=self.step+1 or record.get('native_guards_passed') is not True:
            raise RuntimeError('Fresh guard-accepted post-fetch yield sample required')
        k=record['knife'];g=k['gate_diagnostic']
        if k.get('cut_model')!=DOWNWARD_CUT_MODEL or k.get('raw_normal_rows_complete') is not True:
            raise RuntimeError('Complete downward-edge contact contract required')
        self.step=step
        receipt=dict(model='qualified_edge_secant_softening_v1',sample_step=step,
            stiffness_n_m=self.stiffness,loaded_advance_m=self.progress,
            cut_authorized=False,calibrated=False,fracture_energy_calibrated=False,
            native_material_readback=False,geometry_changed=False)
        if record.get('cut') is True:return dict(receipt,state='released_no_further_softening')
        # Reuse the unsmoothed, complete existing load/alignment/support gates.
        # A low/zero load, wrong face or lost grasp never starts/advances damage.
        if g.get('state')!='qualifying_contact' or g.get('failed_conditions')!=[]:
            return dict(receipt,state='not_qualified_no_material_change')
        edge=np.asarray(k['edge_frame'],float);centre=np.asarray(record['seam_world'],float)
        load=k['edge_signed_resistance_n'];upper=k['tool_contact_upper_bound_n']
        if (edge.shape!=(4,4) or centre.shape!=(3,) or not np.isfinite(np.r_[edge.flat,centre,load,upper]).all()
                or not .2<=load<=upper<=.5 or edge[2,0]<np.cos(np.radians(5))
                or not np.allclose(edge[:3,:3].T@edge[:3,:3],np.eye(3),rtol=0,atol=1e-5)
                or np.linalg.det(edge[:3,:3])<=0):raise RuntimeError('Invalid qualified yield geometry/load')
        relative=edge[:3,3]-centre
        if self.origin is None:
            if load<.23:return dict(receipt,state='elastic_loading')
            rows=[r for r in k['raw_normal_rows'] if r.get('eligible') is True
                  and np.dot(r['impulse_on_knife'], -edge[:3,0])<0]
            depths=[-float(r['separation']) for r in rows]
            if not depths or not np.isfinite(depths).all() or not .0001<=max(depths)<=.0005:
                raise RuntimeError('Measured positive seam indentation required for yield origin')
            self.origin=(relative.copy(),float(edge[2,3]),-edge[:3,0].copy(),max(depths),
                         float(g['leading_normal_cosine']))
            return dict(receipt,state='yield_origin_measured',origin_indentation_m=max(depths))
        origin,z,direction,depth,cosine=self.origin
        travel=min(float(np.dot(relative-origin,direction)),z-float(edge[2,3]))
        if not math.isfinite(travel) or travel>.0005:
            raise RuntimeError('Yield process zone exceeds bounded 0.5 mm measured advance')
        # Maximum NET advance, not summed positive jitter; no healing on return.
        self.progress=max(self.progress,travel)
        proposed=1000.*depth/(depth+cosine*self.progress)
        if not 250.<=proposed<=1000.:
            raise RuntimeError('Yield secant stiffness outside bounded process-zone law')
        self.stiffness=min(self.stiffness,proposed)
        return dict(receipt,state='measured_edge_softening',stiffness_n_m=self.stiffness,
            loaded_advance_m=self.progress,origin_indentation_m=depth,
            law='k0*d0/(d0+normal_projection*net_loaded_advance)',
            constitutive_property='positive_nonincreasing_secant_stiffness_no_force_or_pose_injection',
            coupled_native_energy_balance_verified=False)


class NativeEdgeYield:
    """One cached material update per fresh sample, before the next native step.

Uses the installed Isaac compliant-contact USD backend. USD readback is NOT
called a native material readback; actual response needs native force/travel
qualification. Only two already-bound local shaft surfaces are affected.
"""
    def __init__(self,rig,fixture):
        from pxr import UsdShade
        from .blade_contacts import CROSSBAR_EDGE
        if fixture.knife.edge_mode!=CROSSBAR_EDGE or rig.cut:
            raise ValueError('Intact actual crossbar diagnostic required')
        self.rig=rig;self.fixture=fixture;self.law=EdgeYield();self.pending=None
        self.material_path=rig.root+'/ExperimentalSeamContactMaterial'
        self.prim=rig.stage.GetPrimAtPath(self.material_path)
        if not self.prim:raise ValueError('Original explicit seam compliance material required')
        self.attr=self.prim.GetAttribute('physxMaterial:compliantContactStiffness')
        self.shapes=[rig.stage.GetPrimAtPath(rig.body_paths[i]+'/StemCollider')
                     for i in (rig.cut_index-1,rig.cut_index)]
        for shape in self.shapes:
            bound,_=UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial(materialPurpose='physics')
            if not bound or str(bound.GetPath())!=self.material_path:raise ValueError('Exact seam material bindings required')
        self.current=1000.;self.updates=0;self.last_applied_step=0;self._check()

    def _check(self):
        if (self.attr.Get()!=self.current
                or self.prim.GetAttribute('physxMaterial:compliantContactDamping').Get()!=2.
                or self.prim.GetAttribute('physxMaterial:compliantContactAccelerationSpring').Get() is not False):
            raise RuntimeError('Yield material changed outside owned constitutive update')

    def observe(self,record,*,step):
        if self.pending is not None:raise RuntimeError('Previous material sample was not consumed')
        self._check()
        # Softening must not be used to get a camera/finger past this material.
        allowed={str(p.GetPath()) for p in self.shapes};knife=self.fixture.knife.collider
        for a,b,force in ([] if record.get('cut') is True else record.get('native_contact_pairs_n',[])):
            if (a in allowed or b in allowed) and force>.001 and {a,b}-{knife}-allowed:
                raise RuntimeError('Foreign contact on cutting process zone; no material update')
        evidence=self.law.observe(record,step=step)
        record['seam_yield']=dict(evidence,applied_stiffness_n_m=self.current,material_updates=self.updates)
        self.pending=(step,evidence['stiffness_n_m'])

    def apply(self,*,step):
        from pxr import Usd
        if step==0 and self.pending is None:return
        if self.pending is None or self.pending[0]!=step or step<=self.last_applied_step:
            raise RuntimeError('Stale/reused yield material update refused')
        self._check();_,desired=self.pending
        desired=float(np.float32(desired))
        if desired>self.current+1e-4 or not 250.<=desired<=1000.:raise RuntimeError('Invalid irreversible material update')
        # Deadband reduces USD traffic; NEVER goes ahead of measured softening.
        if self.current-desired>=.5:
            with Usd.EditContext(self.rig.stage,self.rig.stage.GetSessionLayer()):
                if not self.attr.Set(desired):raise RuntimeError('Yield material write failed')
            self.current=desired;self.updates+=1;self._check()
        self.pending=None;self.last_applied_step=step
