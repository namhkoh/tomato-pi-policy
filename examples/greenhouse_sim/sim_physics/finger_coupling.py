"""Explicit compliant equal/opposite-jaw mechanism trial, NOT a plant weld.

The source URDF provides two independent slides; the physical gripper has one
opening DOF. An ideal 1:1 mechanism is an engineering hypothesis. Backlash and
transmission stiffness are uncalibrated. Native contact/effort/slip limits stay.
"""
import numpy as np


def author(stage,fixture):
    from pxr import PhysxSchema
    result=_author(stage,fixture.root,lambda prim:PhysxSchema.PhysxMimicJointAPI.Apply(prim,'rotX'))
    fixture.finger_coupling_trial=result
    fixture.force_closer.physical_gear_coupling_modeled=True
    return result


def _author(stage,root,make_api):
    from pxr import Usd,UsdGeom,UsdPhysics
    if UsdGeom.GetStageMetersPerUnit(stage)!=1. or UsdGeom.GetStageUpAxis(stage)!='Z':
        raise ValueError('Metre Z-up stage required for the jaw mechanism trial')
    joints=[stage.GetPrimAtPath(root+'/joints/gripper_finger_l'+str(i)) for i in (1,2)]
    for i,prim in enumerate(joints):
        joint=UsdPhysics.PrismaticJoint(prim)
        if (not joint or not prim.IsActive()
                or list(joint.GetBody0Rel().GetTargets())!=[root+'/ee_left']
                or list(joint.GetBody1Rel().GetTargets())!=[root+'/ee_finger_l'+str(i+1)]
                or any(str(s).startswith('PhysxMimicJointAPI') for s in prim.GetAppliedSchemas())):
            raise ValueError('Original independent same-palm prismatic fingers required')
        lo,hi=float(joint.GetLowerLimitAttr().Get()),float(joint.GetUpperLimitAttr().Get())
        if not np.allclose([lo,hi],[-.05,0.] if i==0 else [0.,.05],rtol=0,atol=1e-7):
            raise ValueError('Expected original opposite-signed 50 mm slide limits')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        api=make_api(joints[1])
        if not api:raise RuntimeError('Native articulation mimic schema unavailable')
        if not api.CreateReferenceJointRel().SetTargets([joints[0].GetPath()]):
            raise RuntimeError('Mimic reference authoring failed')
        attrs=[(api.CreateGearingAttr(),1.),(api.CreateOffsetAttr(),0.),
               (api.CreateNaturalFrequencyAttr(),480.),(api.CreateDampingRatioAttr(),1.2)]
        for attr,value in attrs:
            if not attr.Set(value) or abs(float(attr.Get())-value)>1e-6:
                raise RuntimeError('Mimic parameter authoring/readback failed')
    return dict(model='experimental_compliant_parallel_jaw_mimic_v1',
        joint_paths=[str(p.GetPath()) for p in joints],equation='q_left_2 + q_left_1 = 0',
        gearing=1.,offset_m=0.,natural_frequency=480.,damping_ratio=1.2,
        parameter_basis='uncalibrated_transmission_prior_dt_times_frequency_one',
        plant_joint_or_contact_changed=False,actuator_caps_changed=False,
        source_layer_changed=False,usd_parameters_checked=True,native_parameters_readback=False,
        tissue_or_grasp_qualified=False)


class CouplingMonitor:
    """Measured jaw sum, not a controller or native-constraint authenticity claim."""
    def __init__(self):self.step=0;self.maximum=0.
    def observe(self,position,velocity,*,step):
        q,v=np.asarray(position,float),np.asarray(velocity,float)
        if (type(step) is not int or step!=self.step+1 or q.shape!=(2,) or v.shape!=(2,)
                or not np.isfinite([q,v]).all()):raise RuntimeError('Fresh finite native jaw state required')
        error=abs(float(q.sum()))
        if error>.0005:raise RuntimeError('Jaw mechanism residual exceeds experimental 0.5 mm bound')
        self.step=step;self.maximum=max(self.maximum,error)
        return dict(model='measured_parallel_jaw_relation_v1',step=step,
            position_sum_m=float(q.sum()),velocity_sum_m_s=float(v.sum()),
            maximum_abs_position_sum_m=self.maximum,limit_m=.0005,
            native_constraint_authenticity_verified=False,grasp_verified=False)
