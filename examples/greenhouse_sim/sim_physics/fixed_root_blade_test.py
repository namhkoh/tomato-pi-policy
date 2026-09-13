import pytest
from pxr import UsdPhysics
from .plant_test import native
from .plant import build
from .knife import ShearGate
from .knife_test import sample
from .root_transition import FixedRootTransition


def event(rig):
    gate=ShearGate(rig.source_target)
    return next(e for i in range(12) if (e:=sample(gate,i*.0001)) is not None)


def test_fixed_root_needs_checked_adapter_even_with_valid_blade_evidence(native):
    stage,record=native;rig=build(stage,record,'SubStem_56',constraint_mode='fixed_articulation')
    with pytest.raises(ValueError,match='checked fixed-root'):
        rig.release_from_blade(event(rig))
    assert not rig.cut and UsdPhysics.Joint.Get(stage,rig.cut_joint_path).GetJointEnabledAttr().Get()


def test_invalid_blade_evidence_never_invokes_transition(native):
    stage,record=native;rig=build(stage,record,'SubStem_56',constraint_mode='fixed_articulation')
    adapter=object.__new__(FixedRootTransition);adapter.rig=rig;called=[]
    adapter.release=lambda:called.append(True)
    bad=event(rig);bad['grasp_slip_m']=.01
    with pytest.raises(ValueError):rig.release_from_blade(bad,transition=adapter)
    assert called==[] and not rig.cut


def test_adapter_for_another_rig_cannot_release(native):
    stage,record=native;rig=build(stage,record,'SubStem_56',constraint_mode='fixed_articulation')
    adapter=object.__new__(FixedRootTransition);adapter.rig=object()
    with pytest.raises(ValueError,match='checked fixed-root'):
        rig.release_from_blade(event(rig),transition=adapter)
    assert not rig.cut
