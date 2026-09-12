import pytest
from sim_physics.shaft_grasp_native_test import fixture,bind


def test_cache_reuses_topology_but_rechecks_each_authored_joint_change():
    f=fixture();a=bind(f)
    first=a._connected();revision=a._connected_revision
    first.clear()
    assert a._connected() and a._connected_revision==revision
    path,joint,pair=a.joints[0]
    joint.GetJointEnabledAttr().Set(False)
    assert a._joint_revision>revision
    assert pair not in a._connected()
    joint.GetJointEnabledAttr().Set(True)
    assert pair in a._connected()
    joint.GetBody0Rel().SetTargets(['/T/Unknown'])
    with pytest.raises(ValueError,match='endpoints changed'):a._connected()
    a.close()


def test_topology_resync_and_closed_notice_cannot_reuse_cached_success():
    f=fixture();a=bind(f);a._connected()
    f.stage.RemovePrim(a.joints[0][0])
    with pytest.raises(ValueError):a._connected()
    a.close()
    f=fixture();a=bind(f);a._connected();a.close()
    with pytest.raises(ValueError):a._connected()
