"""Anonymous USD + stub tensor tests; no SimulationApp, native actor or step."""
import importlib.abc
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from sim_physics import shaft_grasp_native as mod


def pose(x=0., z=0., reverse=False):
    m = np.eye(4)
    m[:3, 3] = [x, 0, z]
    if reverse:
        m[:3, :3] = np.diag([-1., -1., 1.])
    return m


def matrix(prim, value):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(Gf.Matrix4d(np.asarray(value).T.tolist()))


def collision(prim):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.0005)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)


class View:
    def __init__(self, finger, selected):
        self.sensor_count = self.filter_count = 1
        self.sensor_paths, self.filter_paths = [finger], [selected]
        self.max_contact_data_count = 8
        self.calls = []
        self.force = np.zeros(3)
        self.count = 0
        self.start = 0
        self.corrupt = None

    def get_contact_data(self, dt):
        self.calls.append(dt)
        size = self.max_contact_data_count
        f, p, n, d = np.zeros((size, 1)), np.zeros((size, 3)), np.zeros((size, 3)), np.zeros((size, 1))
        if self.count and self.start < size:
            f[self.start, 0] = np.linalg.norm(self.force)
            n[self.start] = self.force / np.linalg.norm(self.force) if np.any(self.force) else [1, 0, 0]
        c, s = np.array([[self.count]]), np.array([[self.start]])
        if self.corrupt == "nan":
            f[self.start, 0] = np.nan
        if self.corrupt == "negative":
            f[self.start, 0] = -.1
        if self.corrupt == "normal":
            n[self.start] = [2, 0, 0]
        if self.corrupt == "layout":
            f = f[:, 0]
        return f, p, n, d, c, s


def fixture(*, selected=2, leaf_selected=False):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/T")
    UsdGeom.Xform.Define(stage, "/R")
    bodies, frames = [], []
    for i in range(5):
        path = "/T/" + ("Support" if i == 0 else "Branch") + f"/Segment_{i:03d}"
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        frame = pose(z=(i-2)*.02)
        matrix(body, frame)
        body.CreateAttribute("tomato:sourceTarget", Sdf.ValueTypeNames.String).Set("seed101_full/SubStem_41")
        cap = UsdGeom.Capsule.Define(stage, path + "/StemCollider")
        cap.CreateAxisAttr("Z"); cap.CreateRadiusAttr(.003); cap.CreateHeightAttr(.014)
        collision(cap.GetPrim())
        bodies.append(path); frames.append(frame)
    for i in range(2, 5):
        joint = UsdPhysics.SphericalJoint.Define(stage, f"/T/Branch/Joint_{i:03d}")
        joint.CreateBody0Rel().SetTargets([bodies[i-1]])
        joint.CreateBody1Rel().SetTargets([bodies[i]])
        joint.CreateJointEnabledAttr(True)
    fingers, fingerframes = [], []
    for i in range(2):
        path = f"/R/ee_finger_l{i+1}"
        body = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        frame = pose(x=.006 if i == 0 else -.006, z=.0295, reverse=i == 1)
        matrix(body, frame)
        cube = UsdGeom.Cube.Define(stage, path + "/restored_collisions/contact_proxy")
        cube.CreateSizeAttr(1.)
        local = pose(x=.005, z=-.0295)
        local[:3, :3] = np.diag([.016, .032, .062])
        matrix(cube.GetPrim(), local); collision(cube.GetPrim())
        fingers.append(path); fingerframes.append(frame)
    if leaf_selected:
        leaf = UsdGeom.Cube.Define(stage, bodies[selected] + "/Leaf_Test/LeafCollider")
        collision(leaf.GetPrim())
    rig = SimpleNamespace(root="/T", body_paths=bodies, cut_index=1,
                          source_target="seed101_full/SubStem_41")
    views = [View(p, bodies[selected]) for p in fingers]
    return SimpleNamespace(stage=stage, rig=rig, frames=np.array(frames),
        fingerframes=np.array(fingerframes), fingers=fingers, views=views, selected=selected)


def bind(f, **kwargs):
    return mod.ShaftGraspNative(f.stage, f.rig, selected_index=f.selected,
        finger_paths=f.fingers, contact_views=f.views, **kwargs)


def add(adapter, f, finger, segment=2, force=.03, *, reverse=False, other=None, point=None):
    direction = np.array([1. if finger == 0 else -1., 0, 0])
    point = [direction[0]*.003, 0, (segment-2)*.02] if point is None else point
    a = adapter.pad_paths[finger]
    b = f.rig.body_paths[segment] + "/StemCollider" if other is None else other
    if reverse:
        a, b = b, a
        direction = -direction
    adapter.add_contact(a, b, point, direction, .01*force*direction, 0.)


def tensor(f, finger, force=.03):
    f.views[finger].force = np.array([force if finger == 0 else -force, 0, 0])
    f.views[finger].count = 1 if force else 0


def evaluate(a, f, **kwargs):
    return a.evaluate(.01, f.frames, f.fingerframes, **kwargs)


def test_import_does_not_load_native_runtime():
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'omni', 'isaacsim', 'carb', 'pxr'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
sys.path[:0] = sys.argv[1:]
import sim_physics.shaft_grasp_native
"""
    r = subprocess.run([sys.executable, "-I", "-B", "-c", script,
        str(Path(mod.__file__).resolve().parents[1]), str(Path(np.__file__).resolve().parents[1])],
        capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert r.returncode == 0, r.stdout + r.stderr


def test_binding_exact_geometry_without_authoring_or_native_calls():
    f = fixture()
    before = f.stage.GetRootLayer().ExportToString()
    a = bind(f)
    assert f.stage.GetRootLayer().ExportToString() == before
    assert len(a.binding_sha256) == 64
    assert a.core.chain[2].half_height_m == pytest.approx(.007)
    for p in a.core.pads:
        np.testing.assert_allclose(p.half_extents_m, [.008, .016, .031])
        np.testing.assert_allclose(p.local_frame, pose(.005, -.0295), atol=1e-12)
        assert (p.face_axis, p.face_sign) == (0, -1)
    assert not any(v.calls for v in f.views)
    a.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_original_header_sign_matches_selected_tensor_and_old_keys(reverse):
    f = fixture(); a = bind(f); assert a.begin_step() == 1
    for i in range(2):
        add(a, f, i, reverse=reverse if i == 0 else not reverse)
        tensor(f, i)
    r = evaluate(a, f, frames_step_id=1)
    assert r["bilateral"] and r["stem_only"] and r["adapter_valid"]
    np.testing.assert_allclose(r["forces"], [[.03, 0, 0], [-.03, 0, 0]])
    assert r["counts"] == [1, 1] and r["min_separation"] == 0
    assert r["opposition_cosine"] == pytest.approx(-1)
    assert all(c["passed"] for c in r["tensor_selected_crosscheck"])
    assert not r["native_grasp_certified"] and not r["training_eligible"]
    assert all(v.calls == [.01] for v in f.views)
    a.close()


def test_connected_neighbor_forces_not_compared_to_selected_tensor():
    f = fixture(); a = bind(f); a.begin_step()
    add(a, f, 0, 1, .024); add(a, f, 0, 2, .006); add(a, f, 1, 3, .04)
    tensor(f, 0, .006)
    r = evaluate(a, f)
    assert r["bilateral"]
    np.testing.assert_allclose(r["forces"], [[.03, 0, 0], [-.04, 0, 0]])
    np.testing.assert_allclose(r["tensor_selected_crosscheck"][0]["callback_selected_force_n"], [.006, 0, 0])
    assert r["tensor_selected_crosscheck"][1]["callback_selected_force_n"] == [0, 0, 0]
    a.close()


def test_selected_body_with_leaf_fails_before_tensor_comparison():
    with pytest.raises(ValueError, match="leaf/extra"):
        bind(fixture(leaf_selected=True))


def test_connected_neighbor_leaf_contact_never_qualifies():
    f = fixture(); a = bind(f); a.begin_step()
    for i in range(2):
        add(a, f, i); tensor(f, i)
    add(a, f, 0, 1, other=f.rig.body_paths[1] + "/Leaf_Test/LeafCollider")
    r = evaluate(a, f)
    assert not r["bilateral"] and not r["stem_only"]
    assert r["rejected"][0]["reason"] == "not_connected_detachable_shaft"
    a.close()


@pytest.mark.parametrize("change", ["disable", "swap_endpoints", "remove"])
def test_cached_joint_live_validation_each_evaluate(change):
    f = fixture(); a = bind(f); a.begin_step()
    add(a, f, 0, 1); add(a, f, 1, 2); tensor(f, 1)
    joint = UsdPhysics.Joint.Get(f.stage, "/T/Branch/Joint_002")
    if change == "disable":
        joint.GetJointEnabledAttr().Set(False)
        r = evaluate(a, f)
        assert not r["bilateral"] and r["connected_pair_count"] == 2
    else:
        if change == "remove":
            f.stage.RemovePrim(str(joint.GetPath()))
        else:
            joint.GetBody0Rel().SetTargets([f.rig.body_paths[3]])
        with pytest.raises(ValueError, match="joint"):
            evaluate(a, f)
    a.close()


def test_same_joint_reenabled_is_read_on_next_step():
    f = fixture(); a = bind(f)
    joint = UsdPhysics.Joint.Get(f.stage, "/T/Branch/Joint_002")
    joint.GetJointEnabledAttr().Set(False)
    a.begin_step(); add(a, f, 0, 1); add(a, f, 1, 3)
    assert not evaluate(a, f)["bilateral"]
    joint.GetJointEnabledAttr().Set(True)
    assert a.begin_step() == 2
    add(a, f, 0, 1); add(a, f, 1, 3)
    assert evaluate(a, f)["bilateral"]
    a.close()


@pytest.mark.parametrize("change", ["leaf_added", "radius", "pad_scale", "collision_disabled"])
def test_geometry_or_coverage_changes_invalidate_binding(change):
    f = fixture(); a = bind(f); a.begin_step()
    if change == "leaf_added":
        collision(UsdGeom.Cube.Define(f.stage, a.selected_body + "/LeafNew/Collider").GetPrim())
    elif change == "radius":
        UsdGeom.Capsule.Get(f.stage, a.selected_collider).GetRadiusAttr().Set(.004)
    elif change == "pad_scale":
        matrix(f.stage.GetPrimAtPath(a.pad_paths[0]), np.diag([.1, .1, .1, 1.]))
    else:
        UsdPhysics.CollisionAPI(f.stage.GetPrimAtPath(a.selected_collider)).GetCollisionEnabledAttr().Set(False)
    with pytest.raises(ValueError, match="adapter"):
        evaluate(a, f)
    a.close()


def test_post_fetch_body_motion_uses_native_frames_not_usd_rest():
    f = fixture(); a = bind(f); a.begin_step()
    for i in range(2):
        add(a, f, i); tensor(f, i)
    f.frames[2, 0, 3] += .01
    r = evaluate(a, f)
    assert not r["bilateral"] and r["rejected"]
    a.close()


def test_direct_body_pose_notices_do_not_invalidate_collider_binding():
    f = fixture(); a = bind(f)
    body = f.stage.GetPrimAtPath(a.selected_body)
    op = UsdGeom.Xformable(body).GetOrderedXformOps()[0]
    op.Set(Gf.Matrix4d(pose(.01).T.tolist()))
    assert a.binding_error is None
    a.begin_step()
    assert not evaluate(a, f)["bilateral"]
    a.close()


@pytest.mark.parametrize("delta", ["sign", "magnitude", "missing_callback"])
def test_tensor_disagreement_fails_closed_without_auto_sign_fit(delta):
    f = fixture(); a = bind(f); a.begin_step()
    for i in range(2):
        if delta != "missing_callback" or i == 1:
            add(a, f, i)
        tensor(f, i)
    if delta == "sign":
        f.views[0].force *= -1
    elif delta == "magnitude":
        f.views[0].force *= 2
    r = evaluate(a, f)
    assert not r["adapter_valid"] and not r["bilateral"] and not r["stem_only"]
    assert not r["tensor_selected_crosscheck"][0]["passed"]
    a.close()


@pytest.mark.parametrize("bad", ["nan", "negative", "normal", "layout", "exhausted", "float_count"])
def test_tensor_buffer_faults_latch_and_propagate(bad):
    f = fixture(); a = bind(f); a.begin_step()
    tensor(f, 0)
    if bad == "exhausted":
        f.views[0].count = 8
    elif bad == "float_count":
        f.views[0].count = 1.5
    else:
        f.views[0].corrupt = bad
    with pytest.raises((ValueError, TypeError)):
        evaluate(a, f)
    with pytest.raises(ValueError):
        a.begin_step()
    a.close()


def test_callback_fault_cannot_be_swallowed_into_a_grasp():
    f = fixture(); a = bind(f); a.begin_step()
    with pytest.raises(ValueError):
        a.add_contact(a.pad_paths[0], a.selected_collider, [0, 0, 0], [0, 0, 0], [0, 0, 0], 0.)
    with pytest.raises(ValueError):
        evaluate(a, f)
    a.close()


def test_step_freshness_once_and_close():
    f = fixture(); a = bind(f)
    with pytest.raises(ValueError):
        evaluate(a, f)
    a.close()
    a = bind(f); a.begin_step()
    with pytest.raises(ValueError, match="Stale"):
        evaluate(a, f, frames_step_id=0)
    a.close()
    a = bind(f); a.begin_step()
    r = evaluate(a, f)
    assert not r["bilateral"]
    with pytest.raises(ValueError):
        evaluate(a, f)
    a.close()
    with pytest.raises(ValueError):
        a.begin_step()


@pytest.mark.parametrize("change", ["sensor", "filter", "counts", "extra_finger_collider", "units", "no_offset", "capsule_scale"])
def test_invalid_initial_bindings_rejected(change):
    f = fixture()
    if change == "sensor":
        f.views[0].sensor_paths = [f.fingers[1]]
    elif change == "filter":
        f.views[0].filter_paths = ["/T/Branch/*"]
    elif change == "counts":
        f.views[0].sensor_count = 2
    elif change == "extra_finger_collider":
        collision(UsdGeom.Cube.Define(f.stage, f.fingers[0] + "/Other").GetPrim())
    elif change == "units":
        UsdGeom.SetStageMetersPerUnit(f.stage, .01)
    elif change == "no_offset":
        f.stage.GetPrimAtPath(f.rig.body_paths[2] + "/StemCollider").RemoveProperty("physxCollision:contactOffset")
    else:
        prim = f.stage.GetPrimAtPath(f.rig.body_paths[2] + "/StemCollider")
        matrix(prim, np.diag([1., 2., 1., 1.]))
    with pytest.raises(ValueError):
        bind(f)


def test_mapping_frames_and_empty_step_are_compatible():
    f = fixture(); a = bind(f); a.begin_step()
    r = a.evaluate(.01, dict(zip(f.rig.body_paths, f.frames)),
                   dict(zip(f.fingers, f.fingerframes)))
    assert not r["bilateral"] and r["counts"] == [0, 0] and r["adapter_valid"]
    a.close()


def test_actual_readonly_rby_asset_cube_transform_when_available():
    asset = Path(__file__).resolve().parents[3] / "data/greenhouse_sim/robots/rby1a_v1.2/rby1a_v1.2.usd"
    if not asset.is_file():
        pytest.skip("original RBY asset not available")
    stage = Usd.Stage.Open(str(asset))
    root = str(stage.GetDefaultPrim().GetPath())
    for i in (1, 2):
        body = stage.GetPrimAtPath(root + f"/ee_finger_l{i}")
        pad = stage.GetPrimAtPath(str(body.GetPath()) + "/restored_collisions/contact_proxy")
        assert pad.IsA(UsdGeom.Cube)
        local, scale = mod._local_geometry(pad, body, UsdGeom)
        np.testing.assert_allclose(local[:3, 3], [.005, 0, -.0295], atol=1e-8)
        np.testing.assert_allclose(scale * float(UsdGeom.Cube(pad).GetSizeAttr().Get()) / 2,
                                   [.008, .016, .031], atol=1e-8)


@pytest.mark.parametrize("change", ["units", "new_joint", "ancestor_scale"])
def test_epoch_metadata_and_topology_changes_invalidate(change):
    f = fixture(); a = bind(f); a.begin_step()
    if change == "units":
        UsdGeom.SetStageMetersPerUnit(f.stage, .01)
    elif change == "new_joint":
        UsdPhysics.FixedJoint.Define(f.stage, "/T/Branch/UnexpectedJoint")
    else:
        UsdGeom.Xformable(f.stage.GetPrimAtPath("/T")).AddScaleOp().Set(Gf.Vec3f(2.))
    with pytest.raises(ValueError):
        evaluate(a, f)
    a.close()


@pytest.mark.parametrize("target", [None, "", "seed101_full/SubStem_44"])
def test_rig_target_required_and_matches_authored_bodies(target):
    f = fixture()
    f.rig.source_target = target
    with pytest.raises(ValueError, match="rig.source_target"):
        bind(f)


def test_rig_target_cannot_change_after_binding():
    f = fixture(); a = bind(f); a.begin_step()
    f.rig.source_target = "other/target"
    with pytest.raises(ValueError, match="source_target"):
        evaluate(a, f)
    a.close()


@pytest.mark.parametrize("location", ["shaft", "finger", "plant_ancestor", "robot_ancestor"])
@pytest.mark.parametrize("kind", ["scale", "reflection", "shear"])
def test_initial_world_basis_rejects_hidden_nonrigid_geometry(location, kind):
    f = fixture()
    path = {"shaft": f.rig.body_paths[2], "finger": f.fingers[0],
            "plant_ancestor": "/T", "robot_ancestor": "/R"}[location]
    m = np.eye(4)
    if kind == "scale":
        m[:3, :3] *= 2
    elif kind == "reflection":
        m[0, 0] = -1
    else:
        m[0, 1] = .2
    matrix(f.stage.GetPrimAtPath(path), m)
    with pytest.raises(ValueError, match="Rigid unscaled"):
        bind(f)


@pytest.mark.parametrize("kind", ["scale", "reflection", "shear"])
def test_existing_body_matrix_edit_cannot_hide_scale(kind):
    f = fixture(); a = bind(f); a.begin_step()
    m = np.eye(4)
    if kind == "scale":
        m[:3, :3] *= 2
    elif kind == "reflection":
        m[0, 0] = -1
    else:
        m[0, 1] = .2
    op = UsdGeom.Xformable(f.stage.GetPrimAtPath(a.selected_body)).GetOrderedXformOps()[0]
    op.Set(Gf.Matrix4d(m.T.tolist()))
    with pytest.raises(ValueError, match="basis"):
        evaluate(a, f)
    a.close()


@pytest.mark.parametrize("edit", ["scale_op", "op_order", "ancestor_translate", "ancestor_orient"])
def test_transform_order_and_ancestor_changes_invalidate(edit):
    f = fixture()
    body = UsdGeom.Xformable(f.stage.GetPrimAtPath(f.rig.body_paths[2]))
    if edit == "scale_op":
        op = body.AddScaleOp(); op.Set(Gf.Vec3f(1.))
    elif edit.startswith("ancestor"):
        ancestor = UsdGeom.Xformable(f.stage.GetPrimAtPath("/R"))
        op = ancestor.AddTranslateOp() if edit.endswith("translate") else ancestor.AddOrientOp()
        op.Set(Gf.Vec3d(0.) if edit.endswith("translate") else Gf.Quatf(1.))
    a = bind(f); a.begin_step()
    if edit == "scale_op":
        op.Set(Gf.Vec3f(1.1))
    elif edit == "op_order":
        body.ClearXformOpOrder()
    elif edit.endswith("translate"):
        op.Set(Gf.Vec3d(.1, 0, 0))
    else:
        op.Set(Gf.Quatf(0., Gf.Vec3f(0, 0, 1)))
    with pytest.raises(ValueError):
        evaluate(a, f)
    a.close()


@pytest.mark.parametrize("edit", ["translate", "orient"])
def test_existing_dynamic_translation_and_orientation_remain_allowed(edit):
    f = fixture()
    body = UsdGeom.Xformable(f.stage.GetPrimAtPath(f.rig.body_paths[2]))
    op = body.AddTranslateOp() if edit == "translate" else body.AddOrientOp()
    op.Set(Gf.Vec3d(0.) if edit == "translate" else Gf.Quatf(1.))
    a = bind(f); a.begin_step()
    op.Set(Gf.Vec3d(.1, 0, 0) if edit == "translate" else Gf.Quatf(0., Gf.Vec3f(0, 0, 1)))
    assert a.binding_error is None
    assert not evaluate(a, f)["bilateral"]
    a.close()


def diagnostic_fault(a, *, reverse=False):
    # Native36's finite noncompressive row, not its previous signed-zero row.
    normal = np.array([-.164892315864563, -.22051431238651276, .9613448977470398])
    impulse = np.array([6.927241003040763e-9, 9.263960265570859e-9, -4.0386769484257456e-8])
    c0, c1 = a.pad_paths[0], a.selected_collider
    if reverse:
        c0, c1 = c1, c0
        normal, impulse = -normal, -impulse
    a.add_contact(c0, c1, [.003, 0, 0], normal, impulse, -1.2320466339588165e-5)
    return c0, c1, normal, impulse


def signed_tensor(view, scalars, normals, *, start=0):
    size = view.max_contact_data_count
    f, p, n, d = np.zeros((size, 1)), np.zeros((size, 3)), np.zeros((size, 3)), np.zeros((size, 1))
    count = len(scalars)
    f[start:start+count, 0] = scalars
    n[start:start+count] = normals
    p[start:start+count] = [.003, 0, 0]
    d[start:start+count, 0] = -1.2320466339588165e-5
    buffers = (f, p, n, d, np.array([[count]]), np.array([[start]]))
    def get(dt):
        view.calls.append(dt)
        return buffers
    view.get_contact_data = get
    return buffers


def diagnostic_report(a, dt=1/240, step_id=1):
    with pytest.raises(ValueError) as caught:
        a.raise_pending_fault(dt, step_id=step_id)
    result = json.loads(str(caught.value))
    assert result == a.last_fault_report
    assert not result['adapter_valid'] and not result['bilateral'] and not result['stem_only']
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize('reverse', [False, True])
def test_diagnostic_drains_first_bad_step_preserving_every_original_sign(reverse):
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    add(a, f, 0)
    # Unrelated normals must also be captured, not silently selected away.
    a.add_contact('/Other/A', '/Other/B', [0, 1, 2], [0, 1, 0], [0, .2, 0], .0001)
    c0, c1, n, j = diagnostic_fault(a, reverse=reverse)
    first = dict(a.last_failed_contact)
    assert a.error is not None and a.core.error is not None
    add(a, f, 1)
    diagnostic_fault(a, reverse=not reverse)
    assert len(a.core.rows) == 1  # Nothing after the failed core row is evidence.
    assert a.last_failed_contact == first
    buffers = signed_tensor(f.views[0], [-1e-5, .03, 1e-30], [[1, 0, 0]]*3, start=2)
    before = [v.copy() for v in buffers]
    signed_tensor(f.views[1], [.04], [[-1, 0, 0]])
    assert not f.views[0].calls and not f.views[1].calls  # No callback-time tensor reads.
    report = diagnostic_report(a)
    rows = report['callback']['rows']
    assert report['callback']['all_delivered_rows_captured'] and len(rows) == 5
    assert [r['row_index'] for r in rows] == list(range(5))
    assert rows[1]['collider0'] == '/Other/A'
    assert rows[2]['collider0'] == c0 and rows[2]['collider1'] == c1
    assert rows[2]['normal'] == n.tolist() and rows[2]['impulse'] == j.tolist()
    assert report['last_failed_contact'] == first and first['row_index'] == 2
    tensor_result = report['tensor_selected'][0]
    assert tensor_result['complete'] and tensor_result['start'] == 2
    assert [r['raw_force_n'] for r in tensor_result['rows']] == [-1e-5, .03, 1e-30]
    assert tensor_result['rows'][0]['raw_impulse_ns'] == -1e-5/240
    assert tensor_result['rows'][2]['raw_impulse_ns'] > 0  # Never drop small positives.
    np.testing.assert_allclose(tensor_result['signed_force_n'], [.03-1e-5, 0, 0])
    for old, new in zip(before, buffers):
        np.testing.assert_array_equal(old, new)
    assert f.views[0].calls == f.views[1].calls == [1/240]
    assert diagnostic_report(a) == report
    assert f.views[0].calls == [1/240]  # Frozen snapshot, not a later reread.
    with pytest.raises(ValueError):
        a.begin_step()
    assert a._step == 1
    with pytest.raises(ValueError):
        evaluate(a, f)
    with pytest.raises(ValueError):
        add(a, f, 0)
    a.close()


def test_diagnostic_evaluate_defensively_halts_before_grasp_or_frames():
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    diagnostic_fault(a)
    with pytest.raises(ValueError) as caught:
        a.evaluate(1/240, None, None, frames_step_id=1)
    assert json.loads(str(caught.value))['diagnostic_only']
    assert not a.core.evaluated and f.views[0].calls == [1/240]
    a.close()


@pytest.mark.parametrize('bad_dt,step', [(0, 1), (-1, 1), (float('nan'), 1),
    (float('inf'), 1), (True, 1), (.01, 0), (.01, 2), (.01, True)])
def test_diagnostic_bad_dt_or_step_blocks_reads_and_next_step(bad_dt, step):
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    diagnostic_fault(a)
    report = diagnostic_report(a, bad_dt, step)
    assert report['validation_error'] and report['tensor_selected'] == []
    assert not f.views[0].calls and not f.views[1].calls
    with pytest.raises(ValueError): a.begin_step()
    a.close()


@pytest.mark.parametrize('bad', ['nan', 'span', 'layout', 'identity', 'read_error', 'full_buffer'])
def test_diagnostic_tensor_failure_is_explicit_and_other_finger_still_captured(bad):
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    diagnostic_fault(a)
    data = signed_tensor(f.views[0], [-1e-5], [[1, 0, 0]])
    if bad == 'nan': data[0][0, 0] = np.nan
    elif bad == 'span': data[4][0, 0] = 9
    elif bad == 'layout': data[0].shape = (8,)
    elif bad == 'identity': f.views[0].filter_paths = ['/Wrong']
    elif bad == 'full_buffer': data[4][0, 0] = 8
    else:
        def fail(dt): raise RuntimeError('read failed')
        f.views[0].get_contact_data = fail
    report = diagnostic_report(a)
    first, second = report['tensor_selected']
    assert not first['complete'] and first['signed_force_n'] is None
    assert second['complete'] and f.views[1].calls == [1/240]
    if bad == 'nan':
        assert isinstance(first['rows'][0]['raw_force_n'], str)
        assert not first['rows'][0]['finite']
    a.close()


def test_diagnostic_bounded_callback_and_combined_tensor_buffers_fail_closed():
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    diagnostic_fault(a)
    for _ in range(300):
        add(a, f, 1)
    for v in f.views:
        v.max_contact_data_count = 400
        signed_tensor(v, [-1e-5]*200, [[1, 0, 0]]*200)
    report = diagnostic_report(a)
    assert report['callback']['rows_seen'] == 301
    assert report['callback']['rows_captured'] == 256
    assert report['callback']['rows_dropped'] == 45
    assert not report['callback']['all_delivered_rows_captured']
    t0, t1 = report['tensor_selected']
    assert len(t0['rows']) == 200 and len(t1['rows']) == 56
    assert t1['rows_dropped'] == 144 and not t1['complete'] and t1['signed_force_n'] is None
    with pytest.raises(ValueError): a.begin_step()
    a.close()


def test_diagnostic_overflow_before_noncompressive_fault_raises_immediately():
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    for _ in range(256):
        a.add_contact('/A', '/B', [0, 0, 0], [1, 0, 0], [0, 0, 0], 0.)
    with pytest.raises(ValueError, match='overflow'):
        a.add_contact('/A', '/B', [0, 0, 0], [1, 0, 0], [0, 0, 0], 0.)
    assert len(a._diagnostic_rows) == 256
    with pytest.raises(ValueError): a.begin_step()
    a.close()


@pytest.mark.parametrize('change', ['closed', 'binding', 'source'])
def test_diagnostic_invalid_epoch_does_not_read_tensors(change):
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    diagnostic_fault(a)
    if change == 'closed': a.close()
    elif change == 'binding': a.binding_error = 'edited'
    else: f.rig.source_target = 'edited'
    report = diagnostic_report(a)
    assert report['validation_error'] and report['tensor_selected'] == []
    assert not f.views[0].calls
    a.close()


@pytest.mark.parametrize('diagnostic', [False, True])
def test_noncompressive_default_and_nonnegative_core_faults_are_not_suppressed(diagnostic):
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=diagnostic); a.begin_step()
    if diagnostic:
        with pytest.raises(ValueError):
            a.add_contact(a.pad_paths[0], a.selected_collider, [0, 0, 0],
                          [1, 0, 0], [1e-4, 1e-4, 0], 0.)
    else:
        with pytest.raises(ValueError): diagnostic_fault(a)
    assert not a._pending_noncompressive
    with pytest.raises(ValueError): a.begin_step()
    a.close()


def test_diagnostic_normal_steps_reset_buffer_and_preserve_grasp_and_strict_tensor_gate():
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    for i in range(2): add(a, f, i); tensor(f, i)
    assert a.raise_pending_fault(.01, step_id=1) is None
    assert evaluate(a, f)['bilateral']
    a.begin_step()
    assert a._diagnostic_rows == [] and a._diagnostic_rows_seen == 0
    signed_tensor(f.views[0], [-1e-20], [[1, 0, 0]])
    with pytest.raises(ValueError, match='Invalid tensor normal contact rows'):
        evaluate(a, f)
    with pytest.raises(ValueError): a.begin_step()
    a.close()


@pytest.mark.parametrize('error_attr', ['error', 'binding_error'])
def test_empty_error_is_still_latched(error_attr):
    f = fixture(); a = bind(f); setattr(a, error_attr, '')
    with pytest.raises(ValueError): a.begin_step()
    assert a._step == 0
    a.close()


@pytest.mark.parametrize('flag', [None, 1, 'yes'])
def test_diagnostic_requires_explicit_boolean_opt_in(flag):
    with pytest.raises(ValueError, match='boolean'):
        bind(fixture(), diagnostic_noncompressive_report=flag)


def test_diagnostic_contact_events_drains_same_pair_and_later_pair_before_halt():
    from sim_physics.contact_events import ContactEvents
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True)
    monitor = ContactEvents(robot_root='/R', target_root='/T', fingers=f.fingers, floor_root='')
    monitor.normal_contact_observer = a
    monitor.begin_step()
    # The bad middle row must not abort the remaining row or later header.
    monitor.consume(a.pad_paths[0], a.selected_collider,
        [[.0003, 0, 0], [-4.2e-8, 0, 0], [.0001, 0, 0]],
        [[.003, 0, 0]]*3, [[1, 0, 0]]*3, [0., 0., 0.])
    monitor.consume(a.pad_paths[1], a.selected_collider,
        [[-.0003, 0, 0]], [[-.003, 0, 0]], [[-1, 0, 0]], [0.])
    assert monitor.error is None  # Deferred ONLY to the mandatory post-fetch guard.
    report = diagnostic_report(a)
    assert report['callback']['rows_captured'] == 4
    assert report['last_failed_contact']['row_index'] == 1
    assert report['callback']['rows'][-1]['collider0'] == a.pad_paths[1]
    with pytest.raises(ValueError): monitor.begin_step()
    assert a._step == 1
    monitor.close(); a.close()


def test_diagnostic_later_malformed_raw_row_is_json_safe_without_entering_core():
    f = fixture(); a = bind(f, diagnostic_noncompressive_report=True); a.begin_step()
    diagnostic_fault(a)
    a.add_contact('/A', '/B', [np.nan, np.inf, -np.inf], [1, 0, 0],
                  [0, 0, 0], np.nan)
    report = diagnostic_report(a)
    row = report['callback']['rows'][1]
    assert all(isinstance(v, str) for v in row['point'])
    assert isinstance(row['separation'], str)
    assert not a.core.rows and not a.core.evaluated
    a.close()


def observed37_bind(f, **kwargs):
    return bind(f, allow_signed_native_normals=True,
                sensor_contract=mod.NATIVE37_SENSOR_CONTRACT, **kwargs)


def observed37_rows(a, f, finger, forces, *, dt=.01):
    """Synthetic sensor matches the OBSERVED contract; not an unsigned API claim."""
    normal = np.array([1. if finger == 0 else -1., 0., 0.])
    points = [[normal[0]*.003, 0., i*.001] for i in range(len(forces))]
    impulses = [normal * (force * dt) for force in forces]
    for point, impulse in zip(points, impulses):
        a.add_contact(a.pad_paths[finger], a.selected_collider, point, normal, impulse, 0.)
    # Magnitudes are ONLY synthetic tensor-report values, never callback evidence.
    buffers = signed_tensor(f.views[finger], [np.linalg.norm(j)/dt for j in impulses],
                            [normal]*len(forces))
    if points:
        buffers[1][:len(points)] = points
        buffers[3][:len(points), 0] = 0.
    return buffers


def test_observed37_magnitude_matches_leave_negative_callback_signed_in_physics():
    f = fixture(); a = observed37_bind(f); a.begin_step()
    observed37_rows(a, f, 0, [-1e-5, .03, 0.])
    observed37_rows(a, f, 1, [.04])
    r = evaluate(a, f)
    assert r['adapter_valid'] and r['bilateral'] and r['compressive_support_gate_applied']
    np.testing.assert_allclose(r['forces'][0], [.03-1e-5, 0, 0], atol=1e-15)
    np.testing.assert_allclose(r['compressive_support_n'], [.03-1e-5, .04], atol=1e-15)
    assert r['negative_normal_impulses'][0]['grasp_scalar_ns'] < 0
    assert r['tensor_integrity_only'] and r['physical_force_source'] == 'signed_callback_only'
    matches = r['tensor_selected_crosscheck'][0]['matches']
    assert len(matches) == 3 and all(m['passed'] for m in matches)
    assert matches[0]['callback_signed_projection_ns'] < 0 < matches[0]['tensor_magnitude_ns']
    assert matches[2]['callback_signed_projection_ns'] == matches[2]['tensor_magnitude_ns'] == 0
    assert r['normal_callback_rows_seen'] == r['normal_callback_rows_captured'] == 4
    contract = r['sensor_contract']
    assert 'observed' in contract['basis'] and 'does not specify unsigned' in contract['basis']
    assert contract['basis_report_sha256'] == mod._NATIVE37_REPORT_SHA256
    assert not contract['native_backend_independently_verified']
    a.close()


@pytest.mark.parametrize('forces', [[-.03], [-.04, .04], [-.03, .04], [-.03, .0499]])
def test_observed37_sensor_integrity_cannot_rescue_tension_cancellation_or_weak_support(forces):
    f = fixture(); a = observed37_bind(f); a.begin_step()
    observed37_rows(a, f, 0, forces)
    observed37_rows(a, f, 1, [.04])
    r = evaluate(a, f)
    assert r['adapter_valid'] and not r['bilateral'] and not r['compressive_support_passed']
    assert r['compressive_support_n'][0] < .02
    a.close()


def test_observed37_uses_geometry_not_buffer_order_or_magnitude_for_matching():
    f = fixture(); a = observed37_bind(f); a.begin_step()
    data = observed37_rows(a, f, 0, [-.01, .04])
    observed37_rows(a, f, 1, [.04])
    for buffer in data[:4]: buffer[:2] = buffer[[1, 0]]
    r = evaluate(a, f)
    assert r['bilateral'] and r['adapter_valid']
    assert [m['tensor_buffer_index'] for m in r['tensor_selected_crosscheck'][0]['matches']] == [1, 0]
    a.close()


@pytest.mark.parametrize('changed', ['point', 'normal', 'separation', 'magnitude', 'missing', 'extra', 'ambiguous'])
def test_observed37_missing_extra_ambiguous_or_mismatched_rows_fail_closed(changed):
    f = fixture(); a = observed37_bind(f); a.begin_step()
    data = observed37_rows(a, f, 0, [-.01, .04])
    observed37_rows(a, f, 1, [.04])
    if changed == 'point': data[1][0, 0] += 2*mod._MATCH_POSITION_M
    elif changed == 'normal': data[2][0, 1] += 2*mod._MATCH_NORMAL
    elif changed == 'separation': data[3][0, 0] += 2*mod._MATCH_SEPARATION_M
    elif changed == 'magnitude': data[0][0, 0] *= 1 + 2*mod._MATCH_IMPULSE_RTOL
    elif changed == 'missing': data[4][0, 0] = 1
    elif changed == 'extra':
        data[4][0, 0] = 3
        data[2][2] = [1, 0, 0]
    else:
        # Duplicate geometry must not be disambiguated by distinct force values.
        a._diagnostic_rows[1]['point'] = list(a._diagnostic_rows[0]['point'])
        data[1][1] = data[1][0]
    r = evaluate(a, f)
    assert not r['adapter_valid'] and not r['bilateral'] and not r['stem_only']
    assert not r['tensor_selected_crosscheck'][0]['passed']
    assert r['tensor_selected_crosscheck'][0]['reason']
    a.close()


@pytest.mark.parametrize('field,tolerance', [('point', mod._MATCH_POSITION_M),
    ('separation', mod._MATCH_SEPARATION_M), ('normal', mod._MATCH_NORMAL)])
@pytest.mark.parametrize('multiplier,expected', [(.5, True), (2., False)])
def test_observed37_geometry_tolerances_predeclared_not_fitted(field, tolerance, multiplier, expected):
    f = fixture(); a = observed37_bind(f); a.begin_step()
    data = observed37_rows(a, f, 0, [.03])
    observed37_rows(a, f, 1, [.04])
    if field == 'point': data[1][0, 0] += multiplier*tolerance
    elif field == 'separation': data[3][0, 0] += multiplier*tolerance
    else: data[2][0, 1] += multiplier*tolerance
    r = evaluate(a, f)
    assert r['adapter_valid'] == expected
    a.close()


@pytest.mark.parametrize('multiplier,expected', [(.5, True), (2., False)])
def test_observed37_impulse_matching_uses_fixed_float32_relative_budget(multiplier, expected):
    f = fixture(); a = observed37_bind(f); a.begin_step()
    # Normal (not subnormal) squared norm: the relative budget alone applies.
    data = observed37_rows(a, f, 0, [1e-10, .03])
    observed37_rows(a, f, 1, [.04])
    data[0][0, 0] *= 1 + multiplier*mod._MATCH_IMPULSE_RTOL
    assert evaluate(a, f)['adapter_valid'] == expected
    a.close()


def test_observed37_never_flips_a_reversed_header_to_fit_sensor_sign():
    f = fixture(); a = observed37_bind(f); a.begin_step()
    add(a, f, 0, reverse=True)
    signed_tensor(f.views[0], [.03], [[1, 0, 0]])
    observed37_rows(a, f, 1, [.04])
    r = evaluate(a, f)
    assert not r['adapter_valid'] and not r['bilateral']
    assert r['tensor_selected_crosscheck'][0]['reason'] == 'selected_header_order_not_qualified_by_native37'
    a.close()


def test_observed37_rejects_negative_tensor_instead_of_taking_absolute_value():
    f = fixture(); a = observed37_bind(f); a.begin_step()
    data = observed37_rows(a, f, 0, [.03])
    data[0][0, 0] *= -1
    with pytest.raises(ValueError, match='Invalid tensor normal contact rows'): evaluate(a, f)
    with pytest.raises(ValueError): a.begin_step()
    a.close()


def test_observed37_geometry_rejection_remains_physical_rejection_despite_sensor_match():
    f = fixture(); a = observed37_bind(f); a.begin_step()
    data = observed37_rows(a, f, 0, [.03])
    observed37_rows(a, f, 1, [.04])
    # Same off-shaft point in both channels is not proof of a valid grasp.
    # Rebuild a step through the public callback, not by editing the core geometry.
    evaluate(a, f); a.begin_step()
    a.add_contact(a.pad_paths[0], a.selected_collider, [.1, 0, 0], [1, 0, 0], [.0003, 0, 0], 0.)
    data = signed_tensor(f.views[0], [.03], [[1, 0, 0]])
    data[1][0] = [.1, 0, 0]; data[3][0, 0] = 0.
    observed37_rows(a, f, 1, [.04])
    r = evaluate(a, f)
    assert r['adapter_valid'] and r['rejected'] and not r['bilateral'] and not r['stem_only']
    a.close()


def test_observed37_matches_selected_only_not_connected_neighbour_tensor_aggregate():
    f = fixture(); a = observed37_bind(f); a.begin_step()
    observed37_rows(a, f, 0, [.03])
    observed37_rows(a, f, 1, [.04])
    add(a, f, 0, segment=3, force=-.005)
    r = evaluate(a, f)
    assert r['bilateral'] and r['adapter_valid']
    assert r['tensor_selected_crosscheck'][0]['callback_contact_count'] == 1
    np.testing.assert_allclose(r['forces'][0], [.025, 0, 0])
    np.testing.assert_allclose(r['tensor_selected_crosscheck'][0]['callback_selected_force_n'], [.03, 0, 0])
    a.close()


@pytest.mark.parametrize('kwargs', [dict(allow_signed_native_normals=True),
    dict(sensor_contract=mod.NATIVE37_SENSOR_CONTRACT), dict(sensor_contract='auto'),
    dict(allow_signed_native_normals=1),
    dict(allow_signed_native_normals=True, sensor_contract=mod.NATIVE37_SENSOR_CONTRACT,
         diagnostic_noncompressive_report=True)])
def test_observed37_explicit_paired_contract_required_and_diagnostic_mode_kept_separate(kwargs):
    with pytest.raises(ValueError): bind(fixture(), **kwargs)


def test_observed37_contract_is_constructor_only_hash_bound_and_default_stays_strict():
    f = fixture(); strict = bind(f); signed = observed37_bind(f)
    assert not strict.core.allow_signed_native_normals and signed.core.allow_signed_native_normals
    assert strict.binding_sha256 != signed.binding_sha256
    for key, value in [('allow_signed_native_normals', False), ('sensor_contract', 'auto')]:
        with pytest.raises(AttributeError): setattr(signed, key, value)
    signed.core._allow_signed_native_normals = False
    with pytest.raises(ValueError, match='mode changed'): signed.begin_step()
    strict.close(); signed.close()


def test_observed37_raw_all_normal_row_accounting_stays_bounded_at_256():
    f = fixture(); a = observed37_bind(f); a.begin_step()
    for _ in range(256): a.add_contact('/A', '/B', [0, 0, 0], [1, 0, 0], [0, 0, 0], 0.)
    with pytest.raises(ValueError, match='overflow'):
        a.add_contact('/A', '/B', [0, 0, 0], [1, 0, 0], [0, 0, 0], 0.)
    assert len(a._diagnostic_rows) == 256
    with pytest.raises(ValueError): a.begin_step()
    a.close()


@pytest.mark.parametrize('sign',[-1.,1.])
@pytest.mark.parametrize('impulse,expected',[(8.774236304117122e-38,True),
    (.5*mod._MATCH_ZERO_COMPONENT_NS,True),(2.*mod._MATCH_ZERO_COMPONENT_NS,False)])
def test_zero_tensor_norm_underflow_keeps_raw_signed_physical_evidence(sign,impulse,expected):
    f=fixture();a=observed37_bind(f);a.begin_step()
    data=observed37_rows(a,f,0,[sign*impulse/.01,.03])
    observed37_rows(a,f,1,[.04]);data[0][0,0]=0.
    r=evaluate(a,f)
    match=r['tensor_selected_crosscheck'][0]['matches'][0]
    assert r['adapter_valid']==expected and match['tensor_zero_underflow']==expected
    assert np.sign(match['callback_signed_projection_ns'])==sign
    if sign<0:
        assert r['negative_normal_impulses'][0]['grasp_scalar_ns']==pytest.approx(-impulse,rel=1e-12,abs=0)
    a.close()


def test_zero_underflow_rule_does_not_allow_tiny_rows_to_manufacture_grasp():
    f=fixture();a=observed37_bind(f);a.begin_step()
    for finger in range(2):
        data=observed37_rows(a,f,finger,[-.5*mod._MATCH_ZERO_COMPONENT_NS/.01])
        data[0][0,0]=0.
    r=evaluate(a,f)
    assert r['adapter_valid'] and not r['bilateral']
    assert all(s<0 for s in r['compressive_support_n'])
    a.close()


@pytest.mark.parametrize('sign',[-1.,1.])
def test_native39_nonzero_squared_subnormal_norm_reconciles_without_losing_sign(sign):
    f=fixture();a=observed37_bind(f);a.begin_step()
    impulse=9.444774048741853e-21
    data=observed37_rows(a,f,0,[sign*impulse/.01,.03])
    observed37_rows(a,f,1,[.04]);data[0][0,0]=9.44485249335351e-21/.01
    r=evaluate(a,f);m=r['tensor_selected_crosscheck'][0]['matches'][0]
    assert r['adapter_valid'] and m['subnormal_squared_norm'] and not m['tensor_zero_underflow']
    assert m['magnitude_error_ns']>m['magnitude_tolerance_ns']
    assert np.sign(m['callback_signed_projection_ns'])==sign
    a.close()


def test_squared_subnormal_budget_does_not_allow_general_small_nonzero_mismatch():
    f=fixture();a=observed37_bind(f);a.begin_step()
    data=observed37_rows(a,f,0,[1e-20/.01,.03])
    observed37_rows(a,f,1,[.04]);data[0][0,0]*=1.001
    r=evaluate(a,f);m=r['tensor_selected_crosscheck'][0]['matches'][0]
    assert not r['adapter_valid'] and not m['subnormal_squared_norm']
    assert m['squared_impulse_error_ns2']>mod._MATCH_SQUARED_ATOL_NS2
    a.close()


@pytest.mark.parametrize('factor,expected',[(.5,True),(2.,False)])
def test_subnormal_squared_norm_has_fixed_absolute_squared_error_boundary(factor,expected):
    f=fixture();a=observed37_bind(f);a.begin_step();impulse=1e-21
    data=observed37_rows(a,f,0,[impulse/.01,.03])
    observed37_rows(a,f,1,[.04])
    data[0][0,0]=np.sqrt(impulse**2+factor*mod._MATCH_SQUARED_ATOL_NS2)/.01
    r=evaluate(a,f);m=r['tensor_selected_crosscheck'][0]['matches'][0]
    assert r['adapter_valid']==expected and m['subnormal_squared_norm']==expected
    a.close()


def test_subnormal_squared_rule_never_substitutes_nonzero_tensor_for_zero_callback():
    f=fixture();a=observed37_bind(f);a.begin_step()
    data=observed37_rows(a,f,0,[0.,.03]);observed37_rows(a,f,1,[.04])
    data[0][0,0]=1e-38/.01
    r=evaluate(a,f);m=r['tensor_selected_crosscheck'][0]['matches'][0]
    assert not r['adapter_valid'] and not m['subnormal_squared_norm'] and not m['tensor_zero_underflow']
    a.close()


def test_oversized_finite_tensor_cannot_publish_nonfinite_squared_diagnostics():
    f=fixture();a=observed37_bind(f);a.begin_step()
    data=observed37_rows(a,f,0,[.03]);observed37_rows(a,f,1,[.04])
    data[0][0,0]=1e200
    with pytest.raises(ValueError,match='Nonfinite selected impulse comparison'):
        evaluate(a,f)
    with pytest.raises(ValueError):a.begin_step()
    a.close()
