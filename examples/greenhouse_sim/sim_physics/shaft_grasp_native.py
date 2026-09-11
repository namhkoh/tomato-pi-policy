"""Read-only binding adapter for ShaftGraspEvidence; no policy or motion control.

Construction snapshots exact authored capsules, Cube pads and joint identities.
A passive USD notice invalidates the binding on geometry/coverage edits. Only
cached joint enabled/body relationships are read per evaluate; there is no stage
traversal, native subscription, SimulationApp, stepping or source authoring.

Main owns the full-report stream and must call begin_step BEFORE each physics
step, forward original-order NORMAL ContactData rows, then evaluate ONCE after
fetch with current body/finger matrices. Do not forward friction anchors, reorder
headers, reuse frames, silently drop callback errors, or reuse after native reset.
USD topology checks cannot independently detect an unreflected native break.

ContactData: normal is collider1 -> collider0; impulse is normal * scalar and
acts on collider0. Force ON finger = +/- impulse/dt, according to header order.
Strict-default tensor crosscheck uses sum(force_scalar * normal) on each finger
sensor, selected StemCollider ONLY. This is NOT a universal signed tensor
contract: native37 matched a negative callback to a positive scalar magnitude.
The explicitly selected native37 contract instead matches point/normal/distance
and impulse magnitude for sensor integrity ONLY. Physical forces stay signed
callback forces; neither absolute values nor tensor scalars enter grasp loads.
Evidence: PhysX PxContactPairPoint / extractContacts; installed _physx.pyi:442;
omni.physics.tensors/api.py:get_contact_data. Sign agreement is checked, never fit.
https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/include/PxSimulationEventCallback.h

The adapter's name denotes its input contract, NOT native grasp certification.
No dwell/slip update, force gate change, contact filtering, or training approval.
"""
import hashlib
import json

import numpy as np

from .shaft_grasp import FingerPad, ShaftCapsule, ShaftGraspEvidence, _index, _pose


NATIVE37_SENSOR_CONTRACT = "native37_magnitude_sensor0_v1"
STRICT_SENSOR_CONTRACT = "strict_signed_vector_v1"
_MATCH_POSITION_M = 1e-7
_MATCH_SEPARATION_M = 1e-7
_MATCH_NORMAL = 1e-6
_MATCH_IMPULSE_RTOL = 8 * float(np.finfo(np.float32).eps)
_MATCH_IMPULSE_ATOL_NS = 8 * float(np.nextafter(np.float32(0), np.float32(1)))
# A three-component float32 norm may flush squared subnormal components to
# zero. This is a numerical envelope, not proof of the opaque backend algorithm.
# Only a ZERO tensor value gets this exception, never a general force tolerance.
_MATCH_ZERO_COMPONENT_NS = float(np.sqrt(float(np.finfo(np.float32).tiny)))
_MATCH_SQUARED_ATOL_NS2 = 8 * float(np.nextafter(np.float32(0), np.float32(1)))
_NATIVE37_REPORT_SHA256 = "aa9d497d1077a54e1ab87d0d71157d9bf35559bc050c6cd6717ab8f7b971e4fa"


def _sensor_contract_record(name):
    if name == STRICT_SENSOR_CONTRACT:
        return dict(name=name, scope="legacy selected signed-vector crosscheck")
    return dict(name=name, scope="selected-row sensor integrity only; not physical force evidence",
        basis="native37 observed behavior; API documentation does not specify unsigned scalars",
        basis_report="data/sim_physics/bimanual_hold_control_20260911_37/report.json",
        basis_report_sha256=_NATIVE37_REPORT_SHA256,
        backend_basis="omni.physics.tensors 110.1.13; caller must bind this measured backend contract",
        native_backend_independently_verified=False,
        selected_header_rule="finger pad is collider0; tensor normal equals original callback normal",
        position_atol_m=_MATCH_POSITION_M, separation_atol_m=_MATCH_SEPARATION_M,
        normal_component_atol=_MATCH_NORMAL, geometry_rtol=0.,
        impulse_rtol=_MATCH_IMPULSE_RTOL, impulse_atol_ns=_MATCH_IMPULSE_ATOL_NS,
        tensor_zero_component_bound_ns=_MATCH_ZERO_COMPONENT_NS,
        subnormal_squared_impulse_atol_ns2=_MATCH_SQUARED_ATOL_NS2,
        tensor_zero_rule="each callback component squared below float32 minimum normal; physical sign retained")


def _diagnostic_text(value):
    """Bound error-path text; malformed native rows must remain JSON serializable."""
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:
        text = "<unrepresentable " + type(value).__name__ + ">"
    return text if len(text) <= 1024 else text[:1024] + "<truncated>"


def _diagnostic_number(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except Exception:
        pass
    return _diagnostic_text(value)


def _diagnostic_vector(value):
    try:
        if len(value) == 3:
            return [_diagnostic_number(value[i]) for i in range(3)]
    except Exception:
        pass
    return _diagnostic_text(value)


def _paths(values):
    values = list(values)
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = list(values[0])
    return [str(p) for p in values]


def _owner(prim, physics):
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(physics.RigidBodyAPI):
            return str(prim.GetPath())
        prim = prim.GetParent()
    raise ValueError("Collider has no rigid body")


def _active(prim, physics):
    return bool(prim and prim.IsActive() and prim.HasAPI(physics.CollisionAPI)
                and physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())


def _offset(prim):
    # Full robot builder authors these explicitly. Unknown defaults are not
    # silently substituted by a guessed offset.
    contact = prim.GetAttribute("physxCollision:contactOffset").Get()
    rest = prim.GetAttribute("physxCollision:restOffset").Get()
    if (contact is None or rest is None or not np.isfinite([contact, rest]).all()
            or not 0 <= contact <= .001 or rest != 0):
        raise ValueError("Explicit <=1 mm contactOffset and zero restOffset required")
    return float(contact)


def _local_geometry(prim, body, geom):
    matrix = np.asarray(geom.XformCache().ComputeRelativeTransform(prim, body)[0], float).T
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid authored collider transform")
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(scale <= 0):
        raise ValueError("Singular collider scale")
    rigid = matrix.copy()
    rigid[:3, :3] /= scale
    return _pose(rigid), scale


def _coverage(stage, body_path, usd, physics):
    prim = stage.GetPrimAtPath(body_path)
    if (not prim or not prim.HasAPI(physics.RigidBodyAPI)
            or not physics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()):
        raise ValueError("Missing/enabled rigid body required: " + body_path)
    return sorted(str(p.GetPath()) for p in usd.PrimRange(prim, usd.TraverseInstanceProxies())
                  if _active(p, physics) and _owner(p, physics) == body_path)


def _body_world_basis(body, geom):
    # Relative collider transforms alone cancel body/ancestor scale. Validate
    # the complete authored world basis too; no scale may be hidden upstream.
    if not body or not body.IsActive():
        raise ValueError("Missing active body for world-basis validation")
    return _pose(np.asarray(geom.XformCache().GetLocalToWorldTransform(body), float).T)


class ShaftGraspNative:
    """Bind once after scene construction; recreate after reset or geometry edits.

    Example integration (main owns all callbacks and the native step):
        observer = ShaftGraspNative(stage, rig, selected_index=fixture.body_index,
            finger_paths=fixture.paths[1:], contact_views=fixture.contact_views)
        observer.begin_step()
        # step/fetch; forward original rows through observer.add_contact(...)
        result = observer.evaluate(dt, runtime.frames, exact_ordered_finger_frames)

    frames: (len(rig.body_paths),4,4) or mapping of those exact body paths.
    nativefingerframes: (2,4,4), finger_paths order, or exact-path mapping.
    An optional frames_step_id may strengthen the producer's freshness check;
    omitting it is the caller's assertion these are post-fetch frames of this step.

    Default pad paths use the actual RBY contact_proxy Cube, not visual meshes.
    Both actual RBY finger inner faces are local -X; finger2's BODY is rotated.
    Custom fixtures may supply exact pad_paths and (axis, sign) pad_faces.

    diagnostic_noncompressive_report is an opt-in, HOLD-ONLY diagnostic. Main
    must prohibit cut/release execution and call raise_pending_fault(dt,
    step_id=...) immediately post-fetch, before other error checks or control.
    It drains only the CURRENT callback step on a noncompressive core fault,
    then reads signed selected tensors and always raises. No next step, grasp
    evidence, motion, reset, file writes or native synchronization is provided.

    Signed compliant evidence requires BOTH allow_signed_native_normals=True
    and sensor_contract=NATIVE37_SENSOR_CONTRACT. This constructor selection is
    never inferred from a run's signs/forces. Native37 measured selected headers
    with the pad as collider0 only; reversed selected headers fail closed.
    Signed mode and strict fault-capture mode are mutually exclusive. Main owns
    any CLI opt-in/restriction and calibration claims; defaults stay strict.
    """
    def __init__(self, stage, rig, *, selected_index, finger_paths, contact_views,
                 pad_paths=None, pad_faces=((0, -1), (0, -1)), max_rows=256,
                 diagnostic_noncompressive_report=False, allow_signed_native_normals=False,
                 sensor_contract=STRICT_SENSOR_CONTRACT):
        from pxr import Tf, Usd, UsdGeom, UsdPhysics

        if type(diagnostic_noncompressive_report) is not bool:
            raise ValueError("Explicit boolean diagnostic_noncompressive_report required")
        if type(allow_signed_native_normals) is not bool:
            raise ValueError("Explicit boolean allow_signed_native_normals required")
        expected_contract = NATIVE37_SENSOR_CONTRACT if allow_signed_native_normals else STRICT_SENSOR_CONTRACT
        if not isinstance(sensor_contract, str) or sensor_contract != expected_contract:
            raise ValueError("Explicit matching signed-mode/sensor contract selection required")
        if diagnostic_noncompressive_report and allow_signed_native_normals:
            raise ValueError("Strict fault capture and signed-native evidence are separate modes")
        self._allow_signed_native_normals = allow_signed_native_normals
        self._sensor_contract = sensor_contract
        self.diagnostic_noncompressive_report = diagnostic_noncompressive_report
        self._pending_noncompressive = False
        self._diagnostic_rows = []
        self._diagnostic_rows_seen = 0
        self._diagnostic_capture_closed = False
        self._diagnostic_error_json = None
        self.last_fault_report = None
        self.stage = stage
        self.rig = rig
        self.closed = False
        self.binding_error = None
        self.error = None
        self.last_failed_contact = None
        self._notice = None
        self._usd, self._geom, self._physics = Usd, UsdGeom, UsdPhysics
        self.body_paths = tuple(map(str, rig.body_paths))
        self.finger_paths = tuple(map(str, finger_paths))
        self.views = tuple(contact_views)
        self.selected_index = _index(selected_index)
        self.cut_index = _index(rig.cut_index)
        if (not self.body_paths or len(set(self.body_paths)) != len(self.body_paths)
                or not self.cut_index <= self.selected_index < len(self.body_paths)
                or len(self.finger_paths) != 2 or len(set(self.finger_paths)) != 2
                or len(self.views) != 2 or len(pad_faces) != 2
                or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
            raise ValueError("Exact ordered shaft/two fingers/views and metre stage required")
        self.selected_body = self.body_paths[self.selected_index]
        self.selected_collider = self.selected_body + "/StemCollider"
        if pad_paths is None:
            pad_paths = [p + "/restored_collisions/contact_proxy" for p in self.finger_paths]
        self.pad_paths = tuple(map(str, pad_paths))
        if len(self.pad_paths) != 2:
            raise ValueError("Two exact pad paths required")

        chain = []
        target_ids = []
        for body_path in self.body_paths:
            body = stage.GetPrimAtPath(body_path)
            collider = stage.GetPrimAtPath(body_path + "/StemCollider")
            if (not _active(collider, UsdPhysics) or not collider.IsA(UsdGeom.Capsule)
                    or _owner(collider, UsdPhysics) != body_path):
                raise ValueError("Exact enabled native StemCollider Capsule required")
            _coverage(stage, body_path, Usd, UsdPhysics)
            _body_world_basis(body, UsdGeom)
            cap = UsdGeom.Capsule(collider)
            if cap.GetAxisAttr().Get() != "Z":
                raise ValueError("Authored shaft capsule must use Z axis")
            frame, scale = _local_geometry(collider, body, UsdGeom)
            if not np.allclose(scale, scale[0], atol=1e-12, rtol=1e-8):
                raise ValueError("Nonuniform capsule scale is unsupported")
            chain.append(ShaftCapsule(
                body_path, str(collider.GetPath()), frame,
                float(cap.GetRadiusAttr().Get()) * scale[0],
                float(cap.GetHeightAttr().Get()) * scale[0] / 2, _offset(collider)))
            target_ids.append(body.GetAttribute("tomato:sourceTarget").Get())
        if not target_ids[0] or any(t != target_ids[0] for t in target_ids):
            raise ValueError("Same authored tomato:sourceTarget required for whole chain")
        self.source_target = str(target_ids[0])
        if (not isinstance(getattr(rig, "source_target", None), str)
                or rig.source_target != self.source_target):
            raise ValueError("rig.source_target must match every authored shaft sourceTarget")

        pads = []
        for path, collider_path, (axis, sign) in zip(
                self.finger_paths, self.pad_paths, pad_faces, strict=True):
            body, prim = stage.GetPrimAtPath(path), stage.GetPrimAtPath(collider_path)
            if (not _active(prim, UsdPhysics) or not prim.IsA(UsdGeom.Cube)
                    or _owner(prim, UsdPhysics) != path
                    or _coverage(stage, path, Usd, UsdPhysics) != [collider_path]):
                raise ValueError("Each tensor finger must contain exactly its bound Cube pad")
            _body_world_basis(body, UsdGeom)
            frame, scale = _local_geometry(prim, body, UsdGeom)
            half = scale * float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            pads.append(FingerPad(path, collider_path, frame, half, axis, sign, _offset(prim)))
        # Tensor filters address a body, not a collider label. Never equate that
        # body aggregate to StemCollider if it also carries an enabled leaf.
        if _coverage(stage, self.selected_body, Usd, UsdPhysics) != [self.selected_collider]:
            raise ValueError("Selected tensor body has leaf/extra collider coverage")

        self.core = ShaftGraspEvidence(chain, pads, selected_index=self.selected_index,
            cut_index=self.cut_index, source_target=self.source_target, max_rows=max_rows,
            allow_signed_native_normals=allow_signed_native_normals)
        self.joints = []
        expected = self.core.links
        seen = set()
        root = stage.GetPrimAtPath(rig.root)
        if not root:
            raise ValueError("Missing authored shaft root")
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            a, b = _paths(joint.GetBody0Rel().GetTargets()), _paths(joint.GetBody1Rel().GetTargets())
            if len(a) != 1 or len(b) != 1 or frozenset((a[0], b[0])) not in expected:
                continue
            pair = frozenset((a[0], b[0]))
            if pair in seen:
                raise ValueError("Ambiguous duplicate shaft joint")
            seen.add(pair)
            self.joints.append((str(prim.GetPath()), joint, (a[0], b[0])))
        if seen != expected:
            raise ValueError("Every adjacent detachable link needs exactly one bound joint")
        self.joints.sort(key=lambda x: x[0])
        for i in range(2):
            self._validate_view(i)

        def pack(shape):
            return dict(body=shape.body, collider=shape.collider,
                local_frame=shape.local_frame.tolist(), contact_offset_m=shape.contact_offset_m,
                geometry=([shape.radius_m, shape.half_height_m] if isinstance(shape, ShaftCapsule)
                          else shape.half_extents_m.tolist()),
                face=None if isinstance(shape, ShaftCapsule) else [shape.face_axis, shape.face_sign])
        binding = dict(source_target=self.source_target,
            shapes=[pack(s) for s in (*chain, *pads)],
            joints=[dict(path=p, body0=ab[0], body1=ab[1]) for p, _, ab in self.joints],
            selected_body=self.selected_body, selected_collider=self.selected_collider)
        if allow_signed_native_normals:
            binding.update(allow_signed_native_normals=True,
                           sensor_contract=_sensor_contract_record(sensor_contract))
        self.binding_sha256 = hashlib.sha256(json.dumps(binding, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self._step = 0
        self._watched_bodies = (*self.body_paths, *self.finger_paths)
        self._joint_parents = {p.rsplit("/", 1)[0] for p, _, _ in self.joints}
        self._notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._changed, stage)

    def _changed(self, notice, sender):
        # USD body matrix edits are read only to reject scale/shear/reflection;
        # they never replace or authenticate the post-fetch native body poses.
        for path in notice.GetResyncedPaths():
            p = str(path.GetPrimPath())
            if p == "/" or any(p == b or p.startswith(b + "/") or b.startswith(p + "/")
                               for b in self._watched_bodies):
                self.binding_error = "Geometry/body coverage resynced; recreate adapter"
            elif p.rsplit("/", 1)[0] in self._joint_parents:
                self.binding_error = "Shaft joint topology resynced; recreate adapter"
        for path in notice.GetChangedInfoOnlyPaths():
            p = str(path.GetPrimPath())
            if p == "/" or any(b.startswith(p + "/") for b in self._watched_bodies):
                self.binding_error = "Stage/ancestor binding changed; recreate adapter"
            elif any(p.startswith(b + "/") for b in self._watched_bodies):
                self.binding_error = "Collider-local geometry/coverage changed; recreate adapter"
            elif p in self._watched_bodies:
                name = path.name
                op = name.split(":")[1] if name.startswith("xformOp:") else None
                if name == "xformOpOrder" or op == "scale":
                    self.binding_error = "Body scale/xformOpOrder changed; recreate adapter"
                elif op in ("translate", "orient", "transform"):
                    try:
                        _body_world_basis(self.stage.GetPrimAtPath(p), self._geom)
                    except Exception as exc:
                        self.binding_error = "Invalid edited body world basis: " + str(exc)
                elif name not in ("physics:velocity", "physics:angularVelocity"):
                    self.binding_error = "Bound body identity/settings changed; recreate adapter"

    def _validate_view(self, i):
        view = self.views[i]
        if (view.sensor_count != 1 or view.filter_count != 1
                or _paths(view.sensor_paths) != [self.finger_paths[i]]
                or _paths(view.filter_paths) != [self.selected_body]
                or _index(view.max_contact_data_count) < 1):
            raise ValueError("Tensor view must bind exact finger -> selected body, without wildcards")

    @property
    def allow_signed_native_normals(self):
        return self._allow_signed_native_normals

    @property
    def sensor_contract(self):
        return self._sensor_contract

    def _healthy(self):
        if getattr(self.rig, "source_target", None) != self.source_target:
            self.binding_error = "rig.source_target changed; recreate adapter"
        if self.core.allow_signed_native_normals != self.allow_signed_native_normals:
            self.binding_error = "Bound core signed-native mode changed; recreate adapter"
        if self.closed or self.binding_error is not None or self.error is not None:
            raise ValueError("Invalid shaft-grasp adapter: " + str(self.binding_error or self.error))

    def begin_step(self):
        self._healthy()
        self._step += 1
        self.core.begin_step(self._step)
        self._diagnostic_rows = []
        self._diagnostic_rows_seen = 0
        self._diagnostic_capture_closed = False
        return self._step

    def add_contact(self, collider0, collider1, point, normal, impulse, separation):
        """Observer for original-order full-report NORMAL rows; copies via core."""
        row = None
        core_called = False
        try:
            if self.diagnostic_noncompressive_report or self.allow_signed_native_normals:
                if self._diagnostic_capture_closed:
                    raise ValueError("Normal contact delivered after post-fetch diagnostic boundary")
                row = dict(step_id=self._step, row_index=self._diagnostic_rows_seen,
                    collider0=_diagnostic_text(collider0), collider1=_diagnostic_text(collider1),
                    point=_diagnostic_vector(point), normal=_diagnostic_vector(normal),
                    impulse=_diagnostic_vector(impulse), separation=_diagnostic_number(separation))
                self._diagnostic_rows_seen += 1
                if len(self._diagnostic_rows) < 256:
                    self._diagnostic_rows.append(row)
                elif not self._pending_noncompressive:
                    raise ValueError("Normal contact diagnostic buffer overflow (256)")
                if self._pending_noncompressive:
                    # The core and adapter stay faulted. Drain only delivered
                    # rows of this step, never ask the faulted core for evidence.
                    return
            self._healthy()
            core_called = True
            self.core.add_contact(collider0, collider1, point, normal, impulse, separation)
        except Exception as exc:
            # Preserve the FIRST failing original-order row, not a later row
            # rejected because this fault is already latched. Never normalize,
            # project or filter these diagnostics, or relax the core checks.
            if self.last_failed_contact is None:
                self.last_failed_contact = dict(
                    step_id=self._step,
                    collider0=_diagnostic_text(collider0),
                    collider1=_diagnostic_text(collider1),
                    point=_diagnostic_vector(point),
                    normal=_diagnostic_vector(normal),
                    impulse=_diagnostic_vector(impulse),
                    separation=_diagnostic_number(separation),
                    error=_diagnostic_text(str(exc)))
                if row is not None:
                    self.last_failed_contact["row_index"] = row["row_index"]
                self.error = json.dumps(
                    {"last_failed_contact": self.last_failed_contact},
                    sort_keys=True, separators=(",", ":"), allow_nan=False)
                if (self.diagnostic_noncompressive_report and core_called
                        and str(exc) == "Normal impulse must be compressive and collinear; no friction"):
                    # Classify for diagnostic exception timing ONLY. The core
                    # has already rejected; no independent acceptance tolerance.
                    n, j = np.asarray(normal, float), np.asarray(impulse, float)
                    with np.errstate(over="ignore", invalid="ignore"):
                        projection = float(j @ n)
                    if np.isfinite(projection) and projection < 0:
                        self._pending_noncompressive = True
                        return
            raise ValueError(self.error) from exc

    def _diagnostic_tensor(self, i, dt, remaining):
        """Bounded raw selected rows, NOT the strict grasp/tensor validator."""
        result = dict(finger=self.finger_paths[i], selected_body=self.selected_body,
            selected_collider=self.selected_collider, rows=[], complete=False,
            signed_force_n=None, error=None)
        try:
            self._validate_view(i)
            view = self.views[i]
            size = _index(view.max_contact_data_count)
            f, p, n, d, c, s = [np.asarray(x) for x in view.get_contact_data(dt)]
            result.update(capacity=size, shapes=[list(x.shape) for x in (f, p, n, d, c, s)],
                count=_diagnostic_number(c[0, 0]) if c.shape == (1, 1) else _diagnostic_text(c),
                start=_diagnostic_number(s[0, 0]) if s.shape == (1, 1) else _diagnostic_text(s))
            if (f.shape != (size, 1) or p.shape != (size, 3) or n.shape != (size, 3)
                    or d.shape != (size, 1) or c.shape != (1, 1) or s.shape != (1, 1)):
                raise ValueError("Unexpected tensor normal contact buffer layout")
            count, start = _index(c[0, 0]), _index(s[0, 0])
            if start + count > size:
                raise ValueError("Invalid tensor contact span")
            take = min(count, remaining)
            result.update(rows_dropped=count-take, capacity_exhausted=start+count >= size)
            finite = True
            force = np.zeros(3)
            for k in range(start, start+take):
                scalar = float(f[k, 0])
                normal = np.asarray(n[k], float)
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    vector = scalar * normal
                    impulse = scalar * dt
                    force += vector
                valid = bool(np.isfinite([scalar, impulse, *p[k], *normal, d[k, 0], *vector]).all())
                finite &= valid
                result["rows"].append(dict(buffer_index=k,
                    raw_force_n=_diagnostic_number(scalar), raw_impulse_ns=_diagnostic_number(impulse),
                    point=_diagnostic_vector(p[k]), normal=_diagnostic_vector(n[k]),
                    separation=_diagnostic_number(d[k, 0]),
                    signed_force_vector_n=_diagnostic_vector(vector), finite=valid))
            self._validate_view(i)
            result["complete"] = bool(take == count and start+count < size and finite
                                      and np.isfinite(force).all())
            if result["complete"]:
                result["signed_force_n"] = force.tolist()
        except Exception as exc:
            result["error"] = _diagnostic_text(exc)
            result["complete"] = False
            result["signed_force_n"] = None
        return result

    def raise_pending_fault(self, dt, *, step_id):
        """First post-fetch guard: snapshot signed tensors then ALWAYS raise on fault.

        No native step or body-frame query. The caller guarantees fetch completed
        and no control/release/motion occurred since it. step_id is checked, not
        independent native synchronization proof. Repeated calls never reread
        tensors; begin_step cannot clear the first fault. Invalid dt/step/binding
        prevents tensor access and remains explicit in the failing report.
        """
        if self._diagnostic_error_json is not None:
            raise ValueError(self._diagnostic_error_json)
        self._diagnostic_capture_closed = True
        validation_error = None
        try:
            if self.core.step_id is None or _index(step_id) != self._step:
                raise ValueError("Same-step post-fetch diagnostic ID required")
            if (isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt)
                    or not np.isfinite(dt) or not 0 < dt <= 1):
                raise ValueError("Finite positive step dt required")
            if self.closed or self.binding_error is not None:
                raise ValueError("Closed or invalid diagnostic binding: " + str(self.binding_error))
            if getattr(self.rig, "source_target", None) != self.source_target:
                raise ValueError("rig.source_target changed")
        except Exception as exc:
            validation_error = _diagnostic_text(exc)
        if not self._pending_noncompressive:
            if validation_error is not None:
                if self.error is None:
                    self.error = validation_error
                raise ValueError(self.error)
            self._healthy()
            return

        report = dict(diagnostic_only=True, adapter_valid=False, bilateral=False, stem_only=False,
            step_id=self._step, post_fetch_step_id=_diagnostic_number(step_id),
            dt_s=_diagnostic_number(dt), source_target=self.source_target,
            binding_sha256=self.binding_sha256, last_failed_contact=self.last_failed_contact,
            validation_error=validation_error, tensor_selected=[],
            native_sensor_synchronization_verified=False,
            frame_freshness="caller_post_fetch_contract",
            callback=dict(capacity=256, rows_seen=self._diagnostic_rows_seen,
                rows_captured=len(self._diagnostic_rows),
                rows_dropped=self._diagnostic_rows_seen-len(self._diagnostic_rows),
                all_delivered_rows_captured=self._diagnostic_rows_seen == len(self._diagnostic_rows),
                coverage_scope="normal rows delivered to this observer; native completeness unverified",
                rows=self._diagnostic_rows))
        if validation_error is None:
            remaining = 256  # Separate shared budget across BOTH selected tensors.
            for i in range(2):
                result = self._diagnostic_tensor(i, float(dt), remaining)
                report["tensor_selected"].append(result)
                remaining -= len(result["rows"])
            if (self.closed or self.binding_error is not None
                    or getattr(self.rig, "source_target", None) != self.source_target):
                report["validation_error"] = "Binding changed during diagnostic tensor read"
        self.last_fault_report = report
        self._diagnostic_error_json = json.dumps(report, sort_keys=True,
            separators=(",", ":"), allow_nan=False)
        self.error = self._diagnostic_error_json
        raise ValueError(self._diagnostic_error_json)

    def _connected(self):
        connected = []
        for path, joint, expected in self.joints:
            if not joint or not joint.GetPrim().IsActive():
                raise ValueError("Cached shaft joint removed: " + path)
            actual = (_paths(joint.GetBody0Rel().GetTargets()), _paths(joint.GetBody1Rel().GetTargets()))
            if actual != ([expected[0]], [expected[1]]):
                raise ValueError("Cached shaft joint endpoints changed: " + path)
            enabled = joint.GetJointEnabledAttr().Get()
            if not isinstance(enabled, bool):
                raise ValueError("Unknown live jointEnabled state")
            if enabled:
                connected.append(expected)
        return connected

    def _tensor_rows(self, i, dt):
        self._validate_view(i)
        view = self.views[i]
        forces, points, normals, distances, counts, starts = view.get_contact_data(dt)
        f, p, n, d = [np.asarray(x, dtype=float) for x in (forces, points, normals, distances)]
        c, s = np.asarray(counts), np.asarray(starts)
        size = _index(view.max_contact_data_count)
        if (f.shape != (size, 1) or p.shape != (size, 3) or n.shape != (size, 3)
                or d.shape != (size, 1) or c.shape != (1, 1) or s.shape != (1, 1)):
            raise ValueError("Unexpected tensor normal contact buffer layout")
        count, start = _index(c[0, 0]), _index(s[0, 0])
        if start + count >= size:
            raise ValueError("Selected tensor contact buffer exhausted or invalid")
        sl = slice(start, start + count)
        if (not all(np.isfinite(x[sl]).all() for x in (f, p, n, d))
                or np.any(f[sl] < 0)
                or (count and not np.allclose(np.linalg.norm(n[sl], axis=1), 1,
                                              atol=1e-4, rtol=0))):
            raise ValueError("Invalid tensor normal contact rows")
        return f[sl, 0].copy(), p[sl].copy(), n[sl].copy(), d[sl, 0].copy(), start

    def _tensor_force(self, i, dt):
        f, _, n, _, _ = self._tensor_rows(i, dt)
        force = np.sum(f[:, None] * n, axis=0)
        if not np.isfinite(force).all():
            raise ValueError("Nonfinite tensor force")
        return force, len(f)

    def _match_selected_rows(self, i, dt, signed_force):
        """Observed native37 magnitude contract, never signed physical evidence.

        Match ONLY identities and geometry; magnitude never disambiguates rows.
        Every selected row, including zero impulse, needs exactly one counterpart.
        Ambiguous neighborhoods fail closed rather than fitting a permutation.
        """
        f, p, n, d, start = self._tensor_rows(i, dt)
        pad = self.pad_paths[i]
        rows = [r for r in self._diagnostic_rows
                if ((r["collider0"] == pad and r["collider1"] == self.selected_collider)
                    or (r["collider1"] == pad and r["collider0"] == self.selected_collider))]
        result = dict(finger=self.finger_paths[i], collider=self.selected_collider,
            comparison="native37_observed_selected_row_geometry_and_impulse_magnitude",
            physical_force_source="signed_callback_only",
            callback_selected_force_n=signed_force.tolist(),
            callback_contact_count=len(rows), tensor_contact_count=len(f),
            matches=[], passed=False, reason=None)
        if any(r["collider0"] != pad for r in rows):
            result["reason"] = "selected_header_order_not_qualified_by_native37"
            return result
        if len(rows) != len(f):
            result["reason"] = "selected_row_count_mismatch"
            return result
        if not rows:
            result["passed"] = True
            return result
        cp = np.asarray([r["point"] for r in rows], float)
        cn = np.asarray([r["normal"] for r in rows], float)
        cd = np.asarray([r["separation"] for r in rows], float)
        cj = np.asarray([r["impulse"] for r in rows], float)
        if not all(np.isfinite(v).all() for v in (cp, cn, cd, cj)):
            raise ValueError("Nonfinite selected callback row")
        position_error = np.max(np.abs(cp[:, None, :] - p[None, :, :]), axis=2)
        normal_error = np.max(np.abs(cn[:, None, :] - n[None, :, :]), axis=2)
        separation_error = np.abs(cd[:, None] - d[None, :])
        candidates = ((position_error <= _MATCH_POSITION_M)
                      & (normal_error <= _MATCH_NORMAL)
                      & (separation_error <= _MATCH_SEPARATION_M))
        if not (np.all(candidates.sum(axis=0) == 1) and np.all(candidates.sum(axis=1) == 1)):
            result["reason"] = "missing_or_ambiguous_selected_geometry_match"
            return result
        for row_index, tensor_index in enumerate(np.argmax(candidates, axis=1)):
            magnitude = float(np.linalg.norm(cj[row_index]))
            scalar = float(cj[row_index] @ (cn[row_index] / np.linalg.norm(cn[row_index])))
            tensor_impulse = float(f[tensor_index] * dt)
            tolerance = _MATCH_IMPULSE_ATOL_NS + _MATCH_IMPULSE_RTOL * max(magnitude, tensor_impulse)
            error = abs(magnitude - tensor_impulse)
            zero_underflow = bool(f[tensor_index] == 0. and magnitude > 0.
                and np.all(np.abs(cj[row_index]) < _MATCH_ZERO_COMPONENT_NS))
            # Relative accuracy degrades when the SUM OF SQUARES is subnormal,
            # even if its nonzero square root is a normal float32 value. Bound
            # the rounding error in squared-impulse units, not physical force.
            squared_error = abs(magnitude*magnitude-tensor_impulse*tensor_impulse)
            subnormal_norm = bool(magnitude > 0. and tensor_impulse > 0.
                and max(magnitude*magnitude,tensor_impulse*tensor_impulse)
                < float(np.finfo(np.float32).tiny) and squared_error <= _MATCH_SQUARED_ATOL_NS2)
            if not np.isfinite([magnitude, scalar, tensor_impulse, tolerance, error,
                    squared_error,magnitude*magnitude,tensor_impulse*tensor_impulse]).all():
                raise ValueError("Nonfinite selected impulse comparison")
            result["matches"].append(dict(callback_row_index=rows[row_index]["row_index"],
                tensor_buffer_index=int(start+tensor_index),
                point_error_max_m=float(position_error[row_index, tensor_index]),
                normal_error_max=float(normal_error[row_index, tensor_index]),
                separation_error_m=float(separation_error[row_index, tensor_index]),
                callback_signed_projection_ns=scalar, callback_magnitude_ns=magnitude,
                tensor_magnitude_ns=tensor_impulse, magnitude_error_ns=error,
                magnitude_tolerance_ns=tolerance, tensor_zero_underflow=zero_underflow,
                squared_impulse_error_ns2=squared_error, subnormal_squared_norm=subnormal_norm,
                passed=bool(error <= tolerance or zero_underflow or subnormal_norm)))
        result["passed"] = all(m["passed"] for m in result["matches"])
        if not result["passed"]:
            result["reason"] = "selected_impulse_magnitude_mismatch"
        return result

    def evaluate(self, dt, frames, nativefingerframes, *, frames_step_id=None):
        """Read cached joints and same-step tensors, then return old-compatible keys.

        Strict-default crosscheck tolerance is 1e-5 N + 1e-4 * maximum vector magnitude,
        numerical reconciliation only, never a change to the 20 mN grasp gate.
        This tolerance/sign convention still requires native qualification.
        A mismatch blocks bilateral/stem_only and remains explicit in diagnostics.
        """
        try:
            if self.diagnostic_noncompressive_report or self.allow_signed_native_normals:
                self.raise_pending_fault(dt, step_id=self._step if frames_step_id is None else frames_step_id)
            self._healthy()
            if self.core.step_id is None or self.core.evaluated:
                raise ValueError("begin_step then evaluate once required")
            if frames_step_id is not None and _index(frames_step_id) != self._step:
                raise ValueError("Stale post-fetch frames")
            if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0 or dt > 1:
                raise ValueError("Finite positive step dt required")
            def ordered(value, paths):
                if isinstance(value, dict):
                    return {p: _pose(value[p]) for p in paths}
                array = np.asarray(value, dtype=float)
                if array.shape != (len(paths), 4, 4):
                    raise ValueError("Exact ordered body frame array required")
                return {p: _pose(m) for p, m in zip(paths, array, strict=True)}
            world = ordered(frames, self.body_paths)
            world.update(ordered(nativefingerframes, self.finger_paths))
            connected = self._connected()
            result = self.core.evaluate(step_id=self._step, frames_step_id=self._step,
                dt=float(dt), body_frames=world, connected_pairs=connected)
            selected = np.zeros((2, 3))
            # Core counts exact selected-collider rows separately from neighbours.
            for i, path in enumerate(self.finger_paths):
                for pair in result["pairs"]:
                    if pair["finger"] == path and pair["collider"] == self.selected_collider:
                        selected[i] += pair["force_n"]
            crosschecks = []
            for i in range(2):
                if self.allow_signed_native_normals:
                    crosschecks.append(self._match_selected_rows(i, float(dt), selected[i]))
                    continue
                force, count = self._tensor_force(i, float(dt))
                error = float(np.linalg.norm(selected[i] - force))
                tolerance = 1e-5 + 1e-4 * max(float(np.linalg.norm(selected[i])), float(np.linalg.norm(force)))
                crosschecks.append(dict(finger=self.finger_paths[i],
                    collider=self.selected_collider, callback_selected_force_n=selected[i].tolist(),
                    tensor_selected_force_n=force.tolist(), tensor_contact_count=count,
                    vector_error_n=error, numerical_tolerance_n=tolerance, passed=error <= tolerance))
            self._healthy()  # No binding edits may sneak in during tensor fetch.
            passed = all(c["passed"] for c in crosschecks)
            result.update(bilateral=bool(result["bilateral"] and passed),
                stem_only=bool(result["stem_only"] and passed),
                adapter_valid=passed, tensor_selected_crosscheck=crosschecks,
                binding_sha256=self.binding_sha256, cached_joint_count=len(self.joints),
                connected_pair_count=len(connected), native_grasp_certified=False,
                source_binding_scope="authored geometry/coverage and same-step force reconciliation",
                frame_freshness="caller_post_fetch_contract",
                force_convention="world force on finger; +J/dt if finger is collider0, -J/dt otherwise")
            if self.allow_signed_native_normals:
                result.update(sensor_contract=_sensor_contract_record(self.sensor_contract),
                    tensor_integrity_only=True, physical_force_source="signed_callback_only",
                    source_binding_scope="authored geometry/coverage and observed-contract selected-row integrity",
                    normal_callback_rows_seen=self._diagnostic_rows_seen,
                    normal_callback_rows_captured=len(self._diagnostic_rows))
            return result
        except Exception as exc:
            if self.error is None:
                self.error = str(exc)
            raise

    def close(self):
        """Release only this adapter's passive USD notice; never touches physics."""
        if self._notice is not None:
            self._notice.Revoke()
            self._notice = None
        self.closed = True
