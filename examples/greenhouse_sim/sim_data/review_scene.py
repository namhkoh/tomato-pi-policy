"""Reversible, session-only anatomy inspection overlays. Never training inputs."""

from pathlib import Path

from .geometry import bounds_by_component, point_bounds_distance

OVERLAY = "/World/Phase1Review"


class ReviewScene:
    def __init__(self, stage, plant_root, paths, report):
        from pxr import Sdf

        if stage.GetPrimAtPath(OVERLAY):
            raise ValueError("Review overlay path is occupied; refusing to overwrite existing scene content")
        self.stage, self.plant_root, self.paths, self.report = stage, plant_root, paths, report
        self.bounds, self.mesh_counts = bounds_by_component(stage, paths)
        self.saved = Sdf.Layer.CreateAnonymous("review_visibility_backup")
        self.visibility_paths = {}
        self.target = None

    def restore_visibility(self):
        from pxr import Sdf

        session = self.stage.GetSessionLayer()
        with Sdf.ChangeBlock():
            for path, existed in self.visibility_paths.items():
                if existed:
                    Sdf.CopySpec(self.saved, path, session, path)
                else:
                    attr = session.GetAttributeAtPath(path)
                    if attr:
                        attr.owner.RemoveProperty(attr)
        self.visibility_paths.clear()

    def close(self):
        from pxr import Usd

        self.restore_visibility()
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.stage.RemovePrim(OVERLAY)

    def select(self, target):
        from pxr import Gf, Usd, UsdGeom

        self.close()
        self.target = target
        key = target["component_id"]
        cache = UsdGeom.XformCache()
        from .review_camera import attachment_world
        attachment = attachment_world(self.stage, self.paths, self.report, target)
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            root = UsdGeom.Xform.Define(self.stage, OVERLAY)
            root.GetPrim().SetCustomDataByKey("review_only", True)
            root.GetPrim().SetCustomDataByKey("training_input_allowed", False)
            marker = UsdGeom.Sphere.Define(self.stage, OVERLAY + "/Attachment_NOT_CutPoint")
            marker.CreateRadiusAttr(0.005)
            marker.AddTranslateOp().Set(attachment)
            marker.CreateDisplayColorAttr([(1.0, 0.8, 0.0)])
            for index, member in enumerate(target["expected_detached_component_ids"]):
                value = self.bounds[member]
                if not value.IsEmpty():
                    self._box(f"{OVERLAY}/Detached_{index}", value, (0.1, 1.0, 0.6))
            for member, component in self.report["components"].items():
                if component["type"] not in {"main_stem", "truss", "fruit", "flower"}:
                    continue
                value = self.bounds[member]
                if not value.IsEmpty() and (value.GetMidpoint() - attachment).GetLength() < 0.4:
                    self._box(f"{OVERLAY}/Protected_{member}", value, (1.0, 0.15, 0.1))
            component_matrix = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.paths[key]))
            for index, chain in enumerate(self.report["components"][key]["capsules_local_m"]):
                points = [component_matrix.Transform(Gf.Vec3d(*p[:3])) for p in chain]
                self._lines(f"{OVERLAY}/Centerline_{index}", points, [len(points)], (0.0, 0.7, 1.0))
        return attachment

    def _lines(self, path, points, counts, color):
        from pxr import Gf, UsdGeom

        curve = UsdGeom.BasisCurves.Define(self.stage, path)
        curve.CreateTypeAttr("linear")
        curve.CreateWrapAttr("nonperiodic")
        curve.CreateCurveVertexCountsAttr(counts)
        curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        curve.CreateWidthsAttr([0.0015])
        curve.SetWidthsInterpolation("constant")
        curve.CreateDisplayColorAttr([color])

    def _box(self, path, bounds, color):
        low, high = bounds.GetMin(), bounds.GetMax()
        corners = [(high[0] if i & 1 else low[0], high[1] if i & 2 else low[1],
                    high[2] if i & 4 else low[2]) for i in range(8)]
        edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if not i & bit]
        self._lines(path, [corners[i] for edge in edges for i in edge], [2] * len(edges), color)

    def isolate(self):
        from pxr import Sdf, Usd, UsdGeom

        if self.visibility_paths or self.target is None:
            self.restore_visibility()
            return
        # Hide non-target meshes, retaining the target's main-stem parent.
        keep = set(self.target["expected_detached_component_ids"])
        keep.add(self.report["components"][self.target["component_id"]]["parent"])
        owners = {path: key for key, path in self.paths.items()}
        session = self.stage.GetSessionLayer()
        meshes = [p for p in self.stage.Traverse() if p.IsA(UsdGeom.Mesh) or p.IsInstance()]
        with Usd.EditContext(self.stage, session):
            for mesh in meshes:
                owner = mesh
                while owner and str(owner.GetPath()) not in owners:
                    owner = owner.GetParent()
                if owner and owners[str(owner.GetPath())] in keep:
                    continue
                attr = UsdGeom.Imageable(mesh).GetVisibilityAttr()
                path = attr.GetPath()
                existed = bool(session.GetAttributeAtPath(path))
                self.visibility_paths[path] = existed
                if existed:
                    Sdf.CreatePrimInLayer(self.saved, mesh.GetPath())
                    Sdf.CopySpec(session, path, self.saved, path)
                attr.Set("invisible")

    def focus(self, viewport, attachment):
        from pxr import Gf, Usd, UsdGeom

        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            camera = UsdGeom.Camera.Define(self.stage, OVERLAY + "/ReviewCamera")
            camera.CreateFocalLengthAttr(24.0)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
            matrix = Gf.Matrix4d().SetLookAt(attachment + Gf.Vec3d(0.65, -0.5, 0.25), attachment, Gf.Vec3d(0, 0, 1))
            UsdGeom.Xformable(camera).MakeMatrixXform().Set(matrix.GetInverse())
            viewport.set_active_camera(str(camera.GetPath()))


class ReviewPanel:
    def __init__(self, stage, records, viewport, output):
        import omni.ui as ui
        from .audit import audit_manifest

        self.viewport, self.stage, self.output = viewport, stage, Path(output)
        self.previous_camera = viewport.camera_path
        self.records = records
        self.reports = [audit_manifest(record["manifest_path"]) for record in records]
        self.entries = [(i, target) for i, report in enumerate(self.reports) for target in report["targets"]]
        if not self.entries:
            raise ValueError("No substems available for review")
        self.index = next((i for i, (_, target) in enumerate(self.entries) if target["status"] == "needs_review"), 0)
        self.scene = None
        self.window = ui.Window("Phase 1 - anatomy review ONLY", width=470, height=520)
        with self.window.frame:
            with ui.VStack(spacing=6):
                self.label = ui.Label("", word_wrap=True, height=150)
                ui.Label("Yellow: attachment, NOT approved cut. Green: detached subtree.\nRed: nearby protected structures. Blue: manifest centerline.", word_wrap=True, height=55)
                with ui.HStack(height=28):
                    ui.Button("Previous", clicked_fn=lambda: self.move(-1))
                    ui.Button("Next", clicked_fn=lambda: self.move(1))
                ui.Button("Isolate target / restore scene", clicked_fn=self.isolate, height=28)
                ui.Label("Reviewer name:", height=20)
                self.reviewer = ui.StringField(height=25).model
                ui.Label("Review notes (required):", height=20)
                self.notes = ui.StringField(height=25).model
                with ui.HStack(height=30):
                    ui.Button("Anatomy confirmed", clicked_fn=lambda: self.save("anatomy_confirmed"))
                    ui.Button("Exclude", clicked_fn=lambda: self.save("excluded"))
                    ui.Button("Unresolved", clicked_fn=lambda: self.save("unresolved"))
                self.message = ui.Label("No safe cut/grasp region or execution approval is set.", word_wrap=True, height=55)
                ui.Button("Close review / restore scene", clicked_fn=self.close, height=28)
        self.window.set_visibility_changed_fn(lambda visible: self.close() if not visible else None)
        self.show()

    def show(self):
        plant_index, target = self.entries[self.index]
        if self.scene:
            self.viewport.set_active_camera(str(self.previous_camera))
            self.scene.close()
        record, report = self.records[plant_index], self.reports[plant_index]
        self.scene = ReviewScene(self.stage, record["plant_root"], record["component_paths"], report)
        attachment = self.scene.select(target)
        self.scene.focus(self.viewport, attachment)
        parent = report["components"][target["component_id"]]["parent"]
        gap = point_bounds_distance(attachment, self.scene.bounds[parent]) if parent else None
        gap_text = "unknown" if gap is None else f"{gap * 1000:.2f} mm (diagnostic only)"
        self.label.text = (f"{self.index + 1}/{len(self.entries)}  {target['target_id']}\n"
                           f"{target['status']}: {', '.join(target['reason_codes'])}\n"
                           f"Detached: {target['descendant_type_counts']}\nParent AABB gap: {gap_text}\n"
                           "Agronomic eligibility: UNREVIEWED\nVisibility: NOT MEASURED\nPhysical execution: NOT TESTED")

    def move(self, delta):
        self.index = (self.index + delta) % len(self.entries)
        self.notes.set_value("")
        self.show()

    def isolate(self):
        self.scene.isolate()

    def save(self, decision):
        from .review import record_review

        index, target = self.entries[self.index]
        try:
            path = record_review(self.reports[index], target["target_id"], decision,
                                 self.reviewer.as_string, self.notes.as_string, self.output)
            self.message.text = f"Saved {path.name}. This is NOT cut or execution approval."
        except (OSError, ValueError) as exc:
            self.message.text = str(exc)

    def close(self):
        if self.scene:
            self.viewport.set_active_camera(str(self.previous_camera))
            self.scene.close()
            self.scene = None
        self.window.set_visibility_changed_fn(None)
        self.window.visible = False
