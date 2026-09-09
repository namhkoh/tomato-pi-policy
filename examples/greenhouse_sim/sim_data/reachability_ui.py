"""Async reach diagnostics and non-colliding candidate outlines, never commands."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
from datetime import datetime, timezone
import uuid

from .reachability import check_reachability
from .reachability_snapshot import capture_snapshot
from .review import _output_directory

GHOST = "/World/Phase1Reachability"


class ReachabilityControls:
    def __init__(self, owner):
        import omni.ui as ui
        self.owner = owner
        self.result = None
        self.task = None
        self.cancel_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anatomy-reach")
        self.epoch = 0
        self.notice = None
        self.closed = False
        self.has_ghost = False
        self.watched = []
        with ui.CollapsableFrame("Automatic reachability (diagnostic only)", collapsed=False):
            with ui.VStack(spacing=4):
                with ui.HStack(height=28):
                    self.run_button = ui.Button("Check both arms", clicked_fn=self.start)
                    ui.Button("Cancel", clicked_fn=self.cancel)
                self.label = ui.Label("Not checked. Fixed base/torso; attachment probe only.\nNo robot motion, grasp approval, or path planning.", word_wrap=True, height=175)
                with ui.HStack(height=26):
                    self.left_button = ui.Button("Show left IK outline", clicked_fn=lambda: self.show_candidate("left"), enabled=False)
                    self.right_button = ui.Button("Show right IK outline", clicked_fn=lambda: self.show_candidate("right"), enabled=False)
                    ui.Button("Hide outline", clicked_fn=self.hide_candidate)

    def invalidate(self, message="Target/scene changed; run reachability again."):
        self.epoch += 1
        self.cancel_event.set()
        self.result = None
        self.left_button.enabled = self.right_button.enabled = False
        self.label.text = message
        self.hide_candidate()

    def cancel(self):
        self.invalidate("Cancelled. No robot motion or review decisions were made.")

    def _scene_changed(self, notice, sender):
        # Camera/overlay activity outside these roots must not cancel the job.
        from pxr import Sdf
        for path in [*notice.GetResyncedPaths(), *notice.GetChangedInfoOnlyPaths()]:
            prim = path.GetPrimPath()
            if any(prim.HasPrefix(Sdf.Path(root)) or Sdf.Path(root).HasPrefix(prim) for root in self.watched):
                self.invalidate(f"Scene changed at {path}; result is stale. Recheck.")
                return

    def start(self):
        import omni.timeline
        from pxr import Tf, Usd
        if self.closed or self.owner.busy or not self.owner.entries or (self.task and not self.task.done()):
            return
        self.invalidate("Reading current robot and target geometry...")
        try:
            if omni.timeline.get_timeline_interface().is_playing():
                raise ValueError("Pause the timeline before checking static reachability")
            if not self.owner.scene or self.owner.scene.visibility_paths:
                raise ValueError("Restore the non-isolated review scene first")
            report, target = self.owner.entries[self.owner.index]
            record = {"plant_root": self.owner.scene.plant_root, "component_paths": self.owner.scene.paths}
            snapshot = capture_snapshot(self.owner.stage, record, report, target,
                                        borrowed=record["plant_root"] == self.owner.borrowed_root)
            directory = _output_directory(report, self.owner.output.parent / "reachability")
            if self.notice:
                self.notice.Revoke()
            self.watched = [snapshot["robot_path"], snapshot["plant_root"], "/World/Gutters"]
            self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._scene_changed, self.owner.stage)
            self.cancel_event = threading.Event()
            self.run_button.enabled = False
            self.label.text = "Searching both arms (up to 4 seconds each). UI remains usable.\nFixed base/torso; orientation unconstrained. No robot movement."
            self.task = asyncio.ensure_future(self._run(snapshot, directory, self.epoch, self.cancel_event))
        except (ValueError, RuntimeError, OSError) as exc:
            self.label.text = str(exc)

    async def _run(self, snapshot, directory, epoch, cancellation):
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self.executor, lambda: check_reachability(snapshot, cancel_event=cancellation))
            if self.closed or epoch != self.epoch or cancellation.is_set():
                return
            result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            result["current_scene_matches_at_completion"] = True
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (uuid.uuid4().hex + ".json")
            temporary = path.with_suffix(".pending")
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self.result = result
            self.left_button.enabled = bool(result["arms"].get("left", {}).get("candidate"))
            self.right_button.enabled = bool(result["arms"].get("right", {}).get("candidate"))
            lines = [f"{snapshot['target_id']} / {snapshot['placement']}"]
            for side, item in result["arms"].items():
                error = item["best_position_error_m"]
                suffix = "" if error is None else f"; error {error * 1000:.2f} mm"
                status = item['status']
                if status == "outside_outer_reach_bound":
                    status = "Out of reach from CURRENT base/torso pose"
                    suffix = f" ({item['shoulder_distance_m']:.2f} m > {item['conservative_outer_reach_m']:.2f} m bound)"
                lines.append(f"{side.upper()}: {status}{suffix}")
                if item["candidate"]:
                    lines.append("  Endpoint: " + item["endpoint_screen"]["status"])
            lines.extend(["Base/torso repositioning NOT searched. This is NOT target ineligibility.",
                          "Fixed base/torso; unconstrained orientation. NOT safe grasp/cut approval.",
                          "Partial endpoint screen only; paths and bimanual execution NOT tested.",
                          f"Saved diagnostic: reachability/{path.name}"])
            self.label.text = "\n".join(lines)
            print("REACHABILITY_READY " + str(path), flush=True)
        except Exception as exc:
            if not self.closed and epoch == self.epoch:
                self.label.text = f"Reachability failed: {exc}. No approval was saved."
                print("REACHABILITY_FAILED " + str(exc), flush=True)
        finally:
            if not self.closed:
                self.run_button.enabled = True

    def hide_candidate(self):
        if self.has_ghost:
            from pxr import Usd
            # Clear the ownership flag first: stage notices can call invalidate.
            self.has_ghost = False
            with Usd.EditContext(self.owner.stage, self.owner.stage.GetSessionLayer()):
                self.owner.stage.RemovePrim(GHOST)

    def show_candidate(self, side):
        from pxr import Gf, Usd, UsdGeom
        if self.closed or self.result is None:
            return
        item = self.result["arms"].get(side, {})
        if not item.get("candidate"):
            return
        self.hide_candidate()
        stage = self.owner.stage
        if stage.GetPrimAtPath(GHOST):
            self.label.text = "Candidate outline path is occupied; no scene content replaced."
            return
        candidate = item["candidate"]
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            root = UsdGeom.Xform.Define(stage, GHOST)
            root.GetPrim().SetCustomDataByKey("training_input_allowed", False)
            root.GetPrim().SetCustomDataByKey("diagnostic_position_ik_only", True)
            for index, capsule in enumerate(candidate["arm_capsules"]):
                curve = UsdGeom.BasisCurves.Define(stage, f"{GHOST}/Arm_{index}")
                curve.CreateTypeAttr("linear")
                curve.CreateCurveVertexCountsAttr([2])
                curve.CreatePointsAttr([Gf.Vec3f(*capsule[k]) for k in ("start_m", "end_m")])
                curve.CreateWidthsAttr([.006])
                curve.SetWidthsInterpolation("constant")
                curve.CreateDisplayColorAttr([(1.0, .45, .1)])
            point = UsdGeom.Sphere.Define(stage, GHOST + "/Probe_NOT_Grasp")
            point.CreateRadiusAttr(.007)
            point.AddTranslateOp().Set(Gf.Vec3d(*candidate["probe_world_m"]))
            point.CreateDisplayColorAttr([(1.0, .45, .1)])
        self.has_ghost = True
        self.label.text += f"\nOrange {side} outline: independent point-IK candidate, NOT a motion plan."

    def close(self):
        self.closed = True
        self.cancel_event.set()
        if self.notice:
            self.notice.Revoke()
            self.notice = None
        self.hide_candidate()
        self.executor.shutdown(wait=False, cancel_futures=True)
