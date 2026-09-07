"""Six-card, explicit-selection anatomy review. Captures are NEVER model inputs."""

import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

from .audit import audit_manifest, ROOT
from .batch import History, attach_cached_geometry, fingerprints, review_queue, target_flags
from .review import REASONS, record_batch, record_review
from .review_scene import ReviewScene


class BatchReviewPanel:
    MODES = ("Mixed sample", "Exceptions", "Unreviewed", "All / revisit")

    def __init__(self, stage, records, viewport, output, *, package=None, history_directories=None):
        import omni.ui as ui

        self.stage, self.viewport, self.output = stage, viewport, Path(output)
        self.previous_camera = viewport.camera_path
        self.records = {Path(r["manifest_path"]).parent.name: r for r in records}
        manifests = sorted((Path(package) / "plants/components").glob("*/manifest.json")) if package else [Path(r["manifest_path"]) for r in records]
        self.reports = [audit_manifest(p) for p in manifests]
        attach_cached_geometry(self.reports, ROOT / "data/sim_data/audits")
        directories = history_directories if history_directories is not None else list((ROOT / "data/sim_data").glob("*/anatomy_reviews"))
        self.history = History(self.reports, [*directories, self.output])
        self.scene = None
        self.borrowed_root = "/World/Phase1ReviewSample"
        if stage.GetPrimAtPath(self.borrowed_root):
            raise ValueError("Sample-plant path occupied; refusing to overwrite")
        self.borrowed_record, self.hidden = None, []
        self.capture_task = None
        self.closed = False
        self.busy = False
        self.seen = set()
        self.page, self.evidence, self.checked = [], {}, {}
        self.index = 0
        self.entries = []
        self.reason_keys = [None, *REASONS]
        self.reviewer = ui.SimpleStringModel("")
        previous = [r for rows in self.history.rows.values() for r in rows]
        if previous:
            self.reviewer.set_value(max(previous, key=lambda r: r["timestamp_utc"])["reviewer"])
        self.notes = ui.SimpleStringModel("")
        self.window = ui.Window("Phase 1 - batch anatomy review", width=520, height=690)
        self.gallery = ui.Window("Phase 1 - six-image review gallery", width=1080, height=800, visible=False)
        self.gallery.set_visibility_changed_fn(self._gallery_visibility)
        with self.window.frame:
            with ui.ScrollingFrame():
                contents = ui.VStack(spacing=5)
            with contents:
                self.summary = ui.Label("", word_wrap=True, height=65)
                ui.Label("Queue (changes do not save decisions):", height=20)
                self.mode = ui.ComboBox(0, *self.MODES).model
                self.mode.add_item_changed_fn(lambda *_: self.reset_queue())
                self.label = ui.Label("", word_wrap=True, height=155)
                ui.Label("Yellow: attachment, NOT cut point. Green: expected detached parts.\nRed: nearby protected organs. Blue: supplied centreline.", word_wrap=True, height=40)
                with ui.HStack(height=28):
                    ui.Button("Previous", clicked_fn=lambda: self.move(-1))
                    ui.Button("Next unreviewed", clicked_fn=lambda: self.move(1))
                ui.Button("Isolate target / restore scene", clicked_fn=self.isolate, height=28)
                ui.Button("Build six-image gallery", clicked_fn=self.start_gallery, height=30)
                ui.Label("Reviewer (required):", height=20)
                ui.StringField(model=self.reviewer, height=25)
                self.reason = ui.ComboBox(0, "Choose a reason...", *[r[0] for r in REASONS.values()]).model
                ui.Label("Additional notes (optional; preset reason is recorded):", height=20)
                ui.StringField(model=self.notes, height=25)
                ui.Button("Save current reason + next", clicked_fn=self.save_current, height=30)
                ui.Label("Saving a new decision supersedes prior decisions for this target; originals are retained.", word_wrap=True, height=28)
                self.message = ui.Label("No cut/grasp or physical approval is set.", word_wrap=True, height=65)
                ui.Button("Close review / restore scene", clicked_fn=self.close, height=28)
        self.window.set_visibility_changed_fn(lambda visible: self.close() if not visible else None)
        self.reset_queue()

    def queue_mode(self):
        return self.MODES[self.mode.get_item_value_model().as_int]

    def update_summary(self):
        reviewed = sum(self.history.complete(t["target_id"]) for r in self.reports for t in r["targets"])
        reviewed_plants = len({key.split('/')[0] for key in self.history.rows})
        total = sum(len(r["targets"]) for r in self.reports)
        self.summary.text = (f"{reviewed} completed reviews; feedback covers {reviewed_plants} plants. No per-target review quota.\n"
                             f"{len(self.reports)} catalog plants / {total} audited targets. {len(self.history.ignored)} stale/unreadable records ignored.\n"
                             "Automated checks are NOT human approval.")

    def superseded(self, target_id):
        return [row["review_id"] for row in self.history.rows.get(target_id, [])]

    def reset_queue(self):
        if self.busy or self.closed:
            return
        self.history.refresh()
        self.seen.clear()
        self.entries = review_queue(self.reports, self.history, self.queue_mode())
        self.index = 0
        self.update_summary()
        self.show()

    def _restore_borrowed(self):
        from pxr import Sdf, Usd

        session = self.stage.GetSessionLayer()
        with Usd.EditContext(self.stage, session):
            for path, backup in self.hidden:
                if backup is not None:
                    Sdf.CopySpec(backup, path, session, path)
                else:
                    attr = session.GetAttributeAtPath(path)
                    if attr:
                        attr.owner.RemoveProperty(attr)
            self.hidden.clear()
            if self.borrowed_record:
                self.stage.RemovePrim(self.borrowed_root)
                self.borrowed_record = None

    def _record(self, report):
        from pxr import Sdf, Usd, UsdGeom
        from .geometry import assemble_plant

        key = report["plant_id"]
        if key in self.records:
            self._restore_borrowed()
            return self.records[key]
        if self.borrowed_record and self.borrowed_record["plant_id"] == key:
            return self.borrowed_record
        self._restore_borrowed()
        anchor = next(iter(self.records.values()))["plant_root"]
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(anchor))
        # At most one additional plant is loaded. Existing session opinions survive.
        paths = assemble_plant(self.stage, self.borrowed_root, report)
        self.borrowed_record = {"plant_id": key, "plant_root": self.borrowed_root, "component_paths": paths}
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            UsdGeom.Xformable(self.stage.GetPrimAtPath(self.borrowed_root)).MakeMatrixXform().Set(matrix)
            for record in self.records.values():
                attr = UsdGeom.Imageable(self.stage.GetPrimAtPath(record["plant_root"])).GetVisibilityAttr()
                path = attr.GetPath()
                backup = None
                if self.stage.GetSessionLayer().GetAttributeAtPath(path):
                    backup = Sdf.Layer.CreateAnonymous("batch_visibility_backup")
                    Sdf.CreatePrimInLayer(backup, path.GetPrimPath())
                    Sdf.CopySpec(self.stage.GetSessionLayer(), path, backup, path)
                self.hidden.append((path, backup))
                attr.Set("invisible")
        return self.borrowed_record

    def _display(self, entry):
        if self.scene:
            self.viewport.set_active_camera(str(self.previous_camera))
            self.scene.close()
            self.scene = None
        report, target = entry
        if report["status"] == "blocked":
            raise ValueError("Structurally blocked asset cannot be assembled; inspect its audit")
        record = self._record(report)
        self.scene = ReviewScene(self.stage, record["plant_root"], record["component_paths"], report)
        attachment = self.scene.select(target)
        self.scene.focus(self.viewport, attachment)

    def description(self, entry):
        report, target = entry
        flags = target_flags(report, target) + self.history.issues(target["target_id"])
        previous = self.history.latest(target["target_id"])
        state = ("Flags: " + ", ".join(flags)) if flags else "Automated structure/AABB checks passed (limited scope)."
        saved = "No saved human decision" if previous is None else f"Saved: {previous['decision']} / {previous.get('review_scope', 'legacy')} - {previous['notes']}"
        return f"{target['target_id']}\n{target['status']}: {', '.join(target['reason_codes'])}\n{state}\n{saved}"

    def card_description(self, entry):
        report, target = entry
        flags = target_flags(report, target) + self.history.issues(target["target_id"])
        previous = self.history.latest(target["target_id"])
        last = "Unreviewed" if previous is None else "Prior: " + previous["decision"]
        checks = f"{len(flags)} flag(s): inspect in 3D" if flags else "Limited automatic checks passed"
        return f"{target['target_id']}\n{target['status']} | {checks}\n{last}"

    def show(self):
        if not self.entries:
            self.label.text = "No targets in this queue. Choose All / revisit or another queue."
            return
        entry = self.entries[self.index]
        self.label.text = self.description(entry) + "\nCut/grasp: NOT APPROVED. Physical execution: NOT TESTED."
        try:
            self._display(entry)
        except (ValueError, RuntimeError) as exc:
            self.message.text = str(exc)

    def move(self, delta):
        if self.busy or not self.entries or self.closed:
            return
        self.index = (self.index + delta) % len(self.entries)
        self.notes.set_value("")
        self.show()

    def isolate(self):
        if not self.busy and self.scene:
            self.scene.isolate()

    def save_current(self):
        key = self.reason_keys[self.reason.get_item_value_model().as_int]
        self.save_reason(key)

    def save(self, decision):
        # Compatibility with the original smoke: an empty identity must reject.
        key = next(k for k, v in REASONS.items() if v[1] == decision)
        self.save_reason(key)

    def save_reason(self, key):
        if self.busy or self.closed or not self.entries:
            return
        try:
            if key not in REASONS:
                raise ValueError("Choose a reason before saving")
            report, target = self.entries[self.index]
            reason = REASONS[key]
            path = record_review(report, target["target_id"], reason[1], self.reviewer.as_string,
                                 self.notes.as_string.strip() or reason[3], self.output, reason_code=key,
                                 supersedes=self.superseded(target["target_id"]))
            self.history.refresh()
            self.update_summary()
            self.entries = review_queue(self.reports, self.history, self.queue_mode())
            # An unresolved item stays in Exceptions but does not immediately repeat.
            if len(self.entries) > 1 and self.entries[self.index % len(self.entries)][1]["target_id"] == target["target_id"]:
                self.index += 1
            self.index %= max(1, len(self.entries))
            self.notes.set_value("")
            self.show()
            self.message.text = f"Saved {path.name}; advanced. No cut approval."
        except (OSError, ValueError) as exc:
            self.message.text = str(exc)

    def start_gallery(self, *, restart=False):
        if self.busy or self.closed:
            return
        if restart:
            self.seen.clear()
        self.history.refresh()
        queue = review_queue(self.reports, self.history, self.queue_mode())
        self.page = [e for e in queue if e[1]["target_id"] not in self.seen][:6]
        self.evidence, self.checked = {}, {}
        self.gallery.visible = True
        self._draw_gallery()
        if self.page:
            self.busy = True
            self.capture_task = asyncio.ensure_future(self._capture_page())

    def _draw_gallery(self):
        import omni.ui as ui

        with self.gallery.frame:
            with ui.VStack(spacing=5):
                ui.Label("REVIEW ONLY: inspect each image, then explicitly select it. Nothing is selected automatically.", height=25)
                with ui.HStack(height=28):
                    ui.Label("Reviewer:", width=75)
                    ui.StringField(model=self.reviewer, width=190)
                    self.gallery_reason = ui.ComboBox(0, "Choose a batch reason...", *[v[0] for v in REASONS.values()]).model
                self.gallery_message = ui.Label("Capturing..." if self.page else "End of this pass. Restart sampling or change the queue.", word_wrap=True, height=45)
                self.image_frames = {}
                for row_index in range(0, len(self.page), 3):
                    with ui.HStack(spacing=8, height=280):
                        for entry in self.page[row_index:row_index + 3]:
                            target_id = entry[1]["target_id"]
                            with ui.VStack(spacing=3):
                                self.image_frames[target_id] = ui.Frame(height=150)
                                with self.image_frames[target_id]:
                                    ui.Label("Waiting for capture", height=150)
                                ui.Label(self.card_description(entry), word_wrap=True, height=88)
                                with ui.HStack(height=25):
                                    check = ui.SimpleBoolModel(False)
                                    ui.CheckBox(model=check, width=24)
                                    self.checked[target_id] = check
                                    ui.Label("I inspected this", width=110)
                                    ui.Button("Inspect in 3D", clicked_fn=lambda e=entry: self.inspect(e))
                with ui.HStack(height=30):
                    ui.Button("Select eligible captured cards", clicked_fn=self.select_captured)
                    ui.Button("Clear selection", clicked_fn=self.clear_selection)
                    ui.Button("Save selected + next page", clicked_fn=self.save_selected)
                    ui.Button("Skip page (no approval)", clicked_fn=self.next_page)
                    ui.Button("Restart sample", clicked_fn=lambda: self.start_gallery(restart=True))
                ui.Label("Saving explicitly replaces prior decisions for selected targets; originals remain. Unselected cards stay unreviewed. No cutting approval.", word_wrap=True, height=35)

    async def _capture_page(self):
        import omni.kit.app
        import omni.ui as ui
        from omni.kit.viewport.utility import capture_viewport_to_file
        from PIL import Image
        from pxr import UsdGeom

        directory = self.output.parent / "review_gallery" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        application = omni.kit.app.get_app()
        try:
            for index, entry in enumerate(self.page):
                report, target = entry
                key = target["target_id"]
                self.gallery_message.text = f"Capturing {index + 1}/{len(self.page)}: {key}. Please wait before moving the camera."
                try:
                    self._display(entry)
                    for _ in range(12):
                        await application.next_update_async()
                    camera = self.stage.GetPrimAtPath(self.viewport.camera_path)
                    camera_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(camera)
                    path = directory / f"card_{index}.png"
                    capture = capture_viewport_to_file(self.viewport, str(path))
                    await asyncio.wait_for(capture.wait_for_result(), timeout=20)
                    deadline = time.monotonic() + 15
                    while True:
                        try:
                            with Image.open(path) as captured:
                                captured.verify()
                            break
                        except (OSError, SyntaxError):
                            if time.monotonic() > deadline:
                                raise RuntimeError("Thumbnail encoder timed out")
                            await application.next_update_async()
                    self.evidence[key] = {"target_id": key, "manifest_sha256": report["manifest_sha256"],
                                          "component_asset_hashes": fingerprints(report),
                                          "image_path": str(path.resolve()), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "camera_to_world": [list(row) for row in camera_matrix],
                                          "isolated": False, "training_input_allowed": False}
                    with self.image_frames[key]:
                        ui.Image(str(path), fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT, height=150)
                except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
                    with self.image_frames[key]:
                        ui.Label(f"Capture unavailable: {exc}\nCannot batch-approve this card.", word_wrap=True)
            (directory / "gallery.json").write_text(json.dumps({"training_input_allowed": False,
                                                               "cards": self.evidence}, indent=2), encoding="utf-8")
            self.gallery_message.text = f"{len(self.evidence)}/{len(self.page)} images ready. Select ONLY inspected cards, choose a reason, then save."
            print(f"REVIEW_GALLERY_READY {directory} captured={len(self.evidence)}/{len(self.page)}", flush=True)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.gallery_message.text = f"Gallery failed: {exc}. No reviews were saved."
            print(f"REVIEW_GALLERY_FAILED {exc}", flush=True)
        finally:
            self.busy = False
            if not self.closed:
                self.show()

    def select_captured(self):
        if not self.busy:
            reason = self.reason_keys[self.gallery_reason.get_item_value_model().as_int]
            eligible = {t["target_id"] for r, t in self.page
                        if reason != "anatomy_matches" or (r["status"] != "blocked" and t["status"] == "needs_review")}
            for key, model in self.checked.items():
                model.set_value(key in self.evidence and key in eligible)

    def clear_selection(self):
        for model in self.checked.values():
            model.set_value(False)

    def inspect(self, entry):
        if self.busy:
            return
        if entry not in self.entries:
            self.entries.append(entry)
        self.index = self.entries.index(entry)
        self.show()
        self.gallery.visible = False
        self.message.text = "Inspect in 3D, then use Save current reason + next. Gallery decisions were not saved."

    def next_page(self):
        if not self.busy:
            self.seen.update(e[1]["target_id"] for e in self.page)
            self.start_gallery()

    def save_selected(self):
        if self.busy:
            return
        selected = [(r, t["target_id"]) for r, t in self.page if self.checked[t["target_id"]].as_bool]
        key = self.reason_keys[self.gallery_reason.get_item_value_model().as_int]
        try:
            path = record_batch(selected, self.reviewer.as_string, key, self.notes.as_string, self.output, self.evidence,
                                supersedes={target: self.superseded(target) for _, target in selected})
            self.history.refresh()
            self.update_summary()
            self.entries = review_queue(self.reports, self.history, self.queue_mode())
            self.index %= max(1, len(self.entries))
            self.notes.set_value("")
            self.message.text = f"Saved {len(selected)} explicit reviews in {path.name}. No cut approval."
            self.next_page()
        except (OSError, ValueError) as exc:
            self.gallery_message.text = str(exc) + ". No batch was saved."

    def _gallery_visibility(self, visible):
        if not visible and self.busy and self.capture_task:
            self.capture_task.cancel()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.capture_task and not self.capture_task.done():
            self.capture_task.cancel()
        if self.scene:
            self.viewport.set_active_camera(str(self.previous_camera))
            self.scene.close()
            self.scene = None
        self._restore_borrowed()
        self.window.set_visibility_changed_fn(None)
        self.gallery.set_visibility_changed_fn(None)
        self.window.visible = False
        self.gallery.visible = False
