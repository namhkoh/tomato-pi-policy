"""Batch review regressions. All human decisions here are temporary fixtures."""

import copy
import hashlib
import json
import unittest
from unittest.mock import Mock, patch

from . import audit_test as fixtures
from .audit import audit_manifest
from .batch import History, attach_cached_geometry, fingerprints, review_queue, target_flags
from .review import record_batch, record_review


class BatchTests(unittest.TestCase):
    setUp = fixtures.AuditTests.setUp
    write = fixtures.AuditTests.write

    def setup_report(self):
        report = self.write()
        report["geometry_audit"] = {"scope": "assembled_translation_and_mesh_aabb_only", "warnings": []}
        return report, report["targets"][0]["target_id"]

    def evidence(self, report, target):
        image = self.root / (target.split("/")[-1] + ".png")
        image.write_bytes(b"test-only image fixture")
        return {target: {"target_id": target, "manifest_sha256": report["manifest_sha256"],
                         "component_asset_hashes": fingerprints(report),
                         "image_path": str(image), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                         "training_input_allowed": False}}

    def test_legacy_history_deduplicates_targets_and_resume_skips_completed(self):
        report, target = self.setup_report()
        for _ in range(2):
            record_review(report, target, "anatomy_confirmed", "fixture", "fixture", self.root / "reviews")
        history = History([report], [self.root / "reviews", self.root / "reviews"])
        self.assertEqual(len(history.rows), 1)
        self.assertEqual(len(history.rows[target]), 2)
        self.assertEqual(review_queue([report], history), [])
        self.assertEqual(len(review_queue([report], history, "All / revisit")), 1)

    def test_legacy_out_of_bounds_requires_scope_not_silent_relabel(self):
        report, target = self.setup_report()
        path = record_review(report, target, "excluded", "fixture", "out of bouds", self.root / "reviews")
        before = path.read_bytes()
        history = History([report], [path.parent])
        self.assertIn("legacy_exclusion_needs_scope", history.issues(target))
        self.assertFalse(history.complete(target))
        self.assertEqual(path.read_bytes(), before)

    def test_workspace_exclusion_is_explicit_and_not_anatomy_approval(self):
        report, target = self.setup_report()
        path = record_review(report, target, "excluded", "fixture", "fixture", self.root / "reviews",
                             reason_code="out_of_workspace")
        row = json.loads(path.read_text())
        self.assertEqual(row["review_scope"], "workspace")
        self.assertFalse(row["cut_approval"])
        self.assertTrue(History([report], [path.parent]).complete(target))

    def test_visibility_unresolved_stays_in_exceptions(self):
        report, target = self.setup_report()
        path = record_review(report, target, "unresolved", "fixture", "fixture", self.root / "reviews", reason_code="out_of_view")
        history = History([report], [path.parent])
        self.assertFalse(history.complete(target))
        self.assertEqual(len(review_queue([report], history, "Exceptions")), 1)

    def test_single_review_retains_view_context_without_changing_anatomy_scope(self):
        report, target = self.setup_report()
        context = {"training_input_allowed": False, "camera_path": "/World/HeadCamera", "resolution": [848, 408]}
        path = record_review(report, target, "unresolved", "fixture", "fixture", self.root / "reviews",
                             reason_code="out_of_view", view_context=context)
        row = json.loads(path.read_text())
        self.assertEqual(row["view_context"], context)
        self.assertEqual(row["review_scope"], "visibility")
        self.assertFalse(row["cut_approval"])

    def test_stale_source_and_unknown_records_do_not_approve_targets(self):
        report, target = self.setup_report()
        path = record_review(report, target, "anatomy_confirmed", "fixture", "fixture", self.root / "reviews")
        report["components"]["Leaf"]["asset_sha256"] = "different"
        (path.parent / "invalid.json").write_text("{")
        history = History([report], [path.parent])
        self.assertFalse(history.complete(target))
        self.assertEqual(len(history.ignored), 2)

    def test_batch_records_only_selected_targets_and_history_loads_it(self):
        report, target = self.setup_report()
        path = record_batch([(report, target)], "fixture", "anatomy_matches", "", self.root / "reviews", self.evidence(report, target))
        batch = json.loads(path.read_text())
        self.assertTrue(batch["explicit_selection"])
        self.assertEqual(len(batch["records"]), 1)
        self.assertFalse(batch["records"][0]["cut_approval"])
        self.assertTrue(batch["records"][0]["notes"])
        self.assertTrue(History([report], [path.parent]).complete(target))

    def test_batch_rejects_no_selection_duplicate_and_no_identity(self):
        report, target = self.setup_report()
        evidence = self.evidence(report, target)
        for entries, reviewer in (([], "fixture"), ([(report, target)] * 2, "fixture"), ([(report, target)], "")):
            with self.subTest(entries=len(entries), reviewer=reviewer), self.assertRaises(ValueError):
                record_batch(entries, reviewer, "anatomy_matches", "", self.root / "reviews", evidence)
        self.assertFalse((self.root / "reviews").exists())

    def test_uncaptured_card_and_changed_image_reject_without_save(self):
        report, target = self.setup_report()
        with self.assertRaises(ValueError):
            record_batch([(report, target)], "fixture", "anatomy_matches", "", self.root / "reviews", {})
        evidence = self.evidence(report, target)
        (self.root / "Petiole.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Thumbnail changed"):
            record_batch([(report, target)], "fixture", "anatomy_matches", "", self.root / "reviews", evidence)
        self.assertFalse((self.root / "reviews").exists())

    def test_excluded_member_makes_entire_batch_fail(self):
        report, target = self.setup_report()
        bad = copy.deepcopy(report)
        bad["targets"][0]["target_id"] = "test_plant/Excluded"
        bad["targets"][0]["status"] = "excluded"
        evidence = self.evidence(report, target)
        evidence.update(self.evidence(bad, "test_plant/Excluded"))
        with self.assertRaisesRegex(ValueError, "excluded"):
            record_batch([(report, target), (bad, "test_plant/Excluded")], "fixture", "anatomy_matches", "", self.root / "reviews", evidence)
        self.assertFalse((self.root / "reviews").exists())

    def test_capture_source_fingerprints_are_checked_after_reaudit(self):
        report, target = self.setup_report()
        evidence = self.evidence(report, target)
        (self.plant / "Leaf.usda").write_text("changed")
        fresh = audit_manifest(self.path)
        with self.assertRaisesRegex(ValueError, "Thumbnail does not match"):
            record_batch([(fresh, target)], "fixture", "anatomy_matches", "", self.root / "reviews", evidence)

    def test_unknown_reason_or_reason_decision_mismatch_rejects(self):
        report, target = self.setup_report()
        with self.assertRaises(ValueError):
            record_review(report, target, "anatomy_confirmed", "fixture", "fixture", self.root / "reviews", reason_code="old_stub")
        with self.assertRaises(ValueError):
            record_batch([(report, target)], "fixture", None, "", self.root / "reviews", self.evidence(report, target))

    def test_source_package_output_is_rejected_for_batch(self):
        report, target = self.setup_report()
        with self.assertRaisesRegex(ValueError, "source package"):
            record_batch([(report, target)], "fixture", "anatomy_matches", "", self.root / "package/reviews", self.evidence(report, target))
        self.assertFalse((self.root / "package/reviews").exists())

    def test_queue_is_reproducible_and_balanced_across_plants(self):
        report, _ = self.setup_report()
        reports = []
        for plant in ("c", "a", "b"):
            clone = copy.deepcopy(report)
            clone["plant_id"] = plant
            clone["targets"] = [dict(clone["targets"][0], target_id=f"{plant}/target_{i}") for i in range(12)]
            reports.append(clone)
        history = History(reports, [])
        first = review_queue(reports, history)
        second = review_queue(list(reversed(reports)), history)
        self.assertEqual([e[1]["target_id"] for e in first], [e[1]["target_id"] for e in second])
        self.assertEqual({e[0]["plant_id"] for e in first[:3]}, {"a", "b", "c"})

    def test_warning_propagation_and_missing_geometry_never_claim_pass(self):
        report, _ = self.setup_report()
        target = report["targets"][0]
        report["geometry_audit"]["warnings"] = [{"component_id": "Leaf", "code": "bad_leaf"}]
        self.assertIn("bad_leaf", target_flags(report, target))
        del report["geometry_audit"]
        self.assertIn("geometry_not_checked", target_flags(report, target))

    def test_geometry_cache_requires_matching_component_hashes(self):
        report, _ = self.setup_report()
        directory = self.root / "audits/20260101"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("{}")
        (directory / "test_plant.json").write_text(json.dumps(report))
        fresh = audit_manifest(self.path)
        attach_cached_geometry([fresh], directory.parent)
        self.assertIn("geometry_audit", fresh)
        del fresh["geometry_audit"]
        fresh["components"]["Leaf"]["asset_sha256"] = "changed"
        attach_cached_geometry([fresh], directory.parent)
        self.assertNotIn("geometry_audit", fresh)

    def test_conflicting_decisions_require_explicit_supersession(self):
        report, target = self.setup_report()
        output = self.root / "reviews"
        first = record_review(report, target, "anatomy_confirmed", "fixture", "fixture", output, reason_code="anatomy_matches")
        second = record_review(report, target, "excluded", "fixture", "fixture", output, reason_code="wrong_anatomy")
        original = {path: path.read_bytes() for path in (first, second)}
        history = History([report], [output])
        self.assertIn("conflicting_decisions_require_explicit_rereview", history.issues(target))
        self.assertFalse(history.complete(target))
        record_review(report, target, "anatomy_confirmed", "fixture", "explicitly rereviewed", output,
                      reason_code="anatomy_matches", supersedes=[row["review_id"] for row in history.rows[target]])
        self.assertTrue(History([report], [output]).complete(target))
        self.assertEqual(original, {path: path.read_bytes() for path in (first, second)})

    def test_corrupt_v2_scope_and_invalid_timestamp_are_not_loaded(self):
        report, target = self.setup_report()
        path = record_review(report, target, "anatomy_confirmed", "fixture", "fixture", self.root / "reviews", reason_code="anatomy_matches")
        row = json.loads(path.read_text())
        for changes in ({"review_scope": "workspace"}, {"timestamp_utc": "not a timestamp"}, {"supersedes_review_ids": [{}]}):
            path.write_text(json.dumps(dict(row, **changes)))
            history = History([report], [path.parent])
            self.assertFalse(history.complete(target))
            self.assertEqual(len(history.ignored), 1)

    def test_saved_confirmation_cannot_override_new_target_exclusion(self):
        report, target = self.setup_report()
        path = record_review(report, target, "anatomy_confirmed", "fixture", "fixture", self.root / "reviews")
        report["targets"][0]["status"] = "excluded"
        history = History([report], [path.parent])
        self.assertFalse(history.complete(target))
        self.assertEqual(len(history.ignored), 1)

    def panel_fixture(self, report):
        from .gallery import BatchReviewPanel

        panel = BatchReviewPanel.__new__(BatchReviewPanel)
        panel.busy = panel.closed = False
        panel.index = 0
        panel.reports = [report]
        panel.history = History([report], [])
        panel.output = self.root / 'reviews'
        panel.reviewer = Mock(as_string='fixture')
        panel.notes = Mock(as_string='')
        panel.message = Mock()
        panel.gallery_message = Mock()
        panel.reason_keys = [None, 'anatomy_matches']
        panel.gallery_reason = Mock()
        panel.gallery_reason.get_item_value_model.return_value.as_int = 1
        panel.queue_mode = Mock(return_value='Mixed sample')
        panel.update_summary = Mock()
        panel.show = Mock()
        panel.next_page = Mock()
        return panel

    def test_gallery_callback_saves_only_checked_cards_and_refreshes_navigation(self):
        report, target = self.setup_report()
        second = copy.deepcopy(report['targets'][0])
        second['target_id'] = 'test_plant/second'
        report['targets'].append(second)
        panel = self.panel_fixture(report)
        panel.page = panel.entries = [(report, t) for t in report['targets']]
        panel.checked = {target: Mock(as_bool=True), second['target_id']: Mock(as_bool=False)}
        panel.evidence = self.evidence(report, target)
        panel.history.directories = [panel.output]
        panel.save_selected()
        self.assertTrue(panel.history.complete(target))
        self.assertFalse(panel.history.complete(second['target_id']))
        self.assertEqual([t['target_id'] for _, t in panel.entries], [second['target_id']])
        panel.next_page.assert_called_once()

    def test_single_save_advances_and_does_not_repeat_completed_target(self):
        report, target = self.setup_report()
        second = copy.deepcopy(report['targets'][0])
        second['target_id'] = 'test_plant/second'
        report['targets'].append(second)
        panel = self.panel_fixture(report)
        panel.entries = [(report, t) for t in report['targets']]
        panel.history.directories = [panel.output]
        panel.stage, panel.viewport, panel.scene = Mock(), Mock(), Mock()
        with patch('sim_data.gallery.view_context', return_value={'training_input_allowed': False}):
            panel.save_reason('anatomy_matches')
        self.assertEqual([t['target_id'] for _, t in panel.entries], [second['target_id']])
        self.assertEqual(panel.index, 0)
        panel.show.assert_called_once()

    def test_selection_button_is_explicit_and_omits_unconfirmable_cards(self):
        report, target = self.setup_report()
        stub = copy.deepcopy(report['targets'][0])
        stub.update(target_id='test_plant/stub', status='excluded')
        panel = self.panel_fixture(report)
        panel.page = [(report, report['targets'][0]), (report, stub)]
        panel.checked = {key: Mock(as_bool=False) for key in (target, stub['target_id'])}
        panel.evidence = {target: {}, stub['target_id']: {}}
        panel.select_captured()
        panel.checked[target].set_value.assert_called_once_with(True)
        panel.checked[stub['target_id']].set_value.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main()
