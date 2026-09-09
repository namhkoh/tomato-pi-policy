import hashlib
import json
from pathlib import Path

import pytest

from sim_data.audit import ROOT
from sim_data.capture_pilot import load_drafts, main, review_markdown

DRAFT = ROOT / "data/sim_data/cut_region_drafts/20260907_235517_89790c/draft_labels.json"


@pytest.mark.skipif(not DRAFT.is_file(), reason="Six-target prototype draft required")
def test_real_pilot_selection_does_not_approve_or_change_drafts():
    before = hashlib.sha256(DRAFT.read_bytes()).hexdigest()
    saved,reports,rows = load_drafts(DRAFT)
    assert [r["draft_id"] for r in rows] == ["B03","B05","B06"]
    assert len(reports)==2
    assert not saved["human_review_performed"]
    assert all(not r["training_label_approved"] and not r["cut_approval"] for r in rows)
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest()==before


@pytest.mark.skipif(not DRAFT.is_file(), reason="Six-target prototype draft required")
@pytest.mark.parametrize("kind",["cut","hash","approval","missing","unfinished"])
def test_modified_or_incomplete_drafts_rejected(tmp_path,kind):
    data=json.loads(DRAFT.read_text())
    if kind=="cut": data["labels"][2]["cut_region_proposal"]["nominal"]["point_plant_m"][0]+=.001
    if kind=="hash": data["source_variants_sha256"]="changed"
    if kind=="approval": data["labels"][2]["cut_approval"]=True
    if kind=="missing": data["labels"].pop()
    if kind=="unfinished": data["state"]="render_failed"
    path=tmp_path/"draft.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_drafts(path)


@pytest.mark.skipif(not DRAFT.is_file(), reason="Six-target prototype draft required")
def test_existing_run_is_not_overwritten(tmp_path):
    sentinel=tmp_path/"keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(SystemExit): main(["--drafts",str(DRAFT),"--output",str(tmp_path)])
    assert sentinel.read_text()=="keep"


def test_review_does_not_call_projected_markers_visible_or_approved():
    text=review_markdown({"samples":[{"sample_id":"sample_0001","target_review_id":"B03",
        "nominal_pixel_xy":[200,100],"depth_status":"foreground_occlusion_evidence"}]})
    assert "NOT an approved training dataset" in text
    assert "HIDDEN geometry" in text
    assert "sample_0001/inputs/rgb.png" in text
    assert "sample_0001/review/overlay.png" in text
