import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from sim_data.depth_preview import COLOURS, colour_depth, generate, labelled_heatmap


def fixture_run(tmp_path):
    run=tmp_path/"run"
    directory=run/"sample_0001"
    inputs=directory/"inputs"
    inputs.mkdir(parents=True)
    depth=np.full((408,848),.6,np.float32)
    depth[0,0]=np.inf
    depth[0,1]=3.
    np.save(inputs/"depth_m.npy",depth,allow_pickle=False)
    Image.fromarray(np.isfinite(depth).astype(np.uint8)*255).save(inputs/"depth_valid.png")
    Image.fromarray(np.full((408,848,3),100,np.uint8)).save(inputs/"rgb.png")
    metadata={"calibration":{"resolution":[848,408],"clipping_range_m":[.04,10.],
        "depth_convention":"optical_axis_z_metres_not_ray_range"},
        "files":{p.relative_to(directory).as_posix():{"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs.iterdir()}}
    (directory/"sample.json").write_text(json.dumps(metadata))
    (run/"manifest.json").write_text(json.dumps({"state":"pilot_ready_for_review",
        "samples":[{"sample_id":"sample_0001","target_review_id":"B03"}]}))
    (run/"review.md").write_text("Original review")
    return run,depth


def test_direct_colour_mapping_preserves_depth_invalids_and_common_metric_scale():
    depth=np.array([[.04,1.,2.,4.,np.inf,np.nan,0.]],np.float32)
    before=depth.copy()
    rgb=colour_depth(depth,np.ones(depth.shape,bool),.04,2.)
    np.testing.assert_array_equal(depth,before)
    np.testing.assert_array_equal(rgb[0,0],COLOURS[0])
    np.testing.assert_array_equal(rgb[0,2],COLOURS[-1])
    np.testing.assert_array_equal(rgb[0,3],COLOURS[-1])
    assert np.all(rgb[0,4:]==rgb[0,4:,0,None])  # Invalid checkerboard, not palette endpoint.
    second=colour_depth(np.array([[1.,.2]],np.float32),np.ones((1,2),bool),.04,2.)
    np.testing.assert_array_equal(rgb[0,1],second[0,0])  # No per-image min/max normalization.


@pytest.mark.parametrize("near,far",[(2,1),(0,0),(-1,2),(np.nan,2),(0,np.inf)])
def test_invalid_display_ranges_rejected(near,far):
    with pytest.raises(ValueError): colour_depth(np.ones((2,2),np.float32),np.ones((2,2),bool),near,far)


def test_mask_and_dtype_are_not_silently_reinterpreted():
    with pytest.raises(ValueError): colour_depth(np.ones((2,2),np.uint16),np.ones((2,2),bool),.04,2)
    with pytest.raises(ValueError): colour_depth(np.ones((2,2),np.float32),np.ones((2,2),np.uint8),.04,2)


def test_legend_does_not_crop_resize_or_cover_depth_pixels():
    depth=np.full((408,848),.6,np.float32)
    valid=np.ones(depth.shape,bool)
    image=labelled_heatmap(depth,valid,.04,2.,"native depth")
    assert image.size==(848,526)
    np.testing.assert_array_equal(np.asarray(image)[:408],colour_depth(depth,valid,.04,2.))


def test_complete_heatmap_export_does_not_write_original_observations_or_labels(tmp_path):
    run,depth=fixture_run(tmp_path)
    originals={p:p.read_bytes() for p in run.rglob("*") if p.is_file()}
    output=generate(run)
    assert all(p.read_bytes()==content for p,content in originals.items())
    result=json.loads((output/"manifest.json").read_text())
    assert result["state"]=="complete_review_only"
    assert result["source_observations_and_metadata_unchanged"]
    assert not result["depth_estimated_from_rgb"] and not result["depth_reconstructed_from_geometry"]
    assert result["samples"][0]["valid_pixels_above_near_display_range"]==1
    assert "SAME linear" in (output/"review.md").read_text()
    for name,digest in result["samples"][0]["files"].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest
    with pytest.raises(FileExistsError): generate(run)


def test_tampered_depth_is_refused_before_any_output_is_created(tmp_path):
    run,_=fixture_run(tmp_path)
    np.save(run/"sample_0001/inputs/depth_m.npy",np.ones((408,848),np.float32),allow_pickle=False)
    with pytest.raises(ValueError,match="hash mismatch"): generate(run)
    assert not (run/"depth_heatmaps").exists()


def test_failed_capture_is_not_promoted_by_visualization(tmp_path):
    run,_=fixture_run(tmp_path)
    (run/"manifest.json").write_text(json.dumps({"state":"failed_do_not_train","samples":[]}))
    with pytest.raises(ValueError,match="completed capture"): generate(run)
    assert not (run/"depth_heatmaps").exists()
