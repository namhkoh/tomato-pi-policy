import numpy as np
from PIL import Image

from sim_data.training_capture_compare import pixel_evidence


def sample(root,rgb_value=100):
    (root/'inputs').mkdir(parents=True)
    (root/'supervision').mkdir()
    Image.fromarray(np.full((408,848,3),rgb_value,np.uint8)).save(root/'inputs/rgb.png')
    Image.fromarray(np.full((408,848),255,np.uint8)).save(root/'inputs/depth_valid.png')
    Image.fromarray(np.full((408,848),255,np.uint8)).save(root/'supervision/target_visible.png')
    np.save(root/'inputs/depth_m.npy',np.full((408,848),1.,np.float32))


def test_shading_variation_is_reported_separately_from_native_geometry(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    sample(a,100); sample(b,112)
    result=pixel_evidence(a,b,[424,204])
    assert result['target_visible_mask_equal'] and result['native_depth_equal_0_2mm']
    assert result['target_mask_IoU']==1.
    assert result['RGB_mean_absolute_difference_0_255']==12.
    assert result['target_ROI_RGB_mean_absolute_difference_0_255']==12.


def test_actual_native_geometry_changes_are_not_hidden(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    sample(a); sample(b)
    mask=np.full((408,848),255,np.uint8); mask[204,424]=0
    Image.fromarray(mask).save(b/'supervision/target_visible.png')
    depth=np.full((408,848),1.,np.float32); depth[204,424]=.9
    np.save(b/'inputs/depth_m.npy',depth)
    result=pixel_evidence(a,b,[424,204])
    assert not result['target_visible_mask_equal'] and not result['native_depth_equal_0_2mm']
    assert result['target_mask_IoU']<1.
