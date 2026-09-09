import json
import numpy as np
from PIL import Image
import pytest

from sim_data.training_evaluate import segment_distance,score
from sim_data.training_export_test import row


def test_interval_distance_uses_segments_not_only_endpoints():
    assert segment_distance([5,2],[[0,0],[10,0]])==2
    assert segment_distance([12,0],[[0,0],[10,0]])==2
    assert segment_distance([0,3],[[0,0],[0,0]])==3
    with pytest.raises(ValueError): segment_distance([float('nan'),0],[[0,0],[10,0]])


def test_abstention_on_localizable_is_counted_as_failure(tmp_path):
    r=row()
    p={'a':dict(status='abstain',cut_point_uv=None,visibility='occluded',next_action='change_viewpoint')}
    s=score([r],p,tmp_path)
    assert s['success_within_5px_including_abstention_failures']==0
    assert s['localized_answer_coverage']==0 and s['median_error_px_conditional_on_answer'] is None


def test_hallucinated_point_on_occlusion_is_false_localization(tmp_path):
    r=row(status='abstain'); p={'a':row()['answer']}
    s=score([r],p,tmp_path)
    assert s['false_localization_rate_on_occluded']==1 and s['status_accuracy']==0


def test_perfect_point_checks_interval_and_visible_target(tmp_path):
    r=row(); r['files']={'label':'label.json','target_mask':'target.png'}
    (tmp_path/'label.json').write_text(json.dumps({'accepted_interval_uv':[[434,204],[444,204]]}))
    mask=np.zeros((408,848),np.uint8); mask[204,434]=255
    Image.fromarray(mask).save(tmp_path/'target.png')
    s=score([r],{'a':r['answer']},tmp_path)
    assert s['success_within_5px_including_abstention_failures']==1
    assert s['projected_interval_hit_rate_2px']==1 and s['visible_target_hit_rate']==1
    json.dumps(s,allow_nan=False)
    with pytest.raises(ValueError): score([r],{},tmp_path)
