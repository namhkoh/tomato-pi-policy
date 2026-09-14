from copy import deepcopy
import pytest
from sim_data.clear_collection_campaign import selected_jobs


def plan():
    return dict(jobs=[dict(job_id='job_001',split='test'),dict(job_id='job_002',split='train'),
                      dict(job_id='job_003',split='validation'),dict(job_id='job_004',split='train')])


def test_explicit_subset_preserves_frozen_source_jobs_and_training_first_order():
    p=plan();before=deepcopy(p)
    assert [j['job_id'] for j in selected_jobs(p)]==['job_002','job_004','job_001','job_003']
    assert [j['job_id'] for j in selected_jobs(p,['job_003','job_004'])]==['job_004','job_003']
    assert p==before


@pytest.mark.parametrize('ids',[[],['missing'],['job_001','job_001'],'job_001'])
def test_invalid_subsets_rejected(ids):
    with pytest.raises(ValueError):selected_jobs(plan(),ids)
