"""Pure campaign scheduling tests; no subprocess, generation or native launch."""
from copy import deepcopy
import pytest
from .campaign import planned_jobs


def inputs():
    jobs=[]
    for i,family in enumerate(('seedA','seedB')):
        anchor=dict(attempt_id=f'{family}_primary',source_family=family,split='train')
        fallback=dict(anchor,attempt_id=f'{family}_fallback')
        jobs.append(dict(source_family=family,job_id=f'donor_{i:03d}',split='train',
                         anchor=anchor,fallback_anchors=[fallback]))
    return dict(jobs=jobs),dict(seedB='seedB_fallback',seedA='seedA_primary')


def test_balanced_order_and_nonoverlapping_retry_seeds():
    s,q=inputs();before=deepcopy((s,q))
    jobs=planned_jobs(s,q,rounds=4,seed_base=4000000)
    assert [j['source_family'] for j in jobs]==['seedA','seedB']*4
    assert [j['round_index'] for j in jobs]==[1,1,2,2,3,3,4,4]
    assert [j['index'] for j in jobs]==list(range(1,9))
    assert jobs[1]['attempt_id']=='seedB_fallback'
    assert len({j['seed']+retry*37 for j in jobs for retry in range(3)})==24
    assert (s,q)==before


@pytest.mark.parametrize('rounds,seed',[(0,1),(101,1),(True,1),(1,-1),(1,2**32),(1,2**32-1)])
def test_bounds(rounds,seed):
    s,q=inputs()
    with pytest.raises(ValueError):planned_jobs(s,q,rounds=rounds,seed_base=seed)


@pytest.mark.parametrize('damage',['empty','unknown','bad_anchor','heldout_job','heldout_anchor','wrong_family','duplicate_family'])
def test_no_unqualified_or_heldout_donors(damage):
    s,q=inputs()
    if damage=='empty':q={}
    elif damage=='unknown':q['new_donor']='whatever'
    elif damage=='bad_anchor':q['seedA']='unknown'
    elif damage=='heldout_job':s['jobs'][0]['split']='validation'
    elif damage=='heldout_anchor':s['jobs'][0]['anchor']['split']='test'
    elif damage=='wrong_family':s['jobs'][0]['anchor']['source_family']='seedB'
    else:s['jobs'].append(deepcopy(s['jobs'][0]))
    with pytest.raises(ValueError):planned_jobs(s,q,rounds=1,seed_base=1000)


def test_scheduled_but_unqualified_donor_is_not_launched():
    s,q=inputs();q.pop('seedB')
    assert {j['source_family'] for j in planned_jobs(s,q,rounds=3,seed_base=1000)}=={'seedA'}
