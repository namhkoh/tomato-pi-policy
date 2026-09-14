import copy
import pytest
from sim_data.clear_geometry_probe import compare_native,memory_reserve


def test_probe_requires_matching_native_geometry_decisions():
    row=dict(target_review_id='target',candidate_id='ground_0001',state='rejected_projected_sampling',
        predicted_diameter_px=7.,predicted_interval_px=11.,base_xy_m=[.4,.1])
    reference=dict(decisions=[copy.deepcopy(row)])
    r=compare_native([row],reference)
    assert r['geometry_decisions_match'] and not r['rendered_visibility_checked']
    for field,value in [('state','geometry_screen_passed_visibility_unknown'),('predicted_diameter_px',8.),
                        ('candidate_id','other'),('base_xy_m',[.5,.1])]:
        with pytest.raises(ValueError):compare_native([dict(row,**{field:value})],reference)
    missing=copy.deepcopy(row);missing.pop('predicted_interval_px')
    with pytest.raises(ValueError):compare_native([missing],reference)


def test_cpu_probe_does_not_ignore_its_own_memory_reserve(monkeypatch):
    from sim_physics import host_memory
    m=dict(read_succeeded=True,commit_limit_bytes=20*2**30,committed_bytes=10*2**30,physical_available_bytes=5*2**30)
    monkeypatch.setattr(host_memory,'memory_snapshot',lambda:m)
    assert memory_reserve()==m
    m['physical_available_bytes']=3*2**30
    with pytest.raises(ValueError,match='reserve'):memory_reserve()
    m['physical_available_bytes']=5*2**30;m['committed_bytes']=13*2**30
    with pytest.raises(ValueError,match='reserve'):memory_reserve()
