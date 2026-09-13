import json
import pytest
from .station_reference_test import fixture
from .cut_priority import from_arguments


def setup_reference(tmp_path):
    args,_,report=fixture(tmp_path)
    args.cut_style='downward';args.knife_edge_mode='source_crossbar_edge_v1'
    report['configuration']['knife_edge_mode']=args.knife_edge_mode
    report['robot']['cut_plan']=dict(plane_tilt_degrees=15.,normal_sign=-1,wing_m=0.)
    args.station_reference_report.write_text(json.dumps(report))
    return args,report


def test_reference_seed_keeps_its_frame_family_without_motion_authority(tmp_path):
    a,_=setup_reference(tmp_path);r=from_arguments(a)
    assert r['tilt']==15. and r['normal_sign']==-1 and r['wing_m']==0.
    assert r['station_reference_frame_family_only'] and r['order_only']
    assert not r['motion_authorized'] and not r['prior_pose_or_path_replayed']
    assert a.cut_priority_report is None  # Do not mutate/pretend explicit input.


def test_explicit_priority_overrides_implicit_reference_family(tmp_path):
    a,r=setup_reference(tmp_path);r['robot']['cut_plan']['plane_tilt_degrees']=-10.
    p=tmp_path/'explicit.json';p.write_text(json.dumps(r));a.cut_priority_report=p
    result=from_arguments(a)
    assert result['tilt']==-10. and 'station_reference_frame_family_only' not in result


@pytest.mark.parametrize('mutation',['gate','task','frame','mode'])
def test_reference_does_not_bypass_evidence_task_family_or_zero_motion_checks(tmp_path,mutation):
    a,r=setup_reference(tmp_path)
    if mutation=='gate':r['gates']['full_forward_cut_stroke_verified']=False
    if mutation=='task':r['configuration']['target']='SubStem_999'
    if mutation=='frame':r['robot']['cut_plan']['plane_tilt_degrees']=90.
    if mutation=='mode':a.native_startup_station_search=False
    a.station_reference_report.write_text(json.dumps(r))
    with pytest.raises(ValueError):from_arguments(a)


def test_default_without_reference_or_priority_does_not_load_any_report(tmp_path):
    a,_=setup_reference(tmp_path);a.station_reference_report=None
    assert from_arguments(a) is None
