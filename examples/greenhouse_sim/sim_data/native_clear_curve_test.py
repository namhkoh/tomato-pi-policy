"""Curved annotation tests use fabricated unit buffers, never dataset examples."""
import numpy as np
import pytest
from sim_data.native_clear_labels import interval_arc_samples,derive
from sim_data.native_clear_contract_test import fixture
from sim_data.cut_regions import propose_cut_region,load_rule
from sim_data.capture_contract import project,transform_points


def test_straight_interval_retains_exact_legacy_samples():
    assert interval_arc_samples([0,.2])==list(np.linspace(.01,.02,11))


def test_knots_are_retained_without_shortcuts_or_oversized_steps():
    knots=np.r_[0,np.arange(.0015,.2,.0015),.2]
    arcs=np.asarray(interval_arc_samples(knots))
    assert len(arcs)==15 and arcs[0]==.01 and arcs[-1]==.02
    assert np.all(np.diff(arcs)>0) and np.max(np.diff(arcs))<=.001+1e-15
    assert all(s in arcs for s in knots if .01<s<.02)


@pytest.mark.parametrize('knots',[[0,.01,.01,.2],[0,.2,.1],[0,float('nan'),.2],[.001,.2]])
def test_invalid_anatomy_is_not_silently_repaired(knots):
    with pytest.raises(ValueError):interval_arc_samples(knots)


def curved_fixture():
    args=list(fixture());meta,report=args[:2]
    component=report['components']['Petiole']
    component['capsules_local_m']=[[[0,0,0,.002],[.0105,0,0,.002],
        [.012,.00015,0,.002],[.0135,0,0,.002],[.2,0,0,.002]]]
    proposal=propose_cut_region(component,report['components']['Main'],load_rule())
    # Independently mirror native capture_scene.target_world_geometry, not the
    # annotation helper under test. The bend lies inside the cut interval.
    points=[]
    samples=proposal['accepted_centerline_interval']['samples']
    for a,b in zip(samples,samples[1:]):
        steps=max(1,int(np.ceil((b['arc_distance_m']-a['arc_distance_m'])/.001)))
        for t in np.linspace(0,1,steps,endpoint=False):
            points.append(np.asarray(a['point_plant_m'])*(1-t)+np.asarray(b['point_plant_m'])*t)
    points.append(samples[-1]['point_plant_m'])
    sup=meta['supervision']
    sup['nominal_world_m']=proposal['nominal']['point_plant_m']
    sup['nominal_projected']=project([sup['nominal_world_m']],meta['calibration'])[0]
    sup['interval_world_m']=transform_points(points,sup['plant_to_world_usd_row_vectors']).tolist()
    return args


def test_curved_native_geometry_qualifies_without_fixed_count():
    args=curved_fixture();result=derive(*args)
    assert result['eligible']
    assert len(result['accepted_interval_uv'])==len(args[0]['supervision']['interval_world_m'])>11
    assert result['native_depth_reconstructed'] is False and result['training_approved'] is False


@pytest.mark.parametrize('fault',['missing_knot','shortcut','stale','occluded'])
def test_curved_mismatch_and_occlusion_are_not_forgiven(fault):
    args=curved_fixture();saved=args[0]['supervision']['interval_world_m']
    if fault=='missing_knot':saved.pop(2)
    elif fault=='shortcut':args[0]['supervision']['interval_world_m']=np.linspace(saved[0],saved[-1],len(saved)).tolist()
    elif fault=='stale':saved[3][0]+=.001
    else:
        p=project([saved[3]],args[0]['calibration'])[0]['pixel_xy'];x,y=np.floor(p).astype(int)
        args[5][y,x]=0
        result=derive(*args)
        assert not result['eligible'] and result['reason']=='cut_interval_not_fully_native_visible'
        return
    with pytest.raises(ValueError,match='Captured cut geometry'):derive(*args)
