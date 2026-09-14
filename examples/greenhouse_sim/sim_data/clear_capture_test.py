import pytest
from sim_data.training_plan import configuration,view_specs


def test_explicit_clear_capture_requires_bounded_real_robot_pose_sampling():
    assert 'clear_capture' not in configuration()
    with pytest.raises(ValueError):configuration(clear_capture=True)
    with pytest.raises(ValueError):configuration(views=13,vary_torso=True,clear_capture=True)
    assert configuration(views=3,vary_torso=True,clear_capture=True)['clear_capture']=='robot_head_close_diffuse_v1'


def test_closer_specs_preserve_robot_side_yaw_and_distinct_windows():
    specs=view_specs(.8,0,'target',2,vary_torso=True,clear_capture=True)
    assert len(specs)==48
    assert specs==view_specs(.8,0,'target',2,vary_torso=True,clear_capture=True)
    for s in specs:
        assert .30<=s['root_x_m']<=.55 and 150<=s['root_yaw_degrees']<=210
        assert abs(s['y_offset_m'])<=.20 and 0<=s['torso_bend_degrees']<=45
    second=view_specs(.8,0,'target',2,vary_torso=True,clear_capture=True,view_offset=2)
    assert not {s['candidate_id'] for s in specs}&{s['candidate_id'] for s in second}


def test_memory_failure_does_not_spawn_native_worker(tmp_path,monkeypatch):
    from sim_data import collection_run as runner
    from sim_physics import host_memory
    p=tmp_path/'plan.json';p.write_text('{}')
    plan=dict(package=str(tmp_path/'source'),configuration=dict(clear_capture='robot_head_close_diffuse_v1'),
              jobs=[dict(job_id='job_001',plant_family='train')])
    monkeypatch.setattr(runner,'load_plan',lambda p:(plan,[]))
    monkeypatch.setattr(host_memory,'preflight',lambda:dict(allowed=False,reasons=['unit reserve']))
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:pytest.fail('Must not launch'))
    with pytest.raises(ValueError,match='memory'):runner.run_jobs(p,tmp_path/'out')


def test_serial_campaign_skips_only_verified_empty_viewpoint_search(tmp_path):
    from sim_data.clear_collection_campaign import outcome
    import json
    folder=tmp_path/'job_001'/'capture';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps(dict(state='blocked_no_screened_viewpoints',samples=[],source_assets_unchanged=True)))
    job=dict(job_id='job_001',returncode=3,timed_out=False,sample_count=0,capture_state='blocked_no_screened_viewpoints')
    result=dict(state='stopped_worker_nonzero_exit',jobs=[job])
    assert outcome(result,tmp_path)=='no_clear_screened_view'
    job['returncode']=1;assert outcome(result,tmp_path)=='failure'
    job['returncode']=3;job['timed_out']=True;assert outcome(result,tmp_path)=='failure'
    assert outcome(dict(state='complete_bounded_batch_pending_visual_review'),tmp_path)=='audited'


def test_opposite_aisle_is_explicit_and_preserves_original_view_sequence():
    import hashlib,json
    with pytest.raises(ValueError):configuration(opposite_aisle=True)
    c=configuration(views=12,vary_torso=True,clear_capture=True,opposite_aisle=True)
    assert c['opposite_aisle'] is True
    legacy=[view_specs(.8,0,t,12,vary_torso=True,clear_capture=True,view_offset=o)
            for t in ('a','b','N_seed23_full_SubStem_41') for o in (0,12)]
    assert hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()=='a2203baa63a78da923a690b660cd4a174d1591c07f05a21a2f08a1b85d2f4060'
    opposite=view_specs(.8,0,'a',12,vary_torso=True,clear_capture=True,opposite_aisle=True)
    for old,new in zip(legacy[0],opposite):
        assert new['root_x_m']==-old['root_x_m'] and -.55<=new['root_x_m']<=-.30
        assert -30<=new['root_yaw_degrees']<=30 and new['opposite_aisle'] is True
        assert new['torso_bend_degrees']==old['torso_bend_degrees']
        assert new['desired_pixel_xy']==old['desired_pixel_xy']
