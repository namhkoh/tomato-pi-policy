from pathlib import Path
import pytest
from .source_station_trial import configure
from .ground_truth_trial import arguments,main
from .benchmark import parser


def test_new_source_does_not_reuse_original_target_specific_station_or_ik():
    old=arguments('new','right_only','cut_action');original=list(old)
    options=configure(old,'seed19_full/SubStem_41');a=parser().parse_args(options)
    assert old==original and a.plant=='seed19_full' and a.target=='SubStem_41'
    assert a.station_pose is None and a.left_ik_seed_degrees is None
    assert a.right_ready_degrees==[0.,-5.,0.,-120.,0.,70.,0.]
    assert a.physics_hz==480 and a.native_startup_clearance and a.native_static_clearance
    assert a.cut_arc_m==.02 and a.grasp_arc_m==.08
    assert a.native_drives_after_cut


@pytest.mark.parametrize('source',['../SubStem_41','seed19_full/../file','seed19_full/Leaf_001',
    'seed19_full/SubStem_41/extra','seed19_full\\SubStem_41','',None])
def test_bad_source_identity_cannot_escape_the_package(source):
    with pytest.raises(ValueError):configure([],source)


def test_current_cli_replaces_all_right_ready_occurrences_and_retains_full_guard_validation(tmp_path,monkeypatch):
    class Validated(Exception):pass
    def stop(*a,**k):raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):
        main(['--output',str(tmp_path/'new'),'--mode','right_only','--milestone','cut_action',
            '--process-zone-trial','--source-station-trial','seed19_full/SubStem_41',
            '--park-left-ready','--greenhouse-trial','--screen-station','--cut-station-orbit'])


def test_new_source_requires_explicit_correct_edge_profile(tmp_path):
    with pytest.raises(SystemExit):
        main(['--output',str(tmp_path/'new'),'--mode','right_only',
            '--source-station-trial','seed19_full/SubStem_41'])
    assert not (tmp_path/'new').exists()


def test_recipe_configuration_cannot_import_usd_before_simulation_app():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',
        "import sys; from sim_physics.source_station_trial import configure; "
        "configure([], 'seed19_full/SubStem_41'); "
        "assert not any(n=='pxr' or n.startswith('pxr.') or n=='omni' or n.startswith('omni.') for n in sys.modules)"],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_scene_reexports_same_ready_constants_without_duplicate_values():
    from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES as pure
    from greenhouse_sim.robot_scene import SDK_READY_POSE_DEGREES as scene
    assert scene is pure


def test_explicit_other_side_is_only_a_bimanual_pose_proposal(tmp_path,monkeypatch):
    from . import benchmark
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda options:captured.append(parser().parse_args(options)))
    main(['--output',str(tmp_path/'new'),'--mode','bimanual','--process-zone-trial',
          '--source-station-trial','seed19_full/SubStem_41','--approach-vector','-1','1','0'])
    a=captured[0]
    assert a.approach_vector==[-1.,1.,0.]
    assert a.grasp_arc_m==.08 and a.grasp_depth_m==.125 and a.native_startup_clearance
    assert a.native_static_clearance and a.force_closure and a.require_retention_screen
    with pytest.raises(SystemExit):
        main(['--output',str(tmp_path/'new'),'--mode','right_only','--process-zone-trial',
              '--approach-vector','-1','1','0'])


def test_equivalent_grasp_roll_and_pitch_do_not_change_physical_guards(tmp_path,monkeypatch):
    from . import benchmark
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda options:captured.append(parser().parse_args(options)))
    main(['--output',str(tmp_path/'new'),'--mode','bimanual','--process-zone-trial',
          '--source-station-trial','seed19_full/SubStem_41','--grasp-roll','180','--grasp-pitch','-20'])
    a=captured[0]
    assert a.grasp_roll==180 and a.grasp_pitch==-20
    assert a.grasp_arc_m==.08 and a.native_static_clearance and a.native_startup_clearance
    assert a.pregrasp_half_aperture_m==.008 and a.finger_actuator_limit_n==.8


@pytest.mark.parametrize('pitch',['nan','inf','60.1','-60.1'])
def test_invalid_pitch_reaches_existing_benchmark_rejection_without_launch(tmp_path,pitch):
    with pytest.raises(ValueError,match='pitch'):
        main(['--output',str(tmp_path/'new'),'--mode','bimanual','--process-zone-trial',
            '--grasp-pitch',pitch])
    assert not (tmp_path/'new').exists()


def test_explicit_neutral_torso_keeps_all_source_and_native_checks(tmp_path,monkeypatch):
    from . import benchmark
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda options:captured.append(parser().parse_args(options)))
    main(['--output',str(tmp_path/'new'),'--mode','bimanual','--process-zone-trial',
          '--source-station-trial','seed19_full/SubStem_41','--torso-degrees',*['0']*6])
    a=captured[0]
    assert a.torso_degrees==[0.]*6 and a.physics_hz==480
    assert a.native_static_clearance and a.native_startup_clearance and a.force_closure


def test_explicit_new_source_seeds_override_old_recipe_only_with_fresh_guards(tmp_path,monkeypatch):
    from . import benchmark
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda options:captured.append(parser().parse_args(options)))
    main(['--output',str(tmp_path/'new'),'--mode','bimanual','--process-zone-trial',
          '--source-station-trial','seed19_full/SubStem_41','--station-pose','.4','.5','90',
          '--right-ready-degrees',*['1']*7,'--left-ik-seed-degrees',*['2']*7])
    a=captured[0]
    assert a.station_pose==[.4,.5,90] and a.right_ready_degrees==[1.]*7 and a.left_ik_seed_degrees==[2.]*7
    assert a.native_static_clearance and a.native_startup_clearance and a.require_retention_screen


def test_conflicting_pose_authorities_are_rejected_before_launch(tmp_path):
    with pytest.raises(SystemExit):
        main(['--output',str(tmp_path/'unused'),'--mode','bimanual','--process-zone-trial',
              '--station-pose','.4','.5','90','--station-proposal-report','prior.json'])
    assert not (tmp_path/'unused').exists()
