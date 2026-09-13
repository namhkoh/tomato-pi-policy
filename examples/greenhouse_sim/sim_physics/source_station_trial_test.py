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
