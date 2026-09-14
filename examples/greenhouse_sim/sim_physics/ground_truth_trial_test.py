from pathlib import Path
import pytest
from .ground_truth_trial import arguments


@pytest.mark.parametrize('mode',['bimanual','right_only'])
@pytest.mark.parametrize('milestone',['full_sequence','cut_action'])
@pytest.mark.parametrize('capture',[False,True])
def test_durable_profile_reaches_validation_boundary_without_launch(tmp_path,monkeypatch,mode,milestone,capture):
    from .benchmark import main,parser
    out=tmp_path/'new_trial';argv=arguments(out,mode,milestone,capture)
    a=parser().parse_args(argv)
    assert a.plant=='seed101_full' and a.target=='SubStem_41'
    assert a.native_station_park_reference and a.native_capsule_sphere_cover
    assert not a.gui and a.capture_milestones is capture and a.physics_hz==480
    assert a.render_hz==(15 if capture else 0)
    assert a.right_only_cut_trial is (mode=='right_only')
    assert a.require_retention_screen is (mode=='bimanual')
    assert a.cut_action_trial is (milestone=='cut_action')
    class Validated(Exception):pass
    def stop(self,*a,**kw):
        assert self==out
        raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(argv)


@pytest.mark.parametrize('mode',['bimanual','right_only'])
@pytest.mark.parametrize('historical',[False,True])
def test_public_launcher_never_silently_uses_the_wrong_physical_edge(monkeypatch,tmp_path,mode,historical):
    from . import benchmark,ground_truth_trial
    from .blade_contacts import CROSSBAR_EDGE,SIDE_EDGE
    from .knife import DOWNWARD_CUT_MODEL,BRITTLE_CUT_MODEL
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda args:captured.append(benchmark.parser().parse_args(args)))
    argv=['--output',str(tmp_path/'new'),'--mode',mode,'--milestone','cut_action']
    if historical:argv.append('--historical-mounting-plate')
    ground_truth_trial.main(argv)
    assert captured[0].knife_edge_mode==(SIDE_EDGE if historical else CROSSBAR_EDGE)
    assert captured[0].cut_model==(BRITTLE_CUT_MODEL if historical else DOWNWARD_CUT_MODEL)
    assert captured[0].source_wrist_contacts is (not historical)


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_process_zone_is_explicit_and_reaches_complete_validation(monkeypatch,tmp_path,mode):
    from . import benchmark,ground_truth_trial
    captured=[]
    def capture(options):captured.extend(options)
    monkeypatch.setattr(benchmark,'main',capture)
    ground_truth_trial.main(['--output',str(tmp_path/'new'),'--mode',mode,'--milestone','cut_action','--process-zone-trial'])
    a=benchmark.parser().parse_args(captured)
    assert a.seam_contact_yield and a.right_ready_degrees[0]==pytest.approx(120.23685010553707)
    assert not benchmark.parser().parse_args(arguments('old',mode,'cut_action')).seam_contact_yield


def test_process_zone_cannot_use_wrong_physical_edge(tmp_path):
    from .ground_truth_trial import main
    with pytest.raises(SystemExit):main(['--output',str(tmp_path/'none'),'--mode','bimanual',
        '--process-zone-trial','--historical-mounting-plate'])
    assert not (tmp_path/'none').exists()


@pytest.mark.parametrize('distance',[None,.01,.04,.08])
def test_explicit_pregrasp_distance_preserves_target_and_guards(monkeypatch,tmp_path,distance):
    from . import benchmark,ground_truth_trial
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda argv:captured.append(benchmark.parser().parse_args(argv)))
    argv=['--output',str(tmp_path/'none'),'--mode','bimanual','--milestone','cut_action','--process-zone-trial']
    if distance is not None:argv+=['--approach-distance',str(distance)]
    ground_truth_trial.main(argv)
    args=captured[0]
    assert args.approach_distance==(.02 if distance is None else distance)
    assert args.grasp_arc_m==.08 and args.cut_arc_m==.02
    assert args.native_startup_clearance and args.native_static_clearance and args.require_retention_screen
    assert args.force_closure and args.physics_hz==480
    assert not (tmp_path/'none').exists()


@pytest.mark.parametrize('value',['0','-.01','.081','nan','inf'])
def test_invalid_pregrasp_distance_never_launches(tmp_path,value):
    from .ground_truth_trial import main
    with pytest.raises(SystemExit):
        main(['--output',str(tmp_path/'none'),'--mode','bimanual','--process-zone-trial',
            '--approach-distance',value])
    assert not (tmp_path/'none').exists()


@pytest.mark.parametrize('extra',[['--mode','right_only','--process-zone-trial'],['--mode','bimanual']])
def test_pregrasp_distance_requires_bimanual_process_zone(tmp_path,extra):
    from .ground_truth_trial import main
    with pytest.raises(SystemExit):
        main(['--output',str(tmp_path/'none'),'--approach-distance','.08',*extra])
    assert not (tmp_path/'none').exists()
