from pathlib import Path
import pytest


@pytest.mark.parametrize('mode',['bimanual','right_only'])
@pytest.mark.parametrize('through',[False,True])
@pytest.mark.parametrize('watch',[False,True])
def test_complete_greenhouse_profile_reaches_precreation_boundary(tmp_path,monkeypatch,mode,through,watch):
    from .ground_truth_trial import main
    class Validated(Exception):pass
    def stop(self,*a,**k):raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'new'),'--mode',mode,'--milestone','cut_action',
          '--process-zone-trial','--greenhouse-trial']
    if through:args+=['--through-stroke-trial']
    if watch:args+=['--watch']
    with pytest.raises(Validated):main(args)


@pytest.mark.parametrize('mutation',[
    dict(isolate_station=True),dict(branch_contact_fixture=True),dict(context_gutters=1),
    dict(native_static_clearance=False),dict(native_startup_clearance=False),
    dict(local_wire_physics=False),dict(stream_trajectory=False),dict(batch_gutter_visuals=False)])
def test_cannot_inherit_isolated_or_incomplete_scene_qualification(tmp_path,monkeypatch,mutation):
    from . import benchmark,ground_truth_trial
    from .greenhouse_cut import validate
    captured=[];monkeypatch.setattr(benchmark,'main',lambda options:captured.append(benchmark.parser().parse_args(options)))
    ground_truth_trial.main(['--output',str(tmp_path/'new'),'--mode','bimanual','--milestone','cut_action',
        '--process-zone-trial','--greenhouse-trial'])
    args=captured[0]
    assert validate(args) and not args.isolate_station and not args.branch_contact_fixture
    for k,v in mutation.items():setattr(args,k,v)
    with pytest.raises(ValueError,match='intact'):validate(args)


def test_intact_inventory_fails_on_excluded_source_component():
    from pxr import Usd,UsdGeom
    from .greenhouse_cut import intact_plant
    s=Usd.Stage.CreateInMemory()
    for p in ('/World/Gutters','/World/GutterWires','/World/Plant/A','/World/Plant/B'):UsdGeom.Xform.Define(s,p)
    record={'component_paths':{'a':'/World/Plant/A','b':'/World/Plant/B'}}
    before=s.GetRootLayer().ExportToString()
    assert intact_plant(s,record)['intact_component_count']==2
    assert before==s.GetRootLayer().ExportToString()
    s.GetPrimAtPath('/World/Plant/B').SetActive(False)
    with pytest.raises(RuntimeError,match='omit'):intact_plant(s,record)


def test_greenhouse_readonly_elbow_search_reaches_precreation_boundary(tmp_path,monkeypatch):
    from .ground_truth_trial import main
    class Validated(Exception):pass
    def stop(self,*a,**k):raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(['--output',str(tmp_path/'new'),'--mode','bimanual',
        '--milestone','cut_action','--process-zone-trial','--greenhouse-trial','--screen-ready-pose'])


@pytest.mark.parametrize('offset',[-.0015,0.,.0015])
def test_contact_aim_comparison_preserves_all_other_trial_options(tmp_path,monkeypatch,offset):
    from . import benchmark,ground_truth_trial
    captured=[];monkeypatch.setattr(benchmark,'main',lambda options:captured.append(benchmark.parser().parse_args(options)))
    base=['--output',str(tmp_path/'new'),'--mode','right_only','--milestone','cut_action',
          '--process-zone-trial','--through-stroke-trial']
    ground_truth_trial.main(base)
    ground_truth_trial.main(base+['--blade-aim-offset-m',str(offset)])
    original,changed=map(vars,captured)
    assert changed.pop('blade_axial_aim_offset_m')==offset
    original.pop('blade_axial_aim_offset_m')
    assert original==changed


@pytest.mark.parametrize('offset',['nan','inf','.0015001','-.002'])
def test_invalid_contact_aim_never_creates_output(tmp_path,offset):
    from .ground_truth_trial import main
    out=tmp_path/'none'
    with pytest.raises(SystemExit):main(['--output',str(out),'--mode','right_only',
        '--process-zone-trial','--blade-aim-offset-m',offset])
    assert not out.exists()
