from collections import Counter
from pathlib import Path
import pytest
from .planting_slots import backdrop_source_slot,target_y
from .planting_slots import backdrop_source_address,validate_side


@pytest.mark.parametrize('target',[0,12,23])
def test_swap_preserves_every_original_asset_including_detailed_neighbor(target):
    def stock(i,slot):
        source=backdrop_source_slot(i,slot,target_side=True)
        if source is None:return 'target'
        if i==13:return 'detailed_neighbor'
        return ('backdrop',source)
    before=[stock(i,12) for i in range(24)]
    after=[stock(i,target) for i in range(24)]
    assert Counter(before)==Counter(after)
    assert after[target]=='target' and after[13]=='detailed_neighbor'
    assert [backdrop_source_slot(i,target,target_side=False) for i in range(24)]==list(range(24))
    assert target_y(target) in (-6.,0.,5.5)


@pytest.mark.parametrize('slot',[True,1,13,24,-1,'0',None])
def test_only_explicit_original_slots(slot):
    with pytest.raises(ValueError):target_y(slot)


def test_public_end_row_requires_explicit_source_and_full_context(tmp_path,monkeypatch):
    from .ground_truth_trial import main
    base=['--output',str(tmp_path/'new'),'--mode','right_only','--milestone','cut_action',
          '--process-zone-trial','--target-row-slot','0']
    with pytest.raises(SystemExit):main(base)
    class Validated(Exception):pass
    def stop(*a,**k):raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(base+['--source-station-trial','seed19_full/SubStem_41','--greenhouse-trial'])


def test_station_proposal_cannot_cross_planting_slots(tmp_path):
    import json
    from .station_proposal_test import fixture
    from .station_proposal import apply_to_arguments
    args,report,_=fixture(tmp_path);args.target_row_slot=0
    args.station_proposal_report.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='planting slot'):apply_to_arguments(args)


@pytest.mark.parametrize('side',[-1,1])
@pytest.mark.parametrize('target',[0,12,23])
def test_cross_side_swap_retains_exact_original_asset_addresses(side,target):
    def assignment(s,i,ts,t):
        address=backdrop_source_address(s,i,t,target_planting_side=ts,selected_gutter=True)
        if address is None:return 'target'
        if (s,i)==(1,13):return 'detailed_neighbor'
        return address
    before=[assignment(s,i,1,12) for s in (-1,1) for i in range(24)]
    after=[assignment(s,i,side,target) for s in (-1,1) for i in range(24)]
    assert len(after)==48 and Counter(after)==Counter(before)
    assert assignment(side,target,side,target)=='target'
    assert assignment(1,13,side,target)=='detailed_neighbor'
    for s in (-1,1):
        for i in range(24):
            assert backdrop_source_address(s,i,target,target_planting_side=side,selected_gutter=False)==(s,i)


@pytest.mark.parametrize('side',[True,False,0,2,-2,1.,'-1',None])
def test_planting_side_is_explicit_integer_sign(side):
    with pytest.raises(ValueError):validate_side(side)


def test_default_address_mapping_is_identical_to_previous_slot_mapping():
    for target in (0,12,23):
        for side in (-1,1):
            for slot in range(24):
                original=backdrop_source_slot(slot,target,target_side=side==1)
                expected=None if original is None else (side,original)
                assert backdrop_source_address(side,slot,target,selected_gutter=True)==expected


def test_station_proposal_cannot_cross_planting_sides(tmp_path):
    import json
    from .station_proposal_test import fixture
    from .station_proposal import apply_to_arguments
    args,report,_=fixture(tmp_path);args.target_planting_side=-1
    args.station_proposal_report.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='planting side'):apply_to_arguments(args)


def test_side_flag_requires_full_scene_and_preserves_guards(tmp_path,monkeypatch):
    from .ground_truth_trial import main
    from . import benchmark
    output=tmp_path/'not_created'
    args=['--output',str(output),'--mode','bimanual','--milestone','cut_action',
          '--process-zone-trial','--target-planting-side','-1']
    with pytest.raises(SystemExit):main(args)
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda argv:captured.append(benchmark.parser().parse_args(argv)))
    main(args+['--source-station-trial','seed71_full/SubStem_42','--greenhouse-trial'])
    a=captured[0]
    assert a.target_planting_side==-1 and a.context_gutters==3 and a.greenhouse_cut_trial
    assert a.native_startup_clearance and a.native_static_clearance and a.require_retention_screen
    assert not a.isolate_station and not a.branch_contact_fixture
    assert not output.exists()


def test_low_level_side_flag_cannot_remove_full_context(tmp_path):
    from .benchmark import main
    output=tmp_path/'not_created'
    with pytest.raises(ValueError,match='intact greenhouse'):
        main(['--output',str(output),'--target-planting-side','-1'])
    assert not output.exists()
