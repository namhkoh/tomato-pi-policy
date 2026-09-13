from collections import Counter
from pathlib import Path
import pytest
from .planting_slots import backdrop_source_slot,target_y


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
