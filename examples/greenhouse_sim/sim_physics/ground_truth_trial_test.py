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
