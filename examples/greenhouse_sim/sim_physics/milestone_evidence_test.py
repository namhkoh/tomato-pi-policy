import pytest
from .bimanual_probe import diagnostic_milestones,diagnostic_detail_views


def schedule(verified=False,cut=None,hold=False):
    return dict(diagnostic_milestones(grasp_verified=verified,grasp_time=6.5,
        hold_control=hold,stroke_start=11.,stroke_end=17.,delay=3.,cut_time=cut))


@pytest.mark.parametrize('hold',[False,True])
def test_no_nominal_time_or_cut_only_can_claim_verified_grasp(hold):
    assert 'grasp' not in schedule(hold=hold)
    assert schedule(verified=True,hold=hold)['grasp']==6.5


def test_release_and_observation_images_follow_actual_cut_not_stroke_schedule():
    assert 'severed' not in schedule() and 'post_cut_2s' not in schedule()
    milestones=schedule(cut=18.495833333333)
    assert milestones['severed']==18.495833333333
    assert milestones['post_cut_2s']==20.495833333333
    assert 'grasp' not in milestones


@pytest.mark.parametrize('name',['knife_precontact','severed','post_cut_2s','full_forward_stroke','knife_unloaded'])
def test_cut_evidence_includes_detail_views_not_only_the_wide_camera(name):
    assert diagnostic_detail_views(name)==[
        ('Right knife mount','knife'),('Grasp plant-side','plant_side'),
        ('Blade plane front','blade_front'),('Blade plane back','blade_back')]


def test_only_relevant_milestones_request_extra_views():
    assert diagnostic_detail_views('grasp')==[
        ('Grasp close-up','close'),('Grasp plant-side','plant_side')]
    assert diagnostic_detail_views('initial')==[]


def test_traversal_and_unloading_images_require_measured_events_not_release():
    args=dict(grasp_verified=True,grasp_time=6.5,hold_control=False,stroke_start=11.,
        stroke_end=17.,delay=3.,cut_time=23.5)
    assert 'full_forward_stroke' not in dict(diagnostic_milestones(**args))
    assert 'knife_unloaded' not in dict(diagnostic_milestones(**args))
    result=dict(diagnostic_milestones(**args,through_time=44.925,retraction_time=78.79))
    assert result['full_forward_stroke']==44.925 and result['knife_unloaded']==78.79
    assert diagnostic_detail_views('final')==diagnostic_detail_views('grasp')


@pytest.mark.parametrize('fail',[False,True])
def test_capture_details_restore_selected_camera_even_on_failure(fail):
    from types import SimpleNamespace as S
    from .bimanual_probe import capture_milestone
    viewport=S(camera_path='user-camera')
    viewport.set_active_camera=lambda p:setattr(viewport,'camera_path',p)
    fixture=S(select_view=viewport.set_active_camera)
    names=[]
    def capture(name):
        names.append(name)
        if fail and name=='final_close':raise RuntimeError('capture failure')
    if fail:
        with pytest.raises(RuntimeError,match='capture failure'):
            capture_milestone('final',capture,fixture,viewport)
    else:
        capture_milestone('final',capture,fixture,viewport)
        assert names==['final','final_close','final_plant_side']
    assert viewport.camera_path=='user-camera'
