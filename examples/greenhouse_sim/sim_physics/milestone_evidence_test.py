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


@pytest.mark.parametrize('name',['knife_precontact','severed','post_cut_2s'])
def test_cut_evidence_includes_detail_views_not_only_the_wide_camera(name):
    assert diagnostic_detail_views(name)==[
        ('Right knife mount','knife'),('Grasp plant-side','plant_side'),
        ('Blade plane front','blade_front'),('Blade plane back','blade_back')]


def test_only_relevant_milestones_request_extra_views():
    assert diagnostic_detail_views('grasp')==[
        ('Grasp close-up','close'),('Grasp plant-side','plant_side')]
    assert diagnostic_detail_views('initial')==[]
