"""Fake-query contracts only; native evidence is reported separately."""
import numpy as np
import pytest
from sim_physics.native_startup_clearance import NativeStartupClearance


def fixture(*,kind='capsule',link='arm',floor=False,**options):
    path='/Robot/'+link+'/collision';obstacle='/Floor/mesh' if floor else '/Plant/leaf'
    records={obstacle:(np.full(3,2.),np.full(3,3.))}
    state=dict(hit=False,guard_fail=False,queries=[],missing=False,foreign=False)
    def guard():
        if state['guard_fail']:raise RuntimeError('epoch changed')
    def box(c,a,h):
        state['queries'].append(('box',c.copy(),h.copy()))
        if state['foreign']:return ['/Unknown/shape']
        if np.all(c==2.5):return [] if state['missing'] else [obstacle,obstacle]
        return [obstacle] if state['hit'] else [path]
    def sphere(c,r):
        state['queries'].append(('sphere',c.copy(),r))
        return [obstacle] if state['hit'] else [path]
    n=NativeStartupClearance(box,sphere,records,[path],guard=guard,floor_root='/Floor' if floor else None,**options)
    shape=(np.array([0,0,-.03]),np.array([0,0,.03]),.02) if kind=='capsule' else (np.zeros(3),np.eye(3),np.full(3,.02))
    shapes=[(path,'/Robot/'+link,link,kind,shape)]
    return n,state,{link:np.eye(4)},shapes


@pytest.mark.parametrize('kind',['capsule','box'])
def test_full_inventory_controls_and_pose_screen_do_not_authorize_motion(kind):
    n,s,world,shapes=fixture(kind=kind)
    result=n.check(world,shapes)
    assert result['passed'] and not result['motion_authorized'] and not result['whole_path_certified']
    n.validate();n.close();r=n.report()
    assert r['initial_positive_controls']==r['final_positive_controls']==1
    assert r['final_validation_passed'] and not r['query_active'] and r['clear_candidates']==1
    assert s['queries'][0][0]==s['queries'][-1][0]=='box'
    np.testing.assert_array_equal(world['arm'],np.eye(4))


@pytest.mark.parametrize('kind',['capsule','box'])
def test_any_native_environment_hit_is_retained(kind):
    n,s,world,shapes=fixture(kind=kind);s['hit']=True
    assert not n.check(world,shapes)['passed']
    assert n.report()['clear_candidates']==0


@pytest.mark.parametrize('link,allowed',[('base',True),('wheel_l',True),('wheel_r',True),('arm',False)])
def test_only_exact_support_links_may_touch_floor(link,allowed):
    n,s,world,shapes=fixture(link=link,floor=True);s['hit']=True
    assert n.check(world,shapes)['passed'] is allowed


@pytest.mark.parametrize('change',['missing','foreign','guard_fail'])
def test_lost_actor_unknown_actor_or_changed_epoch_invalidates_prior_clearance(change):
    n,s,world,shapes=fixture();assert n.check(world,shapes)['passed'];s[change]=True
    with pytest.raises(RuntimeError):n.validate()
    assert not n.report()['final_validation_passed']
    with pytest.raises(RuntimeError):n.check(world,shapes)


def test_missing_positive_actor_cannot_construct_an_empty_scene():
    with pytest.raises(RuntimeError,match='Missing native actor'):
        NativeStartupClearance(lambda *a:[],lambda *a:[],{'/S':(np.zeros(3),np.ones(3))},['/R/b/c'],guard=lambda:None)


@pytest.mark.parametrize('bad',['omit','duplicate','foreign','identity','unknown_kind'])
def test_incomplete_or_misidentified_robot_geometry_rejects(bad):
    n,s,world,shapes=fixture()
    if bad=='omit':shapes=[]
    if bad=='duplicate':shapes=shapes*2
    if bad=='foreign':shapes=[('/Other/shape',*shapes[0][1:])]
    if bad=='identity':shapes=[(shapes[0][0],shapes[0][1],'wheel_l',*shapes[0][3:])]
    if bad=='unknown_kind':shapes=[(*shapes[0][:3],'mesh',shapes[0][4])]
    with pytest.raises(ValueError):n.check(world,shapes)
    assert not n.report()['final_validation_passed']


@pytest.mark.parametrize('margin',[0.,.000999,.051,float('nan'),True])
def test_complete_margin_cannot_be_reduced_or_coerced(margin):
    n,s,world,shapes=fixture()
    with pytest.raises(ValueError):n.check(world,shapes,margin=margin)


def test_native_budget_is_enforced_after_return_and_before_next_query():
    n,s,world,shapes=fixture(kind='box',max_queries=2)
    assert n.check(world,shapes)['passed']
    with pytest.raises(RuntimeError,match='budget'):n.validate()
    assert n.calls==2 and not n.report()['final_validation_passed']


def test_search_reserves_both_environment_and_robot_controls_without_queries():
    n,s,world,shapes=fixture(kind='box',max_queries=5)
    assert n.can_check_with_final_controls(world,shapes) and n.calls==1
    assert n.check(world,shapes)['passed'] and n.calls==2
    assert not n.can_check_with_final_controls(world,shapes) and n.calls==2
    n.validate()
    n._box(np.zeros(3),np.eye(3),np.full(3,.02))  # Owner's final robot control.
    assert n.calls==4 and n.validated and n.active and not n.errors


def test_sphere_cover_budget_cannot_be_mistaken_for_one_capsule_query():
    n,s,world,shapes=fixture(kind='capsule',max_queries=5)
    assert not n.can_check_with_final_controls(world,shapes)
    assert n.calls==1 and n.active
    s['guard_fail']=True
    with pytest.raises(RuntimeError,match='epoch'):
        n.can_check_with_final_controls(world,shapes)
    assert not n.active


def test_expired_native_call_is_not_a_clearance(monkeypatch):
    import sim_physics.native_startup_clearance as module
    n,s,world,shapes=fixture(kind='box');original=n.box_query
    def late(*args):
        hits=original(*args);monkeypatch.setattr(module.time,'perf_counter',lambda:n.started+61);return hits
    n.box_query=late
    with pytest.raises(RuntimeError,match='budget'):n.check(world,shapes)
    assert n.cleared==0


def test_native_query_payload_is_not_truthy_coercion():
    n,s,world,shapes=fixture();n.sphere_query=lambda *a:True
    with pytest.raises(RuntimeError,match='response'):n.check(world,shapes)


@pytest.mark.parametrize('bad',['scaled','nan','reflection'])
def test_nonrigid_proposed_body_is_never_queried(bad):
    n,s,world,shapes=fixture();world['arm'][0,0]={'scaled':2.,'nan':float('nan'),'reflection':-1.}[bad]
    with pytest.raises(ValueError):n.check(world,shapes)
    assert n.calls==1


def test_box_extent_keeps_margin_plus_native_roundoff_reserve():
    n,s,world,shapes=fixture(kind='box');assert n.check(world,shapes)['passed']
    assert np.all(s['queries'][1][2]>.021)


@pytest.mark.parametrize('kind',['capsule','box'])
@pytest.mark.parametrize('blocked',[True,False])
def test_exact_frozen_query_reuse_keeps_rejections_and_final_controls_fresh(kind,blocked):
    n,s,world,shapes=fixture(kind=kind,memoize_queries=True);s['hit']=blocked
    first=n.check(world,shapes);calls=n.calls
    assert n.check(world,shapes)==first and first['passed'] is (not blocked)
    assert n.calls==calls and n.cache_hits>0
    n.validate();assert n.calls==calls+1 and n.validated
    assert not n.report()['positive_controls_cached']
    n.close();assert not n.query_cache


@pytest.mark.parametrize('change',['pose','margin','shape','provider'])
def test_frozen_cache_never_rounds_or_ignores_changed_query_arguments(change):
    n,s,world,shapes=fixture(kind='box',memoize_queries=True)
    assert n.check(world,shapes)['passed'];calls=n.calls;options={}
    if change=='pose':world['arm'][0,3]=np.nextafter(0.,1.)
    if change=='margin':options['margin']=.001001
    if change=='shape':shapes[0][4][2][0]+=.001
    if change=='provider':n.box_query=lambda *a:['/Plant/leaf']
    out=n.check(world,shapes,**options)
    assert n.calls>calls
    if change=='provider':assert not out['passed']


@pytest.mark.parametrize('fault',['epoch','closed','missing_final_actor'])
def test_cached_clearance_is_revoked_on_epoch_or_final_control_failure(fault):
    n,s,world,shapes=fixture(kind='box',memoize_queries=True)
    assert n.check(world,shapes)['passed'] and n.query_cache
    if fault=='epoch':s['guard_fail']=True
    if fault=='closed':n.close()
    if fault=='missing_final_actor':
        s['missing']=True
        with pytest.raises(RuntimeError):n.validate()
    with pytest.raises(RuntimeError):n.check(world,shapes)
    assert not n.validated and not n.active


def test_frozen_query_cache_is_bounded_and_never_owns_mutable_hit_lists():
    n,s,world,shapes=fixture(kind='box',memoize_queries=True)
    returned=['/Robot/arm/collision'];n.box_query=lambda *a:returned
    assert n.check(world,shapes)['passed'];returned.append('/Plant/leaf')
    assert n.check(world,shapes)['passed']  # Original validated snapshot copied.
    # Different poses outside the cache capacity still make native calls.
    n.query_cache={i:frozenset() for i in range(4096)};world['arm'][0,3]=.01
    calls=n.calls;assert not n.check(world,shapes)['passed']
    assert n.calls==calls+1 and len(n.query_cache)==4096


def test_invalid_cache_option_is_not_coerced():
    with pytest.raises(ValueError):fixture(memoize_queries=1)
