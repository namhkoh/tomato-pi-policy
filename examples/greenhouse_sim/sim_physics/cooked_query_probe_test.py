"""Pure ray/measurement tests; native agreement needs the explicit Kit probe."""
import itertools
import numpy as np
import pytest

from .cooked_query_probe import ray_interval_distance, ray_fixture, compare_rays, world_planes


def box_planes(half=1.):
    return np.column_stack([np.vstack([np.eye(3),-np.eye(3)]),np.full(6,-half)])


def test_ray_entry_parallel_miss_inside_and_too_short():
    planes=box_planes()
    assert ray_interval_distance(planes,[-2,0,0],[1,0,0],4)==pytest.approx(1.)
    assert ray_interval_distance(planes,[2,0,0],[-1,0,0],4)==pytest.approx(1.)
    assert ray_interval_distance(planes,[-2,2,0],[1,0,0],4) is None
    assert ray_interval_distance(planes,[0,0,0],[1,0,0],4)==0.
    assert ray_interval_distance(planes,[-2,0,0],[-1,0,0],4) is None
    assert ray_interval_distance(planes,[-2,0,0],[1,0,0],.5) is None


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),0.,-1.])
def test_ray_invalid_extent(bad):
    with pytest.raises(ValueError):
        ray_interval_distance(box_planes(),[-2,0,0],[1,0,0],bad)


def test_ray_rejects_nonunit_direction_and_planes():
    with pytest.raises(ValueError):
        ray_interval_distance(box_planes(),[-2,0,0],[2,0,0],4)
    with pytest.raises(ValueError):
        ray_interval_distance(box_planes()*2,[-2,0,0],[1,0,0],4)


def test_fixture_contains_native_union_not_whole_union_hull():
    cube=np.array(list(itertools.product([-1,1],repeat=3)),float)*.002
    parts=[cube+[-.01,0,0],cube+[.01,0,0]]
    rays=ray_fixture(parts)
    result=compare_rays(rays,lambda path,r:r['expected_m'],'/World/A')
    assert result['passed'] and result['positive_cases']>0 and result['negative_cases']>0
    # Rays down Z through the gap remain misses, unlike one overall convex hull.
    gap=[r for r in rays if abs(r['direction'][2])==1
         and abs(r['origin_m'][0])<1e-8 and abs(r['origin_m'][1])<1e-8]
    assert gap and all(r['expected_m'] is None for r in gap)


@pytest.mark.parametrize('mode',['missing','extra','wrong_distance'])
def test_native_mismatch_fails_closed(mode):
    rows=[dict(origin_m=[-2,0,0],direction=[1,0,0],maximum_m=4.,expected_m=1.),
          dict(origin_m=[-2,3,0],direction=[1,0,0],maximum_m=4.,expected_m=None)]
    def query(path,row):
        assert path=='/World/ExactCollider'
        if mode=='missing': return None
        if mode=='extra': return 1.
        return 1.001 if row['expected_m'] is not None else None
    result=compare_rays(rows,query,'/World/ExactCollider')
    assert not result['passed'] and result['mismatches']==1


def test_empty_or_only_miss_cannot_pass():
    assert not compare_rays([],lambda *args:None,'/World/A')['passed']
    rows=[dict(origin_m=[-2,3,0],direction=[1,0,0],maximum_m=4.,expected_m=None)]
    assert not compare_rays(rows,lambda *args:None,'/World/A')['passed']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,5.])
def test_bad_native_distance(value):
    row=dict(origin_m=[-2,0,0],direction=[1,0,0],maximum_m=4.,expected_m=1.)
    with pytest.raises(ValueError):
        compare_rays([row],lambda *args:value,'/World/A')


def test_bad_parts_and_tolerance():
    with pytest.raises(ValueError):ray_fixture([])
    with pytest.raises(ValueError):ray_fixture([np.zeros((2,3))])
    with pytest.raises(ValueError):compare_rays([],lambda *args:None,'/World/A',tolerance_m=.001)


def test_raw_planes_not_refitted_to_vertices():
    cube=np.array(list(itertools.product([-1,1],repeat=3)),float)*.001
    # A native fitted face plane may extend outside the returned vertex hull.
    hull=ray_fixture([cube])
    raw=ray_fixture([cube],planes=[box_planes(.0012)])
    index=next(i for i,r in enumerate(hull) if r['expected_m'] is not None
               and abs(r['origin_m'][1])<1e-12 and abs(r['origin_m'][2])<1e-12)
    assert hull[index]['expected_m']-raw[index]['expected_m']==pytest.approx(.0002)


def test_plane_covectors_keep_full_transform_scale_and_units():
    matrix=np.array([[0,-3,0,10],[2,0,0,20],[0,0,-4,30],[0,0,0,1]],float)
    planes=world_planes(box_planes(),matrix,.01)
    # World box is centred (.1,.2,.3), half (.03,.02,.04).
    assert ray_interval_distance(planes,[-.1,.2,.3],[1,0,0],1)==pytest.approx(.17)
    assert ray_interval_distance(planes,[.1,.2,.5],[0,0,-1],1)==pytest.approx(.16)
    with pytest.raises(ValueError):world_planes(np.zeros((6,4)),matrix,.01)
    with pytest.raises(ValueError):world_planes(box_planes(),np.zeros((4,4)),1.)
