"""Synthetic pixel fixtures validate pruning, never native qualification."""
from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
from . import near_image_index as index
from . import near_image
from .admission import decoded_rgb_digest


def pin(tmp_path, name, rgb, uv=(130.5,130.5)):
    path=tmp_path/(name+'.png');Image.fromarray(rgb).save(path)
    return index.ImagePin(name,str(path.resolve()),index._sha(path),
        decoded_rgb_digest(rgb.tobytes(),width=rgb.shape[1],height=rgb.shape[0]),uv)


@pytest.mark.parametrize('shape,tile',[((257,273,3),32),((1696,816,3),32),((131,143,3),256),((140,141,3),1)])
def test_partial_cells_lower_bound(shape,tile):
    rng=np.random.default_rng(25)
    a,b=(rng.integers(0,256,shape,dtype=np.uint8) for _ in range(2))
    fa,w=index._features(a,tile);fb,other=index._features(b,tile)
    assert np.array_equal(w,other) and w.sum()==pytest.approx(1)
    assert float(np.sum(np.abs(fa-fb)*w))<=index._mae(a,b)+1e-12


def test_score_matches_existing_metric_exactly():
    rng=np.random.default_rng(91)
    a,b=(rng.integers(0,256,(260,300,3),dtype=np.uint8) for _ in range(2))
    assert index.exact_score(a,b,(150.1,130.4),(160.2,140.4))==near_image.compare(
        a,b,left_junction_uv=[150.1,130.4],right_junction_uv=[160.2,140.4])['score']


def test_all_pairs_match_bruteforce(tmp_path):
    rng=np.random.default_rng(8);a=rng.integers(0,40,(261,270,3),dtype=np.uint8)
    arrays=[a,a.copy(),a+3,np.full_like(a,200),np.full_like(a,205)]
    pins=[pin(tmp_path,str(i),rgb) for i,rgb in enumerate(arrays)]
    g=index.build_graph(pins,inventory_sha256='a'*64,threshold=.04,cache_images=1)
    expected=[[str(i),str(j)] for i in range(len(arrays)) for j in range(i+1,len(arrays))
        if index.exact_score(arrays[i],arrays[j],(130.5,130.5),(130.5,130.5))<=.04]
    assert g['near_image_edges']==expected
    assert g['counts']['resolved_pairs']==10 and g['counts']['bound_pruned_pairs']>0
    assert g['counts']['exact_compared_pairs']>0 and g['complete']
    assert not g['training_approved'] and not g['global_collection_complete'] and not g['threshold_calibrated']
    payload=dict(g);digest=payload.pop('sha256');assert index._digest(payload)==digest


def test_zero_threshold_boundary(tmp_path):
    a=np.full((256,256,3),50,dtype=np.uint8)
    g=index.build_graph([pin(tmp_path,'a',a),pin(tmp_path,'b',a)],inventory_sha256='1'*64,threshold=0)
    assert g['near_image_edges']==[['a','b']] and g['edge_scores']==[0]


def test_local_patch_contributes_even_if_full_rgb_identical(tmp_path):
    a=np.zeros((300,500,3),dtype=np.uint8);a[:,250:]=255
    ps=[pin(tmp_path,'a',a,(100.,150.)),pin(tmp_path,'b',a,(400.,150.))]
    g=index.build_graph(ps,inventory_sha256='1'*64,threshold=.04)
    assert g['near_image_edges']==[] # Exact-image alias merging is downstream.


@pytest.mark.parametrize('field,value',[('sha256','0'*64),('decoded_rgb_sha256','0'*64),
    ('nominal_uv',(2.,2.)),('nominal_uv',(float('nan'),130.)),('path','relative.png')])
def test_bad_native_pin(tmp_path,field,value):
    p=pin(tmp_path,'a',np.zeros((256,256,3),dtype=np.uint8))
    with pytest.raises(ValueError):index.build_graph([replace(p,**{field:value})],inventory_sha256='1'*64,threshold=.04)


def test_source_mutation_fails_at_finish(tmp_path):
    a=np.zeros((256,256,3),dtype=np.uint8);p=pin(tmp_path,'a',a)
    def mutate(state):
        if state['phase']=='pairs':Image.fromarray(a+1).save(p.path)
    with pytest.raises(ValueError,match='changed'):
        index.build_graph([p],inventory_sha256='1'*64,threshold=.04,progress=mutate)


def test_mixed_resolution_and_duplicate_id(tmp_path):
    a=pin(tmp_path,'a',np.zeros((256,256,3),dtype=np.uint8))
    b=pin(tmp_path,'b',np.zeros((257,256,3),dtype=np.uint8))
    with pytest.raises(ValueError,match='Mixed'):index.build_graph([a,b],inventory_sha256='1'*64,threshold=.04)
    with pytest.raises(ValueError,match='Unique'):index.build_graph([a,a],inventory_sha256='1'*64,threshold=.04)


@pytest.mark.parametrize('extra',[{'threshold':-1},{'threshold':float('nan')},
    {'threshold':2},{'tile':0},{'tile':257},{'cache_images':0},{'inventory_sha256':'bad'}])
def test_invalid_parameters(tmp_path,extra):
    p=pin(tmp_path,'a',np.zeros((256,256,3),dtype=np.uint8))
    kwargs=dict(threshold=.04,inventory_sha256='1'*64);kwargs.update(extra)
    with pytest.raises(ValueError):index.build_graph([p],**kwargs)
