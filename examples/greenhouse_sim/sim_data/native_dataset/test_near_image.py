import copy
import numpy as np
import pytest
from .near_image import compare, separation_report


def measure(a=None, b=None, **kwargs):
    a = np.zeros((150,160,3),dtype=np.uint8) if a is None else a
    b = a.copy() if b is None else b
    return compare(a,b,left_junction_uv=kwargs.pop('left_junction_uv',[80.5,75.5]),
                   right_junction_uv=kwargs.pop('right_junction_uv',[80.5,75.5]), **kwargs)


def test_exact_and_no_mutation():
    a=np.random.default_rng(42).integers(0,256,(150,160,3),dtype=np.uint8); before=a.copy()
    result=measure(a)
    assert result['score']==0 and result['decoded_pixels_exact']
    assert not result['training_approved'] and not result['threshold_calibrated']
    assert np.array_equal(a,before)


def test_signed_difference_and_symmetric():
    a=np.zeros((150,160,3),dtype=np.uint8); b=np.full_like(a,255)
    x,y=measure(a,b),measure(b,a)
    assert x==y and x['score']==1 and x['full_frame']['max_delta']==255


def test_edge_origin_pixel_convention():
    assert measure(left_junction_uv=[80.9,75.9])['left_patch_xyxy']==[16,11,145,140]


def test_local_change_is_not_diluted_by_full_frame():
    a=np.zeros((150,160,3),dtype=np.uint8); b=a.copy(); b[75,80]=255
    value=measure(a,b,patch_radius=1)
    assert value['score']==pytest.approx(1/9)
    assert value['full_frame']['mae']<value['junction_patch']['mae']


def test_integer_translation_patch_no_interpolation():
    a=np.zeros((150,160,3),dtype=np.uint8); b=a.copy(); a[75,80]=255; b[76,81]=255
    value=measure(a,b,right_junction_uv=[81.5,76.5],patch_radius=2)
    assert value['junction_patch']['mae']==0
    assert value['full_frame']['mae']>0 and not value['decoded_pixels_exact']


@pytest.mark.parametrize('shape,dtype', [((150,160,3),np.float32),((150,160,4),np.uint8),
    ((150,160),np.uint8),((0,160,3),np.uint8),((150,160,3),bool)])
def test_reject_unsupported_images(shape,dtype):
    with pytest.raises(ValueError):measure(np.zeros(shape,dtype=dtype))


def test_no_resolution_conversion():
    with pytest.raises(ValueError):measure(b=np.zeros((151,160,3),dtype=np.uint8))


@pytest.mark.parametrize('point',[[0,0],[159,149],[160,75],[-1,75],[float('nan'),75],
    [80,float('inf')],[True,75],[80],'point'])
def test_no_clipped_or_invalid_patch(point):
    with pytest.raises(ValueError):measure(left_junction_uv=point)


@pytest.mark.parametrize('radius',[0,-1,64.0,True,1000])
def test_bad_patch_radius(radius):
    with pytest.raises(ValueError):measure(patch_radius=radius)


def control(pair,kind,level=0):
    return dict(pair_id=pair,source_family='donor',source_target='target',split='train',
        evidence_id='external-receipt',control_kind=kind,
        measurement=measure(b=np.full((150,160,3),level,dtype=np.uint8)))


def test_separation_never_certifies():
    report=separation_report([control('a','duplicate',8),control('b','meaningful_change',20)])
    assert report['empirical_controls_separate']
    assert report['candidate_interval']==[8/255,20/255]
    assert not report['calibration_validated'] and not report['control_labels_verified']
    assert report['threshold_selected'] is None


@pytest.mark.parametrize('controls', [[control('a','duplicate',20),control('b','meaningful_change',8)],
    [control('a','duplicate',8),control('b','meaningful_change',8)], [control('a','duplicate',8)]])
def test_overlap_or_missing_control_holds(controls):
    result=separation_report(controls)
    assert not result['empirical_controls_separate'] and result['candidate_interval'] is None


@pytest.mark.parametrize('key,value',[('split','test'),('split','validation'),('control_kind','unknown'),
    ('evidence_id',''),('source_family',''),('pair_id','')])
def test_bad_control_provenance(key,value):
    row=control('a','duplicate');row[key]=value
    with pytest.raises(ValueError):separation_report([row])


def test_duplicate_pair_and_mixed_definitions():
    row=control('a','duplicate')
    with pytest.raises(ValueError):separation_report([row,copy.deepcopy(row)])
    other=control('b','meaningful_change');other['measurement']['patch_radius']=32
    with pytest.raises(ValueError):separation_report([row,other])


@pytest.mark.parametrize('score',[float('nan'),float('inf'),-1,1.1,True,.1])
def test_bad_measurement(score):
    row=control('a','duplicate');row['measurement']['score']=score
    with pytest.raises(ValueError):separation_report([row])
