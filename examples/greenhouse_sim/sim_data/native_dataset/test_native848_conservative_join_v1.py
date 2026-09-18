"""Pure CPU contract fixtures; never dataset reviews or generated evidence."""
import tempfile
from pathlib import Path
import pytest
from .native848_conservative_join_v1 import Pins, filter_identities, identities, shared_cap_selection


def row(i, *, target='seed1/SubStem_1', source='seed1/SubStem_1', family='seed1', split='train'):
    return dict(id=str(i), target_id=target, source_target=source, source_plant_family=family,
        split=split, rgb_sha256='rgb'+str(i), decoded_rgb_sha256='decoded'+str(i),
        source_sample_sha256='sample'+str(i), label_sha256='label'+str(i),
        view_signature='view'+str(i), conservative_camera_signature='camera'+str(i),
        selection_features=[i*.01,0,0,1,0,0,.2,.3])


def test_variants_share_original_cap_without_rewriting_actual_identity():
    original=[row(i) for i in range(12)]
    generated=[row(i,target='seed1_generated_'+str(i)+'/SubStem_1') for i in range(12,30)]
    selected=shared_cap_selection(original+generated)
    assert len(selected)==12
    assert all(r is (original+generated)[int(r['id'])] for r in selected)
    assert {r['source_target'] for r in selected}=={'seed1/SubStem_1'}


def test_fresh_original_control_can_fill_existing_pool_slot():
    assert len(shared_cap_selection([row(i) for i in range(4)]))==4


def test_no_geometry_alias_can_create_source_pool():
    with pytest.raises(ValueError):
        filter_identities([row(1,source='seed1_generated/SubStem_1')],set(),{'seed1':'train'})


def test_hold_propagates_across_transitive_encoded_and_camera_aliases():
    a,b,c=row(1),row(2),row(3)
    b['rgb_sha256']=a['rgb_sha256'];c['conservative_camera_signature']=b['conservative_camera_signature']
    selected,excluded=filter_identities([a,b,c],{('decoded_rgb',c['decoded_rgb_sha256'])},{'seed1':'train'})
    assert not selected and len(excluded)==3


def test_repeated_control_and_reencoded_rgb_have_zero_new_credit():
    a,b,c=row(1),row(2),row(3)
    b['conservative_camera_signature']=a['conservative_camera_signature']
    c['decoded_rgb_sha256']=a['decoded_rgb_sha256']
    selected,excluded=filter_identities([a,b,c],set(),{'seed1':'train'})
    assert selected==[a] and len(excluded)==2


def test_cross_split_identity_is_error_even_if_held():
    a=row(1);b=row(2,family='seed2',target='seed2/SubStem_1',source='seed2/SubStem_1',split='test')
    b['decoded_rgb_sha256']=a['decoded_rgb_sha256']
    with pytest.raises(ValueError): filter_identities([a,b],identities(a),{'seed1':'train','seed2':'test'})


def test_frozen_family_split_and_reused_ids_fail():
    with pytest.raises(ValueError): filter_identities([row(1,split='test')],set(),{'seed1':'train'})
    a,b=row(1),row(2);b['id']=a['id']
    with pytest.raises(ValueError): filter_identities([a,b],set(),{'seed1':'train'})


def test_changed_source_fails_initial_and_final_verification():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
        path=Path(folder)/'source';path.write_bytes(b'one');pins=Pins()
        with pytest.raises(ValueError):pins.add(path,'0'*64)
        pins=Pins();pins.add(path);path.write_bytes(b'two')
        with pytest.raises(ValueError):pins.finish()
