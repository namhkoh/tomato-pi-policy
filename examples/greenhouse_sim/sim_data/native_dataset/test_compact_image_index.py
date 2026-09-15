"""Synthetic storage/index regression fixtures; no native capture qualification.

No monkeypatching. Mutation cases change only tmp_path fixtures through the
index's public progress callback. The no-I/O-write check uses a Python audit
hook to reject writes/extraction and native-array reads during graph building.
"""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from . import bundle, compact_image_index as compact, near_image_index as index
from .admission import decoded_rgb_digest


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _json(path, value):
    _write(path, json.dumps(value, allow_nan=False).encode())


def source(tmp_path, name, rgb=None, uv=(85.5, 75.5), *, eligible=True, image_format='PNG'):
    """Real compact-reader format; intentionally synthetic pixels/buffers."""
    if rgb is None:
        rgb = np.full((150, 170, 3), 40, dtype=np.uint8)
    raw, root = tmp_path / name / 'raw', tmp_path / name / 'compact'
    h, w = rgb.shape[:2]
    for key, array in {
        'inputs/depth_m.npy': np.ones((h, w), dtype=np.float32),
        'supervision/renderer_instance_id.npy': np.zeros((h, w), dtype=np.uint32),
        'supervision/component_id.npy': np.zeros((h, w), dtype=np.uint32),
    }.items():
        stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
        _write(raw / key, stream.getvalue())
    for key, array in {
        'inputs/rgb.png': rgb, 'inputs/depth_valid.png': np.ones((h, w), dtype=np.uint8),
        'supervision/target_visible.png': np.zeros((h, w), dtype=np.uint8),
        'supervision/organ_type.png': np.zeros((h, w), dtype=np.uint8),
    }.items():
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format=image_format if key == 'inputs/rgb.png' else 'PNG')
        _write(raw / key, stream.getvalue())
    _json(raw / 'supervision/identities.json', {'component_catalogue': []})
    files = {p.relative_to(raw).as_posix(): dict(sha256=index._sha(p),
        role='observation' if p.parent.name == 'inputs' else 'ground_truth_supervision')
        for p in raw.rglob('*') if p.is_file()}
    _json(raw / 'sample.json', dict(files=files, training_sample_approved=False))
    _json(raw / 'supervision/label.json', dict(eligible=eligible, training_approved=False,
        nominal_pixel_uv=list(uv), anatomical_attachment_pixel_uv=[0., 0.], target_id='synthetic/target'))
    trace = {'passed': True} if eligible else None
    if trace is not None:
        _json(raw / 'supervision/query_trace.json', trace)
    sample_sha, label_sha = (index._sha(raw / p) for p in ('sample.json', 'supervision/label.json'))
    bundle.pack_sample(raw, root, sample_sha256=sample_sha, label_sha256=label_sha, query_trace=trace)
    decoded_sha = decoded_rgb_digest(rgb.tobytes(), width=w, height=h) if rgb.ndim == 3 and rgb.shape[2] == 3 else '0' * 64
    pin = compact.CompactImagePin(name, str(root.resolve()), index._sha(root / 'bundle.json'),
        sample_sha, label_sha, index._sha(raw / 'inputs/rgb.png'), decoded_sha, tuple(uv))
    return pin, raw


def graph(pins, **options):
    kwargs = dict(inventory_sha256='a' * 64, threshold=.04, cache_images=1)
    kwargs.update(options)
    return compact.build_graph(pins, **kwargs)


def payload(pin, name):
    root = Path(pin.root)
    manifest = json.loads((root / 'bundle.json').read_text())
    return root / manifest['files'][name]['stored_path']


def physical_pin(pin):
    return index.ImagePin(pin.sample_id, str(payload(pin, 'inputs/rgb.png')),
                          pin.encoded_rgb_sha256, pin.decoded_rgb_sha256, pin.nominal_uv)


def tree(root):
    return {str(path.relative_to(root)): index._sha(path) for path in root.rglob('*') if path.is_file()}


def test_mixed_sources_reuse_whole_original_index_exactly(tmp_path):
    rng = np.random.default_rng(76)
    a = rng.integers(0, 50, (151, 179, 3), dtype=np.uint8)
    arrays = [a, a.copy(), a + 3, np.full_like(a, 200), np.full_like(a, 205)]
    pairs = [source(tmp_path, str(i), rgb) for i, rgb in enumerate(arrays)]
    pins = [p for p, _ in pairs]
    raw = index.ImagePin('0', str((pairs[0][1] / 'inputs/rgb.png').resolve()),
                         pins[0].encoded_rgb_sha256, pins[0].decoded_rgb_sha256, pins[0].nominal_uv)
    mixed = [raw, *pins[1:]]
    before = tree(tmp_path)
    actual = graph(mixed)
    base = index.build_graph([raw, *map(physical_pin, pins[1:])],
                            inventory_sha256='a' * 64, threshold=.04, cache_images=1)
    for key in ('image_pins', 'sample_ids', 'near_image_edges', 'edge_scores', 'counts',
                'metric', 'patch_radius', 'patch_anchor', 'dimensions', 'pruning', 'complete'):
        assert actual[key] == base[key]
    expected = [[str(i), str(j)] for i in range(5) for j in range(i + 1, 5)
        if index.exact_score(arrays[i], arrays[j], pins[i].nominal_uv, pins[j].nominal_uv) <= .04]
    assert actual['near_image_edges'] == expected
    assert actual['counts']['bound_pruned_pairs'] > 0 and actual['counts']['exact_compared_pairs'] > 0
    assert actual['base_graph_sha256'] == base['sha256']
    assert actual['base_graph_schema'] == index.SCHEMA
    assert actual['schema'] == compact.SCHEMA != index.SCHEMA
    assert len(actual['compact_image_pins']) == 4
    assert before == tree(tmp_path)
    assert graph(reversed(mixed)) == actual
    body = dict(actual); digest = body.pop('sha256')
    assert index._digest(body) == digest
    assert all(actual[k] is False for k in ('training_approved', 'threshold_calibrated',
        'global_collection_complete', 'depth_recomputed', 'images_modified',
        'images_extracted', 'images_reencoded', 'full_native_audit_performed'))
    for pin in pins[1:]:
        for name in ('sample.json', 'supervision/label.json', 'supervision/query_trace.json', 'inputs/rgb.png'):
            path = payload(pin, name)
            assert actual['input_bindings'][str(path.resolve())] == index._sha(path)
        assert actual['input_bindings'][str(Path(pin.root) / 'bundle.json')] == pin.bundle_sha256
    for module in (compact, index, bundle):
        assert actual['implementation_bindings'][str(Path(module.__file__).resolve())] == index._sha(module.__file__)


def test_compact_no_writes_no_temp_extraction_no_native_array_reads(tmp_path):
    a, _ = source(tmp_path, 'a'); b, _ = source(tmp_path, 'b')
    before = tree(tmp_path)
    active = [True]
    observed_png_reads = []
    def audit(event, args):
        if not active[0]:
            return
        if event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate'):
            raise AssertionError('Graph attempted filesystem mutation: ' + event)
        if event == 'open':
            path, mode, flags = args
            assert not (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            name = str(path)
            assert not name.endswith(('.npy', '.ghn')), 'RGB indexing must not read/decode native arrays'
            if name.endswith('.png'):
                observed_png_reads.append(Path(name).resolve())
    sys.addaudithook(audit)
    try:
        result = graph([a, b], threshold=0)
    finally:
        active[0] = False
    assert result['edge_scores'] == [0.]
    assert set(observed_png_reads) == {payload(p, 'inputs/rgb.png').resolve() for p in (a, b)}
    assert tree(tmp_path) == before


def test_nominal_is_bound_label_input_not_attachment_or_inferred(tmp_path):
    rgb = np.zeros((200, 500, 3), dtype=np.uint8); rgb[:, 250:] = 255
    a, _ = source(tmp_path, 'a', rgb, (100., 100.))
    b, _ = source(tmp_path, 'b', rgb, (400., 100.))
    result = graph([a, b], threshold=0)
    assert a.decoded_rgb_sha256 == b.decoded_rgb_sha256
    assert result['near_image_edges'] == []  # Pixel alias merging is downstream, as in v1.
    assert result['image_pins'][0]['nominal_uv'] == a.nominal_uv
    assert result['source_nominal_coordinates'] == 'caller_supplied_bound_original_label'
    with pytest.raises(ValueError, match='nominal coordinates differ'):
        graph([replace(a, nominal_uv=b.nominal_uv)])


@pytest.mark.parametrize('field', ['bundle_sha256', 'sample_sha256', 'label_sha256',
                                  'encoded_rgb_sha256', 'decoded_rgb_sha256'])
def test_all_external_pins_are_required_and_checked(tmp_path, field):
    pin, _ = source(tmp_path, 'sample')
    for bad in ('0' * 64, '', 'g' * 64, None):
        with pytest.raises(ValueError):
            graph([replace(pin, **{field: bad})])


@pytest.mark.parametrize('when', ['before', 'during'])
@pytest.mark.parametrize('name', ['bundle.json', 'sample.json', 'inputs/rgb.png',
                                 'supervision/label.json', 'supervision/query_trace.json'])
def test_metadata_and_rgb_tampering_never_returns_complete_graph(tmp_path, when, name):
    pin, _ = source(tmp_path, 'sample')
    path = Path(pin.root) / name if name == 'bundle.json' else payload(pin, name)
    mutated = [False]
    def mutate(_state=None):
        if not mutated[0]:
            with path.open('ab') as stream:
                stream.write(b'\n ')  # Valid PNG/JSON can retain decoded identity, but its byte pin changed.
            mutated[0] = True
    if when == 'before':
        mutate()
    with pytest.raises(ValueError):
        graph([pin], progress=mutate if when == 'during' else None)


def test_reader_is_fresh_and_raw_directory_cannot_impersonate_compact(tmp_path):
    pin, raw = source(tmp_path, 'sample')
    cached = bundle.SampleReader(pin.root)
    assert cached.manifest is not None
    with pytest.raises(ValueError, match='complete compact'):
        graph([replace(pin, root=str(raw.resolve()))])
    with pytest.raises(ValueError, match='absolute'):
        graph([replace(pin, root='relative')])
    path = payload(pin, 'sample.json')
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        graph([replace(pin, root=str(cached.root))])


@pytest.mark.parametrize('damage', ['path_escape', 'encoding', 'alias', 'missing_nominal'])
def test_changed_bundle_cannot_bypass_reader_contract(tmp_path, damage):
    pin, _ = source(tmp_path, 'sample')
    marker = Path(pin.root) / 'bundle.json'
    manifest = json.loads(marker.read_text())
    if damage == 'path_escape':
        manifest['files']['inputs/rgb.png']['stored_path'] = '../outside.png'
    elif damage == 'encoding':
        manifest['files']['inputs/rgb.png']['encoding'] = 'native_npy_byteplanes_zstd'
    elif damage == 'alias':
        manifest['files']['inputs/rgb.png']['stored_path'] = manifest['files']['inputs/depth_valid.png']['stored_path']
    else:
        path = payload(pin, 'supervision/label.json')
        label = json.loads(path.read_text()); del label['nominal_pixel_uv']
        _json(path, label)
        entry = manifest['files']['supervision/label.json']
        entry.update(logical_bytes=path.stat().st_size, stored_bytes=path.stat().st_size,
                     logical_sha256=index._sha(path), stored_sha256=index._sha(path))
        manifest['source_label_sha256'] = index._sha(path)
        pin = replace(pin, label_sha256=index._sha(path))
    _json(marker, manifest)
    with pytest.raises(ValueError):
        graph([replace(pin, bundle_sha256=index._sha(marker))])


@pytest.mark.parametrize('uv', [(1., 1.), (float('nan'), 75.), (True, 75.), (85.,), None])
def test_invalid_or_clipped_nominal_not_repaired(tmp_path, uv):
    pin, _ = source(tmp_path, 'sample')
    with pytest.raises(ValueError):
        graph([replace(pin, nominal_uv=uv)])


def test_original_index_rejects_clipped_patch_even_when_label_matches(tmp_path):
    pin, _ = source(tmp_path, 'sample', uv=(1., 1.))
    with pytest.raises(ValueError, match='Complete native junction patch'):
        graph([pin])


@pytest.mark.parametrize('mode', ['L', 'RGBA', 'JPEG'])
def test_native_png_mode_and_format_are_never_converted(tmp_path, mode):
    shape = (150, 170) if mode == 'L' else (150, 170, 4 if mode == 'RGBA' else 3)
    pin, _ = source(tmp_path, 'sample', np.zeros(shape, dtype=np.uint8),
                     image_format='JPEG' if mode == 'JPEG' else 'PNG')
    with pytest.raises(ValueError, match='Unconverted native RGB PNG'):
        graph([pin])


def test_mixed_resolution_duplicate_identity_and_wrong_pin_types_rejected(tmp_path):
    a, raw = source(tmp_path, 'a')
    b, _ = source(tmp_path, 'b', np.zeros((151, 170, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='Mixed native resolutions'):
        graph([a, b])
    duplicate = index.ImagePin(a.sample_id, str((raw / 'inputs/rgb.png').resolve()),
                               a.encoded_rgb_sha256, a.decoded_rgb_sha256, a.nominal_uv)
    for values in ([a, a], [a, duplicate], [], [{}], [bundle.SampleReader(a.root)]):
        with pytest.raises(ValueError):
            graph(values)


def test_no_eligible_trace_and_all_raw_compatibility(tmp_path):
    a, _ = source(tmp_path, 'a', eligible=False)
    assert not payload(a, 'supervision/label.json').with_name('query_trace.json').exists()
    assert graph([a])['counts']['total_pairs'] == 0
    raw = physical_pin(a)
    value = graph([raw])
    assert value['compact_image_pins'] == [] and value['schema'] == compact.SCHEMA
    assert value['image_pins'] == index.build_graph([raw], inventory_sha256='a' * 64, threshold=.04)['image_pins']


def test_caller_nominal_list_snapshot_and_detached_output(tmp_path):
    pin, _ = source(tmp_path, 'a')
    uv = list(pin.nominal_uv)
    mutable = replace(pin, nominal_uv=uv)
    def mutate(_state):
        uv[:] = [1., 1.]
    result = graph([mutable], progress=mutate)
    assert result['compact_image_pins'][0]['nominal_uv'] == pin.nominal_uv
    result['compact_image_pins'][0]['root'] = 'output mutation'
    assert graph([pin])['compact_image_pins'][0]['root'] == pin.root


@pytest.mark.parametrize('options', [dict(threshold=True), dict(threshold=-1), dict(threshold=float('nan')),
    dict(threshold=2), dict(tile=0), dict(tile=257), dict(cache_images=0), dict(inventory_sha256='bad')])
def test_original_bounded_parameters_stay_authoritative(tmp_path, options):
    pin, _ = source(tmp_path, 'sample')
    with pytest.raises(ValueError):
        graph([pin], **options)
