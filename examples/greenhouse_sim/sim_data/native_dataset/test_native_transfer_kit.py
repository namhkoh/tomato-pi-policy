"""Synthetic CPU ZIP tests only; no actual release/20k ZIP or training claim.

Positive three-sample tests patch ONLY the frozen exporter's private count
checker. The public wrapper has no bypass. The normal >=20k hold is also tested.
Relocation subprocesses import code/prepare synthetic RGB only, with model and
simulator imports blocked. All archives/extractions/tampering are tmp_path data.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import warnings
import zipfile

import pytest

from . import native_transfer_kit as kit
from . import native_clear_export as export
from .test_native_clear_export import samples, build_tiny, tree


def tiny_counts(value):
    assert value == dict(train=1, validation=1, test=1)


@pytest.fixture
def tiny_gate(monkeypatch):
    monkeypatch.setattr(export, '_counts', tiny_counts)


@pytest.fixture(scope='module')
def fixture_release(tmp_path_factory, samples):
    root = tmp_path_factory.mktemp('synthetic_native_kit')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export, '_counts', tiny_counts)
        release, result, _ = build_tiny(root, samples)
    pins = root / 'code-pins.json'
    pins.write_bytes(json.dumps(kit.code_inventory(), indent=2).encode() + b'\n')
    return release, result['manifest_sha256'], export.FilePin(str(pins), export._file_sha(pins))


@pytest.fixture(scope='module')
def fixture_zip(fixture_release):
    root, sha, pins = fixture_release
    output = root.parent / 'synthetic-only.zip'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export, '_counts', tiny_counts)
        result = kit.package(root, output, manifest_sha256=sha, code_pins=pins)
    return output, result


def pack(fixture_release, output, **kwargs):
    root, sha, pins = fixture_release
    return kit.package(root, output, manifest_sha256=sha, code_pins=pins, **kwargs)


def check(path):
    return kit.verify(path, archive_sha256=export._file_sha(path))


def test_exact_recursive_code_inventory_and_no_simulator_runtime():
    pins = kit.code_inventory()
    assert len(pins) == 25
    assert set(export._CODE_NAMES) <= set(pins)
    assert {'audit.py', 'capture_contract.py', 'capture_visibility.py', 'cut_regions.py',
            'depth_preview.py', 'training_contract.py'} <= set(pins)
    assert 'native_dataset/native_h200_train.py' in pins and kit.README in pins
    assert 'h200_train.py' not in pins and 'capture_scene.py' not in pins
    assert all(not name.startswith('test_') and not name.endswith('.safetensors') for name in pins)
    for name in ('__init__.py', 'native_dataset/__init__.py'):
        assert kit._entry(kit._ROOT / name, pins[name])['size'] > 0
    for name in pins:
        if name.endswith('.py'):
            assert kit._imports(name, (kit._ROOT / name).read_bytes()) <= set(pins)


def test_import_scan_routes_function_local_imports_conservatively():
    source = b'from .capture_contract import project\ndef simulator_only():\n from .capture_scene import calibration\n'
    assert kit._imports('dataset_review.py', source) == {'capture_contract.py'}
    assert kit._imports('native_dataset/example.py', b'def f():\n from ..capture_contract import project\n') == {'capture_contract.py'}
    with pytest.raises(ValueError): kit._imports('native_dataset/example.py', b'import pxr\n')


def test_public_package_and_verify_keep_real_count_floor(fixture_release, fixture_zip, tmp_path):
    # No tiny_gate fixture: the deliberately small test release MUST fail.
    with pytest.raises(ValueError, match='TRAIN >=20000'):
        pack(fixture_release, tmp_path / 'not-a-release.zip')
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match='TRAIN >=20000'): check(fixture_zip[0])


@pytest.mark.parametrize('compression', ['stored', 'deflated'])
def test_lossless_member_bytes_and_source_buffers(fixture_release, tiny_gate, tmp_path, compression):
    root, manifest_sha, pins = fixture_release
    before = tree(root)
    output = tmp_path / (compression + '.zip')
    result = pack(fixture_release, output, compression=compression)
    assert result['native_source_validator_passed'] and result['byte_integrity_verified']
    assert not result['training_approved'] and not result['gpu_qualified']
    assert result['staging_directory'] is None and not list(tmp_path.glob('*.partial'))
    assert check(output)['release_manifest_sha256'] == manifest_sha
    with zipfile.ZipFile(output) as archive:
        assert {i.compress_type for i in archive.infolist()} == {kit._METHODS[compression]}
        for name, sha in before.items():
            raw = archive.read('release/' + name)
            assert hashlib.sha256(raw).hexdigest() == sha
        for name, sha in kit.code_inventory().items():
            assert hashlib.sha256(archive.read(kit.CODE_PREFIX + name)).hexdigest() == sha
        assert archive.read('SERVER_CODE_PINS.json') == Path(pins.path).read_bytes()
        assert archive.read('README_TRAINING.md') == (kit._ROOT / kit.README).read_bytes()
        assert b'sim_data.native_dataset.native_h200_train' in archive.read('README_TRAINING.md')
        manifest = export._parse(archive.read(kit.KIT_MANIFEST))
        assert manifest['counts'] == {'train': 1, 'validation': 1, 'test': 1}
        assert not manifest['policy']['biological_independence_claimed']
    assert tree(root) == before


@pytest.mark.parametrize('name', ['/absolute', '../outside', 'a/../../x', 'C:/outside',
    'a\\outside', 'a//b', './a', 'a/./b', 'a/', 'a\x00b', 'a\nb', 'CON.txt', 'a/NUL',
    'COM1.log', 'a/LPT9', 'a/trailing.', 'a/trailing ', 'a:stream', 'a?b', 'a*', 'caf\u00e9'])
def test_nonportable_names_rejected(name):
    with pytest.raises(ValueError): kit._name(name)


@pytest.mark.parametrize('names', [['a', 'a'], ['a', 'A'], ['A/x', 'a/y'], ['a', 'a/b'], ['A', 'a/b']])
def test_duplicates_case_and_prefix_collisions(names):
    with pytest.raises(ValueError): kit._names(names)


def minimal_zip(path, names, *, kind=stat.S_IFREG, compression=zipfile.ZIP_STORED):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(path, 'x') as archive:
            for name in names:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (kind | 0o644) << 16
                info.compress_type = compression
                archive.writestr(info, b'fixture')


@pytest.mark.parametrize('names', [['../outside'], ['a', 'a'], ['A', 'a'], ['A/x', 'a/y'], ['a', 'a/b']])
def test_zip_name_attacks_fail_before_manifest(tmp_path, names):
    path = tmp_path / 'attack.zip'; minimal_zip(path, names)
    with pytest.raises(ValueError): check(path)
    assert not (tmp_path.parent / 'outside').exists()


@pytest.mark.parametrize('kind', [stat.S_IFLNK, stat.S_IFDIR, stat.S_IFIFO, stat.S_IFCHR])
def test_zip_links_and_special_files_rejected(tmp_path, kind):
    path = tmp_path / 'attack.zip'; minimal_zip(path, ['member'], kind=kind)
    with pytest.raises(ValueError, match='Links/directories'): check(path)


def test_unsupported_compression_rejected(tmp_path):
    path = tmp_path / 'attack.zip'; minimal_zip(path, ['member'], compression=zipfile.ZIP_BZIP2)
    with pytest.raises(ValueError, match='unsupported ZIP'): check(path)


@pytest.mark.parametrize('location', ['central', 'local', 'both'])
def test_alternate_path_extra_metadata_rejected(tmp_path, location):
    path = tmp_path / 'extra.zip'
    info = kit._info('safe-name', zipfile.ZIP_STORED)
    extra = struct.pack('<HH', 0x7075, 5) + b'other'
    info.extra = extra if location != 'central' else b''
    with zipfile.ZipFile(path, 'x') as archive:
        archive.writestr(info, b'fixture')
        # ZipFile retains this ZipInfo for the central directory; local bytes
        # have already been written. Test each header independently.
        info.extra = extra if location != 'local' else b''
    with pytest.raises(ValueError, match='ZIP extra'): check(path)


@pytest.mark.parametrize('extra', [b'x', struct.pack('<HH', 1, 16) + b'x',
    struct.pack('<HH', 1, 4) + b'1234', (struct.pack('<HH', 1, 8) + b'0' * 8) * 2])
def test_malformed_or_duplicate_zip64_rejected(extra):
    with pytest.raises(ValueError): kit._zip_extra(extra)


@pytest.mark.parametrize('size', [8, 16, 24, 28])
def test_standard_zip64_extra_supported(size):
    kit._zip_extra(struct.pack('<HH', 1, size) + b'\0' * size)


@pytest.mark.parametrize('mutation', ['short', 'signature', 'flags', 'method', 'name', 'extra'])
def test_local_header_conflicts_rejected(mutation):
    info = kit._info('safe', zipfile.ZIP_STORED); info.header_offset = 0
    fields = [b'PK\x03\x04', 20, 0, 0, 0, 0, 0, 0, 0, 4, 0]
    if mutation == 'signature': fields[0] = b'FAIL'
    if mutation == 'flags': fields[2] = 1
    if mutation == 'method': fields[3] = zipfile.ZIP_DEFLATED
    if mutation == 'extra': fields[-1] = 10
    raw = struct.pack('<4s5H3I2H', *fields) + (b'evil' if mutation == 'name' else b'safe')
    if mutation == 'short': raw = raw[:5]
    with pytest.raises(ValueError): kit._member_header(io.BytesIO(raw), info)


def test_empty_source_member_explicitly_unsupported(tmp_path):
    path = tmp_path / 'empty'; path.touch()
    with pytest.raises(ValueError, match='member size'): kit._entry(path, hashlib.sha256(b'').hexdigest())


def rewrite_zip(source, destination, mutate):
    with zipfile.ZipFile(source) as original:
        members = {i.filename: (i, original.read(i)) for i in original.infolist()}
    mutate(members)
    with zipfile.ZipFile(destination, 'x') as changed:
        for info, raw in members.values(): changed.writestr(info, raw)


@pytest.mark.parametrize('mutation', ['content', 'readme', 'size', 'code_pin', 'extra', 'missing', 'counts', 'approval', 'bindings'])
def test_rebound_zip_internal_tampering(fixture_zip, tiny_gate, tmp_path, mutation):
    output = tmp_path / 'changed.zip'
    def change(members):
        info, raw = members[kit.KIT_MANIFEST]
        manifest = export._parse(raw)
        if mutation == 'content':
            name = next(n for n in members if n.endswith('.ghn'))
            meta, value = members[name]; members[name] = (meta, value[:-1] + bytes([value[-1] ^ 1]))
        if mutation == 'readme':
            meta, value = members['README_TRAINING.md']; members['README_TRAINING.md'] = (meta, value.replace(b'native', b'legacy', 1))
        if mutation == 'size': manifest['files']['README_TRAINING.md']['size'] += 1
        if mutation == 'code_pin': manifest['code_pins_sha256'] = '0' * 64
        if mutation == 'extra': members['extra'] = (kit._info('extra', 0), b'extra')
        if mutation == 'missing': del members[kit.CODE_PREFIX + 'audit.py']
        if mutation == 'counts': manifest['counts']['train'] = 20000
        if mutation == 'approval': manifest['policy']['training_approved'] = True
        if mutation == 'bindings': manifest['files'][kit.CODE_PREFIX + 'native_clear_contract.py']['sha256'] = '0' * 64
        members[kit.KIT_MANIFEST] = (info, export._json(manifest))
    rewrite_zip(fixture_zip[0], output, change)
    with pytest.raises((ValueError, KeyError, zipfile.BadZipFile)): check(output)


def test_external_archive_pin_is_required(fixture_zip):
    with pytest.raises(ValueError, match='External archive'): kit.verify(fixture_zip[0], archive_sha256='0' * 64)
    with pytest.raises(TypeError): kit.verify(fixture_zip[0])


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'wrong', 'whitespace_wrong_pin'])
def test_external_code_pins_are_required(fixture_release, tiny_gate, tmp_path, mutation):
    root, sha, _ = fixture_release
    value = kit.code_inventory()
    if mutation == 'missing': del value['audit.py']
    if mutation == 'extra': value['h200_train.py'] = '0' * 64
    if mutation == 'wrong': value['native_dataset/native_h200_train.py'] = '0' * 64
    path = tmp_path / 'pins.json'; path.write_bytes(export._json(value))
    pin = export.FilePin(str(path), export._file_sha(path))
    if mutation == 'whitespace_wrong_pin': path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError): kit.package(root, tmp_path / 'output.zip', manifest_sha256=sha, code_pins=pin)
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('mutation', ['existing', 'case', 'inside_release', 'inside_code', 'wrong_suffix', 'missing_parent', 'relative'])
def test_destination_safety(fixture_release, tiny_gate, tmp_path, mutation):
    output = tmp_path / 'output.zip'
    if mutation == 'existing': output.write_bytes(b'ORIGINAL')
    if mutation == 'case': (tmp_path / 'OUTPUT.ZIP').write_bytes(b'ORIGINAL')
    if mutation == 'inside_release': output = fixture_release[0] / 'output.zip'
    if mutation == 'inside_code': output = kit._ROOT / 'output.zip'
    if mutation == 'wrong_suffix': output = tmp_path / 'output.txt'
    if mutation == 'missing_parent': output = tmp_path / 'missing/output.zip'
    if mutation == 'relative': output = Path('output.zip')
    with pytest.raises((ValueError, FileNotFoundError)): pack(fixture_release, output)
    if mutation in ('existing', 'case'): assert (tmp_path / ('OUTPUT.ZIP' if mutation == 'case' else 'output.zip')).read_bytes() == b'ORIGINAL'


def test_filesystem_symlink_source_rejected(fixture_release, tmp_path):
    linked = tmp_path / 'linked'
    try: linked.symlink_to(fixture_release[0], target_is_directory=True)
    except OSError: pytest.skip('OS does not grant symlink creation; ZIP symlink cases are unconditional')
    with pytest.raises(ValueError, match='Symlink'): kit._regular(linked, directory=True)


def test_partial_failure_never_publishes(fixture_release, tiny_gate, tmp_path, monkeypatch):
    original = kit._write_entry
    def interrupted(*args):
        original(*args)
        raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(kit, '_write_entry', interrupted)
    with pytest.raises(RuntimeError): pack(fixture_release, tmp_path / 'new.zip')
    assert not (tmp_path / 'new.zip').exists()
    stages = list(tmp_path.glob('*.partial'))
    assert len(stages) == 1 and (stages[0] / 'archive.zip').exists()


def test_publication_race_does_not_clobber(fixture_release, tiny_gate, tmp_path, monkeypatch):
    original = os.link
    def race(source, destination):
        Path(destination).write_bytes(b'OTHER_OWNER')
        original(source, destination)
    monkeypatch.setattr(kit.os, 'link', race)
    with pytest.raises(FileExistsError): pack(fixture_release, tmp_path / 'new.zip')
    assert (tmp_path / 'new.zip').read_bytes() == b'OTHER_OWNER'
    assert list(tmp_path.glob('*.partial'))


def test_post_copy_source_tampering_held(fixture_release, tiny_gate, tmp_path, monkeypatch):
    source = tmp_path / 'source'; shutil.copytree(fixture_release[0], source)
    original = kit._write_entry
    last = 'release/splits/validation.jsonl'
    target = next(source.rglob('rgb.png'))
    def change_after_copy(archive, name, entry, method):
        original(archive, name, entry, method)
        if name == last: target.write_bytes(target.read_bytes() + b'changed')
    monkeypatch.setattr(kit, '_write_entry', change_after_copy)
    with pytest.raises(ValueError):
        pack((source, fixture_release[1], fixture_release[2]), tmp_path / 'new.zip')
    assert not (tmp_path / 'new.zip').exists()
    assert list(tmp_path.glob('*.partial'))


def test_readonly_verify_no_extract_or_writes(fixture_zip, tiny_gate, monkeypatch):
    def forbidden(*a, **kw): raise AssertionError('verification must not extract/create files')
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', forbidden)
    monkeypatch.setattr(zipfile.ZipFile, 'extract', forbidden)
    monkeypatch.setattr(kit.tempfile, 'mkdtemp', forbidden)
    before = tree(fixture_zip[0].parent)
    result = check(fixture_zip[0])
    assert result['byte_integrity_verified'] and not result['native_semantics_replayed']
    assert tree(fixture_zip[0].parent) == before


def test_verified_relocation_isolated_imports_and_native_rgb(fixture_zip, tiny_gate, tmp_path):
    check(fixture_zip[0])
    relocated = tmp_path / 'relocated'; relocated.mkdir()
    # Test-only extraction AFTER verification, into an exclusively created directory.
    with zipfile.ZipFile(fixture_zip[0]) as archive: archive.extractall(relocated)
    script = r'''
import sys, pathlib, json, importlib.abc
root = pathlib.Path(sys.argv[1]).resolve()
code = root / 'code/examples/greenhouse_sim'
sys.path.insert(0, str(code))
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pxr','omni','isaacsim','greenhouse_sim','torch','transformers','accelerate','deepspeed'}:
            raise AssertionError('forbidden simulator/model dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from sim_data.native_dataset import native_transfer_kit, native_clear_export, native_clear_processor_preflight
from sim_data.native_dataset import native_training_data, native_h200_train
assert len(native_transfer_kit.code_inventory()) == 25
for name, module in tuple(sys.modules.items()):
    if name == 'sim_data' or name.startswith('sim_data.'):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(code)
record = json.loads((root/'release/splits/train.jsonl').read_text().splitlines()[0])
messages = native_clear_export.model_messages(root/'release', record, supervised=True, crop=True)
assert [item['image'].size for item in messages[1]['content'][:2]] == [(1696,816),(768,768)]
answer = native_training_data.decode_answer(messages[-1]['content'][0]['text'])
assert 1053 < answer['cut_point_uv'][0] < 1054
print(json.dumps({'isolated_native_imports':True,'sizes':[[1696,816],[768,768]],'model_loaded':False}))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(relocated)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(result.stdout)['isolated_native_imports']
    # Original compact encoded files are identical after ZIP round-trip.
    for path in (relocated / 'release').rglob('*.ghn'):
        relative = path.relative_to(relocated).as_posix()
        with zipfile.ZipFile(fixture_zip[0]) as archive: assert path.read_bytes() == archive.read(relative)


def test_cli_no_draft_or_small_count_switch(capsys):
    kit.main(['code-pins'])
    assert len(json.loads(capsys.readouterr().out)) == 25
    for argv in (['package', '--allow-small'], ['package', '--inspection-only'], ['verify']):
        with pytest.raises(SystemExit): kit.main(argv)
