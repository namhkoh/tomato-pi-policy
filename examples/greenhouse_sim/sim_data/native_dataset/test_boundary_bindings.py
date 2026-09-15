"""Synthetic CPU/filesystem tests only; no generated-stage or speedup claim."""
import ast
import hashlib
import os
from pathlib import Path
from types import MappingProxyType

import pytest

from . import boundary_bindings as boundary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sources(tmp_path):
    a, b = tmp_path / 'layer.usda', tmp_path / 'plan-source.bin'
    a.write_bytes(b'original layer')
    b.write_bytes(b'original plan source')
    return a, b


def observe_hashes(monkeypatch):
    calls = []
    original = boundary._digest
    def checked(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(boundary, '_digest', checked)
    return calls


def forbid_hashes(monkeypatch):
    def forbidden(*args):
        pytest.fail('membership/path/pin conflict must fail before hashing')
    monkeypatch.setattr(boundary, '_digest', forbidden)


def test_exact_overlap_hashed_once_without_mutating_inputs(sources, monkeypatch):
    a, b = sources
    current = [str(a), str(a)]
    baseline = {str(a): digest(a)}
    plan = {**baseline, str(b): digest(b)}
    before = (list(current), dict(baseline), dict(plan))
    calls = observe_hashes(monkeypatch)
    result = boundary.verify_boundary(current, baseline, plan)
    assert calls == [str(a), str(b)]
    assert result == dict(schema=boundary.SCHEMA, passed=True, layer_files=1, plan_files=2,
        unique_paths=2, hashes_performed=2, duplicate_hashes_avoided=1,
        layer_path_set_checked=True, cache_scope='this_call_only', stat_only_acceptance=False,
        native_benefit_qualified=False)
    assert (current, baseline, plan) == before


def test_no_overlap_hashes_each_source(sources, monkeypatch):
    a, b = sources
    calls = observe_hashes(monkeypatch)
    result = boundary.verify_boundary([str(a)], {str(a): digest(a)}, {str(b): digest(b)})
    assert result['duplicate_hashes_avoided'] == 0 and calls == [str(a), str(b)]


def test_every_call_rehashes_even_same_objects(sources, monkeypatch):
    a, b = sources
    baseline, plan = {str(a): digest(a)}, {str(b): digest(b)}
    current = [str(a)]
    calls = observe_hashes(monkeypatch)
    for _ in range(3): boundary.verify_boundary(current, baseline, plan)
    assert calls == [str(a), str(b)] * 3


@pytest.mark.parametrize('target', ['layer', 'plan'])
def test_same_size_mtime_tampering_rejected_next_boundary(sources, target):
    a, b = sources
    baseline, plan = {str(a): digest(a)}, {str(b): digest(b)}
    boundary.verify_boundary([str(a)], baseline, plan)
    changed = a if target == 'layer' else b
    info, raw = changed.stat(), changed.read_bytes()
    changed.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(changed, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(ValueError, match='SHA256 changed'):
        boundary.verify_boundary([str(a)], baseline, plan)


@pytest.mark.parametrize('change', ['added', 'removed', 'disappeared', 'different_raw_spelling'])
def test_current_raw_layer_set_checked_before_hashing(sources, monkeypatch, change):
    a, b = sources
    baseline = {str(a): digest(a)}
    plan = {**baseline, str(b): digest(b)}
    current = [str(a)]
    if change == 'added': current.append(str(b))  # Plan pin cannot authorize an extra loaded layer.
    if change == 'removed': current.clear()
    if change == 'disappeared': a.unlink()
    if change == 'different_raw_spelling': current = [str(a.parent) + '/./' + a.name]
    forbid_hashes(monkeypatch)
    with pytest.raises(ValueError, match='path SET'): boundary.verify_boundary(current, baseline, plan)


def test_missing_and_directory_current_entries_filtered_like_source_hashes(sources, tmp_path):
    a, b = sources
    missing = tmp_path / 'missing'
    current = [str(a), str(missing), str(tmp_path)]
    result = boundary.verify_boundary(current, {str(a): digest(a)}, {str(b): digest(b)})
    assert result['layer_files'] == 1


@pytest.mark.parametrize('kind', ['missing', 'directory'])
def test_plan_entries_are_not_silently_filtered(sources, tmp_path, kind):
    a, _ = sources
    target = tmp_path / 'missing' if kind == 'missing' else tmp_path
    with pytest.raises(OSError):
        boundary.verify_boundary([str(a)], {str(a): digest(a)}, {str(target): '0' * 64})


def test_empty_layer_set_and_empty_regular_file_supported(tmp_path):
    path = tmp_path / 'empty'; path.touch()
    result = boundary.verify_boundary([], {}, {str(path): digest(path)})
    assert result['hashes_performed'] == 1 and result['layer_files'] == 0
    assert boundary.verify_boundary([str(path)], {str(path): digest(path)},
        {str(path): digest(path)})['hashes_performed'] == 1


@pytest.mark.parametrize('alias', ['dot', 'separator', 'trailing_separator'])
def test_only_unambiguous_lexical_aliases_merge(sources, monkeypatch, alias):
    a, _ = sources
    raw = str(a.parent) + '/./' + a.name
    if alias == 'separator': raw = str(a.parent) + '//' + a.name
    if alias == 'trailing_separator': raw = str(a) + '/'
    calls = observe_hashes(monkeypatch)
    result = boundary.verify_boundary([str(a)], {str(a): digest(a)}, {raw: digest(a)})
    assert result['hashes_performed'] == 1 and calls == [str(a)]


def test_original_raw_layer_spelling_is_opened(sources, monkeypatch):
    a, _ = sources
    raw = str(a.parent) + '/./' + a.name
    calls = observe_hashes(monkeypatch)
    result = boundary.verify_boundary([raw], {raw: digest(a)}, {str(a): digest(a)})
    assert result['hashes_performed'] == 1 and calls == [raw]


@pytest.mark.parametrize('kind', ['same_path', 'dot_alias', 'inside_plan'])
def test_conflicting_pins_fail_before_hash(sources, monkeypatch, kind):
    a, _ = sources
    raw = str(a.parent) + '/./' + a.name
    baseline = {str(a): digest(a)}
    plan = {str(a) if kind == 'same_path' else raw: '0' * 64}
    if kind == 'inside_plan': plan[str(a)] = digest(a)
    forbid_hashes(monkeypatch)
    with pytest.raises(ValueError, match='Conflicting'): boundary.verify_boundary([str(a)], baseline, plan)


def test_parent_alias_is_held_not_collapsed(sources, monkeypatch):
    a, _ = sources
    raw = str(a.parent) + '/unused/../' + a.name
    forbid_hashes(monkeypatch)
    with pytest.raises(ValueError, match='Parent-traversal'):
        boundary.verify_boundary([str(a)], {str(a): digest(a)}, {raw: digest(a)})


@pytest.mark.skipif(os.name != 'nt', reason='Windows filesystem pathname rules')
@pytest.mark.parametrize('path', [r'C:relative', r'\\?\C:\asset', r'\\.\NUL',
    r'C:\asset:stream', r'C:\folder\NUL', r'C:\folder\CON.txt', r'C:\folder\COM1',
    r'C:\folder\trailing.', r'C:\folder\trailing '])
def test_windows_ambiguous_or_device_paths_held(path):
    with pytest.raises(ValueError): boundary._path(path)


def test_host_case_collision_is_held_even_if_content_pins_match(sources, monkeypatch):
    a, _ = sources
    # Exercise host collision policy without assuming POSIX has case aliases.
    monkeypatch.setattr(boundary.os.path, 'normcase', lambda p: p.lower())
    forbid_hashes(monkeypatch)
    with pytest.raises(ValueError, match='case-alias'):
        boundary.verify_boundary([str(a)], {str(a): digest(a)},
                                 {str(a.parent / a.name.upper()): digest(a)})


def link_directory(target, link):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != 'nt': raise
        import _winapi
        _winapi.CreateJunction(str(target), str(link))


def remove_link(link):
    if link.is_symlink(): link.unlink()
    else: link.rmdir()  # Remove only the test-created junction, never its target.


def test_symlink_or_junction_route_is_separate_and_rehashed(tmp_path, monkeypatch):
    directory = tmp_path / 'original'; directory.mkdir()
    a = directory / 'asset'; a.write_bytes(b'source')
    link = tmp_path / 'alias'; link_directory(directory, link)
    try:
        via_link = link / 'asset'
        calls = observe_hashes(monkeypatch)
        result = boundary.verify_boundary([str(via_link)], {str(via_link): digest(a)}, {str(a): digest(a)})
        assert result['hashes_performed'] == 2 and calls == [str(via_link), str(a)]
        a.write_bytes(b'tamper')
        with pytest.raises(ValueError, match='SHA256 changed'):
            boundary.verify_boundary([str(via_link)], {str(via_link): hashlib.sha256(b'source').hexdigest()},
                                     {str(a): hashlib.sha256(b'source').hexdigest()})
    finally:
        remove_link(link)
    assert a.exists()


def test_hardlinks_not_coalesced_by_inode(sources, tmp_path, monkeypatch):
    a, _ = sources
    alias = tmp_path / 'hardlink'; os.link(a, alias)
    calls = observe_hashes(monkeypatch)
    result = boundary.verify_boundary([str(a)], {str(a): digest(a)}, {str(alias): digest(a)})
    assert result['hashes_performed'] == 2 and calls == [str(a), str(alias)]


@pytest.mark.parametrize('pin', [None, True, 1, '', 'a' * 63, 'a' * 65, 'A' * 64, 'g' * 64, b'a' * 64, '0' * 64 + '\n'])
def test_invalid_sha256_rejected_before_hash(sources, monkeypatch, pin):
    a, _ = sources
    forbid_hashes(monkeypatch)
    with pytest.raises(ValueError, match='SHA256'):
        boundary.verify_boundary([str(a)], {str(a): digest(a)}, {str(a): pin})


@pytest.mark.parametrize('value', [None, 1, True, 'path', b'path', Path('path'), {}])
def test_current_paths_require_explicit_iterable(sources, value):
    a, _ = sources
    with pytest.raises(ValueError): boundary.verify_boundary(value, {str(a): digest(a)}, {str(a): digest(a)})


@pytest.mark.parametrize('value', [None, [], (), 'path', False, 1])
def test_binding_maps_required(sources, value):
    a, _ = sources
    with pytest.raises(ValueError): boundary.verify_boundary([str(a)], value, {str(a): digest(a)})
    with pytest.raises(ValueError): boundary.verify_boundary([str(a)], {str(a): digest(a)}, value)


def test_nonempty_plan_required(sources):
    a, _ = sources
    with pytest.raises(ValueError, match='nonempty plan'):
        boundary.verify_boundary([str(a)], {str(a): digest(a)}, {})


@pytest.mark.parametrize('path', ['', 'relative', '../relative', '/a\x00b', None, 1, b'/bytes'])
def test_invalid_path_keys_rejected(sources, monkeypatch, path):
    a, _ = sources
    forbid_hashes(monkeypatch)
    with pytest.raises(ValueError):
        boundary.verify_boundary([str(a)], {str(a): digest(a)}, {path: digest(a)})


def test_generator_and_readonly_mappings_supported(sources):
    a, b = sources
    baseline = MappingProxyType({str(a): digest(a)})
    plan = MappingProxyType({str(b): digest(b)})
    assert boundary.verify_boundary((p for p in [str(a)]), baseline, plan)['passed']


@pytest.mark.parametrize('which', ['baseline', 'plan'])
def test_caller_map_mutation_during_call_held(sources, monkeypatch, which):
    a, b = sources
    baseline, plan = {str(a): digest(a)}, {str(b): digest(b)}
    original = boundary._digest
    def mutate(path):
        value = original(path)
        (baseline if which == 'baseline' else plan).clear()
        return value
    monkeypatch.setattr(boundary, '_digest', mutate)
    with pytest.raises(ValueError, match='Caller binding map changed'):
        boundary.verify_boundary([str(a)], baseline, plan)


def test_readonly_opens_no_resolve_or_persistent_cache(sources, monkeypatch):
    a, b = sources
    baseline, plan = {str(a): digest(a)}, {str(b): digest(b)}
    original = Path.open
    def readonly(path, mode='r', *args, **kwargs):
        assert mode == 'rb'
        return original(path, mode, *args, **kwargs)
    def forbidden(*args, **kwargs): pytest.fail('No resolve/realpath trust shortcut')
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, 'open', readonly)
        scoped.setattr(Path, 'resolve', forbidden)
        assert boundary.verify_boundary([str(a)], baseline, plan)['passed']


def test_source_has_only_standard_library_imports():
    tree = ast.parse(Path(boundary.__file__).read_bytes())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports.update(a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names)
    assert imports == {'collections.abc', 'hashlib', 'os', 'pathlib', 're'}
