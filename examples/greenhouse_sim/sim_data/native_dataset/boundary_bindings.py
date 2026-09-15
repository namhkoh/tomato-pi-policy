"""Same-boundary SHA256 deduplication; no USD, native runtime or global cache.

verify_boundary(current_layer_paths, baseline_layer_hashes, plan_bindings)
receives REAL caller-owned inputs from the current generated stage. Pass
tuple(layer.realPath for layer in stage.GetUsedLayers()
      if not layer.anonymous and layer.realPath), plus the ORIGINAL loaded-layer
baseline hash dictionary and the current plan's source_bindings. Do not pass a
source-original capture inventory in place of the actual generated-stage list.

As in capture_pilot.source_hashes, current paths are filtered by Path.is_file()
(following links). Their RAW string set must equal the baseline keys BEFORE
normalization/merging. Thus a new missing/nonfile layer is ignored, a disappeared
baseline file fails, and even an equivalent renamed layer key fails. Plan files
are never filtered: missing/unreadable files fail when opened. Empty layer sets
are supported; plan bindings must be nonempty, as in verify_bindings.

Only absolute filesystem paths and lowercase SHA256 pins are accepted. Path
normalization is lexical and host-native (separators and '.'), NOT resolve(),
realpath(), inode identity, or content identity. '..' routes are explicitly held
rather than silently collapsed through a symlink. Windows case-colliding path
spellings are held (case-sensitive NTFS directories exist), not blindly merged.
Windows device/extended/drive-relative paths, reserved names and trailing-dot/
space aliases are unsupported. Distinct symlink
or hardlink routes remain distinct and are hashed separately. Original Path
spellings, not normalized keys, are opened. Empty regular files are supported.

Every unique normalized pathname is fully hashed once in THIS call. The helper
has no digest/stat/content cache across calls and does not move capture or audit
boundaries. It is not an atomic filesystem snapshot, malicious-writer lock, USD
change monitor, dirty-layer check, frame-sync check or native-performance proof.
The caller retains those checks, supplies stable inputs and invokes this helper
at every original boundary. A later source change is checked on the next call;
there is deliberately no second byte sweep inside this single boundary.

Native injection/benefit remains unqualified. Only CPU synthetic correctness
and an explicitly source-original-inventory proxy benchmark have been tested.
"""
from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import re


SCHEMA = 'greenhouse.boundary_bindings.v1'
_SHA256 = re.compile(r'[0-9a-f]{64}')
_WINDOWS_DEVICES = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$', 'CLOCK$'} | {
    prefix + suffix for prefix in ('COM', 'LPT') for suffix in '123456789\u00b9\u00b2\u00b3'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value):
    _require(isinstance(value, str) and value and '\x00' not in value,
             'Explicit nonempty filesystem path string required')
    path = Path(value)
    _require(path.is_absolute(), 'Absolute filesystem path required')
    _require('..' not in path.parts, 'Parent-traversal path alias unsupported')
    if os.name == 'nt':
        _require(not value.replace('/', '\\').startswith(('\\\\?\\', '\\\\.\\')),
                 'Windows device/extended path unsupported')
        _require(':' not in str(path)[len(path.drive):], 'Alternate data stream unsupported')
        _require(all(part == part.rstrip(' .') and part.split('.')[0].upper() not in _WINDOWS_DEVICES
                     for part in path.parts[1:]), 'Windows reserved/trailing path alias unsupported')
    return path


def _bindings(value, *, nonempty=False):
    _require(isinstance(value, Mapping) and (bool(value) or not nonempty),
             'Explicit nonempty plan bindings required' if nonempty else 'Explicit layer baseline mapping required')
    snapshot = dict(value)
    for name, expected in snapshot.items():
        _path(name)
        _require(isinstance(expected, str) and _SHA256.fullmatch(expected) is not None,
                 'Explicit lowercase SHA256 required: ' + name)
    return snapshot


def _merge(*mappings):
    merged, case_spellings = {}, {}
    for mapping in mappings:
        for name, expected in mapping.items():
            # Path applies the same harmless lexical normalization as the
            # original Path(name).open/is_file; traversal was rejected above.
            key = str(Path(name))
            folded = os.path.normcase(key)
            _require(folded not in case_spellings or case_spellings[folded] == key,
                     'Ambiguous case-alias spelling: ' + name)
            case_spellings[folded] = key
            _require(key not in merged or merged[key][1] == expected,
                     'Conflicting source SHA256: ' + name)
            merged.setdefault(key, (name, expected))
    return merged


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_boundary(current_layer_paths, baseline_layer_hashes, plan_bindings):
    """Verify current layer membership, then hash the union once; raise on hold.

    Inputs are path strings and mappings of absolute path strings to SHA256s.
    The returned counts describe THIS completed check, never reusable authority.
    No stage is read implicitly and no filesystem/source state is changed.
    """
    _require(not isinstance(current_layer_paths, (str, bytes, os.PathLike, Mapping)),
             'Explicit iterable of current nonanonymous layer path strings required')
    try:
        current = tuple(current_layer_paths)
    except TypeError as exc:
        raise ValueError('Explicit iterable of current layer paths required') from exc
    baseline = _bindings(baseline_layer_hashes)
    plan = _bindings(plan_bindings, nonempty=True)
    current_files = {name for name in current if _path(name).is_file()}
    _require(current_files == set(baseline), 'Current loaded-layer path SET differs from baseline')
    merged = _merge(baseline, plan)
    for name, expected in merged.values():
        _require(_digest(name) == expected, 'Source SHA256 changed: ' + name)
    # Detect changes to caller-owned expected pins; this is not a file lock.
    _require(dict(baseline_layer_hashes) == baseline and dict(plan_bindings) == plan,
             'Caller binding map changed during verification')
    return dict(schema=SCHEMA, passed=True, layer_files=len(baseline), plan_files=len(plan),
                unique_paths=len(merged), hashes_performed=len(merged),
                duplicate_hashes_avoided=len(baseline) + len(plan) - len(merged),
                layer_path_set_checked=True, cache_scope='this_call_only',
                stat_only_acceptance=False, native_benefit_qualified=False)
