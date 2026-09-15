"""Small native release + exact server-code ZIP64 wrapper. No release creation.

code_inventory() returns the reviewable {sim_data-relative filename: SHA256}
map. Save/authenticate it externally; package(release, destination, *,
manifest_sha256, code_pins=export.FilePin(ABS_JSON, SHA), compression='stored')
requires those explicit pins and the REAL native exporter validator. No draft,
small-count override, calibration/admission creation, model or network APIs.
verify(archive, *, archive_sha256) is streaming, read-only transport verification
with a trusted OUT-OF-BAND whole-ZIP pin, not independent native/admission review.

Layout: release/ (unchanged portable export), code/examples/greenhouse_sim/
sim_data/ (25 exact reviewed files), README_TRAINING.md (native README copy),
SERVER_CODE_PINS.json (exact caller bytes), KIT_MANIFEST.json (sizes/SHA256s).
No executable launch script, dependency installation, model snapshot or actual
processor report is generated. The native README specifies separate server
snapshot/preflight handoffs. Packaging does not qualify that training route.

The code closure includes export._CODE_NAMES, native preflight/data/runner/this
wrapper, package initializers, README/ZeRO-2 and six eager transitive helpers.
Static local imports are checked recursively through this reviewed allowlist.
Unused simulator-only function imports in legacy helper modules are NOT a
supported server API. Those CLIs/assets are intentionally absent. This is not
a general Python dependency resolver or an all-functions portability claim.

Only stored or deflated ZIP members; both preserve all logical/member bytes,
including original compact encoded buffers. No decompression/re-encoding of
native arrays, cropping, image conversion or temp image extraction is performed.
Deflated is recommended for eventual transfer, especially repeated JSON, after
measuring the actual kit; no real-release size or compression ratio is claimed.
Every member is streamed/hash checked during copy and again from the archive;
source inventories/pins are rechecked before exclusive hard-link publication.
The returned SHA256 must be distributed through a trusted channel, not inferred
from an untrusted adjacent checksum. No checksum-sidecar publication race.

Destination parent must already exist. A NEW private .partial directory is used;
failures leave it for diagnosis. Only this invocation's temporary link/empty
directory are removed after success; no overwrite, recursive deletion or resume.
Sources/parent must remain quiescent: checks are not filesystem locks or a
malicious-concurrent-writer sandbox. No public extractor/automatic code execution.
Names must be portable ASCII, case-unambiguous, regular files, with no links,
traversal, devices, streams, directory members or file/directory collisions.
Metadata is O(member count); file content is streamed in 1MiB chunks. Kit metadata
has a 256MiB structural bound; native per-file/metadata 64MiB bounds remain.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import zipfile

from . import native_clear_export as export
from .bundle import MAX_FILE_BYTES, read_bounded


SCHEMA = 'greenhouse.native_transfer_kit.v1'
CODE_PREFIX = 'code/examples/greenhouse_sim/sim_data/'
README = 'native_dataset/README_native_h200_train.md'
KIT_MANIFEST = 'KIT_MANIFEST.json'
MAX_KIT_METADATA = 256 * 1024 * 1024
_ROOT = Path(__file__).absolute().parent.parent
_SOURCE_SHA = export._file_sha(__file__)
_EXTRA_CODE = ('__init__.py', 'native_dataset/__init__.py',
    'native_dataset/native_clear_processor_preflight.py', 'native_dataset/native_training_data.py',
    'native_dataset/native_h200_train.py', 'native_dataset/native_transfer_kit.py', README, 'clear_zero2.json',
    'audit.py', 'capture_contract.py', 'capture_visibility.py', 'cut_regions.py', 'depth_preview.py', 'training_contract.py')
POLICY = dict(training_approved=False, gpu_qualified=False, processor_qualified=False,
    model_weights_included=False, training_performed=False, native_buffers_reencoded=False,
    biological_independence_claimed=False, admission='external_frozen_not_independently_reaudited')
_METHODS = {'stored': zipfile.ZIP_STORED, 'deflated': zipfile.ZIP_DEFLATED}


def _require(ok, message):
    if not ok: raise ValueError(message)


def _name(name):
    _require(isinstance(name, str) and name and all(32 <= ord(c) < 127 for c in name)
             and not any(c in name for c in '\\:<>"|?*'), 'Nonportable archive name')
    parts = name.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'CLOCK$'} | {p + str(i) for p in ('COM', 'LPT') for i in range(1, 10)}
    _require(all(p and p not in ('.', '..') and p == p.rstrip(' .')
                 and p.split('.')[0].upper() not in reserved for p in parts), 'Traversal/device/noncanonical name')
    return name


def _names(names):
    names = list(names)
    _require(len(set(names)) == len(names), 'Duplicate archive member')
    spellings, files = {}, {n.casefold() for n in names}
    for name in names:
        parts = _name(name).split('/')
        for i in range(1, len(parts) + 1):
            prefix = '/'.join(parts[:i]); key = prefix.casefold()
            _require(key not in spellings or spellings[key] == prefix, 'Case-colliding archive path')
            spellings[key] = prefix
            _require(i == len(parts) or key not in files, 'File/directory prefix collision')


def _regular(path, *, directory=False):
    path = Path(path)
    _require(path.is_absolute() and '..' not in path.parts, 'Explicit absolute nontraversing path required')
    for current in [*reversed(path.parents), path]:
        info = current.lstat()
        _require(not stat.S_ISLNK(info.st_mode)
                 and not (getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)),
                 'Symlink/junction/reparse path forbidden')
        _require(stat.S_ISDIR(info.st_mode) if current != path or directory else stat.S_ISREG(info.st_mode),
                 'Regular filesystem entry required')
    return path


def _new_output(path, protected):
    path = Path(path)
    _require(path.is_absolute() and '..' not in path.parts and path.suffix == '.zip', 'New absolute .zip destination required')
    _name(path.name)
    _regular(path.parent, directory=True)
    _require(not any(p.name.casefold() == path.name.casefold() for p in path.parent.iterdir()), 'Destination already exists/case-collides')
    _require(all(not path.is_relative_to(p) and not p.is_relative_to(path) for p in protected), 'Archive must be outside every source')
    return path


def _code_names():
    result = set(export._CODE_NAMES) | set(_EXTRA_CODE)
    _names(result)
    return result


def _imports(name, raw):
    """Reviewed server modules: all local imports; legacy helpers: eager only."""
    tree = ast.parse(raw, filename=name)
    package = ['sim_data', *name.split('/')[:-1]]
    full = name.startswith('native_dataset/')
    def walk(node):
        yield node
        if not full and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)): return
        for child in ast.iter_child_nodes(node): yield from walk(child)
    modules = []
    for node in walk(tree):
        if isinstance(node, ast.Import): modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                _require(node.level <= len(package), 'Import escapes server package')
                base = package[:len(package) - node.level + 1]
                if node.module: modules.append('.'.join(base + node.module.split('.')))
                else: modules.extend('.'.join(base + [a.name]) for a in node.names)
            elif node.module: modules.append(node.module)
    required = set()
    for module in modules:
        _require(module.split('.')[0] not in ('pxr', 'omni', 'isaacsim', 'greenhouse_sim'), 'Simulator import in supported server route')
        if module == 'sim_data': required.add('__init__.py')
        elif module.startswith('sim_data.'):
            relative = module[len('sim_data.'):].replace('.', '/')
            required.add(relative + ('/__init__.py' if relative == 'native_dataset' else '.py'))
    return required


def code_inventory():
    """Read-only code-pin map for external review; no automatic approval/file."""
    _require(export._file_sha(__file__) == _SOURCE_SHA, 'Wrapper changed since import')
    export._check_code(export._LOADED_CODE)
    names, pins = _code_names(), {}
    for name in sorted(names):
        path = _regular(_ROOT / name)
        raw = read_bounded(path)
        if name.endswith('.py'):
            _require(_imports(name, raw) <= names, 'Unreviewed local server dependency: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def _entry(path, sha):
    path = _regular(path)
    size = path.stat().st_size
    _require(0 < size <= MAX_FILE_BYTES, 'Unsupported source member size')
    return dict(path=path, sha256=export._sha(sha), size=size)


def _info(name, compression):
    info = zipfile.ZipInfo(_name(name), date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = compression
    return info


def _stream(source, target=None, *, size):
    total, digest = 0, hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        total += len(block)
        _require(total <= size, 'Member exceeded pinned size')
        digest.update(block)
        if target is not None: target.write(block)
    _require(total == size, 'Member length changed')
    return digest.hexdigest()


def _write_entry(archive, name, entry, method):
    with _regular(entry['path']).open('rb') as source, archive.open(_info(name, method), 'w', force_zip64=True) as target:
        _require(_stream(source, target, size=entry['size']) == entry['sha256'], 'Source changed while packing: ' + name)


def _read_member(archive, infos, name, limit=MAX_FILE_BYTES):
    _require(name in infos and 0 < infos[name].file_size <= limit, 'Missing/oversized archive metadata: ' + name)
    with archive.open(infos[name]) as stream:
        raw = stream.read(limit + 1)
    _require(0 < len(raw) <= limit, 'Metadata exceeded bound')
    return raw


def _zip_extra(raw):
    """Only ZIP64: no alternate Unicode paths or Unix/link metadata."""
    seen = set()
    while raw:
        _require(len(raw) >= 4, 'Truncated ZIP extra field')
        kind, size = struct.unpack('<HH', raw[:4])
        _require(kind == 1 and kind not in seen and size in (8, 16, 24, 28)
                 and len(raw) >= 4 + size, 'Unsupported/ambiguous ZIP extra field')
        seen.add(kind)
        raw = raw[4 + size:]


def _member_header(stream, info):
    _zip_extra(info.extra)
    stream.seek(info.header_offset)
    header = stream.read(30)
    _require(len(header) == 30, 'Truncated local ZIP header')
    signature, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack('<4s5H3I2H', header)
    _require(signature == b'PK\x03\x04' and flags == info.flag_bits and method == info.compress_type,
             'Conflicting local ZIP header')
    _require(stream.read(name_size) == info.filename.encode('ascii'), 'Conflicting local ZIP name')
    extra = stream.read(extra_size)
    _require(len(extra) == extra_size, 'Truncated local ZIP extra field')
    _zip_extra(extra)


def _protocol(archive, infos, kit):
    """Bind the byte inventory to the original frozen release handoff."""
    _require(set(kit) == {'schema', 'policy', 'compression', 'release_manifest_sha256', 'code_pins_sha256', 'counts', 'files'}
             and kit['schema'] == SCHEMA and kit['policy'] == POLICY and kit['compression'] in _METHODS,
             'Unsupported native transfer schema/policy')
    files = kit['files']
    _require(isinstance(files, dict), 'Explicit member inventory required')
    _names(files)
    _require(set(infos) == set(files) | {KIT_MANIFEST}, 'Missing/undeclared archive member')
    for name, entry in files.items():
        _require(isinstance(entry, dict) and set(entry) == {'sha256', 'size'}
                 and type(entry['size']) is int and 0 < entry['size'] <= MAX_FILE_BYTES, 'Invalid pinned member entry')
        export._sha(entry['sha256'])
        _require(infos[name].file_size == entry['size'], 'Member size differs from inventory')
    _require(all(info.compress_type == _METHODS[kit['compression']] for info in infos.values()), 'Mixed/unbound compression method')
    code_raw = _read_member(archive, infos, 'SERVER_CODE_PINS.json')
    _require(hashlib.sha256(code_raw).hexdigest() == kit['code_pins_sha256'], 'Code-pin document changed')
    pins = export._parse(code_raw)
    _require(isinstance(pins, dict) and set(pins) == _code_names(), 'Exact reviewed recursive code inventory required')
    for name, pin in pins.items():
        _require(files[CODE_PREFIX + name]['sha256'] == export._sha(pin), 'Bundled code pin mismatch')
    _require(files['README_TRAINING.md']['sha256'] == pins[README], 'Native training README changed')
    raw = _read_member(archive, infos, 'release/manifest.json')
    _require(hashlib.sha256(raw).hexdigest() == kit['release_manifest_sha256'], 'Release manifest pin changed')
    release = export._parse(raw)
    _require(release.get('schema') == export.SCHEMA and release.get('state') == 'packaged_external_selection'
             and release.get('policy') == export.POLICY
             and release.get('native_contract_sha256') == export.contract.contract_hash(), 'Native release contract changed')
    _require(release.get('implementation_bindings') == {n: pins[n] for n in export._CODE_NAMES}, 'Release/server source binding mismatch')
    expected = {CODE_PREFIX + n for n in pins} | {'SERVER_CODE_PINS.json', 'README_TRAINING.md', 'release/manifest.json'}
    for name, sha in release['files'].items():
        _name(name)
        expected.add('release/' + name)
        _require(files['release/' + name]['sha256'] == export._sha(sha), 'Release/member pin mismatch')
    _require(set(files) == expected, 'Extra code/data/content in kit')
    documents = {name: _read_member(archive, infos, f'release/provenance/{name}.json')
                 for name in ('selection', 'admission', 'frozen_splits')}
    _require(release['provenance'] == {n: hashlib.sha256(raw).hexdigest() for n, raw in documents.items()}, 'Frozen handoff pins changed')
    _, admission, _ = export._handoff(documents)  # Same >=20k floor/exact map; NO public bypass.
    _require(kit['counts'] == admission['counts'], 'Frozen release counts changed')


def verify(archive, *, archive_sha256):
    """No extraction/execution. Trusted external ZIP pin is mandatory."""
    archive = _regular(archive)
    _require(export._file_sha(archive) == export._sha(archive_sha256), 'External archive SHA256 mismatch')
    with zipfile.ZipFile(archive) as handle, archive.open('rb') as headers:
        members = handle.infolist()
        _names([i.orig_filename for i in members])
        for info in members:
            _require(info.orig_filename == info.filename and not info.is_dir() and not (info.flag_bits & 1)
                     and stat.S_IFMT(info.external_attr >> 16) == stat.S_IFREG
                     and not (info.external_attr & 0x410)
                     and info.compress_type in _METHODS.values(), 'Links/directories/encryption/unsupported ZIP member')
            _member_header(headers, info)
        infos = {i.filename: i for i in members}
        raw = _read_member(handle, infos, KIT_MANIFEST, MAX_KIT_METADATA)
        kit = export._parse(raw)
        _protocol(handle, infos, kit)
        for name, entry in kit['files'].items():
            with handle.open(infos[name]) as stream:
                _require(_stream(stream, size=entry['size']) == entry['sha256'], 'Archived member SHA256 mismatch: ' + name)
    _require(export._file_sha(archive) == archive_sha256, 'Archive changed during verification')
    return dict(schema=SCHEMA, archive_sha256=archive_sha256, kit_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        release_manifest_sha256=kit['release_manifest_sha256'], code_pins_sha256=kit['code_pins_sha256'],
        counts=kit['counts'], members=len(infos), compression=kit['compression'], byte_integrity_verified=True,
        native_semantics_replayed=False, training_approved=False, gpu_qualified=False)


def package(release, destination, *, manifest_sha256, code_pins, compression='stored'):
    """Validate a final native release, stream a NEW kit, verify then publish."""
    _require(compression in _METHODS, 'Choose lossless stored or deflated')
    release = _regular(release, directory=True)
    _require(isinstance(code_pins, export.FilePin), 'Explicit external code FilePin required')
    code_path = _regular(code_pins.path)
    destination = _new_output(destination, (release, _ROOT, code_path))
    pins = export._parse(export._pinned(code_pins))
    _require(pins == code_inventory(), 'Exact current reviewed code pins required')
    validation = export.validate(release, manifest_sha256=manifest_sha256)
    counts = dict(validation['counts'])
    del validation  # Do not retain all materialized chat records from validator.
    raw = read_bounded(release / 'manifest.json')
    _require(hashlib.sha256(raw).hexdigest() == manifest_sha256, 'Manifest changed after native validation')
    manifest = export._parse(raw)
    sources = {'release/' + name: _entry(release / name, sha) for name, sha in manifest['files'].items()}
    sources['release/manifest.json'] = _entry(release / 'manifest.json', manifest_sha256)
    sources.update({CODE_PREFIX + name: _entry(_ROOT / name, sha) for name, sha in pins.items()})
    sources['README_TRAINING.md'] = _entry(_ROOT / README, pins[README])
    sources['SERVER_CODE_PINS.json'] = _entry(code_path, code_pins.sha256)
    _names([*sources, KIT_MANIFEST])
    kit = dict(schema=SCHEMA, policy=POLICY, compression=compression, release_manifest_sha256=manifest_sha256,
        code_pins_sha256=code_pins.sha256, counts=counts,
        files={name: {k: value[k] for k in ('sha256', 'size')} for name, value in sources.items()})
    kit_raw = export._json(kit)
    _require(len(kit_raw) <= MAX_KIT_METADATA, 'Kit metadata exceeded structural bound')
    stage = Path(tempfile.mkdtemp(prefix=destination.name + '.', suffix='.partial', dir=destination.parent))
    partial = stage / 'archive.zip'
    with zipfile.ZipFile(partial, 'x', compression=_METHODS[compression], allowZip64=True) as handle:
        for name in sorted(sources): _write_entry(handle, name, sources[name], _METHODS[compression])
        handle.writestr(_info(KIT_MANIFEST, _METHODS[compression]), kit_raw)
    sha = export._file_sha(partial)
    result = verify(partial, archive_sha256=sha)
    for name, entry in sources.items():
        with _regular(entry['path']).open('rb') as stream:
            _require(_stream(stream, size=entry['size']) == entry['sha256'], 'Source changed before publication: ' + name)
    _require(export._regular_tree(release) == set(manifest['files']) | {'manifest.json'}, 'Source release inventory changed')
    _require(code_inventory() == pins, 'Server code changed before publication')
    _new_output(destination, (release, _ROOT, code_path))
    os.link(partial, destination)  # Atomic exclusive file publication; no overwrite.
    # Delete only our own temporary hard link and empty private staging directory.
    try:
        partial.unlink()
        stage.rmdir()
        remaining_stage = None
    except OSError:
        remaining_stage = str(stage)  # Published ZIP is verified; never recursively clean unexpected contents.
    return dict(result, archive=str(destination), bytes=destination.stat().st_size,
                staging_directory=remaining_stage, native_source_validator_passed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('code-pins')
    pack = sub.add_parser('package')
    for name in ('release', 'output', 'code-pins'): pack.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'code-pins-sha256'): pack.add_argument('--' + name, required=True)
    pack.add_argument('--compression', choices=tuple(_METHODS), default='stored')
    check = sub.add_parser('verify')
    check.add_argument('--archive', type=Path, required=True)
    check.add_argument('--archive-sha256', required=True)
    args = parser.parse_args(argv)
    if args.command == 'code-pins': result = code_inventory()
    elif args.command == 'verify': result = verify(args.archive, archive_sha256=args.archive_sha256)
    else:
        result = package(args.release, args.output, manifest_sha256=args.manifest_sha256,
            code_pins=export.FilePin(str(args.code_pins), args.code_pins_sha256), compression=args.compression)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
