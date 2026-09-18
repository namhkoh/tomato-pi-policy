"""Content-bound plant geometry and scene identities shared by all capture hosts.

File paths and root provenance are authentication metadata, never identity fields.
Component metadata is retained conservatively; this is a content identity, not a
claim that different encodings necessarily depict different biological plants.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np

HEX = re.compile(r'^[0-9a-f]{64}$')
ROOT_PROVENANCE = frozenset({
    'generator', 'version', 'schema', 'schema_version', 'seed', 'plant_id', 'plant_name',
    'source_family', 'source_plant_id', 'donor_source_family', 'donor_family',
    'variant_id', 'morphology_id', 'geometry_source_id', 'split', 'split_group',
    'output', 'output_path', 'output_root', 'manifest_path', 'source_manifest',
    'source_manifest_path', 'source_manifest_sha256', 'source_plan', 'source_plan_path',
    'source_plan_sha256', 'source_bindings', 'provenance', 'created_at', 'timestamp',
    'intended_use', 'leaf_transport_policy', 'physics_supported', 'filename',
})


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def numeric(value, shape):
    a = np.asarray(value, dtype=float)
    require(a.shape == shape and np.isfinite(a).all(), 'Invalid identity matrix')
    a = np.round(a, 9)
    a[a == 0] = 0.0
    return a.tolist()


def rigid_rows(value):
    m = np.asarray(value, dtype=float)
    numeric(m, (4, 4))
    require(np.allclose(m[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            and np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-7, rtol=0)
            and abs(np.linalg.det(m[:3, :3]) - 1) < 1e-7, 'Proper rigid row transform required')
    return m


def geometry_identity(manifest_path, authenticated_hashes, extra_dependency_bindings=None, *, allow_new_texture_bindings=False):
    """Hash canonical geometry from an authenticated manifest and its USD closure.

    authenticated_hashes may be a broader flat path->SHA256 source closure. Every
    manifest and component layers must be pinned and current. Extra texture pins
    may come from the current verified asset package. Previously unpinned texture
    content is measured as an explicitly NEW identity input, never represented as
    historical capture authentication. Returned bindings let callers rehash the exact
    immutable closure at operation completion. No mtime-only cache is used.
    """
    from pxr import UsdUtils
    expected = {}
    for path, sha in authenticated_hashes.items():
        key = path_key(path)
        require(HEX.fullmatch(sha) is not None, 'Invalid authenticated digest')
        require(key not in expected or expected[key] == sha, 'Conflicting path aliases')
        expected[key] = sha
    original_keys = set(expected)
    extra_keys = set()
    for path, sha in (extra_dependency_bindings or {}).items():
        key = path_key(path)
        require(HEX.fullmatch(sha) is not None, 'Invalid extra dependency digest')
        require(key not in expected or expected[key] == sha, 'Conflicting extra dependency pin')
        expected[key] = sha
        extra_keys.add(key)
    used = {}
    new_identity_dependencies = {}
    extra_authenticated_dependencies = {}

    def authenticate(path, texture_asset=False):
        path = Path(path).resolve()
        key = path_key(path)
        if key not in expected:
            require(texture_asset and allow_new_texture_bindings is True, 'Unbound geometry dependency: ' + str(path))
            expected[key] = file_sha256(path)
            new_identity_dependencies[str(path)] = expected[key]
        elif key not in original_keys and key in extra_keys:
            extra_authenticated_dependencies[str(path)] = expected[key]
        if str(path) not in used:
            require(file_sha256(path) == expected[key], 'Changed geometry dependency: ' + str(path))
            used[str(path)] = expected[key]
        return used[str(path)]

    manifest_path = Path(manifest_path).resolve()
    manifest_sha = authenticate(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    components = manifest['components']
    require(isinstance(components, list) and components, 'Nonempty component list required')
    require(manifest['component_count'] == len(components), 'Component count differs')
    ids = {c['id'] for c in components}
    require(len(ids) == len(components), 'Duplicate component IDs')
    rows = []
    for component in components:
        require(component.get('parent') is None or component['parent'] in ids,
                'Component parent absent from complete geometry')
        relative = Path(component['file'])
        require(not relative.is_absolute(), 'Component must resolve within its manifest directory')
        asset = (manifest_path.parent / relative).resolve()
        require(asset.is_relative_to(manifest_path.parent), 'Component escapes its plant directory')
        asset_sha = authenticate(asset)
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(asset))
        require(not unresolved, 'Unresolved authored USD dependencies: ' + str(unresolved))
        layer_shas = []
        for layer in layers:
            require(not layer.anonymous and layer.realPath, 'Anonymous geometry dependency')
            layer_shas.append(authenticate(layer.realPath))
        asset_shas = [authenticate(path, texture_asset=True) for path in assets]
        require(asset_sha in layer_shas, 'Dependency resolver omitted the component layer')
        row = deepcopy(component)
        del row['file']
        row['asset_sha256'] = asset_sha
        row['referenced_layer_content_sha256'] = sorted(set(layer_shas) - {asset_sha})
        row['referenced_asset_content_sha256'] = sorted(set(asset_shas))
        rows.append(row)
    rows.sort(key=lambda row: row['id'])
    root = {k: deepcopy(v) for k, v in manifest.items()
            if k != 'components' and k not in ROOT_PROVENANCE}
    payload = dict(schema='greenhouse.plant_geometry_content_identity.v1',
                   root_geometry_metadata=root, components=rows)
    for path, sha in used.items():
        require(file_sha256(path) == sha, 'Geometry changed while computing identity: ' + path)
    return dict(geometry_sha256=digest(payload), payload=payload,
                source_bindings=used, manifest_sha256=manifest_sha,
                component_count=len(rows), referenced_files=len(used),
                new_identity_dependency_bindings=new_identity_dependencies,
                newly_measured_texture_bindings=new_identity_dependencies,
                extra_authenticated_dependency_bindings=extra_authenticated_dependencies,
                identity_upgrade_provenance='Previously unpinned referenced texture bytes are new current identity inputs with opening/closing hashes, not historical capture proof. Extra dependency pins, when supplied, are also verified.')


def scene_identity(census, geometry_hashes_by_plant_root):
    """Same canonical scene payload for original and donor-derived geometries."""
    slots = census['all_plant_roots']
    roots = {s['plant_root'] for s in slots}
    require(len(slots) == len(roots) == 144 and set(geometry_hashes_by_plant_root) == roots,
            'Exact144 current geometry identities required')
    require(census['complete_active_plant_anatomy'] is True, 'Complete plant anatomy required')
    entries = []
    for slot in slots:
        sha = geometry_hashes_by_plant_root[slot['plant_root']]
        require(isinstance(sha, str) and HEX.fullmatch(sha), 'Invalid current geometry digest')
        entries.append(dict(geometry_sha256=sha, source_family=slot['source_family'],
                            split=slot['source_split'], plant_to_world=numeric(
                                rigid_rows(slot['plant_to_world_usd_row_vectors']), (4, 4))))
    entries.sort(key=canonical)
    return dict(schema='greenhouse.full_plant_scene_geometry_identity.v1',
                dataset_split=census['dataset_split'], plants=entries)


def scene_camera_key(scene, calibration):
    require(calibration['resolution'] == [848, 408] and calibration.get('crop_resize') is None,
            'Native848x408 full frame required')
    return digest(dict(schema='greenhouse.fixed_scene_camera_identity.v1',
                       scene_identity_sha256=digest(scene),
                       camera_to_world=numeric(rigid_rows(calibration['camera_to_world_usd_row_vectors']), (4, 4)),
                       intrinsics=numeric(calibration['intrinsics'], (3, 3)), resolution=[848, 408]))
