"""Explicit local data roots for future diverse collection; code/plans remain D."""
from pathlib import Path
import hashlib

WORKSPACE=Path('D:/research/tomato-pi-policy')
LEGACY_DATA=WORKSPACE/'data/sim_data'
FUTURE_DATA=Path('C:/Users/USER/tomato-vlm-data-20260917')
DATA_ROOTS=(LEGACY_DATA,FUTURE_DATA)
DIAGNOSTIC_ROOTS=tuple(p/'diagnostics' for p in DATA_ROOTS)
REVIEW_ROOTS=tuple(p/'dataset_reviews' for p in DATA_ROOTS)
CHECKPOINT_ROOTS=tuple(p/'dataset_checkpoints' for p in DATA_ROOTS)
FRESH_NATIVE_ROOTS=(LEGACY_DATA/'diagnostics/native848_diverse_noon_native_20260917_v1',
                    FUTURE_DATA/'diagnostics/native848_diverse_native_20260917_v1')
SCHEMA='greenhouse.native848_explicit_data_roots.v1'


def resolve_data(path,roots=DATA_ROOTS):
    path=Path(path)
    if not path.is_absolute():raise ValueError('Absolute data path required')
    resolved=path.resolve()
    if not any(root.resolve()==root and resolved.is_relative_to(root) for root in roots):
        raise ValueError('Data path escapes explicit local roots: '+str(path))
    return resolved


def checkpoint_output(path):
    resolved=resolve_data(path,CHECKPOINT_ROOTS)
    if resolved.parent not in CHECKPOINT_ROOTS or resolved.exists():
        raise ValueError('New direct-child cumulative checkpoint required')
    return resolved


def fresh_native(path):
    return resolve_data(path,FRESH_NATIVE_ROOTS)


def diagnostic(path):
    return resolve_data(path,DIAGNOSTIC_ROOTS)

ROOT_MANIFEST=FUTURE_DATA/'dataset_root.json'
ROOT_MANIFEST_SHA256='65c9d79a0993d698524ed2ee61985b276d7fd1838a4258786d9836d1b4881716'


def resolve_evidence(path):
    return resolve_data(path,(WORKSPACE,FUTURE_DATA))


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def root_manifest():
    path=resolve_data(ROOT_MANIFEST)
    if not path.is_file() or _digest(path)!=ROOT_MANIFEST_SHA256:
        raise ValueError('Future dataset root marker differs')
    return dict(path=str(path),sha256=ROOT_MANIFEST_SHA256)


def bindings():
    marker=root_manifest();module=Path(__file__).resolve()
    return {str(module):_digest(module),marker['path']:marker['sha256']}
