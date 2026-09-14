"""Explicit schema dispatch; legacy profiles keep their existing validator."""
from .dataset_review import read_json
from pathlib import Path


def validate(root, *, progress=None):
    from .clear_cutpoint_contract import SCHEMA
    if read_json(Path(root)/'manifest.json').get('schema_version') == SCHEMA:
        from .clear_cutpoint_release import validate as check
    else:
        from .training_export import validate as check
    return check(root,progress=progress)
