"""Propose evidence-insufficient regions from native leaf surfaces, not absence.

RGB/depth are never cropped, edited or inferred. A box scopes the instruction;
the model still receives the original full robot-camera image.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from .dataset_review import read_json, require, safe_file
from .depth_preview import sha256


def leaf_region(components, catalogue, depth, valid, *, size=48):
    require(components.shape == (408, 848) and np.issubdtype(components.dtype, np.integer), 'Invalid native IDs')
    require(depth.shape == components.shape and depth.dtype == np.float32 and valid.shape == depth.shape
            and valid.dtype == bool, 'Expected native optical-Z and validity')
    require(type(size) is int and 32 <= size <= 96 and size % 2 == 0, 'Invalid region size')
    leaves = [c for c in catalogue if c['organ_type'] == 'leaf']
    candidates = sorted(leaves, key=lambda c: (-int(np.count_nonzero(components == c['component_index'])), c['component_index']))
    for leaf in candidates[:12]:
        mask = (components == leaf['component_index']) & valid & np.isfinite(depth) & (depth > 0)
        interior = distance_transform_edt(mask)
        half = size//2
        interior[:half+16] = 0; interior[-half-16:] = 0
        interior[:, :half+16] = 0; interior[:, -half-16:] = 0
        for _ in range(8):
            y, x = np.unravel_index(np.argmax(interior), interior.shape)
            if interior[y, x] < half: break
            box = [int(x-half), int(y-half), int(x+half), int(y+half)]
            roi = np.s_[box[1]:box[3], box[0]:box[2]]
            if mask[roi].all():
                return dict(region_xyxy=box, foreground_component=leaf, native_leaf_fraction=1.,
                    target_eligibility='unknown', hidden_target_existence='not_inferred',
                    no_target_evidence=False, visual_review_required=True)
            interior[max(0,y-8):y+9,max(0,x-8):x+9] = 0
    return None


def propose(directory, metadata, audit, *, size=48):
    directory = Path(directory)
    paths = {}
    for name in ('supervision/component_id.npy', 'supervision/identities.json', 'inputs/depth_m.npy', 'inputs/depth_valid.png'):
        path = safe_file(directory, name); expected = metadata['files'][name]['sha256']
        require(sha256(path) == expected and audit['bindings_sha256'].get(str(path)) == expected,
                'Region evidence is not independently bound native data')
        paths[str(path)] = expected
    from PIL import Image
    result = leaf_region(np.load(directory/'supervision/component_id.npy',allow_pickle=False),
        read_json(directory/'supervision/identities.json')['component_catalogue'],
        np.load(directory/'inputs/depth_m.npy',allow_pickle=False),
        np.asarray(Image.open(directory/'inputs/depth_valid.png')) == 255, size=size)
    if result: result['bindings'] = paths
    return result
