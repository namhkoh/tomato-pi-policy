"""Offline visible-background measurements. Ranking only; never acceptance.

Inputs must be authenticated by the caller. Component ownership uses exact path
ancestors. Coarse groups prove plant pixels, never individual organ pixels.
"""
from collections import Counter
import numpy as np

SCHEMA = 'greenhouse.actual_background_pixel_score.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _inside(path, root):
    return path == root or path.startswith(root + '/')


def score(ids, mapping_value, catalogue, foreground_variants, *, depth, valid,
          coarse_groups=(), tile_rows=4, tile_columns=8):
    require(ids.dtype == np.uint32 and ids.ndim == 2, 'Native uint32 ID image required')
    require(depth.shape == valid.shape == ids.shape and valid.dtype == np.bool_, 'Aligned native buffers required')
    paths = {c['prim_path']: c for c in catalogue}
    indices = {c['component_index']: c for c in catalogue}
    require(len(paths) == len(indices) == len(catalogue), 'Duplicate component ownership')
    fg = set(foreground_variants)
    require(fg and fg <= {c['variant_id'] for c in catalogue}, 'Known foreground instance required')
    groups = {}
    selected = set()
    for g in coarse_groups:
        members = g['member_component_indices']
        require(members and set(members) <= indices.keys(), 'Unknown coarse members')
        variants = {indices[k]['variant_id'] for k in members}
        require(len(variants) == 1 and not variants & fg, 'Coarse group must be one background plant')
        require(not selected.intersection(members) and g['prim_path'] not in groups, 'Duplicate coarse ownership')
        groups[g['prim_path']] = (g, next(iter(variants)))
        selected.update(members)
    mapping = {int(k): v for k, v in mapping_value['renderer_id_to_prim'].items()}
    keys, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    background = np.zeros(len(keys), bool); leaves = background.copy()
    categories = Counter(); plants = Counter(); unknown = []
    for k, rid0 in enumerate(keys):
        rid = int(rid0); n = int(counts[k]); path = mapping.get(rid, '')
        require(isinstance(path, str), 'Renderer path must be text')
        p = path; owner = None
        while p:
            if p in paths:
                owner = paths[p]; break
            p = p.rsplit('/', 1)[0]
        if path in groups:
            g, variant = groups[path]
            saved = mapping_value.get('plant_ID_ownership', {}).get(str(rid), {})
            require(saved.get('kind') == 'coarse_far_group'
                    and saved.get('prim_path') == path
                    and saved.get('member_component_indices') == g['member_component_indices'],
                    'Coarse mapping membership mismatch')
            category = 'background_plant_coarse'; background[k] = True; plants[variant] += n
        elif owner is not None and owner['component_index'] not in selected:
            if owner['variant_id'] in fg:
                category = 'foreground_plant'
            else:
                category = 'background_plant_exact'; background[k] = True
                leaves[k] = owner['organ_type'] == 'leaf'; plants[owner['variant_id']] += n
        elif rid == 0 and not path and np.all(~valid[ids == rid]) and np.all(np.isposinf(depth[ids == rid])):
            category = 'empty_background'
        elif _inside(path, '/World/RBY1'):
            category = 'robot'
        elif any(_inside(path, x) for x in ('/World/Floor', '/World/Ground')):
            category = 'floor_or_ground'
        elif _inside(path, '/World/Gutters'):
            category = 'gutters'
        elif _inside(path, '/World/Environment/GreenHouse'):
            category = 'greenhouse_structure'
        else:
            category = 'unmapped' if not path else 'mapped_unknown'
            unknown.append(dict(renderer_id=rid, prim_path=path, pixels=n))
        categories[category] += n
    mask = background[inv].reshape(ids.shape); leaf = leaves[inv].reshape(ids.shape)
    h, w = ids.shape; fractions = []; occupied = 0
    for ys in np.array_split(np.arange(h), tile_rows):
        row = []
        for xs in np.array_split(np.arange(w), tile_columns):
            value = float(mask[np.ix_(ys, xs)].mean()); row.append(value); occupied += value > 0
        fractions.append(row)
    yy, xx = np.nonzero(mask); spread = None
    if len(xx):
        spread = dict(centroid_uv=[float(xx.mean()), float(yy.mean())],
                      standard_deviation_uv=[float(xx.std()), float(yy.std())],
                      bounds_xyxy=[int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                      horizontal_span_fraction=float((xx.max()-xx.min()+1)/w),
                      vertical_span_fraction=float((yy.max()-yy.min()+1)/h))
    n = ids.size
    return dict(schema=SCHEMA, width=w, height=h, foreground_variants=sorted(fg),
                background_plant_pixel_fraction=float(mask.mean()),
                background_leaf_pixel_fraction_lower_bound=float(leaf.mean()),
                coarse_organ_identity_unavailable=bool(groups),
                visible_background_instance_count=len(plants), background_instance_pixels=dict(plants),
                pixel_categories=dict(categories), pixel_category_fractions={k:v/n for k,v in categories.items()},
                unknown_unmapped_fraction=(categories['unmapped']+categories['mapped_unknown'])/n,
                unknowns=unknown, tiles=dict(rows=tile_rows, columns=tile_columns,
                    occupied_any_plant_pixels=int(occupied), total=tile_rows*tile_columns,
                    background_fractions=fractions, occupancy_is_not_quality_threshold=True),
                spatial_spread=spread, automatic_acceptance=False)


def maximin_views(records, count=4):
    """First highest coverage, then maximin actual optical angle, coverage tie-break.

    Caller supplies one compatible camera group. No invented views or minimum
    angle threshold; actual pair distances are returned for review.
    """
    require(0 < count <= 4, 'Finite 1-4 selection required')
    pending = list(records); chosen = []
    def density(r):
        s = r['score']
        return (s['background_plant_pixel_fraction'], s['background_leaf_pixel_fraction_lower_bound'],
                s['tiles']['occupied_any_plant_pixels'], r['frame_id'])
    def angle(a,b):
        x=-np.asarray(a['camera_to_world'])[2,:3];y=-np.asarray(b['camera_to_world'])[2,:3]
        return float(np.degrees(np.arccos(np.clip(x@y/(np.linalg.norm(x)*np.linalg.norm(y)),-1,1))))
    while pending and len(chosen) < count:
        best = max(pending, key=lambda r: ((min(angle(r,c) for c in chosen),) if chosen else ()) + density(r))
        chosen.append(best); pending.remove(best)
    return dict(selected_frame_ids=[r['frame_id'] for r in chosen], ranking_only=True,
                method='highest_background_fraction_then_maximin_optical_angle_density_tiebreak',
                pairs=[dict(frame_ids=[a['frame_id'],b['frame_id']],optical_axis_degrees=angle(a,b),
                    origin_distance_m=float(np.linalg.norm(np.asarray(a['camera_to_world'])[3,:3]-np.asarray(b['camera_to_world'])[3,:3])))
                    for i,a in enumerate(chosen) for b in chosen[i+1:]])
