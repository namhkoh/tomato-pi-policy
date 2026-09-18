"""Complete per-instance anatomy with indexed native ownership and unchanged labels.

Clones keep the original source family. Every catalogue petiole is enumerated.
Only the existing zero-authenticated-pixel exclusion is accelerated; visible
targets use the frozen 9mm evaluator, including its workspace and clarity gates.
No whole-plant reach shortcut, source edits, native launch, or acceptance.
"""
from pathlib import Path
import numpy as np
from .dataset_review import require
from .capture_visibility import ORGAN_IDS
from . import native848_all_petiole_9mm_v1 as labels

SCHEMA = 'greenhouse.native848_fully_labeled_coverage.v1'


def _unique(rows, key, description):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'Duplicate ' + description)
    return result


def component_catalogue(stage, records, reports, variants):
    """Resolve source reports by family; variant IDs identify individual plants."""
    by_family = _unique(reports, 'plant_id', 'anatomical source family')
    by_root = _unique(variants, 'plant_root', 'plant root')
    _unique(variants, 'variant_id', 'plant instance ID')
    record_by_root = _unique(records, 'plant_root', 'plant record')
    require(set(by_root) == set(record_by_root), 'Incomplete plant records/variants')
    items = []
    for root, record in sorted(record_by_root.items()):
        variant = by_root[root]
        family = variant['source_plant_id']
        require(family in by_family and variant['split_group'] == family,
                'Clone must retain its original anatomical source family')
        require(isinstance(variant['variant_id'], str) and variant['variant_id']
                and '/' not in variant['variant_id'], 'Single-segment unique instance ID required')
        require(not variant.get('added_components') and not variant.get('added_component_paths')
                and variant.get('source_geometry_modified') is False, 'Unmodified complete source plant required')
        report = by_family[family]
        require(Path(record['manifest_path']).resolve() == Path(report['manifest_path']).resolve(),
                'Instance record uses another source manifest')
        # Hash once per source family at the scene/plan boundary, not per clone.
        components = report['components']
        paths = record['component_paths']
        require(set(paths) == set(components), 'Incomplete component paths')
        for key, component in components.items():
            chain = [key]
            parent = component['parent']
            while parent is not None:
                require(parent in components and parent not in chain, 'Invalid anatomical ancestry')
                chain.append(parent)
                parent = components[parent]['parent']
            expected = root + '/' + '/'.join(reversed(chain))
            require(paths[key] == expected, 'Component path differs from original anatomical hierarchy')
            prim = stage.GetPrimAtPath(expected)
            require(bool(prim) and prim.IsActive(), 'Missing/inactive component is not complete anatomy')
            items.append(dict(component_id=key, prim_path=expected, organ_type=component['type'],
                              variant_id=variant['variant_id'], source_plant_id=family, split_group=family))
    result = [dict(item, component_index=i) for i, item in
              enumerate(sorted(items, key=lambda row: row['prim_path']), 1)]
    CatalogueIndex(result)
    return result


class CatalogueIndex:
    """Exact deepest-ancestor ownership; no substring or source-family matching."""
    def __init__(self, catalogue):
        self.catalogue = list(catalogue)
        self.by_path = _unique(self.catalogue, 'prim_path', 'component prim path')
        self.by_identity = {}
        self.by_variant = {}
        indices = []
        for row in self.catalogue:
            path = row['prim_path']
            require(isinstance(path, str) and path.startswith('/') and not path.endswith('/')
                    and '//' not in path, 'Absolute canonical component path required')
            key = (row['variant_id'], row['component_id'])
            require(key not in self.by_identity, 'Duplicate per-instance component identity')
            self.by_identity[key] = row
            self.by_variant.setdefault(row['variant_id'], []).append(row)
            index = row['component_index']
            require(type(index) is int and 0 < index <= np.iinfo(np.uint32).max, 'Invalid component index')
            indices.append(index)
        require(sorted(indices) == list(range(1, len(indices)+1)), 'Complete contiguous component index required')

    def owner(self, path):
        require(isinstance(path, str), 'Renderer identity must be a string path')
        while path:
            if path in self.by_path:
                return self.by_path[path]
            path = path.rsplit('/', 1)[0]
        return None


def component_masks(instances, mapping, catalogue):
    """Same output as capture_visibility.component_masks without catalogue scans."""
    require(isinstance(instances, np.ndarray) and instances.dtype == np.uint32,
            'Native uncolorized uint32 renderer identities required')
    index = catalogue if isinstance(catalogue, CatalogueIndex) else CatalogueIndex(catalogue)
    identifiers, inverse = np.unique(instances, return_inverse=True)
    component_values = np.zeros(len(identifiers), np.uint32)
    organ_values = np.zeros(len(identifiers), np.uint8)
    owners = {}
    for i, rid in enumerate(identifiers):
        rid = int(rid)
        owner = index.owner(mapping.get(rid, ''))
        owners[rid] = owner
        if owner is not None:
            component_values[i] = owner['component_index']
            organ_values[i] = ORGAN_IDS.get(owner['organ_type'], ORGAN_IDS['other_component'])
    return (component_values[inverse].reshape(instances.shape),
            organ_values[inverse].reshape(instances.shape), owners)


def target_inventory(context, reports, catalogue):
    """Enumerate every instance petiole, retaining source-family target lineage."""
    reports = _unique(reports, 'plant_id', 'anatomical source family') if isinstance(reports, list) else reports
    census = context['full_scene_census']
    roots = _unique(census['all_plant_roots'], 'plant_root', 'census root')
    variants = _unique(context['scene_variants'], 'variant_id', 'plant instance ID')
    require(len(roots) == 144 and all(r['active_after_policy'] is True for r in roots.values()),
            'All144 original planting slots must remain active')
    index = CatalogueIndex(catalogue)
    require(set(index.by_variant) == set(variants), 'Catalogue omits a plant instance')
    entries = []
    for row in catalogue:
        variant = variants[row['variant_id']]
        root = roots[variant['plant_root']]
        family = row['source_plant_id']
        require(family == variant['source_plant_id'] == variant['split_group'] == root['source_family']
                and root['source_split'] == context['dataset_split'] and root['authenticated_component_plant'] is True,
                'Unauthenticated or cross-split cloned anatomy')
        require(root['component_count'] == len(index.by_variant[row['variant_id']]), 'Incomplete instance catalogue')
        if row['organ_type'] != 'sub_stem':
            continue
        report = reports[family]
        matches = [t for t in report['targets'] if t['component_id'] == row['component_id']]
        require(len(matches) == 1, 'Unique source anatomical target required')
        target = matches[0]
        component = report['components'][row['component_id']]
        leaves = [key for key in target['expected_detached_component_ids'] if report['components'][key]['type'] == 'leaf']
        semantic = (component.get('deleafed') is False
                    and report['components'].get(component['parent'], {}).get('type') == 'main_stem'
                    and bool(leaves) and not target['protected_descendant_ids'])
        entries.append(dict(target_id=row['variant_id']+'/'+row['component_id'], source_family=family,
                            source_target_id=family+'/'+row['component_id'], source_component_id=row['component_id'], report=report,
                            plant_to_world_usd_row_vectors=root['plant_to_world_usd_row_vectors'],
                            semantic_leaf_petiole_candidate=bool(semantic), anatomy_reason_codes=target['reason_codes']))
    return entries


def _zero_pixel_row(entry):
    # Exactly the pre-existing evaluator result at its first exclusion branch.
    return dict(target_id=entry['target_id'], source_family=entry['source_family'], status='excluded',
                reason='not_visible_zero_authenticated_component_pixels', annotation_epoch=labels.ANNOTATION_EPOCH,
                nominal_arc_m=labels.NOMINAL_ARC_M, visibility_support_arc_m=list(labels.SUPPORT_ARC_M),
                support_is_permissible_cut_interval=False, actual_visual_review=False,
                automated_pass=False, training_approved=False)


def evaluate_frame(metadata, rgb, depth, valid, components, catalogue, target_inventory, workspace_checker, *, scene_coverage=None):
    """Equivalent labels with all11k petioles, no relaxed geometry or visibility."""
    index = CatalogueIndex(catalogue)
    # Run frozen native shape/calibration/static-frame validation and census construction.
    result = labels.evaluate_frame(metadata, rgb, depth, valid, components, catalogue, [], workspace_checker,
                                   scene_coverage=scene_coverage)
    expected = set(result['target_census']['catalogue_petiole_ids'])
    entries = _unique(target_inventory, 'target_id', 'inventory target')
    require(set(entries) <= expected, 'Foreign inventory target')
    visible = set(map(int, np.unique(components)))
    rows = []
    for original in result['targets']:
        target_id = original['target_id']
        if target_id not in entries:
            rows.append(original)
            continue
        entry = entries[target_id]
        try:
            instance, key = target_id.split('/')
            target = index.by_identity[(instance, key)]
            require(entry['report']['plant_id'] == entry['source_family'] == target['source_plant_id'] == target['split_group'], 'Report/source family differs')
            require(target['organ_type'] == 'sub_stem', 'Exact petiole catalogue identity required')
            if target['component_index'] not in visible:
                rows.append(_zero_pixel_row(entry))
            else:
                rows.append(labels.evaluate_target(metadata, rgb, depth, valid, components,
                            index.by_variant[instance], entry, workspace_checker))
        except (ValueError, KeyError, AssertionError) as exc:
            rows.append(dict(target_id=target_id, status='unknown', reason='invalid_target_evidence: '+str(exc),
                             automated_pass=False, actual_visual_review=False, training_approved=False))
    for row in rows:
        instance, component_id = row['target_id'].split('/')
        source = index.by_identity[(instance, component_id)]['source_plant_id']
        row.update(source_family=source, source_component_id=component_id,
                   source_target_id=source+'/'+component_id, plant_instance_id=instance)
    result['targets'] = rows
    census = result['target_census']
    census.update(catalogue_complete=set(entries) == expected,
                  candidate_target_ids=[r['target_id'] for r in rows if r['status'] == 'candidate_pending_visual_review'],
                  excluded_target_ids=[r['target_id'] for r in rows if r['status'] == 'excluded'],
                  unknown_target_ids=[r['target_id'] for r in rows if r['status'] == 'unknown'])
    result['frame_blocked_by_unknown_targets'] = bool(census['unknown_target_ids'])
    return result
