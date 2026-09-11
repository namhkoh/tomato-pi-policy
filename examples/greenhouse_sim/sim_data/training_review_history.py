"""Preserve negative image decisions across task versions; never grant approval."""
from pathlib import Path

from .dataset_review import read_json,require,safe_file
from .depth_preview import sha256

SCHEMAS=('greenhouse.grounding_stratified_visual_QA.v1',
         'greenhouse.grounding_stratified_visual_QA.v2')


def negative_history(bundles):
    from .training_export import review_directory_snapshot,verify_review_directories
    records=[];bindings={};directories={}
    for name in bundles:
        path=Path(name).resolve();bundle=read_json(path);digest=sha256(path)
        require(bundle.get('schema_version') in SCHEMAS,'Unsupported historical review schema')
        bindings[str(path)]=digest;entries={e['id']:e for e in bundle['entries']}
        for folder in (path.parent/'decisions',path.parent/'human_decisions'):
            snapshot=review_directory_snapshot(folder);directories[str(folder)]=snapshot
            for filename,file_hash in snapshot.items():
                decision=read_json(filename);entry=decision['entry']
                require(decision['schema_version']==bundle['schema_version'] and
                    decision['bundle_sha256']==digest and entry==entries.get(entry['id']),
                    'Changed historical review binding')
                require(Path(filename).name==entry['id']+'.json' and
                    decision['reviewer_role'] in ('assistant','human') and
                    decision['decision'] in ('accept','hold','reject') and
                    decision['human_confirmation'] is (decision['reviewer_role']=='human') and
                    decision['physical_execution_approved'] is False,'Invalid historical review scope')
                bindings[filename]=file_hash
                if decision['decision'] not in ('hold','reject'): continue
                card=safe_file(path.parent,entry['card'])
                require(sha256(card)==entry['card_sha256'],'Changed historical negative card')
                bindings[str(card)]=entry['card_sha256']
                records.append(dict(decision=decision,source_decision=filename,source_sha256=file_hash))
    verify_review_directories(directories)
    return dict(schema_version='greenhouse.negative_review_history.v1',records=records,
        bundle_paths=[str(Path(p).resolve()) for p in bundles],
        source_evidence_sha256=bindings,review_directory_snapshots=directories,
        grants_approval=False,negative_rgb_identity_applies_across_task_versions=True)


def pinned_history(path,bundles):
    """Require a separately recorded inventory before beginning the source scan."""
    from .dataset_review import verify_bindings
    from .training_export import verify_review_directories
    require(path is not None,'Complete baseline requires a pinned historical review inventory')
    path=Path(path).resolve();inventory=read_json(path)
    require(inventory.get('schema_version')=='greenhouse.negative_review_history.v1' and
        inventory.get('bundle_paths') and inventory.get('source_evidence_sha256') and
        inventory.get('review_directory_snapshots'),'Empty historical review inventory')
    expected=inventory['bundle_paths'];actual=[str(Path(p).resolve()) for p in bundles]
    require(len(actual)==len(set(actual)) and sorted(actual)==sorted(expected),
        'Historical review bundle omitted or added relative to pinned inventory')
    verify_bindings(inventory['source_evidence_sha256'])
    verify_review_directories(inventory['review_directory_snapshots'])
    # Reconstruct records as well; an inventory cannot invent or drop negatives.
    require(negative_history(bundles)==inventory,'Historical inventory contents changed')
    inventory['source_evidence_sha256'][str(path)]=sha256(path)
    return inventory


def exclude_negatives(rows,evidence):
    require(evidence.get('schema_version')=='greenhouse.negative_review_history.v1' and
        evidence.get('grants_approval') is False and
        evidence.get('negative_rgb_identity_applies_across_task_versions') is True,
        'Invalid negative history scope')
    negatives={}
    for record in evidence['records']:
        decision=record['decision']
        require(decision['decision'] in ('hold','reject') and decision['schema_version'] in SCHEMAS,
            'Negative history cannot grant approval')
        negatives.setdefault(decision['entry']['rgb_sha256'],[]).append(record['source_sha256'])
    kept=[];excluded=[]
    for row in rows:
        if row['rgb_sha256'] not in negatives: kept.append(row)
        else: excluded.append(dict(id=row['id'],family=row['source_plant_family'],
            reason='historical_visual_hold_or_reject_rgb',rgb_sha256=row['rgb_sha256'],
            negative_decisions_sha256=negatives[row['rgb_sha256']]))
    return kept,excluded
