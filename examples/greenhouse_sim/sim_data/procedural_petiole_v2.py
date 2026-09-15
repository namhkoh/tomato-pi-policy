"""Create-only curved/relocated plant variants using original detailed mesh assets.

New centerlines/attachments; rigid leaf blades preserve all source dimensions.
Versioned driver intentionally retains v1 helpers unchanged for frozen assets.
Static inspection only: no source edits, no release novelty or physics approval.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .audit import audit_manifest, safe_asset
from .cut_regions import propose_cut_region, load_rule
from .plant_variants import (load_training_sources, training_envelope,
    PHYSICS_FIELDS, file_hash, digest)
from .plant_variant_usd import copy_component, cut_surface_probe, write_json_new
from .procedural_petiole_geometry import require

VERSION='curved_relocated_rigid_leaf_static.v2'


# The v1 implementation remains byte-identical for frozen v1 receipts. This
# versioned writer shares its geometry/check helpers but explicitly changes the
# per-component deformation policy; no runtime monkeypatching of old modules.
from .procedural_petiole_usd import (plan_change as _plan_change,
    transform_metadata as _transform_metadata, deform_new_copy)
from .procedural_leaf_transport import component_transport


def plan_change(source, key, envelope, seed):
    change = _plan_change(source, key, envelope, seed)
    components = source['report']['components']
    require(all(m == key or (components[m]['type'] == 'leaf' and components[m]['parent'] == key)
                for m in change['members']), 'V2 supports only direct leaves on the target petiole')
    return change


def transform_metadata(raw, change):
    field = component_transport(raw, change)
    return _transform_metadata(raw, dict(change, warp=field))


def generate(plan_path,family,targets,seed,output):
    from pxr import Usd
    output=Path(output).resolve();require(not output.exists(),'New output only')
    plan,sources=load_training_sources(plan_path)
    require(family in sources,'Frozen TRAIN family required')
    source=sources[family];envelope=training_envelope(plan,sources)
    require(targets and len(set(targets))==len(targets),'Explicit unique targets required')
    allowed={t['component_id'] for t in source['job']['targets']}
    require(set(targets)<=allowed,'Target outside frozen source catalogue')
    changes=[plan_change(source,key,envelope,seed+i*104729) for i,key in enumerate(targets)]
    owners={}
    for change in changes:
        for key in change['members']:
            require(key not in owners,'Overlapping target subtrees');owners[key]=change
    source_path=Path(source['report']['manifest_path'])
    require(not output.is_relative_to(source_path.parent),'No writes inside source assets')
    output.mkdir(parents=True)
    try:
        raw=deepcopy(source['raw'])
        raw.update(generator=VERSION,version='2.0.0',seed=seed,physics_supported=False,
                   leaf_transport_policy='rigid_leaf_blade.v1',
                   intended_use='curved_relocated_static_geometry_pending_native_qualification')
        for k in PHYSICS_FIELDS:raw.pop(k,None)
        rows=[];textures={};output_hashes={};geometry=[]
        for old in source['raw']['components']:
            change=owners.get(old['id'])
            row=transform_metadata(old,change) if change else deepcopy(old)
            for k in PHYSICS_FIELDS:row.pop(k,None)
            destination=safe_asset(output,old['file']);destination.parent.mkdir(parents=True,exist_ok=True)
            copy_component(safe_asset(source_path.parent,old['file']),destination,np.eye(3),1.,source_path.parent,textures)
            if change:
                evidence=deform_new_copy(destination,np.asarray(old['transform']['translate']),
                    np.asarray(row['transform']['translate']),component_transport(old,change))
                geometry.extend(dict(component_id=old['id'],**q) for q in evidence)
            rows.append(row);output_hashes[old['file']]=file_hash(destination)
        raw['components']=rows;raw['component_count']=len(rows)
        write_json_new(output/'manifest.json',raw)
        report=audit_manifest(output/'manifest.json')
        require(report['status']!='blocked','Generated manifest structure failed')
        rules=load_rule();targets_out=[]
        for change in changes:
            key=change['component_id'];component=report['components'][key];parent=report['components'][component['parent']]
            proposal=propose_cut_region(component,parent,rules)
            require(proposal['status']=='proposed_geometry_only' and not proposal['geometry_warnings'],
                    'Generated cut interval lacks conservative parent clearance')
            stage=Usd.Stage.Open(str(output/component['file']))
            surface=cut_surface_probe(stage,component,parent)
            require(surface['passed'],'New centerline cut does not lie within the actual deformed surface')
            targets_out.append(dict(component_id=key,source_target_id=change['source_target_id'],
                cut_region_proposal=proposal,cut_surface_probe=surface,
                conservative_view_cap_group=change['source_target_id'],
                # Until a separate global shape/near-duplicate qualification exists,
                # this generator does not reset the original target's view budget.
                shape_novelty_pending=True,training_eligible=False))
        for path,sha in textures.items():output_hashes[path.relative_to(output).as_posix()]=sha
        output_hashes['manifest.json']=file_hash(output/'manifest.json')
        recipe=[{k:v for k,v in c.items() if k not in ('curve','warp')} for c in changes]
        source_bindings={str(source_path):file_hash(source_path)}
        for c in source['report']['components'].values():
            path=safe_asset(source_path.parent,c['file'])
            require(file_hash(path)==c['asset_sha256'],'Source changed while generating')
            source_bindings[str(path)]=c['asset_sha256']
        for relative in output_hashes:
            if relative.endswith(('.png','.jpg','.jpeg')):
                path=safe_asset(source_path.parent,relative)
                require(file_hash(path)==output_hashes[relative],'Source texture changed')
                source_bindings[str(path)]=output_hashes[relative]
        receipt=dict(version=VERSION,leaf_transport_policy='rigid_leaf_blade.v1',state='cpu_curved_rigid_leaf_pending_native_review',source_family=family,
            split='train',split_group=family,variant_id=output.name,source_plan_path=str(Path(plan_path).resolve()),
            source_plan_sha256=file_hash(plan_path),frozen_family_assignments=plan['family_assignments'],
            source_bindings=source_bindings,source_manifest_path=str(source_path),
            output_hashes=output_hashes,recipes=recipe,targets=targets_out,mesh_derivative_diagnostics=geometry,
            training_envelope_sha256=digest(envelope),source_assets_unchanged=True,
            training_eligible=False,native_capture_validated=False,physics_validated=False,
            independent_target_novelty_approved=False,new_donor_family_created=False,
            code_sha256={p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('procedural_petiole_geometry.py'),
                Path(__file__).with_name('procedural_petiole_warp.py'),Path(__file__).with_name('plant_variant_usd.py'),
                Path(__file__).with_name('procedural_petiole_usd.py'),Path(__file__).with_name('procedural_leaf_transport.py'),
                Path(__file__).with_name('procedural_petiole_catalogue.py'),
                Path(__file__).with_name('plant_variants.py'),Path(__file__).with_name('audit.py'),Path(__file__).with_name('cut_regions.py')]})
        write_json_new(output/'qualification.json',receipt)
        print('CURVED_PLANT_CPU_QUALIFIED',output.name,len(targets_out),flush=True)
        return receipt
    except BaseException as exc:
        write_json_new(output/'FAILED.json',dict(error=type(exc).__name__+': '+str(exc),training_eligible=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan',type=Path,required=True);p.add_argument('--family',required=True)
    p.add_argument('--targets',nargs='+',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.source_plan,a.family,a.targets,a.seed,a.output)


if __name__=='__main__':main()
