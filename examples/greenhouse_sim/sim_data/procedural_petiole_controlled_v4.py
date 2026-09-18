"""Explicit signed, directional controls on disjoint petiole/leaf subtrees.

All modified proximal30mm source geometry remains unchanged. The donor family,
TRAIN split, organ graph and original cap group remain authoritative. CPU replay
is not native visibility, morphology novelty or dataset acceptance.
"""
from copy import deepcopy
from pathlib import Path
import json
import math
import traceback
import numpy as np
from .audit import audit_manifest,descendants,safe_asset
from .cut_regions import load_rule,propose_cut_region
from .plant_variants import (PHYSICS_FIELDS,digest,file_hash,load_training_sources,
    training_envelope,parent_tangent,dimensions,normalized)
from .plant_variant_usd import cut_surface_probe,write_json_new
from .procedural_petiole_geometry import require,unit
from .procedural_petiole_usd import donor_curve,deform_new_copy
from .procedural_proximal_shear_v1 import (mesh_guard,controlled_curve,
    transform_metadata,component_transport,proximal_mesh_identity)
from .procedural_distal_deformation_v1 import SignedProximalShear,rotated_source_direction
from .procedural_petiole_catalogue import check_component
from . import procedural_petiole_controlled_v3 as frozen_v3

VERSION='source_frame_multisubtree_signed_control_static.v4'
SCHEMA='greenhouse.multisubtree_controlled_qualification.v4'
CONTROL_SCHEMA='greenhouse.source_frame_signed_directional_control.v4'
STATE='cpu_multisubtree_signed_control_pending_native_review'


def control_for(amplitude_m,direction_angle_degrees=0.):
    require(type(amplitude_m) in (int,float) and math.isfinite(amplitude_m) and amplitude_m!=0.,
        'Explicit finite nonzero signed amplitude required')
    require(type(direction_angle_degrees) in (int,float) and math.isfinite(direction_angle_degrees)
        and -180.<=direction_angle_degrees<=180.,'Finite canonical source-frame angle required')
    return dict(schema=CONTROL_SCHEMA,amplitude_m=float(amplitude_m),
        direction_angle_degrees=float(direction_angle_degrees),
        direction_policy='source_parent_cross_tangent_rotated_about_proximal_tangent',
        taper_policy='exact_proximal_faces_then_smoothstep_to60mm_transverse_shear',
        source_attachment_preserved=True,sampled_source_radii_preserved=True,
        seed_search=False,amplitude_search=False,visual_label=None)


def plan_change_explicit(source, key, envelope, control):
    require(control == control_for(control['amplitude_m'],control['direction_angle_degrees']), 'Explicit control fields changed')
    report = source['report']; component = report['components'][key]
    parent = report['components'][component['parent']]
    require(key in {t['component_id'] for t in source['job']['targets']}
        and component['type'] == 'sub_stem' and component['deleafed'] is False
        and parent['type'] == 'main_stem', 'Frozen intact direct-main-stem target required')
    members = descendants(report['components'], key)
    require(any(m != key for m in members)
        and all(m == key or (report['components'][m]['type'] == 'leaf'
            and report['components'][m]['parent'] == key) for m in members),
        'Only target petiole and its direct leaves are supported')
    old = donor_curve(component); old_absolute = old['points']+component['translation_plant_m']
    anchor = np.asarray(component['attachment_plant_m'], float)
    require(np.allclose(old_absolute[0], anchor, atol=1e-9, rtol=0), 'Source chain does not start at source attachment')
    axis = parent_tangent(component, parent)
    tangent=unit(old_absolute[1]-old_absolute[0]);direction=rotated_source_direction(axis,tangent,control['direction_angle_degrees'])
    guard,protected_faces=mesh_guard(safe_asset(Path(report['manifest_path']).parent,component['file']),
        np.asarray(component['translation_plant_m']),anchor,tangent)
    warp=SignedProximalShear(anchor,tangent,direction,control['amplitude_m'],guard)
    curve,source_samples=controlled_curve(old_absolute,old['radius'],warp)
    change = dict(component_id=key, members=members, curve=curve, warp=warp, control=deepcopy(control),
        direction=direction.tolist(), parent_tangent_plant=axis.tolist(), new_anchor_m=anchor.tolist(),
        source_anchor_m=anchor.tolist(), source_target_id=source['job']['plant_family']+'/'+key,
        source_curve_sha256=digest(dict(points=old_absolute.tolist(), radii=old['radius'].tolist())),
        controlled_curve_sha256=digest(dict(points=curve['points'].tolist(), radii=curve['radius'].tolist(), arc=curve['arc'].tolist())),
        source_centerline_length_m=float(old['arc'][-1]), controlled_centerline_length_m=float(curve['arc'][-1]),
        maximum_centerline_displacement_m=abs(float(control['amplitude_m'])),
        proximal_face_guard_m=guard,protected_source_faces=protected_faces,source_centerline_sample_arcs_m=source_samples.tolist(),
        original_source_capsule_knots_preserved=True,
        blend_sampling_policy='all_source_segment_intersections_with33_longitudinal_levels',
        protected_source_arc_m=[0.,.030],blend_end_longitudinal_m=.060,
        coordinate_policy='projection_on_donor_first_straight_segment',
        direct_leaf_policy='rigid_translation_after60mm_anchor_only',
        source_radius_scale=1., attachment_relocation_m=0., new_independent_donor_family=False,
        novelty_admission_approved=False)
    raw = next(r for r in source['raw']['components'] if r['id'] == key)
    metadata = transform_metadata(raw, change)
    measured, _, _ = dimensions(normalized(metadata), parent, load_rule())
    require(all(envelope['bounds'][k][0] <= value <= envelope['bounds'][k][1] for k,value in measured.items()),
        'Predeclared control outside frozen TRAIN envelope; hold without adjustment')
    require(metadata['attach_point'] == raw['attach_point']
        and metadata['axis']==raw['axis'] and metadata['radius']==raw['radius'], 'Source proximal attachment/axis/radius changed')
    for member in members:
        raw_member=next(r for r in source['raw']['components'] if r['id']==member)
        component_transport(raw_member,change)
    change['training_envelope_measurements'] = measured
    return change


def component_owners(changes):
    require(changes,'At least one explicit subtree control required')
    owners={}
    for change in changes:
        for key in change['members']:
            require(key not in owners,'Overlapping controlled subtrees')
            owners[key]=change
    return owners


def verify_components(source,changes,directory,raw,report):
    owners=component_owners(changes)
    old_rows={c['id']:c for c in source['raw']['components']};new_rows={c['id']:c for c in raw['components']}
    require(old_rows.keys()==new_rows.keys(),'Changed source organ population')
    source_root=Path(source['report']['manifest_path']).parent
    rows=[];serialized=[];changed=[];proximal={}
    for key,old in old_rows.items():
        change=owners.get(key)
        expected=transform_metadata(old,change) if change else {k:v for k,v in old.items() if k not in PHYSICS_FIELDS}
        require(new_rows[key]==expected,'Changed source/replayed manifest geometry:'+key)
        source_path=safe_asset(source_root,old['file']);actual=safe_asset(directory,report['components'][key]['file'])
        visible=frozen_v3.verify_visible_component(source_path,actual,spatial_changes_allowed=change is not None)
        if visible['changed_visible_attributes']:changed.append(key)
        rows.append(dict(component_id=key,**visible))
        if change:
            qa=check_component(source_path,actual,np.asarray(old['transform']['translate']),
                np.asarray(expected['transform']['translate']),component_transport(old,change))
            serialized.extend(dict(component_id=key,**q) for q in qa)
    target_ids={c['component_id'] for c in changes}
    require(target_ids<=set(changed) and set(changed)<=set(owners),'Missing target change or altered protected component')
    for change in changes:
        key=change['component_id']
        proximal[key]=proximal_mesh_identity(safe_asset(source_root,source['report']['components'][key]['file']),
            safe_asset(directory,report['components'][key]['file']),source['report']['components'][key],
            report['components'][key],change['warp'])
    return dict(allowed_changed_components=sorted(owners),changed_visible_components=sorted(changed),
        protected_component_count=len(old_rows)-len(owners),component_attribute_checks=rows,
        target_subtrees_only_visible_change_verified=True),serialized,proximal


def code_bindings():
    bindings=frozen_v3.code_bindings()
    for name in ('procedural_petiole_controlled_v4.py','procedural_petiole_controlled_catalogue_v4.py',
                 'procedural_distal_deformation_v1.py'):
        bindings[name]=file_hash(Path(__file__).with_name(name))
    return bindings


def generate(plan_path,family,primary_target,controls,output):
    from pxr import Usd
    from .geometry import audit_geometry
    output=Path(output).resolve();require(not output.exists(),'New controlled output only')
    plan,sources=load_training_sources(plan_path);require(family in sources,'Frozen TRAIN family required')
    source=sources[family];envelope=training_envelope(plan,sources);source_path=Path(source['report']['manifest_path']).resolve()
    require(not output.is_relative_to(source_path.parent) and not source_path.parent.is_relative_to(output),'Output must be disjoint from source')
    require(isinstance(controls,list) and controls and all(set(c)=={'component_id','control'} for c in controls)
        and len({c['component_id'] for c in controls})==len(controls),'Explicit unique component/control list required')
    require(primary_target in {c['component_id'] for c in controls},'Primary target must be explicitly controlled')
    output.mkdir(parents=True)
    write_json_new(output/'intent.json',dict(version=VERSION,source_family=family,primary_component_id=primary_target,controls=controls,
        source_plan_path=str(Path(plan_path).resolve()),source_plan_sha256=file_hash(plan_path),
        training_eligible=False,native_launched=False,amplitude_search=False))
    try:
        changes=[plan_change_explicit(source,c['component_id'],envelope,c['control']) for c in controls]
        owners=component_owners(changes);raw=deepcopy(source['raw'])
        raw.update(generator=VERSION,version='4.0.0',physics_supported=False,
            leaf_transport_policy='rigid_leaf_blade.v1',intended_use='explicit_multisubtree_static_geometry_pending_native_review')
        for key in PHYSICS_FIELDS:raw.pop(key,None)
        rows=[];textures={};copies={};derivatives=[];output_hashes={}
        for old in source['raw']['components']:
            change=owners.get(old['id']);row=transform_metadata(old,change) if change else deepcopy(old)
            for key in PHYSICS_FIELDS:row.pop(key,None)
            destination=safe_asset(output,old['file']);destination.parent.mkdir(parents=True,exist_ok=True)
            copies[old['id']]=frozen_v3.copy_static_component(safe_asset(source_path.parent,old['file']),destination,source_path.parent,textures)
            if change:
                checks=deform_new_copy(destination,np.asarray(old['transform']['translate']),
                    np.asarray(row['transform']['translate']),component_transport(old,change))
                derivatives.extend(dict(component_id=old['id'],**q) for q in checks)
            rows.append(row);output_hashes[old['file']]=file_hash(destination)
        raw['components']=rows;raw['component_count']=len(rows);write_json_new(output/'manifest.json',raw)
        report=audit_manifest(output/'manifest.json');require(report['status']!='blocked','Generated anatomy audit blocked')
        targets=[];measurements={}
        for change in changes:
            key=change['component_id'];component=report['components'][key];parent=report['components'][component['parent']]
            measured,_,_=dimensions(component,parent,load_rule())
            require(measured==change['training_envelope_measurements']
                and all(envelope['bounds'][k][0]<=v<=envelope['bounds'][k][1] for k,v in measured.items()),'Serialized target left TRAIN envelope')
            proposal=propose_cut_region(component,parent,load_rule())
            require(proposal['status']=='proposed_geometry_only' and not proposal['geometry_warnings'],'Controlled cut geometry warnings')
            surface=cut_surface_probe(Usd.Stage.Open(str(output/component['file'])),component,parent)
            require(surface['passed'],'Controlled cut interval failed actual surface probes')
            measurements[key]=measured
            targets.append(dict(component_id=key,source_target_id=change['source_target_id'],
                conservative_view_cap_group=change['source_target_id'],cut_region_proposal=proposal,
                cut_surface_probe=surface,shape_novelty_pending=True,training_eligible=False))
        geometry=audit_geometry(deepcopy(report))
        require(geometry['maximum_translation_error_m']<=1e-6,'Component assembly frame differs')
        warnings=sorted({w['code'] for w in geometry['warnings'] if w['component_id'] in owners})
        require(not warnings,'Controlled subtree geometry warnings:'+str(warnings))
        proof,serialized,proximal=verify_components(source,changes,output,raw,report)
        source_bindings={str(source_path):file_hash(source_path)}
        for component in source['report']['components'].values():
            path=safe_asset(source_path.parent,component['file'])
            require(file_hash(path)==component['asset_sha256'],'Source mesh changed during generation')
            source_bindings[str(path)]=component['asset_sha256']
        for path,sha in textures.items():
            relative=path.relative_to(output).as_posix();output_hashes[relative]=sha
            donor=safe_asset(source_path.parent,relative);require(file_hash(donor)==sha,'Source texture changed')
            source_bindings[str(donor)]=sha
        output_hashes['manifest.json']=file_hash(output/'manifest.json')
        recipes=[{k:v for k,v in c.items() if k not in ('curve','warp')} for c in changes]
        receipt=dict(schema=SCHEMA,version=VERSION,state=STATE,source_family=family,split='train',split_group=family,
            variant_id=output.name,primary_component_id=primary_target,primary_source_target_id=family+'/'+primary_target,
            source_plan_path=str(Path(plan_path).resolve()),source_plan_sha256=file_hash(plan_path),
            source_manifest_path=str(source_path),frozen_family_assignments=plan['family_assignments'],
            source_bindings=source_bindings,output_hashes=output_hashes,recipes=recipes,targets=targets,
            explicit_controls=controls,leaf_transport_policy='rigid_leaf_blade.v1',static_copy_receipts=copies,
            protected_attribute_proof=proof,proximal_mesh_identity_by_component=proximal,
            serialized_geometry_checks=serialized,mesh_derivative_diagnostics=derivatives,
            training_envelope_sha256=digest(envelope),training_envelope_bounds=envelope['bounds'],training_envelope_measurements=measurements,
            source_assets_unchanged=True,training_eligible=False,native_capture_validated=False,physics_validated=False,
            independent_target_novelty_approved=False,new_donor_family_created=False,source_cap_reset=False,
            geometry_distances_computed=False,visual_label=None,visual_review_performed=False,code_sha256=code_bindings())
        write_json_new(output/'qualification.json',receipt)
        print('MULTISUBTREE_CONTROL_CPU_QUALIFIED',output.name,len(changes),flush=True)
        return receipt
    except BaseException as exc:
        write_json_new(output/'FAILED.json',dict(state='explicit_control_held_without_search',error=type(exc).__name__+': '+str(exc),
            traceback=traceback.format_exc(),controls=controls,training_eligible=False,native_launched=False,
            amplitude_adjusted=False,seed_search=False))
        raise
