"""Standalone unchanged census/answer checks for verified joint annotation epoch2."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image  # shared finite validation adapter namespace
from . import native848_data_roots_v1 as roots
from .capture_contract import project
EPOCH="greenhouse.native848_all_petiole_9mm.v2"
BANNED="seed41_full/SubStem_38"

def require(value,message):
    if not value:raise ValueError(message)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def pin(p):return dict(path=str(Path(p).resolve()),sha256=digest(p))

def read_pin(spec):
    require(set(spec)=={'path','sha256'},'Exact file pin required')
    p=roots.resolve_evidence(spec['path']);require(p.is_file() and digest(p)==spec['sha256'],'Changed pinned evidence')
    return p

def point_at_9mm(chain):
    chain=np.asarray(chain,float)
    require(chain.ndim==2 and chain.shape[1]==3 and len(chain)>=2 and np.isfinite(chain).all(),'Oriented finite world centerline required')
    lengths=np.linalg.norm(np.diff(chain,axis=0),axis=1)
    require((lengths>1e-12).all() and lengths.sum()>=.009,'Degenerate or short centerline')
    cumulative=np.r_[0.,np.cumsum(lengths)];i=min(int(np.searchsorted(cumulative,.009,side='right')-1),len(lengths)-1)
    return chain[i]+(.009-cumulative[i])/lengths[i]*(chain[i+1]-chain[i])

def target_answer(target,cal):
    require(target['status']=='candidate_pending_visual_review' and target['automated_pass'] is True,'Only evaluated candidates can be accepted')
    require(target['annotation_epoch']==EPOCH and target['nominal_arc_m']==.009,'Exact new9mm epoch required')
    require(target['support_is_permissible_cut_interval'] is False,'Visibility support is not cut tolerance')
    geo=target['geometry'];expected=point_at_9mm(geo['oriented_centerline_world_m'])
    attachment_expected=project([geo['oriented_centerline_world_m'][0]],cal)[0]
    require(np.allclose(attachment_expected['camera_optical_xyz_m'],geo['attachment_projected']['camera_optical_xyz_m'],rtol=0,atol=1e-8),'Centerline does not start at actual attachment')
    require(geo['nominal']['arc_m']==.009 and np.allclose(expected,target['cut_world_m'],rtol=0,atol=1e-9)
            and np.allclose(expected,geo['nominal']['world_m'],rtol=0,atol=1e-9),'Cut does not follow9mm arc length')
    projected=project([expected],cal)[0]
    require(projected['projection_status']=='in_frame'
            and np.allclose(projected['pixel_xy'],target['cut_point_uv'],rtol=0,atol=1e-6)
            and np.allclose(projected['camera_optical_xyz_m'],target['cut_optical_xyz_m'],rtol=0,atol=1e-9),'Cut projection/calibration differs')
    workspace=target['workspace']
    require(workspace['target_id']==target['target_id'] and workspace['result']['workspace_passed'] is True
        and np.allclose(workspace['nominal_world_m'],expected,rtol=0,atol=1e-9)
        and workspace['per_frame_camera_FK_verified'] is True
        and workspace['per_frame_cached_solution_FK_verified'] is True
        and isinstance(workspace['solve_input_sha256'],str) and len(workspace['solve_input_sha256'])==64,
        'Workspace proof is not the verified 9 mm problem')
    require(target['visibility']['nominal']['visible'] is True and all(p['visible'] is True for p in target['visibility']['support'])
        and target['local_clarity']['passed'] is True,'Missing new nominal/support/clarity checks')
    junction=target['junction_continuity'];probes=junction['probes']
    arcs=np.asarray([q['arc_m'] for q in probes],float)
    require(junction['attachment_visible'] is True and junction['all_attachment_to_9mm_probes_visible'] is True
        and junction['maximum_arc_step_m']<=.0005 and len(arcs)>=19 and np.isfinite(arcs).all()
        and abs(arcs[0])<1e-12 and abs(arcs[-1]-.009)<1e-12
        and (np.diff(arcs)>0).all() and (np.diff(arcs)<=.0005+1e-12).all()
        and all(q['visible'] is True and q['projected']['projection_status']=='in_frame'
                and q['depth']['status']=='depth_consistent_not_visibility_verified' for q in probes),
        'Entire actual attachment-to-9mm junction must be visible')
    parent=target['parent_context'];z=projected['camera_optical_xyz_m'][2]
    diameter=2*geo['nominal']['radius_m']*min(cal['intrinsics'][0][0],cal['intrinsics'][1][1])/z
    required=max(3.,diameter*.65)
    require(parent['visible_parent_pixels']>=6 and np.isfinite(parent['cut_to_parent_distance_px'])
        and abs(parent['required_distance_px']-required)<1e-8
        and parent['cut_to_parent_distance_px']>=required,'Actual parent visibility/clearance insufficient')
    box=target['target_bbox_xyxy'];require(len(box)==4 and all(type(x) is int for x in box)
        and 0<=box[0]<box[2]<=848 and 0<=box[1]<box[3]<=408,'Visible bbox must be half-open native pixels')
    attachment=geo['attachment_projected'];require(attachment['projection_status']=='in_frame','Visible attachment required')
    return dict(bbox_xyxy=box,attachment_pixel_uv=attachment['pixel_xy'],cut_point_uv=target['cut_point_uv'],
                cut_optical_xyz_m=target['cut_optical_xyz_m'])

def validate_census(annotation):
    require(annotation['schema']==annotation['annotation_epoch']==EPOCH and annotation['training_approved'] is False,'New candidate-only annotation required')
    census=annotation['target_census'];rows=annotation['targets'];ids=[t['target_id'] for t in rows]
    require(len(ids)==len(set(ids)) and sorted(ids)==sorted(census['catalogue_petiole_ids']) and census['complete'] is True and census['catalogue_complete'] is True,'Incomplete/duplicate target census')
    for status,key in [('candidate_pending_visual_review','candidate_target_ids'),('excluded','excluded_target_ids'),('unknown','unknown_target_ids')]:
        require(sorted(t['target_id'] for t in rows if t['status']==status)==sorted(census[key]),'Census status population differs')
    require(all(t['status'] in ('candidate_pending_visual_review','excluded','unknown') for t in rows),'Unknown census status')
    require(not census['unknown_target_ids'] and annotation['frame_blocked_by_unknown_targets'] is False,'Unresolved possible petiole blocks frame')
    require(not any(t['target_id']==BANNED and t['status']=='candidate_pending_visual_review' for t in rows),'Eligible permanently excluded target holds whole frame')
    require(annotation['frame_blocked_by_unverified_full_scene_coverage'] is False
        and census['full_scene_coverage_validation_required'] is True,'Unverified full-scene coverage blocks frame')
    spec=census['full_scene_coverage'];coverage=json.loads(read_pin({k:spec[k] for k in ('path','sha256')}).read_text())
    require(coverage['schema']=='greenhouse.native848_all_petiole_coverage_audit.v1'
        and coverage.get('CPU_fixture_only') is not True and coverage['complete_all_eligible_ground_truth'] is True
        and not coverage['blocking_unknowns'] and not coverage['visible_cross_split_petiole_targets'],
        'Authoritative coverage has unknown/cross-split possible positives')
    require(annotation['observation']==dict(path=coverage['observation_path'],sha256=coverage['observation_sha256']),
        'Coverage belongs to a different actual frame')
    for path,h in coverage['source_bindings'].items():require(digest(path)==h,'Coverage source changed')
    return exactly_one_candidate(rows)

def exactly_one_candidate(rows):
    require(not any(t['status']=='unknown' for t in rows),'Unknown target prevents uniqueness')
    candidates=[t for t in rows if t['status']=='candidate_pending_visual_review']
    require(len(candidates)==1,'Exactly one genuinely eligible petiole required; multi/zero targets held')
    return candidates
