"""Exact authenticated9mm outside-both-arm proof before optional visibility work.

No invisible-pixel or failed-IK substitute: only a positive geometric outer
bound excludes a target. In-bound or unproved targets retain frozen predicates.
"""
from pathlib import Path
import itertools
import numpy as np
from . import native848_fully_labeled_coverage_v1 as prior
from . import native848_all_petiole_9mm_v1 as labels
from . import robot_preview
from .cut_regions import _oriented_chain,_sample
from .capture_contract import transform_points,fingerprint
from .dataset_review import require,verify_bindings
from .depth_preview import sha256
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
from pxr import Usd,UsdGeom
SCHEMA='greenhouse.native848_fully_labeled_coverage.v2'
BOUNDS_SCHEMA='greenhouse.native848_fixed_pose_both_arm_outer_bounds.v1'
EXCLUSION_SCHEMA='greenhouse.native848_exact9mm_both_arm_outer_exclusion.v1'
OUTSIDE_REASON='authenticated_exact9mm_outside_both_arm_outer_bounds'
MARGIN_M=.001
component_catalogue=prior.component_catalogue
component_masks=prior.component_masks
CatalogueIndex=prior.CatalogueIndex
target_inventory=prior.target_inventory


def robot_bounds(metadata,workspace_checker):
    checker=getattr(workspace_checker,'checker',workspace_checker)
    # Both reviewed scene recipes explicitly configure this unchanged stock tool.
    robot_preview.configure_right_tool(checker.stage,checker.root_path,'gripper')
    pose=checker.pose_and_probe(metadata)
    cache=UsdGeom.XformCache();boxes=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    arms={}
    for arm,prefix in (('left','l'),('right','r')):
        ee=checker.stage.GetPrimAtPath(checker.root_path+'/ee_'+arm)
        require(bool(ee),'Actual end effector required')
        inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(ee)).T)
        centers=[]
        for n in (1,2):
            prim=checker.stage.GetPrimAtPath(f'{checker.root_path}/ee_finger_{prefix}{n}/visuals')
            require(bool(prim) and prim.IsActive() and UsdGeom.Imageable(prim).ComputeVisibility()!='invisible','Actual visible gripper required')
            bound=boxes.ComputeWorldBound(prim).ComputeAlignedRange();require(not bound.IsEmpty(),'Actual gripper geometry required')
            corners=np.asarray(list(itertools.product(*zip(bound.GetMin(),bound.GetMax()))))
            local=(inverse@np.c_[corners,np.ones(8)].T).T[:,:3]
            centers.append((local.min(0)+local.max(0))/2)
        probe=np.mean(centers,axis=0)
        if arm=='left':require(np.allclose(probe,pose['probe'],atol=1e-10,rtol=0),'Frozen left workspace probe differs')
        endpoint=checker.model.maximum_endpoint_reach_m(arm)
        arms[arm]=dict(shoulder_world_m=checker.model.arm_shoulder_position_m(arm,pose['base'],pose['torso']).tolist(),
            maximum_endpoint_reach_m=endpoint,actual_finger_midpoint_ee_m=probe.tolist(),
            conservative_probe_reach_m=endpoint+float(np.linalg.norm(probe)))
    bindings=dict(checker.bindings)
    bindings.update({str(Path(m).resolve()):sha256(m) for m in (__file__,robot_preview.__file__)})
    return dict(schema=BOUNDS_SCHEMA,robot_state_sha256=fingerprint(dict(calibration=metadata['calibration'],robot_snapshot=metadata['robot_snapshot'])),
        arms=arms,margin_m=MARGIN_M,fixed_base_and_torso=True,per_frame_camera_FK_verified=True,
        actual_tool_configuration='both_stock_grippers',source_bindings=bindings)


def source_geometry(entry):
    """Same oriented chain, source radius/attachment checks, transform and9mm arc."""
    report=entry['report'];instance,key=entry['target_id'].split('/')
    require(report['plant_id']==entry['source_family'],'Source report identity differs')
    component=report['components'][key]
    require(component['type']=='sub_stem' and component.get('deleafed') is False
        and component.get('parent') in report['components']
        and report['components'][component['parent']]['type']=='main_stem','Intact direct main-stem petiole required')
    matrix=np.asarray(entry['plant_to_world_usd_row_vectors'],float)
    require(matrix.shape==(4,4) and np.isfinite(matrix).all()
        and np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-7,rtol=0)
        and abs(np.linalg.det(matrix[:3,:3])-1.)<1e-7
        and np.allclose(matrix[:,3],[0,0,0,1],atol=1e-9,rtol=0),'Rigid metre-scale plant transform required')
    chain,lengths,reverse,error=_oriented_chain(component,1e-6)
    require(lengths[-1]>=.030,'Petiole too short for unchanged proximal evidence')
    origin=component['translation_plant_m'];point=_sample(chain,lengths,.009,origin)
    world=transform_points([point['point_plant_m']],matrix)[0]
    return dict(nominal=dict(arc_m=.009,world_m=world.tolist(),point_plant_m=point['point_plant_m'],radius_m=point['petiole_radius_m']),
        oriented_centerline_world_m=transform_points([np.asarray(origin)+np.asarray(p[:3]) for p in chain],matrix).tolist(),
        centerline_arc_distances_m=[float(x) for x in lengths],centerline_total_length_m=float(lengths[-1]),
        source_chain_reversed=bool(reverse),attachment_endpoint_error_m=float(error))


def outside_proof(target_id,geometry,bounds):
    require(bounds['schema']==BOUNDS_SCHEMA and bounds['margin_m']==MARGIN_M and set(bounds['arms'])=={'left','right'},'Exact both-arm bound required')
    point=np.asarray(geometry['nominal']['world_m'],float)
    require(point.shape==(3,) and np.isfinite(point).all() and geometry['nominal']['arc_m']==.009,'Exact finite9mm point required')
    distances={arm:float(np.linalg.norm(point-np.asarray(v['shoulder_world_m']))) for arm,v in bounds['arms'].items()}
    if not all(distances[arm]>v['conservative_probe_reach_m']+MARGIN_M for arm,v in bounds['arms'].items()):return None
    return dict(schema=EXCLUSION_SCHEMA,target_id=target_id,nominal_arc_m=.009,nominal_world_m=point.tolist(),
        bounds_sha256=fingerprint(bounds),both_shoulder_distances_m=distances,margin_m=MARGIN_M,
        outside_both_conservative_bounds=True,visibility_assumed=False,ik_search_failure_used=False)


def validate_exclusion(target,bounds):
    require(target['status']=='excluded' and target['reason']==OUTSIDE_REASON
        and target['automated_pass'] is False and target['training_approved'] is False,'Only explicit objective exclusion required')
    geometry=target['geometry'];chain=np.asarray(geometry['oriented_centerline_world_m'],float)
    require(chain.ndim==2 and chain.shape[1]==3 and len(chain)>=2 and np.isfinite(chain).all(),'Authenticated finite centerline required')
    lengths=np.linalg.norm(np.diff(chain,axis=0),axis=1);require((lengths>1e-12).all() and lengths.sum()>=.030,'Nondegenerate authenticated centerline required')
    cumulative=np.r_[0.,np.cumsum(lengths)];i=min(int(np.searchsorted(cumulative,.009,side='right')-1),len(lengths)-1)
    actual=chain[i]+(.009-cumulative[i])/lengths[i]*(chain[i+1]-chain[i])
    require(np.allclose(actual,geometry['nominal']['world_m'],atol=1e-9,rtol=0)
        and np.allclose(actual,target['cut_world_m'],atol=1e-9,rtol=0),'Outside proof point differs from9mm centerline arc')
    expected=outside_proof(target['target_id'],geometry,bounds)
    require(expected is not None and target['outer_workspace_exclusion']==expected,'Outside both-arm evidence differs')
    require('visibility' not in target and 'junction_continuity' not in target,'No fabricated visibility in objective reach exclusion')
    return expected


def verify_annotation_outer_exclusions(annotation,metadata,workspace_checker=None):
    owned=workspace_checker is None
    checker=WorkspaceChecker() if owned else workspace_checker
    try:
        expected=robot_bounds(metadata,checker)
        require(annotation['outer_workspace_bounds']==expected,'Authenticated recorded robot outer spheres differ')
        for target in annotation['targets']:
            if target.get('reason')==OUTSIDE_REASON or 'outer_workspace_exclusion' in target:validate_exclusion(target,expected)
        verify_bindings(expected['source_bindings'])
        return expected
    finally:
        if owned:checker.finish()


def evaluate_frame(metadata,rgb,depth,valid,components,catalogue,target_inventory,workspace_checker,*,scene_coverage=None):
    index=CatalogueIndex(catalogue)
    result=labels.evaluate_frame(metadata,rgb,depth,valid,components,catalogue,[],workspace_checker,scene_coverage=scene_coverage)
    bounds=robot_bounds(metadata,workspace_checker)
    expected=set(result['target_census']['catalogue_petiole_ids']);entries=prior._unique(target_inventory,'target_id','inventory target')
    require(set(entries)<=expected,'Foreign inventory target');visible=set(map(int,np.unique(components)));rows=[]
    for original in result['targets']:
        target_id=original['target_id']
        if target_id not in entries:rows.append(original);continue
        entry=entries[target_id]
        try:
            instance,key=target_id.split('/');target=index.by_identity[(instance,key)]
            require(entry['report']['plant_id']==entry['source_family']==target['source_plant_id']==target['split_group'],'Report/source family differs')
            require(target['organ_type']=='sub_stem','Exact petiole catalogue identity required')
            if target['component_index'] not in visible:row=prior._zero_pixel_row(entry)
            else:
                proof=None
                if entry.get('semantic_leaf_petiole_candidate') is True:
                    geometry=source_geometry(entry);proof=outside_proof(target_id,geometry,bounds)
                if proof is not None:
                    row=dict(target_id=target_id,source_family=entry['source_family'],status='excluded',reason=OUTSIDE_REASON,
                        annotation_epoch=labels.ANNOTATION_EPOCH,nominal_arc_m=.009,visibility_support_arc_m=list(labels.SUPPORT_ARC_M),
                        support_is_permissible_cut_interval=False,actual_visual_review=False,automated_pass=False,training_approved=False,
                        geometry=geometry,cut_world_m=geometry['nominal']['world_m'],outer_workspace_exclusion=proof)
                else:row=labels.evaluate_target(metadata,rgb,depth,valid,components,index.by_variant[instance],entry,workspace_checker)
            rows.append(row)
        except (ValueError,KeyError,AssertionError) as exc:
            rows.append(dict(target_id=target_id,status='unknown',reason='invalid_target_evidence: '+str(exc),automated_pass=False,actual_visual_review=False,training_approved=False))
    for row in rows:
        instance,key=row['target_id'].split('/');source=index.by_identity[(instance,key)]['source_plant_id']
        row.update(source_family=source,source_component_id=key,source_target_id=source+'/'+key,plant_instance_id=instance)
    result['targets']=rows;result['outer_workspace_bounds']=bounds
    result['target_census'].update(catalogue_complete=set(entries)==expected,
        candidate_target_ids=[r['target_id'] for r in rows if r['status']=='candidate_pending_visual_review'],
        excluded_target_ids=[r['target_id'] for r in rows if r['status']=='excluded'],unknown_target_ids=[r['target_id'] for r in rows if r['status']=='unknown'])
    result['frame_blocked_by_unknown_targets']=bool(result['target_census']['unknown_target_ids'])
    return result
