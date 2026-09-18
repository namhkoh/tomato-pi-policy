"""Conservative proposal rejection using full scene anatomy and camera geometry.

This never qualifies visibility or an answer. Inside an outer reach bound is
only potentially reachable; any projected competitor or unknown geometry holds
the proposal. A zero count still requires native census, depth/organ evidence,
per-target workspace and individual visual QA.
"""
from pathlib import Path
from collections import Counter
import numpy as np
from . import native848_all_petiole_9mm_v1 as geometry_api
from . import capture_contract,cut_regions
from .dataset_review import require
from .depth_preview import sha256

SCHEMA='greenhouse.native848_projected_competition_proposal_screen.v1'

class ProjectionCompetition:
    def __init__(self,reports,census,reference_calibration):
        require(census['complete_active_plant_anatomy'] is True,'Complete active anatomy census required')
        self.reference=reference_calibration
        self.pins={str(Path(m.__file__).resolve()):sha256(m.__file__) for m in (geometry_api,capture_contract,cut_regions)}
        self.pins[str(Path(__file__).resolve())]=sha256(__file__)
        by={r['plant_id']:r for r in reports};self.targets=[];self.unknown=[];self.excluded=Counter()
        retained=[p for p in census['all_plant_roots'] if p['active_after_policy']]
        require({p['plant_root'] for p in retained}==set(census['retained_roots'])
                and all(p['authenticated_component_plant'] for p in retained),'Retained plant census differs')
        for plant in retained:
            report=by[plant['source_family']]
            for anatomy in report['targets']:
                cid=anatomy['component_id'];component=report['components'][cid];tid=plant['source_family']+'/'+cid
                leaves=[k for k in anatomy['expected_detached_component_ids'] if report['components'][k]['type']=='leaf']
                if component['type']!='sub_stem' or component.get('deleafed') is not False or component.get('parent') not in report['components'] or report['components'][component['parent']]['type']!='main_stem' or anatomy['protected_descendant_ids'] or not leaves:
                    self.excluded['biologically_ineligible']+=1;continue
                try:
                    g=geometry_api.geometry_9mm(report,cid,plant['plant_to_world_usd_row_vectors'],reference_calibration)
                    attachment=capture_contract.transform_points([component['attachment_plant_m']],plant['plant_to_world_usd_row_vectors'])[0].tolist()
                    self.targets.append(dict(target_id=tid,plant_root=plant['plant_root'],nominal_world_m=g['nominal']['world_m'],
                        junction_world_m=[attachment]+[x['world_m'] for x in g['junction_to_cut']]))
                except (ValueError,KeyError,TypeError) as exc:self.unknown.append(dict(target_id=tid,error=str(exc)))
        require(len({t['target_id'] for t in self.targets})==len(self.targets),'Repeated active target identity')

    def check(self,metadata,target_id,workspace):
        cal=metadata['calibration']
        require(set(cal)==set(self.reference) and all(cal[k]==self.reference[k] for k in cal if k!='camera_to_world_usd_row_vectors'),
                'Prescreen cannot change calibrated optics')
        pose=workspace.checker.pose_and_probe(metadata)
        model=workspace.checker.model
        shoulder=model.arm_shoulder_position_m('left',pose['base'],pose['torso'])
        outer=model.maximum_endpoint_reach_m('left')+float(np.linalg.norm(pose['probe']))
        possible=[];excluded=Counter(self.excluded)
        for target in self.targets:
            if target['target_id']==target_id:continue
            distance=float(np.linalg.norm(np.asarray(target['nominal_world_m'])-shoulder))
            if distance>outer+.001:excluded['outside_conservative_reach_bound']+=1;continue
            nominal=capture_contract.project([target['nominal_world_m']],cal)[0]
            probes=capture_contract.project(target['junction_world_m'],cal)
            if not any(p['projection_status']=='in_frame' for p in [nominal,*probes]):
                excluded['all_attachment_to_nominal_probes_out_of_frame']+=1;continue
            possible.append(dict(target_id=target['target_id'],cut_uv=nominal['pixel_xy'],
                cut_projection_status=nominal['projection_status'],attachment_uv=probes[0]['pixel_xy'],
                all_attachment_to_nominal_probes_in_frame=all(p['projection_status']=='in_frame' for p in [nominal,*probes]),
                shoulder_distance_m=distance,conservative_outer_bound_m=outer,
                actual_visibility_unknown=True,actual_ik_reachability_unknown=True))
        require(any(t['target_id']==target_id for t in self.targets),'Planned target missing from intact anatomy')
        clear=not possible and not self.unknown
        return dict(schema=SCHEMA,target_id=target_id,proposal_survives_geometric_rejection=clear,
            decision='native_candidate_pending_all_actual_checks' if clear else 'hold_projected_competition_or_unknown',
            potential_competitors=possible,potential_competing_petiole_count=len(possible),geometry_unknowns=self.unknown,
            projected_exclusions=dict(excluded),calibrated_optics_unchanged=True,per_pose_camera_FK_verified=True,
            primary_detail_thresholds_never_exclude_alternatives=True,actual_visibility_or_unique_answer_qualified=False,
            conservative_rejection_can_discard_occluded_or_unreachable_alternatives=True,
            native_full_census_and_individual_QA_required=True,training_approved=False,accepted_training_increment=0)
