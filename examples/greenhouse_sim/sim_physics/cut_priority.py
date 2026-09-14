"""Warm-start candidate ORDER from a prior native cut, never reuse its path.

All candidates, current geometric transforms and native checks remain. This
uses privileged engineering trial records, not VLM execution observations.
"""
import hashlib
import json
import math
from pathlib import Path


def validate_frame_family(family):
    """Validate only the existing crossbar proposal set; grant no authority."""
    if (not isinstance(family,dict) or set(family)!={'tilt','normal_sign','wing_m'}
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in family.values())
            or family['tilt'] not in (0.,-10.,10.,-15.,15.)
            or family['normal_sign'] not in (-1,1) or abs(family['wing_m'])>.02):
        raise ValueError('Finite original crossbar candidate family required')
    return dict(tilt=float(family['tilt']),normal_sign=int(family['normal_sign']),wing_m=float(family['wing_m']))


def load(path,args):
    raw=Path(path).read_bytes()
    if len(raw)>2_000_000:raise ValueError('Bounded native cut report required')
    report=json.loads(raw);config=report.get('configuration',{});gates=report.get('gates',{})
    for key in ('plant','target','cut_arc_m','knife_edge_mode','knife_alignment'):
        if config.get(key)!=getattr(args,key):raise ValueError('Cut priority task mismatch: '+key)
    if not all(gates.get(k) is True for k in ('blade_contact_release','full_forward_cut_stroke_verified','right_withdrawal_completed')):
        raise ValueError('Prior native release, traversal and withdrawal required for warm-start receipt')
    plan=report.get('robot',{}).get('cut_plan',{})
    values=[plan.get(k) for k in ('plane_tilt_degrees','normal_sign','wing_m')]
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
            or values[0] not in (0.,-10.,10.,-15.,15.) or values[1] not in (-1,1) or abs(values[2])>.02):
        raise ValueError('Finite original candidate family required')
    return dict(tilt=float(values[0]),normal_sign=int(values[1]),wing_m=float(values[2]),
        source_report=str(Path(path).resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
        order_only=True,prior_pose_or_path_replayed=False,motion_authorized=False)


def from_arguments(args):
    """Keep a reference waiting-pose seed paired with its blade-frame family.

    Previously a station-reference search reused that run's joint seed but
    silently tried only the unrelated default zero-tilt cut frame. This is
    one blade-frame family for zero-motion station search and only candidate
    ordering for execution. Same-task evidence and every fresh native check
    still apply. An explicit cut-priority report remains authoritative.
    """
    path=args.cut_priority_report
    reference=path is None and args.station_reference_report is not None
    if reference:
        from .station_reference import validate_mode
        validate_mode(args)
        path=args.station_reference_report
    if path is None:return None
    if not (args.bimanual_cut and args.cut_style=='downward'
            and args.knife_edge_mode=='source_crossbar_edge_v1'):
        raise ValueError('Cut priority requires explicit actual-crossbar downward diagnostic')
    result=load(path,args)
    if reference:result['station_reference_frame_family_only']=True
    return result


def tilt_order(tilts,priority):
    if priority is None:return tilts
    if priority['tilt'] not in tilts:raise ValueError('Priority outside current tilt family')
    return tuple(sorted(tilts,key=lambda v:v!=priority['tilt']))


def proposal_order(proposals,tilt,priority):
    if priority is None or tilt!=priority['tilt']:return proposals
    def matches(p):
        return p[0] is None and p[1]==priority['normal_sign'] and abs(p[2]-priority['wing_m'])<=1e-9
    if not any(matches(p) for p in proposals):return proposals  # Current geometry/orientation excludes it.
    return sorted(proposals,key=lambda p:not matches(p))
