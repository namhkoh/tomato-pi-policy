"""Additive target/view-group visual QA; no native, annotation or export operations.

Strict Q3 members retain every per-frame gate. Q4 rows are always flagged and
individually inspected. Existing individual receipts and labels stay unchanged.
"""
from collections import defaultdict
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import math
from PIL import Image

SCHEMA = 'greenhouse.native848_diverse_group_inventory.v2'
REVIEW_SCHEMA = 'greenhouse.native848_diverse_group_review.v2'
PROTOCOL_SCHEMA = 'greenhouse.native848_diverse_group_protocol.v1'
WORKSPACE = Path(__file__).resolve().parents[3]
POLICY_PATH = WORKSPACE/'data/sim_data/diagnostics/native848_group_review_user_policy_20260917_v1/policy.json'
PROTOCOL = dict(schema=PROTOCOL_SCHEMA, head_bin_degrees=2.0, pixel_bin=64,
    maximum_group_members=64, q4_always_individually_flagged=True,
    representatives=['first','last','worst_margin','head_extrema','cut_extrema'],
    any_reviewed_ambiguity_holds_whole_group=True,
    all_flagged_candidates_require_individual_review=True,
    unsampled_members_individual_visual_review=False,
    per_image_automated_gates_unchanged=True,
    contact_sheets_are_screening_only=True,
    final_review_requires_full_native_and_exact_unscaled_crop=True)
ALIASES = ('rgb_sha256','decoded_rgb_sha256','conservative_camera_signature')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')


class Pins:
    def __init__(self):
        self.values={}
    def add(self, spec):
        p=Path(spec['path']).resolve(); expected=spec['sha256']
        require(p.is_relative_to(WORKSPACE),'Evidence must remain in workspace')
        require(len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'Invalid pin')
        if str(p) not in self.values:
            require(digest(p)==expected,'Changed artifact: '+str(p))
            self.values[str(p)]=expected
        require(self.values[str(p)]==expected,'Conflicting pin')
        return p
    def json(self,spec):
        return read(self.add(spec))


def pin(path):
    p=Path(path).resolve()
    return dict(path=str(p),sha256=digest(p))


def finite(value):
    x=float(value)
    require(math.isfinite(x),'Nonfinite grouping or quality metric')
    return x


def aliases(value):
    out=set()
    if isinstance(value,dict):
        for k in ALIASES:
            if value.get(k):out.add((k,value[k]))
        for v in value.values():out.update(aliases(v))
    elif isinstance(value,list):
        for v in value:out.update(aliases(v))
    return out


def held_aliases(raw):
    rows=[json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    return aliases(rows)


def describe(row, pins):
    """Read source-bound grouping fields; exporter separately replays all gates."""
    require(row['split']=='train' and row['source_target']!='seed41_full/SubStem_38',
            'Wrong split or excluded physical target')
    for k in ('candidate_for_individual_visual_review','workspace_passed',
              'render_profile_qualified','local_clarity_passed','full_trace_and_anchored_grid_passed'):
        require(row[k] is True,'Per-image gate failed: '+k)
    q4=row.get('input_route')=='bulk_q4'
    if q4:
        require(row.get('q3_passed') is False and all(row.get(k) is True for k in
            ('q4_passed','Q1passed','protected_foreground_passed','conditional_background_passed')),
            'Conditional Q4 gates failed')
    else:
        require(row['input_route']=='bulk_q3' and row['q3_passed'] is True,
                'Only genuine Q3 or individually flagged Q4 rows supported')
    a=row['artifacts']; sample=pins.json(a['sample']); label=pins.json(a['label'])
    trace=pins.json(a['query_trace']); ctx=pins.json(a['context'])
    require(sample['sample_id']==row['sample_id'] and label['eligible'] is True,
            'Sample/label identity differs')
    require(label['target_id']==row['source_target'] and sample['supervision']['target_id']==row['source_target'],
            'Physical target mismatch')
    require(trace['passed'] is True and trace['legacy_trace']['passed'] is True
            and trace['fixed_grid']['passed'] is True,'Trace/grid failed')
    require(label['resolution']==[848,408] and sample['calibration']['resolution']==[848,408],
            'Native resolution differs')
    cal=sample['calibration']; snap=sample['robot_snapshot']
    optical={k:v for k,v in cal.items() if k!='camera_to_world_usd_row_vectors'}
    body={k:v for k,v in snap['joint_degrees'].items() if not k.startswith('head_')}
    head=[finite(snap['joint_degrees'][k]) for k in ('head_0','head_1')]
    cut=[finite(x) for x in label['nominal_pixel_uv']]
    query=[finite(x) for x in label['query_pixel_uv']]
    require(all(0<=p[0]<848 and 0<=p[1]<408 for p in (cut,query)),'Point outside native frame')
    anchor=ctx['anchor']
    identity=dict(target=row['source_target'],family=row['source_family'],
        context_sha256=a['context']['sha256'],anchor_entry=anchor['reference_entry_id'],
        anchor_bank_sha256=anchor['reference_bank_sha256'],body_joints=body,
        robot_root=snap['robot_root_to_world_usd_row_vectors'],
        camera_to_head=snap['camera_to_head_column_vectors'],calibration=optical,
        profile=row['profile'],annotation_epoch=row['annotation_epoch'])
    bins=dict(head=[math.floor(x/PROTOCOL['head_bin_degrees']) for x in head],
        cut=[math.floor(x/PROTOCOL['pixel_bin']) for x in cut],
        query=[math.floor(x/PROTOCOL['pixel_bin']) for x in query])
    c=label['clarity']; t=trace['legacy_trace']; grids=trace['fixed_grid']['probes']
    width=finite(c['width_proxy_px']); length=finite(c['interval_length_px'])
    luma=finite(t['median_luminance']); interior=finite(t['minimum_interior_radius_px'])
    contrast=min(finite(p['usability']['local_contrast_8bit']) for p in grids)
    margin=min(finite(p['usability']['frame_margin_px']) for p in grids)
    metrics=dict(width_px=width,interval_px=length,trace_luma=luma,
        trace_interior_px=interior,minimum_contrast=contrast,minimum_frame_margin=margin,
        worst_margin=min(width/8,length/12,luma/40,interior/2,contrast/6,margin/16))
    flags=[]
    if q4:flags.append('conditional_Q4_background_exception')
    if row.get('flagged') or row.get('flags') or row.get('visual_flags'):flags.append('explicit_source_flag')
    if width<8.5 or length<12.5 or luma<45 or interior<2.5 or contrast<8 or margin<20:
        flags.append('near_automated_quality_boundary')
    if finite(t['dark_fraction'])>0:flags.append('nonzero_trace_dark_fraction')
    if math.dist(cut,query)>160:flags.append('long_query_to_cut_path')
    return dict(sample_id=row['sample_id'],capture_index=int(row['capture_index']),
        group_base=stable(dict(identity=identity,bins=bins)),group_identity=identity,
        region_bins=bins,head=head,cut=cut,query=query,metrics=metrics,flags=flags,
        flagged=bool(flags),rgb_sha256=row['rgb_sha256'],
        label_sha256=a['label']['sha256'],source_sample_sha256=row['source_sample_sha256'])


def select_groups(descriptors):
    grouped=defaultdict(list)
    for d in descriptors:grouped[d['group_base']].append(d)
    groups=[]; records={}
    for base in sorted(grouped):
        values=sorted(grouped[base],key=lambda x:(x['capture_index'],x['sample_id']))
        for offset in range(0,len(values),PROTOCOL['maximum_group_members']):
            block=values[offset:offset+PROTOCOL['maximum_group_members']]
            gid=stable(dict(base=base,members=[x['sample_id'] for x in block]))
            reasons=defaultdict(list)
            def select(x,reason):reasons[x['sample_id']].append(reason)
            select(block[0],'first');select(block[-1],'last')
            select(min(block,key=lambda x:(x['metrics']['worst_margin'],x['sample_id'])),'worst_margin')
            for axis in range(2):
                for field in ('head','cut'):
                    select(min(block,key=lambda x:(x[field][axis],x['sample_id'])),field+str(axis)+'_min')
                    select(max(block,key=lambda x:(x[field][axis],x['sample_id'])),field+str(axis)+'_max')
            for x in block:
                if x['flagged']:select(x,'all_flagged')
            members=[x['sample_id'] for x in block]
            selected=[sid for sid in members if sid in reasons]
            groups.append(dict(group_id=gid,identity=block[0]['group_identity'],
                region_bins=block[0]['region_bins'],sample_ids=members,
                selected_sample_ids=selected,selection_reasons=dict(reasons)))
            for x in block:
                records[x['sample_id']]={**x,'group_id':gid,
                    'selected_for_visual_review':x['sample_id'] in reasons}
    return groups,records


ADAPTER3_SCHEMA = 'greenhouse.native848_diverse_bulk_q3_adapter.v3'
ADAPTER3_PINS = {
    'examples/greenhouse_sim/sim_data/native848_diverse_bulk_adapter_v3.py':
        '3a71e289dd96a2104f146b9e3f965fa8e0a9003f8783f41d8cfdb0e485354512',
    'examples/greenhouse_sim/sim_data/native848_multianchor_plan_v1.py':
        '5fc1e15bd493435fb2db50f11ab9a5c67c82d52daaae7caa3d7b31244f61e31c',
    'examples/greenhouse_sim/sim_data/native848_multianchor_worker_v1.py':
        '261a7f7374de738a17edd0235637dfb0501ca56905782d4127863b4b98164fae',
    'data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_native848_multianchor_v1.py':
        'c92ec4cd9409dfbaed657bd5344ea9cd78380deb1c7cc77c62c135ebea1237b8',
}


def verify_adapter3(result,pins):
    """Dispatch only the exact reviewed multi-anchor adapter/owner/worker chain."""
    if result['schema'] != ADAPTER3_SCHEMA:
        return
    require(result['state']=='original_bulk_q3_normalized_pending_group_or_individual_review'
            and result['input_route']=='bulk_q3'
            and result['authorized_group_or_individual_review_required'] is True,
            'Adapter3 review scope differs')
    adapter=WORKSPACE/'examples/greenhouse_sim/sim_data/native848_diverse_bulk_adapter_v3.py'
    require(result['implementation']==dict(path=str(adapter.resolve()),sha256=ADAPTER3_PINS[
        'examples/greenhouse_sim/sim_data/native848_diverse_bulk_adapter_v3.py']),
        'Unknown adapter3 implementation')
    for relative,expected in ADAPTER3_PINS.items():
        p=(WORKSPACE/relative).resolve()
        require(result['source_bindings'].get(str(p))==expected,'Unbound adapter3 implementation chain')
        pins.add(dict(path=str(p),sha256=expected))


def inventory(result_path,result_sha256,hold_snapshot,policy_spec,pins):
    result=pins.json(dict(path=str(Path(result_path).resolve()),sha256=result_sha256))
    require(result['schema'] in ('greenhouse.native848_diverse_bulk_q3_adapter.v1',
        'greenhouse.native848_diverse_bulk_q4_background_post.v1',
        'greenhouse.native848_diverse_bulk_q3_adapter.v2',
        'greenhouse.native848_diverse_bulk_q3_adapter.v3'),'Unsupported normalization')
    verify_adapter3(result,pins)
    require(policy_spec==pin(POLICY_PATH),'Exact saved user policy required')
    policy=pins.json(policy_spec)
    if result['schema'] in ('greenhouse.native848_diverse_bulk_q3_adapter.v2','greenhouse.native848_diverse_bulk_q3_adapter.v3'):
        require(result['review_policy']==policy_spec and result['authorized_group_or_individual_review_required'] is True,
                'Adapter v2 policy identity differs')
    require(policy['schema']=='greenhouse.native848_group_review_user_policy.v1'
        and policy['preserve_per_image_automated_gates'] is True
        and policy['group_representatives_and_flagged_visual_review_required'] is True
        and policy['individual_review_of_every_image_required'] is False,'User group policy differs')
    raw=pins.add(hold_snapshot).read_bytes(); held=held_aliases(raw)
    rows=result['records']; ids=[r['sample_id'] for r in rows]
    require(len(ids)==len(set(ids)),'Duplicate population ID')
    descriptions=[]; excluded=[]
    for row in rows:
        if aliases(row)&held:
            excluded.append(dict(sample_id=row['sample_id'],reason='preserved_manual_hold_alias'))
        else:descriptions.append(describe(row,pins))
    groups,records=select_groups(descriptions)
    return dict(schema=SCHEMA,protocol=PROTOCOL,implementation=pin(__file__),
        normalized_result=dict(path=str(Path(result_path).resolve()),sha256=result_sha256),
        user_policy=policy_spec,hold_snapshot=hold_snapshot,
        original_candidate_ids=ids,excluded_manual_holds=excluded,
        groups=groups,records=records,source_bindings=pins.values,
        actual_visual_review_performed=False,training_approved=False)


def prepare_group_review(normalized_result_path,result_sha256,hold_ledger_path,output):
    out=Path(output).resolve();require(not out.exists(),'Output exists')
    require(out.is_relative_to(WORKSPACE),'Output outside workspace')
    raw=Path(hold_ledger_path).read_bytes();out.mkdir(parents=True)
    hp=out/'manual_holds_snapshot.jsonl'
    with hp.open('xb') as f:f.write(raw)
    pins=Pins(); inv=inventory(normalized_result_path,result_sha256,pin(hp),pin(POLICY_PATH),pins)
    write(out/'protocol.json',dict(**PROTOCOL,user_policy=inv['user_policy']))
    rows={r['sample_id']:r for r in read(normalized_result_path)['records']}
    samples=[]
    for sid,d in inv['records'].items():
        if not d['selected_for_visual_review']:continue
        row=rows[sid]; a=row['artifacts']; rgb=pins.add(a['rgb'])
        require(digest(rgb)==row['rgb_sha256'],'RGB alias differs')
        label=pins.json(a['label'])
        points=[label['nominal_pixel_uv'],label['query_pixel_uv']]+label['accepted_interval_uv']
        box=[max(0,math.floor(min(p[0] for p in points)-56)),
             max(0,math.floor(min(p[1] for p in points)-56)),
             min(848,math.ceil(max(p[0] for p in points)+56)),
             min(408,math.ceil(max(p[1] for p in points)+56))]
        cp=out/(sid+'.png')
        require(cp.parent==out,'Unsafe sample ID')
        with Image.open(rgb) as im:
            require(im.mode=='RGB' and im.size==(848,408),'Native RGB differs')
            im.crop(box).save(cp)
        samples.append(dict(observation_id=sid,rgb_path=str(rgb),rgb_sha256=row['rgb_sha256'],
            label_path=a['label']['path'],label_sha256=a['label']['sha256'],
            source_sample_sha256=row['source_sample_sha256'],cut_pixel_uv=label['nominal_pixel_uv'],
            query_pixel_uv=label['query_pixel_uv'],crop_box_xyxy=box,crop_path=str(cp),
            crop_sha256=digest(cp),eligible_automated_candidate=True,group_id=d['group_id'],
            flagged=d['flagged'],flag_reasons=d['flags']))
    inv['source_bindings']=pins.values
    write(out/'inventory.json',inv)
    write(out/'visual_packet.json',dict(schema='greenhouse.native848_diverse_group_visual_packet.v2',
        inventory=pin(out/'inventory.json'),samples=samples,screening_only=False,
        actual_visual_review_performed=False))
    return dict(inventory=pin(out/'inventory.json'),packet=pin(out/'visual_packet.json'),
                groups=len(inv['groups']),members=len(inv['records']),selected=len(samples))


def validate_group_review(normalized_result_path,result_sha256,review_path,review_sha256):
    pins=Pins(); review=pins.json(dict(path=str(Path(review_path).resolve()),sha256=review_sha256))
    require(review['schema']==REVIEW_SCHEMA and review['wrapper_result_sha256']==result_sha256,
            'Group review source differs')
    inv=pins.json(review['inventory'])
    require(inv['implementation']==pin(__file__) and inv['protocol']==PROTOCOL,'Changed protocol/helper')
    replay=inventory(normalized_result_path,result_sha256,inv['hold_snapshot'],inv['user_policy'],pins)
    for key in ('normalized_result','original_candidate_ids','excluded_manual_holds','groups','records'):
        require(inv[key]==replay[key],'Changed deterministic group coverage: '+key)
    packet=pins.json(review['packet'])
    require(packet['inventory']==review['inventory'],'Packet inventory differs')
    selected={sid for sid,d in inv['records'].items() if d['selected_for_visual_review']}
    packet_rows={r['observation_id']:r for r in packet['samples']}
    actual={r['sample_id']:r for r in review['reviews']}
    require(len(packet_rows)==len(packet['samples']) and set(packet_rows)==selected,'Packet selection differs')
    require(len(actual)==len(review['reviews']) and set(actual)==selected,
            'Every representative and flagged frame requires exactly one actual review')
    normalized_rows={r['sample_id']:r for r in read(normalized_result_path)['records']}
    for sid,a in actual.items():
        d=inv['records'][sid]; p=packet_rows[sid]
        for k in ('rgb_sha256','label_sha256','source_sample_sha256'):
            require(a[k]==d[k] and p[k]==d[k],'Actual review artifact differs')
        require(a['reviewer_type'] in ('assistant','human') and a['reviewer'].strip()
            and a['reason'].strip(),'Attributed actual decision required')
        require(a['decision'] in ('accept','hold','reject') and
            a['full_native_image_inspected'] is True and
            a['unscaled_lossless_association_crop_inspected'] is True,'Actual full/crop inspection required')
        if a['decision']=='accept':
            require(a['obvious_ghosting'] is False and a['target_query_association_clear'] is True,
                    'Ambiguous or ghosted sample cannot accept')
        row=normalized_rows[sid]
        label=pins.json(row['artifacts']['label'])
        points=[label['nominal_pixel_uv'],label['query_pixel_uv']]+label['accepted_interval_uv']
        expected_box=[max(0,math.floor(min(q[0] for q in points)-56)),
            max(0,math.floor(min(q[1] for q in points)-56)),
            min(848,math.ceil(max(q[0] for q in points)+56)),
            min(408,math.ceil(max(q[1] for q in points)+56))]
        require(p['crop_box_xyxy']==expected_box and
            p['cut_pixel_uv']==label['nominal_pixel_uv'] and p['query_pixel_uv']==label['query_pixel_uv']
            and Path(p['label_path']).resolve()==Path(row['artifacts']['label']['path']).resolve(),
            'Association crop or label path differs')
        rgb=pins.add(dict(path=p['rgb_path'],sha256=p['rgb_sha256']))
        crop=pins.add(dict(path=p['crop_path'],sha256=p['crop_sha256']))
        with Image.open(rgb) as im,Image.open(crop) as cr:
            require(im.mode=='RGB' and im.size==(848,408) and cr.mode=='RGB','Native review image differs')
            exact=im.crop(p['crop_box_xyxy'])
            require(exact.size==cr.size and exact.tobytes()==cr.tobytes(),'Crop is not exact/unscaled')
    decisions={r['group_id']:r for r in review['group_decisions']}
    require(len(decisions)==len(review['group_decisions']) and
            set(decisions)=={g['group_id'] for g in inv['groups']},'Every group needs one decision')
    output={}
    for g in inv['groups']:
        decision=decisions[g['group_id']]
        require(decision['decision'] in ('accept','hold') and decision['reason'].strip(),'Invalid group decision')
        expected='accept' if all(actual[s]['decision']=='accept' for s in g['selected_sample_ids']) else 'hold'
        require(decision['decision']=='hold' or expected=='accept','Ambiguous representative holds whole group')
        for sid in g['sample_ids']:
            d=inv['records'][sid]
            output[sid]=dict(group_id=g['group_id'],review_scope=('individual' if sid in actual else 'target_view_group'),
                individual_visual_review=sid in actual,decision=decision['decision'],
                selected_for_visual_review=d['selected_for_visual_review'],flagged=d['flagged'],
                actual_individual_decision=actual[sid]['decision'] if sid in actual else None,
                actual_review=actual.get(sid))
    for d in inv['excluded_manual_holds']:
        output[d['sample_id']]=dict(group_id=None,review_scope='preserved_manual_hold',
            individual_visual_review=False,decision='hold',selected_for_visual_review=False,flagged=True)
    pins.add(pin(__file__))
    return dict(records=output,source_bindings=pins.values,inventory=review['inventory'],
        packet=review['packet'],user_policy=inv['user_policy'],protocol=PROTOCOL,
        current_global_holds_must_be_rechecked_by_exporter=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result',required=True);p.add_argument('--result-sha256',required=True)
    p.add_argument('--hold-ledger',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(prepare_group_review(a.result,a.result_sha256,a.hold_ledger,a.output)))


if __name__=='__main__':main()

