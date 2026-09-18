"""Opt-in native848 FXAA and advancing render-clock benchmark; no TRAIN admission.

CPU prepare/audit are separate CLI actions. Only an explicit capture action starts
Isaac, under an external serial owner. Frozen direct/reset contracts are unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import importlib.util
import time
import traceback
from copy import deepcopy
import numpy as np
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import jsonable, fingerprint, project, depth_evidence, write_sample
from .capture_sensor import LEGACY_RESOLUTION, sensor_profile, calibration_for_native_resolution
from .native_sensor_payload import validate_native_payload, validate_native_static, decode_native_instances
from .native_greenhouse_pair import assert_same_camera
from .native848_pair_audit_v2 import expected_catalogue, world_from_row, verify_camera, image_array
from . import native848_direct_plan_v2 as direct
from . import native848_reset_probe_v2 as prior
from .capture_visibility import component_masks, interval_visibility, view_quality, write_visibility

ROOT = Path(__file__).resolve().parents[3]
D = ROOT/'data/sim_data/diagnostics'
SCHEMA = 'greenhouse.native848_fast_profile_benchmark_plan.v3'
SAMPLE = 'greenhouse.native848_fast_profile_benchmark_sample.v3'
STATE = 'native848_fast_profile_benchmark_complete_diagnostic_only'
PREDECESSOR = dict(
    module_path=str(Path(__file__).with_name('native848_short_validation_v4.py')),
    module_sha256='2f2ac2243411a0122501c5e3243f45ea2e936f44e2eeaf54f834ec294c52cc8b',
    trial_path=str(D/'native848_short_validation_trial_20260916_v4'),
    result_sha256='9f8577042e32775034804f6ba1bdc44830b4d6399047500795507cc0c176f6c8',
    capture_result_sha256='5a9c323541fe33eff8dc18a5678675a613904417fe98cc4cde698f44481e4d7b',
    audit_sha256='76695169896634e9bc74753fc3db3c0e41a6f98c14fcafd9e3ea277a284030cf',
    owned_exit_sha256='38bc939ff374e2294c670ae59c18c996961ec814378f658be074c473fa84ca92')
BUDGETS = [1, 2, 4]
A, B = 'SubStem_42_view_002', 'SubStem_44_view_002'
PROFILE = 'rt2_fxaa_scheduler_1over60_preserved_native_reference.v2'
NATIVE_REFERENCE_POLICY = 'preserved_callback_clock_nondecreasing_not_frame_identity'
SCHEDULER_POLICY = 'observed_timeline_delta_after_initialization'
CLOCK_DELTA = 1.0/60.0
PROFILE_SETTINGS = {'/rtx/rendermode':'RealTimePathTracing', '/rtx/post/aa/op':2,
    '/rtx-transient/post/aa/limitedOps':False, '/rtx-transient/dlssg/enabled':False,
    '/omni/replicator/captureMotionBlur':False}
OBSERVED_SETTINGS = list(dict.fromkeys([*prior.SETTINGS, *PROFILE_SETTINGS,
    '/rtx/hydra/supportMultiTickRate', '/persistent/rtx/modes/rt2/enabled']))
INSTALLED = Path('D:/isaac-sim-6.0.1/extscache/omni.replicator.core-1.13.27+110.1.1.wx64.r.cp312/omni/replicator/core/scripts')


def apply_profile(rep, settings):
    # Installed API explicitly recommends FXAA for non-sequential data.
    rep.settings.set_render_rtx_realtime(antialiasing='FXAA')
    for key, value in PROFILE_SETTINGS.items():
        settings.set(key, value)
    require(all(settings.get(k)==v for k,v in PROFILE_SETTINGS.items()), 'Candidate setting readback mismatch')
    print('FAST848_EXPLICIT_PROFILE', PROFILE, {k:settings.get(k) for k in OBSERVED_SETTINGS}, flush=True)


def freeze_and_prove_clock_safety(stage):
    from pxr import Usd, UsdPhysics, PhysxSchema
    from .capture_scene import freeze_rigid_bodies
    freeze_rigid_bodies(stage)
    disabled_articulations=[]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                api=PhysxSchema.PhysxArticulationAPI(prim)
                if api.GetArticulationEnabledAttr().Get():
                    api.CreateArticulationEnabledAttr(False)
                    disabled_articulations.append(str(prim.GetPath()))
    bodies=[]; joints=[]; articulations=[]; animated=[]; unsupported=[]
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if path.startswith(('/Render','/Replicator','/Orchestrator')): continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            value=UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            bodies.append([path,value]);require(value is False,'Enabled rigid body in advancing-clock scene')
        if prim.IsA(UsdPhysics.Joint):
            value=UsdPhysics.Joint(prim).GetJointEnabledAttr().Get()
            joints.append([path,value]);require(value is False,'Enabled joint in advancing-clock scene')
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            value=PhysxSchema.PhysxArticulationAPI(prim).GetArticulationEnabledAttr().Get()
            articulations.append([path,value]);require(value is False,'Enabled articulation in advancing-clock scene')
        for attr in prim.GetAttributes():
            if attr.GetNumTimeSamples()>0: animated.append(str(attr.GetPath()))
        for schema in prim.GetAppliedSchemas():
            if any(s in str(schema).lower() for s in ('particle','deformable','cloth')): unsupported.append([path,str(schema)])
    require(not animated and not unsupported,'Animated or unsupported dynamic source cannot use advancing render clock')
    return dict(schema='greenhouse.native848_render_clock_static_scene.v1', rigid_bodies=bodies,
        joints=joints,articulations=articulations,session_disabled_articulations=disabled_articulations,
        authored_time_sampled_attributes=animated,unsupported_dynamic_schemas=unsupported,
        physics_motion_allowed=False,profile=PROFILE,delta_time_seconds=CLOCK_DELTA,
        static_scene_guard_still_required=True)


def single_request(rep, writer, subframes):
    import omni.timeline
    import omni.kit.app
    timeline=omni.timeline.get_timeline_interface();timeline.pause()
    before=writer.sequence;writer.request_index+=1
    updates=[]
    subscription=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        lambda event: updates.append(time.perf_counter()), name='fast848_measure_updates')
    started=time.perf_counter();clock_before=timeline.get_current_time()
    try:
        rep.orchestrator.step(rt_subframes=subframes,pause_timeline=True,
            delta_time=CLOCK_DELTA,wait_for_render=True)
    finally:
        writer.last_step_timing=dict(orchestrator_step_wall_seconds=time.perf_counter()-started,
            app_update_event_count=len(updates),timeline_before_seconds=clock_before,
            timeline_after_seconds=timeline.get_current_time(),delta_time_seconds=CLOCK_DELTA,
            wait_for_render=True,app_update_times_relative=[x-started for x in updates])
        subscription.unsubscribe()
    if writer.capture_error is not None: raise RuntimeError('Native callback failed') from writer.capture_error
    require(writer.latest is not None and writer.sequence>before,'No fresh callback from single request')
    return writer.latest


def verify_clock_request(evidence, folder):
    from fractions import Fraction
    require(evidence['delta_time_seconds']==CLOCK_DELTA and evidence['wait_for_render'] is True
        and evidence['native_reference_time_policy']==NATIVE_REFERENCE_POLICY
        and evidence['scheduler_clock_policy']==SCHEDULER_POLICY,'Changed explicit clock policy')
    for i,request in enumerate(evidence['requests']):
        payload=load_raw(folder/f'callback_{i:02d}')
        reference=list(map(int,payload['reference_time']))
        header=payload['ReferenceTime']
        require(reference==request['reference_time'] and Fraction(*reference)==Fraction(
            int(header['referenceTimeNumerator']),int(header['referenceTimeDenominator'])),
            'Native reference differs from the same callback header')
        previous=request['previous_reference_time']
        if previous is not None:
            require(Fraction(*reference)>=Fraction(*previous),'Native callback reference moved backwards')
        timing=request['step_timing']
        advance=timing['timeline_after_seconds']-timing['timeline_before_seconds']
        initial=request['request_index_after']==1
        require(abs(advance-CLOCK_DELTA)<1e-7 or (initial and abs(advance)<1e-7),
            'Observed scheduler timeline did not advance by requested delta')
        require(timing['delta_time_seconds']==CLOCK_DELTA and timing['wait_for_render'] is True
            and timing['app_update_event_count']>=1,'Missing actual update/timing evidence')
    return evidence


FLAGS = dict(training_approved=False, production_profile_qualified=False,
             accepted_training_increment=0, source_cap_reset=False,
             geometry_novelty_qualified=False, individual_visual_review_performed=False, qualified_for_bulk_production=False, renderer_backend_log_review_required=True)
SOURCES = {
    'transition_plan': D/'native848_direct_camera_fixed56_seed410502_prepare_20260916_v2/plan.json',
    'negative_plan': D/'native848_direct_camera_fixed56_prepare_20260916_v2/plan.json',
    'admission': D/'native848_prefiltered_direct_seed410502_admission_20260916_v1/result.json',
    'review': ROOT/'data/sim_data/dataset_reviews/native848_prefiltered_direct_seed410502_20260916_v1/assistant_reviews.json',
    'negative_sample': D/'native848_direct_camera_fixed56_trial_20260916_v2/capture/SubStem_44_view_002/sample.json',
}
SOURCE_PINS = {
    'transition_plan': '3d40345791744605e48cb69cc9d69474aba82a3557223757caa494c6de1515af',
    'negative_plan': '8be1be3410b0a726737b5a94ab61161b49419532a2ce9d8cb5bf8d5da614a5ad',
    'review': 'bdb263c3d8635f131562064d72b3ef83b4a91620940081f2243b78a0b77c1071',
}


def implementation_bindings():
    root = Path(__file__).resolve().parent
    roots = (root.parent, root.parents[1])
    pending, found = [Path(__file__).resolve()], {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        found[str(path)] = sha256(path)
        base = next(p for p in roots if path.is_relative_to(p))
        package = '.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.'*node.level+(node.module or '')
                name = importlib.util.resolve_name(name, package) if node.level else name
                names = [name]+[name+'.'+a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in roots:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate/'__init__.py') if p.is_file())
    return dict(sorted(found.items()))


def schedule(scene):
    return [dict(name=f'b{budget}_{phase}', budget=budget, sample_id=sample,
                 role=role) for budget in BUDGETS for phase, sample, role in
            ([('A_before', A, 'clear_reference'), ('B', B, 'historical_trace_hold'),
              ('A_return', A, 'clear_reference')] if scene == 'transition' else
             [('fruit_negative', B, 'known_fruit_negative')])]


def prepare():
    plan = dict(schema=SCHEMA, resolution=[848, 408], budgets=BUDGETS,
        predecessor=deepcopy(PREDECESSOR),
        profile=PROFILE, candidate_settings=PROFILE_SETTINGS, clock_delta_seconds=CLOCK_DELTA,
        native_reference_time_policy=NATIVE_REFERENCE_POLICY,scheduler_clock_policy=SCHEDULER_POLICY,
        installed_source_bindings={str(INSTALLED/name):sha256(INSTALLED/name) for name in ('settings.py','orchestrator.py','annotators_default.py','../ogn/python/impl/nodes/OgnWriter.py')}, sources={k:dict(path=str(p.resolve()), sha256=sha256(p)) for k,p in SOURCES.items()},
        implementation_bindings=implementation_bindings(), schedules={s:schedule(s) for s in ('transition','negative')},
        control_order=['A_before','B','A_return','occluder_on','occluder_off'],
        warmup_request_subframes=[8]*7, warmup_per_product=1,
        full_scene_observations=12, control_observations=15,
        equality_is_diagnostic_only=True, reset_api='omni.usd.get_context().reset_renderer_accumulation',
        requires_new_explicit_admission_consumer=True, **FLAGS)
    check(plan)
    return plan


def check(plan, full=False):
    require(plan['predecessor']==PREDECESSOR
        and sha256(PREDECESSOR['module_path'])==PREDECESSOR['module_sha256'],'Changed preserved predecessor module')
    previous=Path(PREDECESSOR['trial_path'])
    for name,key in (('result.json','result_sha256'),('capture/result.json','capture_result_sha256'),
                     ('audit.json','audit_sha256'),('owned_exit.json','owned_exit_sha256')):
        require(sha256(previous/name)==PREDECESSOR[key],'Changed completed v4 evidence '+name)
    terminal=read_json(previous/'result.json');owned=read_json(previous/'owned_exit.json')
    require(terminal['state']=='owned_short848_v4_diagnostic_captured_and_replayed_no_profile_approval'
        and terminal['production_profile_qualified'] is False and terminal['training_approved'] is False
        and all(terminal[k]==PREDECESSOR[k] for k in ('capture_result_sha256','audit_sha256','owned_exit_sha256'))
        and owned['returncode']==0 and owned['method']=='subprocess_wait_on_owned_process'
        and owned['launch_sha256']==sha256(previous/'launch.json')
        and owned['pid']==read_json(previous/'launch.json')['pid']
        and read_json(previous/'audit.json')['state']=='native848_short_validation_saved_buffer_replay_complete'
        and not (previous/'failure.json').exists(),'Incomplete v4 owned capture/replay')
    outcome=read_json(previous/'capture/result.json')['budget_status']
    require(set(outcome)=={'1','2','4'} and outcome['4']['controls_passed'] is True
        and outcome['4']['all_transition_frames_capture_valid'] is True
        and outcome['4']['both_A_frames_automated_passed'] is False
        and all(v['production_profile_qualified'] is False for v in outcome.values()),'V4 follow-up evidence differs')
    require(plan['schema']==SCHEMA and plan['resolution']==[848,408] and plan['budgets']==BUDGETS
        and plan['profile']==PROFILE and plan['schedules']=={s:schedule(s) for s in ('transition','negative')}
        and plan['control_order']==['A_before','B','A_return','occluder_on','occluder_off']
        and plan['warmup_request_subframes']==[8]*7 and plan['warmup_per_product']==1
        and plan['full_scene_observations']==12 and plan['control_observations']==15
        and plan['candidate_settings']==PROFILE_SETTINGS and plan['clock_delta_seconds']==CLOCK_DELTA
        and plan['native_reference_time_policy']==NATIVE_REFERENCE_POLICY and plan['scheduler_clock_policy']==SCHEDULER_POLICY
        and plan['equality_is_diagnostic_only'] is True
        and plan['reset_api']=='omni.usd.get_context().reset_renderer_accumulation'
        and plan['requires_new_explicit_admission_consumer'] is True
        and all(plan[k]==v for k,v in FLAGS.items()), 'Changed short diagnostic scope')
    require(plan['implementation_bindings']==implementation_bindings(), 'Changed implementation closure')
    require(plan['installed_source_bindings']=={str(INSTALLED/name):sha256(INSTALLED/name) for name in ('settings.py','orchestrator.py','annotators_default.py','../ogn/python/impl/nodes/OgnWriter.py')},'Changed installed renderer/orchestrator API')
    verify_bindings(plan['implementation_bindings'])
    require(set(plan['sources'])==set(SOURCES), 'Changed source roles')
    for key, info in plan['sources'].items():
        require(Path(info['path']).resolve()==SOURCES[key].resolve() and sha256(info['path'])==info['sha256'], 'Changed source '+key)
        if key in SOURCE_PINS:
            require(info['sha256']==SOURCE_PINS[key], 'Unexpected historical source '+key)
    loaded = {s:direct.check(read_json(plan['sources'][s+'_plan']['path']), full=full) for s in ('transition','negative')}
    admission = read_json(plan['sources']['admission']['path'])
    verify_bindings(admission['source_bindings'])
    records = {r['sample_id']:r for r in admission['records']}
    clear = records[A]
    require(all(clear[k] is True for k in ('local_clarity_passed','full_trace_and_anchored_grid_passed','workspace_passed')),
        'Clear source role is unsupported')
    require(records[B]['full_trace_and_anchored_grid_passed'] is False, 'B historical hold changed')
    for name in (A,B):
        item=records[name]
        for field in ('source_sample','rgb','label','workspace'):
            require(sha256(item[field+'_path'])==item[field+'_sha256'], 'Changed source proof')
        require(sha256(Path(item['label_path']).parent/'query_trace.json')==item['query_trace_sha256'], 'Changed source trace')
        rec=next(r for r in loaded['transition'][0]['records'] if r['sample_id']==name)
        meta=read_json(item['source_sample_path'])
        assert_same_camera(meta['calibration'],rec['calibration'])
        require(meta['supervision']['target_id']==rec['target_id'], 'Source target/cache mismatch')
    reviews=read_json(plan['sources']['review']['path'])
    require(any(r['sample_id']==A and r['decision']=='accept' and r['rgb_sha256']==clear['rgb_sha256']
        and r['label_sha256']==clear['label_sha256'] and r['source_sample_sha256']==clear['source_sample_sha256']
        and r['full_native_image_inspected'] is True and r['unscaled_lossless_association_crop_inspected'] is True
        for r in reviews), 'Actual reviewed clear source required')
    folder=Path(plan['sources']['negative_sample']['path']).parent
    negative=read_json(folder/'sample.json')
    rec=next(r for r in loaded['negative'][0]['records'] if r['sample_id']==B)
    assert_same_camera(negative['calibration'],rec['calibration'])
    require(negative['supervision']['target_id']==rec['target_id'], 'Negative target/cache mismatch')
    for name,info in negative['files'].items():
        require(sha256(folder/name)==info['sha256'], 'Changed negative buffer')
    ids=np.load(folder/'supervision/renderer_instance_id.npy',allow_pickle=False)
    mapping={int(k):v for k,v in read_json(folder/'supervision/identities.json')['renderer_id_to_prim'].items()}
    probes=prior.interval_probes(negative['supervision']['projected_interval'],
        np.load(folder/'inputs/depth_m.npy',allow_pickle=False),image_array(folder/'inputs/depth_valid.png')==255,ids,mapping)
    require(any('/Fruit_07' in (p.get('prim_path') or '') for p in probes), 'Observed Fruit_07 negative required')
    return loaded


def save_raw(folder, payload):
    folder.mkdir(parents=True,exist_ok=False)
    for key,name in [('rgb','rgba.npy'),('distance_to_image_plane','depth_m.npy')]:
        np.save(folder/name,np.asarray(payload[key]),allow_pickle=False)
    annotation=payload['instance_id_segmentation']
    np.save(folder/'instance_id.npy',np.asarray(annotation['data']),allow_pickle=False)
    write_json(folder/'instance_info.json',jsonable(annotation['info']))
    write_json(folder/'payload.json',jsonable({k:v for k,v in payload.items() if k not in
        ('rgb','distance_to_image_plane','instance_id_segmentation')}))


def load_raw(folder):
    payload=read_json(folder/'payload.json')
    payload['rgb']=np.load(folder/'rgba.npy',allow_pickle=False)
    payload['distance_to_image_plane']=np.load(folder/'depth_m.npy',allow_pickle=False)
    payload['instance_id_segmentation']=dict(data=np.load(folder/'instance_id.npy',allow_pickle=False),
        info=read_json(folder/'instance_info.json'))
    return payload


def save_callback_tree(folder,value):
    """Lossless copied callback before FAST normalization, including every array."""
    folder.mkdir(parents=True,exist_ok=False)
    count=0
    def encode(v):
        nonlocal count
        if isinstance(v,np.ndarray):
            if v.dtype==object:return {'object_array':[encode(x) for x in v.flat],'shape':list(v.shape)}
            name=f'array_{count:04d}.npy';count+=1
            np.save(folder/name,v,allow_pickle=False)
            return {'array':name}
        if isinstance(v,np.generic):return encode(v.item())
        if isinstance(v,bytes):return {'bytes_hex':v.hex()}
        if isinstance(v,dict):return {'dict':[[encode(k),encode(x)] for k,x in v.items()]}
        if isinstance(v,tuple):return {'tuple':[encode(x) for x in v]}
        if isinstance(v,list):return {'list':[encode(x) for x in v]}
        if isinstance(v,float) and not np.isfinite(v):return {'nonfinite':repr(v)}
        if v is None or type(v) in (str,int,float,bool):return {'value':v}
        raise TypeError('Unsupported native callback value '+str(type(v)))
    write_json(folder/'tree.json',encode(value))


def load_callback_tree(folder):
    def decode(v):
        if 'array' in v:return np.load(folder/v['array'],allow_pickle=False)
        if 'object_array' in v:return np.asarray([decode(x) for x in v['object_array']],dtype=object).reshape(v['shape'])
        if 'bytes_hex' in v:return bytes.fromhex(v['bytes_hex'])
        if 'dict' in v:return {decode(k):decode(x) for k,x in v['dict']}
        if 'tuple' in v:return tuple(decode(x) for x in v['tuple'])
        if 'list' in v:return [decode(x) for x in v['list']]
        if 'nonfinite' in v:return float(v['nonfinite'])
        return v['value']
    return decode(read_json(folder/'tree.json'))


def recording_writer(rep,unrequested_folder):
    from .capture_pilot import make_writer
    from .native_instances import normalize_payload
    # Reuse installed annotator registration, never modify its class/instances.
    template=make_writer(rep,include_instances=True,instance_backend='fast')
    class DiagnosticWriter(rep.Writer):
        def __init__(self):
            self.version='native848-fast-profile-benchmark-v3'
            self.annotators=template.annotators
            self.sequence=0;self.request_index=0;self.latest=None;self.capture_error=None
            self.folder=unrequested_folder;self.events=[];self.latest_event=None;self.previous_reference=None;self.last_step_timing=None
        def begin_capture(self,folder):
            self.folder=folder;self.events=[];self.latest=None;self.capture_error=None;self.latest_event=None
        def write(self,data):
            self.sequence+=1
            event=dict(callback_sequence=self.sequence,request_index=self.request_index,normalization_passed=False)
            self.events.append(event)
            folder=self.folder/f'callback_{self.sequence:06d}'
            event['folder']=str(folder.resolve())
            try:
                t=time.perf_counter();raw=deepcopy(data);event['copy_seconds']=time.perf_counter()-t
                t=time.perf_counter();save_callback_tree(folder,raw);event['raw_persistence_seconds']=time.perf_counter()-t
                t=time.perf_counter();self.latest=normalize_payload(raw,'fast');event['normalization_seconds']=time.perf_counter()-t
                event['normalization_passed']=True;self.latest_event=event
            except Exception as exc:
                self.capture_error=exc;event['error']=traceback.format_exc()
                raise
            finally:
                folder.mkdir(parents=True,exist_ok=True)
                write_json(folder/'callback_receipt.json',event)
        def write_metadata(self):self._is_metadata_written=True
    return DiagnosticWriter()


def take(rep, writer, settings, folder, steps):
    """Save all raw callbacks before normalization and each attempt's counters."""
    import omni.usd
    folder.mkdir(parents=True,exist_ok=False)
    writer.begin_capture(folder/'all_callbacks')
    observed={k:jsonable(settings.get(k)) for k in OBSERVED_SETTINGS}
    evidence=dict(requested_steps=steps,settings=observed,requests=[],hidden_settling=False,
        delta_time_seconds=CLOCK_DELTA,wait_for_render=True,
        native_reference_time_policy=NATIVE_REFERENCE_POLICY,scheduler_clock_policy=SCHEDULER_POLICY)
    started=time.perf_counter()
    try:
        evidence['effective_subframe_policy']=[prior.checked_subframe_floor(observed,n) for n in steps]
        value=omni.usd.get_context().reset_renderer_accumulation()
        evidence['reset']=dict(api='omni.usd.get_context().reset_renderer_accumulation',
            returned_without_exception=True,returned_value=jsonable(value),temporal_cleanliness_proven=False)
        for i,n in enumerate(steps):
            seq,req=writer.sequence,writer.request_index
            attempt=dict(subframes=n,callback_sequence_before=seq,request_index_before=req,
                previous_reference_time=writer.previous_reference)
            try:
                payload=single_request(rep,writer,n)
                t=time.perf_counter();save_raw(folder/f'callback_{i:02d}',payload)
                attempt['normalized_persistence_seconds']=time.perf_counter()-t
                attempt['reference_time']=list(map(int,payload['reference_time']))
                writer.previous_reference=attempt['reference_time']
                attempt['normalized_from_callback_sequence']=writer.latest_event['callback_sequence']
            except BaseException:
                attempt['error']=traceback.format_exc();raise
            finally:
                attempt.update(native_requests=writer.request_index-req,request_index_after=writer.request_index,
                    callback_sequence_after=writer.sequence,step_timing=writer.last_step_timing)
                evidence['requests'].append(attempt)
        require(all(r['native_requests']==1 and r['callback_sequence_after']>r['callback_sequence_before']
            for r in evidence['requests']), 'Retry-inflated or stale short capture')
        require(observed=={k:jsonable(settings.get(k)) for k in OBSERVED_SETTINGS}, 'Settings changed during capture')
        verify_clock_request(evidence,folder)
        return payload,evidence
    except BaseException:
        evidence['error']=traceback.format_exc()
        raise
    finally:
        evidence['callbacks']=writer.events
        writer.folder=folder.parent/'unrequested_callbacks';writer.events=[]
        evidence['elapsed_seconds']=time.perf_counter()-started
        write_json(folder/'request_evidence.json',evidence)


def verify_request(folder,steps):
    from .native_instances import normalize_payload
    evidence=read_json(folder/'request_evidence.json')
    require(evidence['requested_steps']==steps and len(evidence['requests'])==len(steps)
        and evidence['hidden_settling'] is False and 'error' not in evidence
        and evidence['reset']['api']=='omni.usd.get_context().reset_renderer_accumulation'
        and evidence['reset']['returned_without_exception'] is True
        and evidence['reset']['temporal_cleanliness_proven'] is False
        and evidence['effective_subframe_policy']==[prior.checked_subframe_floor(evidence['settings'],n) for n in steps],
        'Invalid short budget/reset receipt')
    events=evidence['callbacks']
    require({p.name for p in (folder/'all_callbacks').iterdir()}=={f"callback_{e['callback_sequence']:06d}" for e in events},'Unaccounted raw callbacks')
    require(events and [e['callback_sequence'] for e in events]==list(range(events[0]['callback_sequence'],events[-1]['callback_sequence']+1)),
        'Missing callback evidence')
    for i,(request,n) in enumerate(zip(evidence['requests'],steps)):
        require(request['subframes']==n and request['native_requests']==1
            and request['request_index_after']-request['request_index_before']==1
            and request['callback_sequence_after']>request['callback_sequence_before'] and 'error' not in request,
            'Retry-inflated or stale request')
        applicable=[e for e in events if request['callback_sequence_before']<e['callback_sequence']<=request['callback_sequence_after']]
        require(applicable and all(e['normalization_passed'] and e['request_index']==request['request_index_after'] for e in applicable),
            'Invalid callback/request association')
        selected=next(e for e in applicable if e['callback_sequence']==request['normalized_from_callback_sequence'])
        require(selected==applicable[-1],'Returned payload is not the final fresh callback')
        for event in applicable:
            path=folder/'all_callbacks'/f"callback_{event['callback_sequence']:06d}"
            require(str(path.resolve())==event['folder'] and read_json(path/'callback_receipt.json')==event,'Changed callback identity')
            raw=load_callback_tree(path);normalized=normalize_payload(raw,'fast')
            if event==selected:
                saved=load_raw(folder/f'callback_{i:02d}')
                require(np.array_equal(normalized['rgb'],saved['rgb'])
                    and np.array_equal(normalized['distance_to_image_plane'],saved['distance_to_image_plane'],equal_nan=True),
                    'RGB/depth did not come from the same raw callback')
                ni,nm=decode_native_instances(normalized,LEGACY_RESOLUTION);si,sm=decode_native_instances(saved,LEGACY_RESOLUTION)
                require(np.array_equal(ni,si) and nm==sm,'IDs did not come from the same raw callback')
                require(jsonable({k:v for k,v in normalized.items() if k not in ('rgb','distance_to_image_plane','instance_id_segmentation')})
                    ==read_json(folder/f'callback_{i:02d}'/'payload.json'),'Camera/reference metadata differs from raw callback')
    verify_clock_request(evidence,folder)
    require(all(evidence['settings'].get(k)==v for k,v in PROFILE_SETTINGS.items()),'Unexpected candidate renderer settings')
    return evidence


def freshness(payload, cal, before, after, sequence, previous=None, expect_change=False):
    rgb,depth,valid,reference,token=validate_native_static(payload,cal,before,after,sequence)
    ids,mapping=decode_native_instances(payload,LEGACY_RESOLUTION)
    token['instance_sha256']=hashlib.sha256(ids.tobytes()).hexdigest()
    token['mapping_sha256']=fingerprint(mapping)
    token['reference_time']=reference
    if previous is not None:
        require(token['callback_sequence']>previous['callback_sequence'], 'Stale callback sequence')
        from fractions import Fraction
        require(Fraction(*reference)>=Fraction(*previous['reference_time']),'Native reference moved backwards')
        if expect_change:
            require(all(token[k]!=previous[k] for k in ('rgb_sha256','depth_sha256','instance_sha256')),
                'RGB/depth/IDs did not all change after camera or occluder transition')
    return rgb,depth,valid,reference,token,ids,mapping


def control_check(depth,valid,ids,mapping,expected_depth,expected_prim):
    checks=[]
    for x,y in ((424,204),(470,204)):
        observed=mapping.get(int(ids[y,x]))
        passed=bool(valid[y,x] and abs(float(depth[y,x])-expected_depth)<=.002 and observed==expected_prim)
        checks.append(dict(pixel_xy=[x,y],expected_depth_m=expected_depth,measured_depth_m=float(depth[y,x])
            if valid[y,x] else None,expected_prim=expected_prim,observed_prim=observed,passed=passed))
    return dict(passed=all(r['passed'] for r in checks),checks=checks)


def controls(app, output, renderer):
    output.mkdir(parents=True,exist_ok=False)
    import carb
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,UsdGeom,UsdLux
    from .capture_scene import calibration,scene_guard
    from .capture_pilot import make_writer
    from .review_camera import HEAD_CAMERA
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    cube=UsdGeom.Cube.Define(stage,'/World/CalibrationCube');cube.CreateSizeAttr(1)
    cubeop=cube.AddTranslateOp();cubeop.Set(Gf.Vec3d(0,0,-2.5))
    blocker=UsdGeom.Cube.Define(stage,'/World/KnownOccluder');blocker.CreateSizeAttr(.2)
    blocker.AddTranslateOp().Set(Gf.Vec3d(0,0,-1.1));blocker.CreateVisibilityAttr('invisible')
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1000)
    camera=UsdGeom.Camera.Define(stage,HEAD_CAMERA);camera.CreateFocalLengthAttr(24)
    camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*408/848)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100));cameraop=camera.AddTranslateOp();cameraop.Set(Gf.Vec3d(0))
    product=rep.create.render_product(HEAD_CAMERA,LEGACY_RESOLUTION)
    writer=recording_writer(rep,output/'unrequested_callbacks');writer.attach([product])
    settings=carb.settings.get_settings();apply_profile(rep,settings)
    write_json(output/'clock_safety.json',freeze_and_prove_clock_safety(stage))
    results=[];previous=None
    try:
        take(rep,writer,settings,output/'warmup',[8]*7)
        for budget in BUDGETS:
            for phase in ('A_before','B','A_return','occluder_on','occluder_off'):
                folder=output/f'b{budget}_{phase}';folder.mkdir()
                is_b=phase=='B';on=phase=='occluder_on'
                cameraop.Set(Gf.Vec3d(.15 if is_b else 0,0,0))
                cubeop.Set(Gf.Vec3d(0,0,-2.9 if is_b else -2.5))
                blocker.GetVisibilityAttr().Set('inherited' if on else 'invisible')
                cal=calibration_for_native_resolution(calibration(stage),LEGACY_RESOLUTION)
                before=scene_guard(stage);expected_depth=1. if on else (2.4 if is_b else 2.)
                expected_prim='/World/KnownOccluder' if on else '/World/CalibrationCube'
                receipt=dict(name=folder.name,budget=budget,calibration=cal,static_guard=before,
                    expected_depth_m=expected_depth,expected_prim=expected_prim,passed=False)
                try:
                    payload,request=take(rep,writer,settings,folder/'raw',[budget])
                    values=freshness(payload,cal,before,scene_guard(stage),writer.sequence,previous,phase!='A_before')
                    rgb,depth,valid,reference,token,ids,mapping=values
                    previous=token;check_result=control_check(depth,valid,ids,mapping,expected_depth,expected_prim)
                    receipt.update(passed=check_result['passed'],checks=check_result['checks'],freshness=token,
                        reference_time=reference,request_evidence=request,static_guard_after=scene_guard(stage))
                except Exception:
                    receipt['error']=traceback.format_exc()
                write_json(folder/'result.json',receipt);results.append(receipt)
    finally:
        writer.detach();product.destroy()
    result=dict(observations=results,budgets={str(n):all(r['passed'] for r in results if r['budget']==n) for n in BUDGETS},**FLAGS)
    write_json(output/'result.json',result)
    return result


def full_scene(app, output, plan, scene_name, loaded):
    import omni.replicator.core as rep
    from .native_scene import prepare_native_scene
    from .native848_direct_worker_v2 import apply_cached_pose
    from .generated_capture import substitute_plant
    from .capture_pilot import make_writer,source_hashes
    from .capture_scene import calibration,target_world_geometry
    from .capture_viewpoints import component_catalogue,static_obstacles,scene_triangle_refiner,visible_bounds
    from .training_screen import StaticBoundScreen
    from .static_geometry_cache import StaticGeometryScreenCache
    from .static_guard import StaticSceneMonitor
    from .review_camera import HEAD_CAMERA
    from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
    cache,plans,generated=loaded;by_id={r['sample_id']:r for r in cache['records']}
    anchor=plans[by_id[plan['schedules'][scene_name][0]['sample_id']]['pair_plan_path']]
    scene=prepare_native_scene(app,anchor);stage,robot=scene['stage'],scene['robot']
    apply_profile(rep,scene['settings'])
    original_source_pins=source_hashes(stage)
    changed=substitute_plant(stage,scene['original_variant'],scene['records'],scene['variants'],scene['generated'])
    verify_bindings(original_source_pins)
    substituted_source_pins=source_hashes(stage)
    pins=None
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'clock_safety.json',freeze_and_prove_clock_safety(stage))
    catalogue=component_catalogue(stage,changed['records'],[*scene['reports'],changed['report']],changed['variants'])
    reports={r['plant_id']:r for r in scene['reports']}
    require(catalogue==expected_catalogue(anchor,'generated_variant',reports,generated)
        and len(catalogue)==scene['counts']['components'],'Full scene catalogue changed')
    geometry=StaticGeometryScreenCache(stage,robot['root'],include_generated_plants=True)
    workspace=WorkspaceChecker();monitor=None
    product=rep.create.render_product(HEAD_CAMERA,LEGACY_RESOLUTION)
    writer=recording_writer(rep,output/'unrequested_callbacks');writer.attach([product])
    results=[];previous=None;warmed=False;checked_geometry=False
    try:
        for item in plan['schedules'][scene_name]:
            folder=output/item['name'];folder.mkdir(parents=True)
            receipt=dict(**item,passed_actual_automated_criteria=False,**FLAGS)
            try:
                record=by_id[item['sample_id']];pair=plans[record['pair_plan_path']];row=pair['generated_row']
                pose,cal=apply_cached_pose(stage,robot,record);screen=geometry()
                if not checked_geometry:
                    full=StaticBoundScreen(static_obstacles(stage,robot['root']),scene_triangle_refiner(stage,
                        include_generated_plants=True))(visible_bounds(stage,robot['root']))
                    require(full==screen,'Cached geometry differs from full geometry');checked_geometry=True
                write_json(folder/'geometry_screen.json',screen)
                require(screen['passed'],'Native robot geometry hold');pose['visual_bound_screen']=screen
                variant=next(v for v in changed['variants'] if v['variant_id']==row['variant_id'])
                world=target_world_geometry(stage,variant,row)
                require(all(np.allclose(world[k],world_from_row(pair,row)[k],atol=1e-9,rtol=0) for k in world),'Changed target world geometry')
                if not warmed:
                    take(rep,writer,scene['settings'],output/'warmup',[8]*7);warmed=True
                    # Establish the unchanged guard after explicit writer/product initialization.
                    # No render, update, or exclusion change is inserted after later camera moves.
                    verify_bindings(original_source_pins);verify_bindings(substituted_source_pins)
                    pins=source_hashes(stage)
                    write_json(output/'source_phase_bindings.json',dict(
                        original_source_bindings=original_source_pins,substituted_source_bindings=substituted_source_pins,
                        full_scene_source_bindings=pins,original_sources_verified_after_substitution=True,
                        original_and_substituted_source_contents_verified_after_warmup=True,
                        intentional_substitution_completed=True,
                        warmup_request_sha256=sha256(output/'warmup/request_evidence.json'),
                        full_scene_baseline_phase='after_intentional_substitution_and_explicit56_warmup',
                        source_hash_validation_removed=False))
                    monitor=StaticSceneMonitor(stage,robot['root'])
                    write_json(output/'static_monitor_initialization.json',dict(
                        phase='after_explicit56_warmup_before_first_short_request',
                        warmup_request_sha256=sha256(output/'warmup/request_evidence.json'),
                        first_sample_id=item['sample_id'],baseline_token=monitor.token(),
                        static_guard_exclusions_modified=False))
                require(monitor is not None,'Static scene monitor was not initialized')
                before=monitor.begin()
                payload,request=take(rep,writer,scene['settings'],folder/'raw',[item['budget']])
                expect_change=previous is not None and fingerprint(cal)!=previous['camera_sha256']
                rgb,depth,valid,reference,token,ids,mapping=freshness(payload,cal,before,monitor.token(),
                    writer.sequence,previous,expect_change)
                previous=token
                assert_same_camera(calibration_for_native_resolution(calibration(stage),LEGACY_RESOLUTION),cal)
                require(request['settings']['/rtx/rendermode']==scene['old_manifest']['renderer'],'Renderer changed')
                components,organs,owners=component_masks(ids,mapping,catalogue)
                oldroot=scene['original_variant']['plant_root'];oldids=[i for i,p in mapping.items() if p==oldroot or p.startswith(oldroot+'/')]
                require(not np.isin(ids,oldids).any(),'Original foreground survived generated substitution')
                nominal=project([world['nominal_world_m']],cal)[0];interval=project(world['interval_world_m'],cal)
                target=next(c for c in catalogue if c['variant_id']==row['variant_id'] and c['component_id']==row['component_id'])
                radius=row['cut_region_proposal']['nominal']['petiole_radius_m']
                visibility,mask=interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,target,radius)
                meta=dict(schema_version=SAMPLE,sample_id=item['sample_id'],observation_id=item['name'],scene_case=scene_name,
                    training_sample_approved=False,sensor=sensor_profile(LEGACY_RESOLUTION),calibration=cal,robot_snapshot=pose,
                    scene_counts=scene['counts'],lighting=scene['old_manifest']['lighting'],renderer=scene['old_manifest']['renderer'],
                    rendered_camera_params=jsonable(payload['camera_params']),render_settings=request['settings'],
                    geometry_screen=screen,input_policy=dict(clean_full_scene=True,isolation=False,diagnostic_overlays=False),
                    native_instance_backend='fast',native_target_pixels=int(mask.sum()),old_plant_native_pixels=0,
                    native_instance_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),native_mapping_sha256=fingerprint(mapping),
                    quality=view_quality(cal,nominal,interval,radius,visibility,rgb,mask),
                    source_binding=dict(pair_plan_path=record['pair_plan_path'],pair_plan_sha256=record['pair_plan_sha256'],
                        cache_sha256=sha256(read_json(plan['sources'][scene_name+'_plan']['path'])['cache_path'])),
                    synchronization=dict(method='frozen_scene_single_native_writer_payload',scene_unchanged_during_capture=True,
                        dynamic_recording_supported=False,reference_time=reference,freshness=token,static_guard=before,
                        static_guard_after=monitor.token(),native_render_frame=jsonable(payload['pilot_render_frame']),
                        engine_frame_id_verified=False,render_budget_subframes=item['budget'],profile=PROFILE,
                        request_evidence=request,photometric_identity_to_long_render_claimed=False),
                    supervision=dict(**world,target_id=row['target_id'],source_target_id=pair['source_row']['target_id'],
                        split_group=pair['split_group'],conservative_view_cap_group=pair['conservative_view_cap_group'],
                        cut_region_proposal=row['cut_region_proposal'],nominal_projected=nominal,projected_interval=interval,
                        depth_evidence=depth_evidence(nominal,depth,valid,radius),visibility_evidence=visibility,cut_safety_validated=False))
                verify_camera(meta)
                write_sample(folder/'sample',rgb,depth,valid,meta)
                write_visibility(folder/'sample',ids,mapping,catalogue,components,organs,mask)
                meta=read_json(folder/'sample/sample.json')
                annotation=prior.annotations(meta,generated['report'],rgb,depth,valid,components,catalogue,mask)
                write_json(folder/'annotation.json',annotation)
                probes=prior.interval_probes(interval,depth,valid,ids,mapping);write_json(folder/'interval_probes.json',probes)
                proof=workspace.check_sample(folder/'sample/sample.json',sha256(folder/'sample/sample.json'))
                write_json(folder/'workspace.json',proof)
                quality=bool(annotation['label']['eligible'] and annotation['query_trace'] is not None
                    and annotation['query_trace']['passed'] and proof['result']['workspace_passed'])
                receipt.update(passed_actual_automated_criteria=quality,
                    local_clarity_passed=annotation['baseline_label']['eligible'],
                    selected_full_trace_passed=bool(annotation['label']['eligible'] and annotation['query_trace'] is not None and annotation['query_trace']['passed']),
                    workspace_passed=proof['result']['workspace_passed'],capture_valid=True,
                    fruit_negative_observed=any('/Fruit_07' in (p.get('prim_path') or '') for p in probes),
                    candidate_for_individual_visual_review=quality,render_seconds=request['elapsed_seconds'])
            except Exception:
                receipt['error']=traceback.format_exc()
            write_json(folder/'result.json',receipt)
            write_json(folder/'bindings.json',{str(p.resolve()):sha256(p) for p in folder.rglob('*') if p.is_file()})
            results.append(receipt)
            print('FAST848_PROFILE_V3',scene_name,item['name'],receipt.get('capture_valid',False),receipt['passed_actual_automated_criteria'],flush=True)
    finally:
        if monitor is not None:monitor.close()
        geometry.close();workspace.finish();writer.detach();product.destroy()
    require(pins is not None and source_hashes(stage)==pins,'Source asset map/hash changed after initialized baseline')
    verify_bindings(original_source_pins);verify_bindings(substituted_source_pins);verify_bindings(cache['source_bindings'])
    require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()),'Source layer dirtied')
    return results


def capture(app,output,plan):
    started=time.perf_counter();loaded=check(plan,full=True)
    renderers={read_json(Path(p['source_capture'])/'manifest.json')['renderer'] for _,plans,_ in loaded.values() for p in plans.values()}
    require(len(renderers)==1 and renderers=={'RealTimePathTracing'},'Mixed source renderers')
    control=controls(app,output/'controls',next(iter(renderers)))
    scenes={s:full_scene(app,output/s,plan,s,loaded[s]) for s in ('transition','negative')}
    diagnostics={}
    from .native_render_probe import comparison
    for budget in BUDGETS:
        left=output/'transition'/f'b{budget}_A_before';right=output/'transition'/f'b{budget}_A_return'
        try:
            meta=read_json(left/'sample/sample.json')
            diagnostics[str(budget)]=comparison(load_raw(left/'raw/callback_00'),load_raw(right/'raw/callback_00'),
                meta['calibration'],roi_pixel=meta['supervision']['nominal_projected']['pixel_xy'])
        except Exception:
            diagnostics[str(budget)]=dict(error=traceback.format_exc())
    status={}
    for budget in BUDGETS:
        positive=[r for r in scenes['transition'] if r['budget']==budget and r['role']=='clear_reference']
        negative=next(r for r in scenes['negative'] if r['budget']==budget)
        settings=[r['request_evidence']['settings'] for r in control['observations'] if r['budget']==budget and 'request_evidence' in r]
        settings += [read_json(output/s/r['name']/'raw/request_evidence.json')['settings'] for s,rows in scenes.items() for r in rows if r['budget']==budget and (output/s/r['name']/'raw/request_evidence.json').exists()]
        status[str(budget)]=dict(controls_passed=control['budgets'][str(budget)],
            control_and_scene_settings_equal=bool(settings and all(v==settings[0] for v in settings)),
            all_transition_frames_capture_valid=all(r.get('capture_valid',False) for r in scenes['transition'] if r['budget']==budget),
            both_A_frames_automated_passed=all(r['passed_actual_automated_criteria'] for r in positive),
            negative_correctly_rejected=bool(negative.get('capture_valid') and negative.get('fruit_negative_observed')
                and not negative['passed_actual_automated_criteria']),individual_visual_review_required=True,
            production_profile_qualified=False)
    check(plan)
    write_json(output/'evidence_bindings.json',{str(p.resolve()):sha256(p) for p in output.rglob('*') if p.is_file()})
    return dict(state=STATE,scenes=scenes,controls=control,budget_status=status,
        cross_render_equality_diagnostics=diagnostics,equality_used_for_acceptance=False,
        elapsed_seconds=time.perf_counter()-started,full_scene_observations=12,
        unique_accepted_training_images=0,accepted_unique_training_images_per_second=0,**FLAGS)


def audit(output,plan):
    """Independent raw-callback, registration, full-scene and quality replay."""
    from .collection_plan import load_plan
    from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
    from .capture_visibility import ORGAN_IDS
    loaded=check(plan,full=True);request=read_json(output/'request.json');result=read_json(output/'result.json')
    require(request['plan_sha256']==result['plan_sha256']==sha256(request['plan_path'])
        and read_json(request['plan_path'])==plan and result['state']==STATE
        and set(result['scenes'])=={'transition','negative'} and all(result[k]==v for k,v in FLAGS.items())
        and result['equality_used_for_acceptance'] is False and result['full_scene_observations']==12,'Changed execution scope')
    pins=read_json(output/'evidence_bindings.json');verify_bindings(pins)
    require(not any(output.rglob('unrequested_callbacks')),'Unexpected writer callbacks outside requested captures')
    actual_files={str(p.resolve()) for p in output.rglob('*') if p.is_file() and p.name not in ('result.json','evidence_bindings.json','audit_fast_profile_v3.json')}
    # Per-observation result.json files are also mandatory bound evidence.
    actual_files|={str(p.resolve()) for p in output.rglob('result.json') if p!=output/'result.json'}
    require(set(pins)==actual_files,'Incomplete raw/processed evidence manifest')
    verified=[];workspace=WorkspaceChecker();all_settings={n:[] for n in BUDGETS}
    warmups=[]
    try:
        for group in ('controls','transition','negative'):
            safety=read_json(output/group/'clock_safety.json')
            require(safety['profile']==PROFILE and safety['delta_time_seconds']==CLOCK_DELTA
                and safety['physics_motion_allowed'] is False and safety['static_scene_guard_still_required'] is True
                and safety['authored_time_sampled_attributes']==[] and safety['unsupported_dynamic_schemas']==[]
                and all(row[1] is False for key in ('rigid_bodies','joints','articulations') for row in safety[key]),
                'Unproven advancing-clock static scene')
            warmups.append(verify_request(output/group/'warmup',[8]*7))
        controls_rows=result['controls']['observations'];previous=None
        require([r['name'] for r in controls_rows]==[f'b{n}_{phase}' for n in BUDGETS for phase in plan['control_order']], 'Changed 15-control sequence')
        for row in controls_rows:
            folder=output/'controls'/row['name']
            require(read_json(folder/'result.json')==row,'Control result changed')
            if 'error' in row:
                require(row['passed'] is False,'Failed control was accepted');continue
            evidence=verify_request(folder/'raw',[row['budget']]);all_settings[row['budget']].append(evidence['settings'])
            require(evidence==row['request_evidence'],'Changed control request evidence')
            payload=load_raw(folder/'raw/callback_00');phase=row['name'].split('_',1)[1]
            on=phase=='occluder_on';is_b=phase=='B';matrix=np.eye(4);matrix[3,0]=.15 if is_b else 0
            require(np.allclose(row['calibration']['camera_to_world_usd_row_vectors'],matrix,atol=1e-9,rtol=0)
                and row['calibration']['resolution']==[848,408] and row['calibration']['crop_resize'] is None,'Changed authored control camera')
            require(row['expected_depth_m']==(1. if on else (2.4 if is_b else 2.))
                and row['expected_prim']==('/World/KnownOccluder' if on else '/World/CalibrationCube'),'Changed control geometry expectations')
            rgb,depth,valid,_,token,ids,mapping=freshness(payload,row['calibration'],row['static_guard'],row['static_guard_after'],
                evidence['requests'][0]['callback_sequence_after'],previous,phase!='A_before')
            previous=token;require(token==row['freshness'],'Changed control freshness')
            actual=control_check(depth,valid,ids,mapping,row['expected_depth_m'],row['expected_prim'])
            require(actual['passed']==row['passed'] and actual['checks']==row['checks'],'Control replay differs')
        for scene_name,rows in result['scenes'].items():
            require([{k:r[k] for k in ('name','budget','sample_id','role')} for r in rows]==plan['schedules'][scene_name], 'Changed 12-frame schedule')
            cache,plans,generated=loaded[scene_name];by_id={r['sample_id']:r for r in cache['records']};previous=None
            phase=read_json(output/scene_name/'source_phase_bindings.json')
            verify_bindings(phase['original_source_bindings']);verify_bindings(phase['substituted_source_bindings']);verify_bindings(phase['full_scene_source_bindings'])
            require(phase['original_sources_verified_after_substitution'] is True
                and phase['intentional_substitution_completed'] is True
                and phase['full_scene_baseline_phase']=='after_intentional_substitution_and_explicit56_warmup'
                and phase['original_and_substituted_source_contents_verified_after_warmup'] is True
                and phase['warmup_request_sha256']==sha256(output/scene_name/'warmup/request_evidence.json')
                and phase['source_hash_validation_removed'] is False,'Changed source-phase validation')
            guard=read_json(output/scene_name/'static_monitor_initialization.json')
            require(guard['phase']=='after_explicit56_warmup_before_first_short_request'
                and guard['warmup_request_sha256']==sha256(output/scene_name/'warmup/request_evidence.json')
                and guard['first_sample_id']==plan['schedules'][scene_name][0]['sample_id']
                and guard['static_guard_exclusions_modified'] is False,'Changed monitor initialization boundary')
            for row in rows:
                folder=output/scene_name/row['name'];verify_bindings(read_json(folder/'bindings.json'))
                require(read_json(folder/'result.json')==row,'Changed frame result')
                if (folder/'raw/request_evidence.json').exists():
                    all_settings[row['budget']].append(read_json(folder/'raw/request_evidence.json')['settings'])
                if not row.get('capture_valid'):
                    require('error' in row and row['passed_actual_automated_criteria'] is False,'Unexplained failed observation');continue
                meta=read_json(folder/'sample/sample.json');record=by_id[row['sample_id']];pair=plans[record['pair_plan_path']]
                manifest=read_json(Path(pair['source_capture'])/'manifest.json')
                require(meta['schema_version']==SAMPLE and meta['sample_id']==row['sample_id']
                    and meta['supervision']['target_id']==pair['generated_row']['target_id'] and meta['scene_counts']==pair['expected_scene_counts']
                    and meta['lighting']==manifest['lighting'] and meta['renderer']==manifest['renderer']
                    and meta['input_policy']==dict(clean_full_scene=True,isolation=False,diagnostic_overlays=False)
                    and meta['supervision']['split_group']==pair['split_group']
                    and meta['supervision']['conservative_view_cap_group']==pair['conservative_view_cap_group'],'Changed scene/sample')
                assert_same_camera(meta['calibration'],record['calibration']);verify_camera(meta)
                require(meta['robot_snapshot']['joint_degrees']==record['joint_degrees']
                    and np.allclose(meta['robot_snapshot']['robot_root_to_world_usd_row_vectors'],record['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0)
                    and meta['geometry_screen']==read_json(folder/'geometry_screen.json')==meta['robot_snapshot']['visual_bound_screen']
                    and meta['geometry_screen']['passed'] is True,'Changed embodied pose/geometry proof')
                payload=load_raw(folder/'raw/callback_00');sync=meta['synchronization'];evidence=verify_request(folder/'raw',[row['budget']])
                require(evidence==sync['request_evidence'] and sync['profile']==PROFILE and sync['render_budget_subframes']==row['budget']
                    and meta['render_settings']==evidence['settings'] and evidence['settings']['/rtx/rendermode']==manifest['renderer'],'Changed effective budget/settings')
                changed=previous is not None and fingerprint(meta['calibration'])!=previous['camera_sha256']
                rgb,depth,valid,_,token,ids,mapping=freshness(payload,meta['calibration'],sync['static_guard'],sync['static_guard_after'],
                    evidence['requests'][0]['callback_sequence_after'],previous,changed)
                previous=token;require(token==sync['freshness'],'Changed freshness proof')
                require(meta['native_instance_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
                    and meta['native_mapping_sha256']==fingerprint(mapping),'Changed native fingerprints')
                oldroot=pair['original_variant']['plant_root']
                require(not np.isin(ids,[i for i,p in mapping.items() if p==oldroot or p.startswith(oldroot+'/')]).any(),'Original foreground present')
                require(np.array_equal(rgb,image_array(folder/'sample/inputs/rgb.png'))
                    and np.array_equal(depth,np.load(folder/'sample/inputs/depth_m.npy',allow_pickle=False),equal_nan=True)
                    and np.array_equal(valid,image_array(folder/'sample/inputs/depth_valid.png')==255),'Saved observations differ')
                identities=read_json(folder/'sample/supervision/identities.json');catalogue=identities['component_catalogue']
                _,reports=load_plan(pair['source_collection_plan']);reports={r['plant_id']:r for r in reports}
                require(catalogue==expected_catalogue(pair,'generated_variant',reports,generated)
                    and {int(k):v for k,v in identities['renderer_id_to_prim'].items()}==mapping
                    and identities['organ_ids']==ORGAN_IDS,'Changed full scene catalogue/mapping')
                components,organs,owners=component_masks(ids,mapping,catalogue);world=world_from_row(pair,pair['generated_row'])
                require(all(np.allclose(meta['supervision'][k],v,atol=1e-9,rtol=0) for k,v in world.items()),'Changed world geometry')
                interval=project(world['interval_world_m'],meta['calibration']);nominal=project([world['nominal_world_m']],meta['calibration'])[0]
                target=next(c for c in catalogue if c['variant_id']==pair['generated_row']['variant_id'] and c['component_id']==pair['generated_row']['component_id'])
                vis,mask=interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,target,pair['generated_row']['cut_region_proposal']['nominal']['petiole_radius_m'])
                require(vis==meta['supervision']['visibility_evidence'] and np.array_equal(ids,np.load(folder/'sample/supervision/renderer_instance_id.npy',allow_pickle=False))
                    and np.array_equal(components,np.load(folder/'sample/supervision/component_id.npy',allow_pickle=False))
                    and np.array_equal(organs,image_array(folder/'sample/supervision/organ_type.png'))
                    and np.array_equal(mask,image_array(folder/'sample/supervision/target_visible.png')==255),'Native visibility/mask replay differs')
                annotation=prior.annotations(meta,generated['report'],rgb,depth,valid,components,catalogue,mask)
                require(annotation==read_json(folder/'annotation.json'),'Label/query replay differs')
                probes=prior.interval_probes(interval,depth,valid,ids,mapping)
                require(probes==read_json(folder/'interval_probes.json'),'Probe replay differs')
                proof=read_json(folder/'workspace.json');verify_bindings(proof['source_bindings'])
                fresh=workspace.check_sample(folder/'sample/sample.json',sha256(folder/'sample/sample.json'))
                require(fresh['result']['workspace_passed']==proof['result']['workspace_passed'],'Independent workspace decision differs')
                trace=bool(annotation['label']['eligible'] and annotation['query_trace'] is not None and annotation['query_trace']['passed'])
                quality=bool(trace and fresh['result']['workspace_passed'])
                require(row['passed_actual_automated_criteria']==quality and row['local_clarity_passed']==annotation['baseline_label']['eligible']
                    and row['selected_full_trace_passed']==trace and row['workspace_passed']==fresh['result']['workspace_passed']
                    and row['candidate_for_individual_visual_review']==quality
                    and row['fruit_negative_observed']==any('/Fruit_07' in (p.get('prim_path') or '') for p in probes),'Changed actual frame decision')
                verified.append(str(folder))
        for n in BUDGETS:
            control_pass=all(r['passed'] for r in controls_rows if r['budget']==n)
            require(result['controls']['budgets'][str(n)]==control_pass,'Changed control summary')
            positives=[r for r in result['scenes']['transition'] if r['budget']==n and r['role']=='clear_reference']
            negative=next(r for r in result['scenes']['negative'] if r['budget']==n)
            settings=all_settings[n]
            expected=dict(controls_passed=control_pass,control_and_scene_settings_equal=bool(settings and all(v==settings[0] for v in settings)),
                all_transition_frames_capture_valid=all(r.get('capture_valid',False) for r in result['scenes']['transition'] if r['budget']==n),
                both_A_frames_automated_passed=all(r['passed_actual_automated_criteria'] for r in positives),
                negative_correctly_rejected=bool(negative.get('capture_valid') and negative.get('fruit_negative_observed') and not negative['passed_actual_automated_criteria']),
                individual_visual_review_required=True,production_profile_qualified=False)
            require(result['budget_status'][str(n)]==expected,'Changed per-budget conclusion')
        require(all(w['settings']['/rtx/rendermode']=='RealTimePathTracing' for w in warmups),'Wrong warmup renderer')
    finally:
        workspace.finish()
    check(plan)
    return dict(state='native848_fast_profile_saved_buffer_replay_complete',verified_observations=verified,**FLAGS)


def cpu_tests():
    checked=0
    for budget in BUDGETS:
        require(prior.checked_subframe_floor({},budget)['effective_rt_subframes_from_installed_policy']==budget,'Budget error');checked+=1
        for bad in ({'/omni/replicator/RTSubframes':budget+1},{'/omni/replicator/captureMotionBlur':True}):
            try: prior.checked_subframe_floor(bad,budget)
            except ValueError: checked+=1
            else: raise AssertionError('Invalid short budget accepted')
    depth=np.full((408,848),2.,dtype=np.float32);valid=np.ones(depth.shape,dtype=bool);ids=np.full(depth.shape,2,dtype=np.uint32)
    require(control_check(depth,valid,ids,{2:'/World/CalibrationCube'},2.,'/World/CalibrationCube')['passed'],'Good control rejected');checked+=1
    require(not control_check(depth,valid,ids,{2:'/World/CalibrationCube'},1.,'/World/KnownOccluder')['passed'],'Stale occluder accepted');checked+=1
    depth[204,470]=2.01
    require(not control_check(depth,valid,ids,{2:'/World/CalibrationCube'},2.,'/World/CalibrationCube')['passed'],'Off-axis registration error accepted');checked+=1
    require(len(schedule('transition'))==9 and len(schedule('negative'))==3,'Wrong transition budget');checked+=1
    import tempfile
    from .native_instances import normalize_payload
    with tempfile.TemporaryDirectory(prefix='fast848_profile_v3_cpu_',dir=D) as temporary:
        folder=Path(temporary).resolve()
        require(folder.is_relative_to(D.resolve()),'CPU scratch outside diagnostics')
        callback=folder/'serializer_only'
        raw=dict(rgb=np.full((408,848,4),17,dtype=np.uint8),
            distance_to_image_plane=np.full((408,848),2.,dtype=np.float32),
            instance_id_segmentation_fast=dict(data=np.full((408,848),2,dtype=np.uint32),
                info=dict(ids=np.asarray([2],dtype=np.uint32),labels=['/World/CalibrationCube'])),
            camera_params=dict(test_only=True),reference_time=(0,1),
            ReferenceTime=dict(referenceTimeNumerator=0,referenceTimeDenominator=1),
            serializer_test_values=np.asarray(['label',b'bytes'],dtype=object))
        save_callback_tree(callback,raw)
        restored=load_callback_tree(callback)
        require(isinstance(restored['reference_time'],tuple) and np.array_equal(restored['rgb'],raw['rgb'])
            and np.array_equal(restored['serializer_test_values'],raw['serializer_test_values']), 'Raw callback roundtrip failed');checked+=1
        # Keep callback tree lossless; use supported metadata values for the native replay fixture.
        raw.pop('serializer_test_values');restored.pop('serializer_test_values')
        replay_folder=folder/'all_callbacks/callback_000002';save_callback_tree(replay_folder,raw)
        callback=replay_folder
        normalized=normalize_payload(restored,'fast');save_raw(folder/'callback_00',normalized)
        event=dict(callback_sequence=2,request_index=1,normalization_passed=True,folder=str(callback))
        write_json(callback/'callback_receipt.json',event)
        evidence=dict(requested_steps=[BUDGETS[0]],settings=PROFILE_SETTINGS,hidden_settling=False,callbacks=[event],
            delta_time_seconds=CLOCK_DELTA,wait_for_render=True,
            native_reference_time_policy=NATIVE_REFERENCE_POLICY,scheduler_clock_policy=SCHEDULER_POLICY,
            effective_subframe_policy=[prior.checked_subframe_floor(PROFILE_SETTINGS,BUDGETS[0])],
            reset=dict(api='omni.usd.get_context().reset_renderer_accumulation',returned_without_exception=True,temporal_cleanliness_proven=False),
            requests=[dict(subframes=BUDGETS[0],native_requests=1,request_index_before=0,request_index_after=1,
                callback_sequence_before=1,callback_sequence_after=2,normalized_from_callback_sequence=2,
                previous_reference_time=None,reference_time=[0,1],step_timing=dict(delta_time_seconds=CLOCK_DELTA,
                    wait_for_render=True,app_update_event_count=1,timeline_before_seconds=0.,timeline_after_seconds=0.))])
        write_json(folder/'request_evidence.json',evidence);verify_request(folder,[BUDGETS[0]]);checked+=1
        np.save(folder/'callback_00/rgba.npy',np.zeros_like(raw['rgb']),allow_pickle=False)
        try:verify_request(folder,[BUDGETS[0]])
        except ValueError:checked+=1
        else:raise AssertionError('Mixed-callback RGB accepted')
        np.save(folder/'callback_00/rgba.npy',raw['rgb'],allow_pickle=False)
        evidence['requests'][0]['native_requests']=2;(folder/'request_evidence.json').write_text(__import__('json').dumps(evidence),encoding='utf-8')
        try:verify_request(folder,[BUDGETS[0]])
        except ValueError:checked+=1
        else:raise AssertionError('Retry inflation accepted')
        bad=deepcopy(raw);bad['instance_id_segmentation_fast']['info']={}
        save_callback_tree(folder/'malformed_preserved',bad)
        try:normalize_payload(load_callback_tree(folder/'malformed_preserved'),'fast')
        except ValueError:checked+=1
        else:raise AssertionError('Malformed FAST mapping accepted')
        require((folder/'malformed_preserved/tree.json').is_file(),'Malformed raw callback not preserved');checked+=1
    return dict(passed=True,checks=checked,native_execution=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','capture','audit'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path);parser.add_argument('--plan-sha256')
    args=parser.parse_args(argv);output=args.output.resolve()
    if args.action=='prepare':
        require(not output.exists(),'Create-only preparation output required')
        plan=prepare();tests=cpu_tests();check(plan,full=True)
        output.mkdir(parents=True);write_json(output/'plan.json',plan)
        write_json(output/'cpu_validation.json',tests)
        write_json(output/'handoff.json',dict(plan_path=str(output/'plan.json'),plan_sha256=sha256(output/'plan.json'),
            module_path=str(Path(__file__).resolve()),module_sha256=sha256(__file__),native_launched=False,
            native_owner_required=True,independent_code_review_required=True,**FLAGS))
        print(str(output/'handoff.json'));return
    require(args.plan is not None and sha256(args.plan)==args.plan_sha256,'Exact plan pin required')
    plan=read_json(args.plan);check(plan)
    if args.action=='audit':
        report=audit(output,plan)
        require(not (output/'audit_fast_profile_v3.json').exists(),'Create-only audit required')
        write_json(output/'audit_fast_profile_v3.json',report);return
    require(not output.exists() and output.parent.is_dir(),'Create-only capture output required')
    protected=[Path(v['path']).parent.resolve() for v in plan['sources'].values()]+[args.plan.parent.resolve()]
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),'Output overlaps sources')
    from .native_generated_pair import windows_worker_admission
    from sim_physics.host_memory import preflight
    import shutil
    memory=preflight();disk=shutil.disk_usage(output.parent).free
    require(memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30 and disk>=60*2**30,'Native resource reserve failed')
    process=windows_worker_admission(None);output.mkdir()
    write_json(output/'request.json',dict(plan_path=str(args.plan.resolve()),plan_sha256=args.plan_sha256,
        created_utc=datetime.now(timezone.utc).isoformat(),host_memory_preflight=memory,available_disk_bytes=disk,
        process_admission=process,**FLAGS))
    app=None;success=False
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RealTimePathTracing',anti_aliasing=2,
            sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false','--/persistent/rtx/modes/rt2/enabled=true']))
        result=capture(app,output,plan);result['plan_sha256']=args.plan_sha256
        require(sha256(args.plan)==args.plan_sha256,'Plan changed during capture')
        write_json(output/'result.json',result);success=True
    except BaseException:
        write_json(output/'failure.json',dict(error=traceback.format_exc(),**FLAGS));raise
    finally:
        if app is not None:app.close(exit_code=0 if success else 1)


if __name__=='__main__':
    main()

