"""CPU ledger/owner mutations, actual sealed inventory, unchanged numerical hooks."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib,json,tempfile,types,unittest,inspect
from . import native848_scene_sweep_raw_9mm_v1 as api


def reach(record,passed=True):
    return dict(schema='greenhouse.native848_bulk_workspace.v1',target_id=record['target_id'],
        nominal_world_m=record['pilot_9mm_geometry']['nominal']['world_m'],solve_input_sha256='a'*64,
        identical_problem_reused=False,per_frame_camera_FK_verified=True,per_frame_cached_solution_FK_verified=passed,
        result=dict(workspace_passed=passed),probe_ee_m=[0,0,0],scope='kinematic_position_workspace_and_stationary_other_arm_screen',
        full_scene_arm_collision_checked=False,approach_path_checked=False,physical_cut_approved=False,training_approved=False)


def ledger():
    profile=dict(warmup_steps=[8],request_subframes=8,delta_time_seconds=0,render_settings={'optics':'same'})
    events=[]
    for i in range(1,4):
        events.append(dict(request_index=i,callback_sequence_before=i-1,callback_sequence_after=i,native_requests=1,callback_count=1,
            requested_subframes=8,delta_time_seconds=0,wait_for_render=True,timeline_before_seconds=0,timeline_after_seconds=0,
            reference_time=[i,1],previous_reference_time=None if i==1 else [i-1,1],render_settings=profile['render_settings'],reset_returned_without_exception=True))
    obs=[];frames=[]
    for i,name in enumerate(('a','b'),2):
        token=dict(callback_sequence=i,reference_time=[i,1],camera_sha256=name,depth_sha256='depth'+name,rgb_sha256='rgb'+name)
        obs.append(dict(sample_id=name,observation_id=name,request_evidence=events[i-1],synchronization=dict(request_index=i,callback_sequence=i,reference_time=[i,1],freshness=token)))
        frames.append(dict(observation_id=name,request_index=i))
    result=dict(frames=frames,holds=[dict(sample_id='held',reason='whole_robot_scene_collision',screen={'passed':False})],
        requests=events,warmup_requests=events[:1],request_count=3,callback_count=3)
    records=[dict(sample_id=name,target_id='slot/'+name,pilot_9mm_geometry=dict(nominal=dict(world_m=[0,0,1]))) for name in ('a','held','b')]
    indexed={r['sample_id']:r for r in records}
    for o in obs:o.update(actual_workspace_9mm=reach(indexed[o['sample_id']]),robot_snapshot=dict(source_body_orientation_preserved=False,pose_sampling='new_body_orbit_and_head_fk_v1'))
    result['workspace_stats']=dict(checked_frames=2,source_bindings={})
    return result,records,obs,profile


class LedgerTests(unittest.TestCase):
    def reject(self,fn):
        x=list(ledger());fn(x)
        with self.assertRaises(ValueError):api.validate_ledger(*x)
    def test_positive_single_writer_and_collision_hold(self):api.validate_ledger(*ledger())
    def test_missing_camera(self):self.reject(lambda x:x[1].append(dict(sample_id='missing')))
    def test_held_camera_also_captured(self):self.reject(lambda x:x[0]['holds'][0].update(sample_id='a'))
    def test_fake_collision_hold(self):self.reject(lambda x:x[0]['holds'][0]['screen'].update(passed=True))
    def test_duplicate_id(self):self.reject(lambda x:x[0]['frames'][1].update(observation_id='a'))
    def test_reset_required(self):self.reject(lambda x:x[0]['requests'][2].update(reset_returned_without_exception=False))
    def test_wrong_production_settings(self):self.reject(lambda x:x[0]['requests'][2].update(render_settings={}))
    def test_clock_reset(self):self.reject(lambda x:x[0]['requests'][2].update(request_index=1))
    def test_stale_rgb(self):self.reject(lambda x:x[2][1]['synchronization']['freshness'].update(rgb_sha256='rgba'))
    def test_foreign_event(self):self.reject(lambda x:x[2][1].update(request_evidence=x[0]['requests'][0]))


class OwnerTests(unittest.TestCase):
    def run_case(self,mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            trial=Path(tmp)/'trial';capture=trial/'capture';capture.mkdir(parents=True)
            def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return pin(p)
            def read(spec,within=None):
                p=Path(spec['path']).resolve()
                if within is not None:self.assertTrue(p.is_relative_to(within))
                self.assertEqual(pin(p),{k:spec[k] for k in ('path','sha256')});return json.loads(p.read_text())
            result,records,observations,profile=ledger();profilepin=save(trial/'profile.json',profile);anchor=save(trial/'anchor.json',{})
            req=dict(profile=profilepin,scene_anchor=anchor);rp=save(trial/'request.json',req)
            command=['python.bat','-B','-u','-m',api.producer.MODULE,'--capture','--request',rp['path'],'--request-sha256',rp['sha256'],'--output',str(capture)]
            owner={'ProcessId':5};identity=dict(expected_command=command,owner=owner,command={'ProcessId':7},native={'ProcessId':9})
            if mutation=='owner':identity['owner']={'ProcessId':6}
            ip=save(capture/'native_identity.json',identity)
            context=dict(schema=api.producer.CONTEXT_SCHEMA,request=rp,native_identity=ip,profile=profilepin,scene_anchor=anchor,proposal_stage_scene_collision_pending=True,native_collision_and_workspace_validated_before_each_saved_frame=True,source_bindings={anchor['path']:anchor['sha256']})
            result['workspace_stats']['source_bindings']={anchor['path']:anchor['sha256']};cp=save(capture/'context.json',context);receipts=[]
            for obs in observations:
                obs.update(schema=api.producer.OBSERVATION_SCHEMA,context=cp,training_approved=False,accepted_training_increment=0)
                if mutation=='context':obs['context']={'path':'other','sha256':'wrong'}
                op=save(capture/'frames'/obs['sample_id']/'observation.json',obs)
                receipts.append(dict(op,observation_id=obs['sample_id'],request_index=obs['synchronization']['request_index']))
            result.update(schema=api.producer.RESULT_SCHEMA,request=rp,context=cp,native_identity=ip,frames=receipts,
                training_approved=False,accepted_training_increment=0,stage_count=1,render_product_count=1,scene_assets_unchanged=True)
            if mutation=='two_stages':result['stage_count']=2
            resultpin=save(capture/'result.json',result)
            wp=save(trial/'owned_worker.json',dict(command=command,launcher_pid=7,owner_pid=5))
            ep=save(trial/'owned_exit.json',dict(returncode=1 if mutation=='nonzero' else 0,owned_worker=wp))
            sp=save(trial/'owner_started.json',dict(request=rp,command=command,resources=dict(raw_owner_classification={'metadata':owner})))
            save(trial/'owner_complete.json',dict(result=resultpin,request=rp,native_identity=ip,owned_exit=ep,owner_started=sp,
                frames=2,training_approved=False,accepted_training_increment=0))
            gate=types.SimpleNamespace(same_identity=lambda a,b:a==b,command_child=lambda *a:None,native_child=lambda *a:None)
            with patch.object(api.producer,'check_request',return_value=dict(records=records,profile=profile)),patch.object(api.producer.production,'identity_gate',gate),patch.object(api.producer,'validate_result',return_value=context) as shared:
                value=api.authenticate_capture(capture,read,pin)
                shared.assert_called_once_with(result,Path(rp['path']),capture)
                return value
    def test_actual_pinned_single_owner_success(self):self.assertEqual(len(self.run_case()[-1]),2)
    def test_wrong_owner(self):
        with self.assertRaises(ValueError):self.run_case('owner')
    def test_nonzero_exit(self):
        with self.assertRaises(ValueError):self.run_case('nonzero')
    def test_foreign_context(self):
        with self.assertRaises(ValueError):self.run_case('context')
    def test_unclaimed_second_stage(self):
        with self.assertRaises(ValueError):self.run_case('two_stages')


class GeometryTests(unittest.TestCase):
    def test_tiny_cross_stack_roundoff_passes(self):
        api.validate_geometry(dict(nominal=dict(world_m=[1.+5e-10],projected={'pixel_xy':[2.+5e-7],'projection_status':'in_frame'})),dict(nominal=dict(world_m=[1.],projected={'pixel_xy':[2.],'projection_status':'in_frame'})))
    def test_metric_and_pixel_over_bound_rejected(self):
        for actual,expected in [({'world_m':[1.+2e-9]},{'world_m':[1.]}),({'pixel_xy':[2.+2e-6]},{'pixel_xy':[2.]})]:
            with self.assertRaises(ValueError):api.validate_geometry(actual,expected)
    def test_semantic_or_sample_population_never_tolerated(self):
        for actual,expected in [({'projection_status':'behind'},{'projection_status':'in_frame'}),({'junction_to_cut':[1.]},{'junction_to_cut':[1.,2.]}),({'source_chain_reversed':1},{'source_chain_reversed':True})]:
            with self.assertRaises(ValueError):api.validate_geometry(actual,expected)


class InventoryTests(unittest.TestCase):
    def test_exact_frozen_numerical_functions(self):
        self.assertIs(api.evaluate_frame,api.coverage.evaluate_frame)
        self.assertIs(api.assess_frame,api.ambiguity.assess_frame)
        self.assertIs(api.geometry_9mm,api.labels.geometry_9mm)
        for path,pin in api.FROZEN.items():self.assertEqual(api.sha256(path),pin)
        self.assertNotIn('target',inspect.signature(api.prepare_inventory).parameters)
    def test_actual_sealed_full144_inventory_all11068_petioles(self):
        root=Path(__file__).resolve().parents[3]
        config=root/'data/sim_data/diagnostics/native848_fixed_varied_greenhouse_20260918_v1/environment.json'
        self.assertEqual(api.sha256(config),'5e53caf8cf4c30c80ecdd3fa28795ce51cab6b9d0e2e79be1a82c90c44d85057')
        env=api.read_json(config)
        def read(spec):self.assertEqual(api.sha256(spec['path']),spec['sha256']);return api.read_json(spec['path'])
        census=read(env['expected_census']);reports=read(env['source_anatomy_reports']);catalogue=read(env['expected_catalogue'])
        variants=[dict(plant_root=r['plant_root'],variant_id=r['variant_id'],source_plant_id=r['source_family'],split_group=r['source_family'],
            added_components=[],added_component_paths=[],source_geometry_modified=False) for r in census['all_plant_roots']]
        context=dict(dataset_split='train',full_scene_census=census,scene_variants=variants,source_collection_plan=census['source_collection_plan'])
        inventory,_=api.prepare_inventory(context,census,reports,catalogue,read(census['source_collection_plan']))
        self.assertEqual(len(inventory),11068);self.assertEqual(len({r['target_id'].split('/')[0] for r in inventory}),144)
        self.assertEqual(len({r['source_family'] for r in inventory}),16)
        changed=deepcopy(variants);changed[0]['source_geometry_modified']=True;context['scene_variants']=changed
        with self.assertRaises(ValueError):api.prepare_inventory(context,census,reports,catalogue,read(census['source_collection_plan']))


class RawWorkspaceTests(unittest.TestCase):
    def test_actual_workspace_hold_supported(self):
        x=list(ledger());x[0]['holds']=[dict(sample_id='held',reason='9mm_workspace',workspace=reach(x[1][1],False))]
        api.validate_ledger(*x)
    def test_hold_wrong_target_or_pass_claim_rejected(self):
        for mutate in (lambda w:w.update(target_id='foreign/target'),lambda w:w['result'].update(workspace_passed=True),lambda w:w.update(nominal_world_m=[0,0,2])):
            x=list(ledger());w=reach(x[1][1],False);mutate(w);x[0]['holds']=[dict(sample_id='held',reason='9mm_workspace',workspace=w)]
            with self.assertRaises(ValueError):api.validate_ledger(*x)
    def test_saved_frame_workspace_receipt_cannot_be_borrowed(self):
        x=list(ledger());x[2][0]['actual_workspace_9mm']=deepcopy(x[2][1]['actual_workspace_9mm'])
        with self.assertRaises(ValueError):api.validate_ledger(*x)
    def test_saved_frame_rejects_false_fk_or_physical_approval(self):
        for key,value in [('per_frame_camera_FK_verified',False),('full_scene_arm_collision_checked',True),('training_approved',True)]:
            x=list(ledger());x[2][0]['actual_workspace_9mm'][key]=value
            with self.assertRaises(ValueError):api.validate_ledger(*x)
    def test_body_orbit_must_not_claim_original_orientation(self):
        x=list(ledger());x[2][0]['robot_snapshot']['source_body_orientation_preserved']=True
        with self.assertRaises(ValueError):api.validate_ledger(*x)
    def test_original_annotation_body_is_exactly_unchanged(self):
        from . import native848_scene_sweep_9mm_v1 as old
        import ast
        def numerical_body(fn):
            tree=ast.parse(inspect.getsource(fn))
            class StripSavedReload(ast.NodeTransformer):
                def visit_Assign(self,node):
                    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='read_json' and any(isinstance(t,ast.Name) and t.id in ('metadata','annotation') for t in node.targets):return None
                    return node
            return ast.dump(StripSavedReload().visit(tree),include_attributes=False)
        self.assertEqual(numerical_body(api.evaluate_capture),numerical_body(old.evaluate_capture))
        source=inspect.getsource(api.evaluate_capture)
        self.assertLess(source.index("metadata=read_json(folder/'sample.json')"),source.index('assessment=assess_frame('))
        self.assertLess(source.index("annotation=read_json(folder/'annotation.json')"),source.index('assessment=assess_frame('))
        self.assertEqual(inspect.getsource(api.prepare_inventory),inspect.getsource(old.prepare_inventory))
        self.assertEqual(api.FROZEN,old.FROZEN)

if __name__=='__main__':unittest.main(verbosity=2)
