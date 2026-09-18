"""CPU fixtures: identity/full census boundaries and frozen numeric equivalence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace,ModuleType
from unittest import TestCase,main
from unittest.mock import patch
import ast,inspect,json,sys
import numpy as np
from . import native848_generated_9mm_v1 as generated


def numeric_fixture():
    report=dict(plant_id='seed7_full',components={
        'MainStem_1':dict(type='main_stem',parent=None),
        'SubStem_44':dict(type='sub_stem',parent='MainStem_1',deleafed=False,
            translation_plant_m=[0.,0.,-.25],attachment_plant_m=[0.,0.,-.25],axis_plant=[1.,0.,0.],
            capsules_local_m=[[[0.,0.,0.,.003],[.1,0.,0.,.003]]])})
    entry=dict(target_id='instance/SubStem_44',source_family='seed7_full',split_group='seed7_full',
        source_target_id='seed7_full/SubStem_44',source_component_id='SubStem_44',geometry_source_id='seed7_full',
        geometry_target_id='seed7_full/SubStem_44',generated_geometry=False,morphology_id=None,
        physical_geometry_verified=True,physical_geometry_evidence=None,report=report,
        plant_to_world_usd_row_vectors=np.eye(4).tolist(),semantic_leaf_petiole_candidate=True,anatomy_reason_codes=[])
    catalogue=[dict(component_id=cid,prim_path='/World/P/'+cid,organ_type=typ,variant_id='instance',
        source_plant_id='seed7_full',split_group='seed7_full',geometry_source_id='seed7_full',component_index=i)
        for i,(cid,typ) in enumerate([('MainStem_1','main_stem'),('SubStem_44','sub_stem')],1)]
    cal=dict(camera_to_world_usd_row_vectors=np.eye(4).tolist(),intrinsics=[[500.,0.,424.],[0.,500.,204.],[0.,0.,1.]],
        resolution=[848,408],clipping_range_m=[.01,20.],crop_resize=None,depth_convention='optical_axis_z_metres_not_ray_range')
    metadata=dict(sample_id='CPU_fixture',calibration=cal,robot_snapshot={'fixture':True},
        synchronization=dict(scene_unchanged_during_capture=True,dynamic_recording_supported=False))
    rgb=np.full((408,848,3),160,np.uint8);depth=np.full((408,848),.25,np.float32);valid=np.ones(depth.shape,bool)
    components=np.zeros(depth.shape,np.uint32);components[198:211,424:626]=2;components[185:224,423:425]=1
    return metadata,rgb,depth,valid,components,catalogue,entry


class Workspace:
    def __init__(self,passed=True,status='reachable'):
        self.passed=passed;self.status=status;self.bindings={}
    def check(self,metadata):
        return dict(metadata['supervision'],result=dict(workspace_passed=self.passed,status=self.status),
            per_frame_camera_FK_verified=True,per_frame_cached_solution_FK_verified=True,solve_input_sha256='a'*64)


def bounds():
    return dict(schema=generated.original.BOUNDS_SCHEMA,margin_m=.001,
        arms={side:dict(shoulder_world_m=[0.,0.,0.],conservative_probe_reach_m=10.) for side in ('left','right')},source_bindings={})


def derive(entry,catalogue):
    entry=deepcopy(entry);catalogue=deepcopy(catalogue)
    entry.update(geometry_source_id='derived7',geometry_target_id='derived7/SubStem_44',generated_geometry=True,morphology_id='derived7')
    entry['report']['plant_id']='derived7'
    for row in catalogue:row['geometry_source_id']='derived7'
    return entry,catalogue


class NumericParity(TestCase):
    def test_planned_float_tolerance_does_not_accept_a_changed_cut(self):
        *_,catalogue,entry=numeric_fixture()
        actual=generated.geometry_9mm(entry['report'],'SubStem_44',np.eye(4),numeric_fixture()[0]['calibration'])
        planned=deepcopy(actual);planned['nominal']['world_m'][0]+=.5e-9
        planned['nominal']['projected']['pixel_xy'][0]+=.5e-6
        generated.validate_planned_geometry(actual,planned)
        planned['nominal']['world_m'][0]+=.1e-3
        with self.assertRaises(ValueError):generated.validate_planned_geometry(actual,planned)

    def test_static_numerical_body_matches_frozen_epoch2(self):
        source=inspect.getsource(generated.labels.evaluate_target)
        source=source.replace('def evaluate_target(', 'def _evaluate_geometry_target(',1)
        source=source.replace("require(report['plant_id'] == entry['source_family'], 'Report/source family differs')",
            "require(report['plant_id'] == entry['geometry_source_id'], 'Report/geometry identity differs')",1)
        source=source.replace('joint_ownership.verified_additional_parent(entry, catalogue)','generated_joint_owner(entry, catalogue)',1)
        self.assertEqual(ast.dump(ast.parse(source)),ast.dump(ast.parse(inspect.getsource(generated._evaluate_geometry_target))))

    def test_actual_native_array_fixtures_preserve_every_numeric_output(self):
        for mode in ('pass','narrow','dark','occlusion','invalid_depth','attachment','parent_clearance','workspace'):
            with self.subTest(mode=mode):
                m,rgb,depth,valid,components,catalogue,entry=numeric_fixture();ws=Workspace()
                if mode=='narrow':components[198:202,425:]=0;components[207:211,425:]=0
                if mode=='dark':rgb[:]=20
                if mode=='occlusion':depth[204,442]=.23
                if mode=='invalid_depth':depth[204,442]=np.inf;valid[204,442]=False
                if mode=='attachment':components[204,426:430]=0
                if mode=='parent_clearance':components[201,440:447]=1
                if mode=='workspace':ws=Workspace(False,'outside_outer_reach_bound')
                baseline=generated.labels.evaluate_target(m,rgb,depth,valid,components,catalogue,entry,ws)
                new_entry,new_catalogue=derive(entry,catalogue)
                actual=generated.evaluate_generated_target(m,rgb,depth,valid,components,new_catalogue,new_entry,ws)
                self.assertEqual(actual,baseline)
                if mode=='pass':self.assertEqual(actual['status'],'candidate_pending_visual_review')
                else:self.assertNotEqual(actual['status'],'candidate_pending_visual_review')

    def test_unqualified_geometry_cannot_be_evaluated_as_original(self):
        m,rgb,d,v,mask,catalogue,entry=numeric_fixture();entry,catalogue=derive(entry,catalogue)
        entry['physical_geometry_verified']=False
        with self.assertRaisesRegex(ValueError,'physical geometry'):
            generated.evaluate_generated_target(m,rgb,d,v,mask,catalogue,entry,Workspace())
        mask[:]=0
        result=generated.evaluate_generated_target(m,rgb,d,v,mask,catalogue,entry,Workspace())
        self.assertEqual(result['reason'],'not_visible_zero_authenticated_component_pixels')

    def test_identity_rejects_donor_alias_and_cross_instance_owner(self):
        *_,catalogue,entry=numeric_fixture();entry,catalogue=derive(entry,catalogue)
        for alteration in ('report','donor','catalogue','target'):
            e,c=deepcopy(entry),deepcopy(catalogue)
            if alteration=='report':e['report']['plant_id']=e['source_family']
            if alteration=='donor':e['source_family']='derived7'
            if alteration=='catalogue':c[1]['geometry_source_id']='different'
            if alteration=='target':e['geometry_target_id']='derived7/SubStem_43'
            with self.subTest(alteration=alteration),self.assertRaises(ValueError):generated._identity(e,c)

    def test_fullframe_unqualified_visible_target_stays_unknown(self):
        m,rgb,d,v,mask,catalogue,entry=numeric_fixture();entry,catalogue=derive(entry,catalogue)
        entry['physical_geometry_verified']=False
        with patch.object(generated.original,'robot_bounds',return_value=bounds()):
            result=generated.evaluate_frame(m,rgb,d,v,mask,catalogue,[entry],Workspace())
        self.assertEqual(result['schema'],generated.SCHEMA)
        self.assertEqual(result['targets'][0]['status'],'unknown')
        self.assertTrue(result['frame_blocked_by_unknown_targets'])
        self.assertEqual(result['targets'][0]['source_family'],'seed7_full')
        self.assertEqual(result['targets'][0]['geometry_source_id'],'derived7')


class AlternativeGuard(TestCase):
    def base(self):
        m,rgb,d,v,mask,catalogue,entry=numeric_fixture();entry,catalogue=derive(entry,catalogue)
        with patch.object(generated.original,'robot_bounds',return_value=bounds()):
            annotation=generated.evaluate_frame(m,rgb,d,v,mask,catalogue,[entry],Workspace())
        annotation['target_census']['complete']=True
        annotation['frame_blocked_by_unverified_full_scene_coverage']=False
        return m,annotation

    def test_missing_coverage_and_multiple_strict_candidates_hold(self):
        m,a=self.base();a['target_census']['complete']=False
        self.assertFalse(generated.assess_frame(m,a,Workspace())['single_answer_unambiguous'])
        m,a=self.base();other=deepcopy(a['targets'][0]);other['target_id']='other/SubStem_44';a['targets'].append(other)
        a['target_census']['catalogue_petiole_ids'].append(other['target_id'])
        a['target_census']['candidate_target_ids'].append(other['target_id'])
        self.assertFalse(generated.assess_frame(m,a,Workspace())['single_answer_unambiguous'])

    def test_detail_failed_visible_reachable_alternative_blocks(self):
        m,a=self.base();other=deepcopy(a['targets'][0]);other.update(target_id='other/SubStem_44',
            status='excluded',reason='strict_native_local_clarity_failed',automated_pass=False)
        a['targets'].append(other);a['target_census']['catalogue_petiole_ids'].append(other['target_id'])
        a['target_census']['excluded_target_ids'].append(other['target_id'])
        with patch.object(generated.original,'robot_bounds',return_value=bounds()):
            result=generated.assess_frame(m,a,Workspace())
            unresolved=generated.assess_frame(m,a,Workspace(False,'search_exhausted'))
            outside=generated.assess_frame(m,a,Workspace(False,'outside_outer_reach_bound'))
        self.assertEqual(result['reachable_alternative_ids'],['other/SubStem_44'])
        self.assertEqual(unresolved['unknown_alternative_ids'],['other/SubStem_44'])
        self.assertFalse(result['single_answer_unambiguous']);self.assertFalse(unresolved['single_answer_unambiguous'])
        self.assertTrue(outside['single_answer_unambiguous'])

    def test_tampered_status_census_rejected(self):
        m,a=self.base();a['target_census']['excluded_target_ids']=['bogus']
        with self.assertRaises(ValueError):generated.assess_frame(m,a,Workspace())


class FullPopulationIdentity(TestCase):
    def fixture(self,directory):
        base=Path(directory);reports={};bindings={}
        for gid in ('seed7_full','derived7'):
            folder=base/gid;folder.mkdir();manifest=folder/'manifest.json';manifest.write_text('{}')
            components={}
            for cid,typ,parent in [('MainStem_1','main_stem',None),('SubStem_44','sub_stem','MainStem_1'),('Leaf000','leaf','SubStem_44')]:
                asset=folder/(cid+'.usd');asset.write_text('CPU fixture '+gid+cid)
                bindings[str(asset)]=generated.sha256(asset)
                components[cid]=dict(type=typ,parent=parent,file=asset.name,asset_sha256=bindings[str(asset)],deleafed=False)
            bindings[str(manifest)]=generated.sha256(manifest)
            reports[gid]=dict(plant_id=gid,manifest_path=str(manifest),manifest_sha256=bindings[str(manifest)],components=components,
                targets=[dict(component_id='SubStem_44',expected_detached_component_ids=['SubStem_44','Leaf000'],protected_descendant_ids=[],reason_codes=[])],status='fixture')
        plan=base/'plan.json';plan.write_text(json.dumps(dict(family_assignments={'seed7_full':'train'})));bindings[str(plan)]=generated.sha256(plan)
        q=base/'qualification.json';q.write_text('{}');e=base/'evidence.json';e.write_text('{}')
        pin=lambda p:dict(path=str(p),sha256=generated.sha256(p))
        roots=[];variants=[];catalogue=[]
        for n in range(144):
            instance='instance_'+str(n);root='/World/P'+str(n).zfill(3);gid='derived7' if n==0 else 'seed7_full'
            variants.append(dict(variant_id=instance,plant_root=root,source_plant_id='seed7_full',split_group='seed7_full',
                geometry_source_id=gid,source_geometry_modified=n==0,added_components={},added_component_paths={}))
            roots.append(dict(plant_root=root,active_after_policy=True,authenticated_component_plant=True,source_family='seed7_full',source_split='train',
                manifest_path=reports[gid]['manifest_path'],manifest_sha256=reports[gid]['manifest_sha256'],component_count=3,
                plant_to_world_usd_row_vectors=np.eye(4).tolist()))
            for cid,subpath,typ in [('MainStem_1','MainStem_1','main_stem'),('SubStem_44','MainStem_1/SubStem_44','sub_stem'),('Leaf000','MainStem_1/SubStem_44/Leaf000','leaf')]:
                catalogue.append(dict(component_id=cid,prim_path=root+'/'+subpath,organ_type=typ,variant_id=instance,
                    source_plant_id='seed7_full',split_group='seed7_full',geometry_source_id=gid))
        catalogue=[dict(row,component_index=i) for i,row in enumerate(sorted(catalogue,key=lambda r:r['prim_path']),1)]
        census=dict(schema=generated.CENSUS_SCHEMA,native_population_census_verified=True,source_qualification=pin(q),foreground_root='/World/P000',unchanged_background_count=143,removed_roots=[],active_counts=dict(component_plants=144,backdrop_instances=0),
            all_plant_roots=roots,source_collection_plan=pin(plan))
        census['deterministic_census_sha256']=generated.fingerprint(census)
        context=dict(schema=generated.CONTEXT_SCHEMA,dataset_split='train',native_population_census_verified=True,
            original_143_backgrounds_preserved=True,source_bindings=bindings,source_collection_plan=pin(plan),full_scene_census=census,scene_variants=variants,
            geometry_sources={'derived7':dict(donor_source_family='seed7_full',source_split='train',manifest=pin(Path(reports['derived7']['manifest_path'])),
                qualification=pin(q),physical_geometry_evidence=pin(e))})
        return context,reports,catalogue

    def test_complete144_and_identity_omissions(self):
        authority=ModuleType('sim_data.native848_controlled_9mm_evidence_v2')
        authority.authenticate_9mm_evidence=lambda spec,**kw:dict(radius_qualified_component_ids=['SubStem_44'],modified_component_ids=['SubStem_44'],source_bindings={str(Path(spec['path']).resolve()):spec['sha256']})
        with TemporaryDirectory() as d:
            context,reports,catalogue=self.fixture(d)
            def audit(path):return next(r for r in reports.values() if Path(r['manifest_path'])==Path(path))
            with patch.dict(sys.modules,{authority.__name__:authority}),patch('sim_data.audit.audit_manifest',side_effect=audit):
                entries=generated.build_inventory(context,reports,catalogue)
                self.assertEqual(len(entries),144);self.assertEqual(sum(e['generated_geometry'] for e in entries),1)
                self.assertEqual({e['source_family'] for e in entries},{'seed7_full'})
                self.assertEqual({e['geometry_source_id'] for e in entries},{'seed7_full','derived7'})
                for kind in ('omitted','cross_split','renamed_donor','second_generated','changed_manifest'):
                    c,k=deepcopy(context),deepcopy(catalogue)
                    if kind=='omitted':k.pop()
                    if kind=='cross_split':c['full_scene_census']['all_plant_roots'][1]['source_split']='test'
                    if kind=='renamed_donor':c['scene_variants'][0]['source_plant_id']='derived7'
                    if kind=='second_generated':c['scene_variants'][1]['source_geometry_modified']=True
                    if kind=='changed_manifest':c['full_scene_census']['all_plant_roots'][0]['manifest_sha256']='f'*64
                    census=c['full_scene_census'];census.pop('deterministic_census_sha256');census['deterministic_census_sha256']=generated.fingerprint(census)
                    with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):
                        generated.build_inventory(c,reports,k)



class GeneratedSeamProof(TestCase):
    def test_actual_generated_mesh_joint_requires_same_geometry_and_exact_seam(self):
        from pxr import Usd,UsdGeom
        with TemporaryDirectory() as directory:
            folder=Path(directory);radius=.003
            angle=np.arange(10)*2*np.pi/10
            ring=np.c_[np.zeros(10),radius*np.cos(angle),radius*np.sin(angle)]
            details={
                'MainStem_1':dict(type='main_stem',parent=None,translation_plant_m=[-.02,0.,0.],
                    attachment_plant_m=[-.02,0.,0.],radius_m=.008,
                    capsules_local_m=[[[0.,0.,0.,.008],[0.,.1,0.,.008]],[[.01,0.,0.,radius],[.02,0.,0.,radius]]]),
                'MainStem_2':dict(type='main_stem',parent='MainStem_1',translation_plant_m=[-.01,0.,0.],
                    attachment_plant_m=[-.01,0.,0.],radius_m=.008,capsules_local_m=[[[0.,0.,0.,.008],[0.,.1,0.,.008]]]),
                'SubStem_44':dict(type='sub_stem',parent='MainStem_2',translation_plant_m=[0.,0.,0.],
                    attachment_plant_m=[0.,0.,0.],radius_m=radius,deleafed=False,axis_plant=[1.,0.,0.],
                    capsules_local_m=[[[0.,0.,0.,radius],[.1,0.,0.,radius]]])}
            raw=[]
            for cid,c in details.items():
                asset=folder/(cid+'.usda');stage=Usd.Stage.CreateNew(str(asset))
                mesh=UsdGeom.Mesh.Define(stage,'/Mesh')
                mesh.CreatePointsAttr((ring-np.asarray(c['translation_plant_m'])).tolist())
                mesh.CreateFaceVertexCountsAttr([10]);mesh.CreateFaceVertexIndicesAttr(list(range(10)))
                stage.GetRootLayer().Save();stage=None
                c.update(file=asset.name,asset_sha256=generated.sha256(asset))
                raw.append(dict(id=cid,type=c['type'],parent=c['parent'],file=c['file'],attach_point=c['attachment_plant_m'],
                    capsules=c['capsules_local_m'],radius=c['radius_m'],transform=dict(translate=c['translation_plant_m'])))
            manifest=folder/'manifest.json';manifest.write_text(json.dumps(dict(components=raw)))
            report=dict(plant_id='derived7',components=details,manifest_path=str(manifest),manifest_sha256=generated.sha256(manifest))
            entry=dict(target_id='instance/SubStem_44',report=report,source_family='seed7_full',split_group='seed7_full',
                source_target_id='seed7_full/SubStem_44',geometry_source_id='derived7',geometry_target_id='derived7/SubStem_44')
            catalogue=[dict(variant_id='instance',component_id=cid,organ_type=c['type'],component_index=i,
                source_plant_id='seed7_full',split_group='seed7_full',geometry_source_id='derived7')
                for i,(cid,c) in enumerate(details.items(),1)]
            owners,proof=generated.generated_joint_owner(entry,catalogue)
            self.assertEqual(owners,{1});self.assertEqual(proof['source_family'],'seed7_full')
            self.assertEqual(proof['geometry_source_id'],'derived7');self.assertEqual(proof['allowed_only_below_arc_m'],.008)
            self.assertLessEqual(proof['mesh_seam']['maximum_matching_vertex_distance_m'],2e-6)
            bad=deepcopy(catalogue);bad[0]['geometry_source_id']='another_geometry'
            with self.assertRaises(ValueError):generated.generated_joint_owner(entry,bad)
            # Current mesh no longer shares the ring: even a newly pinned asset
            # cannot retain the earlier seam proof merely by preserving labels.
            path=folder/details['SubStem_44']['file'];stage=Usd.Stage.Open(str(path))
            stage.GetPrimAtPath('/Mesh').GetAttribute('points').Set((ring+[.0001,0.,0.]).tolist())
            stage.GetRootLayer().Save();stage=None
            details['SubStem_44']['asset_sha256']=generated.sha256(path)
            owners,proof=generated.generated_joint_owner(entry,catalogue)
            self.assertEqual(owners,set());self.assertIsNone(proof)


class ClosedCaptureBoundaries(TestCase):
    def fixture(self,directory,*,exit_code=0):
        base=Path(directory);trial=base/'owned';capture=trial/'capture';capture.mkdir(parents=True)
        def write(path,value):
            path.write_text(json.dumps(value));return dict(path=str(path),sha256=generated.sha256(path))
        result=write(capture/'result.json',dict(schema='greenhouse.controlled_capture_result.v1',
            state='captured_pending_complete_annotation_and_visual_review',training_approved=False,accepted_training_increment=0,frames=[]))
        worker=write(trial/'owned_worker.json',dict(launcher_pid=222,owner_pid=111,command=[]))
        exit_pin=write(trial/'owned_exit.json',dict(returncode=exit_code,owned_worker=worker))
        start_pin=write(trial/'owner_started.json',{})
        write(trial/'owner_complete.json',dict(result=result,frames=0,training_approved=False,accepted_training_increment=0,owned_exit=exit_pin,owner_started=start_pin))
        return trial,capture,base/'out'
    def test_failed_owned_child_never_starts_annotation_or_writes_output(self):
        with TemporaryDirectory() as d:
            trial,capture,out=self.fixture(d,exit_code=1)
            with self.assertRaisesRegex(ValueError,'exit0'):generated.evaluate_capture(capture,out)
            self.assertFalse(out.exists())
    def test_changed_saved_capture_breaks_owner_pin(self):
        with TemporaryDirectory() as d:
            trial,capture,out=self.fixture(d)
            result=capture/'result.json';result.write_text(result.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Evidence changed'):generated.evaluate_capture(capture,out)
            self.assertFalse(out.exists())
    def test_other_capture_completion_cannot_be_reused(self):
        with TemporaryDirectory() as d:
            trial,capture,out=self.fixture(d)
            p=trial/'owner_complete.json';value=json.loads(p.read_text())
            value['result']['path']=str(Path(d)/'another_capture/result.json');p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'another capture'):generated.evaluate_capture(capture,out)
            self.assertFalse(out.exists())


class TwoCollectionPlanAuthority(TestCase):
    def fixture(self,directory):
        base=Path(directory)
        def write(name,value):
            path=base/name;path.write_text(json.dumps(value))
            return dict(path=str(path),sha256=generated.sha256(path))
        common=dict(family_assignments={'seed11_full':'train','seed13_full':'validation'},
                    source_bindings_sha256={'source_plant.usd':'a'*64})
        scene=write('overnight.json',dict(common,schedule='camera_original'))
        generator=write('orbit.json',dict(common,schedule='generator_envelope'))
        anchor=write('anchor.json',dict(source_collection_plan=scene['path'],source_bindings={scene['path']:scene['sha256']},
                                      source_family='seed11_full',split='train'))
        context=dict(source_collection_plan=scene,generator_source_collection_plan=generator,
                     full_scene_census=dict(source_collection_plan=scene),dataset_split='train')
        candidates=dict(source_plan=generator,source_family='seed11_full',records=[dict(anchor=anchor,source_family='seed11_full')],
                        source_bindings={anchor['path']:anchor['sha256']})
        return context,candidates
    def repin(self,spec,value):
        path=Path(spec['path']);path.write_text(json.dumps(value));return dict(path=str(path),sha256=generated.sha256(path))
    def test_distinct_original_camera_and_generator_plans_are_preserved(self):
        with TemporaryDirectory() as d:
            context,candidates=self.fixture(d)
            pins=generated.validate_collection_plan_provenance(context,candidates)
            self.assertNotEqual(context['source_collection_plan'],context['generator_source_collection_plan'])
            self.assertEqual(set(pins),{context['source_collection_plan']['path'],context['generator_source_collection_plan']['path'],
                                        candidates['records'][0]['anchor']['path']})
    def test_generator_plan_cannot_replace_camera_authority(self):
        with TemporaryDirectory() as d:
            context,candidates=self.fixture(d)
            context['source_collection_plan']=context['generator_source_collection_plan']
            with self.assertRaisesRegex(ValueError,'roles differ'):generated.validate_collection_plan_provenance(context,candidates)
            context,candidates=self.fixture(d)
            pin=candidates['records'][0]['anchor'];anchor=json.loads(Path(pin['path']).read_text())
            anchor['source_collection_plan']=context['generator_source_collection_plan']['path']
            anchor['source_bindings']={anchor['source_collection_plan']:context['generator_source_collection_plan']['sha256']}
            candidates['records'][0]['anchor']=self.repin(pin,anchor)
            candidates['source_bindings'][pin['path']]=candidates['records'][0]['anchor']['sha256']
            with self.assertRaisesRegex(ValueError,'actual scene collection plan'):
                generated.validate_collection_plan_provenance(context,candidates)
    def test_changed_split_or_source_geometry_is_not_compatible(self):
        for field in ('family_assignments','source_bindings_sha256'):
            with TemporaryDirectory() as d, self.subTest(field=field):
                context,candidates=self.fixture(d);pin=context['generator_source_collection_plan']
                value=json.loads(Path(pin['path']).read_text())
                if field=='family_assignments':value[field]['seed13_full']='train'
                else:value[field]['source_plant.usd']='b'*64
                updated=self.repin(pin,value);context['generator_source_collection_plan']=updated;candidates['source_plan']=updated
                with self.assertRaisesRegex(ValueError,'families or original assets'):
                    generated.validate_collection_plan_provenance(context,candidates)
    def test_camera_anchor_plan_hash_is_required(self):
        with TemporaryDirectory() as d:
            context,candidates=self.fixture(d);pin=candidates['records'][0]['anchor']
            anchor=json.loads(Path(pin['path']).read_text());anchor['source_bindings'][context['source_collection_plan']['path']]='f'*64
            updated=self.repin(pin,anchor);candidates['records'][0]['anchor']=updated;candidates['source_bindings'][pin['path']]=updated['sha256']
            with self.assertRaisesRegex(ValueError,'authenticate the actual scene'):
                generated.validate_collection_plan_provenance(context,candidates)

if __name__=='__main__':main()
