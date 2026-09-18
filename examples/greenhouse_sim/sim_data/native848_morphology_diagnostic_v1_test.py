"""CPU-only diagnostic boundary and lossless persistence checks; no Isaac launch."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
import json
import os
import subprocess
import sys
import numpy as np
from PIL import Image
from . import native848_morphology_diagnostic_v1 as worker


def fixture():
    root=Path(__file__).resolve().parent
    return dict(schema=worker.REQUEST_SCHEMA,
        source_pose_plan=dict(path=str(root/'source_plan.json'),sha256=worker.SOURCE_PLAN_SHA),
        source_pose_id=worker.SOURCE_POSE_ID,target_component_id='SubStem_41',
        variants=[dict(qualification=dict(path=str(root/str(i)/'qualification.json'),sha256=str(i)*64),
                       manifest=dict(path=str(root/str(i)/'manifest.json'),sha256=str(i+2)*64)) for i in (1,2)],
        max_frames=2,frames_per_variant=1,source_family='seed53_full',source_split='train',
        training_approved=False,accepted_training_increment=0,implementation_bindings={str(root/'worker.py'):'a'*64})


class RequestBoundaries(TestCase):
    def test_exact_two_diagnostic_request(self):
        worker.require_request_shape(fixture())

    def test_frame_expansion_or_repeat_rejected(self):
        changes=[('max_frames',3),('max_frames',True),('frames_per_variant',2),
                 ('variants',fixture()['variants']*2),('variants',[fixture()['variants'][0]]*2)]
        for key,value in changes:
            with self.subTest(key=key,value=value):
                request=fixture();request[key]=value
                with self.assertRaises(ValueError):worker.require_request_shape(request)

    def test_old_schema_wrong_source_and_acceptance_rejected(self):
        for key,value in [('schema',worker.api.SCHEMA),('source_family','seed17_full'),
                ('source_split','validation'),('target_component_id','SubStem_42'),
                ('source_pose_id','different_pose'),('training_approved',True),
                ('accepted_training_increment',1),('accepted_training_increment',False)]:
            with self.subTest(key=key):
                request=fixture();request[key]=value
                with self.assertRaises(ValueError):worker.require_request_shape(request)
        request=fixture();request['source_pose_plan']['sha256']='f'*64
        with self.assertRaises(ValueError):worker.require_request_shape(request)

    def test_exact_pin_and_no_extra_request_scope(self):
        for pin in [dict(path='relative',sha256='a'*64),dict(path=str(Path(__file__).resolve()),sha256='z'*64),
                    dict(path=str(Path(__file__).resolve()),sha256='a'*64,approved=True)]:
            with self.assertRaises(ValueError):worker.require_pin(pin)
        request=fixture();request['capture_count']=3
        with self.assertRaises(ValueError):worker.require_request_shape(request)

    def test_import_does_not_initialize_native_usd(self):
        command=[sys.executable,'-B','-c',
            'import sys; from sim_data import native848_morphology_diagnostic_v1; '
            'assert not any(k=="pxr" or k.startswith("pxr.") or k=="isaacsim" for k in sys.modules)']
        result=subprocess.run(command,env=dict(os.environ),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


class DiagnosticGeometry(TestCase):
    def locator(self):
        report={'components':{'MainStem_1':{'type':'main_stem'},'SubStem_41':dict(type='sub_stem',
            parent='MainStem_1',deleafed=False,translation_plant_m=[0.,0.,-1.],
            attachment_plant_m=[0.,0.,-1.],axis_plant=[1.,0.,0.],
            capsules_local_m=[[[0.,0.,0.,.004],[.1,0.,0.,.005]]])}}
        cal=dict(camera_to_world_usd_row_vectors=np.eye(4).tolist(),intrinsics=[[500.,0.,424.],[0.,500.,204.],[0.,0.,1.]],
                 resolution=[848,408],clipping_range_m=[.01,20.])
        return worker.generated_locator(report,'SubStem_41',np.eye(4),cal)

    def test_exact_arc_is_recomputed_and_proxy_never_eligible(self):
        locator=self.locator();g=locator['geometry'];nom=g['nominal']
        self.assertEqual(nom['arc_m'],.009)
        np.testing.assert_allclose(nom['world_m'],[.009,0.,-1.],rtol=0,atol=1e-14)
        self.assertEqual(len(g['junction_to_cut']),19)
        self.assertAlmostEqual(nom['proxy_radius_upper_bound_m'],.00409)
        self.assertNotIn('"radius_m"',json.dumps(locator))
        self.assertFalse(locator['physical_surface_radius_qualified'])
        self.assertFalse(locator['single_answer_verified'])
        self.assertFalse(locator['training_approved'])
        self.assertEqual(locator['eligible_candidate_count'],'not_evaluated')

    def test_native_pixel_owner_and_depth_are_observations_only(self):
        loc=self.locator();depth=np.ones((408,848),dtype=np.float32)
        valid=np.ones(depth.shape,dtype=bool);ids=np.full(depth.shape,19,dtype=np.uint32)
        catalogue=[dict(prim_path='/World/Plant',component_id='ancestor'),
                   dict(prim_path='/World/Plant/SubStem_41',component_id='SubStem_41')]
        result=worker.pixel_diagnostic(loc,depth,valid,ids,{19:'/World/Plant/SubStem_41/Mesh'},catalogue)
        self.assertEqual(result['component_owner']['component_id'],'SubStem_41')
        self.assertEqual(result['centerline_optical_depth_residual_m'],0.)
        self.assertFalse(result['native_visible_eligible_claimed'])
        self.assertFalse(result['depth_consistency_accepted'])
        valid[:]=False
        result=worker.pixel_diagnostic(loc,depth,valid,ids,{},catalogue)
        self.assertIsNone(result['native_optical_depth_m']);self.assertIsNone(result['component_owner'])

    def test_out_of_frame_never_indexes_buffers(self):
        loc=self.locator();loc['geometry']['nominal']['projected'].update(projection_status='out_of_frame',pixel_xy=[900.,200.])
        result=worker.pixel_diagnostic(loc,None,None,None,{},[])
        self.assertNotIn('renderer_id',result);self.assertFalse(result['native_visible_eligible_claimed'])


class NativeIdentity(TestCase):
    def test_executable_is_canonical_once_for_record_and_owner(self):
        args=SimpleNamespace(request=Path(__file__),request_sha256='a'*64,output=Path(__file__).parent/'unused')
        expected=worker.native_command(args)
        self.assertEqual(expected[0],str(Path('D:/isaac-sim-6.0.1/python.bat')))
        native=dict(ProcessId=os.getpid(),ParentProcessId=1002)
        command=dict(ProcessId=1002,ParentProcessId=1001);owner=dict(ProcessId=1001,ParentProcessId=1000)
        gate=SimpleNamespace(snapshot=lambda:[owner,command,native],
            one=lambda rows,pid:next(row for row in rows if row['ProcessId']==pid),
            same_identity=lambda a,b:a==b,command_child=lambda cmd,own,argv:None,native_child=lambda nat,cmd,argv:None)
        with patch.object(worker.production,'identity_gate',gate):
            identity=worker.record_native_identity(args)
            self.assertEqual(identity['command'],expected)
            worker.validate_native_identity(identity,expected,owner_identity=owner)
            stale=deepcopy(identity);stale['command'][0]='D:/different/python.bat'
            with self.assertRaises(ValueError):worker.validate_native_identity(stale,expected)
            with self.assertRaises(ValueError):worker.validate_native_identity(identity,expected,owner_identity=native)


class Persistence(TestCase):
    def test_new_typed_observation_roundtrips_native_arrays_and_pins(self):
        y,x=np.indices((408,848));rgb=np.stack((x%256,y%256,(x+y)%256),axis=-1).astype(np.uint8)
        depth=(x/1000+y/2000).astype(np.float32);ids=(x+848*y).astype(np.uint32)
        valid=(ids%3)!=0;alpha=(ids%251).astype(np.uint8)
        observation=dict(schema=worker.OBSERVATION_SCHEMA,**worker.SCOPE,observation_id='morphology_1',
                         timing={},synchronization=dict(request_index=8))
        with TemporaryDirectory() as directory:
            sink=worker.FrameSink(Path(directory)/'frames',max_frames=1,max_bytes=64*2**20,workers=1)
            try:
                sink.submit(observation,rgb,depth,ids,valid,alpha);receipts=sink.close()
            finally:sink.close()
            self.assertEqual(len(receipts),1);receipt=receipts[0]
            self.assertEqual(receipt['sha256'],worker.sha256(receipt['path']))
            saved=worker.read_json(receipt['path']);self.assertEqual(saved['schema'],worker.OBSERVATION_SCHEMA)
            self.assertFalse(saved['training_approved']);self.assertEqual(saved['accepted_training_increment'],0)
            for pin in saved['files'].values():self.assertEqual(pin['sha256'],worker.sha256(pin['path']))
            np.testing.assert_array_equal(np.asarray(Image.open(saved['files']['rgb']['path'])),rgb)
            with np.load(saved['files']['buffers']['path']) as buffers:
                for key,expected in [('depth_m',depth),('renderer_instance_id',ids),('depth_valid',valid),('rgba_alpha',alpha)]:
                    self.assertEqual(buffers[key].dtype,expected.dtype);np.testing.assert_array_equal(buffers[key],expected)


if __name__=='__main__':main()
