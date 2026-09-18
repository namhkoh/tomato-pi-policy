"""Actual saved census exercises the exact consumer coverage constructor and frozen validator."""
import ast,copy,inspect,json,tempfile,unittest
from pathlib import Path
from . import native848_scene_sweep_9mm_v1 as ordinary
from . import native848_scene_sweep_raw_9mm_v1 as raw
from . import native848_unique9mm_joint_contract_v1 as frozen


class ActualCoverageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[3]
        cls.diagnostic=cls.root/'data/sim_data/diagnostics/native848_sweep55_coverage_contract_fix_20260918_v1'
        summary=json.loads((cls.root/'data/sim_data/diagnostics/native848_fixed144_sweep55_annotation_20260918_v2/result.json').read_text())
        cls.row=summary['records'][0]
        for key in ('annotation','coverage','metadata'):
            spec=cls.row[key];assert frozen.digest(spec['path'])==spec['sha256'];setattr(cls,key,json.loads(Path(spec['path']).read_text()))
        assert len(cls.annotation['targets'])==11068 and cls.row['strict_unique_automated_candidate']
    def make(self,module,folder):
        tree=ast.parse(inspect.getsource(module.evaluate_capture))
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cov' for t in n.targets))
        cov=self.coverage;ann=copy.deepcopy(self.annotation);obspin=ann['observation']
        # Small exact binding union for this isolated contract fixture; no frame is
        # accepted or exported. Production continues to authenticate its full union.
        pins={obspin['path']:obspin['sha256'],str(Path(frozen.__file__).resolve()):frozen.digest(frozen.__file__)}
        ns=dict(obspin=obspin,result=dict(context=cov['context']),census=dict(all_plant_roots=[dict(source_family=k,source_split=v) for k,v in cov['scene_source_families'].items()]),
            unknown=[],annotation=ann,catalogue=range(cov['active_catalogue_components']),inventory=range(cov['catalogue_petiole_count']),context=dict(census=cov['actual_fully_labeled_scene_census']),categories=cov['pixel_categories'],pins=pins)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'<actual coverage constructor>','exec'),ns)
        path=folder/'coverage.json';frozen.write(path,ns['cov']);ann['target_census']['full_scene_coverage']=frozen.pin(path)
        return ann,ns['cov'],path
    def test_both_actual_coverage_constructors_pass_frozen_same_path_contract(self):
        for module in (ordinary,raw):
            with self.subTest(module=module.__name__),tempfile.TemporaryDirectory(dir=self.diagnostic) as tmp:
                ann,cov,_=self.make(module,Path(tmp))
                self.assertEqual(cov['observation'],dict(path=cov['observation_path'],sha256=cov['observation_sha256']))
                self.assertEqual(cov['context'],dict(path=cov['context_path'],sha256=cov['context_sha256']))
                candidate=frozen.validate_census(ann)[0]
                self.assertEqual(candidate['target_id'],'fullpop_Gutter1_R_017/SubStem_41')
                frozen.target_answer(candidate,self.metadata['calibration'])
    def test_previous_missing_fields_reproduces_actual_failure(self):
        with tempfile.TemporaryDirectory(dir=self.diagnostic) as tmp:
            ann,cov,path=self.make(ordinary,Path(tmp));del cov['observation_path'];del cov['observation_sha256']
            path.write_text(json.dumps(cov));ann['target_census']['full_scene_coverage']=frozen.pin(path)
            with self.assertRaisesRegex(KeyError,'observation_path'):frozen.validate_census(ann)
    def test_wrong_actual_observation_still_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.diagnostic) as tmp:
            ann,cov,path=self.make(ordinary,Path(tmp));cov['observation_sha256']='0'*64
            path.write_text(json.dumps(cov));ann['target_census']['full_scene_coverage']=frozen.pin(path)
            with self.assertRaisesRegex(ValueError,'different actual frame'):frozen.validate_census(ann)

if __name__=='__main__':unittest.main(verbosity=2)
