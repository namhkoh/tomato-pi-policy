"""Fresh interpreter guards: the native entry module must not bind USD pre-Kit."""
from pathlib import Path
import ast,inspect,subprocess,sys,unittest


class ImportOrder(unittest.TestCase):
    def test_fresh_entry_import_has_no_pre_app_usd_or_omni(self):
        root=Path(__file__).resolve().parents[3]
        script="""import sys,importlib.abc
class BlockNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('pxr','omni','isaacsim'):
   raise RuntimeError('Pre-SimulationApp native import: '+fullname)
sys.meta_path.insert(0,BlockNative())
from sim_data import native848_generated_farmerge_capture_v1
assert not any(k=='pxr' or k.startswith('pxr.') for k in sys.modules)
print('no_pre_app_native_imports')
"""
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('no_pre_app_native_imports',result.stdout)

    def test_overlay_is_imported_only_inside_post_startup_capture(self):
        path=Path(__file__).with_name('native848_generated_farmerge_capture_v1.py')
        tree=ast.parse(path.read_text())
        def imports(nodes):
            return [n for item in nodes for n in ast.walk(item) if isinstance(n,ast.ImportFrom)
                    and any(a.name=='native848_farmerge_overlay_v1' for a in n.names)]
        self.assertFalse(imports([n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))]))
        capture=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
        self.assertEqual(len(imports(capture.body)),1)


if __name__=='__main__':unittest.main()
