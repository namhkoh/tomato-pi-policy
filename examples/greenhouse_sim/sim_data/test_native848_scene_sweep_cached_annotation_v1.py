"""Source-change, publication-order and scoped-cache guards; no native calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import os,unittest
from . import native848_scene_sweep_cached_annotation_v1 as cache

class CacheGuards(unittest.TestCase):
    def test_changed_same_size_same_mtime_source_blocks_publication(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.dat';source.write_bytes(b'original');before=source.stat()
            session=cache.cache.Session();expected=session.digest(source)
            source.write_bytes(b'modified');os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns))
            self.assertEqual(session.digest(source),expected)
            value=dict(frames_evaluated=1,records=[{}],completed_whole_capture_authenticated=True,
                all_committed_frames_evaluated=True,source_bindings={str(source):expected})
            with self.assertRaisesRegex(ValueError,'changed before final closure'):
                cache.close_and_publish(session,value,root,cache.ordinary,'ordinary',1)
            self.assertFalse((root/'result.json').exists());self.assertFalse((root/'hash_validation_session.json').exists())
    def test_closing_rehash_precedes_receipt_and_terminal(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.dat';source.write_bytes(b'fixed');session=cache.cache.Session();h=session.digest(source)
            value=dict(frames_evaluated=1,records=[{}],completed_whole_capture_authenticated=True,
                all_committed_frames_evaluated=True,source_bindings={str(source):h});events=[];real=cache.real_save
            def save(path,value):
                self.assertTrue(session.closed);self.assertEqual(session.closing_reads,session.hash_reads)
                if Path(path).name=='result.json':self.assertTrue((root/'hash_validation_session.json').exists())
                events.append(Path(path).name);real(path,value)
            with patch.object(cache,'real_save',save):result=cache.close_and_publish(session,value,root,cache.ordinary,'ordinary',1)
            self.assertEqual(events,['hash_validation_session.json','result.json']);self.assertEqual(result['finite_validation'],cache.pin(root/'hash_validation_session.json'))
    def test_scope_restores_frozen_functions_after_exception(self):
        old=cache.ambiguity.dataset.digest;oldverify=cache.ambiguity.verify_bindings;consumerhash=cache.raw.sha256;session=cache.new_session()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with cache.hash_scope(session,cache.raw):
                self.assertNotEqual(cache.ambiguity.dataset.digest,old);self.assertNotEqual(cache.raw.sha256,consumerhash);raise RuntimeError('fixture')
        self.assertIs(cache.ambiguity.dataset.digest,old);self.assertIs(cache.ambiguity.verify_bindings,oldverify);self.assertIs(cache.raw.sha256,consumerhash);self.assertFalse(cache.validation._ACTIVE)
    def test_no_cross_operation_hash_reuse(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'input.dat';source.write_bytes(b'first');a=cache.cache.Session();old=a.digest(source);a.close();source.write_bytes(b'other');b=cache.cache.Session();self.assertNotEqual(b.digest(source),old);self.assertEqual(b.hash_reads,1)
    def test_actual_bytes_required_even_with_expected_pin(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'input.dat';source.write_bytes(b'first');session=cache.cache.Session()
            with self.assertRaisesRegex(ValueError,'Changed pinned file bytes'):session.authenticate(source,'0'*64)
    def test_source_union_cannot_expand_after_closing(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a';b=root/'b';a.write_bytes(b'a');b.write_bytes(b'b');session=cache.cache.Session();session.digest(a);session.close()
            with self.assertRaisesRegex(ValueError,'Cannot expand'):session.digest(b)
    def test_partial_frame_result_cannot_publish(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);session=cache.cache.Session();value=dict(frames_evaluated=1,records=[{}],completed_whole_capture_authenticated=True,all_committed_frames_evaluated=True,source_bindings={})
            with self.assertRaisesRegex(ValueError,'Exact completed frame schedule'):cache.close_and_publish(session,value,root,cache.ordinary,'ordinary',2)
            self.assertFalse((root/'result.json').exists())

if __name__=='__main__':unittest.main()
