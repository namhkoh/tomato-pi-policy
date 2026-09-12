import hashlib
from pathlib import Path
import pytest
from sim_physics.file_integrity import sha256_file


@pytest.mark.parametrize('size',[0,1,1024*1024-1,1024*1024,1024*1024+7,5*1024*1024])
def test_streaming_digest_matches_whole_file_without_read_bytes(tmp_path,monkeypatch,size):
    payload=(b'original source\x00\xff'*(size//17+1))[:size]
    path=tmp_path/'source.usd';path.write_bytes(payload)
    expected=hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(Path,'read_bytes',lambda self:pytest.fail('whole-file allocation'))
    assert sha256_file(path)==expected


def test_missing_source_fails_instead_of_claiming_unchanged(tmp_path):
    with pytest.raises(FileNotFoundError):sha256_file(tmp_path/'missing.usd')


def test_each_read_is_explicitly_memory_bounded(monkeypatch):
    class Stream:
        remaining=3*1024*1024+7
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):
            assert n==1024*1024
            count=min(n,self.remaining);self.remaining-=count
            return b'a'*count
    stream=Stream();monkeypatch.setattr(Path,'open',lambda self,mode:stream)
    assert sha256_file('synthetic')==hashlib.sha256(b'a'*(3*1024*1024+7)).hexdigest()
