import json
import zipfile
import pytest
from sim_data import release_archive as pack
from sim_data.depth_preview import sha256


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'release';root.mkdir();(root/'sample.bin').write_bytes(b'native bytes unchanged')
    (root/'manifest.json').write_text(json.dumps({'files_sha256':{'sample.bin':sha256(root/'sample.bin')}}))
    # Isolate packaging mechanics. This is not a real validated dataset fixture.
    monkeypatch.setattr(pack,'validate',lambda _:{'rows':1})
    return root,tmp_path/'transfer.zip'


def test_archive_contains_only_bound_bytes_and_checksum(fixture):
    root,destination=fixture
    (root/'unrelated_secret.txt').write_text('must not be packaged')
    result=pack.archive(root,destination)
    assert result['sha256']==sha256(destination)
    assert destination.with_suffix('.zip.sha256').read_text().split()[0]==result['sha256']
    with zipfile.ZipFile(destination) as bundle:
        assert set(bundle.namelist())=={'grounding_release/manifest.json','grounding_release/sample.bin'}
        assert bundle.read('grounding_release/sample.bin')==(root/'sample.bin').read_bytes()
    assert not destination.with_suffix('.zip.partial').exists()
    with pytest.raises(ValueError,match='overwrite'): pack.archive(root,destination)


def test_failed_normal_validator_cannot_create_archive(fixture,monkeypatch):
    root,destination=fixture
    def refuse(_): raise ValueError('Incomplete release')
    monkeypatch.setattr(pack,'validate',refuse)
    with pytest.raises(ValueError,match='Incomplete release'): pack.archive(root,destination)
    assert not destination.exists() and not destination.with_suffix('.zip.partial').exists()


def test_changed_source_leaves_only_unpublished_partial(fixture):
    root,destination=fixture
    (root/'sample.bin').write_bytes(b'changed after validation')
    with pytest.raises(ValueError,match='changed during copy'): pack.archive(root,destination)
    assert not destination.exists() and destination.with_suffix('.zip.partial').is_file()
    assert not destination.with_suffix('.zip.sha256').exists()


def test_unsafe_member_and_nested_output_are_rejected(fixture):
    root,destination=fixture
    with pytest.raises(ValueError,match='outside'): pack.archive(root,root/'nested.zip')
    (root/'manifest.json').write_text(json.dumps({'files_sha256':{'../secret':'hash'}}))
    with pytest.raises(ValueError,match='Nonportable'): pack.archive(root,destination)
    assert not destination.exists()
