"""File-backed CPU tests. No native jobs or global-reader monkeypatching."""
import hashlib
import os
from pathlib import Path

import pytest

from .io import Reader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link_dir(target, link):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi
        _winapi.CreateJunction(str(target), str(link))


def unlink_dir(link):
    if link.is_symlink():
        link.unlink()
    else:
        link.rmdir()  # Only the junction, never its target tree.


@pytest.fixture
def asset(tmp_path):
    path = tmp_path / "original.bin"
    path.write_bytes(b"original")
    return path


def test_repeated_paths_keep_two_full_hashes(asset):
    reader = Reader()
    pin = digest(asset)
    for _ in range(100):
        assert reader.bind(asset, pin) == asset
    reader.finish()
    counts = reader.diagnostics()
    assert counts["initial_resolutions"] == counts["final_resolutions"] == 1
    assert counts["initial_hashes"] == counts["final_hashes"] == 1
    assert not counts["stat_only_byte_acceptance"]


def test_noncanonical_route_not_collapsed(asset):
    directory = asset.parent / "ordinary"
    directory.mkdir()
    reader = Reader()
    reader.bind(directory / ".." / asset.name, digest(asset))
    reader.bind(asset, digest(asset))
    reader.finish()
    assert reader.diagnostics()["original_paths"] == 2
    assert reader.diagnostics()["unique_files"] == 1


@pytest.mark.parametrize("operation", ["bind", "json", "map"])
def test_conflicting_pin_before_cached_return(asset, operation):
    reader = Reader()
    reader.bind(asset, digest(asset))
    with pytest.raises(ValueError, match="Conflicting"):
        if operation == "map":
            reader.bindings_from({str(asset): "a" * 64})
        else:
            getattr(reader, operation)(asset, "a" * 64)
    with pytest.raises(ValueError, match="single-use or failed"):
        reader.finish()


@pytest.mark.parametrize("bad", [None, "", "g" * 64, "A" * 64, True])
def test_invalid_pin_cannot_use_cache(asset, bad):
    reader = Reader()
    reader.bind(asset, digest(asset))
    with pytest.raises(ValueError, match="SHA256"):
        reader.bind(asset, bad)


@pytest.mark.parametrize("change", ["digest", "delete", "add"])
def test_same_map_identity_does_not_hide_mutation(asset, change):
    bindings = {str(asset): digest(asset)}
    reader = Reader()
    reader.bindings_from(bindings)
    if change == "digest":
        bindings[str(asset)] = "a" * 64
    elif change == "delete":
        bindings.clear()
    else:
        bindings[str(asset.parent / "missing")] = "a" * 64
    with pytest.raises(ValueError):
        reader.bindings_from(bindings)


def test_map_mutation_before_finish_fails(asset):
    bindings = {str(asset): digest(asset)}
    reader = Reader()
    reader.bindings_from(bindings)
    bindings[str(asset)] = "a" * 64
    with pytest.raises(ValueError, match="Binding map changed"):
        reader.finish()


def test_equal_maps_only_hash_unique_content(asset):
    bindings = {str(asset): digest(asset)}
    reader = Reader()
    for _ in range(20):
        reader.bindings_from(bindings)
        reader.bindings_from(dict(bindings))
    reader.finish()
    assert reader.diagnostics()["initial_hashes"] == reader.diagnostics()["final_hashes"] == 1


def test_same_size_restored_mtime_tamper_fails(asset):
    before = asset.stat()
    reader = Reader()
    reader.bind(asset, digest(asset))
    asset.write_bytes(b"tampered")
    os.utime(asset, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="Source changed"):
        reader.finish()


def test_deleted_file_fails(asset):
    reader = Reader()
    reader.bind(asset, digest(asset))
    asset.unlink()
    with pytest.raises(ValueError, match="hash bound file"):
        reader.finish()


def test_new_reader_always_rehashes(asset):
    pin = digest(asset)
    reader = Reader()
    reader.bind(asset, pin)
    reader.finish()
    with pytest.raises(ValueError, match="single-use"):
        reader.bind(asset, pin)
    asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Changed bound file"):
        Reader().bind(asset, pin)


def test_metadata_hashes_exact_parsed_bytes(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"hello":1}')
    reader = Reader()
    assert reader.json(path, digest(path)) == {"hello": 1}
    assert reader.json(path, digest(path)) == {"hello": 1}
    reader.finish()
    counts = reader.diagnostics()
    assert counts["metadata_hashes"] == counts["initial_hashes"] == counts["final_hashes"] == 1


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_bad_metadata_fails(tmp_path, text):
    path = tmp_path / "metadata.json"
    path.write_text(text)
    reader = Reader()
    with pytest.raises(ValueError):
        reader.json(path, digest(path))
    with pytest.raises(ValueError, match="failed"):
        reader.finish()


def test_metadata_changed_between_hash_and_parse(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"x":1}')
    pin = digest(path)

    class ChangeAfterHash(Reader):
        def _hash(self, path, phase):
            actual = super()._hash(path, phase)
            if phase == "initial":
                Path(path).write_text('{"x":2}')
            return actual

    with pytest.raises(ValueError, match="File changed during read"):
        ChangeAfterHash().json(path, pin)


@pytest.fixture
def routes(tmp_path):
    left, right, alias = (tmp_path / name for name in ("left", "right", "alias"))
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        (directory / "file").write_bytes(b"identical bytes")
    link_dir(left, alias)
    return left, right, alias


def test_aliases_hash_canonical_file_only_twice(routes):
    left, _, alias = routes
    reader = Reader()
    pin = digest(left / "file")
    reader.bind(left / "file", pin)
    reader.bind(alias / "file", pin)
    reader.finish()
    counts = reader.diagnostics()
    assert counts["unique_files"] == counts["initial_hashes"] == counts["final_hashes"] == 1
    assert counts["final_resolutions"] == 2


def test_alias_conflicting_pin_fails(routes):
    left, _, alias = routes
    reader = Reader()
    reader.bind(left / "file", digest(left / "file"))
    with pytest.raises(ValueError, match="Conflicting"):
        reader.bind(alias / "file", "a" * 64)


@pytest.mark.parametrize("noncanonical", [False, True])
def test_retarget_same_bytes_rejected(routes, noncanonical):
    left, right, alias = routes
    for directory in (left, right):
        (directory / "child").mkdir()
    route = alias / "child" / ".." / "file" if noncanonical else alias / "file"
    reader = Reader()
    reader.bind(route, digest(left / "file"))
    unlink_dir(alias)
    link_dir(right, alias)
    reader.bind(route, digest(left / "file"))  # provisional until mandatory finish
    with pytest.raises(ValueError, match="path target changed"):
        reader.finish()


def test_new_symlink_hidden_by_dotdot_rejected(asset):
    ordinary, alternate = asset.parent / "ordinary", asset.parent / "alternate"
    ordinary.mkdir()
    alternate.mkdir()
    reader = Reader()
    reader.bind(ordinary / ".." / asset.name, digest(asset))
    ordinary.rmdir()
    link_dir(alternate, ordinary)
    # Windows normpath can yield the SAME final file; topology still differs.
    with pytest.raises(ValueError, match="path (target|topology) changed"):
        reader.finish()


def test_link_text_change_same_final_target_rejected(routes):
    left, _, alias = routes
    other = alias.parent / "second_alias"
    link_dir(left, other)
    reader = Reader()
    reader.bind(alias / "file", digest(left / "file"))
    unlink_dir(alias)
    link_dir(other, alias)
    with pytest.raises(ValueError, match="topology changed"):
        reader.finish()


def test_indirect_alias_retarget_same_final_target_rejected(routes):
    left, _, alias = routes
    outer, alternate = alias.parent / "outer", alias.parent / "alternate"
    link_dir(alias, outer)
    link_dir(left, alternate)
    reader = Reader()
    reader.bind(outer / "file", digest(left / "file"))
    unlink_dir(alias)
    link_dir(alternate, alias)
    with pytest.raises(ValueError, match="topology changed"):
        reader.finish()


def test_retarget_during_final_hash_rejected(routes):
    left, right, alias = routes

    class RetargetDuringFinish(Reader):
        def _hash(self, path, phase):
            actual = super()._hash(path, phase)
            if phase == "final":
                unlink_dir(alias)
                link_dir(right, alias)
            return actual

    reader = RetargetDuringFinish()
    reader.bind(alias / "file", digest(left / "file"))
    with pytest.raises(ValueError, match="target changed"):
        reader.finish()


def test_safe_path_alias_escape_and_future_output(routes):
    left, right, _ = routes
    with pytest.raises(ValueError, match="escapes"):
        Reader().safe(right, "../alias/file")
    future = left / "future"
    reader = Reader()
    reader.resolve(future)
    link_dir(right, future)
    with pytest.raises(ValueError, match="target changed"):
        reader.finish()


def test_relative_binding_map_rejected(asset):
    with pytest.raises(ValueError, match="Absolute"):
        Reader().bindings_from({asset.name: digest(asset)})
