"""AST/runtime-interface tests, not evidence of native execution or throughput."""
import ast
import inspect
from pathlib import Path
import subprocess
import sys

from .. import native_greenhouse_pair
from . import scene, collector, contracts


def nodes(function):
    return ast.parse(inspect.getsource(function)).body[0].body


def test_exact_robot_application_matches_existing_native_worker_ast():
    old = nodes(native_greenhouse_pair.capture_pair)
    start = next(i for i, n in enumerate(old) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "root")
    expected = old[start:start+3]
    fresh = nodes(scene.restore_pose)
    assert [ast.dump(n) for n in fresh[-3:]] == [ast.dump(n) for n in expected]


def test_one_product_and_scene_outside_case_loop_with_reference_budget():
    tree = ast.parse(inspect.getsource(collector.collect))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "case"]
    assert len(loops) == 1
    loop = loops[0]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    products = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "render_product"]
    stages = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "prepare_scene"]
    assert len(products) == len(stages) == 1
    assert products[0] not in list(ast.walk(loop)) and stages[0] not in list(ast.walk(loop))
    # No 10k-file source verification or USD source scan in the per-frame loop.
    for n in ast.walk(loop):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id not in {"bind_all", "check_plan", "source_hashes", "prepare_plan"}
    source = inspect.getsource(collector.collect)
    assert "range(6)" in source and source.count("step_payload(rep, writer, subframes=8)") == 2
    assert "include_generated_plants=False" in source
    assert "substitute_plant" not in source and "generated_row" not in source
    assert "validate_native_static(" in source and "decode_native_instances(" in source
    assert "derive(" in source and "trace_review(" in source


def test_imports_and_help_do_not_import_isaac_or_create_outputs():
    code = ("import sys; from sim_data.native_original_capture import prepare, collector, audit; "
            "assert 'isaacsim' not in sys.modules and 'omni.usd' not in sys.modules")
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)
    for name in ("prepare", "collector", "audit"):
        result = subprocess.run([sys.executable, "-B", "-m", "sim_data.native_original_capture." + name, "--help"],
                                check=True, capture_output=True, text=True)
        assert "--output" in result.stdout


def test_code_binding_closure_covers_static_core_and_nested_audit():
    names = {Path(p).name for p in contracts.code_bindings()}
    assert {"audit.py", "native_sensor_payload.py", "native_clear_labels.py", "automated_native_review.py",
            "capture_scene.py", "robot_kinematics.py", "static_geometry_cache.py", "native_resolution_smoke.py"} <= names
    assert not any("test_" in Path(p).name for p in contracts.code_bindings())


def test_source_code_changed_after_import_is_not_blessed(tmp_path, monkeypatch):
    import pytest
    path = tmp_path / "unit_module.py"
    path.write_text("unit = 1")
    monkeypatch.setattr(contracts, "LOADED_IMPLEMENTATION", {str(path): contracts.sha256(path)})
    contracts.verify_loaded_code()
    path.write_text("unit = 2")
    with pytest.raises(ValueError, match="Changed or missing pinned"):
        contracts.verify_loaded_code()


def test_writer_creation_and_all_cleanup_are_inside_protected_lifetime():
    tree = ast.parse(inspect.getsource(collector.collect))
    protected = next(n for n in tree.body[0].body if isinstance(n, ast.Try))
    source = ast.unparse(protected)
    assert "make_writer" in source and "writer.attach" in source
    final = "\n".join(ast.unparse(n) for n in protected.finalbody)
    assert all(s in final for s in ("monitor.close", "cache.close", "writer.detach", "product.destroy"))
