"""CPU saved-mesh fixtures mock catalogue verification explicitly.

The real-artifact diagnostic separately exercises the unmocked catalogue.
No tests write original assets, bound implementations, or captures.
"""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from . import morphology as m
from .. import audit, morphology_context

Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def mesh(path, offset=0):
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Plant")
    stage.SetDefaultPrim(root.GetPrim())
    obj = UsdGeom.Mesh.Define(stage, "/Plant/Mesh")
    obj.CreatePointsAttr([(.06, offset, .01), (.07, offset + .02, .015), (.09, offset, .01)])
    obj.CreateFaceVertexCountsAttr([3])
    obj.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()


@pytest.fixture
def saved(tmp_path, monkeypatch):
    source, output = tmp_path / "T0", tmp_path / "variant"
    source.mkdir()
    output.mkdir()
    components = [
        dict(id="Main", type="main_stem", parent=None, file="Main.usda",
             transform=dict(translate=[0, 0, 0]), attach_point=[0, 0, -.1], axis=[0, 0, 1],
             capsules=[[[4, 0, -.1, .005], [4, 0, .2, .005]],
                       [[0, 0, -.1, .005], [0, 0, .2, .005]]]),
        dict(id="Petiole", type="sub_stem", parent="Main", file="Petiole.usda",
             transform=dict(translate=[.01, 0, 0]), attach_point=[.01, 0, 0], axis=[1, 0, 0],
             deleafed=False, capsules=[[[0, 0, 0, .002], [.06, .01, .02, .001]]]),
        dict(id="Leaf", type="leaf", parent="Petiole", file="Leaf.usda",
             transform=dict(translate=[0, 0, 0]), attach_point=[.07, .01, .02], axis=[0, 1, 0], capsules=[]),
    ]
    raw = dict(generator="fixture", version="1", units="meters", up_axis="Z", component_count=3,
               components=components)
    for c in components:
        mesh(source / c["file"])
        mesh(output / c["file"], offset=.03 if c["id"] == "Leaf" else 0)
    write_json(source / "manifest.json", raw)
    generated = deepcopy(raw)
    generated["components"][1]["capsules"][0][1][1] = .03
    write_json(output / "manifest.json", generated)
    assignments = {**{f"T{i}": "train" for i in range(16)},
                   **{f"V{i}": "validation" for i in range(4)}, **{f"E{i}": "test" for i in range(4)}}
    source_pins = {str(p): m._hash(p) for p in source.iterdir()}
    plan = dict(schema_version="greenhouse.grounding_collection_plan.v1",
                state="ready_for_synthetic_grounding_capture", training_dataset_approved=False,
                configuration=dict(clear_capture="robot_head_close_diffuse_v1",
                                   source_geometry="unmodified_native_components"),
                family_assignments=assignments, source_bindings_sha256=source_pins, jobs=[], selection_audit=[])
    for family, split in assignments.items():
        targets = [dict(target_id="T0/Petiole", source_plant_id="T0", split_group="T0")] if family == "T0" else []
        plan["jobs"].append(dict(job_id=family, plant_family=family, split=split, targets=targets,
                                 source_manifest_path=str(source / "manifest.json")))
        plan["selection_audit"].append(dict(family=family, geometry_candidates=len(targets), selected=len(targets),
            exclusions=[dict(target_id=t["target_id"], state="geometry_candidate") for t in targets]))
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    receipt = dict(version=m.SUPPORTED_VERSION, source_family="T0", split="train", split_group="T0",
        source_manifest_path=str(source / "manifest.json"), source_plan_sha256=m._hash(plan_path),
        frozen_family_assignments=assignments, source_bindings=source_pins,
        output_hashes={p.name: m._hash(p) for p in output.iterdir()},
        targets=[dict(component_id="Petiole", source_target_id="T0/Petiole", conservative_view_cap_group="T0/Petiole")])
    write_json(output / "qualification.json", receipt)
    calls = []

    def catalogue(directory, path):
        calls.append((directory, path))
        return dict(qualification_sha256=m._hash(directory / "qualification.json"),
                    source_plan_sha256=m._hash(path), source_family="T0", split_group="T0", variant_id="variant",
                    report=audit.audit_manifest(directory / "manifest.json"),
                    rows=[dict(component_id="Petiole")], rejected=[])

    monkeypatch.setattr(m.plant_variant_catalogue, "load_for_inspection", catalogue)
    return dict(source=source, output=output, plan_path=plan_path, receipt=receipt, plan=plan, calls=calls,
                pin=m.OutputPin(output, m._hash(output / "qualification.json"), plan_path, m._hash(plan_path)))


def repin(saved):
    write_json(saved["plan_path"], saved["plan"])
    saved["receipt"]["source_plan_sha256"] = m._hash(saved["plan_path"])
    write_json(saved["output"] / "qualification.json", saved["receipt"])
    saved["pin"] = m.OutputPin(saved["output"], m._hash(saved["output"] / "qualification.json"),
                               saved["plan_path"], m._hash(saved["plan_path"]))


def test_saved_vertices_chains_ancestry_and_same_output_replay(saved):
    before = {str(p): m._hash(p) for p in saved["output"].iterdir()}
    a = m.extract_output(saved["pin"])
    assert a == m.extract_output(saved["pin"]) and len(saved["calls"]) == 2 and a["holds"] == []
    original, generated = a["records"]
    assert original["context_id"] == generated["source_context_id"]
    assert original["source_target"] == generated["source_target"] == "T0/Petiole"
    assert generated["target_id"] == "variant/Petiole"
    assert generated["parent_chain"]["index"] == 1
    assert generated["input_geometry"]["current_petiole"][1][1] == .03
    assert original["input_geometry"]["current_petiole"][1][1] == .01
    # Actual saved generated vertices, not the source leaf or predicted transport.
    assert generated["input_geometry"]["leaves"][0]["centroid"][1] == pytest.approx(.03 + .02 / 3)
    assert not morphology_context.equivalent(original["descriptor"], generated["descriptor"], .001)
    assert a["frozen_splits"] == saved["plan"]["family_assignments"]
    for flag in ("training_approved", "qualified_geometry", "calibration_validated", "source_cap_reset",
                 "new_biological_family", "native_capture_verified", "global_inventory_complete"):
        assert a[flag] is False
    assert all(row["qualified_geometry"] is False for row in a["records"])
    assert before == {str(p): m._hash(p) for p in saved["output"].iterdir()}


@pytest.mark.parametrize("what", ["qualification", "plan", "generated_mesh", "generated_manifest", "source_mesh", "source_manifest"])
def test_tampering_raises_not_a_domain_hold(saved, what):
    paths = dict(qualification=saved["output"] / "qualification.json", plan=saved["plan_path"],
                 generated_mesh=saved["output"] / "Leaf.usda", generated_manifest=saved["output"] / "manifest.json",
                 source_mesh=saved["source"] / "Leaf.usda", source_manifest=saved["source"] / "manifest.json")
    with paths[what].open("ab") as stream:
        stream.write(b"\n ")
    with pytest.raises(ValueError, match="Changed bound file"):
        m.extract_output(saved["pin"])


def test_tampering_during_catalogue_and_loaded_code_detected(saved, monkeypatch, tmp_path):
    catalogue = m.plant_variant_catalogue.load_for_inspection
    def mutate(directory, path):
        result = catalogue(directory, path)
        with (directory / "Leaf.usda").open("ab") as stream:
            stream.write(b"\n ")
        return result
    monkeypatch.setattr(m.plant_variant_catalogue, "load_for_inspection", mutate)
    with pytest.raises(ValueError, match="Unbound descriptor mesh|Changed bound file"):
        m.extract_output(saved["pin"])
    code = tmp_path / "fixture_code.py"
    code.write_text("old", encoding="utf-8")
    monkeypatch.setattr(m, "_LOADED_CODE", {str(code): m._hash(code)})
    code.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed bound file"):
        m.extract_output(saved["pin"])


@pytest.mark.parametrize("case", ["version", "failed", "split", "ancestry", "escape", "missing_pin", "code_replay"])
def test_unsupported_version_vs_integrity_errors(saved, monkeypatch, case):
    if case == "version":
        saved["receipt"]["version"] = "curved_relocated_petiole_static.v1"
    elif case == "failed":
        (saved["output"] / "FAILED.json").write_text("{}")
    elif case == "split":
        saved["receipt"]["split"] = "test"
    elif case == "ancestry":
        saved["receipt"]["targets"][0]["source_target_id"] = "T1/Petiole"
    elif case == "escape":
        saved["receipt"]["output_hashes"]["../escaped"] = "0" * 64
    elif case == "missing_pin":
        del saved["receipt"]["output_hashes"]["manifest.json"]
    else:
        def refuse(*args):
            raise ValueError("Recipe implementation changed; explicitly regenerate")
        monkeypatch.setattr(m.plant_variant_catalogue, "load_for_inspection", refuse)
    repin(saved)
    if case == "version":
        result = m.extract_output(saved["pin"])
        assert result["records"] == [] and result["holds"][0]["reason"] == "unsupported_generator_version"
        assert saved["calls"] == []
    else:
        with pytest.raises(ValueError):
            m.extract_output(saved["pin"])


@pytest.mark.parametrize("keys", [[], ["Other"], ["Petiole", "Petiole"]])
def test_explicit_target_subset_required(saved, keys):
    with pytest.raises(ValueError, match="subset"):
        m.extract_output(saved["pin"], component_ids=keys)


def test_local_transform_held_without_descriptor(saved):
    p = saved["output"] / "Leaf.usda"
    stage = Usd.Stage.Open(str(p))
    UsdGeom.Xformable(stage.GetDefaultPrim()).AddTranslateOp().Set((1, 0, 0))
    stage.GetRootLayer().Save()
    saved["receipt"]["output_hashes"][p.name] = m._hash(p)
    repin(saved)
    result = m.extract_output(saved["pin"])
    assert result["records"] == []
    assert result["holds"][0]["reason"] == "nonidentity_local_mesh_transform"


def test_nearest_valid_parent_and_ambiguous_or_degenerate_hold():
    parent = dict(translation_plant_m=[0, 0, 0], capsules_local_m=[
        [[0, 0, 0, .01], [0, 0, 0, .01]], [[1, 0, -1, .01], [1, 0, 1, .01]]])
    _, evidence = m._nearest_parent(parent, np.zeros(3))
    assert evidence["index"] == 1 and evidence["skipped"][0]["index"] == 0
    parent["capsules_local_m"].append([[-1, 0, -1, .01], [-1, 0, 1, .01]])
    with pytest.raises(m._Unsupported, match="equidistant"):
        m._nearest_parent(parent, np.zeros(3))
    parent["capsules_local_m"] = parent["capsules_local_m"][:1]
    with pytest.raises(m._Unsupported, match="no_valid"):
        m._nearest_parent(parent, np.zeros(3))


def test_actual_extracted_inputs_rigid_scale_nuisance(saved):
    row = m.extract_output(saved["pin"])["records"][1]
    inputs = row["input_geometry"]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    for scale in (.4, 1., 100.):
        transformed = {}
        for key in ("reference_petiole", "parent", "current_petiole"):
            q = np.array(inputs[key])
            transformed[key] = np.column_stack((q[:, :3] @ rotation.T * scale + [1, 2, 3], q[:, 3] * scale))
        transformed["leaves"] = [dict(attachment=np.array(l["attachment"]) @ rotation.T * scale + [1, 2, 3],
            centroid=np.array(l["centroid"]) @ rotation.T * scale + [1, 2, 3], axis=np.array(l["axis"]) @ rotation.T,
            covariance=rotation @ np.array(l["covariance"]) @ rotation.T * scale ** 2) for l in inputs["leaves"]]
        value = morphology_context.descriptor(**transformed)
        assert morphology_context.equivalent(row["descriptor"], value, 1e-9)
        # Rounded fingerprints may straddle a rounding boundary under numerical
        # nuisance. They are NOT the authoritative tolerance comparator.
        assert len(morphology_context.fingerprint(value)) == 64


@pytest.mark.parametrize("domain,reason", [
    ("no_leaves", "requires_nonempty_direct_leaves_only"),
    ("nested_leaf", "requires_nonempty_direct_leaves_only"),
    ("multiple_chains", "petiole_requires_one_saved_chain"),
    ("parallel_frame", "unsupported_descriptor_frame"),
])
def test_unsupported_geometry_domains_held(saved, monkeypatch, domain, reason):
    loader = m.plant_variant_catalogue.load_for_inspection
    def catalogue(directory, path):
        value = loader(directory, path)
        c = value["report"]["components"]
        if domain == "no_leaves":
            del c["Leaf"]
        elif domain == "nested_leaf":
            c["Second"] = {**c["Leaf"], "id": "Second", "parent": "Leaf"}
        elif domain == "multiple_chains":
            c["Petiole"]["capsules_local_m"] *= 2
        else:
            # Keep the catalogue untouched; exercise an unsupported canonical
            # frame through the descriptor API and an explicit domain exception.
            def refuse(**kwargs):
                raise ValueError("Nondegenerate axis required")
            monkeypatch.setattr(m.morphology_context, "descriptor", refuse)
        return value
    monkeypatch.setattr(m.plant_variant_catalogue, "load_for_inspection", catalogue)
    result = m.extract_output(saved["pin"])
    assert result["records"] == []
    assert result["holds"][0]["reason"].startswith(reason)


def test_saved_mesh_reopened_not_cached_usd_layer(saved):
    path = saved["output"] / "Leaf.usda"
    cached = Usd.Stage.Open(str(path))  # keep an old layer alive
    old = np.asarray(UsdGeom.Mesh.Get(cached, "/Plant/Mesh").GetPointsAttr().Get()).copy()
    # A different file's authored bytes replace the fixture; no USD save/reload.
    other = saved["output"] / "other.usda"
    mesh(other, offset=.08)
    path.write_bytes(other.read_bytes())
    saved["receipt"]["output_hashes"][path.name] = m._hash(path)
    repin(saved)
    result = m.extract_output(saved["pin"])
    assert result["records"][1]["input_geometry"]["leaves"][0]["centroid"][1] == pytest.approx(.08 + .02 / 3)
    np.testing.assert_array_equal(UsdGeom.Mesh.Get(cached, "/Plant/Mesh").GetPointsAttr().Get(), old)


def test_frozen_reservations_and_source_code_verifier_cannot_be_bypassed(saved):
    saved["receipt"]["frozen_family_assignments"] = {**saved["receipt"]["frozen_family_assignments"], "T0": "test"}
    repin(saved)
    with pytest.raises(ValueError, match="Frozen source plan changed"):
        m.extract_output(saved["pin"])


def test_running_capture_and_inventory_modules_are_not_bound():
    assert "inventory.py" not in {Path(p).name for p in m._LOADED_CODE}
    assert "native_capture_v3" not in " ".join(m._LOADED_CODE)
