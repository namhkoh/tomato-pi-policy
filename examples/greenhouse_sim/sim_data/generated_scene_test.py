"""USD-only substitution tests; no Isaac app, physics or rendering."""
from copy import deepcopy

import numpy as np
import pytest

from .generated_capture import substitute_plant
from . import plant_variant_catalogue_test as fixture_module
from .plant_variant_catalogue import load_for_inspection

Usd = fixture_module.Usd
pytestmark = pytest.mark.skipif(Usd is None, reason="USD unavailable")


@pytest.fixture
def generated():
    case = fixture_module.CatalogueTests()
    case.setUp()
    try:
        yield case, load_for_inspection(case.directory, case.plan_path)
    finally:
        case.doCleanups()


def original_stage(c):
    from pxr import Gf, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set((2, 3, 4))
    parent = UsdGeom.Xform.Define(stage, "/World/PackPlants")
    parent.AddRotateZOp().Set(37)
    old = UsdGeom.Xform.Define(stage, "/World/PackPlants/Original")
    old.AddTranslateOp().Set((.195, 0, .9))
    old.AddRotateXOp().Set(11)
    sibling = UsdGeom.Cube.Define(stage, "/World/Neighbor")
    sibling.AddTranslateOp().Set((1, 2, 3))
    original_variant = dict(variant_id=c["source_family"], source_plant_id=c["source_family"],
                           split_group=c["split_group"], plant_root=str(old.GetPath()))
    paths = {key: str(old.GetPath())+"/"+key for key in c["report"]["components"]}
    records = [dict(plant_root=str(old.GetPath()), component_paths=paths),
               dict(plant_root="/World/OtherPlant", component_paths={"other": "/World/OtherPlant/other"})]
    variants = [original_variant, dict(variant_id="Other", plant_root="/World/OtherPlant")]
    return stage, original_variant, records, variants


def test_substitution_preserves_world_placement_and_surroundings(generated):
    from pxr import UsdGeom
    case, c = generated
    stage, old, records, variants = original_stage(c)
    root_before = stage.GetRootLayer().ExportToString()
    records_before, variants_before = deepcopy(records), deepcopy(variants)
    cache = UsdGeom.XformCache()
    before = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(old["plant_root"]))
    neighbor = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Neighbor"))
    result = substitute_plant(stage, old, records, variants, c)
    after = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(result["new_root"]))
    np.testing.assert_allclose(after, before, atol=1e-10)
    np.testing.assert_allclose(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Neighbor")),
                               neighbor, atol=1e-10)
    assert not stage.GetPrimAtPath(old["plant_root"]).IsActive()
    assert stage.GetPrimAtPath(result["new_root"]).IsActive()
    assert set(result["records"][0]["component_paths"]) == set(records[0]["component_paths"])
    assert result["records"][1] == records[1]
    assert records == records_before and variants == variants_before
    assert stage.GetRootLayer().ExportToString() == root_before
    assert result["component_count_preserved"] and not result["source_files_changed"]


@pytest.mark.parametrize("fault", ["wrong_family", "duplicate_record", "duplicate_variant", "missing", "existing_generated"])
def test_ambiguous_or_repeated_substitution_is_rejected(generated, fault):
    from pxr import UsdGeom
    _, c = generated
    stage, old, records, variants = original_stage(c)
    if fault == "wrong_family": old["source_plant_id"] = "Other"
    if fault == "duplicate_record": records.append(deepcopy(records[0]))
    if fault == "duplicate_variant": variants.append(deepcopy(old))
    if fault == "missing": stage.GetPrimAtPath(old["plant_root"]).SetActive(False)
    if fault == "existing_generated":
        UsdGeom.Xform.Define(stage, "/World/GeneratedNativePilot/"+c["variant_id"])
    with pytest.raises(ValueError):
        substitute_plant(stage, old, records, variants, c)
