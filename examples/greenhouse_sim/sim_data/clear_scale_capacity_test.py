from copy import deepcopy
import pytest
from .clear_scale_capacity import inventory, assess, report, SCENARIOS


def plan():
    families = {"train_a": "train", "val_a": "validation", "test_a": "test"}
    result = dict(schema_version="greenhouse.grounding_collection_plan.v1",
        state="ready_for_synthetic_grounding_capture", training_dataset_approved=False,
        configuration=dict(clear_capture="robot_head_close_diffuse_v1", source_geometry="unmodified_native_components"),
        family_assignments=families, jobs=[], selection_audit=[])
    for i, (family, split) in enumerate(families.items()):
        targets = [dict(target_id=f"{family}/p{j}", source_plant_id=family, split_group=family) for j in range(2)]
        result["jobs"].append(dict(job_id=f"job_{i}", plant_family=family, split=split, targets=targets))
        result["selection_audit"].append(dict(family=family, geometry_candidates=3, selected=2,
            exclusions=[dict(target_id=f"{family}/p{j}", state="geometry_candidate") for j in range(3)]))
    return result


def test_identity_capacity_not_render_attempts_or_resolution():
    p = plan()
    p["configuration"].update(view_offset=1000, render_views_per_target=160, resolution=[1696, 816])
    actual = inventory(p)
    assert actual["scheduled_targets"] == 6 and actual["recorded_geometry_candidates"] == 9
    assert actual["scheduled_image_upper_bound"] == 72 and actual["all_geometry_image_upper_bound"] == 108
    assert not actual["repeated_shards_increase_target_capacity"]
    assert not actual["resolution_increases_source_target_capacity"]


def test_20k_training_requires_more_than_one_thousand_six_hundred_targets():
    actual = assess(inventory(plan()), SCENARIOS["20000_train_plus_heldout_provisional"])
    assert actual["per_split"]["train"]["minimum_distinct_targets_at_view_cap"] == 1667
    assert actual["minimum_distinct_targets"] == 2001
    assert actual["requested_total_images"] == 24000
    assert not actual["all_recorded_geometry_capacity_sufficient"]
    assert actual["collection_eta"] is None and not actual["authorizes_collection"]


def test_capacity_pass_still_does_not_authorize_unverified_capture():
    actual = assess(inventory(plan()), {"train": 24, "validation": 24, "test": 24})
    assert actual["scheduled_capacity_sufficient"]
    assert not actual["full_capture_feasibility_established"]
    assert not actual["authorizes_training"]


def test_report_records_confirmed_training_only_minimum(tmp_path):
    from .dataset_review import write_json
    path=tmp_path/'plan.json'
    write_json(path,plan())
    result=report(path)
    assert result['user_minimum_training_images']==20000
    assert result['whether_20k_means_train_or_all_splits']=='training_only_user_confirmed'
    assert result['active_scenario']=='20000_train_plus_heldout_provisional'
    assert not result['training_ready']


@pytest.mark.parametrize("fault", ["split", "lineage", "duplicate_target", "duplicate_family",
    "duplicate_audit", "inflated_count", "bad_selected_count", "missing_geometry", "unknown_state", "approved"])
def test_malformed_or_inflated_inventory_rejected(fault):
    p = plan()
    if fault == "split": p["jobs"][0]["split"] = "test"
    if fault == "lineage": p["jobs"][0]["targets"][0]["split_group"] = "val_a"
    if fault == "duplicate_target": p["jobs"][0]["targets"][1] = deepcopy(p["jobs"][0]["targets"][0])
    if fault == "duplicate_family": p["jobs"].append(deepcopy(p["jobs"][0]))
    if fault == "duplicate_audit": p["selection_audit"].append(deepcopy(p["selection_audit"][0]))
    if fault == "inflated_count": p["selection_audit"][0]["geometry_candidates"] = 99999
    if fault == "bad_selected_count": p["selection_audit"][0]["selected"] = 8
    if fault == "missing_geometry":
        p["selection_audit"][0]["exclusions"][0]["state"] = "excluded"
        p["selection_audit"][0]["geometry_candidates"] -= 1
    if fault == "unknown_state": p["selection_audit"][0]["exclusions"][0]["state"] = "automatically_approved"
    if fault == "approved": p["training_dataset_approved"] = True
    with pytest.raises(ValueError): inventory(p)


@pytest.mark.parametrize("cap", [24, 160, True, 12.0])
def test_cannot_increase_cap_to_make_capacity_pass(cap):
    with pytest.raises(ValueError): inventory(plan(), cap)


@pytest.mark.parametrize("goals", [{"train": 20000}, {"train": True, "validation": 1, "test": 1},
    {"train": 20000, "validation": 0, "test": 1}])
def test_invalid_goals_rejected(goals):
    with pytest.raises(ValueError): assess(inventory(plan()), goals)
