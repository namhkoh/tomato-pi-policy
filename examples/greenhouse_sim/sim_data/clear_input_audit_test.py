from copy import deepcopy
import json

from PIL import Image
import pytest

from .clear_input_audit import check_message_pair


def messages(crop=False):
    images = [dict(type="image", image=Image.new("RGB", (848,408), "green"))]
    if crop:
        images.append(dict(type="image", image=Image.new("RGB", (768,768), "green")))
    answer = dict(status="localized", cut_point_uv=[424.,204.],
                  visibility="clear", next_action="inspect_cut_region")
    train = [dict(role="system", content=[dict(type="text",text="system")]),
             dict(role="user",content=images+[dict(type="text",text="query only")]),
             dict(role="assistant",content=[dict(type="text",text=json.dumps(
                 dict(answer,cut_point_uv=[500.,500.])))])]
    return train, deepcopy(train[:2]), answer


@pytest.mark.parametrize("crop", [False, True])
def test_input_modes_and_coordinate_roundtrip(crop):
    train, inference, answer = messages(crop)
    assert check_message_pair(train, inference, answer, crop) == 0


@pytest.mark.parametrize("fault", ["answer_leak", "prompt", "pixels", "resolution", "answer", "semantics"])
def test_audit_rejects_contract_changes(fault):
    train, inference, answer = messages()
    if fault == "answer_leak": inference.append(train[-1])
    if fault == "prompt": inference[1]["content"][-1]["text"] = "different"
    if fault == "pixels": inference[1]["content"][0]["image"].putpixel((0,0),(255,0,0))
    if fault == "resolution": inference[1]["content"][0]["image"] = Image.new("RGB",(1696,816))
    if fault == "answer": train[-1]["content"][0]["text"] = json.dumps(dict(answer,cut_point_uv=[600.,500.]))
    if fault == "semantics": train[-1]["content"][0]["text"] = json.dumps(dict(answer,visibility="partial",cut_point_uv=[500.,500.]))
    with pytest.raises(ValueError):
        check_message_pair(train, inference, answer, False)
