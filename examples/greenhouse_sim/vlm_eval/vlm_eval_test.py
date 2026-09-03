from __future__ import annotations

import json

import pytest
from PIL import Image

from .openai_compatible import EndpointError
from .openai_compatible import OpenAICompatibleVisionClient
from .openai_compatible import build_chat_payload
from .openai_compatible import encode_image
from .overlay import render_prediction_overlay
from .prompt import build_cutpoint_prompt
from .schema import CutPointPrediction
from .schema import PredictionValidationError
from .schema import extract_json_object


def _cut_prediction() -> dict:
    return {
        "schema_version": "greenhouse.vlm_cutpoint.v1",
        "decision": "cut",
        "primary_view": "left_wrist",
        "target_bbox_px": {"x_min": 20, "y_min": 10, "x_max": 80, "y_max": 70},
        "cut_point_px": {"x": 31, "y": 42},
        "cut_direction_px": {"dx": 0.8, "dy": -0.6},
        "confidence": 0.72,
        "hazards": ["near_main_stem"],
        "reason_codes": ["target_attachment_visible"],
        "rationale": "A visible petiole joins the main stem here.",
    }


def test_fenced_reasoning_response_is_extracted_and_validated() -> None:
    raw = f"Visible evidence considered.\n```json\n{json.dumps(_cut_prediction())}\n```"
    prediction = CutPointPrediction.from_dict(extract_json_object(raw), image_width=100, image_height=80)
    assert prediction.decision == "cut"
    assert prediction.cut_point_px is not None
    assert prediction.cut_point_px.x == 31


def test_cut_point_must_be_inside_target_box() -> None:
    value = _cut_prediction()
    value["cut_point_px"] = {"x": 90, "y": 42}
    with pytest.raises(PredictionValidationError, match="inside"):
        CutPointPrediction.from_dict(value, image_width=100, image_height=80)


def test_abstention_rejects_geometry() -> None:
    value = _cut_prediction()
    value["decision"] = "uncertain"
    with pytest.raises(PredictionValidationError, match="null geometry"):
        CutPointPrediction.from_dict(value, image_width=100, image_height=80)


def test_structured_payload_contains_only_public_image_and_prompt() -> None:
    payload = build_chat_payload(
        model="qwen3-vl-32b-instruct",
        prompt="public instruction",
        image_data_url="data:image/png;base64,AA==",
        max_tokens=256,
        temperature=0.0,
        structured_mode="json_schema",
    )
    serialized = json.dumps(payload)
    assert "greenhouse_vlm_cutpoint_v1" in serialized
    assert "public instruction" in serialized
    assert "SubStem" not in serialized
    assert "api_key" not in serialized


def test_encode_and_overlay_round_trip(tmp_path) -> None:
    image_path = tmp_path / "camera.png"
    Image.new("RGB", (100, 80), color=(20, 80, 20)).save(image_path)
    encoded = encode_image(image_path)
    prediction = CutPointPrediction.from_dict(_cut_prediction(), image_width=100, image_height=80)
    overlay_path = render_prediction_overlay(image_path, prediction, tmp_path / "overlay.png")
    assert encoded.width == 100
    assert encoded.height == 80
    assert encoded.data_url.startswith("data:image/jpeg;base64,")
    assert encoded.transcoded
    assert encoded.sha256 != encoded.submitted_sha256
    assert overlay_path.is_file()
    with Image.open(overlay_path) as overlay:
        assert overlay.size == (100, 80)


def test_prompt_defines_pixel_frame_and_abstention() -> None:
    prompt = build_cutpoint_prompt(
        image_width=1280,
        image_height=720,
        view_name="head",
        instruction="Select one safe lower-leaf cut.",
    )
    assert "1280 pixels wide by 720 pixels high" in prompt
    assert "top-left" in prompt
    assert 'return "uncertain"' in prompt
    for field in (
        "schema_version", "decision", "primary_view", "target_bbox_px", "cut_point_px", "cut_direction_px",
        "confidence", "hazards", "reason_codes", "rationale",
    ):
        assert f'"{field}"' in prompt


def test_auto_mode_falls_back_after_vllm_schema_internal_error(monkeypatch) -> None:
    client = OpenAICompatibleVisionClient(
        base_url="https://inference.invalid/v1",
        api_key="test-only",
    )
    attempted_modes = []

    def fake_post(payload):
        response_format = payload.get("response_format", {})
        mode = response_format.get("type", "none")
        attempted_modes.append(mode)
        if mode == "json_schema":
            raise EndpointError(
                "schema unsupported",
                status=500,
                response_body='{"error":{"type":"InternalServerError"}}',
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(_cut_prediction()),
                    }
                }
            ]
        }, 0.25

    monkeypatch.setattr(client, "_post", fake_post)
    result = client.infer(
        model="qwen3-vl-32b-instruct",
        prompt="test",
        image_data_url="data:image/jpeg;base64,AA==",
        structured_mode="auto",
    )
    assert attempted_modes == ["json_schema", "json_object"]
    assert result.structured_mode == "json_object"
