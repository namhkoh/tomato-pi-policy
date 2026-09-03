"""Canonical, provider-neutral cut-point response schema."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any


DECISIONS = frozenset({"cut", "no_safe_cut", "uncertain"})
HAZARDS = frozenset(
    {
        "near_main_stem",
        "occluded_attachment",
        "overlapping_organs",
        "neighbouring_vine",
        "robot_or_tool_obstruction",
        "greenhouse_obstruction",
        "depth_ambiguous",
    }
)
REASON_CODES = frozenset(
    {
        "target_attachment_visible",
        "petiole_centerline_visible",
        "safe_clearance_visible",
        "target_not_visible",
        "attachment_occluded",
        "ambiguous_target",
        "no_safe_path",
        "insufficient_resolution",
    }
)


class PredictionValidationError(ValueError):
    """Raised when a model response violates the canonical contract."""


def response_json_schema() -> dict[str, Any]:
    """Return the JSON schema sent to compatible inference servers."""

    nullable_point = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }
    nullable_box = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                },
                "required": ["x_min", "y_min", "x_max", "y_max"],
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }
    nullable_direction = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"dx": {"type": "number"}, "dy": {"type": "number"}},
                "required": ["dx", "dy"],
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"const": "greenhouse.vlm_cutpoint.v1"},
            "decision": {"enum": sorted(DECISIONS)},
            "primary_view": {"type": "string", "minLength": 1},
            "target_bbox_px": nullable_box,
            "cut_point_px": nullable_point,
            "cut_direction_px": nullable_direction,
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "hazards": {"type": "array", "items": {"enum": sorted(HAZARDS)}, "uniqueItems": True},
            "reason_codes": {
                "type": "array",
                "items": {"enum": sorted(REASON_CODES)},
                "minItems": 1,
                "uniqueItems": True,
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": [
            "schema_version",
            "decision",
            "primary_view",
            "target_bbox_px",
            "cut_point_px",
            "cut_direction_px",
            "confidence",
            "hazards",
            "reason_codes",
            "rationale",
        ],
        "additionalProperties": False,
    }


@dataclasses.dataclass(frozen=True)
class PixelPoint:
    x: int
    y: int


@dataclasses.dataclass(frozen=True)
class PixelBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def contains(self, point: PixelPoint) -> bool:
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max


@dataclasses.dataclass(frozen=True)
class PixelDirection:
    dx: float
    dy: float


@dataclasses.dataclass(frozen=True)
class CutPointPrediction:
    """Validated model proposal in coordinates of the submitted image."""

    schema_version: str
    decision: str
    primary_view: str
    target_bbox_px: PixelBox | None
    cut_point_px: PixelPoint | None
    cut_direction_px: PixelDirection | None
    confidence: float
    hazards: tuple[str, ...]
    reason_codes: tuple[str, ...]
    rationale: str

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, image_width: int, image_height: int) -> CutPointPrediction:
        """Parse and validate one provider response."""

        if not isinstance(value, dict):
            raise PredictionValidationError("prediction must be a JSON object")
        expected = set(response_json_schema()["required"])
        missing = expected.difference(value)
        extra = set(value).difference(expected)
        if missing or extra:
            raise PredictionValidationError(f"schema keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
        if value["schema_version"] != "greenhouse.vlm_cutpoint.v1":
            raise PredictionValidationError("unsupported schema_version")
        decision = _enum(value["decision"], DECISIONS, "decision")
        primary_view = _nonempty_string(value["primary_view"], "primary_view")
        confidence = _finite_number(value["confidence"], "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise PredictionValidationError("confidence must be in [0, 1]")
        hazards = _enum_list(value["hazards"], HAZARDS, "hazards", allow_empty=True)
        reason_codes = _enum_list(value["reason_codes"], REASON_CODES, "reason_codes", allow_empty=False)
        rationale = _nonempty_string(value["rationale"], "rationale")
        if len(rationale) > 500:
            raise PredictionValidationError("rationale exceeds 500 characters")

        point = _point(value["cut_point_px"], image_width=image_width, image_height=image_height)
        box = _box(value["target_bbox_px"], image_width=image_width, image_height=image_height)
        direction = _direction(value["cut_direction_px"])
        if decision == "cut":
            if point is None or box is None or direction is None:
                raise PredictionValidationError("cut decisions require a point, target box, and direction")
            if not box.contains(point):
                raise PredictionValidationError("cut point must lie inside target_bbox_px")
        elif point is not None or box is not None or direction is not None:
            raise PredictionValidationError("abstaining decisions must use null geometry")

        return cls(
            schema_version="greenhouse.vlm_cutpoint.v1",
            decision=decision,
            primary_view=primary_view,
            target_bbox_px=box,
            cut_point_px=point,
            cut_direction_px=direction,
            confidence=confidence,
            hazards=hazards,
            reason_codes=reason_codes,
            rationale=rationale,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract a JSON object from plain, fenced, or reasoning-prefixed text."""

    if not isinstance(content, str) or not content.strip():
        raise PredictionValidationError("provider returned empty text")
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 : -3].strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "decision" in value:
            return value
    raise PredictionValidationError("provider text does not contain a prediction JSON object")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _enum(value: Any, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise PredictionValidationError(f"{field} must be one of {sorted(allowed)}")
    return value


def _enum_list(value: Any, allowed: frozenset[str], field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise PredictionValidationError(f"{field} must be a {'possibly empty' if allow_empty else 'non-empty'} list")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise PredictionValidationError(f"{field} contains an unsupported value")
    if len(set(value)) != len(value):
        raise PredictionValidationError(f"{field} must not contain duplicates")
    return tuple(value)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise PredictionValidationError(f"{field} must be a finite number")
    return float(value)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictionValidationError(f"{field} must be an integer")
    return value


def _point(value: Any, *, image_width: int, image_height: int) -> PixelPoint | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        raise PredictionValidationError("cut_point_px must contain exactly x and y")
    point = PixelPoint(_integer(value["x"], "cut_point_px.x"), _integer(value["y"], "cut_point_px.y"))
    if not 0 <= point.x < image_width or not 0 <= point.y < image_height:
        raise PredictionValidationError("cut_point_px lies outside the submitted image")
    return point


def _box(value: Any, *, image_width: int, image_height: int) -> PixelBox | None:
    if value is None:
        return None
    keys = {"x_min", "y_min", "x_max", "y_max"}
    if not isinstance(value, dict) or set(value) != keys:
        raise PredictionValidationError("target_bbox_px has invalid keys")
    box = PixelBox(**{key: _integer(value[key], f"target_bbox_px.{key}") for key in keys})
    if not (0 <= box.x_min < box.x_max < image_width and 0 <= box.y_min < box.y_max < image_height):
        raise PredictionValidationError("target_bbox_px is invalid or outside the submitted image")
    return box


def _direction(value: Any) -> PixelDirection | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"dx", "dy"}:
        raise PredictionValidationError("cut_direction_px must contain exactly dx and dy")
    dx = _finite_number(value["dx"], "cut_direction_px.dx")
    dy = _finite_number(value["dy"], "cut_direction_px.dy")
    norm = math.hypot(dx, dy)
    if norm < 0.5 or norm > 1.5:
        raise PredictionValidationError("cut_direction_px must be approximately unit length")
    return PixelDirection(dx=dx / norm, dy=dy / norm)
