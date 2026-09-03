"""Prompts for visual tomato-petiole cut-point evaluation."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a visual perception evaluator for robotic tomato deleafing.
Analyze only visible evidence. Never invent hidden geometry. Return only the requested JSON object."""


DEFAULT_INSTRUCTION = (
    "Select the lowest clearly visible tomato leaf petiole that can be removed safely. "
    "The desired cut is close to its attachment to the main stem while leaving a 2-5 mm stub."
)


def build_cutpoint_prompt(*, image_width: int, image_height: int, view_name: str, instruction: str) -> str:
    """Build a coordinate-explicit prompt without simulator-private labels."""

    return f"""Task instruction:
{instruction}

You are viewing the {view_name!r} RGB camera image. Its original submitted size is exactly
{image_width} pixels wide by {image_height} pixels high. Pixel (0, 0) is the top-left.

Identify a tomato PETIOLE: the smaller stalk connecting a compound leaf/leaf branch to the thick main vertical vine.
Do not select a leaflet blade, leaflet vein, fruit truss, trellis string, robot part, or greenhouse structure.

For a safe cut:
- choose one intended petiole and place cut_point_px on its centreline near the main-stem attachment;
- leave a visually estimated 2-5 mm residual stub and never place the point inside the main stem;
- target_bbox_px must tightly enclose the selected petiole and its attachment region, and contain cut_point_px;
- cut_direction_px is an approximately unit-length image vector showing blade travel across the petiole's short axis;
- report visible hazards conservatively.

If the petiole-to-main-stem attachment is not visible enough to localize safely, return "uncertain". If the target is
visible but no collision-clear cut exists, return "no_safe_cut". Both decisions require null geometry. Do not guess.

Return exactly this JSON contract with every key present:
{{
  "schema_version": "greenhouse.vlm_cutpoint.v1",
  "decision": "cut" or "no_safe_cut" or "uncertain",
  "primary_view": "{view_name}",
  "target_bbox_px": {{"x_min": integer, "y_min": integer, "x_max": integer, "y_max": integer}} or null,
  "cut_point_px": {{"x": integer, "y": integer}} or null,
  "cut_direction_px": {{"dx": number, "dy": number}} or null,
  "confidence": number from 0.0 to 1.0,
  "hazards": [allowed hazard strings],
  "reason_codes": [at least one allowed reason string],
  "rationale": "visible evidence in under two sentences"
}}

Allowed hazards only:
- near_main_stem
- occluded_attachment
- overlapping_organs
- neighbouring_vine
- robot_or_tool_obstruction
- greenhouse_obstruction
- depth_ambiguous

Allowed reason_codes only:
- target_attachment_visible
- petiole_centerline_visible
- safe_clearance_visible
- target_not_visible
- attachment_occluded
- ambiguous_target
- no_safe_path
- insufficient_resolution

For "cut", all three geometry fields must be non-null and the point must lie inside the box. For either abstaining
decision, all three geometry fields must be null. Do not add keys or Markdown fences.
Keep rationale under two sentences and describe visible evidence only."""
