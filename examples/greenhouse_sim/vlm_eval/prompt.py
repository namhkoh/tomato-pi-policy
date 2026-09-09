"""Prompts for visual tomato-petiole cut-point evaluation."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a visual perception evaluator for robotic tomato deleafing.
Use visible plant topology and local image geometry to identify a cut candidate. Never invent hidden geometry.
Return only the requested JSON object."""


DEFAULT_INSTRUCTION = (
    "Select the lowest safely cuttable tomato leaf petiole, where lowest means the visible attachment point with "
    "the largest image y-coordinate. Place the cut just outside the main stem while leaving a 2-5 mm stub."
)


def build_cutpoint_prompt(*, image_width: int, image_height: int, view_name: str, instruction: str) -> str:
    """Build a coordinate-explicit prompt without simulator-private labels."""

    return f"""Task instruction:
{instruction}

You are viewing the {view_name!r} RGB camera image. Its original submitted size is exactly
{image_width} pixels wide by {image_height} pixels high. Pixel (0, 0) is the top-left.
All coordinate fields must use RAW PIXELS in this exact frame: x must be 0-{image_width - 1} and y must be
0-{image_height - 1}. Do not use normalized 0-1000 coordinates or coordinates from an internally resized image.

Use this visual procedure before producing the JSON:
1. Trace the thickest continuous green MAIN STEM through the image. It may be vertical, diagonal, leaning, or curved.
2. Find thinner lateral stalks that branch from it and support multiple leaflets. Those stalks are PETIOLES.
3. Trace each candidate petiole back to its Y-shaped attachment on the visible edge of the main stem.
4. Discard candidates whose attachment itself is hidden. Rank the remaining safe candidates by attachment-point y;
   the largest y is the lowest candidate in this image, regardless of where its distal leaflets appear.
5. Check a short blade corridor through the petiole at its base for other plant organs, greenhouse structure, and robot parts.

Visual consistency checks:
- apply the largest-y rule only after confirming an anatomically valid petiole attachment; do not choose an image-bottom
  location merely because it has a large y-coordinate;
- cut_point_px must visibly land on green basal petiole tissue, never on a leaf blade, empty background, floor, or gutter;
- target_bbox_px must visibly cover that same green basal petiole and its Y-junction. If these checks cannot be satisfied,
  abstain instead of emitting geometry.

Do not select a leaflet blade, leaflet vein, distal leaflet stalk, fruit truss, trellis string, robot part, or greenhouse
structure. Do not assume the most vertical green object is the main stem.

For a safe cut:
- choose one intended petiole and place cut_point_px on its centreline near the main-stem attachment;
- place it just outside the visible main-stem boundary, approximately one petiole width along the petiole, to represent
  a visually estimated 2-5 mm residual stub; never place it inside the main stem;
- target_bbox_px is a LOCAL box around the Y-shaped attachment and short basal petiole segment, not the whole leaf;
  it must contain cut_point_px;
- estimate the basal petiole centreline tangent, then set cut_direction_px to an approximately unit-length image vector
  perpendicular to that tangent, showing blade travel across the petiole's short axis;
- report visible hazards conservatively.

Occlusion of distal leaves alone is not a reason to abstain when the selected basal attachment and local cut corridor are
clearly visible. If no petiole-to-main-stem attachment is visible enough to localize, return "uncertain". If an attachment
is visible but every candidate lacks a collision-clear cut corridor, return "no_safe_cut". Both abstaining decisions
require null geometry. Do not guess through an occluded attachment.

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
Before returning a cut, verify every x and y value is inside the raw-pixel ranges stated above.
Also verify the proposed point and box visibly overlap the selected green petiole base rather than background.
Keep rationale under two sentences and describe visible evidence only."""
