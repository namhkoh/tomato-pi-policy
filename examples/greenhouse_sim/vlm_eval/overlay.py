"""Human-review overlays for VLM cut-point predictions."""

from __future__ import annotations

import math
import pathlib

from .schema import CutPointPrediction


def render_prediction_overlay(
    image_path: str | pathlib.Path,
    prediction: CutPointPrediction,
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """Draw the selected petiole, cut point, direction, and decision."""

    from PIL import Image
    from PIL import ImageDraw

    source = pathlib.Path(image_path)
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, round(min(width, height) / 180))
    text = f"Qwen: {prediction.decision}  confidence={prediction.confidence:.2f}"
    text_box = draw.textbbox((0, 0), text)
    draw.rectangle((4, 4, text_box[2] + 12, text_box[3] + 12), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))

    if prediction.target_bbox_px is not None:
        box = prediction.target_bbox_px
        draw.rectangle((box.x_min, box.y_min, box.x_max, box.y_max), outline=(255, 80, 80), width=line_width)
    if prediction.cut_point_px is not None:
        point = prediction.cut_point_px
        radius = max(5, round(min(width, height) / 45))
        draw.ellipse(
            (point.x - radius, point.y - radius, point.x + radius, point.y + radius),
            outline=(0, 255, 255),
            width=line_width,
        )
        draw.line((point.x - radius, point.y, point.x + radius, point.y), fill=(0, 255, 255), width=line_width)
        draw.line((point.x, point.y - radius, point.x, point.y + radius), fill=(0, 255, 255), width=line_width)
        if prediction.cut_direction_px is not None:
            direction = prediction.cut_direction_px
            length = max(24, round(min(width, height) / 6))
            end_x = point.x + direction.dx * length
            end_y = point.y + direction.dy * length
            draw.line((point.x, point.y, end_x, end_y), fill=(255, 230, 0), width=line_width)
            angle = math.atan2(direction.dy, direction.dx)
            for offset in (-2.55, 2.55):
                head_x = end_x + math.cos(angle + offset) * length * 0.22
                head_y = end_y + math.sin(angle + offset) * length * 0.22
                draw.line((end_x, end_y, head_x, head_y), fill=(255, 230, 0), width=line_width)

    image.save(output)
    return output
