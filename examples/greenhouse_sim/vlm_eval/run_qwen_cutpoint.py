"""Run one Qwen VLM cut-point trial and create inspectable artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from vlm_eval.openai_compatible import EndpointError
    from vlm_eval.openai_compatible import OpenAICompatibleVisionClient
    from vlm_eval.openai_compatible import encode_image
    from vlm_eval.overlay import render_prediction_overlay
    from vlm_eval.prompt import DEFAULT_INSTRUCTION
    from vlm_eval.prompt import SYSTEM_PROMPT
    from vlm_eval.prompt import build_cutpoint_prompt
    from vlm_eval.schema import CutPointPrediction
    from vlm_eval.schema import PredictionValidationError
    from vlm_eval.schema import extract_json_object
else:
    from .openai_compatible import EndpointError
    from .openai_compatible import OpenAICompatibleVisionClient
    from .openai_compatible import encode_image
    from .overlay import render_prediction_overlay
    from .prompt import DEFAULT_INSTRUCTION
    from .prompt import SYSTEM_PROMPT
    from .prompt import build_cutpoint_prompt
    from .schema import CutPointPrediction
    from .schema import PredictionValidationError
    from .schema import extract_json_object


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=pathlib.Path, help="Unannotated RGB camera frame")
    parser.add_argument("--view-name", default="left_wrist")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL"))
    parser.add_argument("--model", default="qwen3-vl-32b-instruct")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--api-key-file", type=pathlib.Path)
    parser.add_argument("--output-directory", type=pathlib.Path)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--submission-format", choices=("jpeg", "original"), default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--structured-mode",
        choices=("auto", "json_schema", "json_object", "none"),
        default="auto",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and write metadata without network access",
    )
    return parser


def _read_api_key(args: argparse.Namespace) -> str:
    if args.api_key_file is not None:
        key = args.api_key_file.expanduser().read_text(encoding="utf-8").strip()
    else:
        key = os.getenv(args.api_key_env, "").strip()
    if not key:
        raise ValueError(
            f"API key unavailable; set {args.api_key_env} or pass --api-key-file pointing to an ignored local file"
        )
    return key


def _default_output_directory() -> pathlib.Path:
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    root = pathlib.Path(__file__).resolve().parents[3]
    return root / "data" / "greenhouse_sim" / "vlm_eval" / "runs" / f"qwen_single_{timestamp}"


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    if not args.base_url:
        raise SystemExit("--base-url or QWEN_BASE_URL is required")
    if args.max_tokens <= 0 or args.timeout_s <= 0.0 or args.temperature < 0.0:
        raise SystemExit("max-tokens and timeout must be positive; temperature must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("jpeg-quality must be in [1, 100]")

    encoded = encode_image(args.image, submission_format=args.submission_format, jpeg_quality=args.jpeg_quality)
    prompt = build_cutpoint_prompt(
        image_width=encoded.width,
        image_height=encoded.height,
        view_name=args.view_name,
        instruction=args.instruction,
    )
    output_directory = (args.output_directory or _default_output_directory()).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "greenhouse.vlm_trial.v1",
        "provider": "openai_compatible",
        "base_url": args.base_url,
        "model": args.model,
        "view_name": args.view_name,
        "instruction": args.instruction,
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt,
        "image": {
            "path": str(encoded.path),
            "width": encoded.width,
            "height": encoded.height,
            "source_sha256": encoded.sha256,
            "submitted_sha256": encoded.submitted_sha256,
            "submitted_mime_type": encoded.submitted_mime_type,
            "transcoded": encoded.transcoded,
        },
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "requested_structured_mode": args.structured_mode,
            "submission_format": args.submission_format,
            "jpeg_quality": args.jpeg_quality,
        },
    }
    _write_json(output_directory / "request.json", metadata)
    if args.dry_run:
        summary = {"status": "dry_run", "output_directory": str(output_directory), **metadata["image"]}
        print(json.dumps(summary, indent=2))
        return 0

    try:
        client = OpenAICompatibleVisionClient(
            base_url=args.base_url,
            api_key=_read_api_key(args),
            timeout_s=args.timeout_s,
        )
        result = client.infer(
            model=args.model,
            prompt=prompt,
            image_data_url=encoded.data_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            structured_mode=args.structured_mode,
        )
        _write_json(output_directory / "raw_response.json", result.response)
        parsed = extract_json_object(result.content)
        prediction = CutPointPrediction.from_dict(
            parsed,
            image_width=encoded.width,
            image_height=encoded.height,
        )
    except (EndpointError, PredictionValidationError, ValueError) as exc:
        error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "http_status": getattr(exc, "status", None),
            "response_body": getattr(exc, "response_body", ""),
        }
        _write_json(output_directory / "error.json", error)
        print(json.dumps(error, indent=2), file=sys.stderr)
        return 2

    prediction_payload = {
        **prediction.to_dict(),
        "latency_s": result.latency_s,
        "structured_mode": result.structured_mode,
        "usage": result.response.get("usage"),
    }
    _write_json(output_directory / "prediction.json", prediction_payload)
    overlay_path = render_prediction_overlay(encoded.path, prediction, output_directory / "overlay.png")
    print(
        json.dumps(
            {
                "status": "ok",
                "prediction": prediction.to_dict(),
                "latency_s": result.latency_s,
                "structured_mode": result.structured_mode,
                "overlay": str(overlay_path),
                "output_directory": str(output_directory),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
