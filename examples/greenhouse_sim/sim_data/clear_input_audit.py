"""Read-only clear-cutpoint input audit; no model, training or release approval."""
import argparse
import json
from pathlib import Path

from PIL import ImageChops

from .dataset_review import read_json, write_json, require, safe_file
from .depth_preview import sha256
from .clear_cutpoint_release import validate
from .qwen_adapter import model_messages
from .qwen_coordinates import answer_to_pixels
from .training_export import read_jsonl


def check_message_pair(training, inference, canonical, query_crop):
    require([m["role"] for m in training] == ["system", "user", "assistant"]
            and [m["role"] for m in inference] == ["system", "user"], "Unexpected roles")
    require(training[0] == inference[0], "Training/inference system prompts differ")
    a, b = training[1]["content"], inference[1]["content"]
    expected = ["image", "image", "text"] if query_crop else ["image", "text"]
    require([v["type"] for v in a] == [v["type"] for v in b] == expected, "Unexpected input modalities")
    require(a[-1] == b[-1], "Training/inference user prompts differ")
    for index, size in enumerate([(848, 408), (768, 768)] if query_crop else [(848, 408)]):
        first, second = a[index]["image"], b[index]["image"]
        require(first.mode == second.mode == "RGB" and first.size == second.size == size,
                "Unexpected full-frame/crop image")
        require(ImageChops.difference(first, second).getbbox() is None, "Training/inference RGB differs")
    decoded = answer_to_pixels(json.loads(training[-1]["content"][0]["text"]))
    require(all(decoded[k] == canonical[k] for k in ("status", "visibility", "next_action")),
            "Coordinate adapter changed answer semantics")
    require(canonical["status"] == "localized", "Clear task must have a visible localization label")
    error = max(abs(a-b) for a,b in zip(decoded["cut_point_uv"], canonical["cut_point_uv"]))
    require(error <= .004240001, "Two-decimal normalized coordinate roundtrip exceeded tolerance")
    return error


def audit(root):
    root = Path(root).resolve()
    checked = validate(root, allow_draft=True)
    manifest = read_json(root/"manifest.json")
    require(manifest["release_profile"] == "clear_cutpoint_v1", "Explicit clear task required")
    manifest_hash = sha256(root/"manifest.json")
    counts, seen, maximum_error = {}, set(), 0.
    for split in ("train", "validation", "test"):
        rows = list(read_jsonl(safe_file(root, f"splits/{split}.jsonl")))
        counts[split] = len(rows)
        for row in rows:
            require(row["id"] not in seen, "Duplicate ID across splits")
            seen.add(row["id"])
            canonical = json.loads(row["messages"][-1]["content"])
            for crop in (False, True):
                training = inference = None
                try:
                    training = model_messages(row, root, include_answer=True,
                        coordinates="normalized_1000", decimals=2, query_crop=crop)
                    inference = model_messages(row, root, include_answer=False,
                        coordinates="normalized_1000", decimals=2, query_crop=crop)
                    maximum_error = max(maximum_error, check_message_pair(training, inference, canonical, crop))
                finally:
                    for messages in (training, inference):
                        if messages:
                            for item in messages[1]["content"]:
                                if item["type"] == "image":
                                    item["image"].close()
        require(counts[split] == manifest["acceptance"]["counts"][split]["rows"], "Split count differs")
    require(sha256(root/"manifest.json") == manifest_hash, "Manifest changed during audit")
    return dict(state="clear_input_format_checked_not_model_or_training_approval",
        source_manifest_sha256=manifest_hash, source_release_state=manifest["state"],
        counts=counts, input_modes=["original_rgb", "original_rgb_plus_query_crop"],
        coordinates="normalized_1000_two_decimals", maximum_roundtrip_error_px=maximum_error,
        matching_training_inference_prompts_and_images=True,
        model_processor_executed=False, model_weights_loaded=False, training_started=False,
        training_approved=False, native_depth_used_as_model_input=False,
        depth_provenance="checked_by_clear_release_validator_not_recomputed_here",
        normal_training_release_validator_still_required=True,
        implementation_sha256={name:sha256(Path(__file__).with_name(name))
            for name in ("clear_input_audit.py", "qwen_adapter.py", "qwen_coordinates.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.dataset.resolve()),
            "New diagnostic output outside immutable dataset required")
    result = audit(args.dataset)
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
