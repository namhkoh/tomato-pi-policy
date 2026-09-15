"""Versioned local contracts. No renderer, writes, or release admission on import."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "greenhouse.original_native_pose_prior_plan.v1"
SAMPLE_SCHEMA = "greenhouse.original_native_sample.v1"
RESULT_SCHEMA = "greenhouse.original_native_result.v1"
AUDIT_SCHEMA = "greenhouse.original_native_automatic_audit.v1"
HEAD_CAMERA = "/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera"
RESOLUTION = [1696, 816]
FROZEN_SPLITS = {"seed" + str(seed) + "_full": split for split, seeds in (
    ("train", (101, 103, 11, 17, 19, 23, 41, 43, 47, 53, 67, 71, 73, 7, 83, 89)),
    ("validation", (13, 29, 37, 97)), ("test", (31, 59, 61, 79))) for seed in seeds}
SCENE_POLICY = dict(profile="robot_head_close_diffuse_v1", day=172, minutes=780,
    intensity=1500, dome_intensity=6000, renderer="RealTimePathTracing",
    source_geometry="unmodified_native_components", render_subframes=56,
    instance_backend="legacy", physical_motion_commanded=False,
    articulated_revolute_joint_count=22, continuous_wheel_degrees=0, prismatic_gripper_m=0)
ADMISSION = dict(training_approved=False, source_cap_reset=False,
    biological_family_credit=0, historical_training_rows=0,
    global_caps_and_duplicate_checks_required=True,
    automatic_review_scope="fresh_native_annotation_only_not_global_release",
    human_review_performed=False, native_depth_reconstructed=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= 64 * 1024 * 1024, "JSON exceeds bounded metadata size")
    return json.loads(path.read_bytes(), object_pairs_hook=_pairs,
                      parse_constant=lambda value: _invalid_constant(value))


def _invalid_constant(value):
    raise ValueError("Non-finite JSON: " + value)


def pin(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Explicit lowercase SHA256 required")
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha256(path) == expected, "Changed or missing pinned file: " + str(path))
    return path


def bind_all(bindings):
    require(isinstance(bindings, dict) and bindings, "Nonempty source bindings required")
    for path, expected in bindings.items():
        require(Path(path).is_absolute() and str(Path(path).resolve()) == path, "Noncanonical bound path")
        pin(path, expected)


def merge_bindings(*groups):
    result = {}
    for group in groups:
        require(isinstance(group, dict), "Bindings must be a map")
        for path, expected in group.items():
            path = str(Path(path).resolve())
            require(path not in result or result[path] == expected, "Conflicting source binding")
            result[path] = expected
    return result


def safe_file(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
            "Relative POSIX file name required")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Source path traversal")
    root = Path(root).resolve()
    path = (root / rel).resolve()
    require(path.is_relative_to(root), "Resolved file escapes source")
    return path


def new_destination(path, protected):
    path = Path(path).resolve()
    require(path != Path(path.anchor) and not path.exists(), "New, non-root destination required")
    for source in protected:
        source = Path(source).resolve()
        require(not path.is_relative_to(source) and not source.is_relative_to(path),
                "Destination overlaps immutable source: " + str(source))
    return path


def write_new(path, value):
    """Exclusive JSON publication. Never overwrite a receipt."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def policy():
    return deepcopy(dict(scene=SCENE_POLICY, admission=ADMISSION, resolution=RESOLUTION,
                         family_assignments=FROZEN_SPLITS))


def code_bindings():
    """Bind local code and the repository-local import closure without importing Kit.

    Includes literal imports inside functions; external Python/Isaac dependencies
    are environmental prerequisites, not authenticated by this source receipt.
    """
    import ast
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    examples = root.parents[1]
    search = (root.parent, examples)
    pending = [Path(__file__).with_name(name) for name in
               ("__init__.py", "contracts.py", "prepare.py", "scene.py", "collector.py", "audit.py")]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        raw = path.read_bytes()
        found[str(path)] = digest(raw)
        base = next((p for p in search if path.is_relative_to(p)), None)
        if base is None:
            continue
        package = ".".join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                module = importlib.util.resolve_name(name, package) if node.level else name
                names = [module] + [module + "." + a.name for a in node.names if a.name != "*"]
            for name in names:
                for folder in search:
                    candidate = folder.joinpath(*name.split("."))
                    for target in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if target.is_file():
                            pending.append(target)
    return dict(sorted(found.items()))


# Pin the code associated with this Python process, not a later on-disk revision.
LOADED_IMPLEMENTATION = code_bindings()


def verify_loaded_code():
    bind_all(LOADED_IMPLEMENTATION)
