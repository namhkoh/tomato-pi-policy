"""Extract mountable robot hardware parts from the supplied CAD.

    "C:/Program Files/FreeCAD 1.0/bin/freecadcmd.exe" \
        examples/greenhouse_sim/extract_robot_hardware.py

Run with FreeCAD's headless interpreter (freecadcmd), not Isaac Sim's. Outputs
are committed so simulator users do not need FreeCAD.

The source assets need normalization before they can be attached safely:

* The end-effector STEP contains a bracket, D405 body, two construction FOV
  cones, and four bolts. Only the bracket and camera body are exported.
* The bracket is re-origined on its mounting face. The STEP's authored D405
  transform is then recorded relative to that mounting frame.
* ``deleaf_knife.stl`` contains two disconnected solids. The straight plate is
  the blade; the U-shaped arc is support geometry. They are exported separately
  in one common distal mounting frame so only the plate receives cut semantics.
* The head bracket already has a useful mounting-face origin and is measured,
  not rewritten.

The STEP FOV cones are construction guides and are not sensor calibration. The
official D405 HD depth and color FOV is 84 x 58 degrees; simulator camera
intrinsics are authored from that specification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import FreeCAD
import Import
import Mesh
import MeshPart

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..")) if "__file__" in globals() else os.getcwd()
_SOURCE = os.path.join(_ROOT, "greenhouse", "robot_assets")
_DERIVED = os.path.join(_SOURCE, "derived")

_BRACKET_LABEL = "카메라브라켓_고속형"
_BRACKET_STEP = "C_CameraBracket_Bent(HighSpeedGripper(R1.1&1.2)).stp"
_D405_PART_LABEL = "D405"
_D405_SOLID_LABEL = "D405_Solid"
_HEAD_BRACKET_STL = "HeadCam_Bracket_D405-Body.stl"
_KNIFE_STL = "deleaf_knife.stl"

_LINEAR_DEFLECTION = 0.05
_ANGULAR_DEFLECTION = 0.35
_NORMALIZED_STEP_TIMESTAMP = "2000-01-01T00:00:00"


def _round_vector(vector, digits: int = 5) -> list[float]:
    return [
        round(float(vector.x), digits),
        round(float(vector.y), digits),
        round(float(vector.z), digits),
    ]


def _bounds(bounds) -> dict:
    return {
        "min_mm": [round(bounds.XMin, 5), round(bounds.YMin, 5), round(bounds.ZMin, 5)],
        "max_mm": [round(bounds.XMax, 5), round(bounds.YMax, 5), round(bounds.ZMax, 5)],
        "size_mm": [round(bounds.XLength, 5), round(bounds.YLength, 5), round(bounds.ZLength, 5)],
    }


def _rotation_matrix(rotation) -> list[list[float]]:
    matrix = rotation.toMatrix()
    return [
        [round(matrix.A11, 9), round(matrix.A12, 9), round(matrix.A13, 9)],
        [round(matrix.A21, 9), round(matrix.A22, 9), round(matrix.A23, 9)],
        [round(matrix.A31, 9), round(matrix.A32, 9), round(matrix.A33, 9)],
    ]


def _normalize_step_header(path: str) -> None:
    """Remove FreeCAD's wall-clock timestamp from an otherwise stable STEP."""
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    normalized, replacements = re.subn(
        r"(FILE_NAME\('[^']*',)'[^']*'",
        rf"\1'{_NORMALIZED_STEP_TIMESTAMP}'",
        source,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(f"could not normalize STEP FILE_NAME timestamp: {path}")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines()) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(normalized)


def _mesh_and_export(shape, name: str, document) -> int:
    feature = document.addObject("Part::Feature", name)
    feature.Shape = shape
    document.recompute()
    step_path = os.path.join(_DERIVED, name + ".step")
    Import.export([feature], step_path)
    _normalize_step_header(step_path)

    mesh = document.addObject("Mesh::Feature", name + "Mesh")
    mesh.Mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=_LINEAR_DEFLECTION,
        AngularDeflection=_ANGULAR_DEFLECTION,
        Relative=False,
    )
    Mesh.export([mesh], os.path.join(_DERIVED, name + ".stl"))
    return int(mesh.Mesh.CountFacets)


def _camera_assembly():
    document = FreeCAD.newDocument("camera_hardware_source")
    Import.insert(os.path.join(_SOURCE, _BRACKET_STEP), document.Name)
    document.recompute()
    return document


def extract_camera_hardware() -> tuple[list[dict], dict]:
    """Extract the bracket/body and preserve the assembly-authored camera pose."""
    source = _camera_assembly()
    bracket = next(
        (obj for obj in source.Objects if obj.Label.startswith(_BRACKET_LABEL)),
        None,
    )
    camera_part = next((obj for obj in source.Objects if obj.Label == _D405_PART_LABEL), None)
    camera_body = next((obj for obj in source.Objects if obj.Label == _D405_SOLID_LABEL), None)
    if bracket is None or camera_part is None or camera_body is None:
        raise RuntimeError("camera STEP is missing the bracket, D405 part, or D405 solid")

    bracket_shape = bracket.Shape.copy()
    bracket_shape.Placement = bracket.getGlobalPlacement()
    source_bounds = bracket_shape.BoundBox
    reorigin = FreeCAD.Vector(-source_bounds.Center.x, -source_bounds.YMin, -source_bounds.ZMin)
    bracket_shape.translate(reorigin)

    bracket_out = FreeCAD.newDocument("camera_bracket_d405")
    bracket_facets = _mesh_and_export(bracket_shape, "camera_bracket_d405", bracket_out)

    camera_shape = camera_body.Shape.copy()
    camera_shape.Placement = camera_body.Placement
    camera_out = FreeCAD.newDocument("d405_body")
    camera_facets = _mesh_and_export(camera_shape, "d405_body", camera_out)

    camera_global = camera_part.getGlobalPlacement()
    camera_in_bracket = camera_global.Base + reorigin
    parts = [
        {
            "part": "camera_bracket_d405",
            "source": _BRACKET_STEP,
            **_bounds(bracket_shape.BoundBox),
            "volume_mm3": round(bracket_shape.Volume, 1),
            "facets": bracket_facets,
            "faces": len(bracket_shape.Faces),
            "frame": "origin at mounting-face centre; +Y away from face; +Z above face",
        },
        {
            "part": "d405_body",
            "source": _BRACKET_STEP,
            **_bounds(camera_shape.BoundBox),
            "volume_mm3": round(camera_shape.Volume, 1),
            "facets": camera_facets,
            "faces": len(camera_shape.Faces),
            "frame": "D405 assembly part frame; camera forward is +Y",
            "depth_baseline_mm": 18.0,
            "optical_origin_mm": [0.0, 19.23, 0.0],
            "depth_fov_degrees": [84.0, 58.0],
            "color_fov_degrees": [84.0, 58.0],
            "specification": "RealSense D400 Series Datasheet, July 2023",
        },
    ]
    mount = {
        "translation_mm": _round_vector(camera_in_bracket),
        "rotation_matrix": _rotation_matrix(camera_global.Rotation),
        "source": "authored D405 App::Part transform in the supplied STEP",
    }
    return parts, mount


def measure_head_bracket() -> dict:
    mesh = Mesh.Mesh(os.path.join(_SOURCE, _HEAD_BRACKET_STL))
    return {
        "part": "head_camera_bracket_d405",
        "source": _HEAD_BRACKET_STL,
        **_bounds(mesh.BoundBox),
        "facets": int(mesh.CountFacets),
        "frame": "origin on the 46 x 36 mm head mounting face; +Z through bracket height",
    }


def extract_knife() -> dict:
    """Separate the flat blade from the curved support in one mounting frame."""
    source_path = os.path.join(_SOURCE, _KNIFE_STL)
    source_mesh = Mesh.Mesh(source_path)
    components = source_mesh.getSeparateComponents()
    if len(components) != 2:
        raise RuntimeError(f"expected two knife components, found {len(components)}")

    blade = max(components, key=lambda component: component.BoundBox.XLength)
    arc = min(components, key=lambda component: component.BoundBox.XLength)
    if blade.BoundBox.ZLength >= arc.BoundBox.ZLength:
        raise RuntimeError("knife component classification no longer identifies a flat blade")

    source_blade_bounds = _bounds(blade.BoundBox)
    source_arc_bounds = _bounds(arc.BoundBox)
    mount_origin = FreeCAD.Vector(
        blade.BoundBox.Center.x,
        blade.BoundBox.YMax,
        0.5 * (blade.BoundBox.ZMin + blade.BoundBox.ZMax),
    )
    translation = FreeCAD.Vector(-mount_origin.x, -mount_origin.y, -mount_origin.z)
    blade.translate(translation.x, translation.y, translation.z)
    arc.translate(translation.x, translation.y, translation.z)

    document = FreeCAD.newDocument("deleaf_knife")
    outputs = []
    for name, component in (("deleaf_knife_blade", blade), ("deleaf_knife_arc", arc)):
        feature = document.addObject("Mesh::Feature", name)
        feature.Mesh = component
        Mesh.export([feature], os.path.join(_DERIVED, name + ".stl"))
        outputs.append(
            {
                "name": name,
                **_bounds(component.BoundBox),
                "facets": int(component.CountFacets),
                "cutting_surface": name.endswith("_blade"),
            }
        )

    with open(source_path, "rb") as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()
    return {
        "part": "deleaf_knife",
        "source": _KNIFE_STL,
        "source_sha256": source_hash,
        "source_bounds": {
            "blade": source_blade_bounds,
            "arc": source_arc_bounds,
        },
        "mount_origin_in_source_mm": _round_vector(mount_origin),
        "frame": "origin at centre of blade distal mounting face; blade projects along -Y",
        "components": outputs,
        "cut_semantics": "flat straight blade only; U-shaped arc is non-cutting support",
    }


def main() -> int:
    os.makedirs(_DERIVED, exist_ok=True)
    camera_parts, camera_mount = extract_camera_hardware()
    report = {
        "schema_version": 2,
        "parts": camera_parts + [measure_head_bracket(), extract_knife()],
        "mounts": {"camera_bracket_to_d405": camera_mount},
    }
    path = os.path.join(_DERIVED, "hardware.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for part in report["parts"]:
        print(f"{part['part']}: {part.get('size_mm', 'component mesh')} -> {_DERIVED}")
    return 0


main()
del sys
