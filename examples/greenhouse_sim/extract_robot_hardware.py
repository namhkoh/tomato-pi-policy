"""Extract mountable robot hardware parts from the supplied CAD.

    "C:/Program Files/FreeCAD 1.0/bin/freecadcmd.exe" \\
        examples/greenhouse_sim/extract_robot_hardware.py

Run with FreeCAD's headless interpreter (freecadcmd), not Isaac Sim's -- this
is a one-off CAD step, and its outputs are committed so that nobody needs
FreeCAD just to run the simulator.

Two things need extracting rather than using as-shipped:

* The camera bracket arrives as an assembly STEP that also contains a D405 body,
  two field-of-view cones and four M3 bolts. Only the bracket is hardware we
  mount, so it is baked out on its own.
* Parts are re-origined onto their own mounting face. Mount transforms then live
  in the robot model where they can be inspected and randomised, instead of
  being baked invisibly into mesh coordinates.

The FOV cones in the assembly are a working-volume guide, not optics: they
subtend about 49 x 35 degrees, whereas a D405 is 87 x 58. Camera intrinsics come
from the datasheet, not from this CAD.
"""

from __future__ import annotations

import json
import os
import sys

import FreeCAD
import Import
import Mesh
import MeshPart

# freecadcmd execs the script rather than importing it, so __file__ is not
# always set; fall back to the working directory, which the docstring requires
# to be the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..")) if "__file__" in globals() else os.getcwd()
_SOURCE = os.path.join(_ROOT, "greenhouse", "robot_assets")
_DERIVED = os.path.join(_SOURCE, "derived")

# Label of the bracket solid inside the assembly ("camera bracket, high-speed").
_BRACKET_LABEL = "\uce74\uba54\ub77c\ube0c\ub77c\ucf13_\uace0\uc18d\ud615"
_BRACKET_STEP = "C_CameraBracket_Bent(HighSpeedGripper(R1.1&1.2)).stp"

# Mesh tolerances: fine enough that M3 holes stay round, coarse enough to keep
# the collision mesh cheap.
_LINEAR_DEFLECTION = 0.05
_ANGULAR_DEFLECTION = 0.35


def _mesh_and_export(shape, name: str, document) -> int:
    feature = document.addObject("Part::Feature", name)
    feature.Shape = shape
    document.recompute()
    Import.export([feature], os.path.join(_DERIVED, name + ".step"))

    mesh = document.addObject("Mesh::Feature", name + "Mesh")
    mesh.Mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=_LINEAR_DEFLECTION,
        AngularDeflection=_ANGULAR_DEFLECTION,
        Relative=False,
    )
    Mesh.export([mesh], os.path.join(_DERIVED, name + ".stl"))
    return int(mesh.Mesh.CountFacets)


def extract_camera_bracket() -> dict:
    """Bake the camera bracket out of the shipped assembly, on its own origin."""
    source = FreeCAD.newDocument("bracket_source")
    Import.insert(os.path.join(_SOURCE, _BRACKET_STEP), source.Name)
    source.recompute()

    matches = [o for o in source.Objects if o.Label.startswith(_BRACKET_LABEL)]
    if not matches:
        raise RuntimeError(f"no bracket solid labelled {_BRACKET_LABEL!r} in {_BRACKET_STEP}")
    bracket = matches[0]

    shape = bracket.Shape.copy()
    shape.Placement = bracket.getGlobalPlacement()
    # Origin at the centre of the mounting face, so +Y runs out along the arm
    # and +Z up off the face.
    box = shape.BoundBox
    shape.translate(FreeCAD.Vector(-box.Center.x, -box.YMin, -box.ZMin))

    out = FreeCAD.newDocument("camera_bracket_d405")
    facets = _mesh_and_export(shape, "camera_bracket_d405", out)

    bounds = shape.BoundBox
    return {
        "part": "camera_bracket_d405",
        "source": _BRACKET_STEP,
        "size_mm": [round(bounds.XLength, 2), round(bounds.YLength, 2), round(bounds.ZLength, 2)],
        "volume_mm3": round(shape.Volume, 1),
        "facets": facets,
        "faces": len(shape.Faces),
    }


def main() -> int:
    os.makedirs(_DERIVED, exist_ok=True)
    report = {"parts": [extract_camera_bracket()]}
    path = os.path.join(_DERIVED, "hardware.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    for part in report["parts"]:
        print(f"{part['part']}: {part['size_mm']} mm, {part['facets']} facets -> {_DERIVED}")
    return 0


# Called unconditionally: freecadcmd execs this file under a module name other
# than __main__, so a standard entry-point guard would never fire.
main()
del sys
