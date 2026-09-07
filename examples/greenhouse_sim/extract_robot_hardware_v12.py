"""Extract the user-supplied v1.2 camera brackets without modifying source CAD.

Run with C:/Program Files/FreeCAD 1.0/bin/python.exe. The original hardware
manifest and v1.0 exports remain available. FreeCAD is not a runtime dependency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import FreeCAD as App
import Mesh
import MeshPart
import Part

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "greenhouse/robot_assets"
DERIVED = ASSETS / "derived"


def bounds(mesh):
    b = mesh.BoundBox
    return {"min_mm": [b.XMin, b.YMin, b.ZMin], "max_mm": [b.XMax, b.YMax, b.ZMax],
            "size_mm": [b.XLength, b.YLength, b.ZLength], "facets": mesh.CountFacets}


def source_info(path):
    return {"source": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    head_source = ASSETS / "HeadCam_Bracket_D405.FCStd"
    wrist_source = ASSETS / "D405_Wrist_Bracket_v2-Body.stl"
    doc = App.openDocument(str(head_source))
    try:
        # Read the saved final BRep. Recomputing with an older FreeCAD can change
        # newer feature types and is deliberately forbidden for this extractor.
        body = doc.getObject("Body")
        shape = body.Shape.copy()
        if shape.isNull() or not shape.isValid() or len(shape.Solids) != 1:
            raise ValueError("Head bracket's saved final solid is invalid")
        head = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.05,
                                      AngularDeflection=0.25, Relative=False)
        head_path = DERIVED / "head_camera_bracket_d405_v12.stl"
        head.write(str(head_path))
        head_holes = [{"radius_mm": round(f.Surface.Radius, 5),
                       "centre_mm": list(f.Surface.Center), "axis": list(f.Surface.Axis)}
                      for f in shape.Faces if type(f.Surface).__name__ == "Cylinder"
                      and abs(f.Surface.Radius - 1.7) < 1e-6]
        wrist = Mesh.Mesh(str(wrist_source))
        if not wrist.isSolid():
            raise ValueError("Wrist bracket is not a closed mesh")
        manifest = json.loads((DERIVED / "hardware.json").read_text(encoding="utf-8"))
        replacements = {
            "head_camera_bracket_d405": {
                "part": "head_camera_bracket_d405", **source_info(head_source), **bounds(head),
                "derived_stl": head_path.name, "saved_tip": body.Tip.Name,
                "mount_holes": head_holes, "frame": "unmodified FCStd Body frame, mm",
            },
            "camera_bracket_d405": {
                "part": "camera_bracket_d405", **source_info(wrist_source), **bounds(wrist),
                "frame": "unmodified supplied v2 STL frame, mm",
                "robot_holes_mm": [[-9, -10, 0], [9, -10, 0]],
                "camera_holes_mm": [[-10, 29, 13], [10, 29, 13]],
            },
        }
        manifest["parts"] = [replacements.get(p["part"], p) for p in manifest["parts"]]
        manifest["robot_model"] = "RBY1_A_v1.2"
        # Simulator-only adapter. Keep the supplied bracket intact, bridge the
        # stock 34 mm screw pair to its 18 mm pair, and stand off the curved
        # wrist housing. Physical fabrication/fastener selection is NOT approved.
        adapter = Part.makeBox(48, 20, 5, App.Vector(-24, -20, -5))
        for x in (-17, 17):
            adapter = adapter.fuse(Part.makeCylinder(2.7, 10, App.Vector(x, -10, -15)))
            adapter = adapter.cut(Part.makeCylinder(1.7, 16, App.Vector(x, -10, -15)))
        for x in (-9, 9):
            adapter = adapter.cut(Part.makeCylinder(1.25, 6, App.Vector(x, -10, -5)))
        if not adapter.isValid() or len(adapter.Solids) != 1:
            raise ValueError("Invalid simulator wrist adapter")
        adapter_mesh = MeshPart.meshFromShape(Shape=adapter, LinearDeflection=0.05,
                                              AngularDeflection=0.25, Relative=False)
        adapter_mesh.write(str(DERIVED / "wrist_adapter_v12.stl"))
        manifest["parts"].append({"part": "wrist_adapter_v12", **bounds(adapter_mesh),
                                  "derived_stl": "wrist_adapter_v12.stl",
                                  "source": "procedural simulator adapter, not a supplied lab part",
                                  "physical_fabrication_approved": False,
                                  "robot_holes_mm": [[-17, -10, -15], [17, -10, -15]],
                                  "bracket_holes_mm": [[-9, -10, 0], [9, -10, 0]]})
        manifest["camera_resolution"] = [848, 408]
        manifest["mounts"]["camera_bracket_to_d405"] = {
            "translation_mm": [0, 29, 13], "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "source": "v2 STL y=29 rear-camera mating face; 20 mm pair aligned to D405 rear holes",
        }
        manifest["mount_validation"] = {
            "head": "26 x 36 mm NECK_2 mesh pair pattern; z=45 mm, y centre=-0.34 mm",
            "wrist": "34-to-18 mm simulator adapter; physical fit and fasteners require lab review",
            "head_source_recomputed": False,
            "camera_intrinsics": "nominal synthetic pinhole, not physical calibration",
        }
        out = DERIVED / "hardware_v1.2.json"
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(out), "head": replacements["head_camera_bracket_d405"],
                          "wrist": replacements["camera_bracket_d405"]}, indent=2))
    finally:
        App.closeDocument(doc.Name)


if __name__ == "__main__":
    main()
