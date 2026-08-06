"""Open the greenhouse deleafing scene in Isaac Sim.

Run from the repository root with Isaac Sim's bundled interpreter. $ISAACSIM
stands for your Isaac Sim install; on Windows use %ISAACSIM%\\python.bat:

    $ISAACSIM/python.sh examples/greenhouse_sim/launch_greenhouse.py
    $ISAACSIM/python.sh examples/greenhouse_sim/launch_greenhouse.py --headless --screenshot shot.png

Interactively this holds the viewport open until the window is closed. Headless
with --screenshot it renders one frame from an inspection camera and exits,
which is how the scene gets checked without a display.

The scene carries no physics yet; this stage is geometry and lighting only.
"""

# Imports here are deliberately deferred: SimulationApp has to be constructed
# before anything from omni or pxr is imported, and the screenshot path pulls in
# Replicator only when it is actually used.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import pathlib
import sys

_DEFAULT_SCENE = pathlib.Path("data/greenhouse_sim/scenes/deleafing_bench.usd")

# An observer standing in the aisle at roughly the height of a robot's head
# camera, framing the planted bed.
_DEFAULT_EYE = (6.2, 5.8, 2.6)
_DEFAULT_TARGET = (7.9, 3.3, 1.4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=pathlib.Path, default=_DEFAULT_SCENE)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--screenshot", type=pathlib.Path, default=None, help="render one frame here, then exit")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=60, help="frames to render before capturing")
    parser.add_argument("--eye", type=float, nargs=3, default=_DEFAULT_EYE)
    parser.add_argument("--target", type=float, nargs=3, default=_DEFAULT_TARGET)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = args.scene.resolve()
    if not scene.exists():
        print(f"scene not found: {scene}\nrun convert_vines_to_usd.py then build_scene.py first")
        return 1

    # SimulationApp must exist before any omni/pxr import.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless or args.screenshot is not None})

    import omni.usd
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom

    omni.usd.get_context().open_stage(str(scene))
    # Opening is asynchronous; let Kit settle before touching the stage.
    for _ in range(10):
        app.update()
    stage = omni.usd.get_context().get_stage()

    camera_path = Sdf.Path("/World/InspectionCamera")
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

    # USD cameras look down their local -Z, which is what SetLookAt encodes;
    # the prim transform is the inverse of that view matrix.
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*args.eye), Gf.Vec3d(*args.target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(view.GetInverse())

    vines = stage.GetPrimAtPath("/World/Vines")
    num_vines = len(vines.GetChildren()) if vines and vines.IsValid() else 0
    print(f"stage:  {scene}")
    print(f"prims:  {sum(1 for _ in stage.Traverse())}")
    print(f"vines:  {num_vines}")

    if args.screenshot is not None:
        import omni.replicator.core as rep

        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        product = rep.create.render_product(str(camera_path), (args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        # The path tracer needs several frames before the image stops being noise.
        for _ in range(args.warmup):
            rep.orchestrator.step(rt_subframes=4)
        _write_png(annotator.get_data(), args.screenshot)
        print(f"wrote {args.screenshot}")
    else:
        while app.is_running():
            app.update()

    app.close()
    return 0


def _write_png(rgba, path: pathlib.Path) -> None:
    """Write an HxWx4 uint8 array without assuming a particular image library."""
    import numpy as np

    array = np.asarray(rgba)
    if array.ndim != 3:
        raise RuntimeError(f"unexpected annotator output shape {array.shape}")
    try:
        from PIL import Image

        Image.fromarray(array[:, :, :3].astype(np.uint8)).save(path)
    except ImportError:
        import struct
        import zlib

        height, width = array.shape[:2]
        rows = b"".join(b"\x00" + array[y, :, :3].astype(np.uint8).tobytes() for y in range(height))

        def chunk(tag: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
            )

        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6))
            + chunk(b"IEND", b"")
        )


if __name__ == "__main__":
    sys.exit(main())
