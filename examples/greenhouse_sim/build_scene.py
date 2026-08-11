"""Compose a deleafing benchmark scene from the greenhouse and the vine assets.

Run from the repository root with Isaac Sim's bundled interpreter, after
convert_vines_to_usd.py. $ISAACSIM stands for your Isaac Sim install; on
Windows use %ISAACSIM%\\python.bat instead:

    $ISAACSIM/python.sh examples/greenhouse_sim/build_scene.py

The written scene sublayers the greenhouse, disables its legacy all-gutter
vines and stale camera-robot payload, and adds sparse benchmark vines to one
gutter. The source art is never modified and scenes can be regenerated freely.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import greenhouse_scene

_DEFAULT_GREENHOUSE = pathlib.Path("greenhouse/gh_tomato_test.usd")
_DEFAULT_VINES = pathlib.Path("data/greenhouse_sim/vines")
_DEFAULT_OUTPUT = pathlib.Path("data/greenhouse_sim/scenes/deleafing_bench.usd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--greenhouse", type=pathlib.Path, default=_DEFAULT_GREENHOUSE)
    parser.add_argument("--vines", type=pathlib.Path, default=_DEFAULT_VINES, help="directory of converted vine USD")
    parser.add_argument("--output", type=pathlib.Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--gutter",
        default="Gutter_01",
        help="single greenhouse gutter whose BedSet segments receive benchmark vines",
    )
    parser.add_argument(
        "--beds",
        type=int,
        default=0,
        help="limit BedSet segments within --gutter; 0 uses the complete gutter",
    )
    parser.add_argument("--plants-per-bed", type=int, default=1, help="0 fills each bed to capacity")
    parser.add_argument(
        "--placement-mode",
        choices=("source", "procedural"),
        default=greenhouse_scene.DEFAULT_PLACEMENT_MODE,
        help="preserve gh_tomato_test stem frames, or use legacy procedural placement",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=greenhouse_scene.DEFAULT_SPACING_M,
        help="plant spacing in procedural mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="yaw seed in procedural mode")
    args = parser.parse_args()

    vines = sorted(args.vines.glob("tomato_*.usd"))
    if not vines:
        print(f"no vine USD found under {args.vines}; run convert_vines_to_usd.py first")
        return 1

    scene = greenhouse_scene.build_scene(
        args.greenhouse,
        vines,
        args.output,
        gutter=args.gutter,
        max_beds=args.beds or None,
        spacing=args.spacing,
        plants_per_bed=args.plants_per_bed or None,
        seed=args.seed,
        placement_mode=args.placement_mode,
    )

    print(f"scene:  {scene.path}")
    print(f"gutter: {scene.gutter}")
    print(f"beds:   {len(scene.beds)}  ({', '.join(scene.beds[:3])}{' ...' if len(scene.beds) > 3 else ''})")
    print(f"vines:  {len(scene.placements)} from {len(vines)} distinct assets")
    for repair in scene.repairs:
        print(f"repair: {repair}")
    for placement in scene.placements[:4]:
        x, y, z = placement.position
        print(
            f"   {placement.prim_path}  {placement.asset.name}  ({x:.3f}, {y:.3f}, {z:.3f})  yaw={placement.yaw_degrees:+.1f}deg"
            f"  source={placement.source_prim or 'procedural'}"
        )
    if len(scene.placements) > 4:
        print(f"   ... {len(scene.placements) - 4} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
