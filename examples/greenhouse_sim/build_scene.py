"""Compose a deleafing benchmark scene from the greenhouse and the vine assets.

Run with Isaac Sim's bundled interpreter, after convert_vines_to_usd.py:

    D:\\isaac-sim\\python.bat examples/greenhouse_sim/build_scene.py

The written scene sublayers the greenhouse and adds only vine placements, so
the source art is never modified and scenes can be regenerated freely.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import greenhouse_scene

_DEFAULT_GREENHOUSE = pathlib.Path("greenhouse/green_house.usd")
_DEFAULT_VINES = pathlib.Path("data/greenhouse_sim/vines")
_DEFAULT_OUTPUT = pathlib.Path("data/greenhouse_sim/scenes/deleafing_bench.usd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--greenhouse", type=pathlib.Path, default=_DEFAULT_GREENHOUSE)
    parser.add_argument("--vines", type=pathlib.Path, default=_DEFAULT_VINES, help="directory of converted vine USD")
    parser.add_argument("--output", type=pathlib.Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--beds", type=int, default=1, help="number of BedSet groups to plant")
    parser.add_argument("--plants-per-bed", type=int, default=8, help="0 fills each bed to capacity")
    parser.add_argument("--spacing", type=float, default=greenhouse_scene.DEFAULT_SPACING_M)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vines = sorted(args.vines.glob("tomato_*.usd"))
    if not vines:
        print(f"no vine USD found under {args.vines}; run convert_vines_to_usd.py first")
        return 1

    scene = greenhouse_scene.build_scene(
        args.greenhouse,
        vines,
        args.output,
        max_beds=args.beds,
        spacing=args.spacing,
        plants_per_bed=args.plants_per_bed or None,
        seed=args.seed,
    )

    print(f"scene:  {scene.path}")
    print(f"beds:   {len(scene.beds)}  ({', '.join(scene.beds[:3])}{' ...' if len(scene.beds) > 3 else ''})")
    print(f"vines:  {len(scene.placements)} from {len(vines)} distinct assets")
    for repair in scene.repairs:
        print(f"repair: {repair}")
    for placement in scene.placements[:4]:
        x, y, z = placement.position
        print(
            f"   {placement.prim_path}  {placement.asset.name}  ({x:.3f}, {y:.3f}, {z:.3f})  yaw={placement.yaw_degrees:+.1f}deg"
        )
    if len(scene.placements) > 4:
        print(f"   ... {len(scene.placements) - 4} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
