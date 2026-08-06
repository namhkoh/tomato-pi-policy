"""Convert the tomato vine GLB assets into per-organ USD.

Run with Isaac Sim's bundled interpreter, which supplies USD and numpy:

    D:\\isaac-sim\\python.bat examples/greenhouse_sim/convert_vines_to_usd.py

Output goes to a gitignored directory; the GLBs under greenhouse/ stay the
source of truth. Every vine is checked against its metadata sidecar as it is
converted, and the run fails if any invariant the benchmark relies on is
broken, so a regenerated asset set cannot quietly degrade the cut sites.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import glb
from greenhouse_sim import organs
from greenhouse_sim import vine_usd

_DEFAULT_SOURCE = pathlib.Path("greenhouse/tomato_glb_20")
_DEFAULT_OUTPUT = pathlib.Path("data/greenhouse_sim/vines")


def check_plant(plant: organs.Plant) -> list[str]:
    """Invariants the deleafing task depends on. Empty result means healthy."""
    issues = []
    counts = plant.metadata.get("counts", {})

    num_sub_stems = len(plant.labelled("SubStem"))
    if "sub_stems" in counts and num_sub_stems != counts["sub_stems"]:
        issues.append(f"found {num_sub_stems} sub-stems, metadata says {counts['sub_stems']}")

    num_trusses = len(plant.labelled("Truss"))
    if "trusses" in counts and num_trusses != counts["trusses"]:
        issues.append(f"found {num_trusses} trusses, metadata says {counts['trusses']}")

    expected_foliage = counts.get("leaves", 0) + counts.get("flowers", 0)
    num_foliage = sum(1 for o in plant.organs if o.tissue is organs.Tissue.FOLIAGE)
    if expected_foliage and num_foliage != expected_foliage:
        issues.append(f"found {num_foliage} foliage organs, metadata says {expected_foliage}")

    # Sub-stem labels must ascend the stem: the deleafing rule is bottom-up, so
    # a label order that disagrees with height would silently mis-score.
    sub_stems = sorted(plant.labelled("SubStem"), key=lambda o: int(re.search(r"\d+", o.label).group()))
    # Organ attachments are still in the GLB's Y-up frame; height is component 1.
    heights = [float(o.attachment[1]) for o in sub_stems if o.attachment is not None]
    if any(b < a - 1e-9 for a, b in itertools.pairwise(heights)):
        issues.append("sub-stem labels are not monotonic in height")

    # A junction at the adjacency limit means the organ was nearly missed and
    # the tree may have been completed by the graft-onto-root fallback.
    gaps = [o.attachment_gap for o in plant.organs if o.tissue is organs.Tissue.STEM and o.parent is not None]
    if gaps and max(gaps) >= organs.ADJACENCY_RADIUS_M:
        issues.append(f"largest stem junction gap {max(gaps) * 1000:.1f} mm reaches the adjacency radius")

    orphans = [o.label for o in plant.organs if o.parent is None and o.index != plant.root]
    if orphans:
        issues.append(f"{len(orphans)} organs are detached from the plant")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=pathlib.Path, default=_DEFAULT_SOURCE, help="directory of tomato_*.glb")
    parser.add_argument("--output", type=pathlib.Path, default=_DEFAULT_OUTPUT, help="directory for the USD assets")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N vines")
    args = parser.parse_args()

    sources = sorted(args.source.glob("tomato_*.glb"))[: args.limit]
    if not sources:
        print(f"no tomato_*.glb found under {args.source}")
        return 1

    failed = []
    for source in sources:
        started = time.perf_counter()
        plant = organs.load_plant(source)
        issues = check_plant(plant)
        result = vine_usd.write_vine(plant, glb.read_glb(source), args.output / f"{source.stem}.usd")

        gaps = [o.attachment_gap for o in plant.organs if o.tissue is organs.Tissue.STEM and o.parent is not None]
        print(
            f"{source.stem}  organs={len(plant.organs):3d}  sub-stems={len(plant.labelled('SubStem')):2d}  "
            f"tris={result.num_triangles:7d}  max-gap={max(gaps) * 1000 if gaps else 0:5.2f}mm  "
            f"{time.perf_counter() - started:4.1f}s  -> {result.path.name}"
        )
        for issue in issues:
            print(f"    ISSUE: {issue}")
            failed.append(f"{source.stem}: {issue}")

    print(f"\nconverted {len(sources)} vines to {args.output}")
    if failed:
        print(f"{len(failed)} invariant violations; the assets are not benchmark-ready")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
