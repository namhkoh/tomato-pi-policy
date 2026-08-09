"""Run isolated Isaac processes and aggregate strict bimanual acceptance trials."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenhouse_sim import repeatability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=("Vine_0000/SubStem_00", "Vine_0000/SubStem_01"),
        help="exact VINE/SUBSTEM keys",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--probe", choices=("left_approach", "right_approach", "full"), default="full"
    )
    parser.add_argument("--motion-steps", type=int, default=180)
    parser.add_argument("--drop-steps", type=int, default=1200)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/repeatability"),
    )
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        default=pathlib.Path("data/greenhouse_sim/repeatability_summary.json"),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="one additional interactive_greenhouse.py argument; repeat as needed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = pathlib.Path(__file__).with_name("interactive_greenhouse.py").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for target in args.targets:
        try:
            vine_name, organ_label = target.split("/", 1)
        except ValueError as exc:
            raise ValueError(f"target must be VINE/SUBSTEM, got {target!r}") from exc
        for seed in args.seeds:
            stem = f"{vine_name}_{organ_label}_seed_{seed}"
            report_path = args.output_dir / f"{stem}.json"
            stdout_path = args.output_dir / f"{stem}.stdout.log"
            stderr_path = args.output_dir / f"{stem}.stderr.log"
            command = [
                sys.executable,
                str(script),
                "--headless",
                "--bimanual-probe",
                args.probe,
                "--target-vine",
                vine_name,
                "--target-organ",
                organ_label,
                "--episode-seed",
                str(seed),
                "--motion-steps",
                str(args.motion_steps),
                "--drop-steps",
                str(args.drop_steps),
                "--report",
                str(report_path),
                *args.extra_arg,
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = {"succeeded": False, "error": "Isaac process produced no report"}
            acceptance = repeatability.acceptance_result(report)
            entries.append(
                {
                    "target": target,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "report": str(report_path),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "acceptance": acceptance,
                    "reported_error": report.get("error")
                    or (report.get("bimanual_probe") or {}).get("error"),
                }
            )
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(
                json.dumps(repeatability.summarize(entries), indent=2), encoding="utf-8"
            )
    summary = repeatability.summarize(entries)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
