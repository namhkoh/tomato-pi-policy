"""Shared Model A v1.2 asset contract; do not mix versions across FK and USD."""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_VERSION = "1.2"
ROBOT_NAME = "RBY1_A_v1.2"
ROBOT_ROOT = "/RBY1_A_v1_2"
DEFAULT_URDF = REPOSITORY_ROOT / "third_party/rby1-sdk/models/rby1a/urdf/model_v1.2.urdf"
# The importer writes sibling configuration layers. Isolate them from v1.0.
DEFAULT_ASSET = REPOSITORY_ROOT / "data/greenhouse_sim/robots/rby1a_v1.2/rby1a_v1.2.usd"
DEFAULT_REPORT = DEFAULT_ASSET.with_suffix(".json")
D405_RESOLUTION = (848, 408)


def joint_limits_degrees(names):
    joints = {j.attrib["name"]: j for j in ET.parse(DEFAULT_URDF).getroot().findall("joint")}
    return tuple(tuple(math.degrees(float(joints[name].find("limit").attrib[bound]))
                       for name in names) for bound in ("lower", "upper"))


def validate_urdf(path=DEFAULT_URDF):
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    if root.attrib.get("name") != ROBOT_NAME:
        raise ValueError(f"Expected {ROBOT_NAME}, got {root.attrib.get('name')}")
    missing = [m.attrib["filename"] for m in root.findall(".//mesh")
               if not (path.parent / m.attrib["filename"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing robot mesh references: {missing}")
    return root
