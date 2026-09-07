"""Read-only mesh datum inspection (run with a Python containing numpy).

Print major planar patches and their boundary loops. Coordinates are mm.
Useful for checking hole patterns in triangulated STL/COLLADA vendor assets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


def triangles(path):
    if path.suffix.lower() == ".stl":
        raw = path.read_bytes()
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) != 84 + 50 * count:
            raise ValueError("Expected binary STL")
        return np.array([struct.unpack_from("<9f", raw, 84 + i * 50 + 12)
                         for i in range(count)]).reshape(-1, 3, 3)
    root = ET.parse(path).getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    result = []
    scale = float(root.find("c:asset/c:unit", ns).attrib["meter"]) * 1000
    for geom in root.findall(".//c:geometry", ns):
        mesh = geom.find("c:mesh", ns)
        sources = {s.attrib["id"]: np.fromstring(s.find("c:float_array", ns).text, sep=" ")
                   for s in mesh.findall("c:source", ns)}
        vertices = {v.attrib["id"]: v.find("c:input[@semantic='POSITION']", ns).attrib["source"][1:]
                    for v in mesh.findall("c:vertices", ns)}
        nodes = root.findall(".//c:node[c:instance_geometry]", ns)
        matches = [n for n in nodes if n.find("c:instance_geometry", ns).attrib["url"] == "#" + geom.attrib["id"]]
        transform = np.eye(4)
        if matches and matches[0].find("c:matrix", ns) is not None:
            transform = np.fromstring(matches[0].find("c:matrix", ns).text, sep=" ").reshape(4, 4)
        for tri in mesh.findall("c:triangles", ns):
            inputs = tri.findall("c:input", ns)
            stride = max(int(i.attrib["offset"]) for i in inputs) + 1
            vertex = next(i for i in inputs if i.attrib["semantic"] == "VERTEX")
            indices = np.fromstring(tri.find("c:p", ns).text, sep=" ", dtype=int).reshape(-1, stride)
            points = sources[vertices[vertex.attrib["source"][1:]]].reshape(-1, 3)
            points = (points @ transform[:3, :3].T + transform[:3, 3]) * scale
            result.extend(points[indices[:, int(vertex.attrib["offset"])]].reshape(-1, 3, 3))
    return np.array(result)


def planar_patches(tris, min_area=15):
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    groups = defaultdict(list)
    for i in np.flatnonzero(norm > 1e-7):
        normal = cross[i] / norm[i]
        key = tuple(np.round(normal, 4)) + (round(float(normal @ tris[i, 0]), 3),)
        groups[key].append(i)
    patches = []
    for key, indices in groups.items():
        area = float(norm[indices].sum() / 2)
        if area < min_area:
            continue
        edges = Counter()
        for triangle in tris[indices]:
            verts = [tuple(np.round(p, 4)) for p in triangle]
            for a, b in zip(verts, verts[1:] + verts[:1]):
                edges[tuple(sorted((a, b)))] += 1
        neighbours = defaultdict(set)
        for (a, b), count in edges.items():
            if count == 1:
                neighbours[a].add(b)
                neighbours[b].add(a)
        loops = []
        while neighbours:
            todo = [next(iter(neighbours))]
            connected = set()
            while todo:
                p = todo.pop()
                if p in connected:
                    continue
                connected.add(p)
                todo.extend(neighbours.pop(p, ()))
            points = np.array(sorted(connected))
            loops.append({"min": points.min(0).round(4).tolist(),
                          "max": points.max(0).round(4).tolist(), "vertices": len(points),
                          "centre": ((points.min(0) + points.max(0))/2).round(4).tolist()})
        patches.append({"normal": key[:3], "plane_offset": key[3], "area_mm2": round(area, 2), "loops": loops})
    return sorted(patches, key=lambda p: -p["area_mm2"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--min-area", type=float, default=15)
    args = parser.parse_args()
    tris = triangles(args.mesh)
    print(json.dumps({"source": str(args.mesh), "triangles": len(tris),
                      "bounds_mm": [tris.min((0, 1)).tolist(), tris.max((0, 1)).tolist()],
                      "patches": planar_patches(tris, args.min_area)}, indent=2))
